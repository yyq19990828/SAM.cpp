#!/usr/bin/env python3
"""Compare layerwise W8/A8 error on frozen original-FP32 calibration samples.

This is an offline numerical study, not a runnable quantized model or a kernel
benchmark. Final segmentation quality requires independent, complete-model tests.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from tools.quantize.calibration import validate_dataset
from tools.convert.convert_sam3 import rename_key
from tools.convert.sam3_artifacts import BPE_SHA256, SAM3_REVISION, artifact_path, read_json, sha256_file, write_json
from tools.convert.sam3_gguf import quantization_module_for_linear_weight


def smooth_scale(activation_absmax, weight_absmax, alpha):
    activation = np.asarray(activation_absmax, dtype=np.float64)
    weight = np.asarray(weight_absmax, dtype=np.float64)
    if (activation.ndim != 1 or not activation.size or activation.shape != weight.shape
            or not np.isfinite(activation).all() or not np.isfinite(weight).all()
            or (activation < 0).any() or (weight < 0).any()
            or not math.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("invalid SmoothQuant channel maxima or alpha")
    scale = np.ones_like(activation)
    active = (activation > 0) & (weight > 0)
    # Log-space evaluation avoids overflow. Zero signal/weight channels need
    # no balancing. Bounded positive factors preserve the float identity.
    logarithm = alpha * np.log(activation[active]) - (1 - alpha) * np.log(weight[active])
    scale[active] = np.exp(np.clip(logarithm, math.log(1e-5), math.log(1e5)))
    return scale.astype(np.float32)


def quantize_rows(values, scale=None):
    """Symmetric signed INT8, round-to-nearest-even, one scale per row by default."""
    values = np.asarray(values)
    if (values.dtype != np.float32 or values.ndim != 2 or min(values.shape) <= 0
            or not np.isfinite(values).all()):
        raise ValueError("quantization requires finite nonempty F32 rows")
    if scale is None:
        maximum = np.abs(values).max(axis=1, keepdims=True)
        scale = np.maximum(maximum / 127, np.finfo(np.float32).tiny)
        scale[maximum == 0] = 1
    scale = np.asarray(scale, dtype=np.float32)
    if scale.shape not in ((), (1, 1), (values.shape[0], 1)) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("INT8 scales must be finite, positive, and scalar or per-row")
    # Float64 division also handles finite inputs paired with very small
    # externally supplied scales without intermediate overflow.
    rounded = np.rint(values.astype(np.float64) / scale)
    clipped = int(np.count_nonzero(np.abs(rounded) > 127))
    return np.clip(rounded, -127, 127).astype(np.int8), scale, clipped


def calibration_inputs(directory, allow_diagnostic=False):
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("kind") != "sam3-vision-activation-calibration" or manifest.get("schema_version") != 1
            or manifest.get("sam3_revision") != SAM3_REVISION or manifest.get("reference_kind") != "official-checkpoint"
            or manifest.get("bpe_sha256") != BPE_SHA256 or type(manifest.get("diagnostic_only")) is not bool):
        raise ValueError("unsupported calibration identity")
    oracle = manifest.get("oracle", {})
    if oracle.get("precision") != "float32" or any(oracle.get(key) is not False for key in ("tf32", "autocast", "compile")):
        raise ValueError("calibration must identify original F32 arithmetic")
    if manifest["diagnostic_only"] and not allow_diagnostic:
        raise ValueError("diagnostic-only calibration cannot select quantization parameters")
    identity = dict(manifest)
    expected = identity.pop("calibration_id", None)
    if hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest() != expected:
        raise ValueError("calibration identity hash mismatch")
    paths = {}
    for key in ("statistics", "samples", "dataset"):
        paths[key] = artifact_path(directory, manifest[key]["file"])
        if sha256_file(paths[key]) != manifest[key]["sha256"]:
            raise ValueError(f"calibration {key} hash mismatch")
    dataset = validate_dataset(read_json(paths["dataset"]))
    fitted = [sample["id"] for sample in dataset["samples"] if sample["split"] == "calibration"]
    observed = manifest["observed_samples"]
    if ((not manifest["diagnostic_only"] and [item["id"] for item in observed] != fitted)
            or not observed or any(item["id"] not in fitted or item["vision_linear_calls"] != 128 for item in observed)):
        raise ValueError("calibration did not observe the declared fit split and all vision linears")
    if len({item["id"] for item in observed}) != len(observed):
        raise ValueError("duplicate observed calibration image")
    schema = read_json(Path(__file__).resolve().parents[2] / "tools/convert/sam3_tensor_schema.json")["tensors"]
    expected_layers = {name: shape for name, shape in schema.items()
                       if quantization_module_for_linear_weight(name) == "vision"}
    statistics = read_json(paths["statistics"])
    if statistics.get("schema_version") != 1 or set(statistics["layers"]) != set(expected_layers):
        raise ValueError("calibration vision inventory differs")
    samples = {}
    with np.load(paths["samples"], allow_pickle=False) as archive:
        if set(archive.files) != set(expected_layers):
            raise ValueError("calibration samples inventory differs")
        for name, shape in expected_layers.items():
            array = archive[name]
            layer = statistics["layers"][name]
            activation = layer["activation"]
            if (array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != shape[0]
                    or not 0 < array.shape[0] <= manifest["sample_rows_per_layer"]
                    or not np.isfinite(array).all() or activation["channels"] != shape[0]
                    or activation["reservoir_rows"] != array.shape[0] or activation["token_rows"] < array.shape[0]
                    or layer["calls"] != len(observed) or layer["input_channel_axis"] != -1
                    or layer["weight_layout"] != "output,input"):
                raise ValueError(f"invalid calibration samples or coverage: {name}")
            for field, values, length in (("activation", activation["absmax"], shape[0]),
                                          ("weight input", layer["weight_input_absmax"], shape[0]),
                                          ("weight output", layer["weight_output_absmax"], shape[1])):
                maxima = np.asarray(values, dtype=np.float64)
                if maxima.shape != (length,) or not np.isfinite(maxima).all() or (maxima < 0).any():
                    raise ValueError(f"invalid {field} maxima: {name}")
            samples[name] = array
    return manifest, statistics["layers"], samples


def study(args):
    import torch
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference Python environment")
    if args.output.exists():
        raise FileExistsError("study output already exists")
    manifest, statistics, samples = calibration_inputs(args.calibration, args.allow_diagnostic)
    inputs = {args.calibration / "manifest.json": sha256_file(args.calibration / "manifest.json"),
              Path(__file__): sha256_file(Path(__file__))}
    for key in ("statistics", "samples", "dataset"):
        inputs[artifact_path(args.calibration, manifest[key]["file"])] = manifest[key]["sha256"]
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != manifest["checkpoint_sha256"]:
        raise ValueError("calibration and study checkpoints differ")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    weights = {}
    for source, tensor in state.items():
        name, _ = rename_key(source)
        if name in samples:
            weights[name] = tensor.detach().float().numpy()
    if set(weights) != set(samples):
        raise ValueError("checkpoint does not cover calibration linears")
    del state
    output_layers, scales = {}, {}
    for index, name in enumerate(sorted(samples), 1):
        x, w = samples[name], weights[name]
        layer = statistics[name]
        if w.shape != (len(layer["weight_output_absmax"]), x.shape[1]):
            raise ValueError(f"weight shape differs from calibration: {name}")
        if not np.array_equal(np.abs(w).max(axis=0), np.asarray(layer["weight_input_absmax"], np.float32)):
            raise ValueError(f"weight identity differs from calibration statistics: {name}")
        def multiply(left, right):
            with torch.inference_mode():
                return (torch.from_numpy(left).to(args.device) @ torch.from_numpy(right).to(args.device).T).cpu().numpy()
        reference = multiply(x, w)
        norm = np.linalg.norm(reference.astype(np.float64))
        def error(actual):
            if not np.isfinite(actual).all():
                raise ValueError(f"non-finite linear output: {name}")
            difference = actual.astype(np.float64) - reference
            return {"relative_l2": float(np.linalg.norm(difference) / max(norm, 1e-30)),
                    "max_abs": float(np.abs(difference).max())}
        candidates = []
        for alpha in (None, 0.0, 0.25, 0.5, 0.75, 1.0):
            scale = (np.ones(x.shape[1], np.float32) if alpha is None else
                     smooth_scale(layer["activation"]["absmax"], layer["weight_input_absmax"], alpha))
            label = "identity" if alpha is None else f"smooth-{alpha:g}"
            xs, ws = x / scale, w * scale
            identity_error = error(multiply(xs, ws))
            if identity_error["relative_l2"] > 1e-5:
                raise ValueError(f"floating reparameterization failed: {name}/{label}")
            qw, sw, _ = quantize_rows(ws)
            decoded_w = qw.astype(np.float32) * sw
            row = {"balance": label, "alpha": alpha, "float_identity": identity_error,
                   "weight_only": error(multiply(xs, decoded_w)), "activation_modes": {}}
            for mode in ("per-token", "static-tensor"):
                maximum = np.max(np.asarray(layer["activation"]["absmax"], np.float64) / scale)
                static = np.float32(max(maximum / 127, np.finfo(np.float32).tiny))
                qa, sa, clipped = quantize_rows(xs, None if mode == "per-token" else static)
                decoded_x = qa.astype(np.float32) * sa
                row["activation_modes"][mode] = {"activation_only": error(multiply(decoded_x, ws)),
                                                   "combined": error(multiply(decoded_x, decoded_w)),
                                                   "clipped_elements": clipped,
                                                   "static_scale": float(static) if mode == "static-tensor" else None}
            candidates.append(row)
            scales[name + "/" + label] = scale
        output_layers[name] = {"input_shape": list(x.shape), "weight_shape": list(w.shape), "candidates": candidates}
        print(f"Studied {index}/{len(samples)}: {name}", flush=True)
    if sha256_file(args.checkpoint) != checkpoint_hash:
        raise ValueError("checkpoint changed during numerical study")
    if any(sha256_file(path) != digest for path, digest in inputs.items()):
        raise ValueError("calibration or study source changed during numerical study")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sam3-study-", dir=args.output.parent))
    try:
        np.savez(staging / "channel-scales.npz", **scales)
        report = {"schema_version": 1, "kind": "sam3-layerwise-w8a8-study", "diagnostic_only": True,
                  "scope": "Fit-sample linear errors only; no model accuracy, latency or kernel support acceptance",
                  "calibration_id": manifest["calibration_id"], "calibration_manifest_sha256": sha256_file(args.calibration / "manifest.json"),
                  "checkpoint_sha256": checkpoint_hash, "device": args.device, "torch": torch.__version__,
                  "threads": args.threads, "cuda_runtime": torch.version.cuda,
                  "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                  "numpy": np.__version__, "tf32": False, "weight_quantization": "symmetric per-output-channel INT8",
                  "rounding": "nearest-even; [-127,127]; float32 scales with minimum normal positive scale",
                  "bias_included": False,
                  "source_sha256": sha256_file(Path(__file__)), "layers": output_layers,
                  "scales": {"file": "channel-scales.npz", "sha256": sha256_file(staging / "channel-scales.npz")}}
        write_json(staging / "study.json", report)
        if args.output.exists():
            raise FileExistsError("study output appeared during export")
        staging.rename(args.output)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "calibration", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    try:
        study(args)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f"quantization study failed: {error}\n")


if __name__ == "__main__":
    main()
