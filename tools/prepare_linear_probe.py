#!/usr/bin/env python3
"""Freeze representative SAM vision linear inputs for diagnostic CUDA probes."""

import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

from convert_sam3 import rename_key
from sam3_artifacts import sha256_file, write_json
from study_activation_quantization import calibration_inputs, smooth_scale


def write_case(path, weights, bias, samples, reference, scale, activation_max, tokens, gelu):
    outputs, inner = weights.shape
    if (samples.ndim != 2 or samples.shape[1] != inner or bias.shape != (outputs,)
            or scale.shape != (inner,) or reference.shape != (samples.shape[0], outputs)
            or not 0 < samples.shape[0] <= tokens <= 16384 or max(outputs, inner) > 16384):
        raise ValueError("invalid probe shapes")
    with path.open("xb") as stream:
        stream.write(struct.pack("<8sIIIIIf", b"SLPROB01", outputs, tokens, inner, len(samples), int(gelu), activation_max))
        for array in (weights, bias, scale, samples, reference):
            if not np.isfinite(array).all():
                raise ValueError("non-finite probe input")
            stream.write(np.asarray(array, dtype="<f4").tobytes(order="C"))


def prepare(args):
    import torch
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference environment")
    if args.output.exists():
        raise FileExistsError("probe inputs already exist")
    manifest, statistics, samples = calibration_inputs(args.calibration)
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != manifest["checkpoint_sha256"]:
        raise ValueError("probe checkpoint differs from calibration")
    source_hashes = {str(Path(__file__).with_name(name)): sha256_file(Path(__file__).with_name(name))
                     for name in ("prepare_linear_probe.py", "study_activation_quantization.py", "convert_sam3.py")}
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    wanted = {f"vit.blocks.{args.block}.{suffix}.{kind}" for suffix in
              ("attn.qkv", "attn.proj", "mlp.lin1", "mlp.lin2") for kind in ("weight", "bias")}
    tensors = {}
    for name, tensor in state.items():
        target, _ = rename_key(name)
        if target in wanted:
            tensors[target] = tensor.detach().float().numpy()
    del state
    if set(tensors) != wanted:
        raise ValueError("checkpoint is missing selected linear tensors")
    args.output.mkdir(parents=True)
    cases = []
    for suffix in ("attn.qkv", "attn.proj", "mlp.lin1", "mlp.lin2"):
        name = f"vit.blocks.{args.block}.{suffix}.weight"
        weight, bias = tensors[name], tensors[name.replace(".weight", ".bias")]
        x = samples[name]
        gelu = suffix == "mlp.lin1"
        with torch.inference_mode():
            reference = torch.from_numpy(x) @ torch.from_numpy(weight).T + torch.from_numpy(bias)
            if gelu:
                reference = torch.nn.functional.gelu(reference, approximate="none")
            reference = reference.numpy()
        for alpha in (0.75, 1.0):
            scale = smooth_scale(statistics[name]["activation"]["absmax"], statistics[name]["weight_input_absmax"], alpha)
            maximum = float(np.max(np.asarray(statistics[name]["activation"]["absmax"], np.float64) / scale))
            filename = f"block-{args.block}-{suffix.replace('.', '-')}-smooth-{alpha:g}.bin"
            write_case(args.output / filename, weight, bias, x, reference, scale, maximum, 5184, gelu)
            cases.append({"file": filename, "sha256": sha256_file(args.output / filename), "layer": name,
                          "alpha": alpha, "m": weight.shape[0], "n": 5184, "k": weight.shape[1],
                          "sample_rows": len(x), "gelu_erf": gelu})
    if sha256_file(args.checkpoint) != checkpoint_hash or any(sha256_file(Path(p)) != h for p, h in source_hashes.items()):
        raise ValueError("checkpoint or preparation source changed")
    write_json(args.output / "manifest.json", {"schema_version": 1, "diagnostic_only": True,
               "scope": "Calibration reservoir rows repeated to actual token count; not independent model quality data",
               "checkpoint_sha256": checkpoint_hash, "calibration_id": manifest["calibration_id"],
               "calibration_manifest_sha256": sha256_file(args.calibration / "manifest.json"),
               "torch": torch.__version__, "numpy": np.__version__, "reference": "CPU F32 matmul + bias + optional erf GELU",
               "tf32": False, "source_sha256": source_hashes, "cases": cases})
    print(f"Prepared {len(cases)} diagnostic linear cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--block", type=int, default=0, choices=range(32))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    try:
        prepare(args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"linear probe preparation failed: {error}\n")


if __name__ == "__main__":
    main()
