#!/usr/bin/env python3
"""Export a supplementary Meta FP32 oracle with weights decoded from an actual quantized GGUF."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from tools.convert.sam3_artifacts import BPE_SHA256, SAM3_REVISION, read_json, sha256_file, write_json
from tools.convert.sam3_gguf import inspect_tensors, read_gguf, tensor_schema, validate_metadata
from tools.validation.validate_image import (QUANTIZED_PRECISIONS, quantization_selection,
                            validate_quantization_sidecar,
                            validate_gguf_module_metadata,
                            validate_schema4_tensor_module_records)


def tensor_error_metrics(actual, reference, zero_norm_limit):
    actual = np.asarray(actual).reshape(-1)
    reference = np.asarray(reference).reshape(-1)
    if actual.size == 0 or actual.shape != reference.shape:
        raise ValueError("weight comparison requires equal, nonempty tensors")
    error_squared = reference_squared = maximum_absolute = 0.0
    for offset in range(0, actual.size, 1024 * 1024):
        left = np.asarray(actual[offset:offset + 1024 * 1024], dtype=np.float64)
        right = np.asarray(reference[offset:offset + 1024 * 1024], dtype=np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("weight comparison received non-finite values")
        delta = left - right
        error_squared += float(np.dot(delta, delta))
        reference_squared += float(np.dot(right, right))
        maximum_absolute = max(maximum_absolute, float(np.abs(delta).max()))
    reference_norm = reference_squared ** 0.5
    normalized_l2 = error_squared ** 0.5 / reference_norm if reference_norm > zero_norm_limit else None
    return {"normalized_l2": normalized_l2, "reference_norm": reference_norm,
            "maximum_absolute_error": maximum_absolute}


def make_dequantized_checkpoint(model_path, checkpoint_path, bpe_path, scratch, source_hf_revision):
    import gguf
    import numpy as np
    import torch

    model_path = Path(model_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    bpe_path = Path(bpe_path).resolve()
    model_manifest_path = model_path.with_suffix(model_path.suffix + ".manifest.json")
    model_manifest = read_json(model_manifest_path)
    precision = model_manifest.get("precision")
    if precision not in QUANTIZED_PRECISIONS:
        raise ValueError("model must use an exact supported quantized image precision")
    storage_profile = model_manifest.get("storage_profile")
    schema_version = model_manifest.get("sam_schema_version")
    schema4 = schema_version == 4
    modules = model_manifest.get("quantization_modules") if schema4 else None
    selection = quantization_selection(precision, storage_profile, modules)
    profile, gates = selection["profile"], selection["gates"]
    if (model_manifest.get("schema_version") != 1 or model_manifest.get("architecture") != "sam3"
            or model_manifest.get("container_format") != "gguf" or model_manifest.get("container_version") != 3
            or schema_version not in (3, 4) or model_manifest.get("task") != "image"
            or model_manifest.get("storage_profile") != profile["storage_profile"]
            or model_manifest.get("arithmetic_profile") != "ggml-quantized-native-v1"):
        raise ValueError("GGUF sidecar does not describe a supported quantized image model")
    validate_quantization_sidecar(model_manifest, allow_custom_quantization=True)
    if sha256_file(checkpoint_path) != model_manifest.get("checkpoint", {}).get("sha256"):
        raise ValueError("original Meta checkpoint SHA-256 differs from the GGUF provenance")
    if sha256_file(model_path) != model_manifest.get("output", {}).get("sha256"):
        raise ValueError("GGUF SHA-256 differs from its sidecar")
    if model_manifest["output"].get("bytes") != model_path.stat().st_size:
        raise ValueError("GGUF size differs from its sidecar")
    if sha256_file(bpe_path) != BPE_SHA256 or model_manifest.get("bpe", {}).get("sha256") != BPE_SHA256:
        raise ValueError("BPE asset does not match the pinned official Meta tokenizer")
    if not isinstance(source_hf_revision, str) or not source_hf_revision.strip():
        raise ValueError("the original checkpoint Hugging Face revision must be recorded")

    schema = read_json(Path(__file__).resolve().parents[2] / "tools/convert/sam3_tensor_schema.json")
    expected = tensor_schema(schema, "image")
    recorded = {item["name"]: item for item in model_manifest.get("tensors", [])}
    if len(recorded) != len(model_manifest.get("tensors", [])) or set(recorded) != set(expected):
        raise ValueError("GGUF sidecar tensor inventory differs from the pinned image schema")
    source_names = [item.get("source_name") for item in model_manifest["tensors"]]
    if (any(not isinstance(name, str) or not name for name in source_names)
            or len(set(source_names)) != len(source_names)
            or any(not isinstance(item.get("source_dtype"), str) or not item["source_dtype"]
                   or item.get("output_dtype") != item.get("dtype")
                   or item.get("conversion") != ("quantized-from-original-f32"
                                                   if item.get("dtype") in QUANTIZED_PRECISIONS else "preserved-f32")
                   for item in model_manifest["tensors"])):
        raise ValueError("GGUF sidecar lacks unique original Meta names/dtypes or actual output dtype records")

    if schema4:
        validate_schema4_tensor_module_records(
            model_manifest["tensors"], precision, storage_profile, modules)

    reader = read_gguf(model_path)
    if schema4:
        validate_gguf_module_metadata(reader, modules)
        validate_metadata(reader, precision, model_manifest["checkpoint"]["sha256"], "image",
                          storage_profile, modules)
        actual = inspect_tensors(reader, precision, expected,
                                 {name: item["shape"] for name, item in recorded.items()},
                                 storage_profile, modules)
    else:
        validate_metadata(reader, precision, model_manifest["checkpoint"]["sha256"], "image", storage_profile)
        actual = inspect_tensors(reader, precision, expected,
                                 {name: item["shape"] for name, item in recorded.items()}, storage_profile)
    for item in actual:
        expected_keys = ("shape", "ggml_shape", "dtype", "offset", "bytes", "sha256")
        if any(item[key] != recorded[item["name"]].get(key) for key in expected_keys):
            raise ValueError(f"{item['name']}: actual GGUF tensor does not match its sidecar")
        if recorded[item["name"]].get("output_dtype") != item["dtype"]:
            raise ValueError(f"{item['name']}: sidecar output dtype differs from the actual GGUF")
        for key in ("source_name", "source_dtype"):
            if not isinstance(recorded[item["name"]].get(key), str) or not recorded[item["name"]][key]:
                raise ValueError(f"{item['name']}: sidecar lacks its original Meta {key}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("original Meta checkpoint must contain a tensor state dictionary")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    quantized_items = [item for item in actual if item["dtype"] in {"q8_0", "q6_k", "q5_k", "q4_k"}]
    if not quantized_items:
        raise ValueError("GGUF contains no quantized tensors to decode")
    zero_norm_limit = gates["comparison"]["zero_norm_reference_threshold"]
    weight_l2_limit = profile["normalized_l2_max"]
    zero_norm_max_abs = profile["zero_norm_max_abs"]
    weight_metrics = {}
    for item in quantized_items:
        source_name = recorded[item["name"]]["source_name"]
        source = state.get(source_name)
        if not isinstance(source, torch.Tensor) or list(source.shape) != item["shape"]:
            raise ValueError(f"{item['name']}: source Meta tensor is missing or has a different shape")
        tensor = tensors[item["name"]]
        values = gguf.dequantize(tensor.data, tensor.tensor_type)
        values = np.asarray(values, dtype=np.float32)
        if values.size != source.numel() or not np.isfinite(values).all():
            raise ValueError(f"{item['name']}: decoded GGUF tensor has an invalid size or non-finite values")
        errors = tensor_error_metrics(values, source.detach().cpu().float().numpy(), zero_norm_limit)
        passed = (errors["maximum_absolute_error"] <= zero_norm_max_abs
                  if errors["normalized_l2"] is None else errors["normalized_l2"] <= weight_l2_limit)
        weight_metrics[item["name"]] = {"source_name": source_name, "dtype": item["dtype"],
                                         **errors, "passed": passed}
        decoded = torch.from_numpy(values.reshape(item["shape"]).copy()).to(dtype=source.dtype)
        state[source_name] = decoded
    del reader, tensors

    checkpoint_out = Path(scratch) / "dequantized-meta-checkpoint.pt"
    torch.save({"model": state}, checkpoint_out)
    checkpoint_sha256 = sha256_file(checkpoint_out)
    quantizer = model_manifest.get("quantizer", {})
    provenance = {
        "schema_version": 1,
        "restored_checkpoint": {"file": checkpoint_out.name, "sha256": checkpoint_sha256},
        "container": {"precision": "f32", "hf_revision": source_hf_revision},
        "reconstruction": {
            "kind": "dequantized-gguf-weights-v1",
            "source_meta_checkpoint_sha256": model_manifest["checkpoint"]["sha256"],
            "sam3_revision": SAM3_REVISION,
            "quantized_precision": precision,
            "storage_profile": storage_profile,
            "gate_set": gates["gate_set"],
            "gates_sha256": selection["gates_sha256"],
            "decoded_tensor_count": len(quantized_items),
            "quantizer": quantizer,
        },
        "weight_compression": {
            "gate_set": gates["gate_set"],
            "gates_sha256": selection["gates_sha256"],
            "normalized_l2_max": weight_l2_limit,
            "zero_norm_max_abs": zero_norm_max_abs,
            "passed": all(item["passed"] for item in weight_metrics.values()),
            "tensors": weight_metrics,
        },
        "quantized_gguf": {
            "file": model_path.name,
            "sha256": model_manifest["output"]["sha256"],
            "manifest_sha256": sha256_file(model_manifest_path),
            "precision": precision,
            "storage_profile": storage_profile,
        },
    }
    if schema4:
        provenance["reconstruction"]["quantization_modules"] = modules
        provenance["weight_compression"]["quantization_modules"] = modules
        provenance["quantized_gguf"]["quantization_modules"] = modules
    provenance_path = Path(scratch) / "supplementary-weights.json"
    write_json(provenance_path, provenance)
    return checkpoint_out, provenance_path


def export(args):
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference-runtime Python environment")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic reference: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sam3-dequantized-", dir=output.parent) as temporary:
        checkpoint, provenance = make_dequantized_checkpoint(
            args.model, args.checkpoint, args.bpe, temporary, args.source_hf_revision)
        command = [sys.executable, str(Path(__file__).with_name("export_reference.py")),
                   "--checkpoint", str(checkpoint), "--cases", str(args.cases.resolve()),
                   "--device", args.device, "--output", str(output),
                   "--supplementary-weights-manifest", str(provenance),
                   "--threads", str(args.threads)]
        if args.sam3_source:
            command.extend(("--sam3-source", str(args.sam3_source.resolve())))
        if args.sam3_runtime_source:
            command.extend(("--sam3-runtime-source", str(args.sam3_runtime_source.resolve())))
        if args.input_root:
            command.extend(("--input-root", str(args.input_root.resolve())))
        command.extend(("--bpe", str(args.bpe.resolve())))
        subprocess.run(command, check=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="actual quantized schema-3/schema-4 GGUF")
    parser.add_argument("--checkpoint", required=True, type=Path, help="original Meta checkpoint used to create the GGUF")
    parser.add_argument("--bpe", required=True, type=Path, help="pinned original Meta BPE asset")
    parser.add_argument("--source-hf-revision", required=True,
                        help="pinned Hugging Face revision of the original checkpoint")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--sam3-source", type=Path, default=os.environ.get("SAM3_SOURCE_DIR"))
    parser.add_argument("--sam3-runtime-source", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("thread count must be positive")
    try:
        print(f"Exporting supplementary decoded-weight Meta diagnostic to {export(args)}")
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"quantized reference export failed: {error}\n")


if __name__ == "__main__":
    main()
