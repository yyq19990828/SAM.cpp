#!/usr/bin/env python3
"""Compare every F32 image GGUF payload with its original checkpoint tensor."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.sam3_artifacts import read_json, sha256_file
from tools.convert.sam3_gguf import read_gguf


def source_values(source, item):
    import numpy as np
    import torch

    conversion = item["conversion"]
    if conversion == "complex-real-pairs":
        if source.dtype != torch.complex64 or not item["name"].endswith(".attn.freqs_cis"):
            raise ValueError(f"{item['name']}: invalid complex-pair conversion")
        source = torch.view_as_real(source)
    elif conversion != "real" or source.dtype != torch.float32:
        raise ValueError(f"{item['name']}: unsupported F32 source conversion")
    values = source.detach().cpu().numpy()
    if item["name"] == "vit.pos_embed":
        if values.shape != (1, 577, 1024):
            raise ValueError("vit.pos_embed: original shape changed")
        values = values[:, 1:, :].reshape(24, 24, 1024)
    if list(values.shape) != item["shape"]:
        raise ValueError(f"{item['name']}: original tensor shape differs")
    return np.ascontiguousarray(values, dtype="<f4")


def verify(model_path, checkpoint_path):
    import gguf
    import numpy as np
    import torch

    model_path, checkpoint_path = Path(model_path), Path(checkpoint_path)
    manifest_path = model_path.with_suffix(model_path.suffix + ".manifest.json")
    manifest = read_json(manifest_path)
    model_sha, checkpoint_sha = sha256_file(model_path), sha256_file(checkpoint_path)
    if (manifest.get("task") != "image" or manifest.get("precision") != "f32" or
            manifest.get("output", {}).get("sha256") != model_sha or
            manifest.get("checkpoint", {}).get("sha256") != checkpoint_sha):
        raise ValueError("F32 GGUF/checkpoint identity differs from its sidecar")
    recorded = {item["name"]: item for item in manifest["tensors"]}
    reader = read_gguf(model_path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if (len(recorded) != len(manifest["tensors"]) or
            len(tensors) != len(reader.tensors) or set(tensors) != set(recorded)):
        raise ValueError("GGUF tensor inventory differs from the sidecar")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no tensor state dictionary")
    source_names = [item["source_name"] for item in recorded.values()]
    if len(set(source_names)) != len(source_names):
        raise ValueError("sidecar repeats an original tensor")
    mismatches = []
    total_values = 0
    conversions = {"real": 0, "complex-real-pairs": 0}
    for name, item in recorded.items():
        source = state.get(item["source_name"])
        tensor = tensors[name]
        if not isinstance(source, torch.Tensor) or tensor.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(f"{name}: missing source or non-F32 GGUF payload")
        expected = source_values(source, item)
        actual = np.ascontiguousarray(tensor.data, dtype="<f4")
        if (expected.size != actual.size or expected.nbytes != item["bytes"] or
                str(source.dtype) != item["source_dtype"]):
            raise ValueError(f"{name}: source/GGUF size or dtype differs")
        expected_sha = hashlib.sha256(expected.tobytes()).hexdigest()
        actual_sha = hashlib.sha256(actual.tobytes()).hexdigest()
        if actual_sha != item["sha256"]:
            raise ValueError(f"{name}: GGUF payload differs from sidecar")
        if expected_sha != actual_sha:
            mismatches.append(name)
        total_values += expected.size
        conversions[item["conversion"]] += 1
    return {"schema_version": 1, "kind": "sam3-f32-gguf-original-weight-parity",
            "complete": True, "passed": not mismatches,
            "tensor_count": len(recorded), "value_count": total_values,
            "conversions": conversions, "mismatches": mismatches,
            "model_sha256": model_sha, "model_manifest_sha256": sha256_file(manifest_path),
            "checkpoint_sha256": checkpoint_sha,
            "verifier_sha256": sha256_file(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.model, args.checkpoint)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
