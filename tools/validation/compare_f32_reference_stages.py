#!/usr/bin/env python3
"""Diagnose native F32 stage differences against an official SAM 3 reference."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.sam3_artifacts import (REQUIRED_TENSORS, read_array, read_json,
                                           read_tensor_index, sha256_file)
from tools.validation.validate_image import tensor_error


def cuda_library(binary):
    loaded = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True)
    paths = re.findall(r"^\s*libggml-cuda\.so\.0\s+=>\s+(\S+)\s+\(",
                       loaded.stdout, re.MULTILINE)
    if len(paths) != 1:
        raise ValueError("test driver must load exactly one GGML CUDA library")
    return Path(paths[0]).resolve()


def compare(reference_root, native_root, binary, model, expected_binary_sha256,
            expected_cuda_sha256):
    reference_root, native_root, binary, model = map(
        Path, (reference_root, native_root, binary, model))
    reference_manifest_path = reference_root / "manifest.json"
    reference_manifest = read_json(reference_manifest_path)
    model_manifest_path = model.with_suffix(model.suffix + ".manifest.json")
    model_manifest = read_json(model_manifest_path)
    library = cuda_library(binary)
    if (sha256_file(binary) != expected_binary_sha256 or
            sha256_file(library) != expected_cuda_sha256 or
            reference_manifest.get("reference_kind") != "official-checkpoint" or
            reference_manifest.get("oracle", {}).get("precision") != "float32" or
            reference_manifest.get("oracle", {}).get("device") != "cuda" or
            model_manifest.get("precision") != "f32" or
            model_manifest.get("output", {}).get("sha256") != sha256_file(model) or
            model_manifest.get("checkpoint", {}).get("sha256") !=
            reference_manifest.get("checkpoint", {}).get("sha256")):
        raise ValueError("frozen driver/library/model or reference identity differs")
    cases_path = reference_root / reference_manifest["cases_manifest"]["file"]
    if sha256_file(cases_path) != reference_manifest["cases_manifest"]["sha256"]:
        raise ValueError("reference case manifest hash differs")
    bindings = {str(path): sha256_file(path) for path in
                (reference_manifest_path, cases_path, model_manifest_path, model,
                 binary, library, Path(__file__),
                 Path(__file__).resolve().parents[1] / "convert/sam3_artifacts.py",
                 Path(__file__).with_name("validate_image.py"))}
    cases = []
    stage_max = {name: {"normalized_l2": 0.0, "maximum_absolute_error": 0.0}
                 for name in REQUIRED_TENSORS}
    identifiers = set()
    for case in reference_manifest["cases"]:
        case_id = case["id"]
        if case_id in identifiers or not re.fullmatch(r"[a-z0-9-]+", case_id):
            raise ValueError("invalid or duplicate case identifier")
        identifiers.add(case_id)
        reference_dir = reference_root / case["directory"]
        actual_dir = native_root / case_id
        input_path = reference_dir / case["input"]
        reference_index_path = reference_dir / "tensors.json"
        reference_results_path = reference_dir / "results.json"
        actual_index_path = actual_dir / "tensors.json"
        actual_results_path = actual_dir / "results.json"
        if (sha256_file(input_path) != case["input_sha256"] or
                sha256_file(reference_index_path) != case["tensors_sha256"] or
                sha256_file(reference_results_path) != case["results_sha256"]):
            raise ValueError(f"{case_id}: reference case files changed")
        actual_results = read_json(actual_results_path)
        runtime = actual_results.get("runtime", {})
        if (actual_results.get("backend") != "cuda" or
                actual_results.get("precision") != "f32" or
                actual_results.get("prompt") != case["prompt"] or
                actual_results.get("storage_profile") != "" or
                actual_results.get("arithmetic_profile") != "" or
                actual_results.get("quantization_modules") != [] or
                actual_results.get("tokenizer_compatibility_repaired") is not False or
                runtime.get("cuda_nodes", 0) <= 0 or
                runtime.get("cpu_nodes") != 0 or runtime.get("metal_nodes") != 0):
            raise ValueError(f"{case_id}: native output is not the CUDA F32 path")
        reference_index = read_tensor_index(reference_dir)
        actual_index = read_tensor_index(actual_dir, require_hash=False)
        if reference_index["token_ids"] != actual_index["token_ids"]:
            raise ValueError(f"{case_id}: token IDs differ")
        for path in (input_path, reference_index_path, reference_results_path,
                     actual_index_path, actual_results_path):
            bindings[str(path)] = sha256_file(path)
        tensors = {}
        for name in REQUIRED_TENSORS:
            reference_meta = reference_index["tensors"][name]
            actual_meta = actual_index["tensors"][name]
            reference = read_array(reference_dir, reference_meta, "float32", require_hash=False)
            actual = read_array(actual_dir, actual_meta, "float32", require_hash=False)
            error = tensor_error(actual, reference)
            tensors[name] = error
            for metric in stage_max[name]:
                value = error[metric]
                if value is not None:
                    stage_max[name][metric] = max(stage_max[name][metric], value)
            for directory, metadata in ((reference_dir, reference_meta),
                                        (actual_dir, actual_meta)):
                path = directory / metadata["file"]
                bindings[str(path)] = sha256_file(path)
        cases.append({"id": case_id, "prompt": case["prompt"], "token_ids_match": True,
                      "cuda_nodes": runtime["cuda_nodes"], "tensors": tensors})
    return {"schema_version": 1, "kind": "sam3-frozen-f32-official-stage-diagnosis",
            "complete": True, "reference_kind": "official-checkpoint",
            "same_operand_arithmetic_gate": "NOT_RUN",
            "whole_recipe_arithmetic_status": "NOT_RUN",
            "reference_checkpoint_sha256": reference_manifest["checkpoint"]["sha256"],
            "native_model_sha256": bindings[str(model)],
            "driver_sha256": bindings[str(binary)],
            "ggml_cuda_sha256": bindings[str(library)],
            "cases": cases, "stage_max": stage_max, "artifact_sha256": bindings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference_root", "native_root", "binary", "model"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--expected-cuda-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.reference_root, args.native_root, args.binary, args.model,
                     args.expected_binary_sha256, args.expected_cuda_sha256)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"cases": len(result["cases"]), "stage_max": result["stage_max"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
