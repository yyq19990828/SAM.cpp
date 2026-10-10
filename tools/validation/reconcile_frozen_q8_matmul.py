#!/usr/bin/env python3
"""Bind independent Q8 CUDA dot checks to the frozen v2 GGML library."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.sam3_artifacts import read_json, sha256_file
from tools.validation.verify_q8_mmq_dots import verify as verify_mmq
from tools.validation.verify_q8_mmvq_dots import verify as verify_mmvq


MMVQ = {"qkv": (3072, 3, 1024), "attn-proj": (1024, 3, 1024),
        "mlp-lin1": (4736, 3, 1024), "mlp-lin2": (1024, 3, 4736),
        "tail": (17, 3, 4736)}
MMQ = {"qkv": (3072, 64, 1024), "attn-proj": (1024, 64, 1024),
       "mlp-lin1": (4736, 64, 1024), "mlp-lin2": (1024, 64, 4736),
       "tail": (17, 5184, 4736), "zero": (2, 64, 128),
       "max-k-ties": (17, 65, 16384)}


def loaded_cuda_library(binary):
    result = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True)
    matches = re.findall(r"^\s*libggml-cuda\.so\.0\s+=>\s+(\S+)\s+\(",
                         result.stdout, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"{binary}: expected one loaded GGML CUDA library")
    return Path(matches[0]).resolve()


def profile_kernels(report, run_report, expected_shape, required):
    run = read_json(run_report)
    if (run.get("complete") is not True or run.get("mode") != "ggml-q8" or
            run.get("cuda_nodes", 0) <= 0 or run.get("cpu_nodes") != 0 or
            tuple(run.get(key) for key in ("m", "n", "k")) != expected_shape):
        raise ValueError(f"{run_report}: profiled probe shape or backend differs")
    result = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", str(report)],
                            capture_output=True, text=True, check=True)
    if not all(kernel in result.stdout for kernel in required):
        raise ValueError(f"{report}: expected CUDA kernels were not profiled")
    return {"path": str(report), "sha256": sha256_file(report),
            "run_report": str(run_report), "run_report_sha256": sha256_file(run_report),
            "shape": list(expected_shape), "kernels": list(required)}


def reconcile(root, mmvq_inputs, mmq_inputs, image_binary, alignment_path,
              vision_export_path, full_export_path, vision_quality_path, full_quality_path):
    root, mmvq_inputs, mmq_inputs, image_binary = map(
        Path, (root, mmvq_inputs, mmq_inputs, image_binary))
    alignment_path, vision_export_path, full_export_path, vision_quality_path, full_quality_path = map(
        Path, (alignment_path, vision_export_path, full_export_path,
               vision_quality_path, full_quality_path))
    alignment, vision_export, full_export, vision_quality, full_quality = map(
        read_json, (alignment_path, vision_export_path, full_export_path,
                   vision_quality_path, full_quality_path))
    library = loaded_cuda_library(image_binary)
    image_entries = [(path, digest) for path, digest in
                     alignment.get("artifact_sha256", {}).items()
                     if Path(path).resolve() == image_binary.resolve()]
    if (alignment.get("complete") is not True or
            alignment.get("image_binary_path") != str(image_binary.resolve()) or
            alignment.get("ggml_cuda_sha256") != sha256_file(library) or
            alignment.get("whole_recipe_arithmetic_status") != "NOT_RUN" or
            len(image_entries) != 1 or image_entries[0][1] != sha256_file(image_binary)):
        raise ValueError("frozen image recipe does not identify this GGML library")
    if any(report.get("complete") is not True or report.get("quality_status") != "FAIL"
           for report in (vision_quality, full_quality)):
        raise ValueError("historical Q8 development quality status changed")
    for export, quality, modules in ((vision_export, vision_quality, ["vision"]),
                                     (full_export, full_quality,
                                      ["vision", "text", "fusion", "decoder"])):
        recipe = export.get("recipe", {})
        library_entries = [(path, digest) for path, digest in
                           export.get("artifact_sha256", {}).items()
                           if Path(path).resolve() == library]
        if (export.get("complete") is not True or recipe.get("backend") != "cuda" or
                recipe.get("weight_precision") != "q8_0" or
                recipe.get("compute_mode") != "f32" or
                recipe.get("quantization_modules") != modules or
                recipe.get("binary_sha256") != sha256_file(image_binary) or
                quality.get("candidate_recipe_sha256") != export.get("recipe_sha256") or
                quality.get("campaign_sha256") != export.get("campaign_sha256") or
                len(library_entries) != 1 or library_entries[0][1] != sha256_file(library)):
            raise ValueError("historical Q8 image recipe/library identity differs")
    binary_dir = root / "bin"
    binaries = {"linear": binary_dir / "sam_ggml_linear_probe",
                "mmvq_staging": binary_dir / "sam_cuda_q8_rhs_probe",
                "mmq_staging": binary_dir / "sam_cuda_q8_mmq_rhs_probe"}
    if any(loaded_cuda_library(binary) != library for binary in binaries.values()):
        raise ValueError("an isolated probe loads a different CUDA library")
    identities = [Path(__file__), image_binary, library, alignment_path,
                  vision_export_path, full_export_path,
                  vision_quality_path, full_quality_path,
                  Path(__file__).with_name("verify_q8_mmvq_dots.py"),
                  Path(__file__).with_name("verify_q8_mmq_dots.py"),
                  Path(__file__).with_name("verify_q8_rhs_staging.py"),
                  Path(__file__).with_name("verify_q8_mmq_rhs_staging.py"),
                  Path(__file__).resolve().parents[1] / "benchmark/ggml_linear_probe.cpp",
                  Path(__file__).resolve().parents[1] / "benchmark/linear_probe.hpp",
                  Path(__file__).resolve().parents[1] / "quantize/cuda_q8_rhs_probe.cu",
                  Path(__file__).resolve().parents[1] / "quantize/cuda_q8_mmq_rhs_probe.cu",
                  *(Path(__file__).resolve().parents[2] / "src/runtime/ggml" / name
                    for name in ("graph.hpp", "resources.hpp", "runtime.hpp", "workspace.hpp")),
                  *(library.parents[3] / "sam-ggml-src" / name
                    for name in ("include/ggml.h", "include/ggml-backend.h",
                                 "src/ggml-cuda/mmq.cuh", "src/ggml-cuda/quantize.cuh")),
                  *binaries.values()]
    bindings = {str(path): sha256_file(path) for path in identities}
    cases = []
    for mode, expected, source, verifier, staging_binary in (
            ("mmvq", MMVQ, mmvq_inputs, verify_mmvq, binaries["mmvq_staging"]),
            ("mmq", MMQ, mmq_inputs, verify_mmq, binaries["mmq_staging"])):
        for name, shape in expected.items():
            input_path = source / ("adversarial" if name in ("zero", "max-k-ties") else
                                   ("matmul-inputs" if mode == "mmvq" else "inputs")) / (name + ".bin")
            directory = root / f"{name}-{mode}"
            receipt_path = directory / "verified.json"
            receipt = read_json(receipt_path)
            checked = verifier(input_path, directory)
            receipt_identity = receipt.get("identity_sha256", {})
            cuda_entries = [(path, digest) for path, digest in receipt_identity.items()
                            if Path(path).resolve() == library]
            if (receipt.get("passed") is not True or checked.get("passed") is not True or
                    tuple(checked[key] for key in ("m", "n", "k")) != shape or
                    any(receipt.get(key) != value for key, value in checked.items()) or
                    receipt_identity.get(str(binaries["linear"])) != sha256_file(binaries["linear"]) or
                    receipt_identity.get(str(staging_binary)) != sha256_file(staging_binary) or
                    len(cuda_entries) != 1 or cuda_entries[0][1] != sha256_file(library) or
                    len(receipt_identity) != 3 or
                    receipt.get("verifier_sha256") !=
                    sha256_file(Path(__file__).with_name(f"verify_q8_{mode}_dots.py"))):
                raise ValueError(f"{name}-{mode}: independent receipt differs")
            for path, digest in checked["artifact_sha256"].items():
                if sha256_file(path) != digest:
                    raise ValueError(f"{path}: retained operand changed")
                bindings[path] = digest
            bindings[str(receipt_path)] = sha256_file(receipt_path)
            cases.append({"name": name, "mode": mode, "m": shape[0], "n": shape[1],
                          "k": shape[2], "checked_dots": checked["checked_dots"],
                          "staging_blocks": checked["staging_blocks"],
                          "relative_l2": checked["relative_l2"],
                          "maximum_absolute_error": checked["maximum_absolute_error"],
                          "receipt": str(receipt_path),
                          "receipt_sha256": bindings[str(receipt_path)]})
    profiles = {
        "mmvq": profile_kernels(root / "profile-mmvq.nsys-rep",
                                root / "profile-mmvq-run/probe.json", MMVQ["qkv"],
                                ("quantize_q8_1", "mul_mat_vec_q")),
        "mmq": profile_kernels(root / "profile-mmq.nsys-rep",
                               root / "profile-mmq-run/probe.json", MMQ["qkv"],
                               ("quantize_mmq_q8_1", "mul_mat_q<")),
    }
    for profile in profiles.values():
        bindings[profile["path"]] = profile["sha256"]
        bindings[profile["run_report"]] = profile["run_report_sha256"]
    return {"schema_version": 1, "kind": "sam3-frozen-v2-q8-cuda-matmul-arithmetic",
            "complete": True, "component_status": "PASS",
            "whole_recipe_arithmetic_status": "NOT_RUN", "deployment_status": "NOT_RUN",
            "vision_q8_development_quality_status": "FAIL",
            "full_q8_development_quality_status": "FAIL",
            "scope": "Selected Q8_0 CUDA MMVQ/MMQ same-operand probes with the frozen v2 GGML library",
            "library_anchor_cache_recipe_sha256": alignment["candidate_recipe_sha256"],
            "vision_q8_recipe_sha256": vision_export["recipe_sha256"],
            "full_q8_recipe_sha256": full_export["recipe_sha256"],
            "ggml_cuda_sha256": sha256_file(library),
            "total_dots": sum(case["checked_dots"] for case in cases),
            "total_staging_blocks": sum(case["staging_blocks"] for case in cases),
            "worst_nonzero_relative_l2": max(case["relative_l2"] for case in cases
                                              if case["relative_l2"] is not None),
            "cases": cases, "profiles": profiles, "artifact_sha256": bindings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "mmvq_inputs", "mmq_inputs", "image_binary", "alignment",
                 "vision_export", "full_export", "vision_quality", "full_quality"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = reconcile(args.root, args.mmvq_inputs, args.mmq_inputs,
                       args.image_binary, args.alignment, args.vision_export,
                       args.full_export, args.vision_quality, args.full_quality)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in
                      ("component_status", "total_dots", "total_staging_blocks",
                       "worst_nonzero_relative_l2")}, indent=2))


if __name__ == "__main__":
    main()
