#!/usr/bin/env python3
"""Bind independent Q8 cache checks to a frozen v2 image recipe."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.validation.precision_acceptance import GATES_PATH, GATES_SHA256, canonical_hash
from tools.validation.verify_precision_q8_cache import check_native_report, sha256_file, verify


CASES = {"original-fpn-0": 256, "original-fpn-1": 256,
         "zero-ties-and-tail-rows": 96}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def loaded_cuda_library(binary):
    result = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True)
    paths = re.findall(r"^\s*libggml-cuda\.so\.0\s+=>\s+(\S+)\s+\(", result.stdout, re.MULTILINE)
    if len(paths) != 1:
        raise ValueError("binary does not resolve exactly one GGML CUDA library")
    return Path(paths[0]).resolve()


def check_recipe_alignment(campaign, parent, candidate, quality, performance,
                           campaign_sha, binary_sha, library_path, library_sha):
    recipes = (parent.get("recipe"), candidate.get("recipe"))
    if (not all(isinstance(value, dict) for value in recipes) or
            any(document.get("complete") is not True for document in (parent, candidate, quality, performance))):
        raise ValueError("frozen recipe reports are incomplete")
    parent_recipe, candidate_recipe = recipes
    if (candidate_recipe.get("backend") != "cuda" or candidate_recipe.get("weight_precision") != "f32" or
            candidate_recipe.get("compute_mode") != "f32" or candidate_recipe.get("feature_cache") != "mixed-q8_0" or
            candidate_recipe.get("storage_profile") != "dense" or candidate_recipe.get("quantization_modules") != [] or
            parent_recipe.get("feature_cache") != "f32"):
        raise ValueError("reports do not describe the F32/mixed-Q8 CUDA recipe")
    without_cache = dict(candidate_recipe)
    without_cache["feature_cache"] = "f32"
    if without_cache != parent_recipe:
        raise ValueError("candidate differs from the parent beyond feature cache")
    parent_hash, candidate_hash = (canonical_hash(recipe) for recipe in recipes)
    if (parent.get("recipe_sha256") != parent_hash or candidate.get("recipe_sha256") != candidate_hash or
            campaign.get("frozen_before_evaluation") is not True or
            candidate_hash not in campaign.get("recipe_sha256", ()) or
            parent_hash not in campaign.get("recipe_sha256", ())):
        raise ValueError("frozen recipe identity differs from the campaign")
    if (any(document.get("campaign_sha256") != campaign_sha for document in
            (parent, candidate, quality, performance)) or
            len({document.get("gates_sha256") for document in
                 (campaign, parent, candidate, quality, performance)}) != 1 or
            len({document.get("dataset_sha256") for document in
                 (campaign, parent, candidate, quality)}) != 1):
        raise ValueError("frozen campaign, gates or dataset identity differs")
    if (quality.get("candidate_recipe_sha256") != candidate_hash or
            quality.get("absolute_quality_status") != "PASS" or
            quality.get("incremental_quality_status") != "PASS" or
            quality.get("quality_status") != "PASS" or
            quality.get("arithmetic_status") != "NOT_RUN" or
            quality.get("deployment_status") != "NOT_RUN" or
            performance.get("candidate_recipe_sha256") != candidate_hash or
            performance.get("baseline_recipe_sha256") != parent_hash or
            performance.get("quality_prerequisites_passed") is not True or
            performance.get("arithmetic_status") != "NOT_RUN" or
            performance.get("deployment_status") != "NOT_RUN"):
        raise ValueError("quality/performance status or recipe identity differs")
    binary_key = next((key for key in candidate.get("artifact_sha256", {})
                       if key.endswith("/sam_precision_image_probe")), None)
    if (binary_key is None or
            any(document.get("artifact_sha256", {}).get(binary_key) != binary_sha
                for document in (parent, candidate)) or
            any(recipe.get("binary_sha256") != binary_sha for recipe in recipes)):
        raise ValueError("frozen image binary identity differs")
    if (any(document.get("artifact_sha256", {}).get(str(library_path.absolute())) != library_sha
            for document in (parent, candidate))):
        raise ValueError("frozen GGML CUDA library identity differs")
    return {"parent_recipe_sha256": parent_hash, "candidate_recipe_sha256": candidate_hash,
            "gates_sha256": candidate["gates_sha256"], "dataset_sha256": candidate["dataset_sha256"],
            "image_binary_path": binary_key}


def reconcile(campaign_path, parent_path, candidate_path, quality_path, performance_path,
              legacy_path, receipts_dir, cache_binary, image_binary, cuda_library):
    campaign_path, parent_path, candidate_path, quality_path, performance_path, legacy_path = map(
        Path, (campaign_path, parent_path, candidate_path, quality_path, performance_path, legacy_path))
    receipts_dir, cache_binary, image_binary, cuda_library = map(
        Path, (receipts_dir, cache_binary, image_binary, cuda_library))
    campaign, parent, candidate, quality, performance, legacy = map(read_json,
        (campaign_path, parent_path, candidate_path, quality_path, performance_path, legacy_path))
    library_resolved = cuda_library.resolve()
    if (loaded_cuda_library(cache_binary) != library_resolved or
            loaded_cuda_library(image_binary) != library_resolved):
        raise ValueError("cache and frozen image probes load different GGML CUDA libraries")
    binary_sha, library_sha = sha256_file(image_binary), sha256_file(cuda_library)
    alignment = check_recipe_alignment(campaign, parent, candidate, quality, performance,
                                       sha256_file(campaign_path), binary_sha, cuda_library, library_sha)
    if sha256_file(GATES_PATH) != GATES_SHA256 or alignment["gates_sha256"] != GATES_SHA256:
        raise ValueError("frozen v2 gate file identity differs")
    if Path(alignment["image_binary_path"]).resolve() != image_binary.resolve():
        raise ValueError("frozen image binary path differs")
    legacy_steps = {(step.get("case"), step.get("mode"), step.get("backend")): step
                    for step in legacy.get("steps", ())}
    cases = []
    bindings = {str(path): sha256_file(path) for path in
                (campaign_path, parent_path, candidate_path, quality_path, performance_path,
                 legacy_path, cache_binary, image_binary, cuda_library,
                 GATES_PATH, Path(__file__), Path(__file__).with_name("verify_precision_q8_cache.py"),
                 Path(__file__).with_name("precision_acceptance.py"))}
    for name, channels in CASES.items():
        receipt_path = receipts_dir / (name + ".json")
        receipt = read_json(receipt_path)
        input_path = legacy_path.parent / "inputs" / (name + ".f32")
        native_dir = legacy_path.parent / (name + "--q8_0--cuda")
        packed_path, decoded_path, report_path = (native_dir / filename for filename in
                                                  ("packed.bin", "decoded.f32", "probe.json"))
        checked = verify(input_path, packed_path, decoded_path, channels)
        native = check_native_report(report_path, checked["rows"], channels)
        old = legacy_steps.get((name, "q8_0", "cuda"))
        if (receipt.get("passed") is not True or checked.get("passed") is not True or
                receipt.get("verifier_sha256") != bindings[str(Path(__file__).with_name("verify_precision_q8_cache.py"))] or
                any(receipt.get(key) != checked[key] for key in
                    ("rows", "channels", "blocks", "scale_tie_allowances", "integer_tie_allowances",
                     "scale_errors", "integer_errors", "decoded_bit_errors", "sha256")) or
                receipt.get("native_probe") != native or
                receipt.get("identity_sha256", {}).get(str(cache_binary)) != sha256_file(cache_binary) or
                receipt.get("identity_sha256", {}).get(str(cuda_library)) != library_sha or
                receipt.get("identity_sha256", {}).get(str(image_binary)) != binary_sha or
                old is None or old.get("packed_sha256") != checked["sha256"][str(packed_path)] or
                old.get("decoded_sha256") != checked["sha256"][str(decoded_path)] or
                old.get("probe_sha256") != native["sha256"] or
                legacy.get("artifact_sha256", {}).get(str(input_path.absolute())) !=
                    checked["sha256"][str(input_path)]):
            raise ValueError("independent receipt differs from retained native cache evidence")
        for path in (receipt_path, input_path, packed_path, decoded_path, report_path):
            bindings[str(path)] = sha256_file(path)
        cases.append({"case": name, "blocks": checked["blocks"],
                      "scale_tie_allowances": checked["scale_tie_allowances"],
                      "integer_tie_allowances": checked["integer_tie_allowances"],
                      "retained_v1_status": old.get("passed"),
                      "independent_status": "PASS", "receipt": str(receipt_path),
                      "receipt_sha256": bindings[str(receipt_path)]})
    return {"schema_version": 1, "kind": "sam3-frozen-q8-cache-arithmetic-alignment",
            "complete": True, "cache_component_status": "PASS",
            "whole_recipe_arithmetic_status": "NOT_RUN", "deployment_status": "NOT_RUN",
            "scope": "Frozen v2 F32/mixed-Q8 CUDA recipe's cache codec, retained original FPN 0/1 and boundary inputs only",
            "campaign_sha256": sha256_file(campaign_path), **alignment,
            "ggml_cuda_sha256": library_sha, "cases": cases,
            "total_blocks": sum(case["blocks"] for case in cases),
            "artifact_sha256": bindings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("campaign", "parent_export", "candidate_export", "quality", "performance",
                     "legacy_codec", "receipts_dir", "cache_binary", "image_binary", "cuda_library"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = reconcile(args.campaign, args.parent_export, args.candidate_export, args.quality,
                       args.performance, args.legacy_codec, args.receipts_dir,
                       args.cache_binary, args.image_binary, args.cuda_library)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
