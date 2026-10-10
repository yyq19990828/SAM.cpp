#!/usr/bin/env python3
"""Run the seven frozen image regressions with versioned spatial, zero-tail gates.

The reference subcommand transcodes complete original F32 tensor dumps. It must
reproduce every stored deployed mask, score and box exactly before those raw
logits can supply the additional ranked masks. It never invents missing masks.
This supplements, rather than replaces, existing validate_image.py acceptance.
"""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.validation.precision_acceptance import (GATES_SHA256, canonical_hash, combine_statuses, compare_objects,
                                  diagnostic_summary, gate_identity, load_gates, object_gates, policy_arguments,
                                  quality_profile, validate_output)
from tools.maintenance.precision_artifacts import (archive_sources, native_recipe, native_snapshot, packages,
                                 runtime_environment, source_snapshot, verify_export_artifacts)
from tools.convert.sam3_artifacts import (BPE_SHA256, SAM3_REVISION, artifact_path, load_case_manifest, read_array,
                           read_json, read_tensor_index, sha256_file, validate_cuda_oracle_provenance,
                           verify_run_artifacts, write_json)


CASES_PATH = Path(__file__).resolve().parents[2] / "tests/data/sam3-image-cases.json"
CASES_SHA256 = "543a2f0a696c6540bf9201c51fb56f3bea979f63c1139f74da78aab4773b9f45"


def fixed_cases():
    if sha256_file(CASES_PATH) != CASES_SHA256:
        raise ValueError("the original seven-case regression contract changed")
    return load_case_manifest(CASES_PATH)["cases"]


def original_bundle(directory):
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("architecture") != "sam3"
            or manifest.get("sam3_revision") != SAM3_REVISION
            or manifest.get("reference_kind") != "official-checkpoint"
            or manifest.get("eligible_for_milestone") is not True
            or manifest.get("bpe", {}).get("sha256") != BPE_SHA256):
        raise ValueError("fixed regression needs the original pinned F32 checkpoint reference")
    oracle = manifest.get("oracle", {})
    validate_cuda_oracle_provenance(oracle)
    if (oracle.get("variant") != "official-unfused-fp32" or oracle.get("device") != "cuda"
            or oracle.get("precision") != "float32" or not oracle.get("fp32_adaptation")
            or any(oracle.get(key) is not False for key in ("compile", "autocast", "tf32"))):
        raise ValueError("fixed reference must declare the CUDA unfused F32 oracle")
    frozen = fixed_cases()
    cases_file = artifact_path(directory, manifest["cases_manifest"]["file"])
    if (sha256_file(cases_file) != CASES_SHA256 or manifest["cases_manifest"]["sha256"] != CASES_SHA256
            or [row.get("id") for row in manifest["cases"]] != [row["id"] for row in frozen]):
        raise ValueError("fixed reference has a different or incomplete case inventory")
    artifacts = {str((directory / "manifest.json").resolve()): sha256_file(directory / "manifest.json"),
                 str(cases_file): CASES_SHA256, str(CASES_PATH): CASES_SHA256}
    for case, expected in zip(manifest["cases"], frozen):
        if any(case.get(key) != value for key, value in expected.items()):
            raise ValueError("fixed input, transform or prompt differs from the original case")
        root = artifact_path(directory, case["directory"])
        for key, name in (("input_sha256", case["input"]), ("tensors_sha256", "tensors.json"),
                          ("results_sha256", "results.json")):
            path = artifact_path(root, name)
            if sha256_file(path) != case[key]:
                raise ValueError("fixed reference input/tensor/result identity changed")
            artifacts[str(path)] = case[key]
    return manifest, artifacts


def selected_parity(payload, results, directory):
    """Reject any lossy reconstruction of the previously frozen deployed output."""
    from pycocotools import mask as masks
    scores, boxes, encoded = validate_output(payload)
    if (results.get("schema_version") != 1 or results.get("score_threshold") != 0.5
            or any(results.get(key) != payload[key] for key in ("width", "height", "prompt"))
            or [row["query_index"] for row in results["detections"]] != np.flatnonzero(scores > 0.5).tolist()):
        raise ValueError("reconstructed deployed query inventory differs from the original")
    artifacts = {}
    for row in results["detections"]:
        query = row["query_index"]
        expected = read_array(directory, row["mask"], "uint8")
        if (scores[query] != row["score"] or not np.array_equal(boxes[query], row["box"])
                or not np.array_equal(masks.decode(encoded[query]), expected)):
            raise ValueError("reconstructed deployed mask, score or box is not bit-exact")
        artifacts[str(artifact_path(directory, row["mask"]["file"]))] = row["mask"]["sha256"]
    return artifacts


def prepare_reference(args):
    import torch
    from tools.validation.export_reference import validate_source
    from tools.validation.export_precision_outputs import capture_ranked
    manifest, artifacts = original_bundle(args.reference)
    source, runtime_source, adaptations = validate_source(args.sam3_source, args.sam3_runtime_source)
    if (manifest["oracle"]["packages"].get("torch") != packages()["torch"]
            or manifest["oracle"]["cuda_runtime"] != torch.version.cuda):
        raise ValueError("postprocessing must retain the original Torch/CUDA version")
    sys.path.insert(0, str(runtime_source))
    artifacts.update(source_snapshot())
    for relative in ("sam3/model/box_ops.py", "sam3/model/data_misc.py"):
        path = runtime_source / relative
        artifacts[str(path)] = sha256_file(path)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    rows = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for case in manifest["cases"]:
            directory = artifact_path(args.reference, case["directory"])
            index = read_tensor_index(directory)
            for tensor in index["tensors"].values():
                artifacts[str(artifact_path(directory, tensor["file"]))] = tensor["sha256"]
            raw = {target: torch.from_numpy(np.array(read_array(directory, index["tensors"][name], "float32"))).cuda()
                   for target, name in (("pred_logits", "class_logits"), ("presence_logit_dec", "presence_logits"),
                                        ("pred_boxes", "pred_boxes"), ("pred_masks", "mask_logits"))}
            results = read_json(directory / "results.json")

            def recorded_tokens(prompts, context_length):
                if prompts != [case["prompt"]] or context_length != 32:
                    raise ValueError("reconstruction requested different token inputs")
                return torch.tensor([index["token_ids"]])

            payload = capture_ranked({"original_width": results["width"], "original_height": results["height"]},
                                     raw, case["prompt"], recorded_tokens, torch)
            artifacts.update(selected_parity(payload, results, directory))
            path = args.output / (case["id"] + ".json")
            write_json(path, payload)
            artifacts[str(path.resolve())] = sha256_file(path)
            rows.append({"id": case["id"], "prompt": case["prompt"], "file": path.name,
                         "sha256": sha256_file(path), "input": str(artifact_path(directory, case["input"])),
                         "input_sha256": case["input_sha256"], "selected_masks_verified": len(results["detections"])})
            print("reconstructed", case["id"], flush=True)
            del raw
    validate_source(source, runtime_source)
    verify_run_artifacts(artifacts)
    archived = archive_sources(args.output, artifacts)
    write_json(args.output / "manifest.json", {"schema_version": 2, "kind": "sam3-fixed-ranked-reference-v2",
                "complete": True, "cases_sha256": CASES_SHA256, "gates_sha256": GATES_SHA256,
                "checkpoint_sha256": manifest["checkpoint"]["sha256"], "bpe_sha256": BPE_SHA256,
                "source_reference_sha256": sha256_file(args.reference / "manifest.json"),
                "source_adaptations": adaptations, "environment": runtime_environment("cuda"), "packages": packages(),
                "scope": "Original seven cases; raw logits transcoded with exact deployed-output parity; no COCO AP",
                "cases": rows, "artifact_sha256": artifacts, "archived_sources": archived})


def load_ranked(directory, kind):
    manifest = read_json(directory / "manifest.json")
    version = manifest.get("schema_version")
    if (manifest.get("kind") != kind or not kind.endswith(f"-v{version}") or manifest.get("complete") is not True
            or manifest.get("cases_sha256") != CASES_SHA256 or manifest.get("gates_sha256") != gate_identity(version)[1]
            or [row.get("id") for row in manifest["cases"]] != [row["id"] for row in fixed_cases()]):
        raise ValueError("incomplete or mismatched fixed ranked run")
    if "recipe" in manifest:
        quality_profile(manifest["recipe"], load_gates(policy_version=version))
        if canonical_hash(manifest["recipe"]) != manifest["recipe_sha256"]:
            raise ValueError("fixed regression recipe identity changed")
    verify_export_artifacts(directory, manifest)
    outputs = {}
    for row, expected in zip(manifest["cases"], fixed_cases()):
        path = artifact_path(directory, row["file"])
        if (sha256_file(path) != row["sha256"] or row["prompt"] != expected["prompt"]
                or sha256_file(Path(row["input"])) != row["input_sha256"]):
            raise ValueError("fixed ranked input, prompt or output changed")
        outputs[row["id"]] = read_json(path)
        validate_output(outputs[row["id"]])
    return manifest, outputs


def compare_cases(reference, candidate, recipe, incremental=False):
    gates = load_gates(policy_version=recipe["schema_version"])
    _, profile = quality_profile(recipe, gates, incremental)
    rows = []
    for case in fixed_cases():
        comparison = compare_objects(reference[case["id"]], candidate[case["id"]], profile, gates["common"])
        result = object_gates([comparison], profile, gates["common"], strict=True)
        # These fixed negative prompts previously contain no deployed detections.
        # An added gray-zone detection must not evade the high-confidence checks.
        if comparison["reference_detections"] == 0:
            passed = comparison["candidate_detections"] == 0
            result["checks"].append({"name": "empty_reference_stays_empty", "value": comparison["candidate_detections"],
                                     "limit": 0, "status": "PASS" if passed else "FAIL"})
            result["status"] = combine_statuses([row["status"] for row in result["checks"]])
        rows.append({"id": case["id"], "status": result["status"], "objects": comparison, "gates": result})
    diagnostics = ({"diagnostics": diagnostic_summary([row["objects"] for row in rows])}
                   if recipe["schema_version"] == 3 else {})
    return {"status": combine_statuses([row["status"] for row in rows]), "cases": rows, **diagnostics,
            "annotation_protection": "NOT_APPLICABLE: fixed fixtures do not have COCO ground truth"}


def run(args):
    reference, reference_outputs = load_ranked(args.reference, "sam3-fixed-ranked-reference-v2")
    version = getattr(args, "policy_version", 2)
    recipe = native_recipe(args.binary, args.model, args.backend, args.compute, args.cache,
                           version, getattr(args, "quality_tier", None))
    if reference["checkpoint_sha256"] != recipe["checkpoint_sha256"]:
        raise ValueError("candidate and fixed original reference checkpoints differ")
    baseline, baseline_outputs = None, None
    if (args.baseline is None) != (args.cache == "f32"):
        raise ValueError("compressed cache needs exactly one uncompressed fixed-regression baseline")
    if args.baseline:
        baseline, baseline_outputs = load_ranked(args.baseline, f"sam3-fixed-ranked-native-v{version}")
        if (baseline["recipe"].get("feature_cache") != "f32"
                or {**baseline["recipe"], "feature_cache": args.cache} != recipe
                or baseline["reference_sha256"] != sha256_file(args.reference / "manifest.json")):
            raise ValueError("fixed cache baseline must retain the same model, compute, device and reference")
    artifacts = {**source_snapshot(), **native_snapshot(args.binary, args.model),
                 str((args.reference / "manifest.json").resolve()): sha256_file(args.reference / "manifest.json")}
    if args.baseline:
        artifacts[str((args.baseline / "manifest.json").resolve())] = sha256_file(args.baseline / "manifest.json")
    for row in reference["cases"]:
        artifacts[row["input"]] = row["input_sha256"]
        artifacts[str(artifact_path(args.reference, row["file"]))] = row["sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    table = args.output / "cases.tsv"
    if any(any(char in row[key] for char in "\t\r\n") for row in reference["cases"] for key in ("id", "input", "prompt")):
        raise ValueError("fixed TSV cannot represent the case input")
    table.write_text("".join(f"{row['id']}\t{row['input']}\t{row['prompt']}\n" for row in reference["cases"]))
    command = ["rtk", "proxy", str(args.binary.resolve()), str(args.model.resolve()), str(table.resolve()),
               args.backend, args.cache, args.compute, str((args.output / "native").resolve())]
    with (args.output / "native.log").open("x") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    native = read_json(args.output / "native/run.json")
    if (native.get("complete") is not True or native.get("cases") != len(reference["cases"])
            or native.get("backend") != args.backend or native.get("feature_cache") != args.cache
            or native.get("cuda_compute") != args.compute):
        raise ValueError("native fixed regression did not complete its declared recipe")
    outputs, rows = {}, []
    for row in reference["cases"]:
        path = args.output / "native" / (row["id"] + ".json")
        value = read_json(path)
        validate_output(value)
        if (value.get("precision") != recipe["weight_precision"]
                or (value.get("storage_profile") or "dense") != recipe["storage_profile"]
                or value.get("feature_cache") != args.cache or value.get("cuda_compute") != args.compute):
            raise ValueError("native fixed output precision differs from the declared recipe")
        outputs[row["id"]] = value
        rows.append({**row, "file": path.relative_to(args.output).as_posix(), "sha256": sha256_file(path)})
    absolute = compare_cases(reference_outputs, outputs, recipe)
    incremental = compare_cases(baseline_outputs, outputs, recipe, True) if baseline_outputs else {"status": "NOT_APPLICABLE"}
    verify_run_artifacts(artifacts)
    verify_export_artifacts(args.reference, reference)
    if baseline:
        verify_export_artifacts(args.baseline, baseline)
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            artifacts[str(path.resolve())] = sha256_file(path)
    archived = archive_sources(args.output, artifacts)
    status = combine_statuses([absolute["status"], incremental["status"]])
    separation = ({"qualification_status": "FAIL" if status == "FAIL" else "NOT_RUN", "diagnostics_affect_quality": False}
                  if version == 3 else {})
    scope = ("Seven spatial v3 task regressions with zero task-tail allowance; fidelity diagnostics do not veto quality"
             if version == 3 else "Seven spatial v2 regressions with zero tail allowance; existing legacy checks remain separately required")
    write_json(args.output / "manifest.json", {"schema_version": version, "kind": f"sam3-fixed-ranked-native-v{version}", "complete": True,
                "cases_sha256": CASES_SHA256, "gates_sha256": gate_identity(version)[1], "recipe": recipe,
                "recipe_sha256": canonical_hash(recipe), "reference_sha256": sha256_file(args.reference / "manifest.json"),
                "baseline_sha256": sha256_file(args.baseline / "manifest.json") if args.baseline else None,
                "regression_status": status, "absolute": absolute, "incremental": incremental,
                "legacy_regression_status": "NOT_RUN", "deployment_status": "NOT_RUN",
                "scope": scope, **separation,
                "cases": rows, "artifact_sha256": artifacts, "archived_sources": archived})
    print({"regression_status": status, "absolute": absolute["status"], "incremental": incremental["status"]}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reference = sub.add_parser("reference", help="transcode verified original raw tensor dumps")
    reference.add_argument("--sam3-source", type=Path, required=True)
    reference.add_argument("--sam3-runtime-source", type=Path, required=True)
    native = sub.add_parser("run", help="run native fixed cases and apply zero-tail spatial gates")
    policy_arguments(native)
    native.add_argument("--binary", type=Path, required=True)
    native.add_argument("--model", type=Path, required=True)
    native.add_argument("--backend", choices=("cpu", "cuda", "metal"), required=True)
    native.add_argument("--compute", choices=("f32", "f16"), default="f32")
    native.add_argument("--cache", choices=("f32", "f16", "mixed-q8_0"), default="f32")
    native.add_argument("--baseline", type=Path)
    for child in (reference, native):
        child.add_argument("--reference", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("use the isolated reference environment")
        if args.output.exists():
            raise FileExistsError("fixed regression output must be a new directory")
        (prepare_reference if args.command == "reference" else run)(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"fixed precision regression failed: {error}\n")


if __name__ == "__main__":
    main()
