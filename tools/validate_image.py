#!/usr/bin/env python3
"""Run test_image and enforce the frozen M1 tensor/detection acceptance gates."""

import argparse
import math
from pathlib import Path
import subprocess
import tempfile

from sam3_artifacts import (BPE_SHA256, REQUIRED_TENSORS, SAM3_REVISION,
                           artifact_path, load_case_manifest, read_array, read_json, read_tensor_index,
                           sha256_file, validate_case_manifest, write_json)
from sam3_gguf import inspect_tensors, read_gguf, validate_metadata, tensor_schema


GATES = {
    "normalized_l2_f32": 1e-3, "normalized_l2_f16": 2e-2,
    "zero_norm_limit": 1e-12, "zero_norm_max_abs": 1e-5,
    "preprocessed_max_abs": 2 / 255, "high_score_min": 0.6,
    "mask_iou_f32": 0.98, "mask_iou_f16": 0.95, "score_max_abs": 0.02,
    "box_dimension_fraction": 0.01, "low_score_max": 0.4, "score_threshold": 0.5,
}


def tensor_error(actual, reference):
    import numpy as np

    if actual.shape != reference.shape or actual.size == 0:
        raise ValueError("tensor comparison requires equal, nonempty shapes")
    actual, reference = actual.reshape(-1), reference.reshape(-1)
    error_squared = reference_squared = maximum_absolute = 0.0
    for offset in range(0, actual.size, 1024 * 1024):
        a = np.asarray(actual[offset:offset + 1024 * 1024], dtype=np.float64)
        r = np.asarray(reference[offset:offset + 1024 * 1024], dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(r).all():
            raise ValueError("tensor comparison received non-finite values")
        delta = a - r
        error_squared += float(np.dot(delta, delta))
        reference_squared += float(np.dot(r, r))
        maximum_absolute = max(maximum_absolute, float(np.abs(delta).max()))
    reference_norm = math.sqrt(reference_squared)
    normalized_l2 = math.sqrt(error_squared) / reference_norm if reference_norm > GATES["zero_norm_limit"] else None
    return {"normalized_l2": normalized_l2, "reference_norm": reference_norm,
            "maximum_absolute_error": maximum_absolute}


def mask_iou(actual, reference):
    import numpy as np

    if actual.shape != reference.shape:
        raise ValueError("mask dimensions differ")
    intersection = int(np.count_nonzero((actual != 0) & (reference != 0)))
    union = int(np.count_nonzero((actual != 0) | (reference != 0)))
    return intersection / union if union else 1.0


def query_scores(tensors):
    import numpy as np

    def sigmoid(value):
        value = np.clip(np.asarray(value, dtype=np.float64), -700, 700)
        return 1 / (1 + np.exp(-value))

    return (sigmoid(tensors["class_logits"]).reshape(200)
            * float(sigmoid(tensors["presence_logits"]).reshape(-1)[0]))


def read_results(directory, scores, case, require_hash):
    result = read_json(Path(directory) / "results.json")
    if result.get("schema_version") != 1 or result.get("prompt") != case["prompt"] or result.get("score_threshold") != 0.5:
        raise ValueError("result schema, prompt, or score threshold differs from the case")
    width, height = result.get("width"), result.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("result has invalid image dimensions")
    detections = result.get("detections")
    if not isinstance(detections, list):
        raise ValueError("result lacks detections (an empty list is a valid empty result)")
    by_query = {}
    for detection in detections:
        query = detection.get("query_index")
        score = detection.get("score")
        box = detection.get("box")
        if type(query) is not int or not 0 <= query < 200 or query in by_query:
            raise ValueError(f"invalid or duplicate query index: {query}")
        if type(score) not in (float, int) or not math.isfinite(score) or not 0.5 < score <= 1:
            raise ValueError(f"query {query}: invalid detected score")
        if abs(score - scores[query]) > 1e-5:
            raise ValueError(f"query {query}: result score disagrees with raw logits")
        if (not isinstance(box, list) or len(box) != 4
                or any(type(value) not in (float, int) or not math.isfinite(value) for value in box)
                or box[0] > box[2] or box[1] > box[3]):
            raise ValueError(f"query {query}: invalid XYXY box")
        metadata = detection.get("mask")
        if not isinstance(metadata, dict) or metadata.get("shape") != [height, width]:
            raise ValueError(f"query {query}: mask dimensions do not match the image")
        read_array(directory, metadata, "uint8", require_hash)
        by_query[query] = detection
    expected = {query for query, score in enumerate(scores) if score > 0.5}
    if set(by_query) != expected:
        raise ValueError(f"selected queries disagree with raw logits: expected={sorted(expected)}, got={sorted(by_query)}")
    return result, by_query


def compare_case(reference_directory, actual_directory, case, precision, backend=None):
    import numpy as np

    gate_precision = "f16" if precision == "hybrid" else precision
    reference_index = read_tensor_index(reference_directory)
    actual_index = read_tensor_index(actual_directory, require_hash=False)
    if reference_index["token_ids"] != actual_index["token_ids"]:
        raise ValueError("token IDs differ from the official tokenizer")
    reference_tensors, actual_tensors = {}, {}
    errors, failures = {}, []
    tolerance = GATES[f"normalized_l2_{gate_precision}"]
    for name in REQUIRED_TENSORS:
        reference = read_array(reference_directory, reference_index["tensors"][name], "float32")
        actual = read_array(actual_directory, actual_index["tensors"][name], "float32", require_hash=False)
        reference_tensors[name], actual_tensors[name] = reference, actual
        metrics = tensor_error(actual, reference)
        if name == "preprocessed_image":
            passed = metrics["maximum_absolute_error"] <= GATES["preprocessed_max_abs"]
        elif metrics["normalized_l2"] is None:
            passed = metrics["maximum_absolute_error"] <= GATES["zero_norm_max_abs"]
        else:
            passed = metrics["normalized_l2"] <= tolerance
        errors[name] = {**metrics, "passed": passed}
        if not passed:
            failures.append(f"{name}: tensor tolerance exceeded")
    reference_scores = query_scores(reference_tensors)
    actual_scores = query_scores(actual_tensors)
    reference_result, reference_detections = read_results(reference_directory, reference_scores, case, True)
    actual_result, actual_detections = read_results(actual_directory, actual_scores, case, False)
    if backend and actual_result.get("backend") != backend:
        raise ValueError("C++ result did not execute on the requested backend")
    runtime = actual_result.get("runtime", {})
    if backend == "metal" and runtime.get("metal_nodes", 0) <= 0:
        raise ValueError("Metal selection has no evidence of actual graph execution on Metal")
    if (reference_result["width"], reference_result["height"]) != (actual_result["width"], actual_result["height"]):
        raise ValueError("source image dimensions differ")
    detection_metrics = []
    width, height = reference_result["width"], reference_result["height"]
    for query, reference_score in enumerate(reference_scores):
        if reference_score >= GATES["high_score_min"]:
            if query not in actual_detections:
                failures.append(f"query {query}: missing high-confidence reference detection")
                continue
            actual = actual_detections[query]
            reference = reference_detections[query]
            iou = mask_iou(read_array(actual_directory, actual["mask"], "uint8", False),
                           read_array(reference_directory, reference["mask"], "uint8"))
            score_error = abs(actual["score"] - reference["score"])
            box_error = np.abs(np.asarray(actual["box"]) - np.asarray(reference["box"]))
            box_fraction = float(np.max(box_error / np.asarray([width, height, width, height])))
            passed = (iou >= GATES[f"mask_iou_{gate_precision}"] and score_error <= GATES["score_max_abs"]
                      and box_fraction <= GATES["box_dimension_fraction"])
            detection_metrics.append({"query_index": query, "mask_iou": iou, "score_absolute_error": score_error,
                                      "box_dimension_fraction": box_fraction, "passed": passed})
            if not passed:
                failures.append(f"query {query}: mask/score/box gate exceeded")
        elif reference_score <= GATES["low_score_max"] and query in actual_detections:
            failures.append(f"query {query}: low-confidence reference query became a detection")
    adjacent = [{"query_index": query, "reference_score": float(reference_scores[query]),
                 "actual_score": float(actual_scores[query]), "reference_selected": query in reference_detections,
                 "actual_selected": query in actual_detections}
                for query in range(200) if 0.4 < reference_scores[query] < 0.6]
    return {"id": case["id"], "passed": not failures, "tensors": errors,
            "detections": detection_metrics, "threshold_adjacent": adjacent, "runtime": runtime,
            "tokenizer_compatibility_repaired": actual_result.get("tokenizer_compatibility_repaired", False),
            "transfers": runtime.get("transfers", "unavailable"), "failures": failures}


def check_provenance(model_path, reference_path, case_path, allow_supplementary=False):
    reference = read_json(reference_path / "manifest.json")
    reference_kind = reference.get("reference_kind")
    if reference_kind == "supplementary-converted-weights":
        if not allow_supplementary or reference.get("eligible_for_milestone") is not False:
            raise ValueError("supplementary references cannot satisfy original-checkpoint acceptance; pass --allow-supplementary for a separate diagnostic")
    elif reference_kind != "official-checkpoint" or reference.get("eligible_for_milestone") is not True:
        raise ValueError("reference does not declare a supported, explicit provenance kind")
    frozen = load_case_manifest(case_path)
    validate_case_manifest(reference)
    if reference_kind == "supplementary-converted-weights":
        supplementary = reference["supplementary_weights"]
        converted_manifest = model_path.with_suffix(model_path.suffix + ".manifest.json")
        model = read_json(converted_manifest)
        if model["checkpoint"]["sha256"] != supplementary["restored_checkpoint"]["sha256"]:
            raise ValueError("supplementary converter output uses a different restored checkpoint")
        model["validation_source"] = "gguf-converter-output-from-supplementary-checkpoint"
        model["supplementary_weights"] = supplementary
    else:
        model = read_json(model_path.with_suffix(model_path.suffix + ".manifest.json"))
        standard_cases = Path(__file__).resolve().parents[1] / "tests/data/sam3-image-cases.json"
        if sha256_file(case_path) != sha256_file(standard_cases):
            raise ValueError("original-checkpoint acceptance requires the complete frozen M1 corpus")
    for manifest in (model, reference, frozen):
        if manifest.get("schema_version") != 1 or manifest.get("sam3_revision") != SAM3_REVISION:
            raise ValueError("unsupported model/reference/case provenance")
    if frozen.get("acceptance") != GATES:
        raise ValueError("case acceptance gates differ from the frozen M1 plan")
    if (model.get("architecture") != "sam3" or model.get("container_format") != "gguf"
            or model.get("container_version") != 3 or model.get("sam_schema_version") not in (1, 2)):
        raise ValueError("converted model requires SAM schema-1/2 GGUF v3; reconvert the original checkpoint")
    precision = model.get("precision")
    if precision not in ("f32", "f16", "hybrid"):
        raise ValueError("converted model has unsupported precision")
    if model["output"].get("bytes") != model_path.stat().st_size:
        raise ValueError("converted model size does not match its manifest")
    if model["output"]["sha256"] != sha256_file(model_path):
        raise ValueError("converted model SHA-256 does not match its manifest")
    task = "video" if model["sam_schema_version"] == 2 else "image"
    if model.get("task", "image") != task:
        raise ValueError("converted model task disagrees with its SAM schema")
    reader = read_gguf(model_path)
    validate_metadata(reader, precision, model["checkpoint"]["sha256"], task)
    schema = read_json(Path(__file__).with_name("sam3_tensor_schema.json"))
    if schema.get("schema_version") != 1 or schema.get("sam3_revision") != SAM3_REVISION:
        raise ValueError("unsupported detector tensor schema")
    expected_tensors = tensor_schema(schema, task)
    recorded = {item["name"]: item for item in model["tensors"]}
    if len(recorded) != len(model["tensors"]) or set(recorded) != set(expected_tensors):
        raise ValueError("GGUF sidecar tensor inventory is missing, duplicated or unknown")
    actual = inspect_tensors(reader, precision, expected_tensors,
                             {name: item["shape"] for name, item in recorded.items()})
    for item in actual:
        if any(item[key] != recorded[item["name"]].get(key)
               for key in ("shape", "ggml_shape", "dtype", "offset", "bytes", "sha256")):
            raise ValueError(f"{item['name']}: GGUF payload disagrees with its sidecar")
    del reader
    if model["checkpoint"]["sha256"] != reference["checkpoint"]["sha256"]:
        raise ValueError("converted model and oracle use different original checkpoints")
    if model["bpe"]["sha256"] != BPE_SHA256 or reference["bpe"]["sha256"] != BPE_SHA256:
        raise ValueError("model/reference tokenizer is not the pinned official BPE")
    oracle = reference.get("oracle", {})
    expected_variant = "official-unfused-fp32" if reference_kind == "official-checkpoint" else "supplementary-converted-weights-unfused-fp32"
    if (oracle.get("variant") != expected_variant or oracle.get("precision") != "float32"
            or oracle.get("compile") is not False or oracle.get("autocast") is not False or oracle.get("tf32") is not False
            or not oracle.get("packages") or not oracle.get("fp32_adaptation")):
        raise ValueError("reference does not identify the recorded unfused FP32 official oracle")
    cases_file = artifact_path(reference_path, reference["cases_manifest"]["file"])
    digest = sha256_file(cases_file)
    if digest != reference["cases_manifest"]["sha256"] or digest != sha256_file(case_path):
        raise ValueError("reference cases differ from the frozen input/tolerance manifest")
    cases = reference.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("reference contains no cases")
    frozen_cases = {case["id"]: case for case in frozen["cases"]}
    if len(frozen_cases) != len(frozen["cases"]):
        raise ValueError("frozen case IDs are duplicated")
    if len(cases) != len(frozen_cases) or {case["id"] for case in cases} != set(frozen_cases):
        raise ValueError("reference is missing required cases or contains duplicate/unknown cases")
    for case in cases:
        if any(case.get(key) != value for key, value in frozen_cases[case["id"]].items()):
            raise ValueError(f"{case['id']}: reference input/prompt differs from the frozen case")
        directory = artifact_path(reference_path, case["directory"])
        for key, filename in (("input_sha256", case["input"]), ("tensors_sha256", "tensors.json"), ("results_sha256", "results.json")):
            if sha256_file(artifact_path(directory, filename)) != case[key]:
                raise ValueError(f"{case['id']}: reference {filename} hash mismatch")
        read_tensor_index(directory)
    return precision, reference, model


def validate(args):
    from sam3_artifacts import freeze_run_artifacts, verify_run_artifacts, freeze_output_files, verify_output_files

    precision, reference, model = check_provenance(args.model, args.reference, args.cases, args.allow_supplementary)
    executable = args.build_dir / "tests/test_image"
    if not executable.is_file():
        raise FileNotFoundError(f"C++ differential executable is missing: {executable}")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        output = args.output.resolve()
    else:
        output = Path(tempfile.mkdtemp(prefix=f"validation-{args.backend}-{precision}-", dir=args.build_dir.resolve()))
    report = {"schema_version": 1, "passed": False, "backend": args.backend, "precision": precision,
              "reference_kind": reference["reference_kind"], "eligible_for_milestone": reference["eligible_for_milestone"],
              "validation_source": model.get("validation_source", "official-checkpoint-converter-output"),
              "model_sha256": model["output"]["sha256"], "checkpoint_sha256": model["checkpoint"]["sha256"],
              "reference_manifest_sha256": sha256_file(args.reference / "manifest.json"),
              "gates": GATES, "cases": []}
    if "supplementary_weights" in model:
        report["supplementary_weights"] = model["supplementary_weights"]
    artifacts = freeze_run_artifacts(args.build_dir, executable, args.model, model["output"]["sha256"])
    report["run_artifact_sha256"] = artifacts
    for case in reference["cases"]:
        directory = artifact_path(args.reference, case["directory"])
        actual_directory = output / case["id"]
        command = [str(executable.resolve()), "--model", str(args.model.resolve()),
                   "--image", str(artifact_path(directory, case["input"])), "--text", case["prompt"],
                   "--backend", args.backend, "--threads", str(args.threads),
                   "--score-threshold", "0.5", "--output", str(actual_directory)]
        try:
            verify_run_artifacts(artifacts)
            print(f"Validating {case['id']} ({args.backend}, {precision})", flush=True)
            with (output / f"{case['id']}.log").open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
            if completed.returncode != 0:
                raise RuntimeError(f"test_image exited {completed.returncode}; see {case['id']}.log")
            verify_run_artifacts(artifacts)
            output_files = freeze_output_files(actual_directory)
            metrics = compare_case(directory, actual_directory, case, precision, args.backend)
            verify_output_files(actual_directory, output_files)
            metrics["output_sha256"] = output_files
            verify_run_artifacts(artifacts)
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
            metrics = {"id": case["id"], "passed": False, "failures": [str(error)]}
        report["cases"].append(metrics)
        write_json(output / "metrics.json", report)
    report["passed"] = len(report["cases"]) == len(reference["cases"]) and all(case["passed"] for case in report["cases"])
    write_json(output / "metrics.json", report)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {output / 'metrics.json'}")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("cpu", "metal"))
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[1] / "tests/data/sam3-image-cases.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-supplementary", action="store_true",
                        help="Run a separately labeled converted-weight diagnostic, never original-checkpoint acceptance")
    args = parser.parse_args()
    if args.threads <= 0 or args.timeout_seconds <= 0 or not math.isfinite(args.timeout_seconds):
        parser.error("thread count and timeout must be positive")
    try:
        parser.exit(validate(args))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f"validation failed: {error}\n")


if __name__ == "__main__":
    main()
