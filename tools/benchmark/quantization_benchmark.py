#!/usr/bin/env python3
"""Advisory SAM 3 comparisons on application images, without quality vetoes."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tools.convert.sam3_artifacts import artifact_path, read_json, sha256_file, verify_run_artifacts, write_json
from tools.validation.ranked_outputs import canonical_hash, mask_ious, spatial_assignment, validate_output
from tools.benchmark.precision_reporting import native_recipe, precision_description, validate_configuration
from tools.quantize.quantization_config import (resolve_runtime_arguments, resolve_performance_arguments,
                                               validate_recipe_configuration)


RUN_KIND = "sam3-quantization-benchmark-run-v1"
REPORT_KIND = "sam3-quantization-reference-benchmark-v1"
ADVICE_METRICS = {
    "missing_reference_rate_max": ("missing_reference_rate", "max"),
    "added_candidate_rate_max": ("added_candidate_rate", "max"),
    "positive_union_iou_mean_min": ("positive_union_iou_mean", "min"),
    "matched_mask_iou_p05_min": ("matched_mask_iou_p05", "min"),
    "score_error_mean_max": ("score_error_mean", "max"),
}


def load_cases(path):
    value = read_json(path)
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or not isinstance(value.get("samples"), list) or not value["samples"]):
        raise ValueError("application cases require schema_version=1 and nonempty samples")
    seen = set()
    for row in value["samples"]:
        if not isinstance(row, dict):
            raise ValueError("application samples must be objects")
        name, prompts = row.get("id"), row.get("prompts")
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
                or name in seen or not isinstance(row.get("image"), str)):
            raise ValueError("invalid/duplicate sample ID or image path")
        if (not isinstance(prompts, list) or not prompts
                or any(not isinstance(p, str) or not p.strip() or any(c in p for c in "\t\r\n") for p in prompts)
                or len(set(prompts)) != len(prompts)):
            raise ValueError("each sample requires unique, nonempty single-line prompts")
        seen.add(name)
    return value["samples"]


def export(args):
    from PIL import Image
    from tools.maintenance.artifact_snapshot import archive_sources, native_snapshot, source_snapshot
    from tools.validation.export_ranked_outputs import export_native, export_original, oracle_recipe
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference environment")
    if args.output.exists():
        raise FileExistsError("benchmark export requires a new output directory")
    verify_run_artifacts(getattr(args, "configuration_inputs", {}))
    samples = load_cases(args.cases)
    validate_configuration(args.backend, args.compute, args.cache, getattr(args, "activation", "backend-selected"))
    recipe = oracle_recipe(args) if args.engine == "original" else native_recipe(args)
    identities = source_snapshot()
    identities.update(getattr(args, "configuration_inputs", {}))
    identities[str(args.cases.resolve())] = sha256_file(args.cases)
    if args.engine == "native":
        identities.update(native_snapshot(args.binary, args.model))
    else:
        identities.update({str(path.resolve()): sha256_file(path) for path in (args.checkpoint, args.bpe)})
    args.output.mkdir(parents=True)
    args.inputs = args.output / "inputs"
    args.inputs.mkdir()
    input_rows = {}
    for sample in samples:
        source = artifact_path(args.input_root, sample["image"])
        identities[str(source.resolve())] = sha256_file(source)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        path = args.inputs / (sample["id"] + ".png")
        image.save(path)
        row = {"file": path.name, "sha256": sha256_file(path), "width": image.width,
               "height": image.height, "source_sha256": identities[str(source.resolve())]}
        input_rows[sample["id"]] = row
        identities[str(path.resolve())] = row["sha256"]
    shutil.copyfile(args.cases, args.output / "cases.json")
    write_json(args.output / "recipe.json", recipe)
    outputs, metadata = (export_original if args.engine == "original" else export_native)(
        args, samples, input_rows, identities, recipe)
    verify_run_artifacts(identities)
    archived = archive_sources(args.output, identities)
    pending = args.output / "manifest.pending.json"
    write_json(pending, {"schema_version": 1, "kind": RUN_KIND, "complete": True,
                "images": len(samples), "sample_ids": [s["id"] for s in samples],
                "cases_sha256": identities[str(args.cases.resolve())],
                "input_images": {name: row["sha256"] for name, row in input_rows.items()},
                "inputs": input_rows, "outputs": outputs, "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
                "activation_policy": "backend-selected; no independent activation setting",
                "kernel_precision_evidence": "NOT_COLLECTED", "artifact_sha256": identities,
                "archived_sources": archived, **metadata})
    # Publish completion only after validating the entire provisional bundle.
    load_bundle(args.output, manifest_name=pending.name)
    pending.replace(args.output / "manifest.json")
    print(f"Exported {len(samples)} application images / {len(outputs)} prompts: {args.output}", flush=True)


def load_bundle(directory, *, manifest_name="manifest.json"):
    manifest_path = directory / manifest_name
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("benchmark manifest must be an object")
    expected_version = {RUN_KIND: 1, "sam3-ranked-precision-output-v2": 2, "sam3-ranked-precision-output-v3": 3}
    if (manifest.get("kind") not in expected_version or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != expected_version[manifest["kind"]] or manifest.get("complete") is not True
            or manifest.get("recipe_sha256") != canonical_hash(manifest["recipe"])):
        raise ValueError("incomplete, unsupported or altered benchmark bundle")
    if read_json(directory / "recipe.json") != manifest["recipe"]:
        raise ValueError("archived recipe differs from the bundle")
    validate_recipe_configuration(manifest["recipe"])
    inputs = manifest["input_images"]
    sample_ids = manifest["sample_ids"]
    if (not isinstance(inputs, dict) or not inputs or len(sample_ids) != len(set(sample_ids))
            or set(sample_ids) != set(inputs) or manifest["images"] != len(inputs)
            or any(not isinstance(d, str) or not re.fullmatch(r"[a-f0-9]{64}", d) for d in inputs.values())):
        raise ValueError("invalid or incomplete input-image identities")
    identities = {str(manifest_path.resolve()): sha256_file(manifest_path),
                  str((directory / "recipe.json").resolve()): sha256_file(directory / "recipe.json")}
    if manifest["kind"] == RUN_KIND:
        cases = directory / "cases.json"
        if sha256_file(cases) != manifest["cases_sha256"]:
            raise ValueError("application case manifest changed")
        samples = load_cases(cases)
        if [s["id"] for s in samples] != sample_ids:
            raise ValueError("application sample inventory changed")
        expected = [(s["id"], i, p) for s in samples for i, p in enumerate(s["prompts"])]
        identities[str(cases.resolve())] = manifest["cases_sha256"]
        if set(manifest["inputs"]) != set(inputs):
            raise ValueError("normalized input inventory changed")
        for name, row in manifest["inputs"].items():
            path = artifact_path(directory / "inputs", row["file"])
            if row["sha256"] != inputs[name] or sha256_file(path) != inputs[name]:
                raise ValueError("normalized image changed")
            identities[str(path.resolve())] = inputs[name]
    else:
        # Legacy receipts remain untouched. Read their frozen cases without
        # applying the old quality budgets or claiming a new final holdout.
        cases = directory / "dataset.json"
        if sha256_file(cases) != manifest["dataset_sha256"]:
            raise ValueError("legacy dataset identity changed")
        rows = {s["id"]: s for s in read_json(cases)["samples"]}
        expected = [(name, i, p) for name in sample_ids for i, p in enumerate(rows[name]["prompts"])]
        identities[str(cases.resolve())] = manifest["dataset_sha256"]
    if [(row["sample_id"], row["prompt_index"], row["prompt"]) for row in manifest["outputs"]] != expected:
        raise ValueError("output prompt inventory differs from the case manifest")
    outputs, paths = {}, set()
    for row in manifest["outputs"]:
        path = artifact_path(directory, row["file"])
        if path in paths or sha256_file(path) != row["sha256"]:
            raise ValueError("duplicate or changed output payload")
        paths.add(path)
        payload = read_json(path)
        validate_output(payload)
        if payload["prompt"] != row["prompt"]:
            raise ValueError("output prompt differs from the case manifest")
        if manifest["kind"] == RUN_KIND:
            size = manifest["inputs"][row["sample_id"]]
            if (payload["width"], payload["height"]) != (size["width"], size["height"]):
                raise ValueError("output dimensions differ from the normalized input")
        outputs[(row["sample_id"], row["prompt_index"], row["prompt"])] = payload
        identities[str(path.resolve())] = row["sha256"]
    verify_run_artifacts(identities)
    return manifest, outputs, identities


def reference_kind(recipe):
    if any(recipe.get(k) != "f32" for k in ("weight_precision", "compute_mode", "feature_cache")):
        raise ValueError("reference must declare F32 weights, compute policy and feature cache")
    if recipe.get("storage_profile") != "dense":
        raise ValueError("reference must use dense F32 storage")
    if recipe.get("engine") not in ("original", "native"):
        raise ValueError("reference must be an original checkpoint or native F32 run")
    return "official-checkpoint" if recipe["engine"] == "original" else "native-f32"


def compare_pair(reference, candidate, score_threshold, matching_iou):
    from pycocotools import mask as masks
    rs, rb, rm = validate_output(reference)
    cs, cb, cm = validate_output(candidate)
    if any(reference[k] != candidate[k] for k in ("width", "height", "prompt", "token_ids")):
        raise ValueError("input dimensions, prompt or token IDs differ")
    r_ids, c_ids = np.flatnonzero(rs > score_threshold), np.flatnonzero(cs > score_threshold)
    r_masks, c_masks = [rm[int(q)] for q in r_ids], [cm[int(q)] for q in c_ids]
    ious = mask_ious(r_masks, c_masks)
    distances = np.max(np.abs(rb[r_ids, None] - cb[c_ids][None, :]) /
                       np.asarray([reference["width"], reference["height"]] * 2), axis=2)
    assignments = spatial_assignment(ious, [True] * len(r_ids), distances, matching_iou)
    paired_r, paired_c = {r for r, _ in assignments}, {c for _, c in assignments}
    pairs = [{"reference_query": int(r_ids[r]), "candidate_query": int(c_ids[c]),
              "mask_iou": float(ious[r, c]), "score_error": float(abs(rs[r_ids[r]] - cs[c_ids[c]])),
              "box_fraction": float(distances[r, c])} for r, c in assignments]
    if not r_masks or not c_masks:
        union_iou = float(not r_masks and not c_masks)
    else:
        union_iou = float(mask_ious([masks.merge(r_masks)], [masks.merge(c_masks)])[0, 0])
    return {"reference_detections": len(r_ids), "candidate_detections": len(c_ids), "matches": pairs,
            "missing_reference_queries": [int(q) for i, q in enumerate(r_ids) if i not in paired_r],
            "added_candidate_queries": [int(q) for i, q in enumerate(c_ids) if i not in paired_c],
            "union_mask_iou": union_iou}


def summarize(rows):
    pairs = [p for row in rows for p in row["matches"]]
    positive = [r["union_mask_iou"] for r in rows if r["reference_detections"]]
    empty = [r for r in rows if not r["reference_detections"]]
    reference = sum(r["reference_detections"] for r in rows)
    candidate = sum(r["candidate_detections"] for r in rows)
    missing = sum(len(r["missing_reference_queries"]) for r in rows)
    added = sum(len(r["added_candidate_queries"]) for r in rows)
    ious = [p["mask_iou"] for p in pairs]
    return {"prompted_pairs": len(rows), "reference_objects": reference, "candidate_objects": candidate,
            "matched_objects": len(pairs), "missing_reference_objects": missing, "added_candidate_objects": added,
            "missing_reference_rate": missing / reference if reference else None,
            "added_candidate_rate": added / candidate if candidate else None,
            "positive_reference_pairs": len(positive), "positive_union_iou_mean": float(np.mean(positive)) if positive else None,
            "reference_empty_pairs": len(empty),
            "reference_empty_with_candidate_objects": sum(bool(r["candidate_detections"]) for r in empty),
            "matched_mask_iou_mean": float(np.mean(ious)) if ious else None,
            "matched_mask_iou_p05": float(np.quantile(ious, .05)) if ious else None,
            "matched_mask_iou_min": min(ious, default=None),
            "score_error_mean": float(np.mean([p["score_error"] for p in pairs])) if pairs else None,
            "box_fraction_max": max((p["box_fraction"] for p in pairs), default=None)}


def advise(summary, limits):
    if limits is None:
        return {"status": "NOT_REQUESTED", "checks": [], "affects_exit_code": False}
    if not isinstance(limits, dict) or not limits or any(k not in ADVICE_METRICS for k in limits):
        raise ValueError("advice requires a nonempty object of documented limit names")
    checks = []
    for name, limit in limits.items():
        if type(limit) not in (float, int) or not np.isfinite(limit) or not 0 <= limit <= 1:
            raise ValueError("advisory limits must be finite numbers in [0, 1]")
        key, direction = ADVICE_METRICS[name]
        value = summary[key]
        outside = value is not None and (value > limit if direction == "max" else value < limit)
        checks.append({"name": name, "value": value, "limit": limit,
                       "status": "NO_DATA" if value is None else "OUTSIDE" if outside else "WITHIN"})
    status = ("OUTSIDE_USER_LIMITS" if any(r["status"] == "OUTSIDE" for r in checks) else
              "INSUFFICIENT_DATA" if any(r["status"] == "NO_DATA" for r in checks) else "WITHIN_USER_LIMITS")
    return {"status": status, "checks": checks, "affects_exit_code": False}


def compare(args):
    if args.output.exists():
        raise FileExistsError("benchmark report requires a new output directory")
    if not np.isfinite(args.score_threshold) or not .5 <= args.score_threshold < 1:
        raise ValueError("ranked export comparisons currently require a score threshold in [0.5, 1)")
    if not np.isfinite(args.matching_iou) or not 0 < args.matching_iou <= 1:
        raise ValueError("matching IoU must be in (0, 1]")
    reference, r_outputs, identities = load_bundle(args.reference)
    candidate, c_outputs, c_hashes = load_bundle(args.candidate)
    identities.update(c_hashes)
    kind = reference_kind(reference["recipe"])
    if reference["input_images"] != candidate["input_images"] or set(r_outputs) != set(c_outputs):
        raise ValueError("reference/candidate image or prompt identities differ")
    if (not reference["recipe"].get("checkpoint_sha256")
            or reference["recipe"]["checkpoint_sha256"] != candidate["recipe"].get("checkpoint_sha256")):
        raise ValueError("reference/candidate checkpoint provenance differs")
    rows = [{"sample_id": key[0], "prompt_index": key[1], "prompt": key[2],
             **compare_pair(r_outputs[key], c_outputs[key], args.score_threshold, args.matching_iou)}
            for key in sorted(r_outputs)]
    summary = summarize(rows)
    limits = read_json(args.advice) if args.advice is not None else None
    if args.advice is not None:
        identities[str(args.advice.resolve())] = sha256_file(args.advice)
    advice = advise(summary, limits)
    verify_run_artifacts(identities)
    args.output.mkdir(parents=True)
    write_json(args.output / "cases.json", rows)
    report = {"schema_version": 1, "kind": REPORT_KIND, "complete": True, "report_validity": "VALID",
              "quality_role": "reference-agreement-only", "production_accuracy": "NOT_MEASURED",
              "deployment_decision": "USER_OWNED", "reference_kind": kind, "images": len(reference["input_images"]),
              "score_threshold": args.score_threshold, "selection_rule": "score > threshold",
              "matching_iou_min": args.matching_iou, "summary": summary, "advice": advice,
              "reference_precision": precision_description(reference), "candidate_precision": precision_description(candidate),
              "reference_recipe_sha256": reference["recipe_sha256"], "candidate_recipe_sha256": candidate["recipe_sha256"],
              "performance": {"status": "NOT_MEASURED", "benefit_labels": []},
              "uncertainty": "descriptive statistics; no bootstrap or independent-holdout claim",
              "input_identity_scope": "exported input hashes; normalized image bytes rechecked for application bundles",
              "producer_provenance_scope": "recorded by exporter; model/binary/kernel execution not recertified by comparison",
              "case_report_sha256": sha256_file(args.output / "cases.json"), "artifact_sha256": identities,
              "evaluator_sha256": sha256_file(Path(__file__)),
              "definitions": {"missing": "unmatched reference objects, not ground-truth false negatives",
                              "added": "unmatched candidate objects, not ground-truth false positives",
                              "positive_union_iou_mean": "macro IoU over reference-positive prompts; empty reference prompts reported separately",
                              "matched_statistics": "matched objects only; read alongside missing/added rates"}}
    write_json(args.output / "report.json", report)
    lines = ["# Quantization reference benchmark", "", f"Report validity: VALID. Advice: {advice['status']}.", "",
             "These measurements describe agreement with the selected reference. Production accuracy is not measured.", "",
             "| Measurement | Value |", "| --- | ---: |"]
    lines.extend(f"| {key} | {value if value is not None else 'NO_DATA'} |" for key, value in summary.items())
    lines.extend(["", "Full per-prompt differences are in `cases.json`. Performance was not measured.", ""])
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Reference benchmark complete: VALID; advice={advice['status']}; {args.output}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("export", help="export application images without a quality policy")
    for name in ("cases", "input-root", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--engine", choices=("original", "native"), required=True)
    for name in ("model", "binary", "checkpoint", "sam3-source", "sam3-runtime-source", "bpe"):
        run.add_argument("--" + name, type=Path)
    run.add_argument("--quantization-config", type=Path, help="shared four-axis configuration; exclusive with separate precision/backend flags")
    run.add_argument("--backend", choices=("cpu", "cuda"))
    run.add_argument("--compute", choices=("f32", "f16"))
    run.add_argument("--cache", choices=("f32", "f16", "mixed-q8_0"))
    run.add_argument("--activation", help="native mode: only backend-selected is implemented")
    measure = commands.add_parser("compare", help="compare bundles; optional limits are advisory")
    for name in ("reference", "candidate", "output"):
        measure.add_argument("--" + name, type=Path, required=True)
    measure.add_argument("--score-threshold", type=float, default=.5)
    measure.add_argument("--matching-iou", type=float, default=.5, help="object matching rule, not a quality floor")
    measure.add_argument("--advice", type=Path, help="optional application limits; never veto a valid report")
    performance = commands.add_parser("performance", help="paired timing/memory without quality prerequisites")
    for name in ("cases", "input-root", "binary", "baseline-model", "candidate-model", "output"):
        performance.add_argument("--" + name, type=Path, required=True)
    performance.add_argument("--backend", choices=("cpu", "cuda"))
    performance.add_argument("--activation", help="only backend-selected is implemented")
    for variant in ("baseline", "candidate"):
        performance.add_argument("--" + variant + "-config", type=Path, help="shared configuration; supply both baseline and candidate files")
        performance.add_argument("--" + variant + "-compute", choices=("f32", "f16"))
        performance.add_argument("--" + variant + "-cache", choices=("f32", "f16", "mixed-q8_0"))
    performance.add_argument("--pairs", type=int, default=3, help="alternating AB/BA process pairs")
    performance.add_argument("--warmups", type=int, default=5)
    performance.add_argument("--iterations", type=int, default=20)
    performance.add_argument("--memory-iterations", type=int, default=1, help="independent memory process iterations; zero warmups")
    performance.add_argument("--limit", type=int, help="select the first N application cases")
    performance.add_argument("--case", action="append", help="select sample ID; repeatable")
    performance.add_argument("--timeout", type=float, default=1800, help="timeout in seconds per child process")
    performance.add_argument("--benefit-min-percent", type=float, default=5, help="advisory tag threshold; never changes exit status")
    summary = commands.add_parser("summarize-performance", help="offline summary of an independent performance run")
    for name in ("run", "output"):
        summary.add_argument("--" + name, type=Path, required=True)
    summary.add_argument("--benefit-min-percent", type=float, default=5)
    inspect = commands.add_parser("inspect-precision", help="inspect stored weights and resolved policies without inference")
    for name in ("model", "output"):
        inspect.add_argument("--" + name, type=Path, required=True)
    inspect.add_argument("--quantization-config", type=Path, help="shared four-axis configuration checked against the GGUF manifest")
    inspect.add_argument("--backend", choices=("cpu", "metal", "cuda"))
    inspect.add_argument("--compute", choices=("f32", "f16"))
    inspect.add_argument("--cache", choices=("f32", "f16", "mixed-q8_0"))
    inspect.add_argument("--activation", help="only backend-selected is implemented")
    args = parser.parse_args(argv)
    try:
        if args.command in ("export", "inspect-precision"):
            resolve_runtime_arguments(args)
        elif args.command == "performance":
            resolve_performance_arguments(args)
        if args.command == "export":
            required = ("checkpoint", "sam3_source", "sam3_runtime_source", "bpe") if args.engine == "original" else ("binary", "model")
            if any(getattr(args, name) is None for name in required):
                raise ValueError("missing engine arguments: " + ", ".join(required))
            if args.engine == "original" and (args.backend != "cuda" or args.compute != "f32" or args.cache != "f32"):
                raise ValueError("the original checkpoint exporter currently requires CUDA F32 compute/cache")
            export(args)
        elif args.command == "compare":
            compare(args)
        elif args.command == "inspect-precision":
            from tools.benchmark.precision_reporting import inspect as describe
            describe(args)
        else:
            from tools.benchmark.quantization_performance import run as time_run, summarize as time_summary
            (time_run if args.command == "performance" else time_summary)(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Quantization benchmark failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
