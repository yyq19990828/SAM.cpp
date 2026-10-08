#!/usr/bin/env python3
"""Evaluate ranked precision outputs against the original checkpoint and COCO."""

import argparse
import contextlib
import io
from pathlib import Path
import sys

from coco_acceptance import evaluate_ranked, paired_bootstrap, quality_gates
from evaluate_coco_screening import prompted_ground_truth
from precision_acceptance import (GATES_SHA256, canonical_hash, combine_statuses, compare_objects,
                                  load_gates, quality_profile, validate_output)
from precision_artifacts import campaign_check, phase_samples, source_snapshot, verify_export_artifacts
from prepare_coco_acceptance import load_precision_dataset
from sam3_artifacts import artifact_path, read_json, sha256_file, verify_run_artifacts, write_json


def load_run(directory):
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("schema_version") != 2 or manifest.get("kind") != "sam3-ranked-precision-output-v2"
            or manifest.get("complete") is not True or manifest.get("gates_sha256") != GATES_SHA256
            or manifest.get("recipe_sha256") != canonical_hash(manifest["recipe"])):
        raise ValueError("unsupported, incomplete or altered precision run")
    if sha256_file(directory / "dataset.json") != manifest["dataset_sha256"]:
        raise ValueError("exported dataset identity differs")
    dataset = load_precision_dataset(directory / "dataset.json")
    eligible = phase_samples(dataset, manifest["phase"])
    sample_ids = manifest["sample_ids"]
    if not sample_ids or len(set(sample_ids)) != len(sample_ids) or manifest["images"] != len(sample_ids):
        raise ValueError("missing or duplicate run sample IDs")
    samples = [row for row in eligible if row["id"] in set(sample_ids)]
    if [row["id"] for row in samples] != sample_ids:
        raise ValueError("run samples differ from the frozen phase")
    if manifest["phase"] == "evaluation" and (samples != eligible or manifest["diagnostic_only"] or not manifest["campaign_sha256"]):
        raise ValueError("final evaluation is partial or lacks a frozen campaign")
    if manifest["phase"] == "evaluation":
        campaign = Path(manifest["campaign_file"])
        if sha256_file(campaign) != manifest["campaign_sha256"]:
            raise ValueError("final campaign file differs from the exported identity")
    if read_json(directory / "recipe.json") != manifest["recipe"]:
        raise ValueError("archived recipe differs from the run")
    verify_export_artifacts(directory, manifest)
    expected = [(sample["id"], index, prompt) for sample in samples for index, prompt in enumerate(sample["prompts"])]
    if [(row["sample_id"], row["prompt_index"], row["prompt"]) for row in manifest["outputs"]] != expected:
        raise ValueError("ranked output prompt inventory differs from the frozen dataset")
    if set(manifest["input_images"]) != set(sample_ids):
        raise ValueError("input-image binding is incomplete")
    paths = [row["file"] for row in manifest["outputs"]]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate ranked output paths")
    return manifest, dataset, samples


def read_outputs(directory, manifest, lookup):
    outputs, identities = {}, {}
    for row in manifest["outputs"]:
        path = artifact_path(directory, row["file"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError("ranked output payload changed")
        value = read_json(path)
        validate_output(value)
        if value["prompt"] != row["prompt"]:
            raise ValueError("ranked output prompt changed")
        key = lookup[(row["sample_id"], row["prompt"])]
        if key in outputs:
            raise ValueError("duplicate prompted image/category output")
        outputs[key] = value
        identities[str(path.resolve())] = row["sha256"]
    if set(outputs) != set(lookup.values()):
        raise ValueError("missing prompted output")
    return outputs, identities


def same_inputs(reference, candidate):
    keys = ("dataset_sha256", "phase", "sample_ids", "input_manifest_sha256", "input_images", "campaign_sha256")
    if any(reference[key] != candidate[key] for key in keys):
        raise ValueError("reference/candidate input or campaign identities differ")
    if reference["recipe"]["checkpoint_sha256"] != candidate["recipe"]["checkpoint_sha256"]:
        raise ValueError("reference and candidate do not derive from the same checkpoint")


def noncrowd_masks(coco, key):
    rows = []
    for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[key[0]], catIds=[key[1]])):
        if not annotation.get("iscrowd", 0):
            rle = coco.annToRLE(annotation)
            rows.append({"size": list(rle["size"]), "counts": rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]})
    return rows


def compare(coco, reference_outputs, candidate_outputs, reference_metrics, candidate_metrics,
            reference_cache, candidate_cache, profile, common, directory, label):
    comparisons = []
    for key in sorted(reference_outputs):
        value = compare_objects(reference_outputs[key], candidate_outputs[key], profile, common, noncrowd_masks(coco, key))
        comparisons.append({"image_id": key[0], "category_id": key[1], **value})
    def progress(done, total):
        write_json(directory / (label + "-bootstrap-progress.json"), {"complete": False, "repetitions": done, "total": total})
        print(f"{label}: bootstrap {done}/{total}", flush=True)
    confidence = paired_bootstrap(reference_cache, candidate_cache, reference_metrics["pairs"], candidate_metrics["pairs"], common, progress)
    result = quality_gates(reference_metrics, candidate_metrics, comparisons, profile, common, confidence, len(reference_cache.image_ids))
    write_json(directory / (label + "-objects.json"), {"complete": True, "profile": profile, "pairs": comparisons})
    return {**result, "bootstrap": confidence, "profile": profile,
            "object_report": label + "-objects.json", "object_report_sha256": sha256_file(directory / (label + "-objects.json"))}


def evaluate(args):
    from pycocotools.coco import COCO
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference environment")
    if args.output.exists():
        raise FileExistsError("precision metrics must use a new directory")
    reference, dataset, samples = load_run(args.reference)
    candidate, _, _ = load_run(args.candidate)
    if reference.get("reference_kind") != "official-checkpoint" or reference["recipe"].get("engine") != "original":
        raise ValueError("absolute quality requires the original checkpoint oracle")
    same_inputs(reference, candidate)
    if candidate["phase"] == "evaluation":
        campaign_check(Path(candidate["campaign_file"]), args.candidate / "dataset.json", candidate["recipe"], "evaluation")
    if sha256_file(args.annotations) != dataset["provenance"]["annotations_sha256"]:
        raise ValueError("COCO annotations differ from the frozen dataset")
    gates = load_gates()
    quality_name, profile = quality_profile(candidate["recipe"], gates)
    identities = source_snapshot()
    for path in (args.annotations, args.reference / "manifest.json", args.candidate / "manifest.json"):
        identities[str(path.resolve())] = sha256_file(path)
    with contextlib.redirect_stdout(io.StringIO()):
        coco, lookup = prompted_ground_truth(COCO(str(args.annotations)), samples)
    r_outputs, hashes = read_outputs(args.reference, reference, lookup)
    identities.update(hashes)
    c_outputs, hashes = read_outputs(args.candidate, candidate, lookup)
    identities.update(hashes)
    print("Computing original and candidate ranked COCO metrics", flush=True)
    r_metrics, r_cache = evaluate_ranked(coco, r_outputs)
    c_metrics, c_cache = evaluate_ranked(coco, c_outputs)
    args.output.mkdir(parents=True)
    absolute = compare(coco, r_outputs, c_outputs, r_metrics, c_metrics, r_cache, c_cache,
                       profile, gates["common"], args.output, "absolute")
    incremental = {"status": "NOT_APPLICABLE" if candidate["recipe"]["feature_cache"] == "f32" else "NOT_RUN"}
    baseline = None
    if args.baseline is not None:
        if candidate["recipe"]["feature_cache"] == "f32":
            raise ValueError("an uncompressed candidate has no cache increment to compare")
        baseline, _, _ = load_run(args.baseline)
        same_inputs(reference, baseline)
        expected = {**candidate["recipe"], "feature_cache": "f32"}
        if baseline["recipe"] != expected:
            raise ValueError("cache baseline must differ only in feature_cache")
        identities[str((args.baseline / "manifest.json").resolve())] = sha256_file(args.baseline / "manifest.json")
        b_outputs, hashes = read_outputs(args.baseline, baseline, lookup)
        identities.update(hashes)
        b_metrics, b_cache = evaluate_ranked(coco, b_outputs)
        _, incremental_profile = quality_profile(candidate["recipe"], gates, incremental=True)
        incremental = compare(coco, b_outputs, c_outputs, b_metrics, c_metrics, b_cache, c_cache,
                              incremental_profile, gates["common"], args.output, "incremental")
        write_json(args.output / "baseline-metrics.json", b_metrics)
    verify_run_artifacts(identities)
    write_json(args.output / "reference-metrics.json", r_metrics)
    write_json(args.output / "candidate-metrics.json", c_metrics)
    status = combine_statuses([absolute["status"], incremental["status"]])
    from precision_artifacts import archive_sources
    archived = archive_sources(args.output, identities)
    write_json(args.output / "metrics.json", {"schema_version": 2, "kind": "sam3-precision-quality-v2", "complete": True,
                "phase": candidate["phase"], "final_evaluation_completed": candidate["phase"] == "evaluation",
                "full_model_qualification": False, "scope": "Prompted COCO subset ranked AP and deployed outputs; no arithmetic or performance certification",
                "images": len(samples), "prompted_pairs": len(lookup), "quality_profile": quality_name,
                "gates_sha256": GATES_SHA256, "dataset_sha256": candidate["dataset_sha256"],
                "candidate_recipe_sha256": candidate["recipe_sha256"], "campaign_sha256": candidate["campaign_sha256"],
                "arithmetic_status": "NOT_RUN", "absolute_quality_status": absolute["status"],
                "incremental_quality_status": incremental["status"], "quality_status": status,
                "performance_labels": [], "deployment_status": "FAIL" if status == "FAIL" else "NOT_RUN",
                "absolute": absolute, "incremental": incremental, "artifact_sha256": identities, "archived_sources": archived,
                "metric_definition": {"ap": "Ranked COCO segm AP .50:.05:.95, maxDets=100, no deployment score cutoff",
                                      "miou": "Macro positive image/prompt union-mask IoU at score > 0.5, excluding crowd pixels",
                                      "negative": "Unannotated-category proxy; increases cannot cancel across prompts"}})
    print(f"Completed precision quality: absolute={absolute['status']}, incremental={incremental['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "annotations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    try:
        evaluate(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"Precision evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
