#!/usr/bin/env python3
"""Freeze complete development recipes and their deployment baselines before holdout inference."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import re

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.archive.precision_v2_v3.evaluate_precision import load_run, same_inputs
from tools.archive.precision_v2_v3.precision_acceptance import canonical_hash, gate_identity, load_gates, quality_profile
from tools.archive.precision_v2_v3.precision_artifacts import (native_snapshot, phase_samples, source_snapshot,
                                                 validate_evaluation_history, verify_export_artifacts)
from tools.convert.sam3_artifacts import read_json, sha256_file, verify_run_artifacts


def development_eligible(metrics, cache):
    """A small development split may miss the final image-count requirement."""
    for name in ("absolute", "incremental"):
        group = metrics[name]
        if name == "incremental" and cache == "f32" and group["status"] == "NOT_APPLICABLE":
            continue
        checks = group.get("checks", [])
        if (group["status"] not in ("PASS", "INCONCLUSIVE") or not checks
                or any(c["status"] not in ("PASS", "NOT_APPLICABLE")
                       and not (c["name"] == "evaluation_images" and c["status"] == "INCONCLUSIVE")
                       for c in checks)):
            raise ValueError("v3 final candidates and baselines must pass development quality")


def freeze(selection_path, output):
    if output.exists():
        raise FileExistsError("campaign output already exists")
    selection = read_json(selection_path)
    version = selection.get("schema_version")
    gate_hash = gate_identity(version)[1]
    if selection.get("kind") != f"sam3-precision-selection-v{version}":
        raise ValueError("unsupported campaign selection")
    entries = selection.get("runs", [])
    ids = [row["id"] for row in entries]
    if (not entries or len(ids) != len(set(ids))
            or any(not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) for name in ids)):
        raise ValueError("run IDs must be unique safe names")
    gates, identities, runs, original = load_gates(policy_version=version), source_snapshot(), [], None
    identities[str(selection_path.resolve())] = sha256_file(selection_path)
    reference = None
    for entry in entries:
        directory = Path(entry["development_run"])
        value, dataset, samples = load_run(directory)
        if value["schema_version"] != version:
            raise ValueError("development runs and campaign policy versions differ")
        if version == 3:
            identities.update(validate_evaluation_history(selection.get("evaluation_history"), dataset))
        if value["phase"] != "development" or samples != phase_samples(dataset, "development"):
            raise ValueError("freeze requires complete development outputs, not a preflight subset")
        if len(phase_samples(dataset, "evaluation")) < gates["common"]["evaluation_images_min"]:
            raise ValueError("the frozen evaluation split is too small")
        if len([row for row in dataset["samples"] if row["split"] == "reserve"]) < 1024:
            raise ValueError("the independent reserve must retain at least 1024 images")
        if reference is not None:
            same_inputs(reference, value)
        reference = value
        recipe = value["recipe"]
        quality_profile(recipe, gates)
        if not recipe.get("environment"):
            raise ValueError("each recipe must bind its hardware and driver environment")
        if recipe["engine"] == "original":
            if original is not None or value.get("reference_kind") != "official-checkpoint":
                raise ValueError("campaign requires exactly one original checkpoint reference")
            original = entry["id"]
        elif (version == 2 or entry.get("performance_baseline") is not None) and entry.get("performance_baseline") not in ids:
            raise ValueError("every native recipe requires a frozen deployment baseline")
        # Completed development sources are archived with their runs. The live
        # export/evaluation closure below is independently frozen for the final.
        verify_export_artifacts(directory, value)
        for name, digest in value["artifact_sha256"].items():
            if name not in value.get("archived_sources", {}):
                if name in identities and identities[name] != digest:
                    raise ValueError("development runs bind conflicting artifacts")
                identities[name] = digest
        identities[str((directory / "manifest.json").resolve())] = sha256_file(directory / "manifest.json")
        rows = {"id": entry["id"], "recipe": recipe, "recipe_sha256": value["recipe_sha256"],
                "development_manifest": str((directory / "manifest.json").resolve()),
                "performance_baseline": entry.get("performance_baseline")}
        if version == 3:
            rows["cache_baseline"] = entry.get("cache_baseline")
        if recipe["engine"] != "original":
            for field, suffix in (("model", ".gguf"), ("binary", None)):
                matches = [name for name, digest in value["artifact_sha256"].items()
                           if digest == recipe[field + "_sha256"] and (suffix is None or name.endswith(suffix))]
                if len(matches) != 1:
                    raise ValueError("cannot identify the exact native model/export executable")
                rows["model" if field == "model" else "export_binary"] = matches[0]
            metric_path = Path(entry["development_quality"])
            metrics = read_json(metric_path)
            verify_export_artifacts(metric_path.parent, metrics)
            if any(metrics["artifact_sha256"].get(name) != digest for name, digest in source_snapshot().items()):
                raise ValueError("development quality must be recomputed after evaluator/source changes")
            if (metrics.get("schema_version") != version or metrics.get("kind") != f"sam3-precision-quality-v{version}" or metrics.get("complete") is not True
                    or metrics.get("phase") != "development" or metrics.get("images") != len(samples)
                    or metrics.get("candidate_recipe_sha256") != value["recipe_sha256"]
                    or metrics.get("dataset_sha256") != value["dataset_sha256"]
                    or metrics.get("gates_sha256") != gate_hash):
                raise ValueError("candidate selection requires matching complete development metrics")
            identities[str(metric_path.resolve())] = sha256_file(metric_path)
            rows["development_quality"] = str(metric_path.resolve())
            rows["development_quality_status"] = metrics["quality_status"]
            if version == 3:
                development_eligible(metrics, recipe["feature_cache"])
        runs.append(rows)
    if original is None:
        raise ValueError("campaign has no original checkpoint reference")
    lookup = {row["id"]: row for row in runs}
    for row in runs:
        if row["recipe"]["engine"] == "original":
            continue
        baseline = lookup.get(row["performance_baseline"])
        if baseline is not None:
            if baseline["recipe"]["engine"] != "native" or baseline["recipe"]["backend"] != row["recipe"]["backend"]:
                raise ValueError("deployment baselines must use the same native backend")
            if version == 3 and baseline["recipe"]["quality_tier"] != row["recipe"]["quality_tier"]:
                raise ValueError("v3 performance baselines require the same declared quality tier")
        if row["recipe"]["feature_cache"] != "f32":
            cache_baseline = lookup.get(row["cache_baseline"]) if version == 3 else baseline
            if cache_baseline is None or cache_baseline["recipe"] != {**row["recipe"], "feature_cache": "f32"}:
                raise ValueError("compressed cache baseline must differ only in feature_cache")
        elif version == 3 and row["cache_baseline"] is not None:
            raise ValueError("an F32 cache has no incremental cache baseline")
    recipe_hashes = [row["recipe_sha256"] for row in runs]
    if len(recipe_hashes) != len(set(recipe_hashes)):
        raise ValueError("the same recipe was selected under multiple names")
    cases_path, benchmark_binary = None, None
    if version == 2 or selection.get("performance_cases") is not None:
        cases_path = Path(selection["performance_cases"])
        cases = read_json(cases_path)
        from tools.archive.precision_v2_v3.precision_performance import validate_cases
        validate_cases(cases, dataset, reference["dataset_sha256"], gates)
        identities[str(cases_path.resolve())] = sha256_file(cases_path)
        identities.update(cases["artifact_sha256"])
        benchmark_binary = Path(selection["benchmark_binary"]).resolve()
        for row in runs:
            if row["recipe"]["engine"] == "native":
                identities.update(native_snapshot(benchmark_binary, Path(row["model"])))
    elif selection.get("benchmark_binary") is not None or any(row["performance_baseline"] is not None for row in runs):
        raise ValueError("performance comparisons require frozen cases and a benchmark binary")
    verify_run_artifacts(identities)
    value = {"schema_version": version, "kind": f"sam3-precision-campaign-v{version}", "frozen_before_evaluation": True,
             "frozen_at": datetime.now(timezone.utc).isoformat(), "gates_sha256": gate_hash,
             "dataset_sha256": reference["dataset_sha256"], "recipe_sha256": recipe_hashes,
             "original_reference": original, "runs": runs, "performance_cases": str(cases_path.resolve()) if cases_path else None,
             "benchmark_binary": str(benchmark_binary) if benchmark_binary else None,
             "artifact_sha256": identities, "reserve_inference_allowed": False,
             "selection_sha256": canonical_hash(selection)}
    if version == 3:
        value["evaluation_history"] = selection["evaluation_history"]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation means a second invocation cannot replace a frozen list.
    import json
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Frozen {len(runs)} recipes: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        freeze(args.selection, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Precision campaign freeze failed: {error}\n")


if __name__ == "__main__":
    main()
