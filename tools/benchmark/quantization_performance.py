"""Application performance measurement without quality or campaign prerequisites."""

import argparse
import math
from pathlib import Path
import shutil
import sys

from tools.benchmark.performance_measurement import run_process
from tools.benchmark.performance_statistics import WORKLOADS, pair_order, summarize_measurements
from tools.benchmark.precision_reporting import expected_runtime_profile, native_recipe, precision_description, validate_configuration
from tools.convert.sam3_artifacts import artifact_path, read_json, sha256_file, verify_run_artifacts, write_json
from tools.maintenance.artifact_snapshot import archive_sources, native_snapshot, source_snapshot
from tools.validation.ranked_outputs import canonical_hash


RUN_KIND = "sam3-application-performance-run-v1"
REPORT_KIND = "sam3-application-performance-report-v1"


def benefit_tags(summary, minimum_percent):
    if not math.isfinite(minimum_percent) or not 0 < minimum_percent < 100:
        raise ValueError("benefit display threshold must be in (0,100) percent")
    limit = 1 - minimum_percent / 100
    result = {}
    for workload in WORKLOADS:
        metrics = {"latency": {"p50_ratio": limit, "p95_ratio": 1.0},
                   "host-memory": {"rss_peak_ratio": limit}}
        if "gpu_peak_ratio" in summary["memory"]:
            metrics["gpu-memory"] = {"gpu_peak_ratio": limit}
        result[workload] = {}
        for label, limits in metrics.items():
            aggregate = {**summary["workloads"][workload]["ratios"], **summary["memory"]}
            point = all(aggregate[key] <= value for key, value in limits.items())
            stable = all(all({**pair[workload], **pair}[key] <= value for key, value in limits.items())
                         for pair in summary["process_pairs"])
            result[workload][label] = ("INCONCLUSIVE" if summary["contaminated"] else
                                      "NOT_DEMONSTRATED" if not point else "UNSTABLE" if not stable else
                                      "OBSERVED_REDUCTION" if len(summary["process_pairs"]) == 1 else "CONSISTENT_REDUCTION")
    return {"minimum_reduction_percent": minimum_percent, "affects_exit_code": False, "workloads": result,
            "scope": "optional descriptive tags; latency requires p50 reduction and no aggregate p95 increase in every pair; inspect per-case tails separately"}


def select_cases(path, limit, requested):
    from tools.benchmark.quantization_benchmark import load_cases
    samples = load_cases(path)
    if limit is not None and (limit < 1 or requested):
        raise ValueError("positive --limit cannot be combined with --case")
    if requested:
        if len(requested) != len(set(requested)) or not set(requested) <= {row["id"] for row in samples}:
            raise ValueError("--case must select unique application sample IDs")
        samples = [row for row in samples if row["id"] in requested]
    if limit is not None:
        samples = samples[:limit]
    if any(len(row["prompts"]) < 2 for row in samples):
        raise ValueError("performance cases need two distinct prompts for the changed-prompt workload")
    return samples


def validate_record(row, recipe, case, protocol):
    if (row.get("recipe_sha256") != canonical_hash(recipe) or row.get("case_sha256") != canonical_hash(case)
            or row.get("protocol_sha256") != canonical_hash(protocol)
            or row.get("precision") != recipe["weight_precision"]
            or (row.get("storage_profile") or "dense") != recipe["storage_profile"]
            or row.get("quantization_modules") != recipe["quantization_modules"]
            or row.get("cuda_compute") != recipe["compute_mode"] or row.get("feature_cache") != recipe["feature_cache"]
            or row.get("arithmetic_profile") != expected_runtime_profile(recipe)
            or row.get("prompt") != case["prompt"] or row.get("alternate") != case["alternate"]
            or row.get("threads") != recipe["threads"]
            or row.get("warmups") != protocol[row["kind"]]["warmups"]
            or row.get("iterations") != protocol[row["kind"]]["iterations"]):
        raise ValueError("performance record differs from recipe, input or protocol")
    stats = row.get("runtime", {})
    if recipe["backend"] == "cuda" and (row.get("cuda_device") != recipe["cuda_device"] or stats.get("cuda_nodes", 0) <= 0
            or any(stats.get(key, 0) for key in ("cpu_nodes", "metal_nodes", "blas_nodes"))):
        raise ValueError("performance record did not retain strict CUDA placement")


def load_run(path):
    run = read_json(path)
    if run.get("schema_version") != 1 or run.get("kind") != RUN_KIND or run.get("complete") is not True:
        raise ValueError("incomplete or unsupported application performance run")
    if set(run["recipes"]) != {"baseline", "candidate"} or run["protocol"].get("graph_observer") is not False:
        raise ValueError("performance run requires two recipes and no graph observer")
    for recipe in run["recipes"].values():
        validate_configuration(recipe["backend"], recipe["compute_mode"], recipe["feature_cache"], recipe["activation"])
        if recipe["backend"] != run["backend"] or recipe["engine"] != "native" or recipe["threads"] != 4:
            raise ValueError("performance recipe backend/engine/threads differ")
    identities = {str(path.resolve()): sha256_file(path)}
    cases = {case["id"]: case for case in run["cases"]}
    if not cases or len(cases) != len(run["cases"]):
        raise ValueError("empty or duplicate application performance cases")
    for case in cases.values():
        image = artifact_path(path.parent, case["image"])
        if sha256_file(image) != case["image_sha256"]:
            raise ValueError("performance image changed")
        identities[str(image.resolve())] = case["image_sha256"]
    rows, seen_files = [], set()
    for item in run["records"]:
        record_path = artifact_path(path.parent, item["file"])
        if record_path in seen_files or sha256_file(record_path) != item["sha256"]:
            raise ValueError("performance record changed or repeated")
        seen_files.add(record_path)
        identities[str(record_path.resolve())] = item["sha256"]
        row = read_json(record_path)
        validate_record(row, run["recipes"][row["variant"]], cases[row["case"]], run["protocol"])
        rows.append(row)
    summary = summarize_measurements(rows, run["cases"], run["backend"], run["protocol"])
    return run, rows, summary, identities


def write_report(run_path, output, minimum_percent):
    run, rows, summary, identities = load_run(run_path)
    tags = benefit_tags(summary, minimum_percent)
    descriptions = {}
    for variant in ("baseline", "candidate"):
        profiles = {row["arithmetic_profile"] for row in rows if row["variant"] == variant}
        if len(profiles) != 1:
            raise ValueError("runtime policy identifier changed within a variant")
        descriptions[variant] = precision_description({"recipe": run["recipes"][variant], "arithmetic_profile": profiles.pop()})
    report = {"schema_version": 1, "kind": REPORT_KIND, "complete": True, "report_validity": "VALID",
              "quality": {"status": "NOT_MEASURED", "required_for_performance": False},
              "deployment_decision": "USER_OWNED", "backend": run["backend"], "cases": len(run["cases"]),
              "protocol": run["protocol"], "summary": summary, "benefit_tags": tags, "precision": descriptions,
              "recipes": run["recipes"],
              "comparison_identity": "identical" if run["recipes"]["baseline"] == run["recipes"]["candidate"] else "different",
              "artifact_sha256": identities,
              "producer_provenance_scope": "recorded by runner; offline summaries recheck inputs and raw records, not current model/binary bytes",
              "hardware_claim_scope": "only the recorded host, device and software; no cross-platform speedup claim"}
    verify_run_artifacts(identities)
    write_json(output / "report.json", report)
    lines = ["# Application performance benchmark", "", f"Measurement: {summary['measurement_status']}. Quality: NOT_MEASURED (not required).", "",
             "Ratios are candidate / baseline; values below 1 are smaller. Valid slowdowns remain completed reports.", "",
             "| Workload | p50 ratio | p95 ratio | Latency tag |", "| --- | ---: | ---: | --- |"]
    for name in WORKLOADS:
        ratios = summary["workloads"][name]["ratios"]
        lines.append(f"| {name} | {ratios['p50_ratio']:.6f} | {ratios['p95_ratio']:.6f} | {tags['workloads'][name]['latency']} |")
    lines.extend(["", "Memory ratios (maximum process peaks): " + str(summary["memory"]), "",
                  "Per-case tails, independent pairs, precision evidence and provenance are in `report.json`.",
                  "Repeated-result timing is a result-cache hit, not model inference. Diagnostic graph observers were disabled.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Performance report complete: {summary['measurement_status']}; {output}", flush=True)
    return report


def run(args):
    from PIL import Image
    if sys.prefix == sys.base_prefix or sys.platform != "linux":
        raise RuntimeError("use the isolated reference environment on Linux for process memory measurement")
    if args.output.exists():
        raise FileExistsError("performance run requires a new output directory")
    protocol = {"pairs": args.pairs, "latency": {"warmups": args.warmups, "iterations": args.iterations},
                "memory": {"warmups": 0, "iterations": args.memory_iterations}, "graph_observer": False}
    if not 1 <= args.pairs <= 100 or not 0 <= args.warmups <= 10000 or not 1 <= min(args.iterations, args.memory_iterations) <= max(args.iterations, args.memory_iterations) <= 10000:
        raise ValueError("invalid process pair/warmup/iteration counts")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("process timeout must be finite and positive")
    if not math.isfinite(args.benefit_min_percent) or not 0 < args.benefit_min_percent < 100:
        raise ValueError("benefit display threshold must be in (0,100) percent")
    samples = select_cases(args.cases, args.limit, args.case)
    recipes, models = {}, {}
    # Validate both configurations before probing the host or launching anything.
    for variant in ("baseline", "candidate"):
        validate_configuration(args.backend, getattr(args, variant + "_compute"), getattr(args, variant + "_cache"), args.activation)
    for variant in ("baseline", "candidate"):
        models[variant] = getattr(args, variant + "_model")
        recipes[variant] = native_recipe(argparse.Namespace(model=models[variant], binary=args.binary, backend=args.backend,
                                                           compute=getattr(args, variant + "_compute"), cache=getattr(args, variant + "_cache"),
                                                           activation=args.activation))
    if recipes["baseline"]["environment"] != recipes["candidate"]["environment"]:
        raise ValueError("hardware or software environment changed between recipes")
    identities = source_snapshot()
    identities[str(args.cases.resolve())] = sha256_file(args.cases)
    for model in set(models.values()):
        identities.update(native_snapshot(args.binary, model))
    args.output.mkdir(parents=True)
    (args.output / "inputs").mkdir()
    shutil.copyfile(args.cases, args.output / "cases.json")
    cases = []
    for sample in samples:
        source = artifact_path(args.input_root, sample["image"])
        identities[str(source.resolve())] = sha256_file(source)
        image_path = args.output / "inputs" / (sample["id"] + ".png")
        with Image.open(source) as image:
            image.convert("RGB").save(image_path)
        cases.append({"id": sample["id"], "image": image_path.relative_to(args.output).as_posix(),
                      "image_sha256": sha256_file(image_path), "prompt": sample["prompts"][0], "alternate": sample["prompts"][1]})
        identities[str(image_path.resolve())] = cases[-1]["image_sha256"]
    records = []
    for kind in ("latency", "memory"):
        for case in cases:
            for pair in range(args.pairs):
                for variant in pair_order(pair):
                    name = f"{kind}-{case['id']}-{pair}-{variant}"
                    print(name, flush=True)
                    process_case = {**case, "image": str((args.output / case["image"]).resolve())}
                    row = run_process(args.binary.resolve(), models[variant].resolve(), recipes[variant], process_case, kind,
                                      args.output / name, protocol=protocol, timeout=args.timeout)
                    row.update(case=case["id"], pair=pair, variant=variant, recipe_sha256=canonical_hash(recipes[variant]),
                               case_sha256=canonical_hash(case), protocol_sha256=canonical_hash(protocol))
                    validate_record(row, recipes[variant], case, protocol)
                    path = args.output / (name + "-record.json")
                    write_json(path, row)
                    records.append({"file": path.name, "sha256": sha256_file(path)})
    verify_run_artifacts(identities)
    archived = archive_sources(args.output, identities)
    pending = args.output / "run.pending.json"
    write_json(pending, {"schema_version": 1, "kind": RUN_KIND, "complete": True, "backend": args.backend,
                         "protocol": protocol, "recipes": recipes, "cases": cases, "records": records,
                         "artifact_sha256": identities, "archived_sources": archived,
                         "observer_scope": "latency has no memory/graph observer; GPU competitors checked before/after latency and during memory; transient latency competitors may be missed"})
    load_run(pending)
    pending.replace(args.output / "run.json")
    return write_report(args.output / "run.json", args.output, args.benefit_min_percent)


def summarize(args):
    if args.output.exists():
        raise FileExistsError("performance summary requires a new output directory")
    # Check first so invalid or partial receipts cannot publish a report.
    load_run(args.run)
    args.output.mkdir(parents=True)
    return write_report(args.run, args.output, args.benefit_min_percent)
