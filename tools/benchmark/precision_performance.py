"""Frozen workloads and conservative, process-paired performance decisions."""

import math
from pathlib import Path

import numpy as np

from tools.validation.precision_acceptance import combine_statuses, gate_identity
from tools.convert.sam3_artifacts import verify_run_artifacts


WORKLOADS = ("full_image", "changed_prompt", "repeated_result")
ORDER = (("baseline", "candidate"), ("candidate", "baseline"), ("baseline", "candidate"))


def validate_cases(value, dataset, dataset_hash, gates):
    version = gates["schema_version"]
    if (value.get("schema_version") != version or value.get("kind") != f"sam3-precision-performance-cases-v{version}"
            or value.get("dataset_sha256") != dataset_hash):
        raise ValueError("invalid frozen performance cases")
    if version == 3 and value.get("gates_sha256") != gate_identity(version)[1]:
        raise ValueError("performance cases use different v3 gates")
    rows = value.get("cases", [])
    ids = [row["id"] for row in rows]
    if len(rows) < gates["performance"]["cases_min"] or len(set(ids)) != len(ids):
        raise ValueError("performance needs at least eight unique cases")
    samples = {row["id"]: row for row in dataset["samples"] if row["split"] in ("calibration", "development")}
    image_ids = []
    for row in rows:
        sample = samples.get(row["sample_id"])
        if sample is None or row["prompt"] not in sample["prompts"] or row["alternate"] not in sample["prompts"]:
            raise ValueError("performance inputs must come only from frozen development prompts")
        if row["prompt"] == row["alternate"]:
            raise ValueError("changed-prompt workload requires a different alternate")
        if value["artifact_sha256"].get(str(Path(row["image"]).resolve())) != row["image_sha256"]:
            raise ValueError("performance image identity is not bound")
        image_ids.append(row["sample_id"])
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("performance cases must cover different images")
    if {row["coverage"] for row in rows} != {"source-small", "source-middle", "source-large", "instances-few",
                                           "instances-many", "objects-small", "objects-large", "negative"}:
        raise ValueError("performance cases lack the predeclared size/instance/negative coverage")
    verify_run_artifacts(value["artifact_sha256"])
    return value


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def assess_performance(records, cases, gates, backend):
    """Raw records are separate processes for latency and memory, for every pair."""
    policy = gates["performance"]
    order = [(row["id"], pair, variant, kind) for kind in ("latency", "memory") for row in cases
             for pair in range(3) for variant in ORDER[pair]]
    expected = set(order)
    lookup = {}
    for record in records:
        key = (record["case"], record["pair"], record["variant"], record["kind"])
        if key not in expected or key in lookup:
            raise ValueError("duplicate or unexpected performance process")
        if record.get("complete") is not True or record.get("backend") != backend:
            raise ValueError("incomplete process or changed backend")
        if type(record.get("process_id")) is not int or record["process_id"] <= 0:
            raise ValueError("a measured process must have a PID")
        if record["warmups"] < policy["warmups_min"] or record["iterations"] < policy["iterations_min"]:
            raise ValueError("performance process has insufficient warmup or measurements")
        if record["kind"] == "latency":
            if record.get("memory_sampler_enabled"):
                raise ValueError("memory sampling cannot contaminate latency measurements")
            for workload in WORKLOADS:
                times = record["timing_ms"][workload]
                if len(times) != record["iterations"] or any(not math.isfinite(t) or t <= 0 for t in times):
                    raise ValueError("invalid or incomplete process timings")
        else:
            if not record.get("memory_sampler_enabled"):
                raise ValueError("memory process has no independent observer")
            for name in ("rss_peak_bytes", *(["gpu_peak_bytes"] if backend != "cpu" else [])):
                if type(record.get(name)) is not int or record[name] <= 0:
                    raise ValueError("missing measured process memory peak")
            if backend != "cpu" and (record.get("gpu_samples", 0) < 2 or record.get("sampled_pid") != record["process_id"]):
                raise ValueError("GPU memory samples must identify the measured PID")
        lookup[key] = record
    if lookup.keys() != expected or len(cases) < policy["cases_min"]:
        raise ValueError("missing independent performance cases/processes")
    if list(lookup) != order:
        raise ValueError("performance processes did not follow the frozen AB/BA/AB order")
    # The same PID cannot represent two records from the same paired sequence.
    # Across distant processes PID reuse by the OS is legal; run IDs bind identity.
    for case in cases:
        for pair in range(3):
            pids = {lookup[(case["id"], pair, variant, kind)]["process_id"]
                    for variant in ("baseline", "candidate") for kind in ("latency", "memory")}
            if len(pids) != 4:
                raise ValueError("latency and memory measurements require independent processes")
    contaminated = any(record.get("other_compute_pids") or record.get("observer_errors") or record.get("diagnostic_only") for record in records)
    per_case, pair_ratios = [], []
    for pair in range(3):
        pair_rows = []
        for case in cases:
            row = {"case": case["id"], "pair": pair, "workloads": {}}
            for workload in WORKLOADS:
                a, b = [np.asarray(lookup[(case["id"], pair, variant, "latency")]["timing_ms"][workload])
                        for variant in ("baseline", "candidate")]
                row["workloads"][workload] = {"p50_ratio": float(np.quantile(b, .50) / np.quantile(a, .50)),
                                               "p95_ratio": float(np.quantile(b, .95) / np.quantile(a, .95)),
                                               "baseline_ms": {"p50": float(np.quantile(a, .50)), "p95": float(np.quantile(a, .95))},
                                               "candidate_ms": {"p50": float(np.quantile(b, .50)), "p95": float(np.quantile(b, .95))}}
            for metric in ("rss_peak", *(["gpu_peak"] if backend != "cpu" else [])):
                a, b = [lookup[(case["id"], pair, variant, "memory")][metric + "_bytes"] for variant in ("baseline", "candidate")]
                row[metric + "_ratio"] = b / a
                row[metric + "_bytes"] = {"baseline": a, "candidate": b}
            pair_rows.append(row)
            per_case.append(row)
        summary = {workload: {name + "_ratio": geometric_mean([row["workloads"][workload][name + "_ratio"] for row in pair_rows])
                               for name in ("p50", "p95")} for workload in WORKLOADS}
        for metric in ("rss_peak", *(["gpu_peak"] if backend != "cpu" else [])):
            summary[metric + "_ratio"] = (max(row[metric + "_bytes"]["candidate"] for row in pair_rows)
                                            / max(row[metric + "_bytes"]["baseline"] for row in pair_rows))
        pair_ratios.append(summary)
    memory = {}
    for metric in ("rss_peak", *(["gpu_peak"] if backend != "cpu" else [])):
        memory[metric + "_ratio"] = (max(row[metric + "_bytes"]["candidate"] for row in per_case)
                                      / max(row[metric + "_bytes"]["baseline"] for row in per_case))
    workloads = {}
    for workload in WORKLOADS:
        aggregate_cases = []
        for case in cases:
            a, b = [np.concatenate([lookup[(case["id"], pair, variant, "latency")]["timing_ms"][workload]
                                    for pair in range(3)]) for variant in ("baseline", "candidate")]
            aggregate_cases.append({"case": case["id"], "p50_ratio": float(np.quantile(b, .5) / np.quantile(a, .5)),
                                    "p95_ratio": float(np.quantile(b, .95) / np.quantile(a, .95))})
        ratios = {name + "_ratio": geometric_mean([row[name + "_ratio"] for row in aggregate_cases]) for name in ("p50", "p95")}
        ratios.update(memory)
        def assess_limits(limits):
            relevant = {name.removesuffix("_max"): limit for name, limit in limits.items()
                        if backend != "cpu" or not name.startswith("gpu_")}
            point_pass = (all(ratios[key] <= limit for key, limit in relevant.items())
                          and all(row["p95_ratio"] <= policy["case_p95_ratio_max"] for row in aggregate_cases))
            stable = all(all(({**pair[workload], **{key: pair[key] for key in memory}})[key] <= limit
                             for key, limit in relevant.items()) for pair in pair_ratios)
            stable = stable and all(row["workloads"][workload]["p95_ratio"] <= policy["case_p95_ratio_max"] for row in per_case)
            return ("INCONCLUSIVE" if contaminated else "FAIL" if not point_pass
                    else "INCONCLUSIVE" if not stable else "PASS")

        labels = {}
        for label, limits in policy["labels"].items():
            labels[label] = ("NOT_APPLICABLE" if backend == "cpu" and "gpu-memory" in label
                             else assess_limits(limits))
            if gates["schema_version"] == 3 and labels[label] == "FAIL":
                labels[label] = "NOT_DEMONSTRATED"
        workloads[workload] = {"ratios": ratios, "cases": aggregate_cases, "labels": labels,
                               "performance_labels": [label for label, status in labels.items() if status == "PASS"]}
        if gates["schema_version"] == 3:
            workloads[workload].update(
                measurement_status="INCONCLUSIVE" if contaminated else "PASS",
                non_regression_status=assess_limits(policy["non_regression"]),
                benefit_status=("PASS" if workloads[workload]["performance_labels"] else
                                "INCONCLUSIVE" if "INCONCLUSIVE" in labels.values() else "NOT_DEMONSTRATED"))
    return {"complete": True, "contaminated": contaminated, "workloads": workloads, "process_pairs": pair_ratios,
            "per_case_process_pairs": per_case, "memory": memory,
            "boundary_policy": "Aggregate point failure is FAIL; crossing a limit between independent pairs is INCONCLUSIVE"}


def deployment_status(arithmetic, absolute, incremental, regression, performance_labels, policy_version=2):
    gate_identity(policy_version)
    if policy_version == 3 and "NOT_APPLICABLE" in (arithmetic, absolute, regression):
        raise ValueError("v3 arithmetic, absolute quality and regression are mandatory")
    prerequisite = combine_statuses([arithmetic, absolute, incremental, regression])
    if prerequisite != "PASS":
        return prerequisite
    return "PASS" if policy_version == 3 or performance_labels else "INCONCLUSIVE"
