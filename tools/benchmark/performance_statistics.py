"""Quality-independent paired latency and process-memory statistics."""

import math
import numpy as np


WORKLOADS = ("full_image", "changed_prompt", "repeated_result")


def pair_order(pair):
    return ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize_measurements(records, cases, backend, protocol):
    if backend not in ("cpu", "cuda") or type(protocol["pairs"]) is not int or not 1 <= protocol["pairs"] <= 100:
        raise ValueError("invalid measurement backend or pair count")
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("duplicate performance cases")
    for kind in ("latency", "memory"):
        if (type(protocol[kind]["warmups"]) is not int or protocol[kind]["warmups"] < 0
                or type(protocol[kind]["iterations"]) is not int or protocol[kind]["iterations"] < 1):
            raise ValueError("invalid warmup/iteration protocol")
    order = [(row["id"], pair, variant, kind) for kind in ("latency", "memory") for row in cases
             for pair in range(protocol["pairs"]) for variant in pair_order(pair)]
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
        if record["warmups"] < protocol[record["kind"]]["warmups"] or record["iterations"] < protocol[record["kind"]]["iterations"]:
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
    if lookup.keys() != expected or not cases:
        raise ValueError("missing independent performance cases/processes")
    if list(lookup) != order:
        raise ValueError("performance processes did not follow the declared alternating AB/BA order")
    # The same PID cannot represent two records from the same paired sequence.
    # Across distant processes PID reuse by the OS is legal; run IDs bind identity.
    for case in cases:
        for pair in range(protocol["pairs"]):
            pids = {lookup[(case["id"], pair, variant, kind)]["process_id"]
                    for variant in ("baseline", "candidate") for kind in ("latency", "memory")}
            if len(pids) != 4:
                raise ValueError("latency and memory measurements require independent processes")
    contaminated = any(record.get("other_compute_pids") or record.get("observer_errors") or record.get("diagnostic_only") for record in records)
    per_case, pair_ratios = [], []
    for pair in range(protocol["pairs"]):
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
        memory[metric + "_bytes"] = {variant: max(row[metric + "_bytes"][variant] for row in per_case)
                                    for variant in ("baseline", "candidate")}
        memory[metric + "_ratio"] = memory[metric + "_bytes"]["candidate"] / memory[metric + "_bytes"]["baseline"]
    workloads = {}
    for workload in WORKLOADS:
        aggregate_cases = []
        for case in cases:
            a, b = [np.concatenate([lookup[(case["id"], pair, variant, "latency")]["timing_ms"][workload]
                                    for pair in range(protocol["pairs"])]) for variant in ("baseline", "candidate")]
            aggregate_cases.append({"case": case["id"],
                                    "baseline_ms": {"p50": float(np.quantile(a, .5)), "p95": float(np.quantile(a, .95))},
                                    "candidate_ms": {"p50": float(np.quantile(b, .5)), "p95": float(np.quantile(b, .95))},
                                    "p50_ratio": float(np.quantile(b, .5) / np.quantile(a, .5)),
                                    "p95_ratio": float(np.quantile(b, .95) / np.quantile(a, .95))})
        workloads[workload] = {"ratios": {name + "_ratio": geometric_mean([row[name + "_ratio"] for row in aggregate_cases])
                                          for name in ("p50", "p95")}, "cases": aggregate_cases}
    return {"complete": True, "contaminated": contaminated, "workloads": workloads,
            "process_pairs": pair_ratios, "per_case_process_pairs": per_case, "memory": memory,
            "measurement_status": "INCONCLUSIVE" if contaminated else "VALID",
            "memory_scope": "maximum process peak across cases/pairs; ratios of maxima, not averaged per-case ratios",
            "uncertainty": "descriptive paired observations; no significance or population-level guarantee"}
