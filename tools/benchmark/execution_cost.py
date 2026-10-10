"""Independent, hash-bound CPU execution diagnostics; never a performance verdict."""

import math
from pathlib import Path
import subprocess

from tools.benchmark.precision_reporting import (expected_runtime_profile, native_recipe,
                                                native_weight_policy_matches, precision_description)
from tools.convert.sam3_artifacts import read_json, sha256_file, verify_run_artifacts, write_json


def finite_nonnegative(value, label):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid observed cost: {label}")


def validate_cost(value):
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1 or value.get("kind") != "sam-cpu-execution-cost-v1"
            or value.get("complete") is not True or value.get("diagnostic_only") is not True
            or value.get("performance_comparable") is not False or value.get("backend") != "cpu"
            or value.get("kernel_internal_rhs_packing") != "NOT_COLLECTED"
            or value.get("kernel_internal_arithmetic") != "NOT_COLLECTED"):
        raise ValueError("cost report must declare observed CPU diagnostics and uncollected kernel internals")
    for key in ("graph_compute_wall_ms", "graph_bind_wall_ms", "graph_arena_peak_bytes"):
        finite_nonnegative(value[key], key)
    if not value["graphs"] or not value["nodes"]:
        raise ValueError("cost report contains no observed graph/nodes")
    graph_total = 0
    for index, graph in enumerate(value["graphs"]):
        if graph["id"] != index or type(graph["compute_calls"]) is not int or graph["compute_calls"] < 1:
            raise ValueError("invalid cost graph execution inventory")
        for key in ("bind_wall_ms", "compute_wall_ms", "arena_bytes"):
            finite_nonnegative(graph[key], key)
        graph_total += graph["compute_wall_ms"]
    categories, node_total = {}, 0
    for node in value["nodes"]:
        if (type(node["graph"]) is not int or not 0 <= node["graph"] < len(value["graphs"])
                or node["backend"] not in ("CPU", "BLAS", "UNASSIGNED")
                or type(node["calls"]) is not int or node["calls"] < 0):
            raise ValueError("cost node placement or execution count is invalid")
        if node["calls"] and node["category"] != "metadata" and node["backend"] == "UNASSIGNED":
            raise ValueError("an executed compute node has no observed backend")
        finite_nonnegative(node["wall_ms"], "node.wall_ms")
        node_total += node["wall_ms"]
        total = categories.setdefault(node["category"], {"calls": 0, "wall_ms": 0})
        total["calls"] += node["calls"]
        total["wall_ms"] += node["wall_ms"]
    if not any(node["calls"] for node in value["nodes"]):
        raise ValueError("cost graph did not execute any observed nodes")
    if not math.isclose(graph_total, value["graph_compute_wall_ms"], rel_tol=1e-6, abs_tol=.001) or node_total > graph_total + .001:
        raise ValueError("cost graph and node time scopes are inconsistent")
    if categories.keys() != value["categories"].keys():
        raise ValueError("cost category inventory differs from observed nodes")
    for category, expected in categories.items():
        reported = value["categories"][category]
        if expected["calls"] != reported["calls"] or not math.isclose(expected["wall_ms"], reported["wall_ms"], rel_tol=1e-6, abs_tol=.001):
            raise ValueError("cost category does not match its observed nodes")
    transfers = {}
    for row in value["transfers"]:
        if row["direction"] not in ("upload", "download") or type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("invalid observed tensor transfer")
        finite_nonnegative(row["wall_ms"], "transfer.wall_ms")
        total = transfers.setdefault(row["direction"], {"bytes": 0, "wall_ms": 0})
        total["bytes"] += row["bytes"]
        total["wall_ms"] += row["wall_ms"]
    if transfers.keys() != value["transfer_totals"].keys():
        raise ValueError("cost transfer inventory differs from observed transfers")
    for direction, expected in transfers.items():
        reported = value["transfer_totals"][direction]
        if expected["bytes"] != reported["bytes"] or not math.isclose(expected["wall_ms"], reported["wall_ms"], rel_tol=1e-6, abs_tol=.001):
            raise ValueError("cost transfer totals differ from observed transfers")


def run(args):
    from tools.maintenance.artifact_snapshot import native_snapshot
    if args.output.exists():
        raise FileExistsError("execution-cost requires a new output directory")
    if (args.backend, args.compute, args.cache) != ("cpu", "f32", "f32"):
        raise ValueError("execution-cost currently supports explicit CPU/F32 compute/F32 cache only")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("execution-cost timeout must be finite and positive")
    recipe = native_recipe(args)
    identities = native_snapshot(args.binary, args.model)
    identities.update(getattr(args, "configuration_inputs", {}))
    identities[str(args.image.resolve())] = sha256_file(args.image)
    verify_run_artifacts(identities)
    args.output.mkdir(parents=True)
    command = [str(args.binary.resolve()), "--model", str(args.model.resolve()), "--image", str(args.image.resolve()),
               "--text", args.text, "--backend", "cpu", "--threads", "4", "--output", str((args.output / "probe").resolve())]
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("execution-cost child process exceeded its timeout; no completed report published") from error
    (args.output / "probe.stdout.log").write_text(process.stdout)
    (args.output / "probe.stderr.log").write_text(process.stderr)
    raw_path = args.output / "probe/execution-cost.json"
    raw = read_json(raw_path)
    if (raw.get("kind") != "sam-execution-cost-run-v1" or raw.get("complete") is not True
            or raw.get("diagnostic_only") is not True or raw.get("performance_comparable") is not False
            or raw.get("model") != str(args.model.resolve()) or raw.get("image") != str(args.image.resolve())
            or raw.get("prompt") != args.text or raw.get("threads") != 4
            or raw.get("precision") != recipe["weight_precision"] or raw.get("storage_profile") != recipe["storage_profile"]
            or raw.get("arithmetic_profile") != expected_runtime_profile(recipe)
            or raw.get("quantization_modules") != recipe["quantization_modules"] or not native_weight_policy_matches(raw, recipe)):
        raise ValueError("cost probe result differs from the requested model/runtime recipe")
    validate_cost(raw["execution_cost"])
    stats = raw["runtime"]
    if not stats["cpu_nodes"] or stats["cuda_nodes"] or stats["metal_nodes"]:
        raise ValueError("cost inference escaped CPU placement")
    for direction, counter in (("upload", "host_upload_bytes"), ("download", "host_download_bytes")):
        if raw["execution_cost"]["transfer_totals"].get(direction, {}).get("bytes", 0) != stats[counter]:
            raise ValueError("cost transfers do not cover the image inference runtime counters")
    verify_run_artifacts(identities)
    cost = raw["execution_cost"]
    report = {"schema_version": 1, "kind": "sam-execution-cost-report-v1", "complete": True,
              "diagnostic_only": True, "performance_comparable": False, "recipe": recipe,
              "precision": precision_description({"recipe": recipe, "arithmetic_profile": raw["arithmetic_profile"]}),
              "image_sha256": identities[str(args.image.resolve())], "prompt": args.text,
              "artifact_sha256": identities, "raw": {"file": "probe/execution-cost.json", "sha256": sha256_file(raw_path)},
              "execution_cost": cost, "model_load_wall_ms": raw["model_load_wall_ms"],
              "rss_peak_bytes": raw["rss_peak_bytes"], "evaluator_sha256": sha256_file(Path(__file__))}
    lines = ["# CPU execution cost diagnostics", "", "Serialized node observation changes dispatch. These times are not performance benchmark results.", "",
             f"Graph compute wall time: {cost['graph_compute_wall_ms']:.6g} ms. Graph bind wall time: {cost['graph_bind_wall_ms']:.6g} ms.",
             "Node times below are already included in graph time. Transfer API times exclude host allocation and weight loading.", "",
             "| Component | Calls or bytes | Wall time (ms) |", "| --- | ---: | ---: |"]
    lines.extend(f"| {name} | {row['calls']} calls | {row['wall_ms']:.6g} |" for name, row in cost["categories"].items())
    lines.extend(f"| {name} | {row['bytes']} bytes | {row['wall_ms']:.6g} |" for name, row in cost["transfer_totals"].items())
    lines.extend(["", "Kernel-internal RHS packing and arithmetic: NOT_COLLECTED. Run the independent performance command to measure gains.", ""])
    (args.output / "report.md").write_text("\n".join(lines))
    write_json(args.output / "report.json", report)
    print(f"CPU execution cost diagnostics complete: {args.output}")
    return report
