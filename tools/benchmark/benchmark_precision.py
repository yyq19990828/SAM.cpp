#!/usr/bin/env python3
"""Run frozen AB/BA/AB latency and independent process-memory measurements."""

import argparse
import csv
import io
from pathlib import Path
import subprocess
import sys
import time

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.validation.precision_acceptance import gate_identity, load_gates
from tools.maintenance.precision_artifacts import (archive_sources, campaign_check, native_recipe, native_snapshot,
                                 source_snapshot, verify_export_artifacts)
from tools.benchmark.precision_performance import ORDER, assess_performance, validate_cases
from tools.validation.prepare_coco_acceptance import load_precision_dataset
from tools.convert.sam3_artifacts import read_json, sha256_file, verify_run_artifacts, write_json


def cuda_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True, timeout=5)
    rows = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) != 3:
            raise ValueError("invalid NVIDIA process memory observation")
        rows.append({"pid": int(row[0].strip()), "uuid": row[1].strip(), "bytes": int(row[2].strip()) * 1024 * 1024})
    return rows


def run_process(binary, model, recipe, case, kind, output, diagnostic=False):
    command = [str(binary), str(model), case["image"], case["prompt"], case["alternate"], recipe["backend"],
               recipe["feature_cache"], recipe["compute_mode"], kind, str(output)]
    gpu, memory = recipe["backend"] == "cuda", kind == "memory"
    before = cuda_processes() if gpu else []
    if before and not diagnostic:
        raise ValueError("GPU has another compute process before the benchmark; no measurements started")
    samples, observer_errors, others, process = [], [], {row["pid"] for row in before}, None
    started = time.monotonic()
    with output.with_suffix(".log").open("x") as log:
        # Launch directly so NVML's PID identifies the measured executable.
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if memory:
                    sample = {"elapsed_ms": (time.monotonic() - started) * 1000, "phase": "load",
                              "rss_bytes": None, "gpu_bytes": None, "gpu_uuids": []}
                    phase = output / "phase.txt"
                    if phase.exists():
                        sample["phase"] = phase.read_text().strip()
                    status = Path(f"/proc/{process.pid}/status")
                    try:
                        status_lines = status.read_text().splitlines()
                    except FileNotFoundError:
                        status_lines = []  # The child may exit between poll and /proc sampling.
                    for line in status_lines:
                        if line.startswith("VmRSS:"):
                            sample["rss_bytes"] = int(line.split()[1]) * 1024
                    if gpu:
                        try:
                            rows = cuda_processes()
                            own = [row for row in rows if row["pid"] == process.pid]
                            others.update(row["pid"] for row in rows if row["pid"] != process.pid)
                            if own:
                                sample["gpu_bytes"] = sum(row["bytes"] for row in own)
                                sample["gpu_uuids"] = [row["uuid"] for row in own]
                        except (OSError, ValueError, subprocess.SubprocessError) as error:
                            observer_errors.append(str(error))
                    samples.append(sample)
                time.sleep(.05 if memory else .1)
            if process.returncode:
                raise RuntimeError(f"benchmark exited {process.returncode}; see {output.with_suffix('.log')}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    if gpu:
        others.update(row["pid"] for row in cuda_processes() if row["pid"] != process.pid)
    value = read_json(output / "result.json")
    if (value.get("complete") is not True or value.get("kind") != kind or value.get("backend") != recipe["backend"]
            or value.get("feature_cache") != recipe["feature_cache"] or value.get("cuda_compute") != recipe["compute_mode"]
            or value.get("precision") != recipe["weight_precision"]
            or (value.get("storage_profile") or "dense") != recipe["storage_profile"]
            or value.get("prompt") != case["prompt"] or value.get("alternate") != case["alternate"]
            or value.get("threads") != recipe["threads"]):
        raise ValueError("benchmark process identity differs from the frozen recipe/case")
    value.update({"process_id": process.pid, "sampled_pid": process.pid if memory and gpu else None,
                  "memory_sampler_enabled": memory, "other_compute_pids": sorted(others), "observer_errors": observer_errors,
                  "diagnostic_only": diagnostic})
    if memory:
        observed = [row for row in samples if row["gpu_bytes"] is not None]
        value["gpu_peak_bytes"] = max((row["gpu_bytes"] for row in observed), default=None)
        value["gpu_samples"] = len(observed)
        value["load_gpu_peak_bytes"] = max((row["gpu_bytes"] for row in observed if row["phase"] == "load"), default=None)
        value["inference_gpu_peak_bytes"] = max((row["gpu_bytes"] for row in observed if row["phase"] != "load"), default=None)
        value["observer_interval_ms_max"] = max((b["elapsed_ms"] - a["elapsed_ms"] for a, b in zip(samples, samples[1:])), default=None)
        value["pool_accounting"] = "Included in sampled process GPU memory; GGML does not expose a separate pool counter"
        write_json(output / "memory-samples.json", {"pid": process.pid, "samples": samples,
                   "scope": "PID-bound NVIDIA MiB observations, target 50 ms plus query time; phase tags use the latest phase file and short GPU peaks can be missed. RSS uses process getrusage high-water."})
    return value


def accepted_quality(path, recipe_hash, campaign_hash, policy_version=2):
    value = read_json(path)
    if (value.get("schema_version") != policy_version or value.get("kind") != f"sam3-precision-quality-v{policy_version}"
            or value.get("complete") is not True or value.get("phase") != "evaluation"
            or value.get("final_evaluation_completed") is not True or value.get("gates_sha256") != gate_identity(policy_version)[1]
            or value.get("candidate_recipe_sha256") != recipe_hash or value.get("campaign_sha256") != campaign_hash
            or value.get("absolute_quality_status") != "PASS"
            or value.get("incremental_quality_status") not in ("PASS", "NOT_APPLICABLE")
            or value.get("quality_status") != "PASS"):
        raise ValueError("performance labels require passing final quality for this exact frozen recipe")
    verify_export_artifacts(path.parent, value)
    return value


def run(args):
    if sys.prefix == sys.base_prefix or sys.platform != "linux":
        raise RuntimeError("use the isolated reference environment on Linux for this measurement runner")
    if args.output.exists():
        raise FileExistsError("performance output must be a new directory")
    campaign = read_json(args.campaign)
    version = campaign["schema_version"]
    gates = load_gates(policy_version=version)
    candidates = {row["id"]: row for row in campaign["runs"]}
    candidate = candidates[args.candidate]
    if (not candidate.get("performance_baseline") or not campaign.get("performance_cases")
            or not campaign.get("benchmark_binary")):
        raise ValueError("this campaign did not freeze a performance comparison for the candidate")
    baseline = candidates[candidate["performance_baseline"]]
    if version == 3 and baseline["recipe"]["quality_tier"] != candidate["recipe"]["quality_tier"]:
        raise ValueError("v3 performance baselines require the same declared quality tier")
    if candidate["id"] == baseline["id"]:
        raise ValueError("a deployment baseline cannot claim an improvement over itself")
    if candidate["recipe"]["backend"] not in ("cpu", "cuda"):
        raise ValueError("this performance runner currently measures CPU or CUDA")
    dataset_path = Path(candidate["development_manifest"]).parent / "dataset.json"
    campaign_hash = campaign_check(args.campaign, dataset_path, candidate["recipe"], "evaluation")
    campaign_check(args.campaign, dataset_path, baseline["recipe"], "evaluation")
    c_quality = accepted_quality(args.candidate_quality, candidate["recipe_sha256"], campaign_hash, version)
    b_quality = accepted_quality(args.baseline_quality, baseline["recipe_sha256"], campaign_hash, version)
    cp, bp = c_quality["absolute"]["profile"], b_quality["absolute"]["profile"]
    mask_floor = "task_mask_iou_min" if version == 3 else "mask_iou_min"
    if any(bp[name] > cp[name] for name in cp if name.endswith("_max")) or bp[mask_floor] < cp[mask_floor]:
        raise ValueError("deployment baseline has not passed an equally strict quality profile")
    cases_path = Path(campaign["performance_cases"])
    cases = validate_cases(read_json(cases_path), load_precision_dataset(dataset_path), campaign["dataset_sha256"], gates)["cases"]
    binary = Path(campaign["benchmark_binary"])
    identities = source_snapshot()
    for path in (args.campaign, args.candidate_quality, args.baseline_quality, cases_path):
        identities[str(path.resolve())] = sha256_file(path)
    for entry in (baseline, candidate):
        recipe = entry["recipe"]
        actual = native_recipe(Path(entry["export_binary"]), Path(entry["model"]), recipe["backend"], recipe["compute_mode"], recipe["feature_cache"],
                               version, recipe.get("quality_tier"))
        if actual != recipe:
            raise ValueError("hardware, model, driver or binary changed after the campaign freeze")
        identities.update(native_snapshot(binary, Path(entry["model"])))
    identities.update(read_json(cases_path)["artifact_sha256"])
    args.output.mkdir(parents=True)
    records = []
    for kind in ("latency", "memory"):
        for case in cases:
            for pair in range(3):
                for variant in ORDER[pair]:
                    entry = baseline if variant == "baseline" else candidate
                    name = f"{kind}-{case['id']}-{pair}-{variant}"
                    print(name, flush=True)
                    row = run_process(binary, Path(entry["model"]), entry["recipe"], case, kind, args.output / name)
                    row.update({"case": case["id"], "pair": pair, "variant": variant})
                    records.append(row)
                    write_json(args.output / (name + ".json"), row)
    result = assess_performance(records, cases, gates, candidate["recipe"]["backend"])
    verify_run_artifacts(identities)
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            identities[str(path.resolve())] = sha256_file(path)
    archived = archive_sources(args.output, identities)
    separation = ({"qualification_status": "NOT_RUN", "performance_required_for_qualification": False}
                  if version == 3 else {})
    write_json(args.output / "performance.json", {"schema_version": version, "kind": f"sam3-precision-performance-v{version}",
               "gates_sha256": gate_identity(version)[1], "campaign_sha256": campaign_hash,
               "candidate_recipe_sha256": candidate["recipe_sha256"], "baseline_recipe_sha256": baseline["recipe_sha256"],
               "quality_prerequisites_passed": True, "backend": candidate["recipe"]["backend"],
               "artifact_sha256": identities, "archived_sources": archived, "records": records, **result,
               "arithmetic_status": "NOT_RUN", "deployment_status": "NOT_RUN", **separation,
               "exclusivity": "No other CUDA compute PID observed before/after each latency process and during independent memory sampling"})
    print({workload: row["performance_labels"] for workload, row in result["workloads"].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "candidate-quality", "baseline-quality", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Precision performance failed: {error}\n")


if __name__ == "__main__":
    main()
