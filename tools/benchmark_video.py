#!/usr/bin/env python3
"""Measure one original-F16 or hybrid M2 cell: 64 frames, 16 warmup, 48 measured.

Requires a passing full original-reference validation for the exact model,
backend and executable, plus an original-module qualification of the independent
one/four-object fixture. This is a performance report, not a numerical oracle.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

from sam3_artifacts import (artifact_path, freeze_output_files, freeze_run_artifacts,
                           read_array, read_json, sha256_file, verify_output_files,
                           verify_run_artifacts, write_json)
from sam3_gguf import HYBRID_PROFILE
from validate_video import check_provenance, read_objects

FRAME_COUNT = 64
WARMUP = 16


def distribution(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    return {"count": len(values), "median_ms": statistics.median(values),
            "p95_ms": ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower),
            "p95_method": "linear interpolation at (n-1)*0.95", "minimum_ms": ordered[0], "maximum_ms": ordered[-1]}


def analyze_run(directory, expected_objects, precision, backend, width, height):
    """Reject incomplete/wrong workloads before computing a benchmark summary."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    profile = HYBRID_PROFILE if precision == "hybrid" else ""
    if (manifest.get("complete") is not True or manifest.get("task") != "text_video"
            or manifest.get("frame_count") != FRAME_COUNT or manifest.get("threads") != 4
            or manifest.get("precision") != precision or manifest.get("storage_profile", "") != profile
            or manifest.get("backend") != backend or manifest.get("max_objects") != 8
            or manifest.get("width") != width or manifest.get("height") != height
            or manifest.get("prompt") != "truck"):
        raise ValueError("benchmark manifest is incomplete or has a different workload/profile/backend")
    if expected_objects not in (1, 4) or precision not in ("f16", "hybrid"):
        raise ValueError("M2 performance cells require 1/4 objects and f16/hybrid storage")
    if type(manifest.get("model_load_ms")) not in (int, float) or not math.isfinite(manifest["model_load_ms"]) or manifest["model_load_ms"] < 0:
        raise ValueError("invalid model-load timing")
    if list((directory / "tensors").rglob("*.bin")):
        raise ValueError("benchmark contains tensor dumps")
    samples, active, initial, output_delays = [], set(), None, []
    last_nodes = 0
    for frame in range(FRAME_COUNT):
        stats = read_json(directory / "trace" / f"{frame:06d}-stats.json")
        trace = read_json(directory / "trace" / f"{frame:06d}.json")
        if trace.get("frame_index") != frame:
            raise ValueError(f"frame {frame}: trace index differs")
        births = [item["id"] for item in trace["births"]]
        removed = trace["removed"]
        if any(type(value) is not int or value < 0 for value in births + removed):
            raise ValueError(f"frame {frame}: malformed object IDs")
        if len(set(births)) != len(births) or active.intersection(births) or not set(removed).issubset(active):
            raise ValueError(f"frame {frame}: duplicate/recycled object IDs")
        active.update(births); active.difference_update(removed)
        if initial is None:
            initial = active.copy()
        if len(active) != expected_objects or active != initial or (frame and (births or removed)):
            raise ValueError(f"frame {frame}: fixed object count/ID continuity failed")
        group_ids = [identifier for group in trace["groups"] for identifier in group["ids"]]
        if frame and (len(group_ids) != expected_objects or set(group_ids) != initial):
            raise ValueError(f"frame {frame}: logical group IDs changed")
        emitted = FRAME_COUNT if frame == FRAME_COUNT - 1 else max(0, frame - 13)
        if (stats["accepted_frames"] != frame + 1 or stats["emitted_frames"] != emitted
                or stats["pending_frames"] != frame + 1 - emitted or stats["pending_high_water"] > 15
                or stats["active_objects"] != expected_objects or stats["retained_records"] > 27 * expected_objects
                or stats["retained_records_high_water"] > 27 * expected_objects):
            raise ValueError(f"frame {frame}: frame/output/state bound differs")
        runtime = stats["runtime"]
        nodes = runtime[f"{backend}_nodes"]
        other = "cpu" if backend == "metal" else "metal"
        if (runtime[f"{other}_nodes"] != 0 or nodes <= last_nodes
                or runtime["vision_encodes"] != frame + 1 or runtime["inferences"] != frame + 1
                or runtime["text_encodes"] != 1):
            raise ValueError(f"frame {frame}: graph placement or per-session encode counts differ")
        last_nodes = nodes
        for name in ("frame_ms", "tracker_ms", "memory_ms"):
            if type(stats[name]) not in (int, float) or not math.isfinite(stats[name]) or stats[name] < 0:
                raise ValueError(f"frame {frame}: invalid stage timing")
        for name in ("weight_buffer_bytes", "compute_buffer_bytes", "process_peak_rss_bytes"):
            if type(runtime[name]) is not int or runtime[name] <= 0:
                raise ValueError(f"frame {frame}: invalid allocation/RSS counter")
        if type(stats["retained_memory_bytes"]) is not int or stats["retained_memory_bytes"] < 0:
            raise ValueError(f"frame {frame}: invalid retained-memory counter")
        result, objects = read_objects(directory / f"{frame:06d}", frame, width, height, False)
        if set(objects) != initial:
            raise ValueError(f"frame {frame}: emitted object count/ID continuity failed")
        for obj in objects.values():
            if not read_array(directory / f"{frame:06d}", obj["mask"], "uint8", False).any():
                raise ValueError(f"frame {frame}: benchmark object mask is empty")
        expected_delay = min(frame + 14, FRAME_COUNT - 1)
        if result["emitted_after_frame"] != expected_delay:
            raise ValueError(f"frame {frame}: first-output/final-drain policy differs")
        output_delays.append(result["emitted_after_frame"])
        samples.append({"frame_index": frame, "warmup": frame < WARMUP, "active_ids": sorted(active),
                        "stats": stats, "emitted_ids": sorted(objects), "emitted_after_frame": result["emitted_after_frame"]})
    if manifest["stats"] != samples[-1]["stats"]:
        raise ValueError("final manifest counters differ from the last frame")
    weights = {sample["stats"]["runtime"]["weight_buffer_bytes"] for sample in samples}
    compute = {sample["stats"]["runtime"]["compute_buffer_bytes"] for sample in samples[WARMUP:]}
    if len(weights) != 1 or len(compute) != 1:
        raise ValueError("weight/compute allocation did not plateau after warmup")
    return {"samples": samples, "timings": {name: distribution([sample["stats"][name] for sample in samples[WARMUP:]])
                                            for name in ("frame_ms", "tracker_ms", "memory_ms")},
            "first_output": {"emitted_after_frame": output_delays[0], "accepted_frames": output_delays[0] + 1,
                             "cumulative_frame_ms": sum(sample["stats"]["frame_ms"] for sample in samples[:15])},
            "final_drain": {"frame_index": 63, "outputs_emitted": 15, "included_in_measured_samples": True},
            "allocation": {"weight_buffer_bytes": next(iter(weights)), "maximum_graph_compute_buffer_bytes": next(iter(compute)),
                           "compute_high_water_constant_after_warmup": True,
                           "retained_records_max": max(sample["stats"]["retained_records"] for sample in samples),
                           "retained_memory_bytes_max": max(sample["stats"]["retained_memory_bytes"] for sample in samples),
                           "scope": "Observed graph compute high-water plateau and bounded retained records; not a universal live-memory bound."},
            "model_load_ms": manifest["model_load_ms"], "final_stats": manifest["stats"]}


def current_rss(parent_pid, executable_name):
    """Read only the timed executable's current RSS, not the wrapper's RSS."""
    text = subprocess.check_output(["/bin/ps", "-axo", "pid=,ppid=,rss=,comm="], text=True, timeout=5)
    rows = [line.strip().split(maxsplit=3) for line in text.splitlines()]
    for row in rows:
        if len(row) == 4 and int(row[1]) == parent_pid and Path(row[3]).name == executable_name:
            return {"pid": int(row[0]), "rss_bytes": int(row[2]) * 1024}
    return None


def conditions():
    result = {"time_unix": time.time(), "platform": platform.platform(), "machine": platform.machine()}
    for name, command in {"power": ["/usr/bin/pmset", "-g", "batt"], "thermal": ["/usr/bin/pmset", "-g", "therm"],
                          "cpu_memory": ["/usr/sbin/sysctl", "machdep.cpu.brand_string", "hw.memsize", "hw.physicalcpu", "hw.logicalcpu"]}.items():
        try:
            value = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[name] = value.stdout.strip() if value.returncode == 0 else "unavailable: " + value.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = "unavailable: " + str(error)
    return result


def run(args):
    if sys.platform != "darwin":
        raise ValueError("this measured M2 protocol requires macOS /usr/bin/time -l; no cross-platform RSS conversion is assumed")
    precision, reference, _ = check_provenance(args.model, args.reference)
    if precision not in ("f16", "hybrid") or args.threads != 4:
        raise ValueError("standard M2 benchmark cells use original F16/hybrid and four threads")
    accepted = read_json(args.validation)
    expected_model = read_json(args.model.with_suffix(args.model.suffix + ".manifest.json"))["output"]["sha256"]
    executable = args.build_dir / "examples/sam_video"
    expected_cases = {case["id"] for case in reference["cases"]}
    accepted_cases = accepted.get("cases", [])
    if (accepted.get("passed") is not True or accepted.get("eligible_for_milestone") is not True
            or accepted.get("precision") != precision or accepted.get("backend") != args.backend
            or accepted.get("model_sha256") != expected_model or accepted.get("binary_sha256") != sha256_file(executable)
            or accepted.get("reference_manifest_sha256") != sha256_file(args.reference / "manifest.json")
            or len(expected_cases) != 5 or len(accepted_cases) != 5
            or {case.get("id") for case in accepted_cases} != expected_cases
            or not all(case.get("passed") is True and case.get("failures") == []
                       and isinstance(case.get("output_sha256"), dict) and case["output_sha256"]
                       for case in accepted_cases)):
        raise ValueError("a passing current full original-reference validation is required for this exact cell")
    verify_run_artifacts(accepted["run_artifact_sha256"])
    fixture = read_json(args.fixture_manifest)
    workload = next((item for item in fixture["workloads"] if item["id"] == args.workload), None)
    qualification = read_json(args.qualification)
    if (fixture.get("complete") is not True or fixture.get("frames") != FRAME_COUNT or fixture.get("warmup_frames") != WARMUP
            or fixture.get("measured_frames") != FRAME_COUNT - WARMUP or fixture.get("prompt") != "truck"
            or fixture.get("max_objects") != 8 or workload is None or workload["expected_objects"] not in (1, 4)
            or qualification.get("passed") is not True or qualification.get("qualified_for_performance") is not True
            or qualification.get("reference_kind") != "official-original-performance-prefix"
            or qualification.get("declared_frame_count") != FRAME_COUNT or qualification.get("prefix_frames") != 17
            or qualification.get("workload") != args.workload or qualification.get("expected_objects") != workload["expected_objects"]
            or qualification.get("fixture_manifest_sha256") != sha256_file(args.fixture_manifest)
            or qualification.get("checkpoint_sha256") != reference["checkpoint"]["sha256"]):
        raise ValueError("fixture needs an original-module 17-frame object/continuity qualification with declared length 64")
    frames = artifact_path(args.fixture_manifest.parent, workload["directory"])
    inputs = {str((frames / f"{i:06d}.png").absolute()): digest for i, digest in enumerate(workload["input_sha256"])}
    if len(inputs) != FRAME_COUNT or len(list(frames.glob("*.png"))) != FRAME_COUNT:
        raise ValueError("performance fixture must contain exactly 64 declared PNG frames")
    verify_run_artifacts(inputs)
    qualified_inputs = {Path(name).name: digest for name, digest in qualification["input_sha256"].items()}
    if {Path(name).name: digest for name, digest in inputs.items()} != qualified_inputs:
        raise ValueError("qualified PNG identities differ from this workload")
    records = qualification.get("records", [])
    expected_ids = records[0]["active_ids"] if records else []
    if (len(records) != 17 or len(expected_ids) != workload["expected_objects"] or len(set(expected_ids)) != len(expected_ids)
            or any(row.get("frame_index") != frame or row.get("active_ids") != expected_ids
                   or row.get("removed") or (frame and row.get("births"))
                   or sorted(item["id"] for item in row.get("visible", [])) != expected_ids
                   or any(item["area"] <= 0 for item in row.get("visible", [])) for frame, row in enumerate(records))
            or [(row["frame_index"], row["emitted_after_frame"]) for row in qualification.get("emitted", [])] != [(0,14),(1,15),(2,16)]):
        raise ValueError("original fixture qualification lacks 17 stable object/ID records and the expected delay")
    if sha256_file(args.recipe_script) != fixture["generator_sha256"]:
        raise ValueError("fixture recipe script changed")
    verify_run_artifacts(qualification["source_sha256"])
    verify_run_artifacts({fixture["source"]: fixture["source_sha256"]})
    artifacts = freeze_run_artifacts(args.build_dir, executable, args.model, expected_model)
    project = Path(__file__).resolve().parents[1]
    sources = [*project.glob("include/**/*.hpp"), *project.glob("examples/*.cpp"), *project.glob("examples/*.hpp"),
               project / "cmake/patches/ggml-precise-metal.patch", project / "cmake/ggml.cmake", project / "tools/sam3_tensor_schema.json",
               *[project / "tools" / name for name in ("benchmark_video.py", "sam3_artifacts.py", "validate_video.py", "validate_image.py", "sam3_gguf.py")],
               args.fixture_manifest, args.qualification, args.validation, args.recipe_script]
    source_hashes = {str(path.absolute()): sha256_file(path) for path in sources}
    args.output.mkdir(parents=True, exist_ok=False)
    actual = args.output / "run"
    command = ["/usr/bin/time", "-l", str(executable.resolve()), "--model", str(args.model.resolve()), "--frames", str(frames),
               "--text", "truck", "--backend", args.backend, "--threads", "4", "--max-objects", "8", "--output", str(actual)]
    report = {"schema_version": 1, "complete": False, "passed": False, "kind": "m2-video-performance-64-16-48",
              "numerical_acceptance": "separate passing full original-reference receipt", "backend": args.backend,
              "precision": precision, "storage_profile": HYBRID_PROFILE if precision == "hybrid" else "",
              "model_sha256": expected_model, "checkpoint_sha256": reference["checkpoint"]["sha256"],
              "workload": args.workload, "expected_objects": workload["expected_objects"], "frame_count": FRAME_COUNT,
              "warmup_frames": WARMUP, "measured_frames": FRAME_COUNT-WARMUP, "threads": 4, "command": command,
              "run_artifact_sha256": artifacts, "source_sha256": source_hashes, "input_sha256": inputs,
              "conditions_before": conditions(), "failures": []}
    write_json(args.output / "benchmark.json", report)
    process, rss_samples, rss_errors = None, [], []
    first_output_observed = None
    try:
        verify_run_artifacts(artifacts); verify_run_artifacts(source_hashes); verify_run_artifacts(inputs)
        with (args.output / "sam_video.log").open("w") as log, (args.output / "current-rss.jsonl").open("w") as rss_log:
            started = time.monotonic()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > args.timeout_seconds:
                    raise TimeoutError("benchmark process timed out")
                if first_output_observed is None and (actual / "000000/results.json").is_file():
                    first_output_observed = elapsed
                try:
                    sample = current_rss(process.pid, executable.name)
                    if sample is not None:
                        available = sorted((actual / "trace").glob("??????-stats.json"))
                        sample.update(elapsed_seconds=elapsed, last_stats_file_frame=int(available[-1].name[:6]) if available else None)
                        rss_samples.append(sample)
                        rss_log.write(json.dumps(sample) + "\n"); rss_log.flush()
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    if str(error) not in rss_errors:
                        rss_errors.append(str(error))
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f"sam_video/time exited {process.returncode}; see sam_video.log")
        verify_run_artifacts(artifacts); verify_run_artifacts(inputs); verify_run_artifacts(source_hashes)
        outputs = freeze_output_files(actual)
        report["output_sha256"] = outputs
        write_json(args.output / "benchmark.json", report)
        report.update(analyze_run(actual, workload["expected_objects"], precision, args.backend, fixture["width"], fixture["height"]))
        import re
        log_text = (args.output / "sam_video.log").read_text()
        peak = re.search(r"^\s*(\d+)\s+maximum resident set size", log_text, re.MULTILINE)
        wall = re.search(r"([0-9.]+)\s+real\s+([0-9.]+)\s+user\s+([0-9.]+)\s+sys", log_text)
        if not peak or not wall:
            raise ValueError("missing /usr/bin/time -l peak RSS or wall-time receipt")
        report["process"] = {"peak_rss_bytes": int(peak.group(1)), "wall_seconds": float(wall.group(1)),
                             "user_seconds": float(wall.group(2)), "system_seconds": float(wall.group(3))}
        report["first_output"]["observed_elapsed_seconds"] = first_output_observed
        report["first_output"]["observation_scope"] = "First output-file observation, sampled about once per second; includes load/decode/file work."
        measured_rss = [sample["rss_bytes"] for sample in rss_samples if sample["last_stats_file_frame"] is not None
                        and WARMUP <= sample["last_stats_file_frame"] < FRAME_COUNT-1]
        report["current_rss"] = {"available": bool(rss_samples), "source": "ps RSS KiB * 1024 of the timed sam_video child",
                                  "sampling_interval_seconds": 1, "samples": rss_samples, "errors": rss_errors,
                                  "after_warmup_before_final_drain_min_bytes": min(measured_rss) if measured_rss else None,
                                  "after_warmup_before_final_drain_max_bytes": max(measured_rss) if measured_rss else None,
                                  "scope": "Asynchronous current RSS observations, tagged by the latest stats file; not exact frame-boundary or peak-RSS substitutes."}
        if not rss_samples:
            report["current_rss"]["unavailable_reason"] = "; ".join(rss_errors) or "timed sam_video child was not observed by ps"
        verify_output_files(actual, outputs); verify_run_artifacts(artifacts); verify_run_artifacts(source_hashes); verify_run_artifacts(inputs)
        report.update(complete=True, passed=True, conditions_after=conditions())
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration, subprocess.SubprocessError) as error:
        report["failures"].append(str(error))
        report["current_rss_partial"] = rss_samples
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            except ProcessLookupError:
                process.wait()
        write_json(args.output / "benchmark.json", report)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output / 'benchmark.json'}")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "model", "reference", "validation", "fixture-manifest", "qualification", "recipe-script", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--workload", required=True, choices=("one-object", "four-object"))
    parser.add_argument("--backend", required=True, choices=("cpu", "metal"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    args = parser.parse_args()
    if args.threads != 4 or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("standard M2 cells require four threads and a positive finite timeout")
    try:
        parser.exit(run(args))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"benchmark failed: {error}\n")


if __name__ == "__main__":
    main()
