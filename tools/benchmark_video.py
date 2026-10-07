#!/usr/bin/env python3
"""Measure an original F32/F16/hybrid cell: 64 frames, 16 warmup, 48 measured.

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
import re
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
from validate_image import CUDA_F16_ARITHMETIC_PROFILE, validate_cuda_compute_mode

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


def encoding_timings(samples):
    """Existing runtime timers; historical reports may omit them entirely."""
    fields = ("image_ms", "inference_ms")
    present = [sum(name in sample["stats"]["runtime"] for name in fields) for sample in samples]
    if not any(present):
        return {"available": False, "reason": "Historical CLI omitted runtime encoding timers."}
    if any(count != len(fields) for count in present):
        raise ValueError("runtime encoding timer fields are incomplete")
    for sample in samples:
        for name in fields:
            value = sample["stats"]["runtime"][name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid runtime encoding timing")
    return {"available": True,
            "image_encoding": distribution([sample["stats"]["runtime"]["image_ms"] for sample in samples[WARMUP:]]),
            "detector_pipeline": distribution([sample["stats"]["runtime"]["inference_ms"] for sample in samples[WARMUP:]]),
            "scope": "image includes ViT/necks/geometry; detector pipeline includes prompt/fusion/detection/masks. Host transfers/allocation included; not GPU-only kernel time."}


def analyze_run(directory, expected_objects, precision, backend, width, height, cuda_device=0, cuda_compute="f32"):
    """Reject incomplete/wrong workloads before computing a benchmark summary."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    validate_cuda_compute_mode(backend, cuda_compute)
    if manifest.get("arithmetic_profile", "") != (CUDA_F16_ARITHMETIC_PROFILE if cuda_compute == "f16" else ""):
        raise ValueError("benchmark arithmetic profile differs from the requested compute mode")
    profile = HYBRID_PROFILE if precision == "hybrid" else ""
    if (manifest.get("complete") is not True or manifest.get("task") != "text_video"
            or manifest.get("frame_count") != FRAME_COUNT or manifest.get("threads") != 4
            or manifest.get("precision") != precision or manifest.get("storage_profile", "") != profile
            or manifest.get("backend") != backend or manifest.get("max_objects") != 8
            or manifest.get("width") != width or manifest.get("height") != height
            or manifest.get("prompt") != "truck"):
        raise ValueError("benchmark manifest is incomplete or has a different workload/profile/backend")
    if backend == "cuda" and (manifest.get("cuda_device") != cuda_device or not manifest.get("device_name")):
        raise ValueError("benchmark CUDA device differs or lacks device identity")
    if expected_objects not in (1, 4) or precision not in ("f32", "f16", "hybrid"):
        raise ValueError("video performance cells require 1/4 objects and f32/f16/hybrid storage")
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
        others = [other for other in ("cpu", "metal", "cuda") if other != backend]
        if (any(runtime.get(f"{other}_nodes", 0) != 0 for other in others)
                or (backend == "cuda" and runtime.get("blas_nodes", 0) != 0) or nodes <= last_nodes
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
    return {"samples": samples, "encoding_timings": encoding_timings(samples),
            "timings": {name: distribution([sample["stats"][name] for sample in samples[WARMUP:]])
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
    if sys.platform == "linux":
        cpuinfo = Path("/proc/cpuinfo").read_text()
        result["cpu_memory"] = {"cpu": next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                                                if line.startswith("model name")), platform.processor()),
                                "logical_cpus": os.cpu_count(),
                                "physical_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")}
        commands = {"nvidia": ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,power.limit,temperature.gpu,clocks.sm,clocks.mem",
                               "--format=csv,noheader"]}
    else:
        commands = {"power": ["/usr/bin/pmset", "-g", "batt"], "thermal": ["/usr/bin/pmset", "-g", "therm"],
                    "cpu_memory": ["/usr/sbin/sysctl", "machdep.cpu.brand_string", "hw.memsize", "hw.physicalcpu", "hw.logicalcpu"]}
    for name, command in commands.items():
        try:
            value = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[name] = value.stdout.strip() if value.returncode == 0 else "unavailable: " + value.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = "unavailable: " + str(error)
    return result


def process_timing(log_text, system):
    if system == "linux":
        match = re.search(r"^SAM_TIME (\d+) ([0-9.]+) ([0-9.]+) ([0-9.]+)$", log_text, re.MULTILINE)
        if not match:
            raise ValueError("missing GNU time peak RSS/wall/user/system receipt")
        return {"peak_rss_bytes": int(match[1]) * 1024, "wall_seconds": float(match[2]),
                "user_seconds": float(match[3]), "system_seconds": float(match[4]),
                "source": "GNU time %M KiB * 1024; %e/%U/%S seconds"}
    peak = re.search(r"^\s*(\d+)\s+maximum resident set size", log_text, re.MULTILINE)
    wall = re.search(r"([0-9.]+)\s+real\s+([0-9.]+)\s+user\s+([0-9.]+)\s+sys", log_text)
    if not peak or not wall:
        raise ValueError("missing /usr/bin/time -l peak RSS or wall-time receipt")
    return {"peak_rss_bytes": int(peak[1]), "wall_seconds": float(wall[1]),
            "user_seconds": float(wall[2]), "system_seconds": float(wall[3]),
            "source": "macOS time -l RSS bytes; real/user/sys seconds"}


def current_cuda_memory(pid):
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                                    "--format=csv,noheader,nounits"], text=True, timeout=5)
    devices = []
    for line in text.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 3 and fields[0] == str(pid):
            devices.append({"gpu_uuid": fields[1], "used_bytes": int(fields[2]) * 1024 * 1024})
    return devices


def validate_qualification_records(qualification, expected_objects):
    records = qualification.get("records", [])
    expected_ids = records[0]["active_ids"] if records else []
    if (len(records) != 17 or len(expected_ids) != expected_objects or len(set(expected_ids)) != len(expected_ids)
            or any(row.get("frame_index") != frame or row.get("active_ids") != expected_ids
                   or row.get("removed") or (frame and row.get("births"))
                   or sorted(item["id"] for item in row.get("visible", [])) != expected_ids
                   or any(item["area"] <= 0 for item in row.get("visible", [])) for frame, row in enumerate(records))
            or [(row["frame_index"], row["emitted_after_frame"]) for row in qualification.get("emitted", [])] != [(0,14),(1,15),(2,16)]):
        raise ValueError("original fixture qualification lacks 17 stable object/ID records and the expected delay")


def run(args):
    cuda_compute = getattr(args, "cuda_compute", None) or "f32"
    validate_cuda_compute_mode(args.backend, cuda_compute)
    if sys.platform not in ("darwin", "linux"):
        raise ValueError("video process timing requires macOS time -l or Linux GNU time")
    precision, reference, _ = check_provenance(args.model, args.reference)
    if precision not in ("f32", "f16", "hybrid") or args.threads != 4:
        raise ValueError("standard video benchmark cells use original F32/F16/hybrid and four threads")
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
    cuda_device = getattr(args, "cuda_device", None) or 0
    if accepted.get("cuda_compute", "f32") != cuda_compute:
        raise ValueError("full acceptance used a different CUDA compute mode")
    if cuda_compute == "f16" and (accepted.get("runtime_arithmetic_profile") != CUDA_F16_ARITHMETIC_PROFILE
                                  or accepted.get("output_quality_passed") is not True):
        raise ValueError("CUDA F16 benchmark requires passing final-output quality acceptance")
    if args.backend == "cuda" and accepted.get("cuda_device") != cuda_device:
        raise ValueError("full numerical acceptance used a different CUDA device")
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
    validate_qualification_records(qualification, workload["expected_objects"])
    if sha256_file(args.recipe_script) != fixture["generator_sha256"]:
        raise ValueError("fixture recipe script changed")
    verify_run_artifacts(qualification["source_sha256"])
    verify_run_artifacts({fixture["source"]: fixture["source_sha256"]})
    artifacts = freeze_run_artifacts(args.build_dir, executable, args.model, expected_model)
    project = Path(__file__).resolve().parents[1]
    sources = [*project.glob("include/**/*.hpp"), *project.glob("examples/*.cpp"), *project.glob("examples/*.hpp"),
               project / "cmake/patches/ggml-precise-metal.patch", project / "cmake/patches/ggml-precise-cuda.patch",
               project / "cmake/ggml.cmake", project / "cmake/prepare_ggml.cmake", project / "tools/sam3_tensor_schema.json",
               *[project / "tools" / name for name in ("benchmark_video.py", "sam3_artifacts.py", "validate_video.py", "validate_image.py", "sam3_gguf.py")],
               args.fixture_manifest, args.qualification, args.validation, args.recipe_script]
    source_hashes = {str(path.absolute()): sha256_file(path) for path in sources}
    args.output.mkdir(parents=True, exist_ok=False)
    actual = args.output / "run"
    time_arguments = ["-f", "SAM_TIME %M %e %U %S"] if sys.platform == "linux" else ["-l"]
    command = ["/usr/bin/time", *time_arguments, str(executable.resolve()), "--model", str(args.model.resolve()), "--frames", str(frames),
               "--text", "truck", "--backend", args.backend, "--threads", "4", "--max-objects", "8", "--output", str(actual)]
    if args.backend == "cuda":
        command.extend(["--cuda-device", str(cuda_device)])
        command.extend(["--cuda-compute", cuda_compute])
    report = {"schema_version": 1, "complete": False, "passed": False, "kind": "m2-video-performance-64-16-48",
              "numerical_acceptance": ("separate passing full original-reference final-output quality receipt"
                                       if cuda_compute == "f16" else "separate passing full original-reference receipt"),
              "backend": args.backend,
              "precision": precision, "storage_profile": HYBRID_PROFILE if precision == "hybrid" else "",
              "model_sha256": expected_model, "checkpoint_sha256": reference["checkpoint"]["sha256"],
              "workload": args.workload, "expected_objects": workload["expected_objects"], "frame_count": FRAME_COUNT,
              "warmup_frames": WARMUP, "measured_frames": FRAME_COUNT-WARMUP, "threads": 4, "command": command,
              "run_artifact_sha256": artifacts, "source_sha256": source_hashes, "input_sha256": inputs,
              "conditions_before": conditions(), "failures": []}
    if args.backend == "cuda":
        report.update(cuda_compute=cuda_compute,
                      arithmetic_profile=CUDA_F16_ARITHMETIC_PROFILE if cuda_compute == "f16" else "")
    write_json(args.output / "benchmark.json", report)
    process, rss_samples, rss_errors, cuda_samples, cuda_errors = None, [], [], [], []
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
                        if args.backend == "cuda":
                            try:
                                if sys.platform == "linux" and "loader" not in report:
                                    maps = Path(f"/proc/{sample['pid']}/maps").read_text()
                                    loaded = {line.split(maxsplit=5)[5] for line in maps.splitlines()
                                              if len(line.split(maxsplit=5)) == 6 and "/libggml-cuda.so" in line}
                                    if loaded:
                                        expected_libraries = {str(Path(path).resolve()): digest for path, digest in artifacts.items()}
                                        if any(path not in expected_libraries or sha256_file(path) != expected_libraries[path]
                                               for path in loaded):
                                            raise RuntimeError("loaded CUDA library differs from the qualified build")
                                        (args.output / "loader.maps").write_text(maps)
                                        report["loader"] = {"pid": sample["pid"], "cuda_libraries": sorted(loaded)}
                                devices = current_cuda_memory(sample["pid"])
                                if devices:
                                    cuda_samples.append({"elapsed_seconds": elapsed, "devices": devices,
                                                         "total_used_bytes": sum(device["used_bytes"] for device in devices)})
                            except (OSError, ValueError, subprocess.SubprocessError) as error:
                                if str(error) not in cuda_errors:
                                    cuda_errors.append(str(error))
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    if str(error) not in rss_errors:
                        rss_errors.append(str(error))
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f"sam_video/time exited {process.returncode}; see sam_video.log")
        if args.backend == "cuda" and sys.platform == "linux" and "loader" not in report:
            raise RuntimeError("video benchmark lacks actual CUDA loader evidence")
        verify_run_artifacts(artifacts); verify_run_artifacts(inputs); verify_run_artifacts(source_hashes)
        outputs = freeze_output_files(actual)
        report["output_sha256"] = outputs
        write_json(args.output / "benchmark.json", report)
        report.update(analyze_run(actual, workload["expected_objects"], precision, args.backend, fixture["width"], fixture["height"], cuda_device, cuda_compute))
        log_text = (args.output / "sam_video.log").read_text()
        report["process"] = process_timing(log_text, sys.platform)
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
        if args.backend == "cuda":
            report["cuda_memory"] = {"available": bool(cuda_samples), "cuda_device": cuda_device,
                                     "sampled_peak_bytes": max((sample["total_used_bytes"] for sample in cuda_samples), default=None),
                                     "samples": cuda_samples, "errors": cuda_errors,
                                     "scope": "nvidia-smi process memory in MiB, sampled about once per second; includes CUDA context/pools, may miss transient peaks."}
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
    parser.add_argument("--backend", required=True, choices=("cpu", "metal", "cuda"))
    parser.add_argument("--cuda-compute", choices=("f32", "f16"), help="Must match the original-reference acceptance receipt")
    parser.add_argument("--cuda-device", type=int, help="Index among CUDA-visible devices (default: 0)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    args = parser.parse_args()
    if args.cuda_device is not None and (args.backend != "cuda" or args.cuda_device < 0):
        parser.error("--cuda-device requires backend cuda and a nonnegative index")
    if args.cuda_compute is not None and args.backend != "cuda":
        parser.error("--cuda-compute requires backend cuda")
    if args.threads != 4 or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("standard M2 cells require four threads and a positive finite timeout")
    try:
        parser.exit(run(args))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"benchmark failed: {error}\n")


if __name__ == "__main__":
    main()
