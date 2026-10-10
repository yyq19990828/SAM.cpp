"""Direct child timing and separate PID-bound process memory observation."""

import csv
import io
from pathlib import Path
import subprocess
import time

from tools.convert.sam3_artifacts import read_json, write_json


def cuda_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True, timeout=5)
    rows = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) != 3:
            raise ValueError("invalid NVIDIA process memory observation")
        rows.append({"pid": int(row[0].strip()), "uuid": row[1].strip(), "bytes": int(row[2].strip()) * 1024 * 1024})
    return rows

def run_process(binary, model, recipe, case, kind, output, diagnostic=False, *, protocol=None, timeout=None):
    command = [str(binary), str(model), case["image"], case["prompt"], case["alternate"], recipe["backend"],
               recipe["feature_cache"], recipe["compute_mode"], kind, str(output)]
    if protocol is not None:
        command.extend([str(protocol[kind]["warmups"]), str(protocol[kind]["iterations"])])
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
                if timeout is not None and time.monotonic() - started > timeout:
                    raise RuntimeError(f"benchmark process timed out after {timeout}s; see {output.with_suffix('.log')}")
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
        try:
            others.update(row["pid"] for row in cuda_processes() if row["pid"] != process.pid)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            observer_errors.append(str(error))
    value = read_json(output / "result.json")
    if (value.get("complete") is not True or value.get("kind") != kind or value.get("backend") != recipe["backend"]
            or value.get("feature_cache") != recipe["feature_cache"] or value.get("cuda_compute") != recipe["compute_mode"]
            or value.get("precision") != recipe["weight_precision"]
            or (value.get("storage_profile") or "dense") != recipe["storage_profile"]
            or value.get("prompt") != case["prompt"] or value.get("alternate") != case["alternate"]
            or value.get("threads") != recipe["threads"]):
        raise ValueError("benchmark process identity differs from the frozen recipe/case")
    if protocol is not None:
        from tools.benchmark.precision_reporting import expected_runtime_profile
        if (value.get("warmups") != protocol[kind]["warmups"] or value.get("iterations") != protocol[kind]["iterations"]
                or value.get("quantization_modules") != recipe["quantization_modules"]
                or (gpu and value.get("cuda_device") != recipe["cuda_device"])
                or value.get("arithmetic_profile") != expected_runtime_profile(recipe)):
            raise ValueError("benchmark process precision policy or measurement protocol differs from the requested recipe")
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
