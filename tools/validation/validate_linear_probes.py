#!/usr/bin/env python3
"""Validate opt-in CUDA linear probes on numerical boundaries and optional real layers."""

import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmark.prepare_linear_probe import write_case
from tools.convert.sam3_artifacts import read_json, sha256_file
from tools.validation.verify_linear_probe import verify


MODES = ("ggml-q8", "ggml-f16", "int8-token-f32", "int8-token-f16", "int8-static-f32", "int8-static-f16")


def synthetic_cases(directory, torch):
    directory.mkdir()
    rng = np.random.default_rng(712)
    m, n, k, rows = 48, 33, 96, 7
    w = rng.normal(0, 0.2, (m, k)).astype(np.float32)
    x = rng.normal(0, 2, (rows, k)).astype(np.float32)
    w[0] = 0
    x[0] = 0
    b = np.linspace(-1, 1, m, dtype=np.float32)
    scale = np.exp2(np.linspace(-2, 2, k)).astype(np.float32)
    cases = []

    def add(name, weights=w, values=x, bias=b, scales=scale, maximum=None, gelu=False,
            modes=MODES, rejection=None, tokens=n):
        with torch.inference_mode():
            reference = torch.from_numpy(values) @ torch.from_numpy(weights).T + torch.from_numpy(bias)
            if gelu:
                reference = torch.nn.functional.gelu(reference, approximate="none")
        if maximum is None:
            maximum = float(np.abs(values / scales).max())
        path = directory / (name + ".bin")
        write_case(path, weights, bias, values, reference.numpy(), scales, maximum, tokens, gelu)
        cases.append({"name": name, "path": path, "modes": modes, "rejection": rejection})

    add("tail-rows-and-zero-channels")
    add("tail-rows-erf", gelu=True)
    add("all-zero", weights=np.zeros_like(w), values=np.zeros_like(x), bias=np.zeros_like(b))
    ties = np.resize(np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 127, -127], np.float32), (rows, k))
    add("nearest-even", values=ties, scales=np.ones(k, np.float32), maximum=127)
    # Static calibration can underestimate production input by much more than
    # INT32's range. Positive saturation must retain its sign before conversion.
    huge = np.resize(np.array([1e25, -1e25, 1e20, -1e20], np.float32), (rows, k))
    add("static-saturation", values=huge, scales=np.ones(k, np.float32), maximum=0.01,
        modes=("int8-static-f32", "int8-static-f16"))
    large_w = np.ones((16, 16384), np.float32)
    large_x = np.ones((3, 16384), np.float32)
    add("max-k-int32-bound", weights=large_w, values=large_x, bias=np.zeros(16, np.float32),
        scales=np.ones(16384, np.float32), tokens=17,
        modes=("int8-token-f32", "int8-static-f32"))
    add("half-overflow", weights=np.ones_like(w), values=np.full_like(x, 1000),
        scales=np.ones(k, np.float32), modes=("int8-token-f16", "int8-static-f16"),
        rejection="non-finite output")
    add("activation-transform-overflow", weights=np.full_like(w, 1e-38), values=np.full_like(x, 1e38),
        scales=np.full(k, 0.01, np.float32), maximum=1,
        modes=("int8-token-f32", "int8-static-f32", "fp8-f32"), rejection="overflowed an activation")
    add("weight-transform-overflow", weights=np.full_like(w, 1e38), values=np.full_like(x, 1e-38),
        scales=np.full(k, 100, np.float32), maximum=1,
        modes=("int8-token-f32", "fp8-f32"), rejection="overflowed a weight")
    base = (directory / "tail-rows-and-zero-channels.bin").read_bytes()
    scale_offset = 32 + 4 * (m * k + m)
    for name, bad in (("zero-scale", 0), ("negative-scale", -1), ("nan-scale", float("nan"))):
        payload = bytearray(base)
        struct.pack_into("<f", payload, scale_offset, bad)
        path = directory / (name + ".bin")
        path.write_bytes(payload)
        cases.append({"name": name, "path": path, "modes": ("ggml-f16", "int8-token-f32"),
                      "rejection": "non-finite" if name == "nan-scale" else "non-positive"})
    for name, payload, message in (("truncated", base[:-4], "length differs"),
                                   ("trailing-data", base + b"extra", "length differs")):
        path = directory / (name + ".bin")
        path.write_bytes(payload)
        cases.append({"name": name, "path": path, "modes": ("ggml-f16", "int8-token-f32"), "rejection": message})
    return cases


def validate(args):
    import torch
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference environment")
    if args.output.exists():
        raise FileExistsError("validation output must be new")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[2]
    binaries = {name: (args.binaries / name).resolve() for name in ("sam_ggml_linear_probe", "sam_cuda_linear_probe")}
    tools_root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), tools_root / "benchmark/cuda_linear_probe.cu",
               tools_root / "benchmark/ggml_linear_probe.cpp", tools_root / "benchmark/linear_probe.hpp",
               tools_root / "validation/verify_linear_probe.py", tools_root / "benchmark/prepare_linear_probe.py"]
    identities = {str(p.resolve()): sha256_file(p) for p in (*sources, *binaries.values())}
    args.output.mkdir(parents=True)
    cases = synthetic_cases(args.output / "inputs", torch)
    if args.inputs:
        manifest_path = args.inputs / "manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("diagnostic_only") is not True or manifest.get("schema_version") != 1:
            raise ValueError("unsupported linear probe manifest")
        identities[str(manifest_path.resolve())] = sha256_file(manifest_path)
        for case in manifest["cases"]:
            filename = case["file"]
            if Path(filename).name != filename or sha256_file(args.inputs / filename) != case["sha256"]:
                raise ValueError("probe input identity differs")
            cases.append({"name": Path(filename).stem, "path": args.inputs / filename, "modes": MODES, "rejection": None})
    for case in cases:
        identities[str(case["path"].resolve())] = sha256_file(case["path"])
    results = []
    for case in cases:
        for mode in case["modes"]:
            name = case["name"] + "--" + mode
            output = args.output / name
            binary = binaries["sam_ggml_linear_probe" if mode.startswith("ggml-") else "sam_cuda_linear_probe"]
            command = ["rtk", "proxy", str(binary), str(case["path"].resolve()), mode, str(output.resolve()), "3"]
            log = args.output / (name + ".log")
            started = time.time()
            with log.open("x") as stream:
                status = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT).returncode
            row = {"case": case["name"], "mode": mode, "command": command, "returncode": status,
                   "started": started, "finished": time.time(), "expected_rejection": case["rejection"]}
            if case["rejection"]:
                row["passed"] = status != 0 and case["rejection"] in log.read_text() and not (output / "probe.json").exists()
            else:
                row["passed"] = status == 0
                if status == 0 and mode != "ggml-q8":
                    checked = verify(case["path"], output, args.threads)
                    with (output / "verification.json").open("x") as stream:
                        json.dump(checked, stream, indent=2, allow_nan=False)
                        stream.write("\n")
                    row["passed"] = checked["passed"]
                    row["decoded_reference_l2"] = checked["relative_l2"]
            results.append(row)
            print(name, "PASS" if row["passed"] else "FAIL", flush=True)
    unchanged = all(sha256_file(Path(p)) == h for p, h in identities.items())
    result = {"schema_version": 1, "complete": True, "passed": unchanged and all(row["passed"] for row in results),
              "diagnostic_only": True, "scope": "Kernel numerics and rejection boundaries; not SAM quality or formal performance",
              "sources_and_inputs_unchanged": unchanged, "artifact_sha256": identities, "steps": results}
    with (args.output / "validation.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="Optional directory prepared by prepare_linear_probe.py")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    try:
        if not validate(args):
            raise SystemExit(1)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"linear probe validation failed: {error}\n")


if __name__ == "__main__":
    main()
