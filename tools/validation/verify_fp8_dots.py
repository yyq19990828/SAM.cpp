"""Independently check sampled CUDA E4M3 probe dots against saved packed bytes."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_e4m3(data):
    bits = np.asarray(data, dtype=np.uint8)
    exponent = (bits >> 3) & 15
    fraction = bits & 7
    if np.any((exponent == 15) & (fraction == 7)):
        raise ValueError("E4M3 operand contains NaN")
    magnitude = np.where(exponent == 0, fraction.astype(np.float64) * 2.0**-9,
                         (1.0 + fraction / 8.0) * np.exp2(exponent.astype(np.int32) - 7))
    return np.where(bits & 128, -magnitude, magnitude)


def verify(input_path, directory):
    with input_path.open("rb") as stream:
        header = stream.read(32)
    if len(header) != 32:
        raise ValueError("truncated linear probe header")
    magic, m, n, k, _, _, _ = struct.unpack("<8s5If", header)
    if magic != b"SLPROB01" or not (0 < m <= 16384 and 0 < n <= 16384 and 0 < k <= 16384):
        raise ValueError("invalid linear probe header")
    failure_path = directory / "kernel-check-failure.json"
    failure = json.loads(failure_path.read_text())
    mode = failure["mode"]
    if mode not in ("fp8-f32", "fp8-f16"):
        raise ValueError("expected failed FP8 probe")
    half = mode.endswith("f16")
    raw_path = directory / ("raw-f16.bin" if half else "raw-f32.bin")
    paths = [input_path, failure_path, directory / "weight.e4m3", directory / "activation.e4m3",
             raw_path, directory / "weight-scale.f32", directory / "activation-scale.f32"]
    before = {str(path): sha256_file(path) for path in paths}
    weight = np.fromfile(paths[2], dtype=np.uint8)
    activation = np.fromfile(paths[3], dtype=np.uint8)
    raw = np.memmap(raw_path, dtype="<f2" if half else "<f4", mode="r")
    if weight.size != m * k or activation.size != n * k or raw.size != m * n:
        raise ValueError("FP8 dump length differs from input shape")
    weight = weight.reshape(m, k)
    activation = activation.reshape(n, k)
    scales = [np.fromfile(path, dtype="<f4") for path in paths[-2:]]
    if any(scale.size != 1 or not np.isfinite(scale[0]) or scale[0] <= 0 for scale in scales):
        raise ValueError("invalid FP8 scale")
    if not np.isclose(scales[0][0], failure["weight_scale"], rtol=1e-9) or not np.isclose(
            scales[1][0], failure["input_scale"], rtol=1e-9):
        raise ValueError("FP8 scale differs from failure record")
    scale = float(scales[0][0]) * float(scales[1][0])
    indices = [m * n - 1 if check == 1 else (check * 104729) % (m * n)
               for check in range(min(m * n, 512))]
    errors = []
    max_error = 0.0
    first_failure = None
    reported = int(failure["index"])
    reported_dot = None
    for index in indices:
        row, token = index % m, index // m
        products = decode_e4m3(weight[row]) * decode_e4m3(activation[token])
        dot = float(np.sum(products, dtype=np.float64))
        expected = float(np.float32(dot * scale))
        if half:
            expected = float(np.float16(expected))
        actual = float(raw[index])
        tolerance = (0.0015 if half else 2e-5) * (1.0 + abs(expected)) + 5e-7 * float(
            np.sum(np.abs(products), dtype=np.float64)) * scale
        error = abs(actual - expected)
        if not np.isfinite(actual) or error > tolerance:
            errors.append(index)
            if first_failure is None:
                first_failure = index
        max_error = max(max_error, error)
        if index == reported:
            reported_dot = dot
            if abs(actual - failure["actual"]) > 1e-6 or abs(expected - failure["expected"]) > 1e-6:
                raise ValueError("reported failed dot differs from saved FP8 dump")
    if first_failure != reported or reported_dot is None:
        raise ValueError("reported first failure differs from independent order")
    if before != {str(path): sha256_file(path) for path in paths}:
        raise ValueError("FP8 evidence changed during verification")
    return {"schema_version": 1, "passed": False, "independent_reference_reproduced": True,
            "mode": mode, "m": m, "n": n, "k": k, "algorithm_id": failure["algorithm_id"],
            "algorithm_index": failure["algorithm_index"], "heuristic_count": failure["heuristic_count"],
            "checked_dots": len(indices), "failed_dots": len(errors), "first_failure": first_failure,
            "first_failure_decoded_dot": reported_dot, "max_abs_difference": max_error,
            "artifact_sha256": before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.input, args.directory)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
