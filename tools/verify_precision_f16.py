#!/usr/bin/env python3
"""Check native F16 cache conversion against an independent integer IEEE-754 oracle."""

import argparse
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from precision_artifacts import archive_sources, runtime_environment, source_snapshot
from sam3_artifacts import sha256_file, verify_run_artifacts, write_json


def rounded_shift(value, shift):
    whole, remainder = divmod(value, 1 << shift)
    half = 1 << (shift - 1)
    return whole + int(remainder > half or remainder == half and whole & 1)


def f32_to_f16_bits(bits):
    """Round a finite binary32 word to binary16, nearest with ties to even."""
    sign, exponent, fraction = (bits >> 16) & 0x8000, (bits >> 23) & 255, bits & 0x7fffff
    if exponent == 255:
        raise ValueError("the native cache contract rejects non-finite inputs")
    power = exponent - 127
    if power < -25:
        return sign
    if power > 15:
        return sign | 0x7c00
    if power < -14:
        return sign | rounded_shift((1 << 23) | fraction, -power - 1)
    return sign | (((power + 15) << 10) + rounded_shift(fraction, 13))


def f16_to_f32_bits(bits):
    sign, exponent, fraction = (bits & 0x8000) << 16, (bits >> 10) & 31, bits & 1023
    if exponent == 31:
        raise ValueError("the native cache contract rejects non-finite payloads")
    if exponent:
        return sign | ((exponent + 112) << 23) | (fraction << 13)
    if not fraction:
        return sign
    highest = fraction.bit_length() - 1
    return sign | ((highest + 103) << 23) | ((fraction - (1 << highest)) << (23 - highest))


def fixture():
    finite_words = [word for word in range(65536) if word & 0x7c00 != 0x7c00]
    exact = np.asarray([f16_to_f32_bits(word) for word in finite_words], dtype="<u4").view("<f4")
    positive = np.asarray([f16_to_f32_bits(word) for word in range(0x7c00)], dtype="<u4").view("<f4")
    midpoints = ((positive[:-1].astype(np.float64) + positive[1:].astype(np.float64)) / 2).astype(np.float32)
    around = np.concatenate([np.nextafter(midpoints, np.float32(-np.inf)), midpoints,
                             np.nextafter(midpoints, np.float32(np.inf))])
    special = np.asarray([0.0, -0.0, np.nextafter(np.float32(0), np.float32(1)),
                          np.finfo(np.float32).tiny, np.nextafter(np.float32(65520), np.float32(0))], dtype=np.float32)
    values = np.concatenate([exact, around, -around, special, -special])
    # Include a final partial GPU block and explicit zero rows, preserving signs.
    values = np.pad(values, (0, (-len(values)) % 32 + 32)).astype("<f4")
    expected = np.asarray([f32_to_f16_bits(int(word)) for word in values.view("<u4")], dtype="<u2")
    decoded = np.asarray([f16_to_f32_bits(int(word)) for word in expected], dtype="<u4")
    assert np.array_equal(np.asarray([f32_to_f16_bits(int(word)) for word in exact.view("<u4")]), finite_words)
    known = [(1.0, 0x3c00), (1.00048828125, 0x3c00), (1.00146484375, 0x3c02),
             (2.0**-24, 1), (2.0**-25, 0), (65504.0, 0x7bff), (65520.0, 0x7c00), (-0.0, 0x8000)]
    for value, word in known:
        assert f32_to_f16_bits(int(np.asarray([value], dtype="<f4").view("<u4")[0])) == word
    return values, expected, decoded, len(finite_words), len(midpoints)


def verify(args):
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference environment")
    if args.output.exists() or len(set(args.backends)) != len(args.backends):
        raise ValueError("use a new output directory and unique backends")
    artifacts = source_snapshot()
    artifacts[str(args.probe.resolve())] = sha256_file(args.probe)
    linked = subprocess.run(["rtk", "proxy", "ldd", str(args.probe)], check=True, text=True, capture_output=True).stdout
    if "not found" in linked:
        raise ValueError("native codec probe has missing shared libraries")
    for line in linked.splitlines():
        match = re.search(r"(?:=>\s+)?(/\S+)\s+\(", line)
        if match:
            path = Path(match.group(1)).resolve()
            artifacts[str(path)] = sha256_file(path)
    environments = {backend: runtime_environment(backend) for backend in args.backends}
    values, expected, decoded, finite_count, midpoint_count = fixture()
    args.output.mkdir(parents=True)
    input_path = args.output / "finite-boundaries.f32"
    values.tofile(input_path)
    artifacts[str(input_path.resolve())] = sha256_file(input_path)
    steps = []
    for backend in args.backends:
        destination = args.output / ("finite-" + backend)
        command = ["rtk", "proxy", str(args.probe.resolve()), str(input_path.resolve()), "f16", backend,
                   str(len(values) // 32), "32", str(destination.resolve()), "1"]
        with (args.output / ("finite-" + backend + ".log")).open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        row = {"backend": backend, "case": "finite-representable-midpoint-neighbors", "returncode": result.returncode,
               "values": len(values), "passed": False}
        if result.returncode == 0:
            actual = np.fromfile(destination / "packed.bin", dtype="<u2")
            restored = np.fromfile(destination / "decoded.f32", dtype="<u4")
            row.update(packed_shape_equal=actual.shape == expected.shape, decoded_shape_equal=restored.shape == decoded.shape)
            if row["packed_shape_equal"] and row["decoded_shape_equal"]:
                row.update(packed_bit_errors=int(np.count_nonzero(actual != expected)),
                           decoded_bit_errors=int(np.count_nonzero(restored != decoded)))
                row["passed"] = row["packed_bit_errors"] == 0 and row["decoded_bit_errors"] == 0
        steps.append(row)
        for name, value, error in (("positive-overflow", 65520.0, "non-finite output"),
                                   ("negative-overflow", -65520.0, "non-finite output"),
                                   ("positive-inf", np.inf, "requires finite F32 input"),
                                   ("negative-inf", -np.inf, "requires finite F32 input"),
                                   ("nan", np.nan, "requires finite F32 input")):
            source = args.output / (name + "-" + backend + ".f32")
            np.full(32, value, dtype="<f4").tofile(source)
            output = args.output / (name + "-" + backend)
            command = ["rtk", "proxy", str(args.probe.resolve()), str(source.resolve()), "f16", backend,
                       "1", "32", str(output.resolve()), "1"]
            result = subprocess.run(command, text=True, capture_output=True)
            log = result.stdout + result.stderr
            (args.output / (name + "-" + backend + ".log")).write_text(log)
            steps.append({"backend": backend, "case": name, "returncode": result.returncode,
                          "expected_rejection": error, "passed": result.returncode != 0 and error in log})
    verify_run_artifacts(artifacts)
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            artifacts[str(path.resolve())] = sha256_file(path)
    archived = archive_sources(args.output, artifacts)
    passed = all(row["passed"] for row in steps)
    write_json(args.output / "arithmetic.json", {"schema_version": 2, "kind": "sam3-f16-codec-arithmetic-v2",
                "complete": True, "passed": passed, "arithmetic_status": "PASS" if passed else "FAIL",
                "scope": "Native F32/F16 cache conversion only; no complete model or W8A8/FP8 qualification",
                "oracle": "Independent integer bit shifts and round-to-nearest ties-to-even, preserving signed zero and F16 subnormals",
                "finite_f16_patterns": finite_count, "positive_rounding_midpoints": midpoint_count,
                "environments": environments, "steps": steps, "artifact_sha256": artifacts, "archived_sources": archived})
    print({"passed": passed, "finite_f16_patterns": finite_count, "steps": len(steps)}, flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    args = parser.parse_args()
    try:
        passed = verify(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"F16 arithmetic validation failed: {error}\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
