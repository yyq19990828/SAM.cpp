#!/usr/bin/env python3
"""Verify every CUDA W8A8 raw INT32 dot with signed INT64 products."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.sam3_artifacts import sha256_file
from tools.validation.verify_linear_probe import read_input


def verify(input_path, run_directory, chunk_tokens=16, identities=()):
    input_path, run_directory = Path(input_path), Path(run_directory)
    if chunk_tokens <= 0:
        raise ValueError("INT64 verifier chunk size must be positive")
    report_path = run_directory / "probe.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    data, _ = read_input(input_path)
    if (report.get("complete") is not True or report.get("raw_int32_saved") is not True or
            report.get("mode") not in ("int8-token-f32", "int8-token-f16", "int8-static-f32", "int8-static-f16") or
            report.get("raw_int32_layout") != "weight[M,K], activation[N,K], dot[N,M], row-major" or
            any(report.get(key) != data[key] for key in ("m", "n", "k")) or
            report.get("gelu_erf") != data["gelu"] or report.get("compute_capability", 0) < 75):
        raise ValueError("incomplete or mismatched CUDA INT8 probe")
    m, n, k = (data[key] for key in ("m", "n", "k"))
    if k * 127 * 127 > np.iinfo(np.int32).max:
        raise ValueError("INT8 dot can overflow INT32")
    files = {"weights": ("weight.i8", m * k, np.int8, (m, k)),
             "activations": ("activation.i8", n * k, np.int8, (n, k)),
             "dots": ("raw-int32.bin", n * m * 4, "<i4", (n, m)),
             "weight_scales": ("weight-scale.f32", m * 4, "<f4", (m,)),
             "activation_scales": ("activation-scale.f32", n * 4, "<f4", (n,))}
    paths = {name: run_directory / filename for name, (filename, _, _, _) in files.items()}
    for name, (_, size, _, _) in files.items():
        if paths[name].stat().st_size != size:
            raise ValueError(f"{name} byte length differs from the declared shape")
    protected = [input_path, report_path, *paths.values(), Path(__file__), *map(Path, identities)]
    before = {str(path): sha256_file(path) for path in protected}
    arrays = {name: np.memmap(paths[name], dtype=dtype, mode="r", shape=shape)
              for name, (_, _, dtype, shape) in files.items()}
    if any((arrays[name] == -128).any() for name in ("weights", "activations")):
        raise ValueError("packed INT8 operand contains -128")
    if any(not np.isfinite(arrays[name]).all() or (arrays[name] <= 0).any()
           for name in ("weight_scales", "activation_scales")):
        raise ValueError("packed INT8 scale is non-finite or non-positive")
    weights = arrays["weights"].astype(np.int64).T
    mismatches = 0
    maximum_error = 0
    worst = None
    for begin in range(0, n, chunk_tokens):
        end = min(begin + chunk_tokens, n)
        expected = arrays["activations"][begin:end].astype(np.int64) @ weights
        actual = arrays["dots"][begin:end].astype(np.int64)
        difference = actual - expected
        failures = difference != 0
        mismatches += int(np.count_nonzero(failures))
        if np.any(failures):
            error = np.abs(difference)
            local = np.unravel_index(int(np.argmax(error)), error.shape)
            if int(error[local]) > maximum_error:
                maximum_error = int(error[local])
                worst = {"token": begin + int(local[0]), "output_row": int(local[1]),
                         "actual": int(actual[local]), "expected": int(expected[local])}
    if any(sha256_file(path) != digest for path, digest in before.items()):
        raise ValueError("INT8 probe or verifier input changed during verification")
    return {"schema_version": 1, "kind": "sam3-w8a8-exact-dot", "complete": True,
            "passed": mismatches == 0, "mode": report["mode"], "m": m, "n": n, "k": k,
            "compared_dots": m * n, "mismatches": mismatches,
            "maximum_integer_error": maximum_error, "worst": worst,
            "scope": "All exported pre-epilogue INT32 dots versus signed INT64 products of the exact packed operands; no model-quality claim",
            "artifact_sha256": before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--identity", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        result = verify(args.input, args.run, args.chunk_tokens, args.identity)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False, sort_keys=True)
            stream.write("\n")
        print(f"{'PASS' if result['passed'] else 'FAIL'}: {result['compared_dots']} exact dots, {result['mismatches']} mismatches")
        if not result["passed"]:
            raise SystemExit(1)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"INT8 dot verification failed: {error}\n")


if __name__ == "__main__":
    main()
