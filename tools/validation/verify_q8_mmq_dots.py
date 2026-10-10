#!/usr/bin/env python3
"""Check CUDA Q8_0 MMQ raw dots against packed Q8_0/Q8_1 operands."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.validation.verify_q8_mmvq_dots import input_shape
from tools.validation.verify_q8_mmq_rhs_staging import verify as verify_staging


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(input_path, directory, row_chunk=64, token_chunk=8):
    input_path, directory = Path(input_path), Path(directory)
    m, n, k, sample_rows = input_shape(input_path)
    if row_chunk <= 0 or token_chunk <= 0:
        raise ValueError("dot chunks must be positive")
    staging_dir = directory / "staging"
    weight_path, rhs_path, raw_path = (directory / name for name in
                                       ("weight.q8_0", "rhs.f32", "raw-dot.f32"))
    report_path, staging_path, staging_report_path = (directory / "probe.json",
                                                      staging_dir / "rhs.q8_1_mmq",
                                                      staging_dir / "probe.json")
    paths = [input_path, weight_path, rhs_path, raw_path, report_path,
             staging_path, staging_report_path]
    padded_k = (k + 511) // 512 * 512
    if (weight_path.stat().st_size != m * k // 32 * 34 or rhs_path.stat().st_size != n * k * 4 or
            raw_path.stat().st_size != n * m * 4 or
            staging_path.stat().st_size != n * padded_k // 128 * 144):
        raise ValueError("Q8 MMQ dot payload length differs")
    report = json.loads(report_path.read_text())
    if (report.get("complete") is not True or report.get("raw_q8_operands_saved") is not True or
            report.get("mode") != "ggml-q8" or report.get("output_type") != "f32" or
            report.get("m") != m or report.get("n") != n or report.get("k") != k or
            report.get("raw_cuda_nodes", 0) <= 0 or report.get("cpu_nodes") != 0):
        raise ValueError("native report does not identify CUDA Q8 raw dots")
    before = {str(path): sha256_file(path) for path in paths}
    source = np.memmap(input_path, dtype="<f4", mode="r", offset=32)
    samples_begin = m * k + m + k
    samples = source[samples_begin:samples_begin + sample_rows * k].reshape(sample_rows, k)
    rhs = np.memmap(rhs_path, dtype="<f4", mode="r").reshape(n, k)
    if not np.array_equal(rhs.view("<u4"), samples[np.arange(n) % sample_rows].view("<u4")):
        raise ValueError("saved Q8 right operand differs from source input")
    staging = verify_staging(rhs_path, staging_path, staging_report_path, k)
    if not staging["passed"]:
        raise ValueError("Q8 MMQ right-hand staging failed its independent oracle")

    blocks = k // 32
    weight_bytes = np.memmap(weight_path, dtype=np.uint8, mode="r").reshape(m, blocks, 34)
    right_bytes = np.memmap(staging_path, dtype=np.uint8, mode="r").reshape(padded_k // 128, n, 144)
    weight_scale = weight_bytes[:, :, :2].copy().view("<f2").reshape(m, blocks).astype(np.float64)
    weight_int = weight_bytes[:, :, 2:].view(np.int8)
    right_scale = right_bytes[:, :, :16].copy().view("<f4").reshape(-1, n, 4)
    right_scale = right_scale.transpose(1, 0, 2).reshape(n, padded_k // 32)[:, :blocks].astype(np.float64)
    right_int = right_bytes[:, :, 16:].view(np.int8).reshape(-1, n, 4, 32)
    right_int = right_int.transpose(1, 0, 2, 3).reshape(n, padded_k // 32, 32)[:, :blocks]
    if (not np.isfinite(weight_scale).all() or (weight_scale < 0).any() or
            (weight_int == -128).any() or np.any((weight_scale == 0) & np.any(weight_int != 0, axis=2))):
        raise ValueError("invalid Q8_0 weight block")
    actual = np.memmap(raw_path, dtype="<f4", mode="r").reshape(n, m)
    if not np.isfinite(actual).all():
        raise ValueError("non-finite native Q8 MMQ dot")
    error_sq = reference_sq = 0.0
    row_error_sq = np.zeros(m, dtype=np.float64)
    row_reference_sq = np.zeros(m, dtype=np.float64)
    maximum_error = 0.0
    worst = (0, 0)
    worst_reference = worst_actual = 0.0
    for token_begin in range(0, n, token_chunk):
        token_end = min(token_begin + token_chunk, n)
        for row_begin in range(0, m, row_chunk):
            row_end = min(row_begin + row_chunk, m)
            products = np.einsum("mbk,nbk->nmb", weight_int[row_begin:row_end].astype(np.int64),
                                 right_int[token_begin:token_end].astype(np.int64), optimize=True)
            reference = np.sum(products * weight_scale[None, row_begin:row_end, :] *
                               right_scale[token_begin:token_end, None, :], axis=2)
            observed = actual[token_begin:token_end, row_begin:row_end].astype(np.float64)
            error = observed - reference
            error_sq += float(np.sum(error * error))
            reference_sq += float(np.sum(reference * reference))
            row_error_sq[row_begin:row_end] += np.sum(error * error, axis=0)
            row_reference_sq[row_begin:row_end] += np.sum(reference * reference, axis=0)
            position = np.unravel_index(np.argmax(np.abs(error)), error.shape)
            if abs(error[position]) > maximum_error:
                maximum_error = float(abs(error[position]))
                worst = (int(token_begin + position[0]), int(row_begin + position[1]))
                worst_reference = float(reference[position])
                worst_actual = float(observed[position])
    relative_l2 = float(np.sqrt(error_sq / reference_sq)) if reference_sq > 1e-30 else None
    row_relative = np.sqrt(row_error_sq / np.maximum(row_reference_sq, 1e-30))
    passed = relative_l2 <= 2e-5 if relative_l2 is not None else maximum_error <= 1e-6
    if before != {str(path): sha256_file(path) for path in paths}:
        raise ValueError("Q8 MMQ operands or native output changed during verification")
    return {"schema_version": 1, "kind": "sam3-q8-mmq-same-operand-dots", "passed": bool(passed),
            "m": m, "n": n, "k": k, "checked_dots": m * n, "relative_l2": relative_l2,
            "maximum_absolute_error": maximum_error, "maximum_row_relative_l2": float(np.max(row_relative)),
            "worst_token": worst[0], "worst_row": worst[1], "worst_reference": worst_reference,
            "worst_actual": worst_actual, "staging_blocks": staging["blocks"],
            "staging_scale_ulp_allowances": staging["scale_ulp_allowances"],
            "staging_integer_tie_allowances": staging["integer_tie_allowances"],
            "artifact_sha256": before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--identity", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = verify(args.input, args.directory)
    result["verifier_sha256"] = sha256_file(__file__)
    result["identity_sha256"] = {str(path): sha256_file(path) for path in args.identity}
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(payload)
    print(payload, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
