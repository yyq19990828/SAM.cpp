#!/usr/bin/env python3
"""Independently verify GGML CUDA MMVQ Q8_1 right-operand staging bytes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BLOCK_VALUES = 32
BLOCK_BYTES = 36


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_blocks(values, payload):
    values = np.asarray(values, dtype="<f4")
    payload = np.asarray(payload, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != BLOCK_VALUES or payload.shape != (len(values), BLOCK_BYTES):
        raise ValueError("Q8_1 source/payload shape differs")
    if not np.isfinite(values).all():
        raise ValueError("Q8_1 source contains non-finite values")
    scale = payload[:, :2].copy().view("<f2").reshape(-1)
    original_sum = payload[:, 2:4].copy().view("<f2").reshape(-1)
    integers = payload[:, 4:].view(np.int8)
    if (not np.isfinite(scale).all() or not np.isfinite(original_sum).all() or
            (scale < 0).any() or (integers == -128).any()):
        raise ValueError("Q8_1 payload has invalid scale, sum or integer")

    maximum = np.max(np.abs(values), axis=1)
    exact_scale = maximum.astype(np.float64) / 127.0
    nearest_scale = exact_scale.astype("<f2")
    if not np.isfinite(nearest_scale).all():
        raise ValueError("Q8_1 source scale overflows F16")
    different_scale = scale.view("<u2") != nearest_scale.view("<u2")
    lower = np.minimum(scale.astype(np.float64), nearest_scale.astype(np.float64))
    upper = np.maximum(scale.astype(np.float64), nearest_scale.astype(np.float64))
    adjacent = np.nextafter(lower.astype("<f2"), np.float16(np.inf)).astype(np.float64) == upper
    midpoint = (lower + upper) / 2.0
    scale_ulp = np.abs(np.spacing(exact_scale.astype("<f4")).astype(np.float64))
    scale_ok = ~different_scale | (adjacent & (np.abs(exact_scale - midpoint) <= 4 * scale_ulp))

    target = np.divide(values.astype(np.float64) * 127.0, maximum[:, None],
                       out=np.zeros(values.shape, dtype=np.float64), where=maximum[:, None] != 0)
    magnitude = np.abs(target)
    whole = np.floor(magnitude)
    fraction = magnitude - whole
    nearest_integer = np.copysign(np.floor(magnitude + 0.5), target).astype(np.int16)
    alternate_integer = np.copysign(np.where(fraction < 0.5, whole + 1, whole), target).astype(np.int16)
    delta = integers.astype(np.int16) - nearest_integer
    target_ulp = np.abs(np.spacing(target.astype("<f4")).astype(np.float64))
    integer_ok = (delta == 0) | ((integers.astype(np.int16) == alternate_integer) &
                                 (np.abs(fraction - 0.5) <= 4 * target_ulp) &
                                 (np.abs(integers.astype(np.int16)) <= 127))

    # CUDA's 32-lane xor-shuffle reduction stores the sum of the original F32
    # values as half. It does not recompute a sum from the quantized integers.
    reduction = values.copy()
    lanes = np.arange(BLOCK_VALUES)
    for offset in (16, 8, 4, 2, 1):
        reduction = (reduction + reduction[:, lanes ^ offset]).astype("<f4")
    expected_sum = reduction[:, 0].astype("<f2")
    if not np.isfinite(expected_sum).all():
        raise ValueError("Q8_1 source sum overflows F16")
    sum_ok = original_sum.view("<u2") == expected_sum.view("<u2")

    zero = maximum == 0
    scale_ok[zero] &= scale[zero] == 0
    integer_ok[zero] = integers[zero] == 0
    return {"blocks": len(values),
            "scale_tie_allowances": int(np.count_nonzero(different_scale & scale_ok)),
            "integer_tie_allowances": int(np.count_nonzero((delta != 0) & integer_ok)),
            "scale_errors": int(np.count_nonzero(~scale_ok)),
            "integer_errors": int(np.count_nonzero(~integer_ok)),
            "sum_errors": int(np.count_nonzero(~sum_ok))}


def verify(input_path, payload_path, report_path, width, chunk_blocks=32768):
    paths = [Path(input_path), Path(payload_path), Path(report_path)]
    if width <= 0 or width % BLOCK_VALUES or chunk_blocks <= 0:
        raise ValueError("Q8_1 requires a positive 32-aligned width and chunk")
    source_bytes = paths[0].stat().st_size
    if source_bytes == 0 or source_bytes % (width * 4):
        raise ValueError("Q8_1 input length differs from whole rows")
    rows = source_bytes // (width * 4)
    blocks = source_bytes // (BLOCK_VALUES * 4)
    if paths[1].stat().st_size != blocks * BLOCK_BYTES:
        raise ValueError("Q8_1 payload length differs")
    report = json.loads(paths[2].read_text())
    if (report.get("complete") is not True or report.get("weight_type") != "q8_0" or
            report.get("staging_path") != "MMVQ" or report.get("width") != width or
            report.get("rows") != rows or report.get("blocks") != blocks or
            report.get("compute_capability", 0) < 89):
        raise ValueError("native report does not identify CUDA Q8_1 MMVQ staging")
    before = {str(path): sha256_file(path) for path in paths}
    source = np.memmap(paths[0], dtype="<f4", mode="r").reshape(blocks, BLOCK_VALUES)
    payload = np.memmap(paths[1], dtype=np.uint8, mode="r").reshape(blocks, BLOCK_BYTES)
    result = {key: 0 for key in ("blocks", "scale_tie_allowances", "integer_tie_allowances",
                                    "scale_errors", "integer_errors", "sum_errors")}
    for begin in range(0, blocks, chunk_blocks):
        end = min(begin + chunk_blocks, blocks)
        part = check_blocks(source[begin:end], payload[begin:end])
        for key, value in part.items():
            result[key] += value
    if before != {str(path): sha256_file(path) for path in paths}:
        raise ValueError("Q8_1 evidence changed during verification")
    result.update(rows=rows, width=width, passed=not any(result[key] for key in
                  ("scale_errors", "integer_errors", "sum_errors")), sha256=before,
                  device=report["device"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("payload", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--identity", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.input, args.payload, args.report, args.width)
    result.update(schema_version=1, kind="sam3-independent-q8-rhs-staging",
                  verifier_sha256=sha256_file(__file__),
                  identity_sha256={str(path): sha256_file(path) for path in args.identity})
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(payload)
    print(payload, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
