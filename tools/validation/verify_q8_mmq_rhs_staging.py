#!/usr/bin/env python3
"""Independently verify pinned GGML CUDA Q8_0 MMQ right-operand staging."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BLOCK_VALUES = 128
BLOCK_BYTES = 144


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
        raise ValueError("Q8 MMQ source/payload shape differs")
    if not np.isfinite(values).all():
        raise ValueError("Q8 MMQ source contains non-finite values")
    values = values.reshape(-1, 4, 32)
    scale = payload[:, :16].copy().view("<f4").reshape(-1, 4)
    integers = payload[:, 16:].view(np.int8).reshape(-1, 4, 32)
    if (not np.isfinite(scale).all() or (scale < 0).any() or
            (integers == -128).any()):
        raise ValueError("Q8 MMQ payload has invalid scale or integer")

    maximum = np.max(np.abs(values), axis=2)
    exact_scale = maximum.astype(np.float64) / 127.0
    nearest_scale = exact_scale.astype("<f4")
    scale_ulp = np.spacing(nearest_scale).astype(np.float64)
    # GGML's CUDA fast divide/reciprocal stores F32 scales. Its bounded
    # roundoff is measured in F32 ulps, unlike MMVQ's F16 scale tie rule.
    scale_ok = np.abs(scale.astype(np.float64) - exact_scale) <= 4 * scale_ulp
    target = np.divide(values.astype(np.float64) * 127.0, maximum[:, :, None],
                       out=np.zeros(values.shape, dtype=np.float64),
                       where=maximum[:, :, None] != 0)
    magnitude = np.abs(target)
    whole = np.floor(magnitude)
    fraction = magnitude - whole
    nearest = np.copysign(np.floor(magnitude + 0.5), target).astype(np.int16)
    adjacent = np.copysign(np.where(fraction < 0.5, whole + 1, whole), target).astype(np.int16)
    difference = integers.astype(np.int16) - nearest
    target_ulp = np.abs(np.spacing(target.astype("<f4")).astype(np.float64))
    integer_ok = (difference == 0) | ((integers.astype(np.int16) == adjacent) &
                    (np.abs(fraction - 0.5) <= 4 * target_ulp) &
                    (np.abs(integers.astype(np.int16)) <= 127))
    zero = maximum == 0
    scale_ok[zero] &= scale[zero] == 0
    integer_ok[zero] = integers[zero] == 0
    return {"blocks": len(values),
            "subblocks": scale.size,
            "scale_ulp_allowances": int(np.count_nonzero((scale.view("<u4") !=
                                                           nearest_scale.view("<u4")) & scale_ok)),
            "integer_tie_allowances": int(np.count_nonzero((difference != 0) & integer_ok)),
            "scale_errors": int(np.count_nonzero(~scale_ok)),
            "integer_errors": int(np.count_nonzero(~integer_ok))}


def verify(input_path, payload_path, report_path, width, row_chunk=4096):
    paths = [Path(input_path), Path(payload_path), Path(report_path)]
    if width <= 0 or width % 32 or row_chunk <= 0:
        raise ValueError("Q8 MMQ requires a positive 32-aligned width and row chunk")
    source_bytes = paths[0].stat().st_size
    if source_bytes == 0 or source_bytes % (width * 4):
        raise ValueError("Q8 MMQ input length differs from whole rows")
    rows = source_bytes // (width * 4)
    padded_width = (width + 511) // 512 * 512
    blocks_per_row = padded_width // BLOCK_VALUES
    blocks = rows * blocks_per_row
    if paths[1].stat().st_size != blocks * BLOCK_BYTES:
        raise ValueError("Q8 MMQ payload length differs")
    report = json.loads(paths[2].read_text())
    if (report.get("complete") is not True or report.get("weight_type") != "q8_0" or
            report.get("staging_path") != "MMQ" or report.get("width") != width or
            report.get("padded_width") != padded_width or report.get("rows") != rows or
            report.get("blocks") != blocks or report.get("compute_capability", 0) < 89):
        raise ValueError("native report does not identify CUDA Q8 MMQ staging")
    before = {str(path): sha256_file(path) for path in paths}
    source = np.memmap(paths[0], dtype="<f4", mode="r").reshape(rows, width)
    payload = np.memmap(paths[1], dtype=np.uint8, mode="r").reshape(blocks_per_row, rows, BLOCK_BYTES)
    result = {key: 0 for key in ("blocks", "subblocks", "scale_ulp_allowances",
                                    "integer_tie_allowances", "scale_errors", "integer_errors")}
    for block in range(blocks_per_row):
        begin, end = block * BLOCK_VALUES, min((block + 1) * BLOCK_VALUES, width)
        for row_begin in range(0, rows, row_chunk):
            row_end = min(row_begin + row_chunk, rows)
            values = np.zeros((row_end - row_begin, BLOCK_VALUES), dtype="<f4")
            if begin < end:
                values[:, :end - begin] = source[row_begin:row_end, begin:end]
            part = check_blocks(values, payload[block, row_begin:row_end])
            for key, value in part.items():
                result[key] += value
    if before != {str(path): sha256_file(path) for path in paths}:
        raise ValueError("Q8 MMQ evidence changed during verification")
    result.update(rows=rows, width=width, padded_width=padded_width,
                  passed=not (result["scale_errors"] or result["integer_errors"]),
                  sha256=before, device=report["device"])
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
    result.update(schema_version=1, kind="sam3-independent-q8-mmq-rhs-staging",
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
