#!/usr/bin/env python3
"""Independently check GGML Q8_0 cache blocks against their F32 inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BLOCK_VALUES = 32
BLOCK_BYTES = 34


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_blocks(values, payload, decoded=None):
    """Return error counts for independent scale, integer and decode checks.

    CUDA fast division and reciprocal may cross a nearest-half or integer tie.
    Only adjacent results within four binary32 ulps of that tie are admitted.
    This tolerance applies to the arithmetic decision, not to stored bytes or
    decoded F32 values. Ordinary non-boundary values must match exactly.
    """
    values = np.asarray(values, dtype="<f4")
    payload = np.asarray(payload, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != BLOCK_VALUES or payload.shape != (len(values), BLOCK_BYTES):
        raise ValueError("Q8_0 values/payload shape differs")
    if decoded is not None and np.asarray(decoded).shape != values.shape:
        raise ValueError("Q8_0 decoded shape differs")
    if not np.isfinite(values).all():
        raise ValueError("Q8_0 source contains non-finite values")

    scale = payload[:, :2].copy().view("<f2").reshape(-1)
    integers = payload[:, 2:].view(np.int8)
    if not np.isfinite(scale).all() or (scale < 0).any() or (integers == -128).any():
        raise ValueError("Q8_0 payload has an invalid scale or integer")
    maximum = np.max(np.abs(values), axis=1)
    exact_scale = maximum.astype(np.float64) / 127.0
    nearest_scale = exact_scale.astype("<f2")
    if not np.isfinite(nearest_scale).all():
        raise ValueError("Q8_0 scale overflows F16")
    different_scale = scale.view("<u2") != nearest_scale.view("<u2")
    # A differing scale must be the adjacent binary16 value across the exact
    # rounding midpoint, and that midpoint must lie within CUDA F32 error.
    lower = np.minimum(scale.astype(np.float64), nearest_scale.astype(np.float64))
    upper = np.maximum(scale.astype(np.float64), nearest_scale.astype(np.float64))
    adjacent = np.nextafter(lower.astype("<f2"), np.float16(np.inf)).astype(np.float64) == upper
    midpoint = (lower + upper) / 2.0
    scale_ulp = np.spacing(exact_scale.astype("<f4")).astype(np.float64)
    allowed_scale = ~different_scale | (adjacent & (np.abs(exact_scale - midpoint) <= 4 * scale_ulp))

    target = np.divide(values.astype(np.float64) * 127.0, maximum[:, None],
                       out=np.zeros(values.shape, dtype=np.float64), where=maximum[:, None] != 0)
    magnitude = np.abs(target)
    whole = np.floor(magnitude)
    fraction = magnitude - whole
    nearest_integer = np.copysign(np.floor(magnitude + 0.5), target).astype(np.int16)
    alternate_integer = np.copysign(np.where(fraction < 0.5, whole + 1, whole), target).astype(np.int16)
    delta = integers.astype(np.int16) - nearest_integer
    half_distance = np.abs(fraction - 0.5)
    target_ulp = np.spacing(target.astype("<f4")).astype(np.float64)
    allowed_integer = (delta == 0) | ((integers.astype(np.int16) == alternate_integer) &
                        (half_distance <= 4 * np.abs(target_ulp)) &
                        (np.abs(integers.astype(np.int16)) <= 127))
    # Zero blocks are exact, including their signed source zeros.
    zero = maximum == 0
    allowed_integer[zero] = integers[zero] == 0
    allowed_scale[zero] &= scale[zero] == 0

    decoded_errors = 0
    if decoded is not None:
        expected = integers.astype("<f4") * scale.astype("<f4")[:, None]
        decoded_errors = int(np.count_nonzero(expected.view("<u4") != np.asarray(decoded, dtype="<f4").view("<u4")))
    return {
        "blocks": len(values),
        "scale_tie_allowances": int(np.count_nonzero(different_scale & allowed_scale)),
        "integer_tie_allowances": int(np.count_nonzero((delta != 0) & allowed_integer)),
        "scale_errors": int(np.count_nonzero(~allowed_scale)),
        "integer_errors": int(np.count_nonzero(~allowed_integer)),
        "decoded_bit_errors": decoded_errors,
    }


def verify(input_path, payload_path, decoded_path, channels, chunk_blocks=65536):
    paths = [Path(input_path), Path(payload_path)]
    if decoded_path is not None:
        paths.append(Path(decoded_path))
    if channels <= 0 or channels % BLOCK_VALUES or chunk_blocks <= 0:
        raise ValueError("Q8_0 requires positive block-aligned channels and chunk size")
    source_size = paths[0].stat().st_size
    if source_size == 0 or source_size % (channels * 4):
        raise ValueError("Q8_0 input length is not a whole number of rows")
    blocks = source_size // (BLOCK_VALUES * 4)
    if paths[1].stat().st_size != blocks * BLOCK_BYTES:
        raise ValueError("Q8_0 payload length differs")
    if decoded_path is not None and paths[2].stat().st_size != source_size:
        raise ValueError("Q8_0 decoded length differs")
    source = np.memmap(paths[0], dtype="<f4", mode="r").reshape(blocks, BLOCK_VALUES)
    payload = np.memmap(paths[1], dtype=np.uint8, mode="r").reshape(blocks, BLOCK_BYTES)
    decoded = None if decoded_path is None else np.memmap(paths[2], dtype="<f4", mode="r").reshape(blocks, BLOCK_VALUES)
    result = {key: 0 for key in ("blocks", "scale_tie_allowances", "integer_tie_allowances",
                                    "scale_errors", "integer_errors", "decoded_bit_errors")}
    for begin in range(0, blocks, chunk_blocks):
        end = min(begin + chunk_blocks, blocks)
        part = check_blocks(source[begin:end], payload[begin:end], None if decoded is None else decoded[begin:end])
        for key, value in part.items():
            result[key] += value
    result.update(rows=source_size // (channels * 4), channels=channels,
                  passed=not any(result[key] for key in ("scale_errors", "integer_errors", "decoded_bit_errors")),
                  sha256={str(path): sha256_file(path) for path in paths})
    return result


def check_native_report(path, rows, channels):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (report.get("complete") is not True or report.get("mode") != "q8_0" or
            report.get("backend") != "cuda" or report.get("rows") != rows or
            report.get("channels") != channels or report.get("cuda_nodes", 0) <= 0 or
            report.get("cpu_nodes") != 0):
        raise ValueError("native probe did not execute the expected CUDA Q8_0 path")
    return {"sha256": sha256_file(path), "device": report["device"],
            "cuda_nodes": report["cuda_nodes"], "cpu_nodes": report["cpu_nodes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="native F32 rows")
    parser.add_argument("payload", type=Path, help="native GGML Q8_0 block bytes")
    parser.add_argument("--decoded", type=Path, help="native decoded F32 rows")
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--native-report", type=Path, help="matching native CUDA probe.json")
    parser.add_argument("--identity", type=Path, action="append", default=[], help="source, binary or linked library to bind")
    parser.add_argument("--output", type=Path, help="new JSON receipt path")
    args = parser.parse_args()
    result = verify(args.input, args.payload, args.decoded, args.channels)
    if args.native_report:
        result["native_probe"] = check_native_report(args.native_report, result["rows"], args.channels)
    result["identity_sha256"] = {str(path): sha256_file(path) for path in args.identity}
    result.update(schema_version=1, kind="sam3-independent-q8-cache-arithmetic",
                  verifier_sha256=sha256_file(Path(__file__)))
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
