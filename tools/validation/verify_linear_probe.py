#!/usr/bin/env python3
"""Check CUDA probe outputs against independently decoded F32 operand products."""

import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.sam3_artifacts import read_json, sha256_file


def read_input(path):
    with path.open("rb") as stream:
        header = stream.read(32)
        if len(header) != 32:
            raise ValueError("truncated probe header")
        magic, m, n, k, rows, gelu, maximum = struct.unpack("<8sIIIIIf", header)
        if (magic != b"SLPROB01" or not 0 < rows <= n <= 16384 or not 0 < m <= 16384
                or not 0 < k <= 16384 or gelu not in (0, 1) or not np.isfinite(maximum) or maximum < 0):
            raise ValueError("invalid probe header")
        size = m * k + m + k + rows * (k + m)
        if size > 2**28 or path.stat().st_size != 32 + size * 4:
            raise ValueError("invalid probe length")
        data = np.fromfile(stream, dtype="<f4")
    if not np.isfinite(data).all():
        raise ValueError("non-finite probe data")
    offset = 0
    arrays = []
    for shape in ((m, k), (m,), (k,), (rows, k), (rows, m)):
        count = int(np.prod(shape))
        arrays.append(data[offset:offset + count].reshape(shape))
        offset += count
    if (arrays[2] <= 0).any():
        raise ValueError("non-positive channel scale")
    return {"m": m, "n": n, "k": k, "rows": rows, "gelu": bool(gelu), "maximum": np.float32(maximum)}, arrays


def int8_rows(values, scale=None):
    if scale is None:
        maximum = np.abs(values).max(axis=1, keepdims=True)
        scale = np.maximum(maximum / np.float32(127), np.finfo(np.float32).tiny)
        scale[maximum == 0] = 1
    # Match the candidate's declared F32 division before nearest-even rounding.
    encoded = np.clip(np.rint(values / scale), -127, 127).astype(np.float32)
    return encoded, scale


def decoded_reference(data, arrays, mode, torch):
    w, bias, scale, x, _ = arrays
    if mode == "ggml-f16":
        result = torch.from_numpy(x.astype(np.float16).astype(np.float32)) @ torch.from_numpy(w.astype(np.float16).astype(np.float32)).T
    else:
        ws, xs = w * scale, x / scale
        if mode.startswith("int8-"):
            qw, sw = int8_rows(ws)
            static = None
            if mode.startswith("int8-static-"):
                static = np.float32(1) if data["maximum"] == 0 else np.maximum(data["maximum"] / np.float32(127), np.finfo(np.float32).tiny)
            qa, sa = int8_rows(xs, static)
            result = torch.from_numpy(qa) @ torch.from_numpy(qw).T
            result = (result * torch.from_numpy(sw[:, 0])) * torch.as_tensor(sa)
        elif mode.startswith("fp8-"):
            w_max = np.abs(ws).max()
            sw = np.float32(1) if w_max == 0 else np.maximum(w_max / np.float32(448), np.finfo(np.float32).tiny)
            sa = np.float32(1) if data["maximum"] == 0 else np.maximum(data["maximum"] / np.float32(448), np.finfo(np.float32).tiny)
            qw = torch.from_numpy(np.clip(ws / sw, -448, 448)).to(torch.float8_e4m3fn).float()
            qa = torch.from_numpy(np.clip(xs / sa, -448, 448)).to(torch.float8_e4m3fn).float()
            result = ((qa @ qw.T) * float(sw)) * float(sa)
            if mode.endswith("f16"):
                result = result.half().float()  # cuBLASLt D is half before bias/erf.
        else:
            raise ValueError("independent decoder supports new CUDA modes and dense F16, not GGML block-Q8")
    result = result + torch.from_numpy(bias)
    if data["gelu"]:
        result = torch.nn.functional.gelu(result, approximate="none")
    if mode != "ggml-f16" and mode.endswith("f16"):
        result = result.half().float()
    return result.numpy()


def verify(input_path, run_directory, threads=2):
    import torch
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference environment")
    report = read_json(run_directory / "probe.json")
    if report.get("complete") is not True or report.get("diagnostic_only") is not True:
        raise ValueError("probe did not complete")
    data, arrays = read_input(input_path)
    if any(report.get(key) != data[key] for key in ("m", "n", "k")) or report.get("gelu_erf") != data["gelu"]:
        raise ValueError("probe shape or epilogue differs")
    identities = {str(path): sha256_file(path) for path in
                  (input_path, run_directory / "probe.json", run_directory / "output.f32", Path(__file__))}
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    with torch.inference_mode():
        reference = decoded_reference(data, arrays, report["mode"], torch)
    if not np.isfinite(reference).all():
        raise ValueError("decoded reference overflowed")
    path = run_directory / "output.f32"
    if path.stat().st_size != data["m"] * data["n"] * 4:
        raise ValueError("probe output has incorrect length")
    values = np.memmap(path, dtype="<f4", mode="r", shape=(data["n"], data["m"]))
    difference_squared, norm_squared, maximum = 0.0, 0.0, 0.0
    for start in range(0, data["n"], 128):
        actual = values[start:start + 128].astype(np.float64)
        expected = reference[np.arange(start, start + len(actual)) % data["rows"]].astype(np.float64)
        if not np.isfinite(actual).all():
            raise ValueError("non-finite actual output")
        difference = actual - expected
        difference_squared += float(np.square(difference).sum())
        norm_squared += float(np.square(expected).sum())
        maximum = max(maximum, float(np.abs(difference).max()))
    relative = float(np.sqrt(difference_squared / max(norm_squared, 1e-60)))
    half_output = report["mode"] != "ggml-f16" and report["mode"].endswith("f16")
    tolerance = 8e-4 if half_output else 2e-5
    passed = relative <= tolerance if norm_squared > 1e-30 else maximum <= 1e-6
    if any(sha256_file(Path(path)) != expected for path, expected in identities.items()):
        raise ValueError("probe or verifier changed during verification")
    return {"schema_version": 1, "diagnostic_only": True, "passed": passed, "mode": report["mode"],
            "scope": "Every output element against CPU F32 products of independently encoded/decoded operands; not model quality",
            "relative_l2": relative, "max_abs": maximum, "relative_l2_limit": tolerance,
            "torch": torch.__version__, "numpy": np.__version__, "artifact_sha256": identities}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    try:
        result = verify(args.input, args.run, args.threads)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"{'PASS' if result['passed'] else 'FAIL'}: {args.output}")
        if not result["passed"]:
            raise SystemExit(1)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"linear probe verification failed: {error}\n")


if __name__ == "__main__":
    main()
