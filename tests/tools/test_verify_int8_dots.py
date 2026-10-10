"""Exact W8A8 dot checks over raw packed operands and malformed receipts."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.benchmark.prepare_linear_probe import write_case
from tools.validation.verify_int8_dots import verify


class ExactInt8DotChecks(unittest.TestCase):
    def fixture(self, root):
        m, n, k = 5, 7, 32
        weights = np.resize(np.asarray([-127, -3, 0, 4, 127], dtype=np.int8), (m, k))
        activations = np.resize(np.asarray([7, -11, 0, 127, -127], dtype=np.int8), (n, k))
        dots = activations.astype(np.int64) @ weights.astype(np.int64).T
        input_path, run = root / "input.bin", root / "run"
        run.mkdir()
        write_case(input_path, weights.astype(np.float32), np.zeros(m, np.float32),
                   activations[:3].astype(np.float32), np.zeros((3, m), np.float32),
                   np.ones(k, np.float32), 127, n, False)
        report = {"complete": True, "raw_int32_saved": True, "mode": "int8-token-f32",
                  "raw_int32_layout": "weight[M,K], activation[N,K], dot[N,M], row-major",
                  "m": m, "n": n, "k": k, "gelu_erf": False, "compute_capability": 89}
        (run / "probe.json").write_text(json.dumps(report))
        weights.tofile(run / "weight.i8")
        activations.tofile(run / "activation.i8")
        dots.astype("<i4").tofile(run / "raw-int32.bin")
        np.ones(m, np.float32).tofile(run / "weight-scale.f32")
        np.ones(n, np.float32).tofile(run / "activation-scale.f32")
        return input_path, run

    def test_all_dots_are_checked_with_tail_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            source, run = self.fixture(Path(directory))
            result = verify(source, run, chunk_tokens=3)
            self.assertTrue(result["passed"])
            self.assertEqual(result["compared_dots"], 35)
            self.assertEqual(result["mismatches"], 0)
            dots = np.fromfile(run / "raw-int32.bin", dtype="<i4").reshape(7, 5)
            dots[-1, -1] += 1
            dots.tofile(run / "raw-int32.bin")
            changed = verify(source, run, chunk_tokens=3)
            self.assertFalse(changed["passed"])
            self.assertEqual(changed["mismatches"], 1)
            self.assertEqual(changed["worst"]["token"], 6)

    def test_rejects_invalid_operand_and_length(self):
        with tempfile.TemporaryDirectory() as directory:
            source, run = self.fixture(Path(directory))
            packed = bytearray((run / "weight.i8").read_bytes())
            packed[0] = 128
            (run / "weight.i8").write_bytes(packed)
            with self.assertRaisesRegex(ValueError, "-128"):
                verify(source, run)
            (run / "raw-int32.bin").write_bytes(b"\0" * 4)
            with self.assertRaisesRegex(ValueError, "length"):
                verify(source, run)

    def test_rejects_unclaimed_or_wrong_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            source, run = self.fixture(Path(directory))
            path = run / "probe.json"
            report = json.loads(path.read_text())
            report["raw_int32_saved"] = False
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "mismatched"):
                verify(source, run)


if __name__ == "__main__":
    unittest.main()
