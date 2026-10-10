"""Reject altered same-operand CUDA Q8 dots and malformed packed operands."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.validation.verify_q8_mmvq_dots import verify


class Q8MmvqDotChecks(unittest.TestCase):
    def test_zero_reference_and_last_dot_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "probe"
            staging = directory / "staging"
            staging.mkdir(parents=True)
            source = root / "input.bin"
            m, n, k = 2, 2, 32
            with source.open("wb") as stream:
                stream.write(struct.pack("<8s5If", b"SLPROB01", m, n, k, n, 0, 1.0))
                for values in (np.zeros((m, k), dtype="<f4"), np.zeros(m, dtype="<f4"),
                               np.ones(k, dtype="<f4"), np.zeros((n, k), dtype="<f4"),
                               np.zeros((n, m), dtype="<f4")):
                    values.tofile(stream)
            (directory / "rhs.f32").write_bytes(bytes(n * k * 4))
            weights = np.zeros((m, 34), dtype=np.uint8)
            weights[:, :2] = np.array([1, 1], dtype="<f2").view("u1").reshape(m, 2)
            weights[0, 2] = 1
            weights.tofile(directory / "weight.q8_0")
            np.zeros((n, m), dtype="<f4").tofile(directory / "raw-dot.f32")
            (directory / "probe.json").write_text(json.dumps({
                "complete": True, "raw_q8_operands_saved": True, "mode": "ggml-q8",
                "output_type": "f32", "m": m, "n": n, "k": k, "raw_cuda_nodes": 1, "cpu_nodes": 0}))
            np.zeros((n, 36), dtype=np.uint8).tofile(staging / "rhs.q8_1")
            (staging / "probe.json").write_text(json.dumps({
                "complete": True, "weight_type": "q8_0", "staging_path": "MMVQ",
                "width": k, "rows": n, "blocks": n, "compute_capability": 89, "device": "test"}))
            self.assertTrue(verify(source, directory)["passed"])
            altered = np.zeros((n, m), dtype="<f4")
            altered[-1, -1] = 0.1
            altered.tofile(directory / "raw-dot.f32")
            result = verify(source, directory)
            self.assertFalse(result["passed"])
            self.assertEqual((result["worst_token"], result["worst_row"]), (1, 1))
            weights[0, 2] = 128
            weights.tofile(directory / "weight.q8_0")
            with self.assertRaisesRegex(ValueError, "invalid Q8_0"):
                verify(source, directory)
            (directory / "rhs.f32").write_bytes(bytes(n * k * 4 - 1))
            with self.assertRaisesRegex(ValueError, "payload length"):
                verify(source, directory)
