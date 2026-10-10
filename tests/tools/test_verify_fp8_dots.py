"""Meaningful boundaries for independent FP8-byte diagnostics."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.validation.verify_fp8_dots import decode_e4m3, verify


class FP8DotChecks(unittest.TestCase):
    def test_e4m3_extremes_and_nan(self):
        values = decode_e4m3(np.array([0x00, 0x01, 0x38, 0x7e, 0xfe], np.uint8))
        np.testing.assert_array_equal(values, [0, 2**-9, 1, 448, -448])
        with self.assertRaisesRegex(ValueError, "NaN"):
            decode_e4m3(np.array([0x7f], np.uint8))

    def test_first_failure_must_match_saved_operands_and_raw_dot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, directory = root / "input.bin", root / "probe"
            directory.mkdir()
            source.write_bytes(struct.pack("<8s5If", b"SLPROB01", 2, 2, 1, 2, 0, 1.0))
            (directory / "weight.e4m3").write_bytes(bytes([0x38, 0x38]))
            (directory / "activation.e4m3").write_bytes(bytes([0x38, 0x38]))
            np.array([0.5, 1, 1, 1], dtype="<f4").tofile(directory / "raw-f32.bin")
            np.array([1], dtype="<f4").tofile(directory / "weight-scale.f32")
            np.array([1], dtype="<f4").tofile(directory / "activation-scale.f32")
            failure = {"mode": "fp8-f32", "index": 0, "actual": 0.5, "expected": 1,
                       "weight_scale": 1, "input_scale": 1, "algorithm_id": 35,
                       "algorithm_index": 0, "heuristic_count": 8}
            (directory / "kernel-check-failure.json").write_text(json.dumps(failure))
            result = verify(source, directory)
            self.assertTrue(result["independent_reference_reproduced"])
            self.assertEqual((result["failed_dots"], result["first_failure"]), (1, 0))
            np.array([1, 1, 1, 1], dtype="<f4").tofile(directory / "raw-f32.bin")
            with self.assertRaisesRegex(ValueError, "reported failed dot differs"):
                verify(source, directory)
            (directory / "weight.e4m3").write_bytes(bytes([0x7f, 0x38]))
            with self.assertRaisesRegex(ValueError, "NaN"):
                verify(source, directory)
