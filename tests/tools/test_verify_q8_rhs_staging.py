"""Q8_1 staging checks for sums, ties and corrupted native bytes."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.validation.verify_q8_rhs_staging import check_blocks, verify


class Q8RhsStagingChecks(unittest.TestCase):
    def test_original_input_sum_is_not_reconstructed_from_integers(self):
        values = np.zeros((1, 32), dtype="<f4")
        values[0, :3] = [127.0, 0.49, 0.49]
        block = np.zeros((1, 36), dtype=np.uint8)
        block[0, :2] = np.array([1], dtype="<f2").view("u1")
        block[0, 2:4] = np.array([127.98], dtype="<f2").view("u1")
        block[0, 4] = 127
        result = check_blocks(values, block)
        self.assertEqual((result["scale_errors"], result["integer_errors"]), (0, 0))
        self.assertEqual(result["sum_errors"], 0)
        block[0, 2:4] = np.array([127], dtype="<f2").view("u1")
        self.assertEqual(check_blocks(values, block)["sum_errors"], 1)

    def test_native_report_and_payload_mutations_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, payload, report = root / "input.f32", root / "rhs.q8_1", root / "probe.json"
            np.zeros((2, 32), dtype="<f4").tofile(source)
            np.zeros((2, 36), dtype=np.uint8).tofile(payload)
            document = {"complete": True, "weight_type": "q8_0", "staging_path": "MMVQ",
                        "width": 32, "rows": 2, "blocks": 2, "compute_capability": 89, "device": "test"}
            report.write_text(json.dumps(document))
            self.assertTrue(verify(source, payload, report, 32)["passed"])
            raw = bytearray(payload.read_bytes())
            raw[4] = 1
            payload.write_bytes(raw)
            self.assertEqual(verify(source, payload, report, 32)["integer_errors"], 1)
            payload.write_bytes(raw[:-1])
            with self.assertRaisesRegex(ValueError, "payload length"):
                verify(source, payload, report, 32)
            np.zeros((2, 36), dtype=np.uint8).tofile(payload)
            document["staging_path"] = "MMQ"
            report.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "native report"):
                verify(source, payload, report, 32)
