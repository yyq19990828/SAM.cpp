"""Check MMQ padding, transposed rows and corrupted native staging bytes."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.validation.verify_q8_mmq_rhs_staging import check_blocks, verify


class Q8MmqRhsStagingChecks(unittest.TestCase):
    def test_f32_scale_and_integer_fields_are_independent(self):
        values = np.zeros((1, 128), dtype="<f4")
        values[0, 0] = 127
        payload = np.zeros((1, 144), dtype=np.uint8)
        payload[0, :4] = np.array([1], dtype="<f4").view("u1")
        payload[0, 16] = 127
        result = check_blocks(values, payload)
        self.assertEqual((result["scale_errors"], result["integer_errors"]), (0, 0))
        payload[0, 16] = 126
        self.assertEqual(check_blocks(values, payload)["integer_errors"], 1)
        payload[0, 16] = 127
        payload[0, :4] = np.array([1.01], dtype="<f4").view("u1")
        self.assertEqual(check_blocks(values, payload)["scale_errors"], 1)

    def test_row_transposition_padding_and_native_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, payload, report = root / "input.f32", root / "rhs.q8_1_mmq", root / "probe.json"
            values = np.zeros((2, 160), dtype="<f4")
            values[0, 0], values[1, 128] = 127, -127
            values.tofile(source)
            blocks = np.zeros((4, 2, 144), dtype=np.uint8)
            blocks[0, 0, :4] = np.array([1], dtype="<f4").view("u1")
            blocks[0, 0, 16] = 127
            blocks[1, 1, :4] = np.array([1], dtype="<f4").view("u1")
            blocks[1, 1, 16] = 129
            blocks.tofile(payload)
            document = {"complete": True, "weight_type": "q8_0", "staging_path": "MMQ",
                        "width": 160, "padded_width": 512, "rows": 2, "blocks": 8,
                        "compute_capability": 89, "device": "test"}
            report.write_text(json.dumps(document))
            self.assertTrue(verify(source, payload, report, 160)["passed"])
            blocks[1, 1, 16] = 0
            blocks.tofile(payload)
            self.assertEqual(verify(source, payload, report, 160)["integer_errors"], 1)
            payload.write_bytes(payload.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "payload length"):
                verify(source, payload, report, 160)
            blocks.tofile(payload)
            document["staging_path"] = "MMVQ"
            report.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "native report"):
                verify(source, payload, report, 160)
