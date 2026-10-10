"""Boundary and tamper checks for the independent Q8 cache arithmetic gate."""

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from tools.validation.verify_precision_q8_cache import check_blocks, check_native_report, verify


class Q8CacheArithmeticChecks(unittest.TestCase):
    def test_zero_ties_and_adjacent_scale(self):
        values = np.zeros((3, 32), dtype="<f4")
        values[1, 0] = 1
        values[1, 1] = np.float32(0.5 / 127)
        values[1, 2] = np.float32(-0.5 / 127)
        values[2, 0] = np.float32(0.06240535154938698)
        payload = np.zeros((3, 34), dtype=np.uint8)
        payload[1, :2] = np.frombuffer(np.float16(1 / 127).tobytes(), dtype=np.uint8)
        payload[1, 2:5] = np.asarray([127, 1, -1], dtype=np.int8).view(np.uint8)
        # This adjacent F16 scale is reached by the pinned CUDA F32 division
        # at an actual real-tensor half midpoint.
        payload[2, :2] = np.frombuffer(np.uint16(0x1006).tobytes(), dtype=np.uint8)
        payload[2, 2] = 127
        scales = payload[:, :2].copy().view("<f2").reshape(-1)
        decoded = payload[:, 2:].view(np.int8).astype("<f4") * scales.astype("<f4")[:, None]
        result = check_blocks(values, payload, decoded)
        self.assertEqual(result["scale_tie_allowances"], 1)
        self.assertTrue(all(result[key] == 0 for key in ("scale_errors", "integer_errors", "decoded_bit_errors")))
        wrong = payload.copy()
        wrong[1, 3] = 2  # The other side of a positive half tie is zero.
        self.assertEqual(check_blocks(values, wrong, decoded)["integer_errors"], 1)

    def test_detects_wrong_integer_scale_and_decoding(self):
        values = np.zeros((1, 32), dtype="<f4")
        values[0, 0] = 1
        values[0, 1] = np.float32(32 / 127)
        payload = np.zeros((1, 34), dtype=np.uint8)
        payload[0, :2] = np.frombuffer(np.float16(1 / 127).tobytes(), dtype=np.uint8)
        payload[0, 2:4] = [127, 32]
        decoded = payload[:, 2:].view(np.int8).astype("<f4") * np.float32(np.float16(1 / 127))
        self.assertEqual(check_blocks(values, payload, decoded)["integer_errors"], 0)
        wrong = payload.copy()
        wrong[0, 3] = 33
        self.assertEqual(check_blocks(values, wrong, decoded)["integer_errors"], 1)
        wrong = payload.copy()
        wrong[0, 0] += 2
        self.assertEqual(check_blocks(values, wrong, decoded)["scale_errors"], 1)
        wrong = payload.copy()
        wrong[0, :2] = np.frombuffer(np.float16(-0.0).tobytes(), dtype=np.uint8)
        self.assertEqual(check_blocks(values, wrong, decoded)["scale_errors"], 1)
        wrong = payload.copy()
        wrong[0, 3] = 128
        with self.assertRaisesRegex(ValueError, "invalid scale or integer"):
            check_blocks(values, wrong, decoded)
        wrong = decoded.copy()
        wrong[0, 1] = np.nextafter(wrong[0, 1], np.float32(np.inf))
        self.assertEqual(check_blocks(values, payload, wrong)["decoded_bit_errors"], 1)

    def test_file_lengths_and_row_layout_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, payload = root / "input.f32", root / "packed.bin"
            np.zeros(64, dtype="<f4").tofile(source)
            np.zeros((2, 34), dtype=np.uint8).tofile(payload)
            self.assertTrue(verify(source, payload, None, 32)["passed"])
            with self.assertRaisesRegex(ValueError, "block-aligned"):
                verify(source, payload, None, 31)
            payload.write_bytes(payload.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "payload length"):
                verify(source, payload, None, 32)

    def test_native_report_must_show_cuda_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            report = {"complete": True, "mode": "q8_0", "backend": "cuda", "rows": 17,
                      "channels": 96, "device": "CUDA0", "cuda_nodes": 2, "cpu_nodes": 0}
            path.write_text(json.dumps(report))
            self.assertEqual(check_native_report(path, 17, 96)["cuda_nodes"], 2)
            report["cpu_nodes"] = 1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "expected CUDA"):
                check_native_report(path, 17, 96)


if __name__ == "__main__":
    unittest.main()
