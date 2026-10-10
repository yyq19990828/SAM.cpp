"""Fixed fixtures must not inherit dataset tail allowances or hide false positives."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.convert.sam3_artifacts import dump_array
from tests.tools import test_precision_acceptance
from tests.tools.test_precision_acceptance import output_fixture
from tools.archive.precision_v2_v3.validate_precision_regression import compare_cases, fixed_cases, selected_parity


class PrecisionRegressionChecks(unittest.TestCase):
    def outputs(self, objects=()):
        return {case["id"]: output_fixture(objects, prompt=case["prompt"]) for case in fixed_cases()}

    def test_small_fixed_suite_requires_zero_object_failures(self):
        recipe = test_precision_acceptance.PrecisionAcceptanceChecks().recipe("q4_k")
        original = self.outputs([(3, .9, (2, 3, 20, 24))])
        self.assertEqual(compare_cases(original, original, recipe)["status"], "PASS")
        changed = copy.deepcopy(original)
        changed["truck-truck"] = output_fixture(prompt="truck")
        result = compare_cases(original, changed, recipe)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["cases"][0]["objects"]["missing_high_objects"], 1)

    def test_new_gray_zone_detection_fails_an_empty_fixed_case(self):
        original = self.outputs()
        changed = copy.deepcopy(original)
        changed["truck-purple-elephant"] = output_fixture([(9, .55, (2, 3, 20, 24))], prompt="purple elephant")
        result = compare_cases(original, changed, test_precision_acceptance.PrecisionAcceptanceChecks().recipe("q4_k"))
        self.assertEqual(result["status"], "FAIL")
        failed = next(row for row in result["cases"] if row["id"] == "truck-purple-elephant")
        self.assertEqual(failed["objects"]["bad_objects"], 0)
        self.assertEqual(failed["gates"]["checks"][-1]["status"], "FAIL")

    def test_cache_increment_is_checked_even_when_absolute_case_passes(self):
        recipe = test_precision_acceptance.PrecisionAcceptanceChecks().recipe("q8_0", "f16", "f16")
        original = self.outputs([(3, .9, (2, 3, 20, 24))])
        changed = self.outputs([(3, .894, (2, 3, 20, 24))])
        self.assertEqual(compare_cases(original, changed, recipe)["status"], "PASS")
        self.assertEqual(compare_cases(original, changed, recipe, incremental=True)["status"], "FAIL")

    def test_raw_reference_transcode_must_preserve_every_deployed_pixel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = output_fixture([(3, .9, (2, 3, 20, 24))], prompt="truck")
            binary = np.zeros((48, 64), dtype=np.uint8)
            binary[3:24, 2:20] = 1
            result = {"schema_version": 1, "width": 64, "height": 48, "prompt": "truck", "score_threshold": .5,
                      "detections": [{"query_index": 3, "score": .9, "box": [2, 3, 20, 24],
                                      "mask": dump_array(root, "mask", binary)}]}
            self.assertEqual(len(selected_parity(payload, result, root)), 1)
            for mutation in ("pixel", "score", "box", "missing"):
                bad = copy.deepcopy(result)
                if mutation == "pixel":
                    binary[0, 0] = 1
                    bad["detections"][0]["mask"] = dump_array(root, "changed-mask", binary)
                elif mutation == "score":
                    bad["detections"][0]["score"] -= .00001
                elif mutation == "box":
                    bad["detections"][0]["box"][0] += .00001
                else:
                    bad["detections"] = []
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    selected_parity(payload, bad, root)


if __name__ == "__main__":
    unittest.main()
