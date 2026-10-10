"""Behavioral checks for precision-specific quality and semantic object parity."""

import copy
import itertools
import unittest

import numpy as np

from tools.archive.precision_v2_v3.precision_acceptance import (combine_statuses, compare_objects, load_gates, object_gates,
                                  quality_profile, required_queries, spatial_assignment, upper_gate,
                                  validate_output, validate_rle)
from tools.quantize.runtime_quantization import encode_mask


def output_fixture(objects=(), width=64, height=48, prompt="object"):
    scores, boxes, binary = [0.0] * 200, [[0.0, 0.0, 0.0, 0.0] for _ in range(200)], {}
    for query, score, box in objects:
        scores[query], boxes[query] = score, list(box)
        value = np.zeros((height, width), dtype=np.uint8)
        x0, y0, x1, y1 = box
        value[y0:y1, x0:x1] = 1
        binary[query] = value
    ranked, required = required_queries(scores)
    empty = encode_mask(np.zeros((height, width), dtype=np.uint8))
    return {"schema_version": 2, "width": width, "height": height, "prompt": prompt,
            "token_ids": [0] * 32, "query_scores": scores, "query_boxes": boxes,
            "ranked_queries": ranked,
            "masks": [{"query_index": q, "mask": encode_mask(binary[q]) if q in binary else copy.deepcopy(empty)}
                      for q in required]}


class PrecisionAcceptanceChecks(unittest.TestCase):
    def setUp(self):
        self.gates = load_gates()
        self.common = self.gates["common"]
        self.profile = self.gates["profiles"]["q8_0"]

    def compare(self, reference, candidate, **kwargs):
        return compare_objects(reference, candidate, self.profile, self.common, **kwargs)

    def recipe(self, precision="f32", compute="f32", cache="f32"):
        quantized = precision.startswith("q")
        return {"schema_version": 2, "task": "image", "backend": "cuda", "weight_precision": precision,
                "compute_mode": compute, "feature_cache": cache,
                "storage_profile": f"image-full-linear-{precision}-v1" if quantized else "dense",
                "quantization_modules": ["vision", "text", "fusion", "decoder"] if quantized else []}

    def test_cache_cannot_widen_parent_budget_and_increments_are_separate(self):
        name, absolute = quality_profile(self.recipe(cache="mixed-q8_0"), self.gates)
        _, incremental = quality_profile(self.recipe(cache="mixed-q8_0"), self.gates, incremental=True)
        self.assertEqual(name, "f32")
        self.assertLess(absolute["ap_drop_max"], incremental["ap_drop_max"])
        name, q4 = quality_profile(self.recipe("q4_k", "f16", "mixed-q8_0"), self.gates)
        self.assertEqual(name, "q4_k")
        self.assertGreater(q4["ap_drop_max"], absolute["ap_drop_max"])
        self.assertEqual(combine_statuses(["FAIL", "PASS"]), "FAIL")

    def test_unknown_or_incomplete_recipe_is_not_mapped_to_loose_budget(self):
        for changes in ({"weight_precision": "int4"}, {"compute_mode": "auto"}, {"feature_cache": "q8"},
                        {"quantization_modules": ["vision"]}, {"storage_profile": ""},
                        {"task": "video"}, {"backend": "auto"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                quality_profile({**self.recipe(), **changes}, self.gates)
        with self.assertRaises(ValueError):
            quality_profile(self.recipe("q4_k", "w8a8"), self.gates)
        with self.assertRaises(ValueError):
            quality_profile(self.recipe(compute="fp8-e4m3"), self.gates)

    def test_permuted_queries_preserve_objects(self):
        original = output_fixture([(3, .9, (2, 3, 20, 24)), (7, .8, (30, 10, 45, 25))])
        actual = output_fixture([(33, .9, (2, 3, 20, 24)), (1, .8, (30, 10, 45, 25))])
        result = self.compare(original, actual)
        self.assertEqual(result["bad_objects"], 0)
        self.assertEqual({(p["reference_query"], p["candidate_query"]) for p in result["matches"]}, {(3, 33), (7, 1)})
        self.assertEqual(len(result["fixed_query_selection_changes"]), 4)

    def test_actual_miss_duplicate_and_low_to_high_are_not_hidden(self):
        original = output_fixture([(0, .9, (2, 3, 20, 24))])
        missing = self.compare(original, output_fixture())
        self.assertEqual(missing["missing_high_objects"], 1)
        duplicate = self.compare(original, output_fixture([(1, .9, (2, 3, 20, 24)), (2, .9, (2, 3, 20, 24))]))
        self.assertEqual(duplicate["extra_high_objects"], 1)
        changed = self.compare(output_fixture([(0, .51, (2, 3, 20, 24))]), original)
        self.assertEqual(changed["bad_objects"], 1)
        self.assertEqual(changed["high_reference_objects"], 0)

    def test_single_pair_must_pass_all_three_quality_limits(self):
        original = output_fixture([(0, .8, (2, 3, 20, 24))])
        candidate = copy.deepcopy(original)
        candidate["query_scores"][0] = .85
        result = self.compare(original, candidate)
        self.assertEqual(result["missing_high_objects"], 0)
        self.assertEqual(result["bad_objects"], 1)
        # Keep the test away from binary representation of the exact .05 boundary.
        candidate["query_scores"][0] = .84
        q4 = compare_objects(original, candidate, self.gates["profiles"]["q4_k"], self.common)
        self.assertEqual(q4["bad_objects"], 0)

    def test_exact_score_boundaries_and_small_object_disappearance(self):
        reference = output_fixture([(0, .6, (1, 1, 3, 3)), (1, .5, (10, 10, 20, 20))])
        result = self.compare(reference, output_fixture())
        self.assertEqual(result["high_reference_objects"], 1)
        self.assertEqual(result["reference_detections"], 1)
        self.assertEqual(result["missing_high_objects"], 1)
        self.assertEqual(result["protected_objects"], 0)

    def test_protected_gt_requires_one_to_one_preservation(self):
        original = output_fixture([(0, .95, (0, 0, 32, 40)), (1, .96, (32, 0, 64, 40))])
        gt = [row["mask"] for row in original["masks"][:2]]
        one = output_fixture([(4, .95, (0, 0, 32, 40))])
        result = self.compare(original, one, ground_truth=gt)
        self.assertEqual(result["protected_objects"], 2)
        self.assertEqual(result["protected_misses"], 1)

    def test_object_rates_do_not_hide_small_or_absent_denominators(self):
        row = self.compare(output_fixture([(0, .9, (0, 0, 10, 10))]), output_fixture([(1, .9, (0, 0, 10, 10))]))
        self.assertEqual(object_gates([row], self.profile, self.common)["status"], "INCONCLUSIVE")
        self.assertEqual(object_gates([row], self.profile, self.common, strict=True)["status"], "PASS")
        bad = self.compare(output_fixture(), output_fixture([(1, .9, (0, 0, 10, 10))]))
        self.assertNotEqual(object_gates([bad], self.profile, self.common)["status"], "PASS")

    def test_ranked_export_retains_low_scores_and_all_deployed_queries(self):
        output = output_fixture([(q, .8, (0, 0, 2, 2)) for q in range(120)])
        self.assertEqual(len(output["ranked_queries"]), 100)
        self.assertEqual(len(output["masks"]), 120)
        validate_output(output)
        output["masks"].pop()
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_output(output)
        sparse = output_fixture()
        self.assertEqual(len(sparse["masks"]), 100)
        validate_output(sparse)

    def test_bad_rle_rejected_before_native_code_and_canonical_round_trip(self):
        rng = np.random.default_rng(17)
        for mask in (np.zeros((5, 6), np.uint8), np.ones((5, 6), np.uint8), rng.integers(0, 2, (5, 6), dtype=np.uint8)):
            from pycocotools import mask as masks
            actual = masks.decode(validate_rle(encode_mask(mask), 5, 6))
            np.testing.assert_array_equal(actual, mask)
        for encoded in ("", "/", "o", "oooo", "1111", "_" * 100):
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                validate_rle({"size": [5, 6], "counts": encoded}, 5, 6)
        invalid = output_fixture()
        invalid["query_scores"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_output(invalid)

    def test_matching_count_priorities_beat_more_attractive_low_objects(self):
        self.assertEqual(spatial_assignment([[.51], [1.0]], [True, False]), [(0, 0)])
        self.assertEqual(spatial_assignment([[.9, .8], [.8, 0]], [True, True]), [(0, 1), (1, 0)])
        self.assertEqual(spatial_assignment(np.ones((2, 2)), [True, True], [[1, 0], [0, 1]]), [(0, 1), (1, 0)])
        self.assertEqual(spatial_assignment(np.ones((2, 2)), [True, True]), [(0, 0), (1, 1)])

    def test_lexicographic_solver_matches_exhaustive_small_assignments(self):
        rng = np.random.default_rng(441)
        for _ in range(30):
            ious = rng.integers(0, 11, (3, 3)) / 10
            high = rng.integers(0, 2, 3)
            actual = spatial_assignment(ious, high)
            def value(pairs):
                return (sum(high[r] for r, _ in pairs), len(pairs), round(sum(ious[r, c] for r, c in pairs), 12))
            possible = []
            for chosen in itertools.product(range(-1, 3), repeat=3):
                cols = [c for c in chosen if c >= 0]
                if len(set(cols)) != len(cols):
                    continue
                pairs = [(r, c) for r, c in enumerate(chosen) if c >= 0]
                if all(ious[r, c] >= .5 for r, c in pairs):
                    possible.append(value(pairs))
            self.assertEqual(value(actual), max(possible))

    def test_confidence_limit_and_missing_statistic_are_not_passes(self):
        self.assertEqual(upper_gate(.001, .002, "ap", .003)["status"], "INCONCLUSIVE")
        self.assertEqual(upper_gate(.003, .002, "ap", .004)["status"], "FAIL")
        self.assertEqual(upper_gate(.003, .002, "ap")["status"], "FAIL")
        self.assertEqual(upper_gate(.001, .002, "ap", .002)["status"], "PASS")
        self.assertEqual(upper_gate(None, .002, "ap")["status"], "INCONCLUSIVE")
        with self.assertRaises(ValueError):
            upper_gate(float("nan"), .002, "ap", .002)


if __name__ == "__main__":
    unittest.main()
