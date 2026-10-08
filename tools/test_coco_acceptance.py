"""Independent COCO comparisons for ranked AP and clustered bootstrap."""

import contextlib
import copy
import io
import json
import unittest

import numpy as np

from coco_acceptance import evaluate_ranked, paired_bootstrap, quality_gates
from precision_acceptance import compare_objects, load_gates, required_queries
from runtime_quantization import encode_mask
import test_coco_screening
from test_precision_acceptance import output_fixture


class CocoAcceptanceChecks(unittest.TestCase):
    def fixture(self):
        _, coco, _, legacy, _ = test_coco_screening.CocoScreeningChecks().fixture()
        outputs = {}
        for key, value in legacy.items():
            output = output_fixture(width=value["width"], height=value["height"], prompt=value["prompt"])
            output["query_scores"], output["query_boxes"] = value["query_scores"], value["query_boxes"]
            output["ranked_queries"], _ = required_queries(output["query_scores"])
            masks = {row["query_index"]: row["mask"] for row in value["detections"]}
            for row in output["masks"]:
                if row["query_index"] in masks:
                    row["mask"] = masks[row["query_index"]]
            outputs[key] = output
        return coco, outputs

    def test_ranked_ap_keeps_low_score_true_objects_while_deployed_miou_drops(self):
        coco, outputs = self.fixture()
        reference, _ = evaluate_ranked(coco, outputs)
        self.assertAlmostEqual(reference["mask_ap"], 1)
        for output in outputs.values():
            output["query_scores"][0] *= .2
        actual, _ = evaluate_ranked(coco, outputs)
        self.assertAlmostEqual(actual["mask_ap"], 1)
        self.assertEqual(actual["positive_union_mask_miou"], 0)
        self.assertEqual(actual["area_coverage"]["small"]["instances"], 2)
        self.assertEqual(sum(row["kind"] == "crowd-only-or-empty" for row in actual["pairs"]), 1)

    def test_cached_area_ap_matches_standard_coco_accumulation(self):
        coco, outputs = self.fixture()
        metrics, cache = evaluate_ranked(coco, outputs)
        for area, index in (("all", 0), ("small", 3), ("medium", 4), ("large", 5)):
            self.assertEqual(cache.ap(area), metrics["coco_statistics"][index])

    def test_complete_metrics_and_gate_report_are_json_serializable(self):
        coco, outputs = self.fixture()
        metrics, _ = evaluate_ranked(coco, outputs)
        gates = load_gates()
        comparisons = [compare_objects(value, value, gates["profiles"]["f32"], gates["common"])
                       for value in outputs.values()]
        report = quality_gates(metrics, metrics, comparisons, gates["profiles"]["f32"], gates["common"],
                               {"ap_upper_bound": 0.0, "miou_upper_bound": 0.0}, len(coco.imgs))
        self.assertEqual(json.loads(json.dumps(metrics, allow_nan=False)), metrics)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_repeated_images_and_equal_score_false_positives_match_cloned_coco(self):
        from pycocotools.coco import COCO
        coco, outputs = self.fixture()
        changed = output_fixture([(0, .9, (0, 0, 2, 2)), (1, .9, (3, 2, 8, 6)),
                                  (2, .9, (0, 0, 2, 2))], width=10, height=8, prompt="person")
        outputs[(11, 1)] = changed
        _, cache = evaluate_ranked(coco, outputs)
        counts = np.asarray([2, 0, 1])
        cloned, cloned_outputs = COCO(), {}
        cloned.dataset = {"info": {}, "images": [], "annotations": [], "categories": copy.deepcopy(coco.dataset["categories"])}
        next_image, next_annotation = 101, 1
        for old_image, repetitions in zip(cache.image_ids, counts):
            for _ in range(repetitions):
                image = {**copy.deepcopy(coco.imgs[old_image]), "id": next_image}
                cloned.dataset["images"].append(image)
                for annotation in coco.imgToAnns[old_image]:
                    cloned.dataset["annotations"].append({**copy.deepcopy(annotation), "image_id": next_image, "id": next_annotation})
                    next_annotation += 1
                for key, output in outputs.items():
                    if key[0] == old_image:
                        cloned_outputs[(next_image, key[1])] = copy.deepcopy(output)
                next_image += 1
        with contextlib.redirect_stdout(io.StringIO()):
            cloned.createIndex()
        actual, _ = evaluate_ranked(cloned, cloned_outputs)
        self.assertAlmostEqual(cache.ap(counts=counts), actual["mask_ap"], places=12)
        self.assertNotAlmostEqual(cache.ap(counts=counts), cache.ap(), places=3)

    def test_pair_bootstrap_preserves_all_prompts_and_matches_identity_exactly(self):
        coco, outputs = self.fixture()
        metrics, cache = evaluate_ranked(coco, outputs)
        common = {**load_gates()["common"], "bootstrap_repetitions": 30}
        result = paired_bootstrap(cache, cache, metrics["pairs"], metrics["pairs"], common)
        # Some resamples contain only the crowd-only image and have no valid GT;
        # those repetitions must remain missing, not be imputed to a perfect AP.
        self.assertIn(result["ap_upper_bound"], (None, 0.0))
        self.assertIn(result["miou_upper_bound"], (None, 0.0))
        self.assertEqual(result, paired_bootstrap(cache, cache, metrics["pairs"], metrics["pairs"], common))
        with self.assertRaisesRegex(ValueError, "identity"):
            paired_bootstrap(cache, cache, metrics["pairs"], metrics["pairs"][:-1], common)

    def test_negative_regressions_are_not_cancelled_by_improvements_elsewhere(self):
        coco, reference = self.fixture()
        candidate = copy.deepcopy(reference)
        reference[(11, 3)] = output_fixture([(0, .9, (0, 0, 2, 2))], width=10, height=8, prompt="cat")
        candidate[(12, 1)] = output_fixture([(0, .9, (0, 0, 2, 2))], width=10, height=8, prompt="person")
        r, _ = evaluate_ranked(coco, reference)
        c, _ = evaluate_ranked(coco, candidate)
        gates = load_gates()
        objects = [compare_objects(reference[key], candidate[key], gates["profiles"]["q8_0"], gates["common"])
                   for key in sorted(reference)]
        result = quality_gates(r, c, objects, gates["profiles"]["q8_0"], gates["common"],
                               {"ap_upper_bound": 1.0, "miou_upper_bound": 0.0}, 3)
        self.assertEqual(result["reference_negative_detections"], 1)
        self.assertEqual(result["candidate_negative_detections"], 1)
        self.assertEqual(result["new_negative_pairs"], 1)
        self.assertEqual(result["extra_negative_detections"], 1)
        checks = {row["name"]: row for row in result["checks"]}
        self.assertEqual(checks["new_negative_rate"]["status"], "FAIL")

    def test_crowd_pixels_do_not_reduce_union_miou(self):
        coco, outputs = self.fixture()
        crowd = np.zeros((8, 10), np.uint8)
        crowd[2:6, 3:4] = 1
        coco.dataset["annotations"].append({"id": 99, "image_id": 11, "category_id": 1,
                                            "bbox": [3, 2, 1, 4], "area": 4, "iscrowd": 1, "segmentation": encode_mask(crowd)})
        with contextlib.redirect_stdout(io.StringIO()):
            coco.createIndex()
        output = output_fixture([(0, .9, (4, 2, 8, 6))], width=10, height=8, prompt="person")
        outputs[(11, 1)] = output
        metrics, _ = evaluate_ranked(coco, outputs)
        self.assertEqual(metrics["positive_union_mask_miou"], 1)


if __name__ == "__main__":
    unittest.main()
