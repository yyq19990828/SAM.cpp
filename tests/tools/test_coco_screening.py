"""Prompt-conditioned COCO evaluation boundaries, crowd handling and empty outputs."""

import contextlib
import copy
import io
import unittest

import numpy as np

from tools.validation.evaluate_coco_screening import annotation_metrics, prompted_ground_truth
from tools.quantize.runtime_quantization import encode_mask


class CocoScreeningChecks(unittest.TestCase):
    def fixture(self):
        from pycocotools.coco import COCO
        mask = np.zeros((8, 10), np.uint8)
        mask[2:6, 3:8] = 1
        coco = COCO()
        coco.dataset = {"info": {}, "images": [{"id": i, "width": 10, "height": 8} for i in (11, 12, 13)],
                        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}, {"id": 3, "name": "cat"}],
                        "annotations": [{"id": index, "image_id": image, "category_id": category,
                                         "bbox": [3, 2, 5, 4], "area": 20, "iscrowd": crowd,
                                         "segmentation": encode_mask(mask)} for index, image, category, crowd in
                                        ((1, 11, 1, 0), (2, 11, 2, 0), (3, 12, 2, 0), (4, 13, 1, 1))]}
        with contextlib.redirect_stdout(io.StringIO()):
            coco.createIndex()
        samples = [
            {"id": "a", "coco_image_id": 11, "prompts": ["person", "cat"], "positive_category_ids": [1], "negative_category_ids": [3]},
            {"id": "b", "coco_image_id": 12, "prompts": ["dog", "person"], "positive_category_ids": [2], "negative_category_ids": [1]},
            {"id": "c", "coco_image_id": 13, "prompts": ["person"], "positive_category_ids": [1], "negative_category_ids": []},
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            subset, _ = prompted_ground_truth(coco, samples)
        outputs = {}
        for key, prompt, positive in (((11, 1), "person", True), ((11, 3), "cat", False),
                                     ((12, 2), "dog", True), ((12, 1), "person", False), ((13, 1), "person", False)):
            scores = [0.0] * 200
            scores[0] = 0.9 if positive else 0.0
            outputs[key] = {"width": 10, "height": 8, "prompt": prompt, "query_scores": scores,
                            "query_boxes": [[3, 2, 8, 6] for _ in range(200)],
                            "detections": [{"query_index": 0, "mask": encode_mask(mask)}] if positive else []}
        return coco, subset, samples, outputs, mask

    def metrics(self, coco, outputs):
        with contextlib.redirect_stdout(io.StringIO()):
            return annotation_metrics(coco, outputs)

    def test_unprompted_ground_truth_excluded_without_losing_negative_pairs(self):
        _, subset, _, outputs, _ = self.fixture()
        self.assertNotIn(2, subset.anns)  # Dog on image 11 was not prompted.
        result = self.metrics(subset, outputs)
        self.assertGreater(result["mask_ap"], 0.999)
        self.assertEqual(result["positive_union_mask_miou"], 1)
        self.assertEqual(result["positive_pairs"], 2)
        self.assertEqual(result["negative_pairs"], 2)
        self.assertEqual(result["excluded_crowd_only_or_empty_pairs"], 1)

    def test_all_empty_detections_are_valid_and_score_zero(self):
        _, subset, _, outputs, _ = self.fixture()
        for output in outputs.values():
            output["query_scores"] = [0.0] * 200
            output["detections"] = []
        result = self.metrics(subset, outputs)
        self.assertEqual(result["mask_ap"], 0)
        self.assertEqual(result["positive_union_mask_miou"], 0)

    def test_negative_false_positive_is_counted_and_reduces_ap(self):
        _, subset, _, outputs, mask = self.fixture()
        outputs[(12, 1)]["query_scores"][0] = 1.0
        outputs[(12, 1)]["detections"] = [{"query_index": 0, "mask": encode_mask(mask)}]
        result = self.metrics(subset, outputs)
        self.assertEqual(result["negative_detection_count"], 1)
        self.assertEqual(result["negative_pairs_with_detections"], 1)
        self.assertLess(result["mask_ap"], 0.99)
        self.assertEqual(result["positive_union_mask_miou"], 1)

    def test_crowd_pixels_excluded_from_union_miou(self):
        _, subset, _, outputs, mask = self.fixture()
        crowd = np.zeros_like(mask)
        crowd[2:6, 3:4] = 1
        subset.dataset["annotations"].append({"id": 10, "image_id": 11, "category_id": 1, "bbox": [3, 2, 1, 4],
                                               "area": 4, "iscrowd": 1, "segmentation": encode_mask(crowd)})
        with contextlib.redirect_stdout(io.StringIO()):
            subset.createIndex()
        mask[crowd != 0] = 0
        outputs[(11, 1)]["detections"][0]["mask"] = encode_mask(mask)
        result = self.metrics(subset, outputs)
        self.assertEqual(result["positive_union_mask_miou"], 1)

    def test_bad_negative_category_and_duplicate_pair_rejected(self):
        coco, _, samples, _, _ = self.fixture()
        invalid = copy.deepcopy(samples)
        invalid[0]["negative_category_ids"] = [2]
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            prompted_ground_truth(coco, invalid)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prompted_ground_truth(coco, samples + [samples[0]])


if __name__ == "__main__":
    unittest.main()
