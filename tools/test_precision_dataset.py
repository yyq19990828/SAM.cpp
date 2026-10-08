"""Previously viewed data, duplicate content and reserved-set boundaries."""

import copy
from pathlib import Path
import tempfile
import unittest

from prepare_coco_acceptance import load_precision_dataset, make_dataset, validate_precision_dataset
from sam3_artifacts import SAM3_REVISION, sha256_file, write_json


class PrecisionDatasetChecks(unittest.TestCase):
    def fixture(self, root):
        (root / "val2017").mkdir()
        value = {"images": [], "categories": [{"id": n, "name": f"class {n}"} for n in range(1, 5)], "annotations": []}
        for image in range(12):
            name = f"{image:012d}.jpg"
            (root / "val2017" / name).write_bytes(f"unique image {image}".encode())
            value["images"].append({"id": image, "file_name": name})
            for category in (1, 2 if image % 2 else 3):
                value["annotations"].append({"image_id": image, "category_id": category, "area": 100, "iscrowd": 0})
        # A fresh ID with previously used content must not be a new holdout.
        (root / "val2017/000000000012.jpg").write_bytes((root / "val2017/000000000000.jpg").read_bytes())
        value["images"].append({"id": 12, "file_name": "000000000012.jpg"})
        value["annotations"].append({"image_id": 12, "category_id": 1, "area": 100, "iscrowd": 0})
        annotations = root / "annotations.json"
        write_json(annotations, value)
        previous = {"schema_version": 1, "kind": "sam3-calibration-dataset", "sam3_revision": SAM3_REVISION,
                    "seed": 1, "provenance": {"annotations_sha256": sha256_file(annotations)}, "samples": []}
        for image, split in ((0, "calibration"), (1, "selection"), (2, "evaluation")):
            name = f"val2017/{image:012d}.jpg"
            previous["samples"].append({"id": f"coco-{image:012d}", "coco_image_id": image, "split": split,
                                        "source_sha256": sha256_file(root / name), "source_group": f"coco-image-{image}",
                                        "image": name, "prompts": ["class 1"]})
        used = root / "previous.json"
        write_json(used, previous)
        return annotations, used

    def test_previous_eval_becomes_development_without_reusing_its_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations, previous = self.fixture(root)
            value = make_dataset(annotations, root, [previous], set(), evaluation=3, reserve=3, seed=42)
            self.assertEqual(value["provenance"]["eligible_unique_images"], 12)
            roles = {row["coco_image_id"]: row["split"] for row in value["samples"]}
            self.assertEqual(roles[0], "calibration")
            self.assertEqual(roles[1], "development")
            self.assertEqual(roles[2], "development")
            self.assertNotIn(12, roles)
            self.assertEqual(value, make_dataset(annotations, root, [previous], set(), 3, 3, 42))
            for row in value["samples"]:
                self.assertEqual(len(row["positive_category_ids"]), 2)
                self.assertEqual(len(row["prompts"]), 3)

    def test_manifest_refuses_relabelled_used_image_or_overlapping_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations, previous = self.fixture(root)
            value = make_dataset(annotations, root, [previous], set(), 3, 3)
            bad = copy.deepcopy(value)
            bad["samples"][0]["split"] = "evaluation"
            with self.assertRaisesRegex(ValueError, "leaked"):
                validate_precision_dataset(bad)
            bad = copy.deepcopy(value)
            bad["samples"][-1]["source_sha256"] = bad["samples"][0]["source_sha256"]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_precision_dataset(bad)

    def test_source_change_and_insufficient_unused_population_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations, previous = self.fixture(root)
            value = make_dataset(annotations, root, [previous], set(), 3, 3)
            path = root / "precision.json"
            write_json(path, value)
            load_precision_dataset(path, root)
            with self.assertRaisesRegex(ValueError, "insufficient"):
                make_dataset(annotations, root, [previous], set(), 5, 5)
            (root / value["samples"][-1]["image"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed image"):
                load_precision_dataset(path, root)

    def test_extra_consumed_ids_and_fixed_regression_hashes_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations, previous = self.fixture(root)
            value = make_dataset(annotations, root, [previous], {sha256_file(root / "val2017/000000000003.jpg")},
                                 3, 3, additional_used_ids=[4])
            fresh = {row["coco_image_id"] for row in value["samples"] if row["split"] in ("evaluation", "reserve")}
            self.assertTrue(fresh.isdisjoint({0, 1, 2, 3, 4, 12}))


if __name__ == "__main__":
    unittest.main()
