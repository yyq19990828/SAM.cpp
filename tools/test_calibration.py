"""Calibration leakage, exact moments, and bounded sampling regressions."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibration import ChannelStats, layer_seed, load_dataset, validate_dataset
from prepare_coco_calibration import make_dataset
from sam3_artifacts import BPE_SHA256, SAM3_REVISION, sha256_file, write_json
from study_activation_quantization import calibration_inputs, smooth_scale, quantize_rows


def dataset():
    return {"schema_version": 1, "kind": "sam3-calibration-dataset", "sam3_revision": SAM3_REVISION,
            "seed": 7, "samples": [
                {"id": "fit", "split": "calibration", "source_group": "photo-a", "image": "a.ppm",
                 "source_sha256": "a" * 64, "prompts": ["car", "absent object"]},
                {"id": "holdout", "split": "evaluation", "source_group": "photo-b", "image": "b.ppm",
                 "source_sha256": "b" * 64, "prompts": ["person"]}]}


class CalibrationChecks(unittest.TestCase):
    def test_study_rejects_diagnostic_and_changed_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"schema_version": 1, "kind": "sam3-vision-activation-calibration",
                        "sam3_revision": SAM3_REVISION, "bpe_sha256": BPE_SHA256,
                        "reference_kind": "official-checkpoint", "diagnostic_only": True,
                        "oracle": {"precision": "float32", "tf32": False, "autocast": False, "compile": False}}
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "diagnostic-only"):
                calibration_inputs(root)
            manifest["diagnostic_only"] = False
            manifest["calibration_id"] = "0" * 64
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "identity hash"):
                calibration_inputs(root)
            manifest["oracle"]["tf32"] = True
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "original F32"):
                calibration_inputs(root)

    def test_smoothing_identity_and_zero_channels(self):
        scale = smooth_scale([100, 1, 0], [1, 100, 0], 0.5)
        np.testing.assert_allclose(scale, [10, 0.1, 1], rtol=1e-7)
        values = np.array([[3.25, -5.5, 0], [7.125, 2.3, 0]], np.float32)
        weights = np.array([[0.2, 31.5, 0], [1, -2, 0]], np.float32)
        np.testing.assert_allclose((values / scale) @ (weights * scale).T, values @ weights.T, rtol=2e-7, atol=1e-6)
        for alpha in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                smooth_scale([1], [1], alpha)
        with self.assertRaises(ValueError):
            smooth_scale([1, 2], [1], 0.5)

    def test_int8_rounding_clipping_and_row_scales(self):
        values = np.array([[0.5, 1.5, 2.5, -1.5, 200], [0, 0, 0, 0, 0]], np.float32)
        encoded, scale, clipped = quantize_rows(values, np.float32(1))
        np.testing.assert_array_equal(encoded, [[0, 2, 2, -2, 127], [0, 0, 0, 0, 0]])
        self.assertEqual(encoded.dtype, np.int8)
        self.assertEqual(clipped, 1)
        encoded, scale, clipped = quantize_rows(np.array([[127, -127, 0], [254, -254, 0], [0, 0, 0]], np.float32))
        np.testing.assert_array_equal(scale, [[1], [2], [1]])
        np.testing.assert_array_equal(encoded, [[127, -127, 0], [127, -127, 0], [0, 0, 0]])
        self.assertEqual(clipped, 0)
        for invalid in (np.float32(0), np.float32(-1), np.float32(np.inf), np.ones((2, 5), np.float32)):
            with self.assertRaises(ValueError):
                quantize_rows(values, invalid)

    def test_coco_subsets_and_annotation_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            value = {"images": [], "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "dog"}],
                     "annotations": []}
            for identifier in range(8):
                name = f"{identifier}.ppm"
                (root / "images" / name).write_bytes(f"image {identifier}".encode())
                value["images"].append({"id": identifier, "file_name": name})
                value["annotations"].append({"image_id": identifier, "category_id": identifier % 2 + 1,
                                             "area": 100 if identifier % 3 else 10000})
            # Duplicate bytes under another COCO ID must not cross splits.
            (root / "images/8.ppm").write_bytes((root / "images/0.ppm").read_bytes())
            value["images"].append({"id": 8, "file_name": "8.ppm"})
            value["annotations"].append({"image_id": 8, "category_id": 1, "area": 100})
            annotation_path = root / "annotations.json"
            write_json(annotation_path, value)
            first = make_dataset(annotation_path, root, "images", (2, 2, 3), 19)
            value["images"].reverse()
            value["annotations"].reverse()
            write_json(annotation_path, value)
            second = make_dataset(annotation_path, root, "images", (2, 2, 3), 19)
            self.assertEqual(first["samples"], second["samples"])
            self.assertEqual(first["provenance"]["eligible_unique_images"], 8)
            self.assertEqual(len({sample["source_sha256"] for sample in first["samples"]}), 7)
            for sample in first["samples"]:
                self.assertTrue(set(sample["positive_category_ids"]).isdisjoint(sample["negative_category_ids"]))
                self.assertEqual(len(sample["prompts"]), 2)
            with self.assertRaisesRegex(ValueError, "not enough"):
                make_dataset(annotation_path, root, "images", (3, 3, 3), 19)

    def test_source_leakage(self):
        for field in ("source_group", "source_sha256", "image"):
            with self.subTest(field=field):
                value = dataset()
                value["samples"][1][field] = value["samples"][0][field]
                with self.assertRaisesRegex(ValueError, "leakage"):
                    validate_dataset(value)
        value = dataset()
        duplicate = copy.deepcopy(value["samples"][0])
        duplicate.update(id="crop", image="crop.ppm", source_sha256="c" * 64, split="selection")
        value["samples"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_dataset(value)

    def test_regression_exclusion_is_explicit(self):
        value = dataset()
        with self.assertRaisesRegex(ValueError, "regression"):
            validate_dataset(value, {"a" * 64})
        self.assertIs(validate_dataset(value, {"a" * 64}, diagnostic=True), value)
        self.assertIs(validate_dataset(value, {"b" * 64}), value)

    def test_manifest_boundaries(self):
        for patch in ({"id": "../escape"}, {"image": "../outside.ppm"}, {"source_sha256": "g" * 64},
                      {"prompts": ["car", "car"]}, {"prompts": [""]}, {"split": "train"},
                      {"transform": {"crop": [1, 2, 3, 4]}}):
            with self.subTest(patch=patch):
                value = dataset()
                value["samples"][0].update(patch)
                with self.assertRaises(ValueError):
                    validate_dataset(value)

    def test_all_split_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = dataset()
            for sample in value["samples"]:
                path = root / sample["image"]
                path.write_bytes(sample["id"].encode())
                sample["source_sha256"] = sha256_file(path)
            write_json(root / "dataset.json", value)
            self.assertEqual(load_dataset(root / "dataset.json", root), value)
            (root / "b.ppm").write_bytes(b"replaced holdout")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_dataset(root / "dataset.json", root)

    def test_exact_moments_and_bounded_chunk_independent_reservoir(self):
        values = np.random.default_rng(4).normal(size=(1001, 7)).astype(np.float32)
        values[9, 2] = 2000  # Rare outlier must survive in the exact extrema.
        whole, chunked = ChannelStats(7, 31, 5), ChannelStats(7, 31, 5)
        whole.update(values)
        for start in range(0, len(values), 17):
            chunked.update(values[start:start + 17])
            self.assertLessEqual(len(chunked.samples), 31)
        np.testing.assert_array_equal(whole.samples, chunked.samples)
        summary = chunked.summary()
        np.testing.assert_array_equal(summary["maximum"], values.max(axis=0))
        np.testing.assert_array_equal(summary["minimum"], values.min(axis=0))
        np.testing.assert_allclose(summary["mean"], values.mean(axis=0, dtype=np.float64), rtol=1e-12)
        np.testing.assert_allclose(summary["rms"], np.sqrt(np.square(values, dtype=np.float64).mean(axis=0)), rtol=1e-12)
        self.assertEqual(summary["token_rows"], 1001)
        self.assertEqual(summary["reservoir_rows"], 31)
        self.assertEqual(summary["absmax"][2], 2000)

    def test_stats_empty_zero_and_invalid_inputs(self):
        stats = ChannelStats(3, 8)
        with self.assertRaisesRegex(ValueError, "empty"):
            stats.summary()
        stats.update(np.empty((0, 3), np.float32))
        stats.update(np.zeros((4, 3), np.float32))
        self.assertEqual(stats.summary()["rms"], [0, 0, 0])
        for invalid in (np.zeros((2, 4), np.float32), np.zeros((2, 3), np.float64),
                        np.full((1, 3), np.nan, np.float32), np.full((1, 3), np.inf, np.float32)):
            with self.assertRaises(ValueError):
                stats.update(invalid)
        self.assertEqual(stats.rows, 4)

    def test_independent_reproducible_layer_seeds(self):
        self.assertEqual(layer_seed(9, "layer-a"), layer_seed(9, "layer-a"))
        self.assertNotEqual(layer_seed(9, "layer-a"), layer_seed(9, "layer-b"))


if __name__ == "__main__":
    unittest.main()
