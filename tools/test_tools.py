#!/usr/bin/env python3
"""Small parser/comparison regressions; no model inference or parity claim."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gguf
import numpy as np

from convert_sam3 import bytes_to_unicode, convert, rename_key, tensor_array
from export_reference import load_case_manifest, load_detector_checkpoint
from prepare_reference_source import prepare
from generate_video_cases import generate as generate_video_cases, recipe_frame
from sam3_artifacts import (SAM3_REVISION, artifact_path, dump_array, read_array,
                           read_json, read_tensor_index)
from sam3_gguf import (canonical_shape, converted_array, inspect_tensors, read_gguf,
                       validate_metadata, write_metadata)
from validate_image import GATES, check_provenance, mask_iou, read_results, tensor_error


class ToolChecks(unittest.TestCase):
    @staticmethod
    def tokenizer_fixture():
        base = list(bytes_to_unicode().values())
        merges = [(left, right) for left in base for right in base][:48894]
        vocab = base + [token + "</w>" for token in base] + [left + right for left, right in merges]
        vocab += ["<start_of_text>", "<end_of_text>"]
        return vocab, merges

    @staticmethod
    def gguf_fixture(path, mutate_metadata=None):
        vocab, merges = ToolChecks.tokenizer_fixture()
        writer = gguf.GGUFWriter(path, "sam3")
        write_metadata(writer, "f16", "a" * 64, vocab, merges)
        if mutate_metadata is not None:
            mutate_metadata(writer)
        values = np.asarray([[1.0003, -2.333], [3.1415927, 4.00001]], dtype=np.float32)
        arrays = {"linear.weight": values, "token_embed.weight": values,
                  "linear.bias": values[0],
                  "singleton.weight": np.asarray([[[5.0003, 6.666, 7.1234], [8.0003, 9.666, 10.1234]]], dtype=np.float32)}
        converted = {name: converted_array(name, array, "f16") for name, array in arrays.items()}
        for name, array in converted.items():
            dtype = gguf.GGMLQuantizationType.F16 if array.itemsize == 2 else gguf.GGMLQuantizationType.F32
            writer.add_tensor_info(name, canonical_shape(array.shape), array.dtype, array.nbytes, raw_dtype=dtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for array in converted.values():
            writer.write_tensor_data(array.reshape(canonical_shape(array.shape)), tensor_endianess=gguf.GGUFEndian.LITTLE)
        writer.close()
        return arrays, vocab, merges

    def test_official_checkpoint_unused_neck_boundary(self):
        import torch

        model = torch.nn.Linear(2, 1)
        neck_key = "detector.backbone.vision_backbone.sam2_convs.0.conv_1x1.bias"
        detector = {"detector.weight": torch.tensor([[3.0, 4.0]]),
                    "detector.bias": torch.tensor([5.0])}
        skipped = load_detector_checkpoint(model, {**detector, neck_key: torch.zeros(256)})
        torch.testing.assert_close(model(torch.tensor([1.0, 2.0])), torch.tensor([16.0]))
        self.assertEqual([item["name"] for item in skipped], [neck_key])
        self.assertTrue(skipped[0]["reason"])
        invalid = [
            {**detector, "detector.unknown": torch.zeros(1)},
            {**detector, neck_key.replace("conv_1x1", "unknown"): torch.zeros(256)},
            {**detector, neck_key: torch.zeros(257)},
            {**detector, neck_key: torch.zeros(256, dtype=torch.int32)},
            {"detector.weight": detector["detector.weight"]},
        ]
        for checkpoint in invalid:
            with self.subTest(keys=list(checkpoint)):
                with self.assertRaises(ValueError):
                    load_detector_checkpoint(model, checkpoint)

    def test_tensor_record_and_precision(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            arrays, vocab, merges = self.gguf_fixture(path)
            reader = read_gguf(path)
            precision, actual_vocab, actual_merges = validate_metadata(reader, "f16", "a" * 64)
            self.assertEqual(actual_vocab, vocab)
            self.assertEqual(actual_merges, [" ".join(pair) for pair in merges])
            self.assertEqual(reader.get_field("sam3.vision.feed_forward_length").contents(), 4736)
            expected = {name: list(reversed(canonical_shape(array.shape))) for name, array in arrays.items()}
            original_shapes = {name: list(array.shape) for name, array in arrays.items()}
            inventory = inspect_tensors(reader, precision, expected, original_shapes)
            for tensor, item in zip(reader.tensors, inventory):
                np.testing.assert_array_equal(tensor.data.reshape(-1), arrays[tensor.name].astype(tensor.data.dtype).reshape(-1))
                self.assertEqual(item["offset"] % 32, 0)
            self.assertEqual([item["dtype"] for item in inventory], ["float16", "float32", "float32", "float16"])
            self.assertEqual(inventory[-1]["ggml_shape"], [3, 2])
            with self.assertRaisesRegex(ValueError, "original shape"):
                inspect_tensors(reader, precision, expected, {**original_shapes, "linear.weight": [3, 2]})
        for invalid in (1e10, float("nan"), float("inf")):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    converted_array("linear.weight", np.asarray([[invalid]], dtype=np.float32), "f16")
        self.assertEqual(converted_array("linear.weight", np.ones((2, 2), dtype=np.float32), "f32").dtype, np.dtype("<f4"))
        phases = torch.tensor([[1 + 2j, 3 + 4j]], dtype=torch.complex64)
        np.testing.assert_array_equal(tensor_array("vit.blocks.0.attn.freqs_cis", phases), [[[1, 2], [3, 4]]])
        with self.assertRaises(ValueError):
            tensor_array("linear.weight", phases)

    def test_case_ids_rejected_before_model_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = {"id": "safe-case", "prompt": "truck", "score_threshold": 0.5,
                    "source_sha256": "0" * 64}
            frozen = {"schema_version": 1, "sam3_revision": SAM3_REVISION,
                      "acceptance": GATES, "cases": [case]}
            reference = {**frozen, "reference_kind": "supplementary-converted-weights",
                         "eligible_for_milestone": False,
                         "supplementary_weights": {"restored_checkpoint": {"sha256": "a" * 64}}}
            cases_path = root / "cases.json"
            reference_path = root / "manifest.json"
            model_path = root / "missing-model.gguf"
            for collection in ("frozen", "reference"):
                for ids in (["../escape"], [str(root / "escape")], ["duplicate", "duplicate"]):
                    with self.subTest(collection=collection, ids=ids):
                        bad_cases = [{**case, "id": case_id} for case_id in ids]
                        cases_path.write_text(json.dumps({**frozen, "cases": bad_cases if collection == "frozen" else [case]}))
                        reference_path.write_text(json.dumps({**reference, "cases": bad_cases if collection == "reference" else [case]}))
                        with self.assertRaisesRegex(ValueError, "supplementary references cannot"):
                            check_provenance(model_path, root, cases_path)
                        with self.assertRaisesRegex(ValueError, "invalid or duplicate case ID"):
                            check_provenance(model_path, root, cases_path, allow_supplementary=True)
            cases_path.write_text(json.dumps(frozen))
            self.assertEqual(load_case_manifest(cases_path)["cases"], [case])
            self.assertFalse(model_path.exists())
            self.assertFalse((root / "escape.log").exists())

    def test_incompatible_vector_rejected_before_publication(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            bpe = root / "fixture.bpe"
            bpe.write_bytes(b"disposable tokenizer fixture")
            output = root / "vector.gguf"
            schema = {"schema_version": 1, "sam3_revision": SAM3_REVISION,
                      "tensors": {"ddec.norm.bias": [2]}, "unused_tracker_tensors": {}}
            vocab, merges = self.tokenizer_fixture()
            torch.save({"detector.transformer.decoder.norm.bias": torch.tensor([[1.0003, -2.333]])}, checkpoint)
            with patch("convert_sam3.load_tokenizer", return_value=(vocab, merges)), \
                    patch("sam3_artifacts.read_json", return_value=schema):
                with self.assertRaisesRegex(ValueError, "canonical SAM precision"):
                    convert(checkpoint, bpe, "f16", output)
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix(".gguf.manifest.json").exists())
                self.assertFalse(list(root.glob(".sam3-*")))
                torch.save({"detector.transformer.decoder.norm.bias": torch.tensor([1.0003, -2.333])}, checkpoint)
                manifest = convert(checkpoint, bpe, "f16", output)
            self.assertEqual(manifest["tensors"][0]["dtype"], "float32")
            np.testing.assert_array_equal(read_gguf(output).tensors[0].data, [np.float32(1.0003), np.float32(-2.333)])

    def test_video_conversion_preserves_image_subset_and_is_exclusive(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, bpe = root / "checkpoint.pt", root / "fixture.bpe"
            bpe.write_bytes(b"disposable tokenizer fixture")
            schema = {"schema_version": 1, "sam3_revision": SAM3_REVISION,
                      "tensors": {"ddec.norm.bias": [2]},
                      "unused_tracker_tensors": {"no_mem_embed": [2]}}
            torch.save({"detector.transformer.decoder.norm.bias": torch.tensor([1.0003, -2.333]),
                        "tracker.no_mem_embed": torch.tensor([[[3.14159, -4.0001]]])}, checkpoint)
            vocab, merges = self.tokenizer_fixture()
            with patch("convert_sam3.load_tokenizer", return_value=(vocab, merges)), \
                    patch("sam3_artifacts.read_json", return_value=schema):
                image = convert(checkpoint, bpe, "f16", root / "image.gguf")
                video = convert(checkpoint, bpe, "f16", root / "video.gguf", "video")
                with self.assertRaises(FileExistsError):
                    convert(checkpoint, bpe, "f16", root / "video.gguf", "video")
            self.assertEqual(video["sam_schema_version"], 2)
            self.assertEqual(video["task"], "video")
            self.assertEqual(len(video["tensors"]), 2)
            self.assertEqual(video["skipped"], [])
            self.assertEqual(len(image["skipped"]), 1)
            image_entry = image["tensors"][0]
            video_entry = next(item for item in video["tensors"] if item["name"] == image_entry["name"])
            for key in ("ggml_shape", "dtype", "sha256", "bytes"):
                self.assertEqual(image_entry[key], video_entry[key])
            reader = read_gguf(root / "video.gguf")
            validate_metadata(reader, "f16", video["checkpoint"]["sha256"], "video")
            with self.assertRaises(ValueError):
                validate_metadata(reader, "f16", video["checkpoint"]["sha256"])
            self.assertEqual(reader.get_field("sam3.tracker.memory_storage").contents(), "bf16")
            self.assertEqual(rename_key("tracker.no_mem_embed", "video"), ("no_mem_embed", None))
            self.assertEqual(rename_key("detector.backbone.vision_backbone.sam2_convs.0.conv_1x1.bias", "video"),
                             ("neck.trk.0.conv_1x1.bias", None))
            with self.assertRaises(ValueError):
                rename_key("tracker.no_mem_embed", "unknown")

    def test_video_metadata_rejects_invalid_profile_and_storage(self):
        vocab, merges = self.tokenizer_fixture()
        mutations = [
            lambda writer: writer.add_string("sam.task", "text_image"),
            lambda writer: writer.add_uint32("sam.schema_version", 1),
            lambda writer: writer.add_uint32("sam3.tracker.attention.head_length", 64),
            lambda writer: writer.add_string("sam3.tracker.memory_storage", "f32"),
            lambda writer: writer.add_string("sam3.tracker.policy", "unknown"),
            lambda writer: writer.add_int32("sam3.tracker.memory_length", 64),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for index, mutate in enumerate(mutations):
                path = Path(temporary) / f"mutated-{index}.gguf"
                writer = gguf.GGUFWriter(path, "sam3")
                write_metadata(writer, "f32", "a" * 64, vocab, merges, "video")
                mutate(writer)
                writer.add_tensor("probe.weight", np.zeros((2, 2), dtype=np.float32))
                writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
                with self.subTest(index=index), self.assertRaises(ValueError):
                    validate_metadata(read_gguf(path), "f32", "a" * 64, "video")

    def test_video_recipe_boundaries_and_generation_status(self):
        from PIL import Image
        from sam3_artifacts import sha256_file

        source = Image.new("RGB", (1800, 1200), (220, 40, 10))
        self.assertEqual(recipe_frame("motion", source, 12).tobytes(), source.tobytes())
        entry_start, entry_late = recipe_frame("entry", source, 0), recipe_frame("entry", source, 30)
        self.assertEqual(entry_start.getpixel((1799, 600)), (127, 127, 127))
        self.assertEqual(entry_late.getpixel((1799, 600)), (220, 40, 10))
        occluded = recipe_frame("occlusion", source, 20)
        for point in ((64, 256), (1743, 919)):
            self.assertEqual(occluded.getpixel(point), (127, 127, 127))
        for point in ((63, 256), (1744, 919), (100, 920)):
            self.assertEqual(occluded.getpixel(point), (220, 40, 10))
        self.assertEqual(recipe_frame("occlusion", source, 24).tobytes(), source.tobytes())
        self.assertEqual(recipe_frame("hotstart-removal", source, 1).getpixel((0, 0)), (220, 40, 10))
        self.assertEqual(recipe_frame("hotstart-removal", source, 2).getpixel((0, 0)), (127, 127, 127))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "fixture.png"
            Image.new("RGB", (7, 3), (220, 40, 10)).save(path)
            frozen = {"sam3_revision": SAM3_REVISION, "pillow_version": "11.2.1", "cases": [
                {"id": "negative", "frames": 2, "source": path.name, "source_sha256": sha256_file(path),
                 "prompt": "purple elephant"}]}
            with patch("generate_video_cases.read_json", return_value=frozen):
                output = root / "generated"
                generate_video_cases(root, output)
                manifest = read_json(output / "manifest.json")
                self.assertTrue(manifest["complete"])
                self.assertFalse(manifest["eligible_for_milestone"])
                self.assertFalse(manifest["reference_behavior_verified"])
                frames = manifest["cases"][0]["frames_manifest"]
                self.assertEqual([item["index"] for item in frames], [0, 1])
                for item in frames:
                    self.assertEqual(sha256_file(output / item["file"]), item["sha256"])
                with self.assertRaises(FileExistsError):
                    generate_video_cases(root, output)
                frozen["cases"][0]["source_sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    generate_video_cases(root, root / "bad-output")
                self.assertFalse((root / "bad-output").exists())

    def test_reference_source_output_does_not_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "sam3").mkdir(parents=True)
            (source / "sam3/example.py").write_text("fixture = 1\n")
            (source / "LICENSE").write_text("disposable fixture license\n")
            alias = root / "source-alias"
            alias.symlink_to(source / "sam3", target_is_directory=True)
            original = sorted(path.relative_to(source).as_posix() for path in source.rglob("*"))
            with patch("prepare_reference_source.validate_source", return_value=(source.resolve(), source.resolve(), [])):
                for output in (source / "sam3/new-parent/runtime", alias / "symlink-parent/runtime"):
                    with self.subTest(output=str(output)):
                        with patch("prepare_reference_source.shutil.copytree", side_effect=AssertionError("copy must not begin")):
                            with self.assertRaisesRegex(ValueError, "overlap"):
                                prepare(source, output)
                        self.assertEqual(sorted(path.relative_to(source).as_posix() for path in source.rglob("*")), original)
                output = root / "outside-parent/runtime"
                with patch("prepare_reference_source.PATCHES", []):
                    prepare(source, output)
                self.assertEqual((output / "sam3/example.py").read_bytes(), (source / "sam3/example.py").read_bytes())
                self.assertEqual(sorted(path.relative_to(source).as_posix() for path in source.rglob("*")), original)

    def test_gguf_rejection_and_output_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "model.gguf"
            self.gguf_fixture(path)
            original = path.read_bytes()
            path.write_bytes(original + b"unexpected")
            with self.assertRaisesRegex(ValueError, "trailing data"):
                read_gguf(path)
            path.write_bytes(original[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                read_gguf(path)
            bad_metadata = [
                lambda writer: writer.add_int32("sam3.vision.feed_forward_length", 4736),
                lambda writer: writer.add_uint32("sam3.vision.feed_forward_length", 4625),
                lambda writer: writer.add_array("sam3.vision.global_attention_blocks", [7, 15, 23, 31]),
                lambda writer: writer.add_string("sam.task", "text_image\0suffix"),
            ]
            for mutate in bad_metadata:
                with self.subTest(mutation=bad_metadata.index(mutate)):
                    self.gguf_fixture(path, mutate)
                    with self.assertRaises(ValueError):
                        validate_metadata(read_gguf(path), "f16", "a" * 64)
            path.write_bytes(original)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                convert(root / "missing.pt", root / "missing.bpe", "f16", path)
            self.assertEqual(path.read_bytes(), original)
            path.unlink()
            sidecar = path.with_suffix(".gguf.manifest.json")
            sidecar.write_text("retain this manifest")
            with self.assertRaises(FileExistsError):
                convert(root / "missing.pt", root / "missing.bpe", "f16", path)
            self.assertFalse(path.exists())
            self.assertEqual(sidecar.read_text(), "retain this manifest")
            legacy = root / "legacy.ggml"
            legacy.write_bytes(b"3mas" + bytes(20))
            with self.assertRaisesRegex(ValueError, "reconvert"):
                read_gguf(legacy)

    def test_strict_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = dump_array(root, "tensor", np.asarray([1, -2], dtype=np.float32), "C")
            np.testing.assert_array_equal(read_array(root, metadata), [1, -2])
            metadata["shape"] = [1]
            with self.assertRaises(ValueError):
                read_array(root, metadata)
            with self.assertRaises(ValueError):
                artifact_path(root, "../outside.bin")
            (root / "bad.json").write_text('{"score":NaN}')
            with self.assertRaises(ValueError):
                read_json(root / "bad.json")
            (root / "bad.json").write_text('{"score":1,"score":2}')
            with self.assertRaises(ValueError):
                read_json(root / "bad.json")
            (root / "tensors.json").write_text(json.dumps({"schema_version": 1, "byte_order": "little", "token_ids": [0] * 32, "tensors": {}}))
            with self.assertRaisesRegex(ValueError, "missing required tensors"):
                read_tensor_index(root)

    def test_comparison_boundaries(self):
        reference = np.asarray([3, 4], dtype=np.float32)
        self.assertAlmostEqual(tensor_error(reference + [0, 1], reference)["normalized_l2"], 0.2)
        zero = tensor_error(np.asarray([1e-6]), np.asarray([0.0]))
        self.assertIsNone(zero["normalized_l2"])
        self.assertLess(zero["maximum_absolute_error"], GATES["zero_norm_max_abs"])
        self.assertEqual(mask_iou(np.zeros((2, 2)), np.zeros((2, 2))), 1)
        self.assertEqual(mask_iou(np.asarray([1, 0]), np.asarray([1, 1])), 0.5)
        with self.assertRaises(ValueError):
            tensor_error(np.asarray([float("nan")]), np.asarray([1.0]))

    def test_empty_detection_and_fail_closed_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results.json").write_text(json.dumps({"schema_version": 1, "width": 2, "height": 3,
                                                         "prompt": "empty", "score_threshold": 0.5, "detections": []}))
            scores = np.zeros(200)
            _, detections = read_results(root, scores, {"prompt": "empty"}, False)
            self.assertEqual(detections, {})
            scores[0] = 0.51
            with self.assertRaisesRegex(ValueError, "selected queries disagree"):
                read_results(root, scores, {"prompt": "empty"}, False)
            (root / "manifest.json").write_text(json.dumps({"reference_kind": "supplementary-converted-weights", "eligible_for_milestone": False}))
            (root / "cases.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "supplementary references cannot"):
                check_provenance(root / "missing-model.gguf", root, root / "cases.json")

    def test_exact_byte_mapping_and_detector_renaming(self):
        mapping = bytes_to_unicode()
        self.assertEqual(len(mapping), 256)
        self.assertEqual(len(set(mapping.values())), 256)
        self.assertEqual(mapping[0], "\u0100")
        self.assertEqual(mapping[32], "\u0120")
        self.assertEqual(rename_key("detector.transformer.decoder.layers.0.cross_attn.in_proj_weight")[0], "ddec.layers.0.ca.in_proj_weight")
        self.assertIsNone(rename_key("tracker.no_mem_embed")[0])
        with self.assertRaises(ValueError):
            rename_key("detector.unsupported.weight")


if __name__ == "__main__":
    unittest.main()
