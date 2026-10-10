#!/usr/bin/env python3
"""Small parser/comparison regressions; no model inference or parity claim."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gguf
import numpy as np

from tools.convert.convert_sam3 import (GGML_REVISION, bytes_to_unicode, convert, pinned_gguf_version,
                          quantizer_identity, rename_key, tensor_array)
from tools.validation.export_reference import load_case_manifest, load_detector_checkpoint
from tools.validation.prepare_reference_source import prepare
from tools.benchmark.generate_video_cases import generate as generate_video_cases, recipe_frame
from tools.convert.sam3_artifacts import (SAM3_REVISION, artifact_path, dump_array, read_array,
                           read_json, read_tensor_index)
from tools.convert.sam3_gguf import (canonical_shape, converted_array, inspect_tensors, quantization_profile,
                       quantized_tensor_type, read_gguf, validate_metadata, write_metadata)
from tests.tools.test_quantization_tools import QuantizationChecks
from tests.tools.test_modular_quantization_tools import ModularQuantizationChecks
from tests.tools.test_calibration import CalibrationChecks
from tests.tools.test_runtime_quantization import RuntimeQuantizationChecks
from tests.tools.test_cache_quantization import CacheQuantizationChecks
from tests.tools.test_coco_screening import CocoScreeningChecks
from tests.tools.test_precision_acceptance import PrecisionAcceptanceChecks
from tests.tools.test_coco_acceptance import CocoAcceptanceChecks
from tests.tools.test_precision_dataset import PrecisionDatasetChecks
from tests.tools.test_precision_artifacts import PrecisionArtifactChecks
from tests.tools.test_precision_performance import PrecisionPerformanceChecks
from tests.tools.test_precision_regression import PrecisionRegressionChecks
from tests.tools.test_precision_v3 import PrecisionV3Checks, PrecisionV3ArtifactChecks
from tests.tools.test_quantization_benchmark import QuantizationBenchmarkChecks
from tests.tools.test_quantization_performance import QuantizationPerformanceChecks
from tests.tools.test_verify_precision_q8_cache import Q8CacheArithmeticChecks
from tests.tools.test_verify_int8_dots import ExactInt8DotChecks
from tests.tools.test_verify_fp8_dots import FP8DotChecks
from tests.tools.test_verify_q8_rhs_staging import Q8RhsStagingChecks
from tests.tools.test_verify_q8_mmvq_dots import Q8MmvqDotChecks
from tests.tools.test_verify_q8_mmq_rhs_staging import Q8MmqRhsStagingChecks
from tests.tools.test_verify_q8_mmq_dots import Q8MmqDotChecks
from tests.tools.test_reconcile_precision_cache_arithmetic import FrozenCacheAlignmentChecks
from tests.tools.test_tool_entrypoints import ToolEntrypointChecks
from tests.tools.test_video_benchmark import VideoBenchmarkChecks
from tests.tools.test_documentation import DocumentationChecks
from tools.validation.validate_image import GATES, check_provenance, mask_iou, read_results, tensor_error


class ToolChecks(unittest.TestCase):
    def test_process_timing_preserves_platform_rss_units(self):
        from tools.benchmark.benchmark_video import process_timing

        linux = process_timing("output\nSAM_TIME 2048 1.25 0.50 0.10\n", "linux")
        macos = process_timing("1.25 real 0.50 user 0.10 sys\n2097152 maximum resident set size\n", "darwin")
        self.assertEqual(linux["peak_rss_bytes"], 2097152)
        self.assertEqual(macos["peak_rss_bytes"], linux["peak_rss_bytes"])
        self.assertEqual(linux["wall_seconds"], macos["wall_seconds"])
        with self.assertRaises(ValueError):
            process_timing("SAM_TIME missing\n", "linux")

    def test_cuda_oracle_requires_math_and_device_provenance(self):
        from tools.convert.sam3_artifacts import validate_cuda_oracle_provenance

        validate_cuda_oracle_provenance({"device": "cpu"})
        oracle = {"device": "cuda", "sdpa_backend": "math", "cuda_runtime": "12.8",
                  "device_name": "NVIDIA test GPU", "cublas_workspace_config": ":4096:8",
                  "sdpa_adaptation": {"replacement": "explicit math attention"}}
        validate_cuda_oracle_provenance(oracle)
        for name in ("sdpa_backend", "cuda_runtime", "device_name", "cublas_workspace_config", "sdpa_adaptation"):
            incomplete = {key: value for key, value in oracle.items() if key != name}
            with self.assertRaisesRegex(ValueError, "CUDA oracle"):
                validate_cuda_oracle_provenance(incomplete)

    def test_visual_comparison_uses_actual_masks_and_rejects_altered_receipts(self):
        from PIL import Image
        from tools.visualization.render_image_comparison import render, panel
        from tools.convert.sam3_artifacts import sha256_file, write_json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, actual = root / "reference", root / "actual"
            for directory in (reference / "sample", actual / "sample"):
                directory.mkdir(parents=True)
            Image.new("RGB", (2, 2), "white").save(reference / "sample/input.ppm")
            case = {"id": "sample", "directory": "sample", "input": "input.ppm",
                    "prompt": "object", "score_threshold": 0.2}
            masks = [np.asarray([[1, 1], [0, 0]], dtype=np.uint8),
                     np.asarray([[1, 0], [0, 0]], dtype=np.uint8)]
            for directory, mask in zip((reference / "sample", actual / "sample"), masks):
                metadata = dump_array(directory, "mask-0", mask)
                result = {"width": 2, "height": 2, "prompt": "object", "score_threshold": 0.2,
                          "backend": "cpu", "precision": "q8_0", "storage_profile": "image-vision-linear-q8_0-v1",
                          "detections": [{"query_index": 0, "score": 0.8, "box": [0, 0, 2, 2], "mask": metadata}]}
                if directory.parent == actual:
                    result["detections"][0]["box"][0] = 0.5
                write_json(directory / "results.json", result)
            case.update(input_sha256=sha256_file(reference / "sample/input.ppm"),
                        results_sha256=sha256_file(reference / "sample/results.json"))
            write_json(reference / "manifest.json", {"reference_kind": "supplementary-converted-weights", "cases": [case]})
            write_json(actual / "metrics.json", {"backend": "cpu",
                "reference_manifest_sha256": sha256_file(reference / "manifest.json"),
                "cases": [{"id": "sample", "passed": False, "tensors": {},
                           "output_sha256": {p.name: sha256_file(p) for p in (actual / "sample").iterdir()}}]})
            with patch("tools.visualization.render_image_comparison.panel", wraps=panel) as rendered_panels:
                evidence = render(reference, [("Q8_0", actual)], root / "gallery")
            titles = [call.args[1] for call in rendered_panels.call_args_list]
            self.assertIn("Supplementary reference", titles)
            self.assertNotIn("Meta FP32 reference", titles)
            compared = evidence["cases"][0]["comparisons"][0]
            self.assertEqual(compared["minimum_mask_iou"], 0.5)
            self.assertEqual(compared["detections"][0]["box_dimension_fraction"], 0.25)
            self.assertFalse(compared["output_quality_passed"])
            self.assertTrue((root / "gallery/sample-comparison.jpg").is_file())
            with self.assertRaises(FileExistsError):
                render(reference, [("Q8_0", actual)], root / "gallery")
            (actual / "sample/mask-0.bin").write_bytes(b"\0\0\0\0")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                render(reference, [("Q8_0", actual)], root / "altered")
            self.assertFalse((root / "altered").exists())
            write_json(reference / "manifest.json", {"reference_kind": "test-fixture", "cases": [case, case]})
            with self.assertRaisesRegex(ValueError, "duplicate reference case IDs"):
                render(reference, [("Q8_0", actual)], root / "duplicate")
            self.assertFalse((root / "duplicate").exists())

    def test_private_archive_integrity_and_exclusive_output(self):
        from tools.maintenance.archive_validation import create_bundle, verify_bundle

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "receipt.json").write_text('{"passed": true}')
            output = root / "saved"
            manifest = create_bundle([source], output)
            self.assertEqual(len(verify_bundle(manifest.parent)["files"]), 1)
            with self.assertRaises(ValueError):
                create_bundle([source], source / "nested")
            with self.assertRaises(FileExistsError):
                create_bundle([source], output)
            (manifest.parent / "extra.txt").write_text("unlisted")
            with self.assertRaises(ValueError):
                verify_bundle(manifest.parent)
            (manifest.parent / "extra.txt").unlink()
            (manifest.parent / "external").symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                verify_bundle(manifest.parent)
            (manifest.parent / "external").unlink()
            saved = manifest.read_text()
            altered = json.loads(saved)
            altered["bytes"] += 1
            manifest.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                verify_bundle(manifest.parent)
            manifest.write_text(saved)
            verify_bundle(manifest.parent)
            (manifest.parent / "data/00-source/receipt.json").write_text("corrupted")
            with self.assertRaises(ValueError):
                verify_bundle(manifest.parent)

    def test_bilingual_measurement_table_drift(self):
        from tools.maintenance.check_docs import table_facts

        english = "| Mixed F16/F32 | 38.994 / 4.927 | 5.576 / 2.660 |"
        chinese = "| 混合 F16/F32 | 38.994 / 4.927 | 5.576 / 2.660 |"
        self.assertEqual(table_facts(english), table_facts(chinese))
        self.assertNotEqual(table_facts(english), table_facts(chinese.replace("5.576", "5.575")))

    def test_encoding_timers_keep_historical_missing_data_explicit(self):
        from tools.benchmark.benchmark_video import encoding_timings

        samples = [{"stats": {"runtime": {}}} for _ in range(64)]
        self.assertFalse(encoding_timings(samples)["available"])
        for sample in samples:
            sample["stats"]["runtime"]["image_ms"] = 10
        with self.assertRaises(ValueError):
            encoding_timings(samples)
        for frame, sample in enumerate(samples):
            sample["stats"]["runtime"] = {"image_ms": 1000 if frame < 16 else 10,
                                            "inference_ms": 1000 if frame < 16 else 20}
        timings = encoding_timings(samples)
        self.assertEqual(timings["image_encoding"]["count"], 48)
        self.assertEqual(timings["image_encoding"]["median_ms"], 10)
        self.assertEqual(timings["detector_pipeline"]["median_ms"], 20)
        del samples[-1]["stats"]["runtime"]["image_ms"]
        with self.assertRaises(ValueError):
            encoding_timings(samples)

    def test_qualification_reuse_requires_identical_inputs_and_valid_prefix(self):
        from tools.benchmark.generate_video_benchmark import SOURCE_SHA256
        from tools.benchmark.qualify_video_benchmark import CHECKPOINT_SHA256, reuse_qualification
        from tools.convert.sam3_artifacts import sha256_file, write_json

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("old", "new"):
                frames = root / name / "one-object"
                frames.mkdir(parents=True)
                for frame in range(64):
                    (frames / f"{frame:06d}.png").write_bytes(b"same trusted fixture bytes")
                hashes = [sha256_file(frames / f"{frame:06d}.png") for frame in range(64)]
                fixture = {"complete": True, "frames": 64, "warmup_frames": 16, "measured_frames": 48,
                           "prompt": "truck", "max_objects": 8, "width": 1800, "height": 1200,
                           "pillow_version": "11.2.1", "source_sha256": SOURCE_SHA256,
                           "workloads": [{"id": "one-object", "expected_objects": 1, "directory": "one-object",
                                          "positions_xy_mirror_phase": [[500, 330, False, 0]], "input_sha256": hashes}]}
                write_json(root / name / "fixture.json", fixture)
            source = root / "source.py"
            source.write_text("trusted source")
            parent = {"complete": True, "passed": True, "qualified_for_performance": True,
                      "reference_kind": "official-original-performance-prefix", "checkpoint_sha256": CHECKPOINT_SHA256,
                      "sam3_revision": SAM3_REVISION, "declared_frame_count": 64, "prefix_frames": 17,
                      "workload": "one-object", "expected_objects": 1,
                      "fixture_manifest_sha256": sha256_file(root / "old/fixture.json"),
                      "input_sha256": {str(root / "old/one-object" / f"{frame:06d}.png"): hashes[frame] for frame in range(64)},
                      "source_sha256": {str(source): sha256_file(source)},
                      "records": [{"frame_index": frame, "active_ids": [0], "births": [0] if frame == 0 else [],
                                   "removed": [], "visible": [{"id": 0, "area": 1}]} for frame in range(17)],
                      "emitted": [{"frame_index": frame, "emitted_after_frame": frame + 14} for frame in range(3)]}
            write_json(root / "parent.json", parent)
            result = reuse_qualification(root / "new/fixture.json", "one-object", root / "parent.json",
                                         root / "old/fixture.json", root / "reuse.json")
            self.assertEqual(read_json(result)["reused_qualification"]["parent_sha256"], sha256_file(root / "parent.json"))
            parent["records"][-1]["active_ids"] = []
            write_json(root / "parent.json", parent)
            with self.assertRaises(ValueError):
                reuse_qualification(root / "new/fixture.json", "one-object", root / "parent.json",
                                    root / "old/fixture.json", root / "invalid.json")
            (root / "new/one-object/000063.png").write_bytes(b"different input")
            with self.assertRaises(ValueError):
                reuse_qualification(root / "new/fixture.json", "one-object", root / "parent.json",
                                    root / "old/fixture.json", root / "different.json")

    def test_converter_cli_video_precision_default(self):
        import io
        from tools.convert import convert_sam3
        common = ["convert_sam3.py", "--checkpoint", "source.pt", "--bpe", "bpe.gz", "--output", "output.gguf"]
        for task, explicit, expected in (("video", None, "hybrid"), ("video", "f16", "f16"),
                                         ("video", "f32", "f32"), ("video", "hybrid", "hybrid"),
                                         ("image", "f16", "f16"), ("image", "f32", "f32"),
                                         ("image", "q8_0", "q8_0")):
            with self.subTest(task=task, explicit=explicit):
                arguments = common + ["--task", task] + (["--precision", explicit] if explicit else [])
                with patch("sys.argv", arguments), patch.object(convert_sam3, "convert") as convert_mock:
                    convert_sam3.main()
                    convert_mock.assert_called_once_with(Path("source.pt"), Path("bpe.gz"), expected,
                                                         Path("output.gguf"), task, None, None)
        with patch("sys.argv", common + ["--precision", "q6_k", "--quantizer", "/tmp/sam_quantize_rows"]), \
                patch.object(convert_sam3, "convert") as convert_mock:
            convert_sam3.main()
            convert_mock.assert_called_once_with(Path("source.pt"), Path("bpe.gz"), "q6_k",
                                                 Path("output.gguf"), "image", Path("/tmp/sam_quantize_rows"), None)
        with patch("sys.argv", common + ["--precision", "q8_0", "--storage-profile",
                                           "image-vision-linear-q8_0-v1"]), \
                patch.object(convert_sam3, "convert") as convert_mock:
            convert_sam3.main()
            convert_mock.assert_called_once_with(Path("source.pt"), Path("bpe.gz"), "q8_0",
                                                 Path("output.gguf"), "image", None,
                                                 "image-vision-linear-q8_0-v1")
        for task_arguments in ([], ["--task", "image"]):
            with self.subTest(image_arguments=task_arguments):
                with patch("sys.argv", common + task_arguments), patch.object(convert_sam3, "convert") as convert_mock:
                    with patch("sys.stderr", new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit) as caught:
                        convert_sam3.main()
                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn("--precision", stderr.getvalue())
                    convert_mock.assert_not_called()

    def test_converter_requires_the_locked_gguf_package(self):
        self.assertEqual(pinned_gguf_version(), "0.19.0")
        with patch("tools.convert.convert_sam3.importlib.metadata.version", return_value="0.19.1"):
            with self.assertRaisesRegex(ValueError, "requires pinned gguf==0.19.0"):
                pinned_gguf_version()
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "must-not-publish.gguf"
                with self.assertRaisesRegex(ValueError, "requires pinned gguf==0.19.0"):
                    convert(Path(temporary) / "missing.pt", Path(temporary) / "missing.bpe",
                            "q8_0", output)
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix(".gguf.manifest.json").exists())

    def test_quantizer_identity_hashes_linked_target_and_encoder_library(self):
        from tools.convert.sam3_artifacts import sha256_file
        from tools.validation.validate_image import SAM_LEGACY_PATCHED_GGML_BUILD_COMMIT, SAM_PATCHED_GGML_BUILD_COMMIT
        from tools.convert.sam3_gguf import GGML_VERSION, LEGACY_GGML_QUANTIZER_IDENTITIES

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            linked_library, encoder_library = root / "libggml.dylib", root / "libquantize.so"
            linked_library.write_bytes(b"linked ggml target")
            encoder_library.write_bytes(b"actual quantize_chunk provider")
            identity = {"ggml_revision": GGML_REVISION, "ggml_build_commit": GGML_REVISION,
                        "ggml_version": GGML_VERSION, "quantization_version": 2,
                        "ggml_library_path": str(linked_library),
                        "ggml_quantize_library_path": str(encoder_library)}
            helper = root / "sam_quantize_rows"
            short_dot_patch = Path(__file__).resolve().parents[2] / "cmake/patches/ggml-short-dot-cuda.patch"
            current_build_commit = SAM_PATCHED_GGML_BUILD_COMMIT + "-" + sha256_file(short_dot_patch)[:12]
            for commit in (GGML_REVISION, SAM_LEGACY_PATCHED_GGML_BUILD_COMMIT,
                           SAM_PATCHED_GGML_BUILD_COMMIT, current_build_commit):
                with self.subTest(commit=commit):
                    identity["ggml_build_commit"] = commit
                    helper.write_text("#!/usr/bin/env python3\nprint(" + repr(json.dumps(identity)) + ")\n")
                    helper.chmod(0o755)
                    executable, provenance = quantizer_identity(helper)
                    self.assertEqual(executable, helper.resolve())
                    self.assertEqual(provenance["ggml_revision"], GGML_REVISION)
                    self.assertEqual(provenance["ggml_build_commit"], commit)
                    self.assertEqual(provenance["ggml_version"], GGML_VERSION)
                    self.assertEqual(provenance["ggml_library"]["sha256"], sha256_file(linked_library))
                    self.assertEqual(provenance["ggml_quantize_library"]["sha256"], sha256_file(encoder_library))
            identity["ggml_build_commit"] = current_build_commit + "-unverified"
            helper.write_text("#!/usr/bin/env python3\nprint(" + repr(json.dumps(identity)) + ")\n")
            with self.assertRaisesRegex(ValueError, "pinned GGML"):
                quantizer_identity(helper)
            for revision, version, commit in LEGACY_GGML_QUANTIZER_IDENTITIES:
                identity.update(ggml_revision=revision, ggml_version=version, ggml_build_commit=commit)
                helper.write_text("#!/usr/bin/env python3\nprint(" + repr(json.dumps(identity)) + ")\n")
                with self.subTest(legacy_generator=commit), self.assertRaisesRegex(ValueError, "revision"):
                    quantizer_identity(helper)
            identity.update(ggml_revision=GGML_REVISION, ggml_version=GGML_VERSION)
            helper.write_text("#!/usr/bin/env python3\nprint(" + repr(json.dumps(identity)) + ")\n")
            with self.assertRaisesRegex(ValueError, "pinned GGML"):
                quantizer_identity(helper)

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

    def test_q8_profile_conversion_keeps_logical_dimensions_and_f32_tensors(self):
        import torch

        matrix_name = "vit.blocks.0.attn.qkv.weight"
        vector_name = "ddec.norm.bias"
        matrix = torch.linspace(-1.3, 1.7, 64, dtype=torch.float32).reshape(2, 32)
        vector = torch.tensor([1.0003, -2.333], dtype=torch.float32)
        schema = {"schema_version": 1, "sam3_revision": SAM3_REVISION,
                  "tensors": {matrix_name: [32, 2], vector_name: [2]},
                  "unused_tracker_tensors": {}}
        vocab, merges = self.tokenizer_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, bpe, output = root / "original.pt", root / "fixture.bpe", root / "q8.gguf"
            torch.save({"detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight": matrix,
                        "detector.transformer.decoder.norm.bias": vector}, checkpoint)
            bpe.write_bytes(b"fixture BPE identity is mocked by this test")
            with patch("tools.convert.convert_sam3.load_tokenizer", return_value=(vocab, merges)), \
                    patch("tools.convert.sam3_artifacts.read_json", return_value=schema):
                manifest = convert(checkpoint, bpe, "q8_0", output)
                with self.assertRaises(FileExistsError):
                    convert(checkpoint, bpe, "q8_0", output)

            reader = read_gguf(output)
            precision, actual_vocab, actual_merges = validate_metadata(
                reader, "q8_0", manifest["checkpoint"]["sha256"])
            self.assertEqual(precision, "q8_0")
            self.assertEqual(actual_vocab, vocab)
            self.assertEqual(actual_merges, [" ".join(pair) for pair in merges])
            self.assertEqual(reader.get_field("sam.schema_version").contents(), 3)
            self.assertEqual(reader.get_field("sam.storage_profile").contents(), "image-linear-q8_0-v1")
            self.assertEqual(reader.get_field("general.quantization_version").contents(), 2)
            self.assertEqual(reader.get_field("general.file_type").contents(), 7)
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            packed = tensors[matrix_name]
            self.assertEqual(packed.shape.tolist(), [32, 2])
            self.assertEqual(packed.tensor_type, gguf.GGMLQuantizationType.Q8_0)
            self.assertEqual(packed.data.shape, (2, 34))
            decoded = gguf.quants.dequantize(packed.data, packed.tensor_type)
            self.assertEqual(decoded.shape, (2, 32))
            self.assertTrue(np.isfinite(decoded).all())
            np.testing.assert_allclose(decoded, matrix.numpy(), rtol=0, atol=0.02)
            self.assertEqual(tensors[vector_name].tensor_type, gguf.GGMLQuantizationType.F32)
            np.testing.assert_array_equal(tensors[vector_name].data, vector.numpy())
            inventory = {item["name"]: item for item in manifest["tensors"]}
            self.assertEqual(inventory[matrix_name]["dtype"], "q8_0")
            self.assertEqual(inventory[matrix_name]["output_dtype"], "q8_0")
            self.assertEqual(inventory[matrix_name]["bytes"], 68)
            self.assertEqual(inventory[vector_name]["dtype"], "float32")
            self.assertEqual(manifest["sam_schema_version"], 3)
            self.assertEqual(manifest["storage_profile"], "image-linear-q8_0-v1")
            self.assertEqual(manifest["arithmetic_profile"], "ggml-quantized-native-v1")
            self.assertEqual(quantized_tensor_type("vit.blocks.32.attn.qkv.weight", (2, 32), "q8_0"), None)
            with self.assertRaisesRegex(ValueError, "not divisible"):
                quantized_tensor_type(matrix_name, (2, 31), "q8_0")

    def test_vision_profiles_quantize_only_128_vit_matrices(self):
        schema = read_json(Path(__file__).resolve().parents[2] / "tools/convert/sam3_tensor_schema.json")
        for precision in ("q8_0", "q6_k", "q5_k", "q4_k"):
            with self.subTest(precision=precision):
                legacy = quantization_profile(precision)
                vision_profile = f"image-vision-linear-{precision}-v1"
                vision = quantization_profile(precision, vision_profile)
                self.assertTrue(legacy["quantize_text_linear"])
                self.assertFalse(vision["quantize_text_linear"])
                self.assertEqual(legacy["profile_status"], "diagnostic")
                self.assertEqual(vision["storage_profile"], f"image-vision-linear-{precision}-v1")

                legacy_types, vision_types = {}, {}
                for name, dimensions in schema["tensors"].items():
                    shape = list(reversed(dimensions))
                    legacy_types[name] = quantized_tensor_type(name, shape, precision)
                    vision_types[name] = quantized_tensor_type(name, shape, precision, vision["storage_profile"])
                self.assertEqual(sum(qtype is not None for qtype in legacy_types.values()), 224)
                self.assertEqual(sum(qtype is not None for qtype in vision_types.values()), 128)
                self.assertEqual(len(vision_types), 1133)
                self.assertEqual(sum(qtype is None for qtype in vision_types.values()), 1005)
                text_weights = {name for name, qtype in legacy_types.items()
                                if name.startswith("text.blocks.") and qtype is not None}
                self.assertEqual(len(text_weights), 96)
                self.assertTrue(all(vision_types[name] is None for name in text_weights))
                if precision == "q8_0":
                    self.assertEqual(sum(qtype == gguf.GGMLQuantizationType.Q8_0
                                         for qtype in vision_types.values()), 128)
                else:
                    self.assertEqual(sum(qtype == getattr(gguf.GGMLQuantizationType, precision.upper())
                                         for qtype in vision_types.values()), 96)
                    self.assertEqual(sum(qtype == gguf.GGMLQuantizationType.Q8_0
                                         for qtype in vision_types.values()), 32)

    def test_vision_q8_keeps_text_f32_bytes_and_rejects_mismatched_profiles(self):
        import torch

        vit_name = "vit.blocks.0.attn.qkv.weight"
        text_name = "text.blocks.0.attn.in_proj.weight"
        bias_name = "ddec.norm.bias"
        vit = torch.linspace(-1.2, 1.4, 64, dtype=torch.float32).reshape(2, 32)
        text = torch.linspace(0.3, 2.1, 64, dtype=torch.float32).reshape(2, 32)
        bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
        schema = {"schema_version": 1, "sam3_revision": SAM3_REVISION,
                  "tensors": {vit_name: [32, 2], text_name: [32, 2], bias_name: [2]},
                  "unused_tracker_tensors": {}}
        vocab, merges = self.tokenizer_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, bpe = root / "original.pt", root / "fixture.bpe"
            output = root / "vision-q8.gguf"
            torch.save({"detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight": vit,
                        "detector.backbone.language_backbone.encoder.transformer.resblocks.0.attn.in_proj_weight": text,
                        "detector.transformer.decoder.norm.bias": bias}, checkpoint)
            bpe.write_bytes(b"fixture BPE identity is mocked by this test")
            selected_profile = "image-vision-linear-q8_0-v1"
            with patch("tools.convert.convert_sam3.load_tokenizer", return_value=(vocab, merges)), \
                    patch("tools.convert.sam3_artifacts.read_json", return_value=schema):
                manifest = convert(checkpoint, bpe, "q8_0", output, "image", None, selected_profile)
                for bad_precision, bad_task, bad_profile in (
                        ("q8_0", "image", "image-vision-linear-q6_k-v1"),
                        ("q8_0", "video", selected_profile),
                        ("f16", "image", selected_profile)):
                    bad_output = root / f"bad-{bad_precision}-{bad_task}.gguf"
                    with self.subTest(precision=bad_precision, task=bad_task, profile=bad_profile), \
                            self.assertRaises(ValueError):
                        convert(checkpoint, bpe, bad_precision, bad_output, bad_task, None, bad_profile)
                    self.assertFalse(bad_output.exists())

            reader = read_gguf(output)
            self.assertEqual(validate_metadata(reader, "q8_0", manifest["checkpoint"]["sha256"],
                                               "image", selected_profile)[0], "q8_0")
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(tensors[vit_name].tensor_type, gguf.GGMLQuantizationType.Q8_0)
            self.assertEqual(tensors[text_name].tensor_type, gguf.GGMLQuantizationType.F32)
            np.testing.assert_array_equal(tensors[text_name].data, text.numpy())
            inventory = {item["name"]: item for item in manifest["tensors"]}
            self.assertEqual(inventory[text_name]["dtype"], "float32")
            self.assertEqual(inventory[text_name]["output_dtype"], "float32")
            self.assertEqual(inventory[text_name]["bytes"], text.numel() * 4)
            self.assertEqual(inventory[text_name]["conversion"], "preserved-f32")
            self.assertEqual(sum(item["dtype"].startswith("q") for item in manifest["tensors"]), 1)
            self.assertEqual(manifest["storage_profile"], selected_profile)
            self.assertEqual(manifest["profile_status"], "candidate")
            self.assertFalse(manifest["quantization"]["quantize_text_linear"])
            self.assertFalse(manifest["options"]["quantize_text_linear"])

    def test_k_profiles_use_exact_matrix_rules_and_decode_the_trailing_q6_scale(self):
        for precision, expected_type in (("q6_k", gguf.GGMLQuantizationType.Q6_K),
                                         ("q5_k", gguf.GGMLQuantizationType.Q5_K),
                                         ("q4_k", gguf.GGMLQuantizationType.Q4_K)):
            self.assertEqual(quantized_tensor_type("vit.blocks.0.attn.qkv.weight", (4, 256), precision),
                             expected_type)
            self.assertEqual(quantized_tensor_type("vit.blocks.0.mlp.lin2.weight", (1024, 4736), precision),
                             gguf.GGMLQuantizationType.Q8_0)
            self.assertIsNone(quantized_tensor_type("vit.blocks.0.attn.qkv.bias", (256,), precision))
            self.assertIsNone(quantized_tensor_type("text.blocks.24.attn.in_proj.weight", (256, 256), precision))
            with self.assertRaisesRegex(ValueError, "fallback is restricted"):
                quantized_tensor_type("vit.blocks.0.attn.qkv.weight", (4, 240), precision)

        q6 = np.zeros((1, 210), dtype=np.uint8)
        q6[0, 0:2] = [0, 0x7c]  # ql data resembles a non-finite F16 value, not a scale.
        from tools.convert.sam3_gguf import validate_quantized_payload
        validate_quantized_payload("q6-probe", q6, gguf.GGMLQuantizationType.Q6_K)
        q6[0, 208:210] = [0, 0x7c]  # Q6_K's actual `d` is the trailing F16 at bytes 208..209.
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_quantized_payload("q6-probe", q6, gguf.GGMLQuantizationType.Q6_K)

    def test_k_gguf_profiles_retain_exact_types_and_q8_row_fallback(self):
        profile_specs = (("q6_k", gguf.GGMLQuantizationType.Q6_K, 18),
                         ("q5_k", gguf.GGMLQuantizationType.Q5_K, 16),
                         ("q4_k", gguf.GGMLQuantizationType.Q4_K, 14))
        names = ("vit.blocks.0.attn.qkv.weight", "vit.blocks.0.mlp.lin2.weight")
        original_shapes = {names[0]: [2, 256], names[1]: [2, 4736]}
        expected = {name: list(reversed(canonical_shape(shape))) for name, shape in original_shapes.items()}
        vocab, merges = self.tokenizer_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            for precision, main_type, file_type in profile_specs:
                with self.subTest(precision=precision):
                    path = Path(temporary) / f"{precision}.gguf"
                    writer = gguf.GGUFWriter(path, "sam3")
                    write_metadata(writer, precision, "a" * 64, vocab, merges)
                    for name, shape, qtype in (
                            (names[0], original_shapes[names[0]], main_type),
                            (names[1], original_shapes[names[1]], gguf.GGMLQuantizationType.Q8_0)):
                        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
                        packed = np.zeros((shape[0], shape[1] // block_size * type_size), dtype=np.uint8)
                        writer.add_tensor_info(name, packed.shape, packed.dtype, packed.nbytes, raw_dtype=qtype)
                    writer.write_header_to_file()
                    writer.write_kv_data_to_file()
                    writer.write_ti_data_to_file()
                    for name, qtype in ((names[0], main_type), (names[1], gguf.GGMLQuantizationType.Q8_0)):
                        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
                        shape = original_shapes[name]
                        packed = np.zeros((shape[0], shape[1] // block_size * type_size), dtype=np.uint8)
                        writer.write_tensor_data(packed, tensor_endianess=gguf.GGUFEndian.LITTLE)
                    writer.close()
                    reader = read_gguf(path)
                    self.assertEqual(validate_metadata(reader, precision, "a" * 64)[0], precision)
                    self.assertEqual(reader.get_field("general.file_type").contents(), file_type)
                    inventory = inspect_tensors(reader, precision, expected, original_shapes)
                    self.assertEqual([item["dtype"] for item in inventory],
                                     [precision, "q8_0"])

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
            with patch("tools.convert.convert_sam3.load_tokenizer", return_value=(vocab, merges)), \
                    patch("tools.convert.sam3_artifacts.read_json", return_value=schema):
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
                      "unused_tracker_tensors": {"no_mem_embed": [2],
                                                 "sam_dec.pred_obj_score_head.layers.2.weight": [256]}}
            score_weight = torch.arange(256, dtype=torch.float32).reshape(1, 256) / 1000 + 1.0003
            torch.save({"detector.transformer.decoder.norm.bias": torch.tensor([1.0003, -2.333]),
                        "tracker.no_mem_embed": torch.tensor([[[3.14159, -4.0001]]]),
                        "tracker.sam_mask_decoder.pred_obj_score_head.layers.2.weight": score_weight}, checkpoint)
            vocab, merges = self.tokenizer_fixture()
            with patch("tools.convert.convert_sam3.load_tokenizer", return_value=(vocab, merges)), \
                    patch("tools.convert.sam3_artifacts.read_json", return_value=schema):
                image = convert(checkpoint, bpe, "f16", root / "image.gguf")
                video = convert(checkpoint, bpe, "f16", root / "video.gguf", "video")
                with self.assertRaises(FileExistsError):
                    convert(checkpoint, bpe, "f16", root / "video.gguf", "video")
            self.assertEqual(video["sam_schema_version"], 2)
            self.assertEqual(video["task"], "video")
            self.assertEqual(len(video["tensors"]), 3)
            self.assertEqual(video["skipped"], [])
            self.assertEqual(len(image["skipped"]), 2)
            score_entry = next(item for item in video["tensors"]
                               if item["name"] == "sam_dec.pred_obj_score_head.layers.2.weight")
            self.assertEqual(score_entry["shape"], [1, 256])
            self.assertEqual(score_entry["ggml_shape"], [256])
            self.assertEqual(score_entry["dtype"], "float32")
            score_tensor = next(item for item in read_gguf(root / "video.gguf").tensors
                                if item.name == score_entry["name"])
            np.testing.assert_array_equal(score_tensor.data.reshape(1, 256), score_weight.numpy())
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

    def test_hybrid_profile_retains_original_values_and_rejects_misdeclarations(self):
        import torch

        values = torch.tensor([[1.0003, -2.333], [3.1415927, 4.00001]])
        keys = ["detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
                "detector.backbone.vision_backbone.sam2_convs.2.conv_1x1.weight",
                "tracker.obj_ptr_proj.layers.0.weight",
                "detector.backbone.vision_backbone.convs.2.conv_1x1.weight",
                "detector.backbone.language_backbone.resizer.weight"]
        names = [rename_key(key, "video")[0] for key in keys]
        expected = {name: [2, 2] for name in names}
        vocab, merges = self.tokenizer_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, bpe = root / "original.pt", root / "fixture.bpe"
            torch.save({key: values.clone() for key in keys}, checkpoint)
            bpe.write_bytes(b"disposable BPE fixture")
            with patch("tools.convert.convert_sam3.load_tokenizer", return_value=(vocab, merges)), \
                    patch("tools.convert.convert_sam3.tensor_schema", return_value=expected):
                baseline = convert(checkpoint, bpe, "f16", root / "f16.gguf", "video")
                hybrid = convert(checkpoint, bpe, "hybrid", root / "hybrid.gguf", "video")
            reader = read_gguf(root / "hybrid.gguf")
            self.assertEqual(validate_metadata(reader, "hybrid", hybrid["checkpoint"]["sha256"], "video")[0], "hybrid")
            self.assertEqual(hybrid["storage_profile"], "visual-tracker-f32-v1")
            original_shapes = {name: [2, 2] for name in names}
            inspect_tensors(reader, "hybrid", expected, original_shapes)
            old = {item["name"]: item for item in baseline["tensors"]}
            for tensor in reader.tensors:
                if tensor.name in names[:3]:
                    self.assertEqual(tensor.tensor_type, gguf.GGMLQuantizationType.F32)
                    np.testing.assert_array_equal(tensor.data, values.numpy())
                else:
                    item = next(item for item in hybrid["tensors"] if item["name"] == tensor.name)
                    self.assertEqual(item["sha256"], old[tensor.name]["sha256"])
                    self.assertEqual(item["dtype"], "float16")
            with self.assertRaises(ValueError):
                inspect_tensors(reader, "f16", expected, original_shapes)
            with self.assertRaises(ValueError):
                validate_metadata(reader, "f16", hybrid["checkpoint"]["sha256"], "video")
            for index, declaration in enumerate(("unknown", 7)):
                path = root / f"invalid-profile-{index}.gguf"
                writer = gguf.GGUFWriter(path, "sam3")
                write_metadata(writer, "hybrid", "a" * 64, vocab, merges, "video")
                if isinstance(declaration, str):
                    writer.add_string("sam.storage_profile", declaration)
                else:
                    writer.add_uint32("sam.storage_profile", declaration)
                writer.add_tensor("probe.weight", values.numpy())
                writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
                with self.assertRaises(ValueError):
                    validate_metadata(read_gguf(path), task="video")
            with self.assertRaises(ValueError):
                write_metadata(gguf.GGUFWriter(root / "invalid-image.gguf", "sam3"),
                               "hybrid", "a" * 64, vocab, merges, "image")

    def test_video_recipe_boundaries_and_generation_status(self):
        from PIL import Image
        from tools.convert.sam3_artifacts import sha256_file

        source = Image.new("RGB", (1800, 1200), (220, 40, 10))
        self.assertEqual(recipe_frame("motion", source, 12).tobytes(), source.tobytes())
        entry_start, entry_late = recipe_frame("entry", source, 15), recipe_frame("entry", source, 16)
        self.assertEqual(entry_start.getpixel((1799, 600)), (127, 127, 127))
        self.assertEqual(entry_late.getpixel((1799, 600)), (220, 40, 10))
        asymmetric = source.copy()
        asymmetric.paste((10, 40, 220), (0, 0, 900, 1200))
        mirrored_entry = recipe_frame("entry", asymmetric, 16)
        self.assertEqual(mirrored_entry.getpixel((1000, 600)), (220, 40, 10))
        self.assertEqual(mirrored_entry.getpixel((1799, 600)), (10, 40, 220))
        occluded = recipe_frame("occlusion", source, 20)
        for point in ((64, 256), (1743, 919)):
            self.assertEqual(occluded.getpixel(point), (127, 127, 127))
        for point in ((63, 256), (1744, 919), (100, 920)):
            self.assertEqual(occluded.getpixel(point), (220, 40, 10))
        self.assertEqual(recipe_frame("occlusion", source, 24).tobytes(), source.tobytes())
        self.assertEqual(recipe_frame("hotstart-removal", source, 1).getpixel((0, 0)), (220, 40, 10))
        hotstart = recipe_frame("hotstart-removal", source, 2)
        self.assertEqual(hotstart.getpixel((64, 256)), (127, 127, 127))
        self.assertEqual(hotstart.getpixel((1743, 759)), (127, 127, 127))
        self.assertEqual(hotstart.getpixel((1744, 759)), (220, 40, 10))
        self.assertEqual(hotstart.getpixel((100, 760)), (220, 40, 10))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "fixture.png"
            Image.new("RGB", (7, 3), (220, 40, 10)).save(path)
            frozen = {"sam3_revision": SAM3_REVISION, "pillow_version": "11.2.1", "cases": [
                {"id": "negative", "frames": 2, "source": path.name, "source_sha256": sha256_file(path),
                 "prompt": "purple elephant"}]}
            with patch("tools.benchmark.generate_video_cases.read_json", return_value=frozen):
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
            with patch("tools.validation.prepare_reference_source.validate_source", return_value=(source.resolve(), source.resolve(), [])):
                for output in (source / "sam3/new-parent/runtime", alias / "symlink-parent/runtime"):
                    with self.subTest(output=str(output)):
                        with patch("tools.validation.prepare_reference_source.shutil.copytree", side_effect=AssertionError("copy must not begin")):
                            with self.assertRaisesRegex(ValueError, "overlap"):
                                prepare(source, output)
                        self.assertEqual(sorted(path.relative_to(source).as_posix() for path in source.rglob("*")), original)
                output = root / "outside-parent/runtime"
                with patch("tools.validation.prepare_reference_source.PATCHES", []):
                    prepare(source, output)
                self.assertEqual((output / "sam3/example.py").read_bytes(), (source / "sam3/example.py").read_bytes())
                self.assertEqual(sorted(path.relative_to(source).as_posix() for path in source.rglob("*")), original)

    def test_cuda_reference_preserves_device_cache_precomputation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "sam3/model").mkdir(parents=True)
            (source / "LICENSE").write_text("disposable fixture license\n")
            original = 'cache = make_cache(device="cuda")\n'
            adapted = original.replace('device="cuda"', 'device="cpu"')
            patches = []
            for name in ("position_encoding.py", "decoder.py"):
                filename = "sam3/model/" + name
                (source / filename).write_text(original)
                patches.append({"file": filename, "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                                "adapted_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
                                "replace": [original, adapted], "reason": "CPU fixture cache"})
            with patch("tools.validation.prepare_reference_source.validate_source", return_value=(source.resolve(), source.resolve(), [])), \
                    patch("tools.validation.prepare_reference_source.PATCHES", patches):
                for device in ("cpu", "cuda"):
                    output = root / device
                    prepare(source, output, device=device)
                    for change in patches:
                        self.assertEqual((output / change["file"]).read_text(), adapted if device == "cpu" else original)
                        self.assertEqual((source / change["file"]).read_text(), original)
                    self.assertEqual(len(read_json(output / "adaptations.json")["changes"]), 2 if device == "cpu" else 0)

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

    def test_video_propagation_trace_serializes_original_ids_and_ties(self):
        import torch
        from tools.validation.export_video_reference import Capture

        capture = Capture.__new__(Capture)
        capture.ids = [np.int64(4), np.int64(9)]
        capture.propagation = []
        capture.candidate_iou = torch.tensor([[0.99, 0.7, 0.7, 0.4], [0.99, 0.4, 0.6, 0.8]])
        capture.record_propagation(capture.candidate_iou[:, 1:])
        restored = json.loads(json.dumps(capture.propagation, allow_nan=False))
        self.assertEqual([item["id"] for item in restored], [4, 9])
        self.assertEqual([item["mask_index"] for item in restored], [1, 3])
        self.assertEqual([item["pointer_index"] for item in restored], [1, 3])
        self.assertEqual([len(item["iou"]) for item in restored], [4, 4])
        with self.assertRaisesRegex(ValueError, "batch"):
            capture.record_propagation(capture.candidate_iou[:1, 1:])

    def test_validators_reject_model_and_library_replacement_during_execution(self):
        from types import SimpleNamespace
        from tools.validation import validate_image
        from tools.validation import validate_video
        from tools.convert.sam3_artifacts import sha256_file

        for validator, target in ((validate_image, "tests/test_image"), (validate_video, "examples/sam_video")):
            for changed_artifact in ("model", "library"):
                with self.subTest(validator=validator.__name__, changed=changed_artifact), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary); build, reference = root / "build", root / "reference"
                    executable = build / target; executable.parent.mkdir(parents=True)
                    executable.write_bytes(b"disposable executable fixture")
                    library = build / "libggml-base.dylib"; library.write_bytes(b"original library")
                    model = root / "model.gguf"; model.write_bytes(b"original model")
                    manifest = model.with_suffix(".gguf.manifest.json"); manifest.write_text("{}")
                    (reference / "case").mkdir(parents=True)
                    (reference / "manifest.json").write_text("{}")
                    case = {"id": "case", "directory": "case", "input": "input.png", "prompt": "truck"}
                    oracle = {"reference_kind": "official-checkpoint", "eligible_for_milestone": False,
                              "max_objects": 8, "cases": [case]}
                    provenance = {"output": {"sha256": sha256_file(model)}, "checkpoint": {"sha256": "a" * 64}}
                    args = SimpleNamespace(build_dir=build, model=model, reference=reference, cases=root / "cases.json",
                                           output=root / "output", backend="cpu", threads=4, timeout_seconds=1,
                                           allow_supplementary=False, allow_diagnostic=True)
                    checked = ("f16", oracle, provenance) if validator is validate_image else ("f16", oracle, GATES)
                    def replace(*unused, **kwargs):
                        (model if changed_artifact == "model" else library).write_bytes(b"replaced during inference")
                        return SimpleNamespace(returncode=0)
                    with patch.object(validator, "check_provenance", return_value=checked), \
                            patch.object(validator, "compare_case", return_value={"id": "case", "passed": True}) as compared, \
                            patch.object(validator.subprocess, "run", side_effect=replace), \
                            patch.object(validate_video, "MODEL_HASHES", {"f16": sha256_file(model)}):
                        self.assertEqual(validator.validate(args), 1)
                        compared.assert_not_called()
                    report = read_json(args.output / "metrics.json")
                    self.assertFalse(report["passed"])
                    self.assertIn("artifact changed", report["cases"][0]["failures"][0])

        from tools.convert.sam3_artifacts import freeze_output_files, verify_output_files
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary); tensor = output / "tensor.bin"
            tensor.write_bytes(b"original output")
            frozen = freeze_output_files(output)
            verify_output_files(output, frozen)
            tensor.write_bytes(b"altered output")
            with self.assertRaisesRegex(ValueError, "output changed"):
                verify_output_files(output, frozen)

    def test_benchmark_rejects_incomplete_or_changed_workloads(self):
        import copy
        from tools.benchmark import benchmark_video
        from tools.convert.sam3_artifacts import write_json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "trace").mkdir(); (root / "tensors").mkdir()
            for frame in range(64):
                directory = root / f"{frame:06d}"; directory.mkdir()
                mask = dump_array(directory, "mask", np.array([[1,0],[0,0]], dtype=np.uint8))
                write_json(directory / "results.json", {"schema_version": 1, "frame_index": frame,
                    "emitted_after_frame": min(frame+14,63),
                    "objects": [{"id": 7, "score": .9, "box": [0,0,1,1], "mask": mask}]})
                emitted = 64 if frame == 63 else max(0, frame-13)
                stats = {"accepted_frames": frame+1, "emitted_frames": emitted, "pending_frames": frame+1-emitted,
                         "pending_high_water": min(frame+1,15), "active_objects": 1, "retained_records": min(frame+1,27),
                         "retained_records_high_water": min(frame+1,27), "retained_memory_bytes": min(frame+1,27)*32,
                         "frame_ms": 10000 if frame < 16 else (1000 if frame == 63 else frame-15),
                         "tracker_ms": 1, "memory_ms": .5,
                         "runtime": {"vision_encodes": frame+1, "inferences": frame+1, "text_encodes": 1,
                                     "metal_nodes": (frame+1)*10, "cpu_nodes": 0, "weight_buffer_bytes": 100,
                                     "compute_buffer_bytes": 200, "process_peak_rss_bytes": 300}}
                write_json(root / "trace" / f"{frame:06d}-stats.json", stats)
                write_json(root / "trace" / f"{frame:06d}.json", {"frame_index": frame,
                    "births": [{"id":7}] if frame == 0 else [], "removed": [], "groups": [] if frame == 0 else [{"ids":[7]}]})
            manifest = {"complete": True, "task": "text_video", "frame_count": 64, "threads": 4,
                        "precision": "f16", "storage_profile": "", "backend": "metal", "max_objects": 8,
                        "width": 2, "height": 2, "prompt": "truck", "model_load_ms": 5, "stats": stats}
            write_json(root / "manifest.json", manifest)
            analyze = lambda: benchmark_video.analyze_run(root, 1, "f16", "metal", 2, 2)
            valid = analyze()
            self.assertEqual(valid["timings"]["frame_ms"]["count"], 48)
            self.assertEqual(valid["timings"]["frame_ms"]["median_ms"], 24.5)
            self.assertAlmostEqual(valid["timings"]["frame_ms"]["p95_ms"], 45.65)
            self.assertEqual(valid["timings"]["frame_ms"]["maximum_ms"], 1000)
            self.assertEqual(valid["first_output"]["emitted_after_frame"], 14)
            f32_manifest = copy.deepcopy(manifest); f32_manifest["precision"] = "f32"
            write_json(root / "manifest.json", f32_manifest)
            f32 = benchmark_video.analyze_run(root, 1, "f32", "metal", 2, 2)
            self.assertEqual(f32["timings"], valid["timings"])
            self.assertEqual(f32["first_output"], valid["first_output"])
            with self.assertRaisesRegex(ValueError, "profile"):
                analyze()
            write_json(root / "manifest.json", manifest)
            # Counts remain one, but retiring/replacing an ID must not be accepted as continuity.
            trace_path = root / "trace/000032.json"; trace = read_json(trace_path)
            changed = copy.deepcopy(trace); changed.update(births=[{"id":8}], removed=[7])
            write_json(trace_path, changed)
            with self.assertRaisesRegex(ValueError, "continuity"):
                analyze()
            write_json(trace_path, trace)
            missing = root / "trace/000063-stats.json"; saved = missing.read_bytes(); missing.unlink()
            with self.assertRaises(OSError):
                analyze()
            missing.write_bytes(saved)
            altered = copy.deepcopy(manifest); altered["complete"] = False
            write_json(root / "manifest.json", altered)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analyze()
            write_json(root / "manifest.json", manifest)
            path = root / "trace/000040-stats.json"; before = read_json(path)
            altered = copy.deepcopy(before); altered["runtime"]["cpu_nodes"] = 1; write_json(path, altered)
            with self.assertRaisesRegex(ValueError, "placement"):
                analyze()
            altered = copy.deepcopy(before); altered["runtime"]["compute_buffer_bytes"] = 201; write_json(path, altered)
            with self.assertRaisesRegex(ValueError, "plateau"):
                analyze()
            write_json(path, before)
            self.assertEqual(len(analyze()["samples"]), 64)
            manifest.update(backend="cuda", cuda_device=0, device_name="NVIDIA test GPU")
            write_json(root / "manifest.json", manifest)
            for frame in range(64):
                path = root / "trace" / f"{frame:06d}-stats.json"
                sample = read_json(path)
                sample["runtime"]["cuda_nodes"] = sample["runtime"].pop("metal_nodes")
                sample["runtime"]["metal_nodes"] = 0
                write_json(path, sample)
            manifest["stats"] = read_json(root / "trace/000063-stats.json")
            write_json(root / "manifest.json", manifest)
            cuda = lambda: benchmark_video.analyze_run(root, 1, "f16", "cuda", 2, 2)
            self.assertEqual(cuda()["timings"], valid["timings"])
            with self.assertRaisesRegex(ValueError, "CUDA device"):
                benchmark_video.analyze_run(root, 1, "f16", "cuda", 2, 2, cuda_device=1)
            manifest["arithmetic_profile"] = "ggml-cuda-f16-v1"
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "arithmetic profile"):
                cuda()
            self.assertEqual(benchmark_video.analyze_run(root, 1, "f16", "cuda", 2, 2,
                                                        cuda_compute="f16")["timings"], valid["timings"])
            manifest["arithmetic_profile"] = ""
            write_json(root / "manifest.json", manifest)
            path = root / "trace/000040-stats.json"; sample = read_json(path)
            for counter in ("cpu_nodes", "metal_nodes", "blas_nodes"):
                sample["runtime"][counter] = 1; write_json(path, sample)
                with self.assertRaisesRegex(ValueError, "placement"):
                    cuda()
                sample["runtime"][counter] = 0
        with patch.object(benchmark_video.subprocess, "check_output", return_value="100 1 32 /usr/bin/time\n101 100 2048 /tmp/sam_video\n102 1 99999 /elsewhere/sam_video\n"):
            self.assertEqual(benchmark_video.current_rss(100, "sam_video"), {"pid":101, "rss_bytes":2097152})

    def test_video_oracle_behavior_and_incomplete_provenance(self):
        from tools.validation.export_video_reference import verify_behavior
        from tools.validation.validate_video import check_provenance as check_video_provenance

        visible = lambda *ids: {"objects": [{"id": value} for value in ids]}
        self.assertTrue(verify_behavior("motion", [visible(4), visible(4)], []))
        self.assertFalse(verify_behavior("motion", [visible(4), visible(5)], []))
        entry_traces = [{"births": [], "removed": []} for _ in range(18)]
        entry_traces[16]["births"] = [{"id": 5}]
        self.assertTrue(verify_behavior("entry", [visible(4)] * 16 + [visible(4, 5)] * 2, entry_traces))
        self.assertFalse(verify_behavior("entry", [visible(4)] * 16 + [visible(4, 5)] * 2, []))
        self.assertFalse(verify_behavior("entry", [visible(4)] * 16 + [visible(4, 5), visible(5)], entry_traces))
        self.assertFalse(verify_behavior("entry", [visible(4)] * 18, []))
        trace = [{"births": [{"id": 4}], "removed": []}, {"births": [], "removed": [4]}]
        self.assertTrue(verify_behavior("hotstart-removal", [visible(), visible()], trace))
        self.assertFalse(verify_behavior("hotstart-removal", [visible(4), visible()], trace))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps({"complete": False}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                check_video_provenance(root / "missing-model.gguf", root)
            self.assertFalse((root / "missing-model.gguf").exists())

    def test_video_comparison_keeps_birth_mapping_and_rejects_fallback(self):
        from tools.validation.validate_video import compare_case, read_objects

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); reference, actual = root / "reference", root / "actual"
            case = {"id": "motion", "frames": 2, "width": 2, "height": 2, "prompt": "truck", "token_ids": [0] * 32}
            stats = {"accepted_frames": 2, "emitted_frames": 2, "pending_frames": 0, "pending_high_water": 2,
                     "retained_records": 2, "active_objects": 1, "runtime": {"metal_nodes": 20, "cpu_nodes": 0}}
            for directory, identifier in ((reference, 4), (actual, 9)):
                (directory / "trace").mkdir(parents=True)
                for frame in range(2):
                    target = directory / f"{frame:06d}"; target.mkdir()
                    mask = dump_array(target, "mask", np.array([[1, 0], [0, 0]], dtype=np.uint8))
                    record = {"schema_version": 1, "frame_index": frame, "emitted_after_frame": 1,
                              "objects": [{"id": identifier, "score": 0.9, "box": [0, 0, 0, 0], "mask": mask}]}
                    (target / "results.json").write_text(json.dumps(record))
                    trace = {"frame_index": frame, "groups": [], "births": [{"id": identifier}] if frame == 0 else [], "removed": [],
                             "propagation": [{"id": identifier, "mask_index": 2, "pointer_index": 2,
                                              "iou": [0.99, 0.7, 0.9, 0.8]}] if frame else []}
                    (directory / "trace" / f"{frame:06d}.json").write_text(json.dumps(trace))
                    (directory / "trace" / f"{frame:06d}-stats.json").write_text(json.dumps(stats))
                    tensors = directory / "tensors" / f"{frame:06d}"; tensors.mkdir(parents=True)
                    metadata = dump_array(tensors, "pointer", np.asarray([1, 2], dtype=np.float32))
                    (tensors / "tensors.json").write_text(json.dumps({"schema_version": 1, "byte_order": "little",
                                                                    "tensors": {f"object.{identifier}.pointer": metadata}}))
            manifest = {"complete": True, "task": "text_video", "frame_count": 2, "precision": "f32", "backend": "metal",
                        "max_objects": 8, "stats": stats, **{key: case[key] for key in ("width", "height", "prompt", "token_ids")}}
            (actual / "manifest.json").write_text(json.dumps(manifest))
            gates = read_json(Path(__file__).resolve().parents[2] / "tests/data/sam3-video-cases.json")["acceptance"]
            result = compare_case(reference, actual, case, "f32", "metal", gates, 8)
            self.assertTrue(result["passed"]); self.assertEqual(result["id_mapping"], {9: 4})
            trace_path = actual / "trace/000001.json"; trace = read_json(trace_path)
            trace["propagation"][0]["pointer_index"] = 1; trace_path.write_text(json.dumps(trace))
            failed = compare_case(reference, actual, case, "f32", "metal", gates, 8)
            self.assertFalse(failed["passed"])
            self.assertIn("frame 1: propagated mask/pointer candidate differs", failed["failures"])
            trace["propagation"][0]["pointer_index"] = 2; trace_path.write_text(json.dumps(trace))
            manifest.update(precision="hybrid", storage_profile="visual-tracker-f32-v1")
            (actual / "manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(compare_case(reference, actual, case, "hybrid", "metal", gates, 8)["passed"])
            manifest.update(precision="f32", storage_profile="")
            (actual / "manifest.json").write_text(json.dumps(manifest))
            # A propagated object cannot acquire a new ID by matching its mask.
            path = actual / "000001/results.json"; changed = read_json(path)
            changed["objects"][0]["id"] = 10; path.write_text(json.dumps(changed))
            with self.assertRaises(KeyError):
                compare_case(reference, actual, case, "f32", "metal", gates, 8)
            changed["objects"][0]["id"] = 9; changed["objects"][0]["box"][0] = "bad"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "malformed"):
                read_objects(path.parent, 1, 2, 2, False)
            manifest["stats"]["runtime"]["cpu_nodes"] = 1
            (actual / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "CPU fallback"):
                compare_case(reference, actual, case, "f32", "metal", gates, 8)
            changed["objects"][0]["box"][0] = 0
            path.write_text(json.dumps(changed))
            manifest.update(backend="cuda", cuda_device=0, device_name="NVIDIA test GPU")
            manifest["stats"]["runtime"] = {"cuda_nodes": 20, "cpu_nodes": 0, "metal_nodes": 0}
            (actual / "manifest.json").write_text(json.dumps(manifest))
            for frame in range(2):
                (actual / "trace" / f"{frame:06d}-stats.json").write_text(json.dumps(manifest["stats"]))
            self.assertTrue(compare_case(reference, actual, case, "f32", "cuda", gates, 8)["passed"])
            with self.assertRaisesRegex(ValueError, "selected-GPU"):
                compare_case(reference, actual, case, "f32", "cuda", gates, 8, cuda_device=1)
            manifest["arithmetic_profile"] = "ggml-cuda-f16-v1"
            (actual / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "arithmetic profile"):
                compare_case(reference, actual, case, "f32", "cuda", gates, 8)
            tensor_directory = actual / "tensors/000001"
            tensor_index = read_json(tensor_directory / "tensors.json")
            tensor_index["tensors"]["object.9.pointer"] = dump_array(
                tensor_directory, "pointer", np.asarray([5, 6], dtype=np.float32))
            (tensor_directory / "tensors.json").write_text(json.dumps(tensor_index))
            fast = compare_case(reference, actual, case, "f32", "cuda", gates, 8, cuda_compute="f16")
            self.assertTrue(fast["passed"])
            self.assertFalse(fast["tensor_fidelity_passed"])
            # Final-quality mode permits another internally consistent candidate,
            # while retaining identity, selection-rule and finite-value checks.
            trace["propagation"][0].update(mask_index=1, pointer_index=1, iou=[0.99, 0.95, 0.9, 0.8])
            trace_path.write_text(json.dumps(trace))
            fast = compare_case(reference, actual, case, "f32", "cuda", gates, 8, cuda_compute="f16")
            self.assertTrue(fast["passed"])
            self.assertFalse(fast["internal_candidate_equivalence_passed"])
            self.assertEqual(len(fast["internal_candidate_diagnostics"]), 1)
            trace["propagation"][0]["pointer_index"] = 2
            trace_path.write_text(json.dumps(trace))
            self.assertFalse(compare_case(reference, actual, case, "f32", "cuda", gates, 8, cuda_compute="f16")["passed"])
            trace["propagation"][0]["pointer_index"] = 4
            trace_path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "candidate scores/indices"):
                compare_case(reference, actual, case, "f32", "cuda", gates, 8, cuda_compute="f16")
            trace["propagation"][0]["pointer_index"] = 1
            trace["propagation"][0]["iou"][1] = float("nan")
            trace_path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                compare_case(reference, actual, case, "f32", "cuda", gates, 8, cuda_compute="f16")
            trace["propagation"][0]["iou"][1] = 0.95
            trace["propagation"].append(dict(trace["propagation"][0]))
            trace_path.write_text(json.dumps(trace))
            self.assertFalse(compare_case(reference, actual, case, "f32", "cuda", gates, 8, cuda_compute="f16")["passed"])
            trace["propagation"].pop()
            trace_path.write_text(json.dumps(trace))
            changed["objects"][0]["score"] = 0.8
            path.write_text(json.dumps(changed))
            self.assertFalse(compare_case(reference, actual, case, "f32", "cuda", gates, 8, cuda_compute="f16")["passed"])
            changed["objects"][0]["score"] = 0.9
            path.write_text(json.dumps(changed))
            manifest["arithmetic_profile"] = ""
            (actual / "manifest.json").write_text(json.dumps(manifest))
            tensor_index["tensors"]["object.9.pointer"] = dump_array(
                tensor_directory, "pointer", np.asarray([1, 2], dtype=np.float32))
            (tensor_directory / "tensors.json").write_text(json.dumps(tensor_index))
            sample = dict(manifest["stats"])
            sample["runtime"] = {"cuda_nodes": 20, "cpu_nodes": 1, "metal_nodes": 0}
            (actual / "trace/000001-stats.json").write_text(json.dumps(sample))
            failed = compare_case(reference, actual, case, "f32", "cuda", gates, 8)
            self.assertFalse(failed["passed"])
            self.assertIn("frame 1: missing CUDA compute or graph fallback", failed["failures"])


if __name__ == "__main__":
    unittest.main()
