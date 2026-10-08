"""Focused unit checks for frozen quantization gates and diagnostic separation."""

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import gguf

from tools.convert.sam3_artifacts import PAB_REVISION, sha256_file, write_json
from tools.validation.export_quantized_reference import tensor_error_metrics
from tools.convert.sam3_gguf import (QUANTIZATION_MODULES,
                       quantization_module_for_linear_weight,
                       quantized_tensor_type,
                       quantization_profile as storage_quantization_profile)
from tools.validation.validate_image import (GATES, QUANTIZATION_GATES, QUANTIZATION_GATES_PATH,
                            QUANTIZATION_GATES_SHA256, VISION_QUANTIZATION_GATES,
                            VISION_QUANTIZATION_GATES_PATH, VISION_QUANTIZATION_GATES_SHA256,
                            OUTPUT_QUALITY_GATES, OUTPUT_QUALITY_GATES_PATH, OUTPUT_QUALITY_GATES_SHA256,
                            FULL_QUANTIZATION_GATES, FULL_QUANTIZATION_GATES_PATH,
                            FULL_QUANTIZATION_GATES_SHA256, FULL_OUTPUT_QUALITY_GATES,
                            FULL_OUTPUT_QUALITY_GATES_PATH, FULL_OUTPUT_QUALITY_GATES_SHA256,
                            QUANTIZED_PRECISIONS, native_arithmetic_tensor_passed, quantization_profile,
                            quantization_selection, output_quality_selection,
                            canonical_module_selection, validate_gguf_module_metadata,
                            validate_schema4_tensor_module_records,
                            quantization_release_eligibility,
                            validate_diagnostic_case_descriptor,
                            validate_existing_validation_profile, PINNED_GGML_REVISION,
                            PINNED_GGML_VERSION, SAM_PATCHED_GGML_BUILD_COMMIT,
                            allowed_runtime_arithmetic_profiles, runtime_arithmetic_profiles_valid,
                            validate_existing_run_receipt,
                            validate_quantization_sidecar)


class QuantizationChecks(unittest.TestCase):
    @staticmethod
    def compare_case_fixture(precision, reference_scores, actual_scores, reference_detections,
                             actual_detections, high_mask_iou=1.0, model=None, runtime_modules=None,
                             backend="cpu", result_override=None, cuda_compute="f32", preprocessed_error=0.0):
        import numpy as np
        from tools.validation import validate_image
        names = ("preprocessed_image", "mask_logits")
        index = {"schema_version": 1, "byte_order": "little", "token_ids": [0] * 32,
                 "tensors": {name: name for name in names}}

        def read_fixture_array(directory, metadata, expected_dtype=None, require_hash=True):
            if expected_dtype == "uint8":
                return np.ones((2, 2), dtype=np.uint8)
            if metadata == "preprocessed_image":
                return np.asarray([0.0, 1.0], dtype=np.float32) + (preprocessed_error if Path(directory).name == "actual" else 0)
            if metadata == "mask_logits":
                return np.asarray([2.0, 2.0] if Path(directory).name == "actual" else [1.0, 1.0], dtype=np.float32)
            raise AssertionError(metadata)

        result_base = {"schema_version": 1, "width": 2, "height": 2, "prompt": "truck",
                       "score_threshold": 0.5, "detections": []}
        reference_result = dict(result_base)
        arithmetic_profiles = {"cpu": "ggml-quantized-weights-f32-v1", "metal": "ggml-quantized-native-v1",
                               "cuda": "ggml-quantized-cuda-native-v1"}
        actual_result = {**result_base, "backend": backend, "precision": precision,
                         "storage_profile": (model or {}).get("storage_profile", ""),
                         "arithmetic_profile": (arithmetic_profiles[backend] if precision in QUANTIZED_PRECISIONS
                                                else ""),
                         "runtime": {"cpu_nodes": int(backend == "cpu"), "blas_nodes": 0,
                                     "metal_nodes": int(backend == "metal"), "cuda_nodes": int(backend == "cuda")}}
        if backend == "cuda":
            actual_result.update(device_name="CUDA fixture", cuda_device=0)
        if result_override is not None:
            actual_result.update(result_override)
        if (model or {}).get("sam_schema_version") == 4:
            actual_result["quantization_modules"] = ((model or {}).get("quantization_modules")
                                                       if runtime_modules is None else runtime_modules)
        case = {"id": "fixture", "prompt": "truck"}
        with (patch.object(validate_image, "REQUIRED_TENSORS", names),
              patch.object(validate_image, "read_tensor_index", return_value=index),
              patch.object(validate_image, "read_array", side_effect=read_fixture_array),
              patch.object(validate_image, "query_scores", side_effect=[reference_scores, actual_scores]),
              patch.object(validate_image, "read_results", side_effect=[
                  (reference_result, reference_detections), (actual_result, actual_detections)]),
              patch.object(validate_image, "mask_iou", return_value=high_mask_iou)):
            return validate_image.compare_case(Path("reference"), Path("actual"), case,
                                               precision, backend, model, cuda_compute=cuda_compute)

    def test_cuda_f16_quality_is_explicit_and_keeps_hard_gates(self):
        import numpy as np

        scores = np.full(200, 0.1); scores[0] = 0.9
        high = {"score": 0.9, "box": [0, 0, 1, 1], "mask": {"file": "mask.bin"}}
        compare = lambda **kwargs: self.compare_case_fixture(
            "f32", scores, scores, {0: high}, {0: high}, backend="cuda", **kwargs)
        self.assertFalse(compare()["passed"])  # Large intermediate tensor error.
        with self.assertRaisesRegex(ValueError, "arithmetic profile"):
            compare(cuda_compute="f16")  # A default-mode receipt cannot qualify as fast.
        fast = {"cuda_compute": "f16", "result_override": {"arithmetic_profile": "ggml-cuda-f16-v1"}}
        accepted = compare(high_mask_iou=0.96, **fast)
        self.assertTrue(accepted["passed"])
        self.assertFalse(accepted["tensor_fidelity_passed"])
        self.assertFalse(compare(high_mask_iou=0.94, **fast)["passed"])
        self.assertFalse(compare(preprocessed_error=0.1, **fast)["passed"])
        with self.assertRaisesRegex(ValueError, "arithmetic profile"):
            compare(result_override={"arithmetic_profile": "ggml-cuda-f16-v1"})
        self.assertFalse(runtime_arithmetic_profiles_valid("cuda", ["ggml-quantized-cuda-f16-v1"]))
        self.assertTrue(runtime_arithmetic_profiles_valid(
            "cuda", ["ggml-quantized-cuda-f16-v1"], cuda_compute="f16"))
        with self.assertRaises(ValueError):
            allowed_runtime_arithmetic_profiles("cpu", cuda_compute="f16")
        quantized = self.compare_case_fixture(
            "q8_0", scores, scores, {0: high}, {0: high}, backend="cuda", cuda_compute="f16",
            model={"storage_profile": "image-linear-q8_0-v1"},
            result_override={"arithmetic_profile": "ggml-quantized-cuda-f16-v1"})
        self.assertTrue(quantized["passed"])
        self.assertEqual(quantized["output_gate_precision"], "quantized-profile")

    def test_frozen_profiles_and_legacy_gates(self):
        self.assertEqual(QUANTIZATION_GATES_SHA256,
                         "819d9de1ebe638d223db9d57c8ac652b36707fae3d36124735bdc65d681ba2cc")
        self.assertEqual(sha256_file(QUANTIZATION_GATES_PATH), QUANTIZATION_GATES_SHA256)
        self.assertEqual(set(QUANTIZATION_GATES["profiles"]), {"q8_0", "q6_k", "q5_k", "q4_k"})
        self.assertEqual(VISION_QUANTIZATION_GATES_SHA256,
                         "6bdb91602e8a89e8b29b208282b846b541d8fbe499ae36955e4f905e9eff7a93")
        self.assertEqual(sha256_file(VISION_QUANTIZATION_GATES_PATH), VISION_QUANTIZATION_GATES_SHA256)
        self.assertEqual(VISION_QUANTIZATION_GATES["parent_gate_sha256"], QUANTIZATION_GATES_SHA256)
        for precision, expected in zip(("q8_0", "q6_k", "q5_k", "q4_k"),
                                       [(0.02, 0.96, 0.02, 0.01), (0.04, 0.94, 0.03, 0.015),
                                        (0.06, 0.92, 0.04, 0.02), (0.1, 0.9, 0.05, 0.03)]):
            old = quantization_selection(precision, f"image-linear-{precision}-v1")
            vision = quantization_selection(precision, f"image-vision-linear-{precision}-v1")
            fields = ("normalized_l2_max", "high_confidence_mask_iou_min",
                      "score_absolute_error_max", "box_dimension_fraction_max")
            self.assertEqual(tuple(old["profile"][key] for key in fields), expected)
            self.assertEqual(old["profile"]["normalized_l2_max"], vision["profile"]["normalized_l2_max"])
            self.assertEqual(old["gate_set"], QUANTIZATION_GATES["gate_set"])
            self.assertEqual(vision["gate_set"], VISION_QUANTIZATION_GATES["gate_set"])
            self.assertNotEqual(old["gates_sha256"], vision["gates_sha256"])
            self.assertNotEqual(old["profile"]["storage_profile"], vision["profile"]["storage_profile"])
            self.assertEqual(old["profile"]["native_arithmetic_normalized_l2_max"],
                             vision["profile"]["native_arithmetic_normalized_l2_max"])
        self.assertEqual(GATES, {
            "normalized_l2_f32": 1e-3, "normalized_l2_f16": 2e-2,
            "zero_norm_limit": 1e-12, "zero_norm_max_abs": 1e-5,
            "preprocessed_max_abs": 2 / 255, "high_score_min": 0.6,
            "mask_iou_f32": 0.98, "mask_iou_f16": 0.95, "score_max_abs": 0.02,
            "box_dimension_fraction": 0.01, "low_score_max": 0.4, "score_threshold": 0.5,
        })

    def test_precision_selection_fails_closed(self):
        for precision, storage in (("f16", "image-linear-f16-v1"), ("f32", "image-linear-f32-v1"),
                                   ("hybrid", "visual-tracker-f32-v1"), ("q3_k", "image-linear-q3_k-v1"),
                                   ("Q8_0", "image-linear-q8_0-v1"), ("q8_0", None),
                                   ("q8_0", "image-vision-linear-q6_k-v1"), ("q8_0", "unknown")):
            with self.subTest(precision=precision, storage=storage), self.assertRaises(ValueError):
                quantization_profile(precision, storage)

    def test_full_schema4_gates_are_frozen_and_inherit_the_existing_precision_limits(self):
        self.assertEqual(FULL_QUANTIZATION_GATES_SHA256,
                         "17447e93cbec08233a9be3b50f43574da057d5ce05f79493e662fb757071d0bb")
        self.assertEqual(sha256_file(FULL_QUANTIZATION_GATES_PATH), FULL_QUANTIZATION_GATES_SHA256)
        self.assertEqual(FULL_QUANTIZATION_GATES["quantization_modules"], list(QUANTIZATION_MODULES))
        self.assertEqual(FULL_OUTPUT_QUALITY_GATES_SHA256,
                         "f796add8c03adc078797c6c5f93bb439c214044007578b40a63af5d99a342aa3")
        self.assertEqual(sha256_file(FULL_OUTPUT_QUALITY_GATES_PATH), FULL_OUTPUT_QUALITY_GATES_SHA256)
        self.assertEqual(FULL_OUTPUT_QUALITY_GATES["parent_tensor_gate_sha256"], FULL_QUANTIZATION_GATES_SHA256)
        self.assertFalse(FULL_OUTPUT_QUALITY_GATES["selected_detection_quality"]["tensor_fidelity_is_primary_pass_gate"])
        for precision in ("q8_0", "q6_k", "q5_k", "q4_k"):
            old = quantization_selection(precision, f"image-linear-{precision}-v1")
            full = quantization_selection(precision, f"image-full-linear-{precision}-v1")
            self.assertEqual(full["family"], "full")
            self.assertEqual(full["profile"]["storage_profile"], f"image-full-linear-{precision}-v1")
            self.assertEqual(full["profile"]["normalized_l2_max"], old["profile"]["normalized_l2_max"])
            self.assertEqual(full["profile"]["high_confidence_mask_iou_min"],
                             old["profile"]["high_confidence_mask_iou_min"])
            self.assertEqual(full["profile"]["score_absolute_error_max"],
                             old["profile"]["score_absolute_error_max"])
            self.assertEqual(full["profile"]["box_dimension_fraction_max"],
                             old["profile"]["box_dimension_fraction_max"])

    def test_schema4_profile_selection_is_exact_and_custom_never_becomes_full(self):
        modules = list(QUANTIZATION_MODULES)
        full = quantization_selection("q8_0", "image-full-linear-q8_0-v1", modules)
        custom_all = quantization_selection("q8_0", "image-modules-linear-q8_0-v1", modules)
        self.assertEqual(full["family"], "full")
        self.assertEqual(custom_all["family"], "custom")
        self.assertEqual(custom_all["profile"]["profile_status"], "diagnostic")
        self.assertEqual(custom_all["profile"]["quantization_modules"], modules)
        for invalid in (None, [], ["vision", "vision"], ["other"], ["text", "vision"]):
            with self.subTest(modules=invalid), self.assertRaises(ValueError):
                quantization_selection("q8_0", "image-modules-linear-q8_0-v1", invalid)
        with self.assertRaisesRegex(ValueError, "all four"):
            quantization_selection("q8_0", "image-full-linear-q8_0-v1", ["vision"])
        self.assertEqual(canonical_module_selection(["vision", "text"]), ["vision", "text"])

    def test_module_inventory_includes_fused_projection_weight_suffixes(self):
        self.assertEqual(quantization_module_for_linear_weight("fenc.layers.0.sa.in_proj_weight"), "fusion")
        self.assertEqual(quantization_module_for_linear_weight("ddec.layers.0.ca_text.in_proj_weight"), "decoder")
        self.assertEqual(quantized_tensor_type(
            "fenc.layers.0.sa.in_proj_weight", [768, 256], "q8_0",
            "image-modules-linear-q8_0-v1", ["fusion"]), gguf.GGMLQuantizationType.Q8_0)

    def test_schema4_sidecar_requires_modules_and_custom_opt_in(self):
        modules = list(QUANTIZATION_MODULES)
        for storage_profile, selected_modules, status in (
                ("image-full-linear-q8_0-v1", modules, "candidate"),
                ("image-modules-linear-q8_0-v1", ["vision", "decoder"], "diagnostic")):
            profile = storage_quantization_profile("q8_0", storage_profile, selected_modules)
            sidecar = {
                "sam_schema_version": 4, "precision": "q8_0", "storage_profile": storage_profile,
                "quantization_modules": selected_modules, "profile_status": status,
                "converter_revision": PAB_REVISION, "converter_sha256": "1" * 64,
                "gguf_helper_sha256": "2" * 64, "requirements_lock_sha256": "3" * 64,
                "gguf_package_version": "0.19.0",
                "quantization": {"version": 2, "ggml_quantization_version": 2,
                                 "gguf_file_type": profile["file_type"], "ggml_type": profile["ggml_type"],
                                 "block_elements": profile["block_elements"],
                                 "block_bytes": profile["block_bytes"],
                                 "row_block_fallback": profile.get("fallback_type"),
                                 "quantize_text_linear": "text" in selected_modules},
                "quantizer": {"implementation": "gguf-py", "version": "0.19.0",
                              "module": "gguf.quants.quantize", "ggml_quantization_version": 2},
            }
            if storage_profile.startswith("image-full"):
                validate_quantization_sidecar(sidecar)
            else:
                with self.assertRaisesRegex(ValueError, "diagnostic only"):
                    validate_quantization_sidecar(sidecar)
                validate_quantization_sidecar(sidecar, allow_custom_quantization=True)
                sidecar["quantization_modules"] = modules
                sidecar["quantization"]["quantize_text_linear"] = True
                validate_quantization_sidecar(sidecar, allow_custom_quantization=True)
                self.assertEqual(storage_quantization_profile("q8_0", storage_profile, modules)["profile_status"],
                                 "diagnostic")

    def test_schema4_gguf_module_csv_and_runtime_array_are_bound_to_sidecar(self):
        class Field:
            def __init__(self, types, value):
                self.types = types
                self.value = value

            def contents(self):
                return self.value

        class Reader:
            def __init__(self, field):
                self.field = field

            def get_field(self, key):
                self.asserted_key = key
                return self.field

        modules = ["vision", "text"]
        validate_gguf_module_metadata(Reader(Field([gguf.GGUFValueType.STRING], ",".join(modules))), modules)
        for field in (None, Field([gguf.GGUFValueType.UINT32], 2),
                      Field([gguf.GGUFValueType.STRING], "text,vision")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "GGUF module metadata"):
                validate_gguf_module_metadata(Reader(field), modules)
        scores = __import__("numpy").full(200, 0.1)
        scores[0] = 0.9
        detection = {"query_index": 0, "score": 0.9, "box": [0.0, 0.0, 2.0, 2.0],
                     "mask": {"file": "mask.bin", "dtype": "uint8", "shape": [2, 2]}}
        model = {"sam_schema_version": 4, "storage_profile": "image-modules-linear-q8_0-v1",
                 "quantization_modules": modules}
        result = self.compare_case_fixture("q8_0", scores, scores, {0: detection}, {0: detection}, model=model)
        self.assertTrue(result["output_quality_passed"])
        self.assertEqual(result["quantization_modules"], modules)
        mismatch = self.compare_case_fixture("q8_0", scores, scores, {0: detection}, {0: detection},
                                             model=model, runtime_modules=["vision"])
        self.assertFalse(mismatch["output_quality_passed"])
        self.assertTrue(any("quantization modules differ" in failure for failure in mismatch["output_quality_failures"]))

    def test_custom_profile_cannot_qualify_for_milestone_or_release(self):
        selection = quantization_selection("q8_0", "image-modules-linear-q8_0-v1", ["vision", "text"])
        flags = quantization_release_eligibility(
            selection, {"profile_status": "diagnostic"}, {"eligible_for_milestone": True}, True)
        self.assertFalse(flags["eligible_for_milestone"])
        self.assertFalse(flags["release_support_eligible"])

    def test_sidecar_accepts_conversion_snapshot_hashes_and_pins_gguf_package(self):
        for storage_profile in ("image-linear-q8_0-v1", "image-vision-linear-q8_0-v1"):
            profile = storage_quantization_profile("q8_0", storage_profile)
            sidecar = {
                "precision": "q8_0", "storage_profile": storage_profile,
                "profile_status": profile["profile_status"], "converter_revision": PAB_REVISION,
                "converter_sha256": "1" * 64, "gguf_helper_sha256": "2" * 64,
                "requirements_lock_sha256": "3" * 64, "gguf_package_version": "0.19.0",
                "quantization": {"version": 2, "ggml_quantization_version": 2,
                                 "gguf_file_type": profile["file_type"], "ggml_type": profile["ggml_type"],
                                 "block_elements": profile["block_elements"], "block_bytes": profile["block_bytes"],
                                 "row_block_fallback": profile.get("fallback_type"),
                                 "quantize_text_linear": profile["quantize_text_linear"]},
                "quantizer": {"implementation": "gguf-py", "version": "0.19.0",
                              "module": "gguf.quants.quantize", "ggml_quantization_version": 2},
            }
            validate_quantization_sidecar(sidecar)
            sidecar["quantization"]["quantize_text_linear"] = not profile["quantize_text_linear"]
            with self.assertRaisesRegex(ValueError, "pinned conversion/profile"):
                validate_quantization_sidecar(sidecar)
        sidecar["quantization"]["quantize_text_linear"] = profile["quantize_text_linear"]
        sidecar["gguf_package_version"] = "0.20.0"
        with self.assertRaisesRegex(ValueError, "pinned conversion/profile"):
            validate_quantization_sidecar(sidecar)

    def test_k_sidecar_requires_pinned_quantize_provider_but_allows_static_library(self):
        storage_profile = "image-vision-linear-q6_k-v1"
        profile = storage_quantization_profile("q6_k", storage_profile)
        sidecar = {
            "precision": "q6_k", "storage_profile": storage_profile,
            "profile_status": profile["profile_status"], "converter_revision": PAB_REVISION,
            "converter_sha256": "1" * 64, "gguf_helper_sha256": "2" * 64,
            "requirements_lock_sha256": "3" * 64, "gguf_package_version": "0.19.0",
            "quantization": {"version": 2, "ggml_quantization_version": 2,
                             "gguf_file_type": profile["file_type"], "ggml_type": profile["ggml_type"],
                             "block_elements": profile["block_elements"], "block_bytes": profile["block_bytes"],
                             "row_block_fallback": profile.get("fallback_type"),
                             "quantize_text_linear": profile["quantize_text_linear"]},
            "quantizer": {"implementation": "ggml_quantize_chunk", "fallback_implementation": "gguf-py",
                          "fallback_version": "0.19.0", "ggml_revision": PINNED_GGML_REVISION,
                          "ggml_build_commit": PINNED_GGML_REVISION, "ggml_version": PINNED_GGML_VERSION,
                          "ggml_quantization_version": 2,
                          "helper": {"file": "sam_quantize_rows", "sha256": "4" * 64},
                          "ggml_library": {"file": "libggml-base.dylib", "sha256": "5" * 64},
                          "ggml_quantize_library": {"file": "libggml-quantize.a", "sha256": "6" * 64}},
        }
        validate_quantization_sidecar(sidecar)
        sidecar["quantizer"]["ggml_build_commit"] = SAM_PATCHED_GGML_BUILD_COMMIT
        validate_quantization_sidecar(sidecar)
        from tools.convert.sam3_gguf import LEGACY_GGML_QUANTIZER_IDENTITIES
        current_identity = {key: sidecar["quantizer"][key]
                            for key in ("ggml_revision", "ggml_version", "ggml_build_commit")}
        for revision, version, commit in LEGACY_GGML_QUANTIZER_IDENTITIES:
            sidecar["quantizer"].update(ggml_revision=revision, ggml_version=version, ggml_build_commit=commit)
            validate_quantization_sidecar(sidecar)
            for field, current in current_identity.items():
                previous = sidecar["quantizer"][field]
                sidecar["quantizer"][field] = current
                with self.subTest(legacy=commit, mixed_field=field), self.assertRaisesRegex(ValueError, "pinned GGML"):
                    validate_quantization_sidecar(sidecar)
                sidecar["quantizer"][field] = previous
        sidecar["quantizer"]["ggml_build_commit"] += "-unverified"
        with self.assertRaisesRegex(ValueError, "pinned GGML encoder/provider"):
            validate_quantization_sidecar(sidecar)
        sidecar["quantizer"].update(current_identity)
        del sidecar["quantizer"]["ggml_quantize_library"]
        with self.assertRaisesRegex(ValueError, "pinned GGML encoder/provider"):
            validate_quantization_sidecar(sidecar)
        sidecar["quantizer"]["ggml_quantize_library"] = {"file": "libggml-quantize.a", "sha256": "6" * 64}
        sidecar["quantizer"]["ggml_version"] = "0.25.2"
        with self.assertRaisesRegex(ValueError, "pinned GGML encoder/provider"):
            validate_quantization_sidecar(sidecar)

    def test_native_arithmetic_uses_its_own_nonzero_and_zero_limits(self):
        for storage_profile in ("image-linear-q5_k-v1", "image-vision-linear-q5_k-v1"):
            profile = quantization_profile("q5_k", storage_profile)
            limit = profile["native_arithmetic_normalized_l2_max"]
            self.assertTrue(native_arithmetic_tensor_passed(
                {"normalized_l2": limit, "maximum_absolute_error": 100.0}, profile))
            self.assertFalse(native_arithmetic_tensor_passed(
                {"normalized_l2": limit + 1e-6, "maximum_absolute_error": 0.0}, profile))
            zero_limit = profile["zero_norm_max_abs"]
            self.assertTrue(native_arithmetic_tensor_passed(
                {"normalized_l2": None, "maximum_absolute_error": zero_limit}, profile))
            self.assertFalse(native_arithmetic_tensor_passed(
                {"normalized_l2": None, "maximum_absolute_error": zero_limit + 1e-8}, profile))

    def test_retained_receipt_cannot_cross_gate_families(self):
        old = quantization_selection("q8_0", "image-linear-q8_0-v1")
        vision = quantization_selection("q8_0", "image-vision-linear-q8_0-v1")
        model = {"precision": "q8_0", "storage_profile": "image-linear-q8_0-v1",
                 "output": {"sha256": "a" * 64}, "checkpoint": {"sha256": "b" * 64}}
        receipt = {"schema_version": 1, "backend": "cpu", "precision": "q8_0",
                   "model_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
                   "reference_manifest_sha256": "c" * 64,
                   "gate_set": old["gate_set"], "gates_sha256": old["gates_sha256"],
                   "storage_profile": "image-linear-q8_0-v1"}
        validate_existing_validation_profile(receipt, "cpu", model, "c" * 64, old)
        with self.assertRaisesRegex(ValueError, "different profile, gate"):
            validate_existing_validation_profile(receipt, "cpu", model, "c" * 64, vision)
        model["storage_profile"] = "image-vision-linear-q8_0-v1"
        receipt.update(gate_set=vision["gate_set"], gates_sha256=vision["gates_sha256"],
                       storage_profile="image-vision-linear-q8_0-v1")
        validate_existing_validation_profile(receipt, "cpu", model, "c" * 64, vision)

    def test_retained_dual_axis_receipt_binds_output_and_tensor_gate_families(self):
        tensor = quantization_selection("q8_0", "image-vision-linear-q8_0-v1")
        output = output_quality_selection("q8_0", "image-vision-linear-q8_0-v1")
        model = {"precision": "q8_0", "storage_profile": "image-vision-linear-q8_0-v1",
                 "output": {"sha256": "a" * 64}, "checkpoint": {"sha256": "b" * 64}}
        receipt = {"schema_version": 1, "backend": "cpu", "precision": "q8_0",
                   "model_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
                   "reference_manifest_sha256": "c" * 64,
                   "gate_set": output["gates"]["gate_set"], "gates_sha256": output["gates_sha256"],
                   "output_quality_gate_set": output["gates"]["gate_set"],
                   "output_quality_gates_sha256": output["gates_sha256"],
                   "tensor_fidelity_gate_set": tensor["gates"]["gate_set"],
                   "tensor_fidelity_gates_sha256": tensor["gates_sha256"],
                   "storage_profile": model["storage_profile"]}
        validate_existing_validation_profile(receipt, "cpu", model, "c" * 64, tensor)
        receipt["tensor_fidelity_gates_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "output-quality or tensor-fidelity"):
            validate_existing_validation_profile(receipt, "cpu", model, "c" * 64, tensor)
        receipt["tensor_fidelity_gates_sha256"] = tensor["gates_sha256"]
        receipt["gates_sha256"] = tensor["gates_sha256"]
        with self.assertRaisesRegex(ValueError, "output-quality or tensor-fidelity"):
            validate_existing_validation_profile(receipt, "cpu", model, "c" * 64, tensor)

    def test_schema4_tensor_module_and_reason_records_are_recomputed(self):
        modules = ["fusion"]
        profile = "image-modules-linear-q8_0-v1"
        items = [
            {"name": "fenc.layers.0.sa.in_proj_weight", "shape": [768, 256],
             "module": "fusion", "quantization_reason": "quantized-linear"},
            {"name": "text.blocks.0.attn.in_proj.weight", "shape": [3072, 1024],
             "module": "text", "quantization_reason": "module-not-selected"},
        ]
        validate_schema4_tensor_module_records(items, "q8_0", profile, modules)
        for key, value in (("module", "text"), ("quantization_reason", "outside-linear-whitelist")):
            corrupted = [dict(item) for item in items]
            corrupted[0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "allocation policy"):
                validate_schema4_tensor_module_records(corrupted, "q8_0", profile, modules)

    def test_existing_run_receipt_binds_binary_gate_metrics_and_source_hashes(self):
        selection = quantization_selection("q8_0", "image-linear-q8_0-v1")
        model_sha = "a" * 64
        artifacts = {"/retained/test_image": "b" * 64, "/retained/libggml.dylib": "c" * 64,
                     "/retained/model.gguf": model_sha, "/retained/model.gguf.manifest.json": "d" * 64}
        model = {"output": {"sha256": model_sha}}
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            output = parent / "validation"
            output.mkdir()
            metrics = {"run_artifact_sha256": artifacts}
            metrics_path = output / "metrics.json"
            write_json(metrics_path, metrics)
            driver = str(Path(__file__).resolve().parents[2] / "build/quantization/20261003/quant_acceptance.py")
            run = {"complete": True, "metrics_sha256": sha256_file(metrics_path),
                   "model_sha256": model_sha, "gate_sha256": selection["gates_sha256"],
                   "model_unchanged": True, "gate_unchanged": True,
                   "binary_and_libraries_unchanged": True,
                   "binary_and_libraries_before": {"sha256": artifacts},
                   "binary_and_libraries_after": {"sha256": dict(artifacts)},
                   "source_sha256": {driver: "1" * 64},
                   "source_sha256_after": {driver: "2" * 64}}
            write_json(parent / "run.json", run)
            verified = validate_existing_run_receipt(output, metrics, model, selection)
            self.assertEqual(verified["source_hash_delta_paths"], [driver])
            self.assertEqual(verified["binary_and_library_sha256"], artifacts)
            run["binary_and_libraries_after"]["sha256"]["/retained/libggml.dylib"] = "e" * 64
            write_json(parent / "run.json", run)
            with self.assertRaisesRegex(ValueError, "binary/library hashes"):
                validate_existing_run_receipt(output, metrics, model, selection)
            run["binary_and_libraries_after"]["sha256"] = dict(artifacts)
            run["source_sha256_after"] = {"/repo/src/model.cpp": "2" * 64}
            write_json(parent / "run.json", run)
            with self.assertRaisesRegex(ValueError, "source hash"):
                validate_existing_run_receipt(output, metrics, model, selection)

    def test_output_quality_gate_is_separate_and_profile_bound(self):
        self.assertEqual(OUTPUT_QUALITY_GATES_SHA256,
                         "c69dd6fee4abc789e421c407c7f954e3fdfd7f04a2a79288d3192816385f11e8")
        self.assertEqual(sha256_file(OUTPUT_QUALITY_GATES_PATH), OUTPUT_QUALITY_GATES_SHA256)
        self.assertFalse(OUTPUT_QUALITY_GATES["selected_detection_quality"]["tensor_fidelity_is_primary_pass_gate"])
        for precision in ("q8_0", "q6_k", "q5_k", "q4_k"):
            for prefix in ("image-linear-", "image-vision-linear-", "image-full-linear-"):
                storage = f"{prefix}{precision}-v1"
                selected = output_quality_selection(precision, storage)
                tensor_gate = quantization_selection(precision, storage)["profile"]
                self.assertEqual(selected["profile"]["mask_iou_min"], tensor_gate["high_confidence_mask_iou_min"])
                self.assertEqual(selected["profile"]["score_absolute_error_max"], tensor_gate["score_absolute_error_max"])
                self.assertEqual(selected["profile"]["box_dimension_fraction_max"], tensor_gate["box_dimension_fraction_max"])

    def test_output_gate_passes_valid_objects_while_tensor_fidelity_reports_failure(self):
        import numpy as np

        scores = np.full(200, 0.1)
        scores[0] = 0.9
        boxes = [0.0, 0.0, 2.0, 2.0]
        detection = {"query_index": 0, "score": 0.9, "box": boxes,
                     "mask": {"file": "mask.bin", "dtype": "uint8", "shape": [2, 2]}}
        profile = "image-vision-linear-q8_0-v1"
        result = self.compare_case_fixture(
            "q8_0", scores, scores, {0: detection}, {0: detection},
            model={"storage_profile": profile})
        self.assertTrue(result["passed"])
        self.assertTrue(result["output_quality_passed"])
        self.assertFalse(result["tensor_fidelity_passed"])
        self.assertFalse(result["tensors"]["mask_logits"]["passed"])
        self.assertTrue(any("mask_logits" in failure for failure in result["tensor_fidelity_failures"]))
        self.assertEqual(result["gate_set"], OUTPUT_QUALITY_GATES["gate_set"])

    def test_output_gate_rejects_missing_low_false_positive_and_bad_mask(self):
        import numpy as np

        scores = np.full(200, 0.1)
        scores[0] = 0.9
        scores[1] = 0.3
        actual_scores = scores.copy()
        base_detection = {"query_index": 0, "score": 0.9, "box": [0.0, 0.0, 2.0, 2.0],
                          "mask": {"file": "mask.bin", "dtype": "uint8", "shape": [2, 2]}}
        model = {"storage_profile": "image-vision-linear-q8_0-v1"}
        missing = self.compare_case_fixture("q8_0", scores, actual_scores, {0: base_detection}, {}, model=model)
        self.assertFalse(missing["output_quality_passed"])
        false_positive_scores = scores.copy(); false_positive_scores[1] = 0.7
        false_positive = {1: {**base_detection, "query_index": 1, "score": 0.7}}
        extra = self.compare_case_fixture("q8_0", scores, false_positive_scores,
                                          {0: base_detection}, {0: base_detection, **false_positive}, model=model)
        self.assertFalse(extra["output_quality_passed"])
        bad_mask = self.compare_case_fixture("q8_0", scores, actual_scores,
                                             {0: base_detection}, {0: base_detection},
                                             high_mask_iou=0.5, model=model)
        self.assertFalse(bad_mask["output_quality_passed"])

    def test_adjacent_query_selection_changes_are_reported_not_misclassified(self):
        import numpy as np

        ref_scores = np.full(200, 0.1); ref_scores[0] = 0.9; ref_scores[1] = 0.55
        act_scores = ref_scores.copy(); act_scores[1] = 0.45
        high = {"query_index": 0, "score": 0.9, "box": [0.0, 0.0, 2.0, 2.0],
                "mask": {"file": "mask.bin", "dtype": "uint8", "shape": [2, 2]}}
        adjacent = {"query_index": 1, "score": 0.55, "box": [0.0, 0.0, 2.0, 2.0],
                    "mask": {"file": "mask.bin", "dtype": "uint8", "shape": [2, 2]}}
        result = self.compare_case_fixture("q8_0", ref_scores, act_scores, {0: high, 1: adjacent}, {0: high},
                                           model={"storage_profile": "image-linear-q8_0-v1"})
        self.assertTrue(result["output_quality_passed"])
        self.assertEqual([item["query_index"] for item in result["threshold_adjacent"]], [1])

    def test_legacy_f32_f16_hybrid_primary_pass_semantics_stay_tensor_gated(self):
        import numpy as np

        scores = np.full(200, 0.1); scores[0] = 0.9
        high = {"query_index": 0, "score": 0.9, "box": [0.0, 0.0, 2.0, 2.0],
                "mask": {"file": "mask.bin", "dtype": "uint8", "shape": [2, 2]}}
        for precision, storage_profile in (("f32", ""), ("f16", ""), ("hybrid", "visual-tracker-f32-v1")):
            with self.subTest(precision=precision):
                result = self.compare_case_fixture(precision, scores, scores, {0: high}, {0: high},
                                                   model={"storage_profile": storage_profile,
                                                          "arithmetic_profile": ""})
                self.assertFalse(result["passed"])
                self.assertNotIn("output_quality_passed", result)
                self.assertTrue(any("mask_logits" in failure for failure in result["failures"]))

    def test_runtime_arithmetic_profiles_are_backend_and_receipt_bound(self):
        self.assertEqual(allowed_runtime_arithmetic_profiles("cpu"), {"ggml-quantized-weights-f32-v1"})
        self.assertEqual(allowed_runtime_arithmetic_profiles("metal"), {"ggml-quantized-native-v1"})
        self.assertEqual(allowed_runtime_arithmetic_profiles("cuda"), {"ggml-quantized-cuda-native-v1"})
        self.assertTrue(runtime_arithmetic_profiles_valid("cpu", ["ggml-quantized-weights-f32-v1"]))
        self.assertFalse(runtime_arithmetic_profiles_valid("cpu", ["ggml-quantized-native-v1"]))
        self.assertTrue(runtime_arithmetic_profiles_valid(
            "cpu", ["ggml-quantized-native-v1"], allow_historical_cpu_native=True))
        self.assertFalse(runtime_arithmetic_profiles_valid(
            "cpu", ["ggml-quantized-native-v1", "ggml-quantized-weights-f32-v1"],
            allow_historical_cpu_native=True))
        self.assertTrue(runtime_arithmetic_profiles_valid("metal", ["ggml-quantized-native-v1"]))
        self.assertFalse(runtime_arithmetic_profiles_valid("metal", ["ggml-quantized-weights-f32-v1"]))
        self.assertTrue(runtime_arithmetic_profiles_valid("cuda", ["ggml-quantized-cuda-native-v1"]))
        for profile in ("ggml-quantized-native-v1", "ggml-quantized-weights-f32-v1"):
            self.assertFalse(runtime_arithmetic_profiles_valid("cuda", [profile], allow_historical_cpu_native=True))
        for backend in ("cpu", "metal"):
            self.assertFalse(runtime_arithmetic_profiles_valid(backend, ["ggml-quantized-cuda-native-v1"]))
        self.assertFalse(runtime_arithmetic_profiles_valid("cuda", []))
        self.assertFalse(runtime_arithmetic_profiles_valid(
            "cuda", ["ggml-quantized-cuda-native-v1", "ggml-quantized-native-v1"]))

    def test_cuda_quantized_output_requires_its_arithmetic_and_selected_device(self):
        import numpy as np

        scores = np.full(200, 0.1); scores[0] = 0.9
        high = {"query_index": 0, "score": 0.9, "box": [0.0, 0.0, 2.0, 2.0],
                "mask": {"file": "mask.bin", "dtype": "uint8", "shape": [2, 2]}}
        model = {"storage_profile": "image-vision-linear-q8_0-v1"}
        compare = lambda **override: self.compare_case_fixture(
            "q8_0", scores, scores, {0: high}, {0: high}, model=model,
            backend="cuda", result_override=override)
        self.assertTrue(compare()["output_quality_passed"])
        for profile in ("ggml-quantized-native-v1", "ggml-quantized-weights-f32-v1"):
            invalid = compare(arithmetic_profile=profile)
            self.assertFalse(invalid["output_quality_passed"])
            self.assertTrue(any("arithmetic profile" in failure for failure in invalid["failures"]))
        for override in ({"cuda_device": 1}, {"device_name": ""},
                         {"runtime": {"cuda_nodes": 0}},
                         *({"runtime": {"cuda_nodes": 1, counter: 1}}
                           for counter in ("cpu_nodes", "metal_nodes", "blas_nodes"))):
            with self.assertRaisesRegex(ValueError, "CUDA acceptance"):
                compare(**override)

    def test_supplementary_case_must_reuse_exact_transform_and_input(self):
        frozen = {"id": "crop", "prompt": "truck", "score_threshold": 0.5,
                  "source_sha256": "a" * 64, "transform": {"crop_xyxy": [1, 2, 5, 6]}}
        original = {**frozen, "input_sha256": "b" * 64}
        diagnostic = {**frozen, "directory": "crop", "input": "crop/input.ppm", "input_sha256": "b" * 64}
        validate_diagnostic_case_descriptor(diagnostic, frozen, original)
        changed_crop = {**diagnostic, "transform": {"crop_xyxy": [2, 2, 5, 6]}}
        with self.assertRaisesRegex(ValueError, "case descriptor"):
            validate_diagnostic_case_descriptor(changed_crop, frozen, original)
        changed_input = {**diagnostic, "input_sha256": "c" * 64}
        with self.assertRaisesRegex(ValueError, "input differs"):
            validate_diagnostic_case_descriptor(changed_input, frozen, original)

    def test_weight_error_diagnostic_handles_zero_reference_norm(self):
        nonzero = tensor_error_metrics([1.0, 2.0], [1.0, 3.0], 1e-12)
        self.assertAlmostEqual(nonzero["normalized_l2"], (1 / 10) ** 0.5)
        self.assertEqual(nonzero["maximum_absolute_error"], 1.0)
        zero = tensor_error_metrics([0.0001, -0.0001], [0.0, 0.0], 1e-12)
        self.assertIsNone(zero["normalized_l2"])
        self.assertEqual(zero["maximum_absolute_error"], 0.0001)


if __name__ == "__main__":
    unittest.main()
