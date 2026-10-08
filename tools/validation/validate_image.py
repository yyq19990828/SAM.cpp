#!/usr/bin/env python3
"""Run test_image and enforce the frozen M1 tensor/detection acceptance gates."""

import argparse
import math
from pathlib import Path
import sys
import re
import subprocess
import tempfile

import gguf

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.sam3_artifacts import (BPE_SHA256, PAB_REVISION, REQUIRED_TENSORS, SAM3_REVISION,
                           artifact_path, load_case_manifest, read_array, read_json, read_tensor_index,
                           sha256_file, validate_case_manifest, validate_cuda_oracle_provenance, write_json)
from tools.convert.sam3_gguf import (QUANTIZATION_MODULES as GGUF_QUANTIZATION_MODULES,
                       QUANTIZATION_VERSION, LEGACY_GGML_QUANTIZER_BUILD_COMMITS, canonical_quantization_modules, inspect_tensors,
                       quantization_module_for_tensor, quantization_reason_for_tensor,
                       quantization_profile as storage_quantization_profile, read_gguf,
                       validate_metadata, tensor_schema)


GATES = {
    "normalized_l2_f32": 1e-3, "normalized_l2_f16": 2e-2,
    "zero_norm_limit": 1e-12, "zero_norm_max_abs": 1e-5,
    "preprocessed_max_abs": 2 / 255, "high_score_min": 0.6,
    "mask_iou_f32": 0.98, "mask_iou_f16": 0.95, "score_max_abs": 0.02,
    "box_dimension_fraction": 0.01, "low_score_max": 0.4, "score_threshold": 0.5,
}

QUANTIZATION_GATES_PATH = Path(__file__).resolve().parents[2] / "tests/data/sam3-quantization-gates.json"
VISION_QUANTIZATION_GATES_PATH = Path(__file__).resolve().parents[2] / "tests/data/sam3-vision-quantization-gates.json"
OUTPUT_QUALITY_GATES_PATH = Path(__file__).resolve().parents[2] / "tests/data/sam3-quantization-output-gates.json"
FULL_QUANTIZATION_GATES_PATH = Path(__file__).resolve().parents[2] / "tests/data/sam3-full-image-quantization-gates.json"
FULL_OUTPUT_QUALITY_GATES_PATH = Path(__file__).resolve().parents[2] / "tests/data/sam3-full-image-output-gates.json"
QUANTIZATION_GATES = read_json(QUANTIZATION_GATES_PATH)
QUANTIZATION_GATES_SHA256 = sha256_file(QUANTIZATION_GATES_PATH)
VISION_QUANTIZATION_GATES = read_json(VISION_QUANTIZATION_GATES_PATH)
VISION_QUANTIZATION_GATES_SHA256 = sha256_file(VISION_QUANTIZATION_GATES_PATH)
OUTPUT_QUALITY_GATES = read_json(OUTPUT_QUALITY_GATES_PATH)
OUTPUT_QUALITY_GATES_SHA256 = sha256_file(OUTPUT_QUALITY_GATES_PATH)
FULL_QUANTIZATION_GATES = read_json(FULL_QUANTIZATION_GATES_PATH)
FULL_QUANTIZATION_GATES_SHA256 = sha256_file(FULL_QUANTIZATION_GATES_PATH)
FULL_OUTPUT_QUALITY_GATES = read_json(FULL_OUTPUT_QUALITY_GATES_PATH)
FULL_OUTPUT_QUALITY_GATES_SHA256 = sha256_file(FULL_OUTPUT_QUALITY_GATES_PATH)
FROZEN_QUANTIZATION_GATES_SHA256 = "819d9de1ebe638d223db9d57c8ac652b36707fae3d36124735bdc65d681ba2cc"
FROZEN_VISION_QUANTIZATION_GATES_SHA256 = "6bdb91602e8a89e8b29b208282b846b541d8fbe499ae36955e4f905e9eff7a93"
FROZEN_OUTPUT_QUALITY_GATES_SHA256 = "c69dd6fee4abc789e421c407c7f954e3fdfd7f04a2a79288d3192816385f11e8"
FROZEN_FULL_QUANTIZATION_GATES_SHA256 = "17447e93cbec08233a9be3b50f43574da057d5ce05f79493e662fb757071d0bb"
FROZEN_FULL_OUTPUT_QUALITY_GATES_SHA256 = "f796add8c03adc078797c6c5f93bb439c214044007578b40a63af5d99a342aa3"
PINNED_GGML_REVISION = "353b63b439f27ab2cc19dac97ab1681ba6d2d084"
PINNED_GGML_VERSION = "0.25.3"
GGML_PRECISION_PATCH = Path(__file__).resolve().parents[2] / "cmake/patches/ggml-precise-metal.patch"
SAM_LEGACY_PATCHED_GGML_BUILD_COMMIT = f"{PINNED_GGML_REVISION[:8]}-sam-{sha256_file(GGML_PRECISION_PATCH)[:12]}"
GGML_CUDA_PRECISION_PATCH = Path(__file__).resolve().parents[2] / "cmake/patches/ggml-precise-cuda.patch"
SAM_PATCHED_GGML_BUILD_COMMIT = f"{SAM_LEGACY_PATCHED_GGML_BUILD_COMMIT}-{sha256_file(GGML_CUDA_PRECISION_PATCH)[:12]}"
CPU_QUANTIZED_ARITHMETIC_PROFILE = "ggml-quantized-weights-f32-v1"
METAL_QUANTIZED_ARITHMETIC_PROFILE = "ggml-quantized-native-v1"
CUDA_QUANTIZED_ARITHMETIC_PROFILE = "ggml-quantized-cuda-native-v1"
CUDA_F16_ARITHMETIC_PROFILE = "ggml-cuda-f16-v1"
CUDA_F16_QUANTIZED_ARITHMETIC_PROFILE = "ggml-quantized-cuda-f16-v1"
QUANTIZED_PRECISIONS = frozenset({"q8_0", "q6_k", "q5_k", "q4_k"})
_EXPECTED_PRECISION_NAMES = {"q8_0", "q6_k", "q5_k", "q4_k"}
QUANTIZATION_MODULES = tuple(GGUF_QUANTIZATION_MODULES)
FULL_QUANTIZATION_MODULES = list(QUANTIZATION_MODULES)
CUSTOM_MODULE_PROFILE_PREFIX = "image-modules-linear-"
_GATE_FILES = (
    ("legacy", QUANTIZATION_GATES, QUANTIZATION_GATES_SHA256,
     FROZEN_QUANTIZATION_GATES_SHA256, "sam3-image-quantization-v1",
     "frozen-before-model-runs", "image-linear-"),
    ("vision", VISION_QUANTIZATION_GATES, VISION_QUANTIZATION_GATES_SHA256,
     FROZEN_VISION_QUANTIZATION_GATES_SHA256, "sam3-vision-image-quantization-v1",
     "frozen-before-new-profile-model-runs", "image-vision-linear-"),
)
QUANTIZATION_PROFILES_BY_STORAGE_PROFILE = {}

for family, gates, digest, frozen_digest, gate_set, status, profile_prefix in _GATE_FILES:
    profiles = gates.get("profiles", {})
    if (digest != frozen_digest or gates.get("schema_version") != 1
            or gates.get("gate_set") != gate_set or gates.get("status") != status
            or len(profiles) != 4 or set(profiles) != _EXPECTED_PRECISION_NAMES):
        raise ValueError(f"missing, changed, or unfrozen {family} SAM 3 image quantization gates")
    if family == "vision" and gates.get("parent_gate_sha256") != FROZEN_QUANTIZATION_GATES_SHA256:
        raise ValueError("vision gates do not identify their frozen original quality thresholds")
    for precision, profile in profiles.items():
        if (profile.get("precision") != precision
                or not isinstance(profile.get("storage_profile"), str)
                or not profile["storage_profile"].startswith(profile_prefix)
                or profile["storage_profile"] in QUANTIZATION_PROFILES_BY_STORAGE_PROFILE):
            raise ValueError(f"{family} gates contain an invalid or duplicate storage profile")
        QUANTIZATION_PROFILES_BY_STORAGE_PROFILE[profile["storage_profile"]] = {
            "family": family, "precision": precision, "profile": profile,
            "gate_set": gates["gate_set"], "gates": gates, "gates_sha256": digest}

SCHEMA3_QUANTIZATION_PROFILES_BY_STORAGE_PROFILE = dict(QUANTIZATION_PROFILES_BY_STORAGE_PROFILE)

FULL_GATE_PROFILES = FULL_QUANTIZATION_GATES.get("profiles", {})
if (FULL_QUANTIZATION_GATES_SHA256 != FROZEN_FULL_QUANTIZATION_GATES_SHA256
        or FULL_QUANTIZATION_GATES.get("schema_version") != 1
        or FULL_QUANTIZATION_GATES.get("gate_set") != "sam3-full-image-quantization-v1"
        or FULL_QUANTIZATION_GATES.get("status") != "frozen-before-full-profile-model-runs"
        or FULL_QUANTIZATION_GATES.get("parent_gate_sha256") != FROZEN_QUANTIZATION_GATES_SHA256
        or FULL_QUANTIZATION_GATES.get("quantization_modules") != FULL_QUANTIZATION_MODULES
        or set(FULL_GATE_PROFILES) != _EXPECTED_PRECISION_NAMES):
    raise ValueError("missing, changed, or unfrozen full-image quantization gates")
for precision, profile in FULL_GATE_PROFILES.items():
    old = QUANTIZATION_GATES["profiles"][precision]
    if ({key: value for key, value in profile.items() if key != "storage_profile"}
            != {key: value for key, value in old.items() if key != "storage_profile"}
            or profile.get("storage_profile") != f"image-full-linear-{precision}-v1"):
        raise ValueError("full-image quantization gates changed the frozen per-precision thresholds")
    QUANTIZATION_PROFILES_BY_STORAGE_PROFILE[profile["storage_profile"]] = {
        "family": "full", "precision": precision,
        "profile": {**profile, "profile_status": "candidate",
                    "quantization_modules": FULL_QUANTIZATION_MODULES},
        "gate_set": FULL_QUANTIZATION_GATES["gate_set"],
        "gates": FULL_QUANTIZATION_GATES,
        "gates_sha256": FULL_QUANTIZATION_GATES_SHA256}

for precision in _EXPECTED_PRECISION_NAMES:
    old = QUANTIZATION_GATES["profiles"][precision]
    vision = VISION_QUANTIZATION_GATES["profiles"][precision]
    if {key: value for key, value in old.items() if key != "storage_profile"} != {
            key: value for key, value in vision.items() if key != "storage_profile"}:
        raise ValueError("vision quantization gates changed the original quality thresholds")

OUTPUT_QUALITY_PROFILES = OUTPUT_QUALITY_GATES.get("profiles", {})
if (OUTPUT_QUALITY_GATES_SHA256 != FROZEN_OUTPUT_QUALITY_GATES_SHA256
        or OUTPUT_QUALITY_GATES.get("schema_version") != 1
        or OUTPUT_QUALITY_GATES.get("gate_set") != "sam3-image-output-quality-v1"
        or OUTPUT_QUALITY_GATES.get("status") != "frozen-before-output-quality-regrading"
        or OUTPUT_QUALITY_GATES.get("parent_tensor_gate_sha256") != {
            QUANTIZATION_GATES["gate_set"]: FROZEN_QUANTIZATION_GATES_SHA256,
            VISION_QUANTIZATION_GATES["gate_set"]: FROZEN_VISION_QUANTIZATION_GATES_SHA256}
        or set(OUTPUT_QUALITY_PROFILES) != set(SCHEMA3_QUANTIZATION_PROFILES_BY_STORAGE_PROFILE)
        or OUTPUT_QUALITY_GATES.get("hard_requirements") != {
            "frozen_original_input_bytes": True,
            "official_token_ids_equal": True,
            "canonical_tensor_inventory_and_shapes": True,
            "all_required_tensor_outputs_finite": True,
            "preprocessed_image_max_abs": 2 / 255,
            "score_threshold": 0.5,
            "high_confidence_reference_score_min": 0.6,
            "low_confidence_reference_score_max": 0.4,
            "threshold_adjacent_interval": [0.4, 0.6],
            "selected_query_sets_match_outside_threshold_adjacent_interval": True,
            "every_high_confidence_reference_query_remains_selected": True,
            "no_low_confidence_reference_query_becomes_selected": True,
        }
        or OUTPUT_QUALITY_GATES.get("selected_detection_quality", {}).get("tensor_fidelity_is_primary_pass_gate") is not False):
    raise ValueError("missing, changed, or unfrozen quantized image output-quality gates")

for storage_profile, raw_selection in SCHEMA3_QUANTIZATION_PROFILES_BY_STORAGE_PROFILE.items():
    output_profile = OUTPUT_QUALITY_PROFILES[storage_profile]
    raw_profile = raw_selection["profile"]
    if (output_profile.get("precision") != raw_selection["precision"]
            or output_profile.get("mask_iou_min") != raw_profile["high_confidence_mask_iou_min"]
            or output_profile.get("score_absolute_error_max") != raw_profile["score_absolute_error_max"]
            or output_profile.get("box_dimension_fraction_max") != raw_profile["box_dimension_fraction_max"]):
        raise ValueError("output-quality thresholds must retain the original per-profile object values")
    if (OUTPUT_QUALITY_GATES["hard_requirements"].get("preprocessed_image_max_abs")
            != raw_selection["gates"]["comparison"]["preprocessed_image_max_abs"]):
        raise ValueError("output-quality preprocessing limit differs from the frozen tensor gate")

FULL_OUTPUT_QUALITY_PROFILES = FULL_OUTPUT_QUALITY_GATES.get("profiles", {})
if (FULL_OUTPUT_QUALITY_GATES_SHA256 != FROZEN_FULL_OUTPUT_QUALITY_GATES_SHA256
        or FULL_OUTPUT_QUALITY_GATES.get("schema_version") != 1
        or FULL_OUTPUT_QUALITY_GATES.get("gate_set") != "sam3-full-image-output-quality-v1"
        or FULL_OUTPUT_QUALITY_GATES.get("status") != "frozen-before-full-profile-model-runs"
        or FULL_OUTPUT_QUALITY_GATES.get("parent_tensor_gate_sha256") != FROZEN_FULL_QUANTIZATION_GATES_SHA256
        or FULL_OUTPUT_QUALITY_GATES.get("quantization_modules") != FULL_QUANTIZATION_MODULES
        or set(FULL_OUTPUT_QUALITY_PROFILES)
        != {profile["storage_profile"] for profile in FULL_GATE_PROFILES.values()}
        or FULL_OUTPUT_QUALITY_GATES.get("hard_requirements") != OUTPUT_QUALITY_GATES.get("hard_requirements")
        or FULL_OUTPUT_QUALITY_GATES.get("selected_detection_quality")
        != OUTPUT_QUALITY_GATES.get("selected_detection_quality")):
    raise ValueError("missing, changed, or unfrozen full-image output-quality gates")
for storage_profile, raw_selection in ((profile["storage_profile"], profile)
                                       for profile in FULL_GATE_PROFILES.values()):
    output_profile = FULL_OUTPUT_QUALITY_PROFILES[storage_profile]
    if (output_profile.get("precision") != raw_selection["precision"]
            or output_profile.get("mask_iou_min") != raw_selection["high_confidence_mask_iou_min"]
            or output_profile.get("score_absolute_error_max") != raw_selection["score_absolute_error_max"]
            or output_profile.get("box_dimension_fraction_max") != raw_selection["box_dimension_fraction_max"]
            or FULL_OUTPUT_QUALITY_GATES["hard_requirements"].get("preprocessed_image_max_abs")
            != FULL_QUANTIZATION_GATES["comparison"]["preprocessed_image_max_abs"]):
        raise ValueError("full-image output-quality thresholds differ from the frozen tensor gate")

def canonical_module_selection(modules):
    if not isinstance(modules, list):
        raise ValueError("schema-4 quantization_modules must be a JSON array")
    canonical = canonical_quantization_modules(modules)
    if modules != canonical:
        raise ValueError("quantization_modules must use the canonical module order")
    return canonical


def validate_gguf_module_metadata(reader, modules):
    field = reader.get_field("sam.quantization.modules")
    if (field is None or field.types != [gguf.GGUFValueType.STRING]
            or field.contents() != ",".join(modules)):
        raise ValueError("GGUF module metadata disagrees with the schema-4 sidecar quantization_modules")


def quantization_selection(precision, storage_profile, quantization_modules=None):
    selection = QUANTIZATION_PROFILES_BY_STORAGE_PROFILE.get(storage_profile)
    if selection is not None:
        if selection["precision"] != precision:
            raise ValueError(f"unsupported quantized precision/storage profile pair: {precision}/{storage_profile}")
        if selection["family"] == "full":
            if quantization_modules is not None and canonical_module_selection(quantization_modules) != FULL_QUANTIZATION_MODULES:
                raise ValueError("full storage profile requires all four quantization modules")
        elif quantization_modules is not None:
            raise ValueError("schema-3 quantization profiles do not accept quantization_modules")
        return selection
    if not isinstance(storage_profile, str):
        raise ValueError(f"unsupported quantized precision/storage profile pair: {precision}/{storage_profile}")
    expected_custom = f"{CUSTOM_MODULE_PROFILE_PREFIX}{precision}-v1"
    if storage_profile != expected_custom or precision not in QUANTIZED_PRECISIONS:
        raise ValueError(f"unsupported quantized precision/storage profile pair: {precision}/{storage_profile}")
    modules = canonical_module_selection(quantization_modules)
    full = FULL_GATE_PROFILES[precision]
    profile = {**full, "storage_profile": storage_profile, "profile_status": "diagnostic",
               "quantization_modules": modules}
    return {"family": "custom", "precision": precision, "profile": profile,
            "gate_set": FULL_QUANTIZATION_GATES["gate_set"], "gates": FULL_QUANTIZATION_GATES,
            "gates_sha256": FULL_QUANTIZATION_GATES_SHA256}


def quantization_profile(precision, storage_profile=None, quantization_modules=None):
    if storage_profile is None:
        raise ValueError("an exact versioned storage profile is required to select quantization gates")
    return quantization_selection(precision, storage_profile, quantization_modules)["profile"]


def output_quality_selection(precision, storage_profile, quantization_modules=None):
    raw_selection = quantization_selection(precision, storage_profile, quantization_modules)
    if raw_selection["family"] in ("full", "custom"):
        output_gates, output_profiles = FULL_OUTPUT_QUALITY_GATES, FULL_OUTPUT_QUALITY_PROFILES
        output_key = f"image-full-linear-{precision}-v1"
    else:
        output_gates, output_profiles = OUTPUT_QUALITY_GATES, OUTPUT_QUALITY_PROFILES
        output_key = storage_profile
    return {"precision": precision, "storage_profile": storage_profile,
            "profile": output_profiles[output_key],
            "gates": output_gates, "gates_sha256": FULL_OUTPUT_QUALITY_GATES_SHA256
            if raw_selection["family"] in ("full", "custom") else OUTPUT_QUALITY_GATES_SHA256,
            "raw_tensor_selection": raw_selection}


def quantization_release_eligibility(selection, model, reference, passed):
    custom = selection["family"] == "custom"
    milestone_eligible = reference.get("eligible_for_milestone") is True and not custom
    return {"eligible_for_milestone": milestone_eligible,
            "release_support_eligible": (passed is True and milestone_eligible
                                         and model.get("profile_status") != "diagnostic")}


def validate_cuda_compute_mode(backend, cuda_compute):
    if cuda_compute not in ("f32", "f16") or (cuda_compute == "f16" and backend != "cuda"):
        raise ValueError("CUDA compute must be f32 or explicit CUDA f16")


def allowed_runtime_arithmetic_profiles(backend, allow_historical_cpu_native=False, cuda_compute="f32"):
    validate_cuda_compute_mode(backend, cuda_compute)
    if backend == "cuda":
        return {CUDA_F16_QUANTIZED_ARITHMETIC_PROFILE if cuda_compute == "f16" else CUDA_QUANTIZED_ARITHMETIC_PROFILE}
    if backend == "metal":
        return {METAL_QUANTIZED_ARITHMETIC_PROFILE}
    if backend == "cpu":
        profiles = {CPU_QUANTIZED_ARITHMETIC_PROFILE}
        if allow_historical_cpu_native:
            profiles.add(METAL_QUANTIZED_ARITHMETIC_PROFILE)
        return profiles
    raise ValueError(f"unsupported quantized backend: {backend}")


def runtime_arithmetic_profiles_valid(backend, profiles, allow_historical_cpu_native=False, cuda_compute="f32"):
    observed = set(profiles)
    return (len(observed) == 1
            and observed.issubset(allowed_runtime_arithmetic_profiles(backend, allow_historical_cpu_native, cuda_compute)))


def validate_existing_validation_profile(receipt, backend, model, reference_manifest_sha256, selection):
    gates = selection["gates"]
    dual_axis_fields = ("output_quality_gate_set", "output_quality_gates_sha256",
                        "tensor_fidelity_gate_set", "tensor_fidelity_gates_sha256")
    dual_axis = any(key in receipt for key in dual_axis_fields)
    if dual_axis:
        if any(key not in receipt for key in dual_axis_fields):
            raise ValueError("existing validation receipt has incomplete output/tensor gate identities")
        schema4 = model.get("sam_schema_version") == 4
        modules = model.get("quantization_modules") if schema4 else None
        output_selection = output_quality_selection(
            model.get("precision"), model.get("storage_profile"), modules)
        output_gates = output_selection["gates"]
        expected_gate_fields = {
            "gate_set": output_gates["gate_set"],
            "gates_sha256": output_selection["gates_sha256"],
            "output_quality_gate_set": output_gates["gate_set"],
            "output_quality_gates_sha256": output_selection["gates_sha256"],
            "tensor_fidelity_gate_set": gates["gate_set"],
            "tensor_fidelity_gates_sha256": selection["gates_sha256"],
        }
        if any(receipt.get(key) != value for key, value in expected_gate_fields.items()):
            raise ValueError("existing validation receipt uses different output-quality or tensor-fidelity gates")
    elif (receipt.get("gate_set") != gates["gate_set"]
          or receipt.get("gates_sha256") != selection["gates_sha256"]):
        raise ValueError("existing validation receipt uses a different profile, gate, model, oracle, or backend")
    if (receipt.get("schema_version") != 1 or receipt.get("backend") != backend
            or receipt.get("precision") != model.get("precision")
            or receipt.get("model_sha256") != model.get("output", {}).get("sha256")
            or receipt.get("checkpoint_sha256") != model.get("checkpoint", {}).get("sha256")
            or receipt.get("reference_manifest_sha256") != reference_manifest_sha256
            or receipt.get("storage_profile") != model.get("storage_profile")):
        raise ValueError("existing validation receipt uses a different profile, gate, model, oracle, or backend")


def validate_existing_run_receipt(existing_output, metrics, model, tensor_selection):
    """Verify the producer receipt before regrading saved outputs without inference."""
    run_path = Path(existing_output).resolve().parent / "run.json"
    if not run_path.is_file():
        raise ValueError("retained outputs lack their immutable run receipt")
    run = read_json(run_path)
    before = run.get("binary_and_libraries_before", {}).get("sha256")
    after = run.get("binary_and_libraries_after", {}).get("sha256")
    artifacts = metrics.get("run_artifact_sha256")
    if (run.get("complete") is not True or run.get("metrics_sha256") != sha256_file(Path(existing_output) / "metrics.json")
            or run.get("model_sha256") != model.get("output", {}).get("sha256")
            or run.get("gate_sha256") != tensor_selection["gates_sha256"]
            or run.get("model_unchanged") is not True or run.get("gate_unchanged") is not True
            or run.get("binary_and_libraries_unchanged") is not True
            or not isinstance(before, dict) or before != after
            or not isinstance(artifacts, dict) or artifacts != before):
        raise ValueError("run receipt does not bind the retained metrics, model, raw gate, and binary/library hashes")
    source_before, source_after = run.get("source_sha256"), run.get("source_sha256_after")
    if (not isinstance(source_before, dict) or not isinstance(source_after, dict)
            or set(source_before) != set(source_after)
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in source_before.values())
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in source_after.values())):
        raise ValueError("run receipt has incomplete source hash snapshots")
    changed_sources = [path for path, digest in source_before.items() if source_after[path] != digest]
    ignored_driver = str(Path(__file__).resolve().parents[2] / "build/quantization/20261003/quant_acceptance.py")
    if any(path != ignored_driver for path in changed_sources):
        raise ValueError(f"retained run source hashes changed outside the acceptance driver: {changed_sources}")
    return {"run_receipt_sha256": sha256_file(run_path), "source_hash_delta_paths": changed_sources,
            "binary_and_library_sha256": before}


def validate_schema4_tensor_module_records(tensors, precision, storage_profile, modules):
    """Recompute the schema-4 module and allocation reason for every sidecar tensor."""
    modules = canonical_module_selection(modules)
    quantization_selection(precision, storage_profile, modules)
    for item in tensors:
        name = item.get("name")
        shape = item.get("shape")
        if not isinstance(name, str) or not isinstance(shape, list):
            raise ValueError("schema-4 tensor sidecar lacks a name or source shape")
        expected_module = quantization_module_for_tensor(name)
        expected_reason = quantization_reason_for_tensor(
            name, shape, precision, storage_profile, modules)
        if (item.get("module") != expected_module
                or item.get("quantization_reason") != expected_reason):
            raise ValueError(f"{name}: schema-4 module/reason disagrees with the converter allocation policy")


def validate_quantization_sidecar(model, allow_custom_quantization=False):
    precision = model["precision"]
    storage_profile = model.get("storage_profile")
    family = QUANTIZATION_PROFILES_BY_STORAGE_PROFILE.get(storage_profile, {}).get("family")
    if storage_profile == f"{CUSTOM_MODULE_PROFILE_PREFIX}{precision}-v1":
        family = "custom"
    schema4 = family in ("full", "custom")
    modules = canonical_module_selection(model.get("quantization_modules")) if schema4 else None
    if family == "custom" and not allow_custom_quantization:
        raise ValueError("custom module quantization is diagnostic only; pass --allow-custom-quantization explicitly")
    selection = quantization_selection(precision, storage_profile, modules)
    expected = storage_quantization_profile(precision, storage_profile, modules)
    schema_version = model.get("sam_schema_version", expected.get("schema_version"))
    if schema_version != expected.get("schema_version"):
        raise ValueError("quantized model schema disagrees with its exact storage profile")
    if not schema4 and "quantization_modules" in model:
        raise ValueError("schema-3 sidecars must not declare schema-4 quantization_modules")
    quantization = model.get("quantization", {})
    expected_quantization = {
        "version": QUANTIZATION_VERSION,
        "ggml_quantization_version": QUANTIZATION_VERSION,
        "gguf_file_type": expected["file_type"],
        "ggml_type": expected["ggml_type"],
        "block_elements": expected["block_elements"],
        "block_bytes": expected["block_bytes"],
        "row_block_fallback": expected.get("fallback_type"),
        "quantize_text_linear": ("text" in modules if schema4 else expected["quantize_text_linear"]),
    }
    lock_hash = model.get("requirements_lock_sha256")
    if (quantization != expected_quantization
            or model.get("profile_status") != expected["profile_status"]
            or model.get("converter_revision") != PAB_REVISION
            or model.get("gguf_package_version") != "0.19.0"
            or any(not isinstance(model.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", model[key])
                   for key in ("converter_sha256", "gguf_helper_sha256"))
            or not isinstance(lock_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", lock_hash)):
        raise ValueError("quantized model sidecar does not match the pinned conversion/profile contract")
    if expected["storage_profile"] != storage_profile:
        raise ValueError("quantized model sidecar storage profile disagrees with the quantization profile")
    if schema4 and modules != expected["modules"]:
        raise ValueError("schema-4 sidecar quantization_modules disagree with the exact storage profile")
    quantizer = model.get("quantizer", {})
    if precision == "q8_0":
        if (quantizer.get("implementation") != "gguf-py"
                or quantizer.get("version") != model.get("gguf_package_version")
                or quantizer.get("module") != "gguf.quants.quantize"
                or quantizer.get("ggml_quantization_version") != QUANTIZATION_VERSION):
            raise ValueError("Q8_0 sidecar has incomplete gguf-py quantizer provenance")
    else:
        helper = quantizer.get("helper", {})
        library = quantizer.get("ggml_library", {})
        quantize_library = quantizer.get("ggml_quantize_library", {})
        if (quantizer.get("implementation") != "ggml_quantize_chunk"
                or quantizer.get("fallback_implementation") != "gguf-py"
                or quantizer.get("fallback_version") != model.get("gguf_package_version")
                or quantizer.get("ggml_revision") != PINNED_GGML_REVISION
                or quantizer.get("ggml_build_commit") not in (
                    PINNED_GGML_REVISION, SAM_LEGACY_PATCHED_GGML_BUILD_COMMIT, SAM_PATCHED_GGML_BUILD_COMMIT,
                    *LEGACY_GGML_QUANTIZER_BUILD_COMMITS)
                or quantizer.get("ggml_version") != PINNED_GGML_VERSION
                or quantizer.get("ggml_quantization_version") != QUANTIZATION_VERSION
                or not isinstance(helper.get("file"), str) or not helper["file"]
                or not isinstance(library.get("file"), str) or not library["file"]
                or not isinstance(quantize_library.get("file"), str) or not quantize_library["file"]
                or not isinstance(helper.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", helper["sha256"])
                or not isinstance(library.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", library["sha256"])
                or not isinstance(quantize_library.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", quantize_library["sha256"])):
            raise ValueError("K-quant sidecar has incomplete pinned GGML encoder/provider provenance")


def tensor_error(actual, reference, zero_norm_limit=None):
    import numpy as np

    if actual.shape != reference.shape or actual.size == 0:
        raise ValueError("tensor comparison requires equal, nonempty shapes")
    actual, reference = actual.reshape(-1), reference.reshape(-1)
    error_squared = reference_squared = maximum_absolute = 0.0
    for offset in range(0, actual.size, 1024 * 1024):
        a = np.asarray(actual[offset:offset + 1024 * 1024], dtype=np.float64)
        r = np.asarray(reference[offset:offset + 1024 * 1024], dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(r).all():
            raise ValueError("tensor comparison received non-finite values")
        delta = a - r
        error_squared += float(np.dot(delta, delta))
        reference_squared += float(np.dot(r, r))
        maximum_absolute = max(maximum_absolute, float(np.abs(delta).max()))
    reference_norm = math.sqrt(reference_squared)
    zero_norm_limit = GATES["zero_norm_limit"] if zero_norm_limit is None else zero_norm_limit
    normalized_l2 = math.sqrt(error_squared) / reference_norm if reference_norm > zero_norm_limit else None
    return {"normalized_l2": normalized_l2, "reference_norm": reference_norm,
            "maximum_absolute_error": maximum_absolute}


def mask_iou(actual, reference):
    import numpy as np

    if actual.shape != reference.shape:
        raise ValueError("mask dimensions differ")
    intersection = int(np.count_nonzero((actual != 0) & (reference != 0)))
    union = int(np.count_nonzero((actual != 0) | (reference != 0)))
    return intersection / union if union else 1.0


def query_scores(tensors):
    import numpy as np

    def sigmoid(value):
        value = np.clip(np.asarray(value, dtype=np.float64), -700, 700)
        return 1 / (1 + np.exp(-value))

    return (sigmoid(tensors["class_logits"]).reshape(200)
            * float(sigmoid(tensors["presence_logits"]).reshape(-1)[0]))


def read_results(directory, scores, case, require_hash):
    result = read_json(Path(directory) / "results.json")
    if result.get("schema_version") != 1 or result.get("prompt") != case["prompt"] or result.get("score_threshold") != 0.5:
        raise ValueError("result schema, prompt, or score threshold differs from the case")
    width, height = result.get("width"), result.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("result has invalid image dimensions")
    detections = result.get("detections")
    if not isinstance(detections, list):
        raise ValueError("result lacks detections (an empty list is a valid empty result)")
    by_query = {}
    for detection in detections:
        query = detection.get("query_index")
        score = detection.get("score")
        box = detection.get("box")
        if type(query) is not int or not 0 <= query < 200 or query in by_query:
            raise ValueError(f"invalid or duplicate query index: {query}")
        if type(score) not in (float, int) or not math.isfinite(score) or not 0.5 < score <= 1:
            raise ValueError(f"query {query}: invalid detected score")
        if abs(score - scores[query]) > 1e-5:
            raise ValueError(f"query {query}: result score disagrees with raw logits")
        if (not isinstance(box, list) or len(box) != 4
                or any(type(value) not in (float, int) or not math.isfinite(value) for value in box)
                or box[0] > box[2] or box[1] > box[3]):
            raise ValueError(f"query {query}: invalid XYXY box")
        metadata = detection.get("mask")
        if not isinstance(metadata, dict) or metadata.get("shape") != [height, width]:
            raise ValueError(f"query {query}: mask dimensions do not match the image")
        read_array(directory, metadata, "uint8", require_hash)
        by_query[query] = detection
    expected = {query for query, score in enumerate(scores) if score > 0.5}
    if set(by_query) != expected:
        raise ValueError(f"selected queries disagree with raw logits: expected={sorted(expected)}, got={sorted(by_query)}")
    return result, by_query


def compare_case(reference_directory, actual_directory, case, precision, backend=None, model=None,
                 allow_historical_cpu_native=False, cuda_device=0, cuda_compute="f32"):
    import numpy as np

    validate_cuda_compute_mode(backend, cuda_compute)
    fast_compute = cuda_compute == "f16"
    quantized = precision in QUANTIZED_PRECISIONS
    if quantized:
        selected_storage_profile = ((model or {}).get("storage_profile") if model is not None
                                    else f"image-linear-{precision}-v1")
        schema4 = (model or {}).get("sam_schema_version") == 4
        selected_modules = (model or {}).get("quantization_modules") if schema4 else None
        selection = quantization_selection(precision, selected_storage_profile, selected_modules)
        tensor_profile, tensor_gates = selection["profile"], selection["gates"]
        output_selection = output_quality_selection(precision, selected_storage_profile, selected_modules)
        output_profile, output_gates = output_selection["profile"], output_selection["gates"]
    else:
        selected_storage_profile = ""
        tensor_profile = tensor_gates = output_profile = output_gates = output_selection = None
    gate_precision = "f16" if precision == "hybrid" else precision
    output_gate_precision = "f16" if fast_compute else gate_precision
    reference_index = read_tensor_index(reference_directory)
    actual_index = read_tensor_index(actual_directory, require_hash=False)
    if reference_index["token_ids"] != actual_index["token_ids"]:
        raise ValueError("token IDs differ from the official tokenizer")
    reference_tensors, actual_tensors = {}, {}
    errors, tensor_failures, output_failures = {}, [], []
    tensor_comparison = tensor_gates["comparison"] if quantized else None
    output_hard = output_gates["hard_requirements"] if quantized else None
    tolerance = tensor_profile["normalized_l2_max"] if quantized else GATES[f"normalized_l2_{gate_precision}"]
    zero_norm_limit = tensor_comparison["zero_norm_reference_threshold"] if quantized else GATES["zero_norm_limit"]
    zero_norm_max_abs = tensor_profile["zero_norm_max_abs"] if quantized else GATES["zero_norm_max_abs"]
    for name in REQUIRED_TENSORS:
        reference = read_array(reference_directory, reference_index["tensors"][name], "float32")
        actual = read_array(actual_directory, actual_index["tensors"][name], "float32", require_hash=False)
        reference_tensors[name], actual_tensors[name] = reference, actual
        metrics = tensor_error(actual, reference, zero_norm_limit)
        if name == "preprocessed_image":
            tensor_limit = tensor_comparison["preprocessed_image_max_abs"] if quantized else GATES["preprocessed_max_abs"]
            tensor_passed = metrics["maximum_absolute_error"] <= tensor_limit
            output_passed = (metrics["maximum_absolute_error"] <= output_hard["preprocessed_image_max_abs"]
                             if quantized else tensor_passed)
        elif metrics["normalized_l2"] is None:
            tensor_passed = metrics["maximum_absolute_error"] <= zero_norm_max_abs
            output_passed = True
        else:
            tensor_passed = metrics["normalized_l2"] <= tolerance
            output_passed = True
        errors[name] = {**metrics, "passed": tensor_passed,
                        **({"tensor_fidelity_passed": tensor_passed,
                           "output_quality_passed": output_passed} if quantized or fast_compute else {})}
        if not tensor_passed:
            tensor_failures.append(f"{name}: tensor fidelity tolerance exceeded")
        if (quantized or fast_compute) and not output_passed:
            output_failures.append(f"{name}: preprocessing output-quality limit exceeded")
    reference_scores = query_scores(reference_tensors)
    actual_scores = query_scores(actual_tensors)
    reference_result, reference_detections = read_results(reference_directory, reference_scores, case, True)
    actual_results_valid = True
    try:
        actual_result, actual_detections = read_results(actual_directory, actual_scores, case, False)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        if not quantized:
            raise
        actual_result, actual_detections = {"runtime": {}}, {}
        actual_results_valid = False
        output_failures.append(f"invalid C++ result structure: {error}")
    if backend and actual_result.get("backend") != backend:
        if quantized:
            output_failures.append("C++ result did not execute on the requested backend")
        else:
            raise ValueError("C++ result did not execute on the requested backend")
    if quantized:
        expected_storage_profile = selected_storage_profile
        allowed_arithmetic_profiles = allowed_runtime_arithmetic_profiles(
            backend or actual_result.get("backend"), allow_historical_cpu_native, cuda_compute)
        runtime_arithmetic_profile = actual_result.get("arithmetic_profile", "")
        if runtime_arithmetic_profile not in allowed_arithmetic_profiles:
            output_failures.append("C++ result arithmetic profile is unsupported for this backend/receipt type")
    else:
        expected_storage_profile = ((model or {}).get("storage_profile", "") if model is not None
                                    else "visual-tracker-f32-v1" if precision == "hybrid" else "")
        runtime_arithmetic_profile = (model or {}).get("arithmetic_profile", "") if model is not None else ""
        if fast_compute:
            runtime_arithmetic_profile = CUDA_F16_ARITHMETIC_PROFILE
    if actual_result.get("precision") != precision:
        if quantized:
            output_failures.append("C++ result precision differs from the converted model")
        else:
            raise ValueError("C++ result precision differs from the converted model")
    if actual_result.get("storage_profile", "") != expected_storage_profile:
        if quantized:
            output_failures.append("C++ result storage profile differs from the converted model")
        else:
            raise ValueError("C++ result storage profile differs from the converted model")
    if quantized and (model or {}).get("sam_schema_version") == 4:
        if actual_result.get("quantization_modules") != (model or {}).get("quantization_modules"):
            output_failures.append("C++ result quantization modules differ from the schema-4 model")
    if not quantized and actual_result.get("arithmetic_profile", "") != runtime_arithmetic_profile:
        raise ValueError("C++ result arithmetic profile differs from the converted model")
    runtime = actual_result.get("runtime", {})
    if backend == "cuda":
        if (runtime.get("cuda_nodes", 0) <= 0 or any(runtime.get(f"{other}_nodes", 0) for other in ("cpu", "metal", "blas"))
                or actual_result.get("cuda_device") != cuda_device or not actual_result.get("device_name")):
            raise ValueError("CUDA acceptance requires work on the selected GPU with zero CPU/Metal/BLAS compute fallback")
    if backend == "metal" and runtime.get("metal_nodes", 0) <= 0:
        if quantized:
            output_failures.append("Metal selection has no evidence of actual graph execution on Metal")
        else:
            raise ValueError("Metal selection has no evidence of actual graph execution on Metal")
    if quantized and backend == "metal" and (runtime.get("cpu_nodes", 0) or runtime.get("blas_nodes", 0)):
        output_failures.append("quantized Metal acceptance requires zero CPU/BLAS graph fallback")
    if quantized and backend == "cpu" and (runtime.get("metal_nodes", 0) or runtime.get("cuda_nodes", 0)):
        output_failures.append("CPU quantized acceptance unexpectedly executed graph nodes on a GPU")
    dimensions_match = ((reference_result["width"], reference_result["height"])
                        == (actual_result.get("width"), actual_result.get("height")))
    if not dimensions_match:
        if quantized:
            output_failures.append("source image dimensions differ")
        else:
            raise ValueError("source image dimensions differ")
    detection_metrics = []
    width, height = reference_result["width"], reference_result["height"]
    high_score_min = output_hard["high_confidence_reference_score_min"] if quantized else GATES["high_score_min"]
    low_score_max = output_hard["low_confidence_reference_score_max"] if quantized else GATES["low_score_max"]
    for query, reference_score in enumerate(reference_scores):
        if quantized and not dimensions_match:
            break
        if reference_score >= high_score_min:
            if query not in actual_detections:
                output_failures.append(f"query {query}: missing high-confidence reference detection")
                continue
            actual = actual_detections[query]
            reference = reference_detections[query]
            iou = mask_iou(read_array(actual_directory, actual["mask"], "uint8", False),
                           read_array(reference_directory, reference["mask"], "uint8"))
            score_error = abs(actual["score"] - reference["score"])
            box_error = np.abs(np.asarray(actual["box"]) - np.asarray(reference["box"]))
            box_fraction = float(np.max(box_error / np.asarray([width, height, width, height])))
            mask_iou_min = output_profile["mask_iou_min"] if quantized else GATES[f"mask_iou_{output_gate_precision}"]
            score_max = output_profile["score_absolute_error_max"] if quantized else GATES["score_max_abs"]
            box_max = output_profile["box_dimension_fraction_max"] if quantized else GATES["box_dimension_fraction"]
            passed = iou >= mask_iou_min and score_error <= score_max and box_fraction <= box_max
            detection_metrics.append({"query_index": query, "mask_iou": iou, "score_absolute_error": score_error,
                                      "box_dimension_fraction": box_fraction, "passed": passed})
            if not passed:
                output_failures.append(f"query {query}: mask/score/box output-quality gate exceeded")
        elif reference_score <= low_score_max and query in actual_detections:
            output_failures.append(f"query {query}: low-confidence reference query became a detection")
    threshold_low, threshold_high = (output_hard["threshold_adjacent_interval"] if quantized else (0.4, 0.6))
    adjacent = [{"query_index": query, "reference_score": float(reference_scores[query]),
                 "actual_score": float(actual_scores[query]), "reference_selected": query in reference_detections,
                 "actual_selected": query in actual_detections}
                for query in range(200) if threshold_low < reference_scores[query] < threshold_high]
    if quantized or fast_compute:
        adjacent_queries = {item["query_index"] for item in adjacent}
        selection_changes = sorted(query for query in range(200)
                                   if query not in adjacent_queries
                                   and ((query in reference_detections) != (query in actual_detections)))
        if selection_changes:
            output_failures.append(f"selected query set changed outside the threshold-adjacent interval: {selection_changes}")
        failures = output_failures
    else:
        failures = tensor_failures + output_failures
    result = {"id": case["id"], "passed": not failures, "tensors": errors,
            "detections": detection_metrics, "threshold_adjacent": adjacent, "runtime": runtime,
            "tokenizer_compatibility_repaired": actual_result.get("tokenizer_compatibility_repaired", False),
            "transfers": runtime.get("transfers", "unavailable"), "failures": failures,
            **({"output_quality_passed": not output_failures,
               "tensor_fidelity_passed": not tensor_failures,
               "output_quality_failures": output_failures,
               "tensor_fidelity_failures": tensor_failures,
               "quantization_profile": tensor_profile["storage_profile"],
               "result_structure_valid": actual_results_valid,
               "gate_set": output_gates["gate_set"],
               "gates_sha256": output_selection["gates_sha256"],
               "tensor_fidelity_gate_set": tensor_gates["gate_set"],
               "tensor_fidelity_gates_sha256": selection["gates_sha256"],
               "runtime_arithmetic_profile": runtime_arithmetic_profile,
               **({"quantization_modules": selected_modules} if selection["family"] in ("full", "custom") else {})} if quantized else {})}
    if fast_compute:
        result.update(cuda_compute=cuda_compute, runtime_arithmetic_profile=runtime_arithmetic_profile,
                      output_quality_passed=not output_failures, tensor_fidelity_passed=not tensor_failures,
                      output_quality_failures=output_failures, tensor_fidelity_failures=tensor_failures,
                      tensor_fidelity_is_release_gate=False,
                      output_gate_precision="quantized-profile" if quantized else "f16")
    return result


def compare_native_arithmetic(dequantized_reference_directory, actual_directory, case, precision, storage_profile,
                              quantization_modules=None):
    """Compare native GGML output to Meta output using the actual GGUF-decoded weights."""
    selection = quantization_selection(precision, storage_profile, quantization_modules)
    profile, gates = selection["profile"], selection["gates"]
    reference_index = read_tensor_index(dequantized_reference_directory)
    actual_index = read_tensor_index(actual_directory, require_hash=False)
    if reference_index["token_ids"] != actual_index["token_ids"]:
        raise ValueError("dequantized Meta diagnostic token IDs differ from the official tokenizer")
    errors = {}
    failures = []
    zero_limit = gates["comparison"]["zero_norm_reference_threshold"]
    limit = profile["native_arithmetic_normalized_l2_max"]
    for name in REQUIRED_TENSORS:
        reference = read_array(dequantized_reference_directory, reference_index["tensors"][name], "float32")
        actual = read_array(actual_directory, actual_index["tensors"][name], "float32", require_hash=False)
        metrics = tensor_error(actual, reference, zero_limit)
        passed = native_arithmetic_tensor_passed(metrics, profile)
        errors[name] = {**metrics, "passed": passed}
        if not passed:
            failures.append(f"{name}: native arithmetic diagnostic tolerance exceeded")
    return {"passed": not failures, "normalized_l2_max": limit,
            "gate_set": gates["gate_set"], "gates_sha256": selection["gates_sha256"],
            "tensors": errors, "failures": failures}


def compare_dequantized_meta_output(original_reference_directory, dequantized_reference_directory, case,
                                    precision, storage_profile, quantization_modules=None):
    """Measure end-to-end compression error while both sides use Meta FP32 arithmetic."""
    import numpy as np

    selection = quantization_selection(precision, storage_profile, quantization_modules)
    profile, gates = selection["profile"], selection["gates"]
    original_index = read_tensor_index(original_reference_directory)
    decoded_index = read_tensor_index(dequantized_reference_directory)
    if original_index["token_ids"] != decoded_index["token_ids"]:
        raise ValueError("dequantized Meta diagnostic token IDs differ from the official tokenizer")
    original_tensors, decoded_tensors = {}, {}
    tensor_metrics, failures = {}, []
    comparison = gates["comparison"]
    zero_limit = comparison["zero_norm_reference_threshold"]
    for name in REQUIRED_TENSORS:
        original = read_array(original_reference_directory, original_index["tensors"][name], "float32")
        decoded = read_array(dequantized_reference_directory, decoded_index["tensors"][name], "float32")
        original_tensors[name], decoded_tensors[name] = original, decoded
        metrics = tensor_error(decoded, original, zero_limit)
        if name == "preprocessed_image":
            passed = metrics["maximum_absolute_error"] <= comparison["preprocessed_image_max_abs"]
        elif metrics["normalized_l2"] is None:
            passed = metrics["maximum_absolute_error"] <= profile["zero_norm_max_abs"]
        else:
            passed = metrics["normalized_l2"] <= profile["normalized_l2_max"]
        tensor_metrics[name] = {**metrics, "passed": passed}
        if not passed:
            failures.append(f"{name}: Meta decoded-weight compression gate exceeded")
    original_scores = query_scores(original_tensors)
    decoded_scores = query_scores(decoded_tensors)
    original_result, original_detections = read_results(original_reference_directory, original_scores, case, True)
    decoded_result, decoded_detections = read_results(dequantized_reference_directory, decoded_scores, case, True)
    if (original_result["width"], original_result["height"]) != (decoded_result["width"], decoded_result["height"]):
        raise ValueError("decoded-weight Meta diagnostic image dimensions differ")
    detections = []
    width, height = original_result["width"], original_result["height"]
    for query, score in enumerate(original_scores):
        if score >= comparison["high_confidence_reference_score_min"]:
            if query not in decoded_detections:
                failures.append(f"query {query}: dequantized Meta lost a high-confidence detection")
                continue
            original = original_detections[query]
            decoded = decoded_detections[query]
            iou = mask_iou(read_array(dequantized_reference_directory, decoded["mask"], "uint8"),
                           read_array(original_reference_directory, original["mask"], "uint8"))
            score_error = abs(decoded["score"] - original["score"])
            box_error = np.abs(np.asarray(decoded["box"]) - np.asarray(original["box"]))
            box_fraction = float(np.max(box_error / np.asarray([width, height, width, height])))
            passed = (iou >= profile["high_confidence_mask_iou_min"]
                      and score_error <= profile["score_absolute_error_max"]
                      and box_fraction <= profile["box_dimension_fraction_max"])
            detections.append({"query_index": query, "mask_iou": iou,
                               "score_absolute_error": score_error,
                               "box_dimension_fraction": box_fraction, "passed": passed})
            if not passed:
                failures.append(f"query {query}: decoded-weight Meta mask/score/box gate exceeded")
        elif score <= comparison["low_confidence_reference_score_max"] and query in decoded_detections:
            failures.append(f"query {query}: low-confidence Meta query became a decoded-weight detection")
    adjacent = [{"query_index": query, "original_score": float(original_scores[query]),
                 "dequantized_score": float(decoded_scores[query]),
                 "original_selected": query in original_detections,
                 "dequantized_selected": query in decoded_detections}
                for query in range(200) if 0.4 < original_scores[query] < 0.6]
    return {"passed": not failures, "normalized_l2_max": profile["normalized_l2_max"],
            "gate_set": gates["gate_set"], "gates_sha256": selection["gates_sha256"],
            "tensors": tensor_metrics, "detections": detections,
            "threshold_adjacent": adjacent, "failures": failures}


def native_arithmetic_tensor_passed(metrics, profile):
    if metrics["normalized_l2"] is None:
        return metrics["maximum_absolute_error"] <= profile["zero_norm_max_abs"]
    return metrics["normalized_l2"] <= profile["native_arithmetic_normalized_l2_max"]


def check_provenance(model_path, reference_path, case_path, allow_supplementary=False,
                     allow_custom_quantization=False):
    reference = read_json(reference_path / "manifest.json")
    reference_kind = reference.get("reference_kind")
    if reference_kind == "supplementary-converted-weights":
        if not allow_supplementary or reference.get("eligible_for_milestone") is not False:
            raise ValueError("supplementary references cannot satisfy original-checkpoint acceptance; pass --allow-supplementary for a separate diagnostic")
    elif reference_kind != "official-checkpoint" or reference.get("eligible_for_milestone") is not True:
        raise ValueError("reference does not declare a supported, explicit provenance kind")
    frozen = load_case_manifest(case_path)
    validate_case_manifest(reference)
    if reference_kind == "supplementary-converted-weights":
        supplementary = reference["supplementary_weights"]
        converted_manifest = model_path.with_suffix(model_path.suffix + ".manifest.json")
        model = read_json(converted_manifest)
        if model["checkpoint"]["sha256"] != supplementary["restored_checkpoint"]["sha256"]:
            raise ValueError("supplementary converter output uses a different restored checkpoint")
        model["validation_source"] = "gguf-converter-output-from-supplementary-checkpoint"
        model["supplementary_weights"] = supplementary
    else:
        model = read_json(model_path.with_suffix(model_path.suffix + ".manifest.json"))
        standard_cases = Path(__file__).resolve().parents[2] / "tests/data/sam3-image-cases.json"
        if sha256_file(case_path) != sha256_file(standard_cases):
            raise ValueError("original-checkpoint acceptance requires the complete frozen M1 corpus")
    for manifest in (model, reference, frozen):
        if manifest.get("schema_version") != 1 or manifest.get("sam3_revision") != SAM3_REVISION:
            raise ValueError("unsupported model/reference/case provenance")
    if frozen.get("acceptance") != GATES:
        raise ValueError("case acceptance gates differ from the frozen M1 plan")
    if (model.get("architecture") != "sam3" or model.get("container_format") != "gguf"
            or model.get("container_version") != 3 or model.get("sam_schema_version") not in (1, 2, 3, 4)):
        raise ValueError("converted model requires a supported SAM schema GGUF v3; reconvert the original checkpoint")
    precision = model.get("precision")
    quantized = precision in QUANTIZED_PRECISIONS
    if quantized:
        storage_profile = model.get("storage_profile")
        schema4 = model.get("sam_schema_version") == 4
        modules = canonical_module_selection(model.get("quantization_modules")) if schema4 else None
        if schema4 and storage_profile == f"{CUSTOM_MODULE_PROFILE_PREFIX}{precision}-v1" and not allow_custom_quantization:
            raise ValueError("custom module quantization is diagnostic only; pass --allow-custom-quantization explicitly")
        selection = quantization_selection(precision, storage_profile, modules)
        profile, gates = selection["profile"], selection["gates"]
        if (model.get("sam_schema_version") not in (3, 4) or model.get("task") != "image"
                or model.get("storage_profile") != profile["storage_profile"]
                or model.get("arithmetic_profile") != gates["native_arithmetic"]["required_profile"]):
            raise ValueError("quantized model schema, task, storage profile, or arithmetic profile is unsupported")
        if schema4 != (selection["family"] in ("full", "custom")):
            raise ValueError("quantized model schema disagrees with its versioned profile family")
        validate_quantization_sidecar(model, allow_custom_quantization)
        if reference_kind != "official-checkpoint" or reference.get("eligible_for_milestone") is not True:
            raise ValueError("quantized release acceptance requires the original official Meta checkpoint oracle")
    elif precision not in ("f32", "f16", "hybrid"):
        raise ValueError("converted model has unsupported precision; refusing implicit precision mapping")
    elif model.get("sam_schema_version") == 3:
        raise ValueError("SAM schema 3 requires a supported quantized image precision")
    elif model.get("sam_schema_version") == 4:
        raise ValueError("SAM schema 4 requires a supported quantized image precision")
    if model["output"].get("bytes") != model_path.stat().st_size:
        raise ValueError("converted model size does not match its manifest")
    if model["output"]["sha256"] != sha256_file(model_path):
        raise ValueError("converted model SHA-256 does not match its manifest")
    task = "video" if model["sam_schema_version"] == 2 else "image"
    if model.get("task", "image") != task:
        raise ValueError("converted model task disagrees with its SAM schema")
    reader = read_gguf(model_path)
    if quantized and model.get("sam_schema_version") == 4:
        validate_gguf_module_metadata(reader, modules)
        validate_metadata(reader, precision, model["checkpoint"]["sha256"], task,
                          model.get("storage_profile"), modules)
    else:
        validate_metadata(reader, precision, model["checkpoint"]["sha256"], task,
                          model.get("storage_profile") if quantized else None)
    schema = read_json(Path(__file__).resolve().parents[2] / "tools/convert/sam3_tensor_schema.json")
    if schema.get("schema_version") != 1 or schema.get("sam3_revision") != SAM3_REVISION:
        raise ValueError("unsupported detector tensor schema")
    expected_tensors = tensor_schema(schema, task)
    recorded = {item["name"]: item for item in model["tensors"]}
    if len(recorded) != len(model["tensors"]) or set(recorded) != set(expected_tensors):
        raise ValueError("GGUF sidecar tensor inventory is missing, duplicated or unknown")
    if quantized:
        source_names = [item.get("source_name") for item in model["tensors"]]
        if (any(not isinstance(name, str) or not name for name in source_names)
                or len(set(source_names)) != len(source_names)
                or any(not isinstance(item.get("source_dtype"), str) or not item["source_dtype"]
                       or item.get("output_dtype") != item.get("dtype")
                       or item.get("conversion") != ("quantized-from-original-f32"
                                                       if item.get("dtype") in QUANTIZED_PRECISIONS else "preserved-f32")
                       for item in model["tensors"])):
            raise ValueError("quantized sidecar lacks unique original Meta tensor names/dtypes")
        if model.get("sam_schema_version") == 4:
            validate_schema4_tensor_module_records(
                model["tensors"], precision, model["storage_profile"], modules)
    if quantized and model.get("sam_schema_version") == 4:
        actual = inspect_tensors(reader, precision, expected_tensors,
                                 {name: item["shape"] for name, item in recorded.items()},
                                 model.get("storage_profile"), modules)
    else:
        actual = inspect_tensors(reader, precision, expected_tensors,
                                 {name: item["shape"] for name, item in recorded.items()},
                                 model.get("storage_profile") if quantized else None)
    for item in actual:
        compared_keys = ("shape", "ggml_shape", "dtype", "offset", "bytes", "sha256",
                         "quantization_block_elements", "quantization_block_bytes")
        if any(item[key] != recorded[item["name"]].get(key)
               for key in compared_keys if key in item):
            raise ValueError(f"{item['name']}: GGUF payload disagrees with its sidecar")
    del reader
    if model["checkpoint"]["sha256"] != reference["checkpoint"]["sha256"]:
        raise ValueError("converted model and oracle use different original checkpoints")
    if model["bpe"]["sha256"] != BPE_SHA256 or reference["bpe"]["sha256"] != BPE_SHA256:
        raise ValueError("model/reference tokenizer is not the pinned official BPE")
    oracle = reference.get("oracle", {})
    validate_cuda_oracle_provenance(oracle)
    expected_variant = "official-unfused-fp32" if reference_kind == "official-checkpoint" else "supplementary-converted-weights-unfused-fp32"
    if (oracle.get("variant") != expected_variant or oracle.get("precision") != "float32"
            or oracle.get("compile") is not False or oracle.get("autocast") is not False or oracle.get("tf32") is not False
            or not oracle.get("packages") or not oracle.get("fp32_adaptation")):
        raise ValueError("reference does not identify the recorded unfused FP32 official oracle")
    cases_file = artifact_path(reference_path, reference["cases_manifest"]["file"])
    digest = sha256_file(cases_file)
    if digest != reference["cases_manifest"]["sha256"] or digest != sha256_file(case_path):
        raise ValueError("reference cases differ from the frozen input/tolerance manifest")
    cases = reference.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("reference contains no cases")
    frozen_cases = {case["id"]: case for case in frozen["cases"]}
    if len(frozen_cases) != len(frozen["cases"]):
        raise ValueError("frozen case IDs are duplicated")
    if len(cases) != len(frozen_cases) or {case["id"] for case in cases} != set(frozen_cases):
        raise ValueError("reference is missing required cases or contains duplicate/unknown cases")
    for case in cases:
        if any(case.get(key) != value for key, value in frozen_cases[case["id"]].items()):
            raise ValueError(f"{case['id']}: reference input/prompt differs from the frozen case")
        directory = artifact_path(reference_path, case["directory"])
        for key, filename in (("input_sha256", case["input"]), ("tensors_sha256", "tensors.json"), ("results_sha256", "results.json")):
            if sha256_file(artifact_path(directory, filename)) != case[key]:
                raise ValueError(f"{case['id']}: reference {filename} hash mismatch")
        read_tensor_index(directory)
    return precision, reference, model


def check_dequantized_reference(path, model_path, model, original_reference_path, case_path, precision):
    """Validate a non-release Meta oracle rebuilt from the exact quantized GGUF payload."""
    if precision not in QUANTIZED_PRECISIONS:
        raise ValueError("dequantized-weight diagnostics are available only for quantized image profiles")
    reference_path = Path(path)
    reference = read_json(reference_path / "manifest.json")
    supplementary = reference.get("supplementary_weights", {})
    derivation = supplementary.get("reconstruction", {})
    quantized = supplementary.get("quantized_gguf", {})
    weight_compression = supplementary.get("weight_compression", {})
    storage_profile = model.get("storage_profile")
    schema4 = model.get("sam_schema_version") == 4
    modules = canonical_module_selection(model.get("quantization_modules")) if schema4 else None
    selection = quantization_selection(precision, storage_profile, modules)
    profile, gates = selection["profile"], selection["gates"]
    expected_quantized_names = {item["name"] for item in model["tensors"]
                                if item.get("dtype") in {"q8_0", "q6_k", "q5_k", "q4_k"}}
    sidecar_path = Path(model_path).with_suffix(Path(model_path).suffix + ".manifest.json")
    if (reference.get("schema_version") != 1 or reference.get("sam3_revision") != SAM3_REVISION
            or reference.get("reference_kind") != "supplementary-converted-weights"
            or reference.get("eligible_for_milestone") is not False
            or supplementary.get("schema_version") != 1
            or supplementary.get("restored_checkpoint", {}).get("sha256") != reference.get("checkpoint", {}).get("sha256")
            or supplementary.get("container", {}).get("precision") != "f32"
            or not supplementary.get("container", {}).get("hf_revision")
            or derivation.get("kind") != "dequantized-gguf-weights-v1"
            or derivation.get("source_meta_checkpoint_sha256") != model["checkpoint"]["sha256"]
            or derivation.get("decoded_tensor_count") != len(expected_quantized_names)
            or derivation.get("quantizer") != model.get("quantizer")
            or derivation.get("storage_profile") != storage_profile
            or (schema4 and derivation.get("quantization_modules") != modules)
            or derivation.get("gate_set") != gates["gate_set"]
            or derivation.get("gates_sha256") != selection["gates_sha256"]
            or quantized.get("sha256") != model["output"]["sha256"]
            or quantized.get("manifest_sha256") != sha256_file(sidecar_path)
            or quantized.get("precision") != precision
            or quantized.get("storage_profile") != storage_profile
            or (schema4 and quantized.get("quantization_modules") != modules)
            or weight_compression.get("gate_set") != gates["gate_set"]
            or weight_compression.get("gates_sha256") != selection["gates_sha256"]
            or (schema4 and weight_compression.get("quantization_modules") != modules)
            or weight_compression.get("normalized_l2_max") != profile["normalized_l2_max"]
            or weight_compression.get("zero_norm_max_abs") != profile["zero_norm_max_abs"]):
        raise ValueError("supplementary reference is not provenance-bound to this quantized GGUF")
    measured_weights = weight_compression.get("tensors")
    if not isinstance(measured_weights, dict) or set(measured_weights) != expected_quantized_names:
        raise ValueError("dequantized Meta diagnostic lacks per-tensor weight compression measurements")
    for name, metrics in measured_weights.items():
        if (not isinstance(metrics, dict) or type(metrics.get("passed")) is not bool
                or metrics.get("dtype") != next(item["dtype"] for item in model["tensors"] if item["name"] == name)
                or not isinstance(metrics.get("source_name"), str) or not metrics["source_name"]
                or not math.isfinite(metrics.get("reference_norm", float("nan")))
                or not math.isfinite(metrics.get("maximum_absolute_error", float("nan")))
                or metrics["reference_norm"] < 0 or metrics["maximum_absolute_error"] < 0
                or (metrics.get("normalized_l2") is not None
                    and (not math.isfinite(metrics["normalized_l2"]) or metrics["normalized_l2"] < 0))):
            raise ValueError(f"{name}: malformed weight compression diagnostic")
    if weight_compression.get("passed") is not all(item["passed"] for item in measured_weights.values()):
        raise ValueError("dequantized Meta weight compression summary disagrees with its tensor measurements")
    oracle = reference.get("oracle", {})
    validate_cuda_oracle_provenance(oracle)
    if (oracle.get("variant") != "supplementary-converted-weights-unfused-fp32"
            or oracle.get("precision") != "float32" or oracle.get("compile") is not False
            or oracle.get("autocast") is not False or oracle.get("tf32") is not False
            or not oracle.get("packages") or not oracle.get("fp32_adaptation")):
        raise ValueError("dequantized Meta diagnostic is not a recorded unfused FP32 export")
    validate_case_manifest(reference)
    if reference.get("bpe", {}).get("sha256") != model.get("bpe", {}).get("sha256"):
        raise ValueError("dequantized Meta diagnostic uses a different tokenizer asset")
    original = read_json(Path(original_reference_path) / "manifest.json")
    original_cases = {case["id"]: case for case in original.get("cases", [])}
    cases = {case["id"]: case for case in reference.get("cases", [])}
    if len(cases) != len(reference.get("cases", [])) or set(cases) != set(original_cases):
        raise ValueError("dequantized Meta diagnostic cases differ from the original Meta oracle")
    frozen_cases = {case["id"]: case for case in load_case_manifest(case_path)["cases"]}
    if set(frozen_cases) != set(cases):
        raise ValueError("dequantized Meta diagnostic cases differ from the frozen inputs")
    if reference.get("cases_manifest", {}).get("sha256") != sha256_file(case_path):
        raise ValueError("dequantized Meta diagnostic cases do not match the frozen case manifest")
    for case_id, case in cases.items():
        validate_diagnostic_case_descriptor(case, frozen_cases[case_id], original_cases[case_id])
        directory = artifact_path(reference_path, case["directory"])
        for key, filename in (("input_sha256", case["input"]),
                              ("tensors_sha256", "tensors.json"),
                              ("results_sha256", "results.json")):
            if sha256_file(artifact_path(directory, filename)) != case.get(key):
                raise ValueError(f"{case_id}: dequantized Meta diagnostic {filename} hash mismatch")
        read_tensor_index(directory)
    return reference, cases


def validate_diagnostic_case_descriptor(diagnostic_case, frozen_case, original_case):
    """Bind supplementary Meta outputs to the exact frozen transform and RGB input bytes."""
    if any(diagnostic_case.get(key) != value for key, value in frozen_case.items()):
        raise ValueError(f"{frozen_case['id']}: dequantized Meta diagnostic changed a frozen case descriptor")
    if (diagnostic_case.get("input_sha256") != original_case.get("input_sha256")
            or diagnostic_case.get("source_sha256") != original_case.get("source_sha256")):
        raise ValueError(f"{frozen_case['id']}: dequantized Meta diagnostic input differs from the original oracle")


def validate(args):
    from tools.convert.sam3_artifacts import freeze_run_artifacts, verify_run_artifacts, freeze_output_files, verify_output_files

    cuda_compute = getattr(args, "cuda_compute", None) or "f32"
    validate_cuda_compute_mode(args.backend, cuda_compute)
    precision, reference, model = check_provenance(
        args.model, args.reference, args.cases, args.allow_supplementary,
        getattr(args, "allow_custom_quantization", False))
    quantized = precision in QUANTIZED_PRECISIONS
    schema4 = model.get("sam_schema_version") == 4
    quantization_modules = model.get("quantization_modules") if schema4 else None
    selection = quantization_selection(precision, model.get("storage_profile"), quantization_modules) if quantized else None
    profile = selection["profile"] if quantized else None
    quantization_gates = selection["gates"] if quantized else None
    output_selection = output_quality_selection(
        precision, model.get("storage_profile"), quantization_modules) if quantized else None
    output_gates = output_selection["gates"] if quantized else None
    diagnostic_cases = None
    diagnostic_reference = None
    diagnostic_path = getattr(args, "dequantized_reference", None)
    if diagnostic_path:
        if not quantized:
            raise ValueError("--dequantized-reference applies only to quantized image profiles")
        diagnostic_reference, diagnostic_cases = check_dequantized_reference(
            diagnostic_path, args.model, model, args.reference, args.cases, precision)
    existing_root = Path(args.existing_output).resolve() if getattr(args, "existing_output", None) else None
    if existing_root is not None and cuda_compute != "f32":
        raise ValueError("CUDA F16 compute acceptance requires a fresh run, not existing-output regrading")
    previous_report, previous_cases, previous_run_integrity = None, {}, None
    if existing_root is not None:
        if not quantized:
            raise ValueError("--existing-output regrading applies only to quantized image profiles")
        previous_report = read_json(existing_root / "metrics.json")
        case_list = previous_report.get("cases")
        if not isinstance(case_list, list) or any(not isinstance(item, dict) for item in case_list):
            raise ValueError("existing validation receipt has a malformed case list")
        previous_cases = {item.get("id"): item for item in case_list}
        expected_artifacts = previous_report.get("run_artifact_sha256")
        sidecar = args.model.with_suffix(args.model.suffix + ".manifest.json")
        validate_existing_validation_profile(previous_report, args.backend, model,
                                             sha256_file(args.reference / "manifest.json"), selection)
        previous_run_integrity = validate_existing_run_receipt(existing_root, previous_report, model, selection)
        if (len(previous_cases) != len(case_list or [])
                or set(previous_cases) != {case["id"] for case in reference["cases"]}
                or not isinstance(expected_artifacts, dict)
                or expected_artifacts.get(str(args.model.absolute())) != model["output"]["sha256"]
                or expected_artifacts.get(str(sidecar.absolute())) != sha256_file(sidecar)):
            raise ValueError("existing validation receipt does not match this model, oracle, backend, and frozen gates")
        for case in reference["cases"]:
            expected_files = previous_cases[case["id"]].get("output_sha256")
            actual_dir = existing_root / case["id"]
            if not isinstance(expected_files, dict) or freeze_output_files(actual_dir) != expected_files:
                raise ValueError(f"{case['id']}: retained C++ output differs from its validation receipt")
        executable = None
    else:
        executable = args.build_dir / "tests/test_image"
        if not executable.is_file():
            raise FileNotFoundError(f"C++ differential executable is missing: {executable}")
    if args.output:
        output = args.output.resolve()
        if existing_root is not None and (output == existing_root or output.is_relative_to(existing_root)
                                          or existing_root.is_relative_to(output)):
            raise ValueError("diagnostic receipt output must be separate from retained C++ case outputs")
        args.output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix=f"validation-{args.backend}-{precision}-", dir=args.build_dir.resolve()))
    if existing_root is not None and (output == existing_root or output.is_relative_to(existing_root)
                                      or existing_root.is_relative_to(output)):
        raise ValueError("diagnostic receipt output must be separate from retained C++ case outputs")
    report = {"schema_version": 1, "passed": False, "backend": args.backend, "precision": precision,
              "reference_kind": reference["reference_kind"], "eligible_for_milestone": reference["eligible_for_milestone"],
              "validation_source": model.get("validation_source", "official-checkpoint-converter-output"),
              "model_sha256": model["output"]["sha256"], "checkpoint_sha256": model["checkpoint"]["sha256"],
              "reference_manifest_sha256": sha256_file(args.reference / "manifest.json"),
              "gates": GATES, "cases": []}
    if args.backend == "cuda":
        report["cuda_device"] = getattr(args, "cuda_device", None) or 0
        report["cuda_compute"] = cuda_compute
    if cuda_compute == "f16":
        report.update(acceptance_mode="cuda-f16-output-quality-v1", tensor_fidelity_is_release_gate=False,
                      output_gate_precision="quantized-profile" if quantized else "f16")
    if quantized:
        from collections import Counter

        allowed_profiles = allowed_runtime_arithmetic_profiles(args.backend, existing_root is not None, cuda_compute)
        report.update({"gate_set": output_gates["gate_set"],
                       "gates_sha256": output_selection["gates_sha256"],
                       "gates": output_gates,
                       "output_quality_gate_set": output_gates["gate_set"],
                       "output_quality_gates_sha256": output_selection["gates_sha256"],
                       "tensor_fidelity_gate_set": quantization_gates["gate_set"],
                       "tensor_fidelity_gates_sha256": selection["gates_sha256"],
                       "tensor_fidelity_gates": quantization_gates,
                       "profile_family": selection["family"],
                       "profile_status": model["profile_status"],
                       "storage_profile": profile["storage_profile"],
                       "arithmetic_profile": model["arithmetic_profile"],
                       "runtime_arithmetic_profiles_allowed": sorted(allowed_profiles),
                       "output_quality_passed": False,
                       "tensor_fidelity_passed": False,
                       "runtime_arithmetic_profile_valid": False,
                       "regrading_existing_output": existing_root is not None,
                       "quantization": model["quantization"],
                       "quantizer": model["quantizer"],
                       "converter_sha256": model["converter_sha256"],
                       "gguf_helper_sha256": model["gguf_helper_sha256"],
                       "requirements_lock_sha256": model["requirements_lock_sha256"],
                       "gguf_package_version": model["gguf_package_version"],
                       "tensor_dtype_distribution": dict(sorted(Counter(item["dtype"] for item in model["tensors"]).items()))})
        if schema4:
            report["quantization_modules"] = quantization_modules
            report["custom_quantization_diagnostic"] = selection["family"] == "custom"
            if selection["family"] == "custom":
                report["eligible_for_milestone"] = False
                report["release_support_eligible"] = False
                report["reference_eligible_for_milestone"] = reference["eligible_for_milestone"]
    if diagnostic_reference is not None:
        report["dequantized_meta_reference_manifest_sha256"] = sha256_file(diagnostic_path / "manifest.json")
        report["native_arithmetic_diagnostic_is_release_gate"] = False
        report["weight_compression_diagnostic"] = diagnostic_reference["supplementary_weights"]["weight_compression"]
        report["weight_compression_diagnostic_is_release_gate"] = False
    if "supplementary_weights" in model:
        report["supplementary_weights"] = model["supplementary_weights"]
    if existing_root is not None:
        artifacts = previous_report["run_artifact_sha256"]
        report["reused_validation_receipt_sha256"] = sha256_file(existing_root / "metrics.json")
        report["reused_run_receipt_sha256"] = previous_run_integrity["run_receipt_sha256"]
        report["reused_source_hash_delta_paths"] = previous_run_integrity["source_hash_delta_paths"]
        report["reused_binary_and_library_sha256"] = previous_run_integrity["binary_and_library_sha256"]
        report["reused_case_output_directory"] = str(existing_root)
    else:
        artifacts = freeze_run_artifacts(args.build_dir, executable, args.model, model["output"]["sha256"])
    report["run_artifact_sha256"] = artifacts
    for case in reference["cases"]:
        directory = artifact_path(args.reference, case["directory"])
        actual_directory = existing_root / case["id"] if existing_root is not None else output / case["id"]
        try:
            if existing_root is None:
                command = [str(executable.resolve()), "--model", str(args.model.resolve()),
                           "--image", str(artifact_path(directory, case["input"])), "--text", case["prompt"],
                           "--backend", args.backend, "--threads", str(args.threads),
                           "--score-threshold", "0.5", "--output", str(actual_directory)]
                if args.backend == "cuda":
                    command.extend(["--cuda-device", str(getattr(args, "cuda_device", None) or 0)])
                    command.extend(["--cuda-compute", cuda_compute])
                verify_run_artifacts(artifacts)
                print(f"Validating {case['id']} ({args.backend}, {precision})", flush=True)
                with (output / f"{case['id']}.log").open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=args.timeout_seconds)
                if completed.returncode != 0:
                    raise RuntimeError(f"test_image exited {completed.returncode}; see {case['id']}.log")
                verify_run_artifacts(artifacts)
            output_files = freeze_output_files(actual_directory)
            if existing_root is not None and output_files != previous_cases[case["id"]]["output_sha256"]:
                raise ValueError("retained C++ output changed during diagnostic comparison")
            metrics = compare_case(directory, actual_directory, case, precision, args.backend, model,
                                   allow_historical_cpu_native=existing_root is not None,
                                   cuda_device=getattr(args, "cuda_device", None) or 0, cuda_compute=cuda_compute)
            if diagnostic_cases is not None:
                diagnostic_case = diagnostic_cases[case["id"]]
                diagnostic_directory = artifact_path(diagnostic_path, diagnostic_case["directory"])
                try:
                    metrics["weight_compression_output_diagnostic"] = compare_dequantized_meta_output(
                        directory, diagnostic_directory, case, precision, model["storage_profile"],
                        quantization_modules)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    metrics["weight_compression_output_diagnostic"] = {"passed": False, "failures": [str(error)]}
                try:
                    metrics["native_arithmetic_diagnostic"] = compare_native_arithmetic(
                        diagnostic_directory, actual_directory, case, precision, model["storage_profile"],
                        quantization_modules)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    metrics["native_arithmetic_diagnostic"] = {"passed": False, "failures": [str(error)]}
            verify_output_files(actual_directory, output_files)
            metrics["output_sha256"] = output_files
            if existing_root is None:
                verify_run_artifacts(artifacts)
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
            metrics = {"id": case["id"], "passed": False, "failures": [str(error)]}
            if quantized or cuda_compute == "f16":
                metrics.update(output_quality_passed=False, tensor_fidelity_passed=False,
                               output_quality_failures=[str(error)],
                               tensor_fidelity_failures=["tensor fidelity could not be fully evaluated after validation error"])
        report["cases"].append(metrics)
        write_json(output / "metrics.json", report)
    complete_cases = len(report["cases"]) == len(reference["cases"])
    if quantized:
        report["output_quality_passed"] = (complete_cases and all(
            case.get("output_quality_passed") is True for case in report["cases"]))
        report["tensor_fidelity_passed"] = (complete_cases and all(
            case.get("tensor_fidelity_passed") is True for case in report["cases"]))
        report["tensor_fidelity_failures"] = [
            {"case_id": case["id"], "failures": case.get("tensor_fidelity_failures", [])}
            for case in report["cases"] if case.get("tensor_fidelity_passed") is not True]
        observed_runtime_profiles = sorted({case["runtime_arithmetic_profile"] for case in report["cases"]
                                             if case.get("runtime_arithmetic_profile")})
        allow_historical_cpu_native = existing_root is not None and args.backend == "cpu"
        report["runtime_arithmetic_profiles_observed"] = observed_runtime_profiles
        report["runtime_arithmetic_profile_valid"] = runtime_arithmetic_profiles_valid(
            args.backend, observed_runtime_profiles, allow_historical_cpu_native, cuda_compute)
        report["runtime_arithmetic_profile"] = observed_runtime_profiles[0] if len(observed_runtime_profiles) == 1 else None
        report["passed"] = report["output_quality_passed"] and report["runtime_arithmetic_profile_valid"]
        eligibility = quantization_release_eligibility(selection, model, reference, report["passed"])
        report["release_support_eligible"] = eligibility["release_support_eligible"]
        if schema4:
            report["eligible_for_milestone"] = eligibility["eligible_for_milestone"]
            if selection["family"] == "custom":
                report["reference_eligible_for_milestone"] = reference["eligible_for_milestone"]
    else:
        report["passed"] = complete_cases and all(case["passed"] for case in report["cases"])
        if cuda_compute == "f16":
            report.update(output_quality_passed=report["passed"],
                          tensor_fidelity_passed=complete_cases and all(
                              case.get("tensor_fidelity_passed") is True for case in report["cases"]),
                          runtime_arithmetic_profile=CUDA_F16_ARITHMETIC_PROFILE)
    if diagnostic_cases is not None:
        report["weight_compression_output_diagnostic_passed"] = (
            len(report["cases"]) == len(diagnostic_cases)
            and all(case.get("weight_compression_output_diagnostic", {}).get("passed") is True
                    for case in report["cases"]))
        report["native_arithmetic_diagnostic_passed"] = (
            len(report["cases"]) == len(diagnostic_cases)
            and all(case.get("native_arithmetic_diagnostic", {}).get("passed") is True for case in report["cases"]))
    write_json(output / "metrics.json", report)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {output / 'metrics.json'}")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("cpu", "metal", "cuda"))
    parser.add_argument("--cuda-device", type=int, help="Index among CUDA-visible devices (default: 0)")
    parser.add_argument("--cuda-compute", choices=("f32", "f16"), help="CUDA arithmetic; f16 uses final-output quality gates")
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[2] / "tests/data/sam3-image-cases.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dequantized-reference", type=Path,
                        help="Compare native GGML arithmetic against a supplementary Meta export of the actual decoded GGUF weights")
    parser.add_argument("--existing-output", type=Path,
                        help="Reuse case outputs and hashes from a prior validation receipt without rerunning C++ inference")
    parser.add_argument("--allow-supplementary", action="store_true",
                        help="Run a separately labeled converted-weight diagnostic, never original-checkpoint acceptance")
    parser.add_argument("--allow-custom-quantization", action="store_true",
                        help="Allow a schema-4 custom module profile for diagnostics only; it is never release eligible")
    args = parser.parse_args()
    if args.cuda_device is not None and (args.backend != "cuda" or args.cuda_device < 0):
        parser.error("--cuda-device requires backend cuda and a nonnegative index")
    if args.cuda_compute is not None and args.backend != "cuda":
        parser.error("--cuda-compute requires backend cuda")
    if args.threads <= 0 or args.timeout_seconds <= 0 or not math.isfinite(args.timeout_seconds):
        parser.error("thread count and timeout must be positive")
    try:
        parser.exit(validate(args))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f"validation failed: {error}\n")


if __name__ == "__main__":
    main()
