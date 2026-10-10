#!/usr/bin/env python3
"""Resolve weight, activation, compute and image-feature cache policies without inference."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.sam3_artifacts import read_json, sha256_file, write_json
from tools.quantize.weight_policy import (CUSTOM_MODULE_PROFILE_PREFIX, FULL_MODULE_PROFILE_PREFIX,
                                         MIXED_STORAGE_PROFILE, TENSOR_MIXED_STORAGE_PROFILE, QUANTIZATION_PROFILES,
                                         canonical_quantization_modules, mixed_quantization_profile, quantization_profile)


CONFIG_KIND = "sam-quantization-config"
WEIGHTS = ("f32", "f16", *QUANTIZATION_PROFILES, "mixed")
CACHES = {"f32": ["f32", "f32", "f32"], "f16": ["f16", "f16", "f16"],
          "mixed-q8_0": ["q8_0", "q8_0", "f32"]}
CONTEXTS = ("conversion", "inspect", "benchmark", "public-api", "original-reference")


def configuration_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def object_keys(value, required, optional, label):
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - optional:
        raise ValueError(f"{label} requires {sorted(required)}; optional keys: {sorted(optional)}; unknown keys are rejected")


def choice(value, choices, label):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"unsupported {label}: {value!r}; choose from {', '.join(choices)}")
    return value


def validate_configuration(backend, compute, cache, activation="backend-selected"):
    choice(backend, ("cpu", "metal", "cuda"), "backend")
    choice(compute, ("f32", "f16"), "compute policy")
    choice(cache, CACHES, "image-feature-cache policy")
    if activation != "backend-selected":
        raise ValueError("native activation precision is backend-selected; independent INT8/FP8/F16 activation settings are not implemented")
    if backend != "cuda" and (compute != "f32" or cache != "f32"):
        raise ValueError("native reduced compute/cache settings currently require explicit CUDA")


def resolve_weights(value, schema_version=1):
    if schema_version in (2, 3):
        required = {"precision", "base_precision", "module_precisions"}
        if schema_version == 3:
            required.add("tensor_precisions")
        object_keys(value, required, {"storage_profile", "policy_sha256"}, f"schema-{schema_version} weights")
        if value["precision"] != "mixed":
            raise ValueError(f"configuration schema {schema_version} requires precision=mixed; use schema 1 for uniform presets")
        policy = {key: value[key] for key in ("base_precision", "module_precisions", "tensor_precisions", "policy_sha256") if key in value}
        profile = mixed_quantization_profile(policy, value.get("storage_profile"))
        return {"precision": "mixed", "storage_profile": profile["storage_profile"],
                **{key: profile[key] for key in ("base_precision", "module_precisions", "tensor_precisions", "policy_sha256") if key in profile}}
    if isinstance(value, dict) and set(value) & {"overrides", "module_precisions", "tensor_precisions"}:
        raise ValueError("schema 1 selects modules using one precision; use schema 2 for module formats or schema 3 for tensor overrides")
    object_keys(value, {"precision"}, {"storage_profile", "modules"}, "weights")
    precision = choice(value["precision"], WEIGHTS[:-1], "schema-1 weight precision")
    storage, modules = value.get("storage_profile"), value.get("modules")
    if storage is not None and not isinstance(storage, str):
        raise ValueError("weights.storage_profile must be a string")
    if modules is not None and not isinstance(modules, list):
        raise ValueError("weights.modules must be a list of module names, not a precision mapping")
    if precision in ("f32", "f16"):
        if storage not in (None, "dense") or modules not in (None, []):
            raise ValueError("dense weights do not support quantization modules or a quantized storage profile")
        return {"precision": precision, "storage_profile": "dense", "modules": []}
    if storage is None and modules is None:
        raise ValueError("quantized configurations require explicit weights.modules or weights.storage_profile")
    if storage is not None and not storage.startswith((CUSTOM_MODULE_PROFILE_PREFIX, FULL_MODULE_PROFILE_PREFIX)):
        profile = quantization_profile(precision, storage)
        selected = ["vision", "text"] if profile["quantize_text_linear"] else ["vision"]
        if modules not in (None, []) and canonical_quantization_modules(modules) != selected:
            raise ValueError("weights.modules disagrees with the selected storage profile")
    else:
        if modules == []:
            if storage is None or storage.startswith(CUSTOM_MODULE_PROFILE_PREFIX):
                raise ValueError("custom quantization requires a nonempty module list")
            modules = None
        profile = quantization_profile(precision, storage, modules)
        selected = profile["modules"] or (["vision", "text"] if profile["quantize_text_linear"] else ["vision"])
    return {"precision": precision, "storage_profile": profile["storage_profile"], "modules": selected}


def normalize_config(value):
    object_keys(value, {"schema_version", "kind", "task", "backend", "weights", "activation", "compute", "cache"}, set(), "configuration")
    if type(value["schema_version"]) is not int or value["schema_version"] not in (1, 2, 3) or value["kind"] != CONFIG_KIND:
        raise ValueError("configuration requires schema_version=1/2/3 and kind=sam-quantization-config")
    if value["task"] != "image":
        raise ValueError("the unified configuration currently supports SAM 3 image tools only; video policies are separate")
    for axis in ("activation", "compute", "cache"):
        object_keys(value[axis], {"mode"}, set(), axis)
    validate_configuration(value["backend"], value["compute"]["mode"], value["cache"]["mode"], value["activation"]["mode"])
    return {"schema_version": value["schema_version"], "kind": CONFIG_KIND, "task": "image", "backend": value["backend"],
            "weights": resolve_weights(value["weights"], value["schema_version"]), "activation": {"mode": value["activation"]["mode"]},
            "compute": {"mode": value["compute"]["mode"]}, "cache": {"mode": value["cache"]["mode"]}}


def validate_context(config, context):
    choice(context, CONTEXTS, "configuration context")
    if context == "benchmark" and config["backend"] == "metal":
        raise ValueError("current image export/performance probes support CPU/CUDA; Metal configuration can be inspected only")
    if context == "public-api" and config["cache"]["mode"] != "f32":
        raise ValueError("reduced image-feature cache is a private benchmark probe setting; not exposed by the public API")
    if context == "original-reference" and (config["backend"] != "cuda" or config["weights"]["precision"] != "f32"
            or config["compute"]["mode"] != "f32" or config["cache"]["mode"] != "f32"):
        raise ValueError("the original checkpoint exporter currently requires CUDA F32 weights/compute/cache")


def configuration_receipt(config, source=None):
    config = normalize_config(config)
    return {"config": config, "resolved_sha256": configuration_hash(config), "source": source or {"kind": "cli-and-model"}}


def validate_receipt(receipt):
    object_keys(receipt, {"config", "resolved_sha256", "source"}, set(), "configuration receipt")
    config = normalize_config(receipt["config"])
    if config != receipt["config"] or configuration_hash(config) != receipt["resolved_sha256"]:
        raise ValueError("resolved configuration identity changed")
    source = receipt["source"]
    if not isinstance(source, dict) or source.get("kind") not in ("json", "cli-and-model"):
        raise ValueError("invalid configuration source")
    if source["kind"] == "json" and (not isinstance(source.get("file"), str)
            or not isinstance(source.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"])):
        raise ValueError("invalid configuration file identity")
    return config


def load_config(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024:
        raise ValueError("quantization configuration exceeds 64 KiB")
    before = sha256_file(path)
    config = normalize_config(read_json(path))
    if sha256_file(path) != before:
        raise ValueError("configuration changed while reading")
    return configuration_receipt(config, {"kind": "json", "file": path.name, "sha256": before})


def model_weights(manifest):
    storage = manifest.get("storage_profile") or "dense"
    modules = manifest.get("quantization_modules", [])
    if not isinstance(storage, str) or not isinstance(modules, list):
        raise ValueError("invalid model storage profile or quantization module metadata")
    if manifest["precision"] == "mixed":
        policy = {key: manifest[key] for key in ("base_precision", "module_precisions", "tensor_precisions", "policy_sha256") if key in manifest}
        weights = resolve_weights({"precision": "mixed", "storage_profile": storage, **policy}, 3 if "tensor_precisions" in policy else 2)
        selected = mixed_quantization_profile(policy, storage)["modules"]
        if modules != selected:
            raise ValueError("quantization_modules disagrees with the mixed format allocation")
        return weights
    if storage.startswith((CUSTOM_MODULE_PROFILE_PREFIX, FULL_MODULE_PROFILE_PREFIX)):
        if not isinstance(modules, list) or not modules or canonical_quantization_modules(modules) != modules:
            raise ValueError("schema-4 model manifest requires canonical nonempty quantization_modules")
    elif modules:
        raise ValueError("model quantization_modules metadata is defined only for schema-4 modular profiles")
    return resolve_weights({"precision": manifest["precision"], "storage_profile": storage, "modules": modules})


def resolve_model_configuration(args, manifest):
    actual = model_weights(manifest)
    receipt = getattr(args, "quantization_configuration", None)
    if receipt is not None:
        config = validate_receipt(receipt)
        if config["weights"] != actual:
            raise ValueError("quantization configuration weights differ from the model conversion manifest")
        if (config["backend"], config["compute"]["mode"], config["cache"]["mode"], config["activation"]["mode"]) != (
                args.backend, args.compute, args.cache, getattr(args, "activation", "backend-selected")):
            raise ValueError("runtime settings differ from the resolved quantization configuration")
        return receipt
    return configuration_receipt({"schema_version": 3 if "tensor_precisions" in actual else 2 if actual["precision"] == "mixed" else 1,
                                  "kind": CONFIG_KIND, "task": "image", "backend": args.backend,
                                  "weights": actual, "activation": {"mode": getattr(args, "activation", "backend-selected")},
                                  "compute": {"mode": args.compute}, "cache": {"mode": args.cache}})


def validate_recipe_configuration(recipe):
    if "quantization_configuration" not in recipe:
        return  # Completed older reports remain readable without rewriting receipts.
    if recipe.get("engine") == "original":
        validate_context(validate_receipt(recipe["quantization_configuration"]), "original-reference")
    resolve_model_configuration(SimpleNamespace(backend=recipe["backend"], compute=recipe["compute_mode"],
                                               cache=recipe["feature_cache"], activation=recipe.get("activation", "backend-selected"),
                                               quantization_configuration=recipe["quantization_configuration"]),
                                {"precision": recipe["weight_precision"], "storage_profile": recipe["storage_profile"],
                                 "quantization_modules": recipe["quantization_modules"],
                                 **{key: recipe[key] for key in ("base_precision", "module_precisions", "tensor_precisions", "policy_sha256") if key in recipe}})


def conversion_options(receipt):
    config = validate_receipt(receipt)
    validate_context(config, "conversion")
    weights = config["weights"]
    if weights["precision"] == "mixed":
        return {"precision": "mixed", "storage_profile": weights["storage_profile"], "quantize_modules": None}
    storage = weights["storage_profile"]
    # Preserve the converter's mutually exclusive profile/module arguments.
    modules = weights["modules"] if storage.startswith(CUSTOM_MODULE_PROFILE_PREFIX) else None
    return {"precision": weights["precision"], "storage_profile": None if storage == "dense" or modules else storage,
            "quantize_modules": modules}


def resolve_runtime_arguments(args):
    path = getattr(args, "quantization_config", None)
    names = ("backend", "compute", "cache", "activation")
    if path is not None:
        if any(getattr(args, name, None) is not None for name in names):
            raise ValueError("--quantization-config cannot be combined with --backend/--compute/--cache/--activation")
        receipt = load_config(path)
        config = receipt["config"]
        args.quantization_configuration = receipt
        args.configuration_inputs = {str(path.resolve()): receipt["source"]["sha256"]}
        args.backend = config["backend"]
        args.compute, args.cache, args.activation = (config[axis]["mode"] for axis in ("compute", "cache", "activation"))
        context = "original-reference" if getattr(args, "engine", None) == "original" else (
            "inspect" if args.command == "inspect-precision" else "benchmark")
        validate_context(config, context)
    else:
        for name, default in zip(names, ("cuda", "f32", "f32", "backend-selected")):
            if getattr(args, name, None) is None:
                setattr(args, name, default)
        validate_configuration(args.backend, args.compute, args.cache, args.activation)


def resolve_performance_arguments(args):
    paths = [getattr(args, variant + "_config", None) for variant in ("baseline", "candidate")]
    legacy = ("backend", "activation", "baseline_compute", "baseline_cache", "candidate_compute", "candidate_cache")
    if any(path is not None for path in paths):
        if any(path is None for path in paths):
            raise ValueError("performance configurations require both --baseline-config and --candidate-config")
        if any(getattr(args, name, None) is not None for name in legacy):
            raise ValueError("performance config files cannot be combined with backend/activation/compute/cache CLI settings")
        receipts = {variant: load_config(path) for variant, path in zip(("baseline", "candidate"), paths)}
        for receipt in receipts.values():
            validate_context(receipt["config"], "benchmark")
        if receipts["baseline"]["config"]["backend"] != receipts["candidate"]["config"]["backend"]:
            raise ValueError("paired performance configurations must use the same backend")
        args.quantization_configurations = receipts
        args.configuration_inputs = {str(path.resolve()): receipt["source"]["sha256"]
                                     for path, receipt in zip(paths, receipts.values())}
        args.backend = receipts["baseline"]["config"]["backend"]
        args.activation = "backend-selected"
        for variant, receipt in receipts.items():
            setattr(args, variant + "_compute", receipt["config"]["compute"]["mode"])
            setattr(args, variant + "_cache", receipt["config"]["cache"]["mode"])
    else:
        for name, default in zip(legacy, ("cuda", "backend-selected", "f32", "f32", "f32", "f32")):
            if getattr(args, name, None) is None:
                setattr(args, name, default)
    for variant in ("baseline", "candidate"):
        validate_configuration(args.backend, getattr(args, variant + "_compute"), getattr(args, variant + "_cache"), args.activation)


def capabilities():
    return {"schema_version": 1, "kind": "sam-quantization-capabilities", "scope": "SAM 3 image tools; source-defined support, not hardware qualification",
            "weights": {"precisions": list(WEIGHTS), "modules": ["vision", "text", "fusion", "decoder"],
                        "custom_selection": "one quantized format across selected linear modules; other tensors stay F32",
                        "exceptions": "protected F32 tensors; K-block vision MLP row fallback Q8_0",
                        "mixed_module_formats": ["f32", *QUANTIZATION_PROFILES],
                        "mixed_configuration_schema": 2, "mixed_storage_profile": MIXED_STORAGE_PROFILE,
                        "tensor_configuration_schema": 3, "tensor_storage_profile": TENSOR_MIXED_STORAGE_PROFILE,
                        "tensor_overrides": "exact eligible canonical linear names; tensor > module > base; no unused entries",
                        "mixed_module_or_tensor_f16": "NOT_IMPLEMENTED"},
            "activation": {"modes": ["backend-selected"], "independent_int8_fp8_f16": "NOT_IMPLEMENTED",
                           "studies": "PyTorch/offline studies do not enable native runtime modes"},
            "compute": {"cpu": ["f32"], "metal": ["f32"], "cuda": ["f32", "f16"],
                        "meaning": "backend policy/hints; no universal operand or accumulation dtype guarantee"},
            "cache": {"kind": "host image-feature cache, not LLM KV", "cpu": ["f32"], "metal": ["f32"],
                      "cuda": list(CACHES), "levels": CACHES, "reduced_mode_scope": "private native image probes"},
            "entrypoints": {"conversion": "weights applied; other axes retained as intended runtime settings",
                            "benchmark": "CPU/CUDA; independent reference and performance reports",
                            "inspect": "CPU/Metal/CUDA; no inference",
                            "public_api": "no JSON loader; BackendOptions translates compute; reduced cache unavailable",
                            "video": "NOT_IN_THIS_CONFIGURATION"},
            "execution_evidence": "kernel arithmetic NOT_COLLECTED until independently traced"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="resolve a configuration without loading weights or initializing a GPU")
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--context", choices=CONTEXTS, default="benchmark")
    validate.add_argument("--output", type=Path)
    available = commands.add_parser("capabilities", help="show supported combinations and implementation limits")
    available.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError("configuration reports require a new output file")
        result = capabilities()
        if args.command == "validate":
            receipt = load_config(args.config)
            validate_context(receipt["config"], args.context)
            result = {"schema_version": 1, "kind": "sam-quantization-config-validation", "valid": True,
                      "context": args.context, "configuration": receipt, "inference_executed": False,
                      "hardware_qualification": "NOT_MEASURED", "kernel_arithmetic": "NOT_COLLECTED"}
        if args.output is None:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"quantization configuration failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
