"""Describe requested precision, resolved policies and bounded execution evidence."""

from collections import Counter
from pathlib import Path

from tools.convert.sam3_artifacts import read_json, sha256_file, verify_run_artifacts, write_json
from tools.quantize.quantization_config import (CACHES, WEIGHTS, resolve_model_configuration,
                                               validate_configuration, validate_recipe_configuration)


def mixed_weight_fields(value):
    return {key: value[key] for key in ("base_precision", "module_precisions", "tensor_precisions", "policy_sha256") if key in value}


def native_weight_policy_matches(value, recipe):
    return recipe["weight_precision"] != "mixed" or mixed_weight_fields(value) == mixed_weight_fields(recipe)


def has_quantized_weights(recipe):
    if recipe["weight_precision"] == "mixed":
        inventory = recipe.get("weight_inventory", {})
        if inventory.get("tensor_count", 0):
            return any(name.startswith("q") for name in inventory["types"])
        return any(value != "f32" for value in (*recipe["module_precisions"].values(), *recipe.get("tensor_precisions", {}).values()))
    return recipe["weight_precision"] in WEIGHTS[2:]


def conversion_manifest(model):
    sidecar = model.with_suffix(model.suffix + ".manifest.json")
    manifest = read_json(sidecar)
    if manifest.get("task") != "image" or manifest.get("precision") not in WEIGHTS:
        raise ValueError("benchmark requires a supported converted SAM 3 image model")
    digest = sha256_file(model)
    if manifest["output"]["sha256"] != digest:
        raise ValueError("model differs from its conversion manifest")
    return manifest, digest, sha256_file(sidecar)


def native_recipe(args, *, collect_environment=True):
    from tools.maintenance.artifact_snapshot import runtime_environment
    activation = getattr(args, "activation", "backend-selected")
    validate_configuration(args.backend, args.compute, args.cache, activation)
    manifest, model_digest, sidecar_digest = conversion_manifest(args.model)
    configuration = resolve_model_configuration(args, manifest)
    storage = manifest.get("storage_profile") or "dense"
    recipe = {"task": "image", "engine": "native", "backend": args.backend,
            "weight_precision": manifest["precision"], "storage_profile": storage,
            "quantization_modules": manifest.get("quantization_modules", []),
            "activation": activation, "compute_mode": args.compute, "feature_cache": args.cache, "threads": 4,
            "checkpoint_sha256": manifest["checkpoint"]["sha256"], "model_sha256": model_digest,
            "binary_sha256": sha256_file(args.binary), "conversion_manifest_sha256": sidecar_digest,
            "model_bytes": args.model.stat().st_size, "cuda_device": 0 if args.backend == "cuda" else None,
            "weight_inventory": weight_inventory(manifest), "quantization_configuration": configuration}
    if manifest["precision"] == "mixed":
        recipe.update(mixed_weight_fields(manifest))
    if collect_environment:
        recipe["environment"] = runtime_environment(args.backend)
    return recipe


def weight_inventory(manifest):
    rows = manifest.get("tensors", [])
    counts, sizes, reasons = Counter(), Counter(), Counter()
    seen = set()
    for row in rows:
        if row["name"] in seen or type(row["bytes"]) is not int or row["bytes"] <= 0:
            raise ValueError("invalid conversion tensor inventory")
        seen.add(row["name"])
        counts[row["dtype"]] += 1
        sizes[row["dtype"]] += row["bytes"]
        if row.get("quantization_reason"):
            reasons[row["quantization_reason"]] += 1
    return {"evidence": "hash-bound conversion manifest" if rows else "NOT_COLLECTED",
            "tensor_count": len(rows), "types": {key: {"tensors": counts[key], "bytes": sizes[key]} for key in sorted(counts)},
            "assignment_reasons": dict(sorted(reasons.items())),
            "scope": "stored weights; excludes runtime casts, activations and image caches"}


def expected_runtime_profile(recipe):
    quantized = has_quantized_weights(recipe)
    if recipe["compute_mode"] == "f16":
        return "ggml-quantized-cuda-f16-v1" if quantized else "ggml-cuda-f16-v1"
    if not quantized:
        return ""
    return {"cpu": "ggml-quantized-weights-f32-v1", "cuda": "ggml-quantized-cuda-native-v1"}.get(
        recipe["backend"], "ggml-quantized-native-v1")


def precision_description(manifest):
    recipe = manifest["recipe"]
    validate_recipe_configuration(recipe)
    backend, compute, cache = (recipe[key] for key in ("backend", "compute_mode", "feature_cache"))
    activation = recipe.get("activation", "backend-selected")
    validate_configuration(backend, compute, cache, activation)
    original = recipe.get("engine") == "original"
    quantized = has_quantized_weights(recipe)
    observed = manifest.get("arithmetic_profile")
    if original:
        compute_scope = "PyTorch F32 oracle with autocast and TF32 disabled; see exporter provenance"
        path = "original reference implementation"
    else:
        if backend == "cuda" and compute == "f16":
            compute_scope = ("F16 operand hint for eligible dense F32/F16 MUL_MAT; eligible unmasked attention uses F16 accumulation hint; "
                             "quantized MUL_MAT, masked attention and other operations retain their own backend policies")
        else:
            compute_scope = "F32 accumulation hint for MUL_MAT/attention; operand representation and kernel internals remain backend-selected"
        path = ("stored F16 weights promoted to F32 when loading on CPU" if backend == "cpu" and recipe["weight_precision"] == "f16" else
                "quantized weights cast to F32 before CPU MUL_MAT" if quantized and backend == "cpu" else
                "native quantized CUDA dispatch may stage RHS as Q8 and use integer dot products, or decode to floating-point operands"
                if quantized and backend == "cuda" else "floating-point operands selected by the backend")
    return {"weight_storage": recipe["weight_precision"], "storage_profile": recipe["storage_profile"],
            "configuration": recipe.get("quantization_configuration"),
            "quantization_modules": recipe["quantization_modules"], "backend": backend,
            "requested": {"weight_storage": recipe["weight_precision"], "activation": activation,
                          "compute": compute, "image_feature_cache": cache},
            "weights": {"inventory": manifest.get("weight_inventory", recipe.get("weight_inventory", {"evidence": "NOT_COLLECTED"})),
                        **mixed_weight_fields(recipe),
                        "uniform_dtype_guarantee": False,
                        "note": "preset names do not describe every tensor; protected tensors stay F32 and K-block exceptions can use Q8_0"},
            "activation_policy": "backend-selected; no independent native activation setting",
            "activation": {"independent_setting_supported": False, "policy": "backend-selected",
                           "scope": "operator tensor types and temporary kernel RHS representations can differ",
                           "int8_fp8_studies": "separate numerical prototypes; not native model execution modes"},
            "requested_compute_mode": compute,
            "compute": {"scope": compute_scope, "source_resolved_path": path,
                        "convolution_columns_type": "f16" if backend == "cuda" and compute == "f16" else "f32",
                        "universal_operand_dtype_guarantee": False, "universal_accumulator_dtype_guarantee": False},
            "feature_cache": cache,
            "cache": {"kind": "host image-encoder features reused across text prompts",
                      "levels": [{"level": i, "storage_type": dtype} for i, dtype in enumerate(CACHES[cache])],
                      "policy_evidence": "source-resolved", "is_llm_kv_cache": False,
                      "native_public_api_setting": False, "benchmark_probe_setting": not original,
                      "scope": "feature levels 0/1/2; text results and other runtime caches are separate"},
            "execution_evidence": {"runtime_policy_identifier": observed,
                                   "runtime_policy_evidence": "producer-reported" if observed is not None else "NOT_COLLECTED",
                                   "expected_native_policy_identifier": None if original else expected_runtime_profile(recipe),
                                   "graph_operand_types": "NOT_COLLECTED",
                                   "kernel_internal_arithmetic": "NOT_COLLECTED"},
            "kernel_precision_evidence": "NOT_COLLECTED"}


def inspect(args):
    validate_configuration(args.backend, args.compute, args.cache, args.activation)
    if args.output.exists():
        raise FileExistsError("precision description requires a new output file")
    verify_run_artifacts(getattr(args, "configuration_inputs", {}))
    manifest, digest, sidecar_digest = conversion_manifest(args.model)
    configuration = resolve_model_configuration(args, manifest)
    inventory = weight_inventory(manifest)
    # Read bounded GGUF tensor headers, not the multi-GB dequantized payload.
    from tools.convert.sam3_gguf import read_gguf, validate_metadata, quantized_tensor_type
    reader = read_gguf(args.model)
    storage = manifest.get("storage_profile") or "dense"
    modules = manifest.get("quantization_modules", [])
    policy_options = {"mixed_policy": mixed_weight_fields(manifest)} if manifest["precision"] == "mixed" else {}
    validate_metadata(reader, manifest["precision"], manifest["checkpoint"]["sha256"],
                      storage_profile=None if storage == "dense" else storage,
                      quantization_modules=(modules or None) if manifest["precision"] != "mixed" else None,
                      **policy_options)
    if manifest["precision"] == "mixed":
        for tensor in reader.tensors:
            wanted = quantized_tensor_type(tensor.name, list(reversed(tensor.shape)), "mixed", storage, **policy_options)
            if tensor.tensor_type.name != (wanted.name if wanted is not None else "F32"):
                raise ValueError(f"{tensor.name}: GGUF type differs from the mixed module policy")
    observed = {tensor.name: (tensor.tensor_type.name.lower(), tensor.n_bytes) for tensor in reader.tensors}
    aliases = {"float32": "f32", "float16": "f16"}
    expected = {row["name"]: (aliases.get(row["dtype"], row["dtype"]), row["bytes"]) for row in manifest["tensors"]}
    if observed != expected:
        raise ValueError("GGUF tensor storage differs from conversion manifest")
    inventory["evidence"] = "GGUF tensor headers and hash-bound conversion manifest"
    recipe = {"engine": "native", "backend": args.backend, "weight_precision": manifest["precision"],
              "storage_profile": storage, "quantization_modules": modules, "compute_mode": args.compute,
              "feature_cache": args.cache, "activation": args.activation,
              "weight_inventory": inventory, **mixed_weight_fields(manifest),
              "quantization_configuration": configuration}
    result = {"schema_version": 1, "kind": "sam3-precision-description-v1", "complete": True,
              "model_sha256": digest, "conversion_manifest_sha256": sidecar_digest,
              "inference_executed": False,
              "precision": precision_description({"recipe": recipe, "weight_inventory": inventory})}
    verify_run_artifacts(getattr(args, "configuration_inputs", {}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"Stored precision inspected; inference not executed: {args.output}", flush=True)
    return result
