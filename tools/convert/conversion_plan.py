"""Read-only conversion allocation and bounded disk estimates; never encode payloads."""

from collections import Counter
import json
from pathlib import Path

import gguf

from tools.convert.sam3_artifacts import SAM3_REVISION, read_json, sha256_file, verify_run_artifacts
from tools.convert.sam3_gguf import (METADATA_LIMIT, canonical_shape, tensor_assignment, tensor_schema,
                                    write_metadata, write_profile_metadata)
from tools.quantize.tensor_policy import validate_tensor_precisions


def allocation_rows(expected, precision, storage_profile=None, modules=None, mixed_policy=None):
    if mixed_policy is not None and "tensor_precisions" in mixed_policy:
        validate_tensor_precisions(mixed_policy["tensor_precisions"], expected)
    return [tensor_assignment(name, dimensions, precision, storage_profile, modules, mixed_policy=mixed_policy)
            for name, dimensions in sorted(expected.items())]


def summarize_allocations(rows):
    types, modules, reasons = {}, {}, Counter()
    for row in rows:
        for groups, key in ((types, row["resolved_dtype"]), (modules, row["module"] or "other")):
            item = groups.setdefault(key, {"tensors": 0, "elements": 0, "bytes": 0})
            item["tensors"] += 1
            item["elements"] += row["elements"]
            item["bytes"] += row["bytes"]
        reasons[row["quantization_reason"]] += 1
    quantized = [row for row in rows if row["resolved_dtype"].startswith("q")]
    elements = sum(row["elements"] for row in rows)
    return {"tensor_count": len(rows), "elements": elements, "payload_bytes": sum(row["bytes"] for row in rows),
            "padded_payload_bytes": sum(row["padded_bytes"] for row in rows),
            "types": dict(sorted(types.items())), "modules": dict(sorted(modules.items())),
            "assignment_reasons": dict(sorted(reasons.items())),
            "quantized_tensor_count": len(quantized),
            "quantized_element_fraction": sum(row["elements"] for row in quantized) / elements if elements else 0,
            "effective_payload_bits_per_element": sum(row["bytes"] for row in rows) * 8 / elements if elements else 0}


def conversion_plan(expected, precision, task="image", storage_profile=None, modules=None, mixed_policy=None,
                    *, tokenizer=None):
    rows = allocation_rows(expected, precision, storage_profile, modules, mixed_policy)
    summary = summarize_allocations(rows)
    writer = gguf.GGUFWriter(None, "sam3", endianess=gguf.GGUFEndian.LITTLE)
    if tokenizer is None:
        write_profile_metadata(writer, precision, "0" * 64, task, storage_profile, modules, mixed_policy=mixed_policy)
    else:
        write_metadata(writer, precision, "0" * 64, *tokenizer, task, storage_profile, modules, mixed_policy=mixed_policy)
    # The pinned writer's actual packer is used, without opening an output file.
    metadata_bytes = sum(len(writer._pack_val(key, gguf.GGUFValueType.STRING, add_vtype=False)) +
                         len(writer._pack_val(value.value, value.type, add_vtype=True, sub_type=value.sub_type))
                         for key, value in writer.kv_data[0].items())
    directory_bytes = sum(24 + len(row["name"].encode("utf-8")) + 8 * len(row["ggml_shape"]) for row in rows)
    known_start = (24 + metadata_bytes + directory_bytes + 31) // 32 * 32
    if known_start > METADATA_LIMIT:
        raise ValueError("conversion metadata and tensor directory exceed the supported bound")
    exact_bytes = known_start + summary["padded_payload_bytes"] if tokenizer is not None else None
    lower = known_start + summary["padded_payload_bytes"]
    upper = exact_bytes if exact_bytes is not None else METADATA_LIMIT + summary["padded_payload_bytes"]
    scratch = max((row["elements"] * 4 + row["bytes"] for row in rows
                   if row["resolved_dtype"] in ("q6_k", "q5_k", "q4_k")), default=0)
    return {"summary": summary, "tensors": rows,
            "gguf_size": {"exact_bytes": exact_bytes, "lower_bound_bytes": lower, "upper_bound_bytes": upper,
                          "evidence": "pinned writer metadata serialization and resolved layout" if exact_bytes is not None else
                                      "schema layout; tokenizer strings unavailable; metadata bounded at 16 MiB",
                          "tokenizer_included": tokenizer is not None, "alignment": 32},
            "disk_budget": {"model_upper_bound_bytes": upper, "native_row_scratch_upper_bound_bytes": scratch,
                            "additional_bytes_excluding_manifest_and_filesystem_overhead": upper + scratch,
                            "manifest_bytes": "NOT_ESTIMATED", "source_checkpoints_included": False,
                            "note": "temporary model is published by hard link; it is not a second model copy"},
            "runtime_memory": {"peak_bytes": "NOT_MEASURED",
                               "cpu_f32_cast_payload_if_all_quantized_weights_materialized_bytes":
                                   sum(row["elements"] * 4 for row in rows if row["resolved_dtype"].startswith("q")),
                               "note": "cast payload is a static inventory, not a simultaneous workspace/RSS peak"},
            "requires_native_encoder_for_conversion": any(row["resolved_dtype"] in ("q6_k", "q5_k", "q4_k") for row in rows)}


def preflight_checkpoint(path, schema, task, precision):
    import torch
    from tools.convert.convert_sam3 import rename_key, unused_tracker_key
    before = path.stat()
    # Storage is memory-mapped: inspect metadata, not all tensor values.
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("checkpoint preview requires a mmap-compatible PyTorch checkpoint") from error
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must contain a nonempty tensor state dictionary")
    expected, inventory = tensor_schema(schema, task), {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("checkpoint state dictionary contains a non-tensor entry")
        name, reason = rename_key(key, task)
        if name is None:
            if reason.startswith("unused") and (not tensor.is_floating_point() or tensor.is_complex() or
                    schema["unused_tracker_tensors"].get(unused_tracker_key(key)) != list(reversed(canonical_shape(tensor.shape)))):
                raise ValueError(f"unrecognized or incompatible unused tracker tensor: {key}")
            continue
        if name in inventory:
            raise ValueError(f"duplicate converted tensor name: {name}")
        if precision == "mixed" and tensor.dtype not in (torch.float32, torch.complex64):
            raise ValueError(f"{name}: mixed conversion requires the original F32 checkpoint")
        shape = list(tensor.shape)
        if tensor.is_complex():
            if not name.endswith(".attn.freqs_cis") or tensor.dtype != torch.complex64:
                raise ValueError(f"{name}: unsupported complex checkpoint tensor")
            shape.append(2)
        elif not tensor.is_floating_point():
            raise ValueError(f"{name}: checkpoint tensors must be floating-point")
        if name == "vit.pos_embed":
            if shape != [1, 577, 1024]:
                raise ValueError("vit.pos_embed must have the original source shape")
            shape = [24, 24, 1024]
        dimensions = list(reversed(canonical_shape(shape)))
        if expected.get(name) != dimensions:
            raise ValueError(f"{name}: checkpoint shape differs from the canonical schema")
        inventory[name] = {"source_name": key, "source_dtype": str(tensor.dtype), "source_shape": list(tensor.shape)}
    if set(inventory) != set(expected):
        raise ValueError("checkpoint preview has missing or unknown canonical tensors")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("checkpoint changed during preview")
    return inventory, {"file": path.name, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                       "sha256": "NOT_COLLECTED", "validation": "mmap names/shapes/dtypes; values NOT_SCANNED"}


def preview(args, *, configuration=None, configuration_inputs=None):
    from tools.convert.convert_sam3 import load_tokenizer, pinned_gguf_version
    pinned_gguf_version()
    if args.output.suffix != ".json":
        raise ValueError("--dry-run --output must name a new .json preview report")
    if args.output.exists():
        raise FileExistsError("preview requires a new output file")
    verify_run_artifacts(configuration_inputs or {})
    schema_path = Path(__file__).with_name("sam3_tensor_schema.json")
    schema = read_json(schema_path)
    if schema.get("schema_version") != 1 or schema.get("sam3_revision") != SAM3_REVISION:
        raise ValueError("unsupported detector tensor schema")
    expected = tensor_schema(schema, args.task)
    policy = None
    if args.precision == "mixed":
        policy = {key: value for key, value in configuration["config"]["weights"].items()
                  if key in ("base_precision", "module_precisions", "tensor_precisions", "policy_sha256")}
    tokenizer = load_tokenizer(args.bpe) if args.bpe is not None else None
    result = {"schema_version": 1, "kind": "sam3-conversion-preview-v1", "complete": True,
              "task": args.task, "precision": args.precision, "configuration": configuration,
              "tensor_schema_sha256": sha256_file(schema_path),
              "checkpoint_preflight": "NOT_REQUESTED", "payload_generated": False, "inference_executed": False,
              **conversion_plan(expected, args.precision, args.task, args.storage_profile, args.quantize_modules,
                                policy, tokenizer=tokenizer)}
    if args.checkpoint is not None:
        inventory, identity = preflight_checkpoint(args.checkpoint, schema, args.task, args.precision)
        result["checkpoint_preflight"] = identity
        for row in result["tensors"]:
            row.update(inventory[row["name"]])
    if args.bpe is not None:
        result["tokenizer"] = {"file": args.bpe.name, "sha256": sha256_file(args.bpe)}
    verify_run_artifacts(configuration_inputs or {})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Previewed {len(expected)} tensors; no model payload generated: {args.output}")
    return result
