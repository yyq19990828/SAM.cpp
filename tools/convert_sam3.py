#!/usr/bin/env python3
"""Convert pinned SAM 3 image/video weights to task-specific GGUF v3 containers.

Key mapping/container adapted from PABannier/sam3.cpp, MIT License,
Copyright (c) 2025-2026 Pierre-Antoine Bannier. See THIRD_PARTY_NOTICES.md.
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import gguf

from sam3_artifacts import BPE_SHA256, PAB_REVISION, SAM3_REVISION, sha256_file, write_json
from sam3_gguf import (KEEP_F32, HYBRID_PROFILE, HYBRID_F32_PREFIXES, QUANTIZATION_PROFILES,
                       QUANTIZATION_VERSION, QUANTIZED_ARITHMETIC_PROFILE, bytes_to_unicode,
                       canonical_shape, converted_array, inspect_tensors, quantized_array,
                       quantization_profile, quantization_module_for_tensor, quantized_tensor_type,
                       quantization_reason_for_tensor,
                       read_gguf, validate_metadata,
                       write_metadata, tensor_schema, validate_quantized_payload,
                       SUPPORTED_STORAGE_PROFILES)

GGML_REVISION = "353b63b439f27ab2cc19dac97ab1681ba6d2d084"
GGUF_PACKAGE_VERSION = "0.19.0"


def pinned_gguf_version(version=None):
    version = importlib.metadata.version("gguf") if version is None else version
    if version != GGUF_PACKAGE_VERSION:
        raise ValueError(f"converter requires pinned gguf=={GGUF_PACKAGE_VERSION}, found {version}")
    return version


def load_tokenizer(bpe_path):
    if sha256_file(bpe_path) != BPE_SHA256:
        raise ValueError("BPE asset does not match the pinned official source SHA-256")
    with gzip.open(bpe_path, "rt", encoding="utf-8") as stream:
        lines = stream.read().split("\n")
    merges = [tuple(line.split()) for line in lines[1:49152 - 256 - 2 + 1]]
    if len(merges) != 48894 or any(len(pair) != 2 for pair in merges) or len(set(merges)) != len(merges):
        raise ValueError("invalid official BPE merge inventory")
    vocab = list(bytes_to_unicode().values())
    vocab += [value + "</w>" for value in vocab]
    vocab += ["".join(pair) for pair in merges]
    vocab += ["<start_of_text>", "<end_of_text>"]
    if len(vocab) != 49408 or len(set(vocab)) != len(vocab):
        raise ValueError("invalid official BPE vocabulary")
    return vocab, merges


def rename_key(key, task="image"):
    if task not in ("image", "video"):
        raise ValueError(f"unsupported SAM 3 task: {task}")
    if key.startswith("tracker."):
        return (unused_tracker_key(key), None) if task == "video" else (None, "unused tracker tensor")
    if not key.startswith("detector."):
        raise ValueError(f"unrecognized checkpoint tensor: {key}")
    if any(pattern in key for pattern in ("attn_mask", ".dac_", "_dn_", "text_projection")):
        return None, "deterministic buffer or unused training/pooled-text tensor"
    replacements = (
        ("detector.backbone.vision_backbone.trunk.", "vit."),
        (".mlp.fc1.", ".mlp.lin1."), (".mlp.fc2.", ".mlp.lin2."),
        ("detector.backbone.vision_backbone.convs.", "neck.det."),
        ("detector.backbone.vision_backbone.sam2_convs.", "neck.trk."),
        ("detector.backbone.language_backbone.encoder.transformer.resblocks.", "text.blocks."),
        ("detector.backbone.language_backbone.encoder.token_embedding.", "text.token_embed."),
        ("detector.backbone.language_backbone.encoder.positional_embedding", "text.pos_embed"),
        ("detector.backbone.language_backbone.encoder.ln_final.", "text.ln_final."),
        ("detector.backbone.language_backbone.resizer.", "text.resizer."),
        (".attn.in_proj_weight", ".attn.in_proj.weight"),
        (".attn.in_proj_bias", ".attn.in_proj.bias"),
        (".mlp.c_fc.", ".mlp.fc1."), (".mlp.c_proj.", ".mlp.fc2."),
        ("detector.transformer.encoder.layers.", "fenc.layers."),
        (".cross_attn_image.", ".ca."),
        ("detector.transformer.decoder.layers.", "ddec.layers."),
        ("detector.transformer.decoder.", "ddec."),
        (".cross_attn.", ".ca."), (".self_attn.", ".sa."),
        (".catext_norm.", ".norm_ca_text."),
        ("detector.geometry_encoder.", "geom."), ("geom.encode.", "geom.layers."),
        ("detector.segmentation_head.", "seg."),
        ("detector.dot_prod_scoring.", "scoring."),
    )
    name = key
    for source, target in replacements:
        name = name.replace(source, target)
    if name.startswith("neck.trk.") and task == "image":
        return None, "unused interactive tracker neck tensor"
    if name.startswith("detector."):
        raise ValueError(f"unrecognized detector tensor: {key}")
    return name, None


def unused_tracker_key(key):
    replacements = (
        ("detector.backbone.vision_backbone.sam2_convs.", "neck.trk."),
        (".attn.in_proj_weight", ".attn.in_proj.weight"), (".attn.in_proj_bias", ".attn.in_proj.bias"),
        (".cross_attn_image.", ".ca."), (".cross_attn.", ".ca."), (".self_attn.", ".sa."),
        ("tracker.transformer.encoder.layers.", "mem_attn.layers."),
        ("tracker.transformer.encoder.norm.", "mem_attn.norm."),
        ("tracker.maskmem_backbone.", "mem_enc."), ("mem_enc.fuser.layers.", "mem_enc.fuser."),
        ("mem_enc.mask_downsampler.encoder.", "mem_enc.ds."),
        ("tracker.sam_prompt_encoder.", "sam_pe."),
        ("sam_pe.pe_layer.positional_encoding_gaussian_matrix", "sam_pe.pe_gaussian"),
        ("sam_pe.mask_downscaling.", "sam_pe.mask_ds."),
        ("tracker.sam_mask_decoder.", "sam_dec."),
        ("sam_dec.transformer.layers.", "sam_dec.twoway."),
        ("sam_dec.transformer.final_attn_token_to_image.", "sam_dec.final_attn."),
        ("sam_dec.transformer.norm_final_attn.", "sam_dec.final_norm."),
        ("sam_dec.output_upscaling.", "sam_dec.upscale."),
        ("sam_dec.output_hypernetworks_mlps.", "sam_dec.hyper."),
        ("tracker.obj_ptr_proj.", "obj_ptr_proj."), ("tracker.obj_ptr_tpos_proj.", "obj_ptr_tpos_proj."),
        ("tracker.no_obj_ptr", "no_obj_ptr"), ("tracker.no_mem_embed", "no_mem_embed"),
        ("tracker.no_mem_pos_enc", "no_mem_pos_enc"), ("tracker.no_obj_embed_spatial", "no_obj_embed_spatial"),
        ("tracker.maskmem_tpos_enc", "mem_enc.tpos_enc"), ("tracker.mask_downsample.", "trk_mask_ds."),
    )
    for source, target in replacements:
        key = key.replace(source, target)
    return key


def tensor_array(name, tensor):
    import numpy as np
    import torch

    if tensor.is_complex():
        if not name.endswith(".attn.freqs_cis") or tensor.dtype != torch.complex64:
            raise ValueError(f"{name}: only pinned complex64 RoPE frequency tensors are supported")
        tensor = torch.view_as_real(tensor).contiguous()
    elif not tensor.is_floating_point():
        raise ValueError(f"{name}: checkpoint tensors must be floating-point")
    array = tensor.detach().cpu().float().numpy()
    if name == "vit.pos_embed":
        if array.shape != (1, 577, 1024):
            raise ValueError(f"vit.pos_embed: expected [1,577,1024], got {list(array.shape)}")
        array = array[:, 1:, :].reshape(24, 24, 1024)
    if not 1 <= array.ndim <= 4 or any(value <= 0 or value > 2**31 - 1 for value in array.shape):
        raise ValueError(f"{name}: invalid tensor dimensions: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: non-finite checkpoint tensor")
    return array


def quantizer_identity(quantizer):
    if not quantizer.is_absolute():
        raise ValueError("--quantizer must be an absolute executable path")
    quantizer = quantizer.resolve(strict=True)
    if not quantizer.is_file() or not os.access(quantizer, os.X_OK):
        raise ValueError("--quantizer must name an executable file")
    try:
        result = subprocess.run([str(quantizer), "--identity"], check=True, capture_output=True, text=True)
        identity = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read quantizer identity: {error}") from error
    if (not isinstance(identity, dict)
            or identity.get("ggml_revision") != GGML_REVISION
            or type(identity.get("quantization_version")) is not int
            or identity.get("quantization_version") != QUANTIZATION_VERSION):
        raise ValueError("quantizer GGML revision or quantization version is unsupported")
    patch_hash = sha256_file(Path(__file__).resolve().parents[1] / "cmake/patches/ggml-precise-metal.patch")
    expected_patched_commit = f"{GGML_REVISION[:8]}-sam-{patch_hash[:12]}"
    build_commit = identity.get("ggml_build_commit")
    if build_commit not in (GGML_REVISION, expected_patched_commit) or identity.get("ggml_version") != "0.25.3":
        raise ValueError("quantizer is not linked to the pinned GGML 0.25.3 source")

    libraries = {}
    for field, label in (("ggml_library_path", "ggml_library"),
                          ("ggml_quantize_library_path", "ggml_quantize_library")):
        library_value = identity.get(field)
        if not isinstance(library_value, str) or not library_value:
            raise ValueError(f"quantizer did not report {label} path")
        library_path = Path(library_value)
        if not library_path.is_absolute():
            raise ValueError(f"quantizer {label} path is not absolute")
        try:
            library_path = library_path.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"quantizer {label} is unavailable: {error}") from error
        if not library_path.is_file():
            raise ValueError(f"quantizer {label} path is not a file")
        libraries[label] = {"file": library_path.name, "sha256": sha256_file(library_path)}
    return quantizer, {"ggml_revision": GGML_REVISION,
                       "ggml_build_commit": build_commit,
                       "ggml_version": identity["ggml_version"],
                       "ggml_quantization_version": QUANTIZATION_VERSION,
                       "helper": {"file": quantizer.name, "sha256": sha256_file(quantizer)},
                       **libraries}


def quantize_native_rows(name, array, qtype, quantizer, workdir):
    import numpy as np

    values = np.asarray(array, dtype="<f4", order="C")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name}: native quantization requires a finite F32 matrix")
    rows, row_width = map(int, values.shape)
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    if row_width % block_size:
        raise ValueError(f"{name}: ne[0] is not divisible by the {qtype.name} block size")
    expected_bytes = rows * (row_width // block_size) * type_size
    stem = hashlib.sha256(name.encode("utf-8")).hexdigest()
    input_path, output_path = workdir / f"{stem}.f32le", workdir / f"{stem}.packed"
    with input_path.open("xb") as stream:
        values.tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        result = subprocess.run(
            [str(quantizer), "--type", qtype.name.lower(), "--rows", str(rows),
             "--row-width", str(row_width), "--input", str(input_path), "--output", str(output_path)],
            check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        raise ValueError(f"{name}: GGML row quantizer failed: {detail}") from error
    except OSError as error:
        raise ValueError(f"{name}: cannot run GGML row quantizer: {error}") from error
    try:
        reported_bytes = int(result.stdout.strip(), 10)
    except ValueError as error:
        raise ValueError(f"{name}: GGML row quantizer returned an invalid byte count") from error
    packed = np.fromfile(output_path, dtype=np.uint8)
    if reported_bytes != expected_bytes or packed.size != expected_bytes:
        raise ValueError(f"{name}: GGML row quantizer returned an unexpected payload size")
    packed = packed.reshape(rows, row_width // block_size * type_size)
    validate_quantized_payload(name, packed, qtype)
    input_path.unlink()
    output_path.unlink()
    return packed


def convert(checkpoint, bpe_path, precision, output, task="image", quantizer=None, storage_profile=None,
            quantize_modules=None):
    quantized = precision in QUANTIZATION_PROFILES
    gguf_version = importlib.metadata.version("gguf")
    if storage_profile is not None and quantize_modules is not None:
        raise ValueError("--quantize-modules cannot be combined with --storage-profile")
    quantization_modules = None
    if quantized:
        gguf_version = pinned_gguf_version(gguf_version)
        profile = quantization_profile(precision, storage_profile, quantize_modules)
        storage_profile = profile["storage_profile"]
        if profile["schema_version"] == 4:
            quantization_modules = profile["modules"]
    elif storage_profile is not None or quantize_modules is not None:
        raise ValueError("quantization module/profile selection requires a quantized image precision")
    if quantized and task != "image":
        raise ValueError("quantized storage profiles are defined only for image models")
    if precision == "hybrid" and task != "video":
        raise ValueError("hybrid storage is defined only for full video models")
    import torch

    output = Path(output).resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.suffix != ".gguf":
        raise ValueError("output must use .gguf; reconvert from the original checkpoint")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite model or manifest: {output}")
    quantizer_path = None
    quantizer_provenance = None
    if quantized and precision != "q8_0":
        if quantizer is None:
            raise ValueError(f"--quantizer is required for {precision}")
        quantizer_path, quantizer_provenance = quantizer_identity(Path(quantizer))
    elif quantizer is not None:
        raise ValueError("--quantizer is only used by Q6_K/Q5_K/Q4_K profiles")
    vocab, merges = load_tokenizer(bpe_path)
    schema_path = Path(__file__).with_name("sam3_tensor_schema.json")
    from sam3_artifacts import read_json
    schema = read_json(schema_path)
    if schema.get("schema_version") != 1 or schema.get("sam3_revision") != SAM3_REVISION:
        raise ValueError("unsupported detector tensor schema")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must contain a nonempty tensor state dictionary")
    renamed, skipped = {}, []
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("checkpoint state dictionary contains a non-tensor entry")
        name, reason = rename_key(key, task)
        if name is None:
            if reason.startswith("unused"):
                unused_name = unused_tracker_key(key)
                dimensions = list(reversed(tensor.shape))
                while len(dimensions) > 1 and dimensions[-1] == 1:
                    dimensions.pop()
                if (not tensor.is_floating_point() or tensor.is_complex()
                        or schema["unused_tracker_tensors"].get(unused_name) != dimensions):
                    raise ValueError(f"unrecognized or incompatible unused tracker tensor: {key}")
            skipped.append({"name": key, "shape": list(tensor.shape), "reason": reason})
            continue
        if name in renamed:
            raise ValueError(f"duplicate converted tensor name: {name}")
        renamed[name] = (key, tensor)
    expected = tensor_schema(schema, task)
    if set(renamed) != set(expected):
        raise ValueError(f"checkpoint schema mismatch; missing={sorted(set(expected) - set(renamed))}; "
                         f"unknown={sorted(set(renamed) - set(expected))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = []
    published = []
    writer = None
    quant_workdir = None
    try:
        if quantizer_path is not None:
            quant_workdir = Path(tempfile.mkdtemp(prefix=".sam3-quantize-", dir=output.parent))
        descriptor, model_tmp = tempfile.mkstemp(prefix=".sam3-convert-", dir=output.parent)
        os.close(descriptor)
        temporary.append(Path(model_tmp))
        checkpoint_sha256 = sha256_file(checkpoint)
        writer = gguf.GGUFWriter(model_tmp, "sam3", endianess=gguf.GGUFEndian.LITTLE)
        write_metadata(writer, precision, checkpoint_sha256, vocab, merges, task, storage_profile,
                       quantization_modules)
        original_shapes = {}
        for name, (_, tensor) in sorted(renamed.items()):
            array = tensor_array(name, tensor)
            shape = canonical_shape(array.shape)
            dimensions = list(reversed(shape))
            if dimensions != expected[name]:
                raise ValueError(f"{name}: expected GGML dimensions {expected[name]}, got {dimensions}")
            original_shapes[name] = list(array.shape)
            qtype = (quantized_tensor_type(name, array.shape, precision, storage_profile, quantization_modules)
                     if quantized else None)
            if qtype is not None:
                packed = (quantized_array(name, array, precision, storage_profile, quantization_modules)
                          if qtype == gguf.GGMLQuantizationType.Q8_0
                          else quantize_native_rows(name, array, qtype, quantizer_path, quant_workdir))
                # gguf-py converts a packed uint8 byte-shape back to logical dimensions here.
                writer.add_tensor_info(name, packed.shape, packed.dtype, packed.nbytes, raw_dtype=qtype)
                del packed
            else:
                converted = converted_array(name, array, "f32" if quantized else precision)
                dtype = gguf.GGMLQuantizationType.F16 if converted.itemsize == 2 else gguf.GGMLQuantizationType.F32
                writer.add_tensor_info(name, shape, converted.dtype, converted.nbytes, raw_dtype=dtype)
                del converted
            del array
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for name, (_, tensor) in sorted(renamed.items()):
            array = tensor_array(name, tensor)
            qtype = (quantized_tensor_type(name, array.shape, precision, storage_profile, quantization_modules)
                     if quantized else None)
            if qtype is not None:
                packed = (quantized_array(name, array, precision, storage_profile, quantization_modules)
                          if qtype == gguf.GGMLQuantizationType.Q8_0
                          else quantize_native_rows(name, array, qtype, quantizer_path, quant_workdir))
                writer.write_tensor_data(packed, tensor_endianess=gguf.GGUFEndian.LITTLE)
                del packed
            else:
                converted = converted_array(name, array, "f32" if quantized else precision)
                writer.write_tensor_data(converted.reshape(canonical_shape(array.shape)),
                                         tensor_endianess=gguf.GGUFEndian.LITTLE)
                del converted
            del array
        writer.flush()
        for stream in writer.fout:
            os.fsync(stream.fileno())
        writer.close()
        reader = read_gguf(model_tmp)
        _, actual_vocab, actual_merges = validate_metadata(reader, precision, checkpoint_sha256, task,
                                                           storage_profile, quantization_modules)
        if actual_vocab != vocab or actual_merges != [" ".join(pair) for pair in merges]:
            raise ValueError("GGUF tokenizer readback differs from the pinned source")
        inventory = inspect_tensors(reader, precision, expected, original_shapes, storage_profile,
                                    quantization_modules)
        del reader
        for item in inventory:
            source_name, tensor = renamed[item["name"]]
            item.update(source_name=source_name, source_dtype=str(tensor.dtype),
                        conversion="complex-real-pairs" if tensor.is_complex() else "real")
        manifest = {
            "schema_version": 1, "architecture": "sam3", "container_format": "gguf",
            "container_version": 3,
            "sam_schema_version": profile["schema_version"] if quantized else (2 if task == "video" else 1),
            "task": task,
            "precision": precision, "sam3_revision": SAM3_REVISION, "converter_revision": PAB_REVISION,
            "converter_sha256": sha256_file(__file__),
            "gguf_helper_sha256": sha256_file(Path(__file__).with_name("sam3_gguf.py")),
            "gguf_package_version": gguf_version,
            "checkpoint": {"file": Path(checkpoint).name, "sha256": checkpoint_sha256},
            "bpe": {"file": Path(bpe_path).name, "sha256": sha256_file(bpe_path)},
            "output": {"file": output.name, "sha256": sha256_file(model_tmp), "bytes": Path(model_tmp).stat().st_size},
            "tokenizer": {"vocab_size": len(vocab), "merge_count": len(merges), "context_length": 32,
                          "sot_id": 49406, "eot_id": 49407},
            "options": {"detector_only": task == "image", "preserve_f32": list(KEEP_F32), "one_dimensional_f32": True},
            "tensors": inventory, "skipped": skipped,
        }
        if quantized:
            manifest.update(storage_profile=profile["storage_profile"],
                            arithmetic_profile=QUANTIZED_ARITHMETIC_PROFILE,
                            profile_status=profile["profile_status"],
                            quantization={"version": QUANTIZATION_VERSION,
                                          "ggml_quantization_version": QUANTIZATION_VERSION,
                                          "gguf_file_type": profile["file_type"],
                                          "ggml_type": profile["ggml_type"],
                                          "block_elements": profile["block_elements"],
                                          "block_bytes": profile["block_bytes"],
                                          "row_block_fallback": profile.get("fallback_type")})
            if profile["schema_version"] == 3:
                manifest["quantization"]["quantize_text_linear"] = profile["quantize_text_linear"]
                manifest["options"] = {"quantized_image_linear_weights": True,
                                        "quantize_text_linear": profile["quantize_text_linear"],
                                        "other_tensor_storage": "float32",
                                        "row_block_fallback": profile.get("fallback_type")}
            else:
                manifest["quantization"]["quantize_text_linear"] = "text" in profile["modules"]
                manifest["quantization_modules"] = list(profile["modules"])
                manifest["options"] = {"quantized_image_linear_weights": True,
                                        "other_tensor_storage": "float32",
                                        "row_block_fallback": profile.get("fallback_type")}
                for item in inventory:
                    item["module"] = quantization_module_for_tensor(item["name"])
                    item["quantization_reason"] = quantization_reason_for_tensor(
                        item["name"], original_shapes[item["name"]], precision, storage_profile,
                        quantization_modules)
            manifest["quantizer"] = {"implementation": "gguf-py",
                                     "version": gguf_version,
                                     "module": "gguf.quants.quantize",
                                     "ggml_quantization_version": QUANTIZATION_VERSION}
            if quantizer_provenance is not None:
                manifest["quantizer"] = {"implementation": "ggml_quantize_chunk",
                                         "fallback_implementation": "gguf-py",
                                         "fallback_version": gguf_version,
                                         **quantizer_provenance}
            manifest["requirements_lock_sha256"] = sha256_file(Path(__file__).with_name("requirements.lock"))
            for item in inventory:
                item["output_dtype"] = item["dtype"]
                item["conversion"] = "quantized-from-original-f32" if item["dtype"].startswith("q") else "preserved-f32"
        if precision == "hybrid":
            manifest["storage_profile"] = HYBRID_PROFILE
            manifest["options"]["original_f32_prefixes"] = list(HYBRID_F32_PREFIXES)
        descriptor, manifest_tmp = tempfile.mkstemp(prefix=".sam3-manifest-", dir=output.parent)
        os.close(descriptor)
        temporary.append(Path(manifest_tmp))
        write_json(manifest_tmp, manifest)
        # Hard links publish complete files exclusively; a failing second publication rolls back the first.
        os.link(manifest_tmp, manifest_path)
        published.append(manifest_path)
        os.link(model_tmp, output)
        published.append(output)
        print(f"Converted {len(inventory)} tensors to {output} ({manifest['output']['bytes']} bytes)")
        return manifest
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
        for path in temporary:
            path.unlink(missing_ok=True)
        if quant_workdir is not None:
            shutil.rmtree(quant_workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("image", "video"), default="image")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bpe", required=True, type=Path)
    parser.add_argument("--precision", choices=("f32", "f16", "hybrid", "q8_0", "q6_k", "q5_k", "q4_k"),
                        help="weight storage precision: defaults to hybrid for video; required for image")
    parser.add_argument("--quantizer", type=Path, help="absolute path to sam_quantize_rows for K profiles")
    parser.add_argument("--storage-profile", choices=SUPPORTED_STORAGE_PROFILES,
                        help="exact versioned image allocation profile; omitted preserves legacy behavior")
    parser.add_argument("--quantize-modules",
                        help="comma-separated SAM 3 modules for schema-4 custom quantization: vision,text,fusion,decoder")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.precision is None:
        if args.task != "video":
            parser.error("--precision is required for --task image")
        args.precision = "hybrid"
    try:
        arguments = (args.checkpoint, args.bpe, args.precision, args.output, args.task,
                     args.quantizer, args.storage_profile)
        if args.quantize_modules is None:
            convert(*arguments)
        else:
            convert(*arguments, args.quantize_modules)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"conversion failed: {error}\n")


if __name__ == "__main__":
    main()
