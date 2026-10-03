#!/usr/bin/env python3
"""Convert pinned SAM 3 image/video weights to task-specific GGUF v3 containers.

Key mapping/container adapted from PABannier/sam3.cpp, MIT License,
Copyright (c) 2025-2026 Pierre-Antoine Bannier. See THIRD_PARTY_NOTICES.md.
"""

import argparse
import gzip
import importlib.metadata
import os
from pathlib import Path
import tempfile

import gguf

from sam3_artifacts import BPE_SHA256, PAB_REVISION, SAM3_REVISION, sha256_file, write_json
from sam3_gguf import (KEEP_F32, HYBRID_PROFILE, HYBRID_F32_PREFIXES, bytes_to_unicode, canonical_shape, converted_array,
                       inspect_tensors, read_gguf, validate_metadata, write_metadata, tensor_schema)


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


def convert(checkpoint, bpe_path, precision, output, task="image"):
    import torch

    if precision == "hybrid" and task != "video":
        raise ValueError("hybrid storage is defined only for full video models")
    output = Path(output).resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.suffix != ".gguf":
        raise ValueError("output must use .gguf; reconvert from the original checkpoint")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite model or manifest: {output}")
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
    try:
        descriptor, model_tmp = tempfile.mkstemp(prefix=".sam3-convert-", dir=output.parent)
        os.close(descriptor)
        temporary.append(Path(model_tmp))
        checkpoint_sha256 = sha256_file(checkpoint)
        writer = gguf.GGUFWriter(model_tmp, "sam3", endianess=gguf.GGUFEndian.LITTLE)
        write_metadata(writer, precision, checkpoint_sha256, vocab, merges, task)
        original_shapes = {}
        for name, (_, tensor) in sorted(renamed.items()):
            array = tensor_array(name, tensor)
            shape = canonical_shape(array.shape)
            dimensions = list(reversed(shape))
            if dimensions != expected[name]:
                raise ValueError(f"{name}: expected GGML dimensions {expected[name]}, got {dimensions}")
            converted = converted_array(name, array, precision)
            original_shapes[name] = list(array.shape)
            dtype = gguf.GGMLQuantizationType.F16 if converted.itemsize == 2 else gguf.GGMLQuantizationType.F32
            writer.add_tensor_info(name, shape, converted.dtype, converted.nbytes, raw_dtype=dtype)
            del array, converted
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for name, (_, tensor) in sorted(renamed.items()):
            array = tensor_array(name, tensor)
            converted = converted_array(name, array, precision)
            writer.write_tensor_data(converted.reshape(canonical_shape(array.shape)),
                                     tensor_endianess=gguf.GGUFEndian.LITTLE)
            del array, converted
        writer.flush()
        for stream in writer.fout:
            os.fsync(stream.fileno())
        writer.close()
        reader = read_gguf(model_tmp)
        _, actual_vocab, actual_merges = validate_metadata(reader, precision, checkpoint_sha256, task)
        if actual_vocab != vocab or actual_merges != [" ".join(pair) for pair in merges]:
            raise ValueError("GGUF tokenizer readback differs from the pinned source")
        inventory = inspect_tensors(reader, precision, expected, original_shapes)
        del reader
        for item in inventory:
            source_name, tensor = renamed[item["name"]]
            item.update(source_name=source_name, source_dtype=str(tensor.dtype),
                        conversion="complex-real-pairs" if tensor.is_complex() else "real")
        manifest = {
            "schema_version": 1, "architecture": "sam3", "container_format": "gguf",
            "container_version": 3, "sam_schema_version": 2 if task == "video" else 1, "task": task,
            "precision": precision, "sam3_revision": SAM3_REVISION, "converter_revision": PAB_REVISION,
            "converter_sha256": sha256_file(__file__),
            "gguf_helper_sha256": sha256_file(Path(__file__).with_name("sam3_gguf.py")),
            "gguf_package_version": importlib.metadata.version("gguf"),
            "checkpoint": {"file": Path(checkpoint).name, "sha256": checkpoint_sha256},
            "bpe": {"file": Path(bpe_path).name, "sha256": sha256_file(bpe_path)},
            "output": {"file": output.name, "sha256": sha256_file(model_tmp), "bytes": Path(model_tmp).stat().st_size},
            "tokenizer": {"vocab_size": len(vocab), "merge_count": len(merges), "context_length": 32,
                          "sot_id": 49406, "eot_id": 49407},
            "options": {"detector_only": task == "image", "preserve_f32": list(KEEP_F32), "one_dimensional_f32": True},
            "tensors": inventory, "skipped": skipped,
        }
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("image", "video"), default="image")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bpe", required=True, type=Path)
    parser.add_argument("--precision", choices=("f32", "f16", "hybrid"),
                        help="weight storage precision: defaults to hybrid for video; required for image")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.precision is None:
        if args.task != "video":
            parser.error("--precision is required for --task image")
        args.precision = "hybrid"
    try:
        convert(args.checkpoint, args.bpe, args.precision, args.output, args.task)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"conversion failed: {error}\n")


if __name__ == "__main__":
    main()
