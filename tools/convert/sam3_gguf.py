"""SAM 3 image schemas 1/3/4/5 and video schema 2 on the pinned GGUF tools."""

import hashlib
import math
from pathlib import Path
import re
import struct

import gguf
import numpy as np

from tools.convert.sam3_artifacts import BPE_SHA256, SAM3_REVISION
from tools.quantize.weight_policy import (
    QUANTIZATION_MODULES, MAX_QUANTIZATION_MODULE_CSV_LENGTH,
    FULL_MODULE_PROFILE_PREFIX, CUSTOM_MODULE_PROFILE_PREFIX, MODULE_TENSOR_PREFIXES,
    QUANTIZATION_PROFILES, SUPPORTED_STORAGE_PROFILES,
    MIXED_STORAGE_PROFILE, canonical_quantization_modules, quantization_profile,
    parse_mixed_module_precisions,
)


# The original [1,256] score projection is a canonical GGML vector, stored F32.
KEEP_F32 = ("embed", "tpos", "pe_gaussian", "token", "no_obj", "no_mem", "gamma", "freqs_cis",
            "sam_dec.pred_obj_score_head.layers.2.weight")
HYBRID_PROFILE = "visual-tracker-f32-v1"
HYBRID_F32_PREFIXES = ("vit.", "neck.trk.", "mem_attn.", "mem_enc.", "sam_pe.", "sam_dec.",
                      "obj_ptr_proj.", "obj_ptr_tpos_proj.", "trk_mask_ds.")
QUANTIZATION_VERSION = 2
GGML_REVISION = "d7cb574130e6f01ad25b3289685489200febcd74"
GGML_VERSION = "0.26.0"
_PATCH_DIRECTORY = Path(__file__).resolve().parents[2] / "cmake/patches"
_METAL_PATCH_HASH = hashlib.sha256((_PATCH_DIRECTORY / "ggml-precise-metal.patch").read_bytes()).hexdigest()
_CUDA_PATCH_HASH = hashlib.sha256((_PATCH_DIRECTORY / "ggml-precise-cuda.patch").read_bytes()).hexdigest()
_SHORT_DOT_PATCH_HASH = hashlib.sha256((_PATCH_DIRECTORY / "ggml-short-dot-cuda.patch").read_bytes()).hexdigest()
GGML_QUANTIZER_BUILD_COMMITS = (
    GGML_REVISION,
    f"{GGML_REVISION[:8]}-sam-{_METAL_PATCH_HASH[:12]}",
    f"{GGML_REVISION[:8]}-sam-{_METAL_PATCH_HASH[:12]}-{_CUDA_PATCH_HASH[:12]}",
    f"{GGML_REVISION[:8]}-sam-{_METAL_PATCH_HASH[:12]}-{_CUDA_PATCH_HASH[:12]}-{_SHORT_DOT_PATCH_HASH[:12]}",
)
# Historical encoders remain valid provenance for existing packed weights.
# New conversions use the current encoder; encoder changes need not be bitwise
# identical merely because the packed format's quantization version is unchanged.
LEGACY_GGML_QUANTIZER_BUILD_COMMITS = (
    "353b63b439f27ab2cc19dac97ab1681ba6d2d084",
    "353b63b4-sam-0a0b80dd15c2",
    "353b63b4-sam-0a0b80dd15c2-28b1260af845",
    "353b63b4-sam-0a0b80dd15c2-08e802442ae8",
    "353b63b4-sam-0a0b80dd15c2-9417f66f5488",
    "353b63b4-sam-0a0b80dd15c2-e62f040cf989",
    "353b63b4-sam-0a0b80dd15c2-2e0b508b9ca2",
)
LEGACY_GGML_QUANTIZER_IDENTITIES = tuple(
    ("353b63b439f27ab2cc19dac97ab1681ba6d2d084", "0.25.3", commit)
    for commit in LEGACY_GGML_QUANTIZER_BUILD_COMMITS
)


def supported_ggml_quantizer(revision, version, build_commit, *, allow_legacy=False):
    if not all(isinstance(value, str) for value in (revision, version, build_commit)):
        return False
    if revision == GGML_REVISION and version == GGML_VERSION and build_commit in GGML_QUANTIZER_BUILD_COMMITS:
        return True
    return allow_legacy and (revision, version, build_commit) in LEGACY_GGML_QUANTIZER_IDENTITIES


QUANTIZED_ARITHMETIC_PROFILE = "ggml-quantized-native-v1"
_IMAGE_QUANTIZED_MATRIX = re.compile(
    r"(?:vit\.blocks\.(\d+)\.(?:attn\.(?:qkv|proj)|mlp\.lin[12])"
    r"|text\.blocks\.(\d+)\.(?:attn\.(?:in_proj|out_proj)|mlp\.fc[12]))\.weight\Z"
)
_MODULE_LINEAR_PATTERNS = {
    "vision": (re.compile(
        r"vit\.blocks\.[0-9]+\.(?:attn\.(?:qkv|proj)|mlp\.lin[12])\.weight\Z"),),
    "text": (re.compile(
        r"text\.blocks\.[0-9]+\.(?:attn\.(?:in_proj|out_proj)|mlp\.fc[12])\.weight\Z"),
        re.compile(r"text\.resizer\.weight\Z")),
    "fusion": (re.compile(
        r"fenc\.layers\.[0-9]+\.(?:sa\.in_proj_weight|sa\.out_proj\.weight|"
        r"ca\.in_proj_weight|ca\.out_proj\.weight|linear[12]\.weight)\Z"),),
    "decoder": (
        re.compile(
            r"ddec\.layers\.[0-9]+\.(?:sa\.in_proj_weight|sa\.out_proj\.weight|"
            r"ca\.in_proj_weight|ca\.out_proj\.weight|ca_text\.in_proj_weight|"
            r"ca_text\.out_proj\.weight|linear[12]\.weight)\Z"),
        re.compile(r"ddec\.(?:bbox_embed\.layers\.[012]|presence_token_head\.layers\.[01]|"
                   r"ref_point_head\.layers\.[0-9]+)\.weight\Z"),
        re.compile(r"ddec\.boxRPB_embed_[xy]\.layers\.[0-9]+\.weight\Z"),
        re.compile(
            r"geom\.layers\.[0-9]+\.(?:sa\.in_proj_weight|sa\.out_proj\.weight|"
            r"ca\.in_proj_weight|ca\.out_proj\.weight|linear[12]\.weight)\Z"),
        re.compile(r"geom\.(?:points_direct_project|boxes_direct_project|points_pool_project|"
                   r"points_pos_enc_project|boxes_pos_enc_project|final_proj)\.weight\Z"),
        re.compile(r"scoring\.(?:hs_proj|prompt_proj)\.weight\Z"),
        re.compile(r"scoring\.prompt_mlp\.layers\.[0-9]+\.weight\Z"),
        re.compile(r"seg\.cross_attend_prompt\.(?:in_proj_weight|out_proj\.weight)\Z"),
        re.compile(r"seg\.mask_predictor\.mask_embed\.layers\.[0-9]+\.weight\Z"),
    ),
}
_MODULAR_LAYER_INDEX_RULES = (
    (re.compile(r"^vit\.blocks\.([^.]+)\."), 32),
    (re.compile(r"^text\.blocks\.([^.]+)\."), 24),
    (re.compile(r"^fenc\.layers\.([^.]+)\."), 6),
    (re.compile(r"^ddec\.layers\.([^.]+)\."), 6),
    (re.compile(r"^ddec\.bbox_embed\.layers\.([^.]+)\."), 3),
    (re.compile(r"^ddec\.presence_token_head\.layers\.([^.]+)\."), 3),
    (re.compile(r"^ddec\.ref_point_head\.layers\.([^.]+)\."), 2),
    (re.compile(r"^ddec\.boxRPB_embed_[xy]\.layers\.([^.]+)\."), 2),
    (re.compile(r"^geom\.layers\.([^.]+)\."), 3),
    (re.compile(r"^scoring\.prompt_mlp\.layers\.([^.]+)\."), 2),
    (re.compile(r"^seg\.mask_predictor\.mask_embed\.layers\.([^.]+)\."), 3),
)
_MODULAR_F32_ROW_WIDTHS = {
    "geom.points_direct_project.weight": 2,
    "geom.boxes_direct_project.weight": 4,
    "geom.boxes_pos_enc_project.weight": 258,
}
_MODULAR_CANONICAL_VECTOR_F32 = {"ddec.presence_token_head.layers.2.weight"}
IMAGE_PARAMETERS = {
    "sam3.vision.image_size": 1008, "sam3.vision.patch_size": 14,
    "sam3.vision.embedding_length": 1024, "sam3.vision.block_count": 32,
    "sam3.vision.attention.head_count": 16, "sam3.vision.feed_forward_length": 4736,
    "sam3.vision.window_size": 24, "sam3.text.embedding_length": 1024,
    "sam3.text.attention.head_count": 16, "sam3.text.block_count": 24,
    "sam3.text.context_length": 32, "sam3.text.vocab_size": 49408,
    "sam3.text.output_length": 256, "sam3.neck.embedding_length": 256,
    "sam3.fusion.block_count": 6, "sam3.fusion.attention.head_count": 8,
    "sam3.fusion.feed_forward_length": 2048, "sam3.decoder.block_count": 6,
    "sam3.decoder.attention.head_count": 8, "sam3.decoder.feed_forward_length": 2048,
    "sam3.decoder.query_count": 200, "sam3.geometry.block_count": 3,
}
GLOBAL_BLOCKS = [7, 15, 23, 31]
STRING_METADATA = {
    "general.architecture": "sam3", "sam.task": "text_image",
    "sam.source.code_revision": SAM3_REVISION, "sam.tokenizer.sha256": BPE_SHA256,
    "tokenizer.ggml.model": "clip",
}
UINT32_METADATA = {
    "sam.schema_version": 1, "tokenizer.ggml.bos_token_id": 49406,
    "tokenizer.ggml.eos_token_id": 49407, "tokenizer.ggml.padding_token_id": 0,
    **IMAGE_PARAMETERS,
}
VIDEO_PARAMETERS = {
    "sam3.tracker.embedding_length": 256, "sam3.tracker.memory_length": 64,
    "sam3.tracker.attention.block_count": 4, "sam3.tracker.attention.head_count": 1,
    "sam3.tracker.attention.head_length": 256, "sam3.tracker.memory_position_count": 7,
    "sam3.tracker.conditioning_frame_count": 4, "sam3.tracker.pointer_candidate_count": 16,
    "sam3.tracker.mask_memory_size": 1152,
}
VIDEO_STRINGS = {
    "sam3.tracker.policy": "meta-sam3-temporal-v1",
    "sam3.tracker.feature_storage": "bf16", "sam3.tracker.memory_storage": "bf16",
    "sam3.video.preprocessing": "pillow-bicubic-f16-normalize-v1",
}


def task_metadata(task):
    if task not in ("image", "video"):
        raise ValueError(f"unsupported SAM 3 task: {task}")
    strings, integers = dict(STRING_METADATA), dict(UINT32_METADATA)
    if task == "video":
        strings.update(VIDEO_STRINGS)
        strings["sam.task"] = "text_video"
        integers.update(VIDEO_PARAMETERS)
        integers["sam.schema_version"] = 2
    return strings, integers


def tensor_schema(schema, task="image"):
    task_metadata(task)
    result = dict(schema["tensors"])
    if task == "video":
        if set(result) & set(schema["unused_tracker_tensors"]):
            raise ValueError("image and tracker schemas overlap")
        result.update(schema["unused_tracker_tensors"])
    return result


METADATA_LIMIT = 16 * 1024 * 1024


def bytes_to_unicode():
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    code_points = byte_values[:]
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            code_points.append(256 + len(code_points) - 188)
    return dict(zip(byte_values, map(chr, code_points)))


def storage_dtype(name, shape, precision):
    if precision not in ("f32", "f16", "hybrid"):
        raise ValueError(f"unsupported precision: {precision}")
    keep = any(part in name for part in KEEP_F32) or (precision == "hybrid" and name.startswith(HYBRID_F32_PREFIXES))
    use_f16 = precision != "f32" and len(shape) >= 2 and not keep
    return np.dtype("<f2" if use_f16 else "<f4")


def quantization_module_for_tensor(name):
    """Return the canonical SAM 3 module containing an image tensor name."""
    if not isinstance(name, str):
        return None
    for module in QUANTIZATION_MODULES:
        if name.startswith(MODULE_TENSOR_PREFIXES[module]):
            return module
    return None


def _canonical_modular_layer_indices(name):
    for pattern, limit in _MODULAR_LAYER_INDEX_RULES:
        match = pattern.match(name)
        if match is not None:
            token = match.group(1)
            if not re.fullmatch(r"(?:0|[1-9][0-9]*)", token):
                return False
            return int(token) < limit
    return True


def quantization_module_for_linear_weight(name):
    """Return the module for a graph-consumed dense-linear weight, else None."""
    module = quantization_module_for_tensor(name)
    if (module is not None and _canonical_modular_layer_indices(name)
            and any(pattern.fullmatch(name) for pattern in _MODULE_LINEAR_PATTERNS[module])):
        return module
    return None


def quantized_tensor_type(name, shape, precision, storage_profile=None, quantization_modules=None, *, mixed_policy=None):
    """Return the exact GGML type for a whitelisted image matrix, or None for F32."""
    profile = quantization_profile(precision, storage_profile, quantization_modules, mixed_policy=mixed_policy)
    if profile["schema_version"] == 5:
        module = quantization_module_for_tensor(name)
        requested = profile["module_precisions"].get(module, "f32")
        if requested == "f32":
            return None
        return quantized_tensor_type(name, shape, requested, quantization_modules=[module])
    if profile["schema_version"] == 4:
        module = quantization_module_for_tensor(name)
        if module is None or module not in profile["modules"]:
            return None
        if name in _MODULAR_CANONICAL_VECTOR_F32:
            if len(canonical_shape(shape)) != 1:
                raise ValueError(f"{name}: source shape disagrees with its canonical one-dimensional F32 layout")
            return None
        if quantization_module_for_linear_weight(name) is None:
            return None
        if name in _MODULAR_F32_ROW_WIDTHS:
            if len(shape) != 2 or int(shape[-1]) != _MODULAR_F32_ROW_WIDTHS[name]:
                raise ValueError(f"{name}: source shape disagrees with its canonical F32 row layout")
            return None
        box_rpb_layer0 = re.fullmatch(r"ddec\.boxRPB_embed_[xy]\.layers\.0\.weight", name)
        if box_rpb_layer0:
            if len(shape) != 2 or int(shape[-1]) != 2:
                raise ValueError(f"{name}: source shape disagrees with its canonical F32 row layout")
            return None
        if len(canonical_shape(shape)) != 2:
            return None
        row_width = int(shape[-1])
        if row_width <= 0:
            raise ValueError(f"{name}: invalid quantized row width")
        block = profile["block_elements"]
        ggml_type = profile["ggml_type"]
        if row_width % block:
            if not ("fallback_type" in profile and module == "vision"
                    and name.startswith("vit.blocks.") and name.endswith(".mlp.lin2.weight")
                    and row_width == 4736):
                return None
            ggml_type = profile["fallback_type"]
        qtype = gguf.GGMLQuantizationType[ggml_type]
        if row_width % gguf.GGML_QUANT_SIZES[qtype][0]:
            raise ValueError(f"{name}: ne[0] is not divisible by the {qtype.name} fallback block size")
        return qtype

    match = _IMAGE_QUANTIZED_MATRIX.fullmatch(name)
    if match is None:
        return None
    vision_block, text_block = match.groups()
    if ((vision_block is not None and int(vision_block) >= 32)
            or (text_block is not None and int(text_block) >= 24)):
        return None
    if text_block is not None and not profile["quantize_text_linear"]:
        return None
    if len(shape) != 2:
        raise ValueError(f"{name}: quantized image weights must be rank-2 matrices")
    row_width = int(shape[-1])  # GGML ne[0], the innermost contiguous dimension.
    if row_width <= 0:
        raise ValueError(f"{name}: invalid quantized row width")
    block = profile["block_elements"]
    ggml_type = profile["ggml_type"]
    if row_width % block:
        if "fallback_type" not in profile:
            raise ValueError(f"{name}: ne[0] is not divisible by the {precision} block size")
        if not (vision_block is not None and name.endswith(".mlp.lin2.weight") and row_width == 4736):
            raise ValueError(f"{name}: K-format fallback is restricted to vit.blocks.*.mlp.lin2.weight with ne[0]=4736")
        ggml_type = profile["fallback_type"]
    qtype = gguf.GGMLQuantizationType[ggml_type]
    if row_width % gguf.GGML_QUANT_SIZES[qtype][0]:
        raise ValueError(f"{name}: ne[0] is not divisible by the {qtype.name} fallback block size")
    return qtype


def quantization_reason_for_tensor(name, shape, precision, storage_profile, quantization_modules, *, mixed_policy=None):
    """Describe schema-4/5 tensor allocation without changing legacy sidecars."""
    profile = quantization_profile(precision, storage_profile, quantization_modules, mixed_policy=mixed_policy)
    if profile["schema_version"] == 5:
        module = quantization_module_for_tensor(name)
        requested = profile["module_precisions"].get(module, "f32")
        if requested == "f32":
            return "module-f32" if module is not None else "outside-module-policy"
        return quantization_reason_for_tensor(name, shape, requested, None, [module])
    if profile["schema_version"] != 4:
        raise ValueError("per-tensor module reasons are only defined for schema 4")
    module = quantization_module_for_tensor(name)
    if module not in profile["modules"]:
        return "module-not-selected"
    qtype = quantized_tensor_type(name, shape, precision, storage_profile, quantization_modules)
    if qtype is not None:
        if qtype.name == "Q8_0" and precision != "q8_0":
            return "q8-row-fallback"
        return "quantized-linear"
    if len(canonical_shape(shape)) < 2:
        return "canonical-one-dimensional"
    if quantization_module_for_linear_weight(name) is not None:
        if len(shape) >= 2 and int(shape[-1]) % profile["block_elements"]:
            return "row-block-ineligible"
        return "unsupported-linear-layout"
    if any(part in name.lower() for part in ("embed", "position", "pos_embed", "reference_points", "token")):
        return "embedding-or-position"
    if name.endswith(".bias") or "norm" in name.lower():
        return "bias-or-normalization"
    if len(shape) > 2:
        return "convolution-or-spatial-weight"
    if len(shape) < 2:
        return "vector-or-normalization"
    return "outside-linear-whitelist"


def quantized_array(name, array, precision, storage_profile=None, quantization_modules=None, *, mixed_policy=None):
    """Q8_0 encode from F32 with pinned gguf-py; K formats use the native helper."""
    qtype = quantized_tensor_type(name, array.shape, precision, storage_profile, quantization_modules, mixed_policy=mixed_policy)
    if qtype is None:
        raise ValueError(f"{name}: tensor is not in the quantized image matrix whitelist")
    if qtype != gguf.GGMLQuantizationType.Q8_0:
        raise ValueError(f"{name}: {qtype.name} encoding requires sam_quantize_rows")
    values = np.asarray(array, dtype=np.float32, order="C")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite quantization input")
    packed = gguf.quants.quantize(values, qtype)
    expected_bytes = values.size // gguf.GGML_QUANT_SIZES[qtype][0] * gguf.GGML_QUANT_SIZES[qtype][1]
    if packed.dtype != np.uint8 or packed.nbytes != expected_bytes:
        raise ValueError(f"{name}: gguf-py returned an invalid {qtype.name} payload")
    validate_quantized_payload(name, packed, qtype)
    return np.ascontiguousarray(packed)


def validate_quantized_payload(name, packed, qtype):
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    payload = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if payload.size == 0 or payload.size % type_size:
        raise ValueError(f"{name}: invalid {qtype.name} block byte count")
    chunk_bytes = max(type_size, 1024 * 1024 // type_size * type_size)
    for offset in range(0, payload.size, chunk_bytes):
        with np.errstate(over="ignore", invalid="ignore"):
            decoded = gguf.quants.dequantize(payload[offset:offset + chunk_bytes], qtype)
        if decoded.size == 0 or not np.isfinite(decoded).all():
            raise ValueError(f"{name}: {qtype.name} contains a non-finite scale or dequantized value")
    if payload.size // type_size * block_size == 0:
        raise ValueError(f"{name}: empty {qtype.name} tensor")


def converted_array(name, array, precision):
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: non-finite checkpoint tensor")
    dtype = storage_dtype(name, array.shape, precision)
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(array, dtype=dtype, order="C")
    if not np.isfinite(converted).all():
        raise ValueError(f"{name}: conversion overflows {precision}")
    return converted


def canonical_shape(shape):
    shape = list(shape)
    while len(shape) > 1 and shape[0] == 1:
        shape.pop(0)
    return tuple(shape)


def validate_tokenizer(tokens, merges):
    if len(tokens) != 49408 or len(set(tokens)) != 49408:
        raise ValueError("invalid GGUF tokenizer vocabulary")
    base = list(bytes_to_unicode().values())
    if tokens[:512] != base + [token + "</w>" for token in base]:
        raise ValueError("GGUF tokenizer does not contain the pinned byte vocabulary")
    if tokens[-2:] != ["<start_of_text>", "<end_of_text>"]:
        raise ValueError("GGUF tokenizer special tokens are not canonical")
    if len(merges) != 48894 or len(set(merges)) != 48894:
        raise ValueError("invalid GGUF tokenizer merge inventory")
    symbols = set(tokens)
    total_bytes = 0
    for values, limit in ((tokens, 2048), (merges, 2049)):
        for symbol in values:
            length = len(symbol.encode("utf-8"))
            if not length or length > limit or "\0" in symbol:
                raise ValueError("invalid GGUF tokenizer symbol")
            total_bytes += length
    if total_bytes > METADATA_LIMIT:
        raise ValueError("GGUF tokenizer exceeds 16 MiB")
    for rank, entry in enumerate(merges):
        pair = entry.split(" ")
        if (len(pair) != 2 or any(not token or token not in symbols for token in pair)
                or "".join(pair) != tokens[512 + rank]):
            raise ValueError(f"GGUF tokenizer merge rank {rank} is inconsistent")


def write_metadata(writer, precision, checkpoint_sha256, tokens, merges, task="image", storage_profile=None,
                   quantization_modules=None, *, mixed_policy=None):
    quantized = precision in QUANTIZATION_PROFILES or precision == "mixed"
    if quantized:
        if task != "image":
            raise ValueError("quantized storage profiles are defined only for image models")
        profile = quantization_profile(precision, storage_profile, quantization_modules, mixed_policy=mixed_policy)
    else:
        if storage_profile is not None or quantization_modules is not None or mixed_policy is not None:
            raise ValueError("quantization module/profile selection requires a quantized image precision")
        storage_dtype("", (1,), precision)
    if precision == "hybrid" and task != "video":
        raise ValueError("hybrid storage is defined only for full video models")
    if not re.fullmatch("[0-9a-f]{64}", checkpoint_sha256):
        raise ValueError("checkpoint SHA-256 must be 64 lowercase hexadecimal characters")
    merge_strings = [" ".join(pair) for pair in merges]
    validate_tokenizer(tokens, merge_strings)
    writer.add_custom_alignment(32)
    file_type = profile["file_type"] if quantized else int(precision != "f32")
    writer.add_file_type(file_type)
    if precision == "hybrid":
        writer.add_string("sam.storage_profile", HYBRID_PROFILE)
    strings, integers = task_metadata(task)
    if quantized:
        integers["sam.schema_version"] = profile["schema_version"]
    for key, value in strings.items():
        if key != "general.architecture":  # The official writer adds architecture in its constructor.
            writer.add_string(key, value)
    writer.add_string("sam.source.checkpoint_sha256", checkpoint_sha256)
    for key, value in integers.items():
        writer.add_uint32(key, value)
    if quantized:
        writer.add_string("sam.storage_profile", profile["storage_profile"])
        writer.add_uint32("general.quantization_version", QUANTIZATION_VERSION)
        if profile["schema_version"] == 4:
            writer.add_string("sam.quantization.modules", profile["modules_csv"])
        if profile["schema_version"] == 5:
            writer.add_string("sam.quantization.base_precision", profile["base_precision"])
            writer.add_string("sam.quantization.module_precisions", profile["module_precisions_csv"])
            writer.add_string("sam.quantization.policy_sha256", profile["policy_sha256"])
    writer.add_key_value("sam3.vision.global_attention_blocks", GLOBAL_BLOCKS,
                         gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.UINT32)
    writer.add_token_list(tokens)
    writer.add_token_merges(merge_strings)


class BoundedReader(gguf.GGUFReader):
    """Bound the pinned reader's metadata access before it constructs array views."""

    def _get(self, offset, dtype, count=1, override_order=None):
        if (getattr(self, "reading_metadata", True)
                and int(offset) + np.dtype(dtype).itemsize * int(count) > METADATA_LIMIT):
            raise ValueError("GGUF metadata exceeds 16 MiB")
        return super()._get(offset, dtype, count, override_order)

    def _push_field(self, field, skip_sum=False):
        if "\0" in field.name:
            raise ValueError("GGUF metadata key contains NUL")
        if field.types == [gguf.GGUFValueType.STRING] and "\0" in field.contents():
            raise ValueError("GGUF metadata string contains NUL")
        if (field.types == [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING]
                and any("\0" in value for value in field.contents())):
            raise ValueError("GGUF metadata string array contains NUL")
        return super()._push_field(field, skip_sum)

    def _build_tensor_info(self, offset, count):
        result = super()._build_tensor_info(offset, count)
        self.metadata_end = result[0]
        return result

    def _build_tensors(self, start_offset, fields):
        if start_offset > METADATA_LIMIT or self.alignment != 32:
            raise ValueError("GGUF metadata size or alignment is unsupported")
        if np.any(self.data[self.metadata_end:start_offset]):
            raise ValueError("GGUF metadata padding must be zero")
        expected_offset = 0
        for field in fields:
            _, name, _, dimensions, raw_type, relative_offset = field.parts
            dims = [int(value) for value in dimensions]
            if (b"\0" in name.tobytes() or not 1 <= len(dims) <= 4
                    or any(value <= 0 or value > 2**31 - 1 for value in dims)
                    or (len(dims) > 1 and dims[-1] == 1)):
                raise ValueError("GGUF tensor name or dimensions are not canonical")
            try:
                tensor_type = gguf.GGMLQuantizationType(int(raw_type[0]))
            except ValueError as error:
                raise ValueError("GGUF tensor type is unsupported") from error
            if tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
                block_size, type_size = 1, 4 if tensor_type == gguf.GGMLQuantizationType.F32 else 2
            else:
                try:
                    block_size, type_size = gguf.GGML_QUANT_SIZES[tensor_type]
                except KeyError as error:
                    raise ValueError("GGUF tensor type is unsupported") from error
                if dims[0] % block_size:
                    raise ValueError("GGUF quantized tensor ne[0] is not divisible by its block size")
            elements = math.prod(dims)
            if elements > 2**63 - 1:
                raise ValueError("GGUF tensor element count overflows signed 64-bit limits")
            if elements % block_size:
                raise ValueError("GGUF tensor element count is not divisible by its block size")
            size = elements // block_size * type_size
            if size > 2**63 - 1:
                raise ValueError("GGUF tensor byte count overflows signed 64-bit limits")
            if int(relative_offset[0]) != expected_offset:
                raise ValueError("GGUF tensor data must be contiguous in directory order")
            offset = int(start_offset) + expected_offset
            padded_size = (size + 31) // 32 * 32
            if offset + padded_size > self.data.size:
                raise ValueError("GGUF tensor data or final padding is truncated")
            if np.any(self.data[offset + size:offset + padded_size]):
                raise ValueError("GGUF tensor padding must be zero")
            expected_offset += padded_size
        if int(start_offset) + expected_offset != self.data.size:
            raise ValueError("GGUF file contains trailing data")
        self.reading_metadata = False
        return super()._build_tensors(start_offset, fields)


def read_gguf(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:4] != b"GGUF":
        raise ValueError("model requires GGUF v3; reconvert the original checkpoint")
    _, version, tensors, metadata = struct.unpack("<4sIQQ", header)
    if version != 3 or not 1 <= tensors <= 4096 or not 1 <= metadata <= 256:
        raise ValueError("unsupported GGUF version or header counts")
    try:
        return BoundedReader(path)
    except (IndexError, KeyError, OverflowError, UnicodeError) as error:
        raise ValueError(f"malformed GGUF metadata: {error}") from error


def validate_metadata(reader, precision=None, checkpoint_sha256=None, task="image", storage_profile=None,
                      quantization_modules=None, *, mixed_policy=None):
    def require(key, types, expected=None):
        field = reader.get_field(key)
        if field is None or field.types != types:
            raise ValueError(f"GGUF metadata {key} has missing or incompatible type")
        value = field.contents()
        if expected is not None and value != expected:
            raise ValueError(f"GGUF metadata {key} disagrees with the supported SAM 3 schema")
        return value

    strings, integers = task_metadata(task)
    for key, value in strings.items():
        require(key, [gguf.GGUFValueType.STRING], value)
    for key, value in integers.items():
        if key != "sam.schema_version":
            require(key, [gguf.GGUFValueType.UINT32], value)
    schema_version = require("sam.schema_version", [gguf.GGUFValueType.UINT32])
    if reader.get_field("general.name") is not None:
        require("general.name", [gguf.GGUFValueType.STRING])
    require("sam3.vision.global_attention_blocks", [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32], GLOBAL_BLOCKS)
    if reader.get_field("general.alignment") is not None:
        require("general.alignment", [gguf.GGUFValueType.UINT32], 32)
    file_type = require("general.file_type", [gguf.GGUFValueType.UINT32])
    mixed_keys = ("sam.quantization.base_precision", "sam.quantization.module_precisions", "sam.quantization.policy_sha256")
    if schema_version != 5 and (mixed_policy is not None or any(reader.get_field(key) is not None for key in mixed_keys)):
        raise ValueError("mixed weight policy metadata requires SAM schema 5")
    if schema_version in (3, 4, 5):
        if task != "image":
            raise ValueError(f"SAM schema {schema_version} currently supports image models only")
        profile_name = require("sam.storage_profile", [gguf.GGUFValueType.STRING])
        actual_precision = None
        if schema_version == 5:
            actual_precision = "mixed"
            if profile_name != MIXED_STORAGE_PROFILE or quantization_modules is not None or reader.get_field("sam.quantization.modules") is not None:
                raise ValueError("schema 5 requires its mixed storage profile and module format mapping")
            policy = {"base_precision": require(mixed_keys[0], [gguf.GGUFValueType.STRING]),
                      "module_precisions": parse_mixed_module_precisions(require(mixed_keys[1], [gguf.GGUFValueType.STRING])),
                      "policy_sha256": require(mixed_keys[2], [gguf.GGUFValueType.STRING])}
            profile = quantization_profile("mixed", profile_name, mixed_policy=policy)
            if mixed_policy is not None and profile != quantization_profile("mixed", profile_name, mixed_policy=mixed_policy):
                raise ValueError("GGUF mixed policy differs from the requested allocation")
            profile_modules = None
        elif schema_version == 3:
            actual_precision = next((name for name, base in QUANTIZATION_PROFILES.items()
                                     if profile_name in (base["storage_profile"], base["vision_storage_profile"])), None)
            if reader.get_field("sam.quantization.modules") is not None:
                raise ValueError("schema 3 cannot declare schema-4 quantization modules")
            if quantization_modules is not None:
                raise ValueError("quantization modules require a schema-4 quantized image")
            profile_modules = None
        else:
            actual_precision = next((name for name in QUANTIZATION_PROFILES
                                     if profile_name in (f"{FULL_MODULE_PROFILE_PREFIX}{name}-v1",
                                                         f"{CUSTOM_MODULE_PROFILE_PREFIX}{name}-v1")), None)
            raw_modules = require("sam.quantization.modules", [gguf.GGUFValueType.STRING])
            profile_modules = canonical_quantization_modules(raw_modules)
            if raw_modules != ",".join(profile_modules):
                raise ValueError("GGUF quantization modules are not in canonical order")
            if quantization_modules is not None:
                expected_modules = canonical_quantization_modules(quantization_modules)
                if isinstance(quantization_modules, str):
                    canonical_expected = ",".join(expected_modules)
                else:
                    canonical_expected = expected_modules
                if quantization_modules != canonical_expected:
                    raise ValueError("requested quantization modules are not in canonical order")
                if profile_modules != expected_modules:
                    raise ValueError("GGUF quantization modules disagree with the requested modules")
        if actual_precision is None:
            raise ValueError("unsupported SAM image quantization profile")
        if schema_version != 5:
            profile = quantization_profile(actual_precision, profile_name, profile_modules)
        if file_type != profile["file_type"]:
            raise ValueError("GGUF file type disagrees with the SAM quantization profile")
        if storage_profile is not None and storage_profile != profile_name:
            raise ValueError("GGUF storage profile disagrees with the requested profile")
        require("general.quantization_version", [gguf.GGUFValueType.UINT32], QUANTIZATION_VERSION)
    elif schema_version == integers["sam.schema_version"]:
        if file_type not in (0, 1):
            raise ValueError("GGUF storage precision disagrees with its provenance")
        actual_precision = "f16" if file_type else "f32"
        if reader.get_field("sam.storage_profile") is not None:
            require("sam.storage_profile", [gguf.GGUFValueType.STRING], HYBRID_PROFILE)
            if task != "video" or file_type != 1:
                raise ValueError("hybrid storage requires a mixed full-video container")
            actual_precision = "hybrid"
    else:
        raise ValueError("unsupported SAM schema version")
    if precision is not None and precision != actual_precision:
        raise ValueError("GGUF storage precision disagrees with its provenance")
    if storage_profile is not None and schema_version not in (3, 4, 5):
        raise ValueError("storage profile is only valid for a quantized image")
    if quantization_modules is not None and schema_version != 4:
        raise ValueError("quantization modules are only valid for a schema-4 image")
    digest = require("sam.source.checkpoint_sha256", [gguf.GGUFValueType.STRING], checkpoint_sha256)
    if not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("invalid GGUF checkpoint SHA-256")
    tokens = require("tokenizer.ggml.tokens", [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING])
    merges = require("tokenizer.ggml.merges", [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING])
    validate_tokenizer(tokens, merges)
    return actual_precision, tokens, merges


def inspect_tensors(reader, precision, expected, original_shapes=None, storage_profile=None,
                    quantization_modules=None, *, mixed_policy=None):
    if {tensor.name for tensor in reader.tensors} != set(expected):
        raise ValueError("GGUF tensor inventory differs from the SAM 3 schema")
    inventory = []
    for tensor in reader.tensors:
        name = tensor.name
        dimensions = [int(value) for value in tensor.shape]
        if dimensions != expected[name]:
            raise ValueError(f"{name}: GGUF dimensions differ from the SAM 3 schema")
        original_shape = original_shapes[name] if original_shapes is not None else list(reversed(dimensions))
        if (not isinstance(original_shape, list) or not 1 <= len(original_shape) <= 4
                or any(type(value) is not int or value <= 0 or value > 2**31 - 1 for value in original_shape)
                or list(reversed(canonical_shape(original_shape))) != dimensions):
            raise ValueError(f"{name}: original shape does not match the GGUF dimensions")
        if precision in QUANTIZATION_PROFILES or precision == "mixed":
            wanted_type = quantized_tensor_type(name, original_shape, precision, storage_profile,
                                                quantization_modules, mixed_policy=mixed_policy)
            if wanted_type is None:
                wanted_type = gguf.GGMLQuantizationType.F32
            dtype_name = "float32" if wanted_type == gguf.GGMLQuantizationType.F32 else wanted_type.name.lower()
        else:
            dtype = storage_dtype(name, original_shape, precision)
            if dtype != storage_dtype(name, list(reversed(dimensions)), precision):
                raise ValueError(f"{name}: source rank selects storage incompatible with canonical SAM precision")
            wanted_type = gguf.GGMLQuantizationType.F16 if dtype.itemsize == 2 else gguf.GGMLQuantizationType.F32
            dtype_name = "float16" if dtype.itemsize == 2 else "float32"
        if tensor.tensor_type != wanted_type:
            raise ValueError(f"{name}: GGUF dtype disagrees with the precision policy")
        if tensor.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            flat = tensor.data.reshape(-1)
            for offset in range(0, flat.size, 1024 * 1024):
                if not np.isfinite(flat[offset:offset + 1024 * 1024]).all():
                    raise ValueError(f"{name}: non-finite GGUF tensor")
        else:
            validate_quantized_payload(name, tensor.data, tensor.tensor_type)
        inventory.append({"name": name, "shape": list(original_shape), "ggml_shape": dimensions,
                          "dtype": dtype_name,
                          "offset": int(tensor.data_offset), "bytes": int(tensor.n_bytes),
                          "sha256": hashlib.sha256(memoryview(tensor.data).cast("B")).hexdigest()})
        if tensor.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            block_size, type_size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            inventory[-1].update(quantization_block_elements=block_size, quantization_block_bytes=type_size)
    return inventory
