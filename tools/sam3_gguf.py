"""SAM 3 schema 1 on the pinned official GGUF writer/reader."""

import hashlib
import math
from pathlib import Path
import re
import struct

import gguf
import numpy as np

from sam3_artifacts import BPE_SHA256, SAM3_REVISION


KEEP_F32 = ("embed", "tpos", "pe_gaussian", "token", "no_obj", "no_mem", "gamma", "freqs_cis")
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
    if precision not in ("f32", "f16"):
        raise ValueError(f"unsupported precision: {precision}")
    use_f16 = precision == "f16" and len(shape) >= 2 and not any(part in name for part in KEEP_F32)
    return np.dtype("<f2" if use_f16 else "<f4")


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


def write_metadata(writer, precision, checkpoint_sha256, tokens, merges):
    storage_dtype("", (1,), precision)
    if not re.fullmatch("[0-9a-f]{64}", checkpoint_sha256):
        raise ValueError("checkpoint SHA-256 must be 64 lowercase hexadecimal characters")
    merge_strings = [" ".join(pair) for pair in merges]
    validate_tokenizer(tokens, merge_strings)
    writer.add_custom_alignment(32)
    writer.add_file_type(int(precision == "f16"))
    for key, value in STRING_METADATA.items():
        if key != "general.architecture":  # The official writer adds architecture in its constructor.
            writer.add_string(key, value)
    writer.add_string("sam.source.checkpoint_sha256", checkpoint_sha256)
    for key, value in UINT32_METADATA.items():
        writer.add_uint32(key, value)
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
            if int(raw_type[0]) not in (int(gguf.GGMLQuantizationType.F32), int(gguf.GGMLQuantizationType.F16)):
                raise ValueError("GGUF tensor type must be F32 or F16")
            size = math.prod(dims) * (4 if int(raw_type[0]) == 0 else 2)
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


def validate_metadata(reader, precision=None, checkpoint_sha256=None):
    def require(key, types, expected=None):
        field = reader.get_field(key)
        if field is None or field.types != types:
            raise ValueError(f"GGUF metadata {key} has missing or incompatible type")
        value = field.contents()
        if expected is not None and value != expected:
            raise ValueError(f"GGUF metadata {key} disagrees with the supported SAM 3 schema")
        return value

    for key, value in STRING_METADATA.items():
        require(key, [gguf.GGUFValueType.STRING], value)
    for key, value in UINT32_METADATA.items():
        require(key, [gguf.GGUFValueType.UINT32], value)
    if reader.get_field("general.name") is not None:
        require("general.name", [gguf.GGUFValueType.STRING])
    require("sam3.vision.global_attention_blocks", [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32], GLOBAL_BLOCKS)
    if reader.get_field("general.alignment") is not None:
        require("general.alignment", [gguf.GGUFValueType.UINT32], 32)
    file_type = require("general.file_type", [gguf.GGUFValueType.UINT32])
    if file_type not in (0, 1) or (precision is not None and file_type != int(precision == "f16")):
        raise ValueError("GGUF storage precision disagrees with its provenance")
    digest = require("sam.source.checkpoint_sha256", [gguf.GGUFValueType.STRING], checkpoint_sha256)
    if not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("invalid GGUF checkpoint SHA-256")
    tokens = require("tokenizer.ggml.tokens", [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING])
    merges = require("tokenizer.ggml.merges", [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING])
    validate_tokenizer(tokens, merges)
    return "f16" if file_type else "f32", tokens, merges


def inspect_tensors(reader, precision, expected, original_shapes=None):
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
        dtype = storage_dtype(name, original_shape, precision)
        if dtype != storage_dtype(name, list(reversed(dimensions)), precision):
            raise ValueError(f"{name}: source rank selects storage incompatible with canonical SAM precision")
        wanted_type = gguf.GGMLQuantizationType.F16 if dtype.itemsize == 2 else gguf.GGMLQuantizationType.F32
        if tensor.tensor_type != wanted_type:
            raise ValueError(f"{name}: GGUF dtype disagrees with the precision policy")
        flat = tensor.data.reshape(-1)
        for offset in range(0, flat.size, 1024 * 1024):
            if not np.isfinite(flat[offset:offset + 1024 * 1024]).all():
                raise ValueError(f"{name}: non-finite GGUF tensor")
        inventory.append({"name": name, "shape": list(original_shape), "ggml_shape": dimensions,
                          "dtype": "float16" if dtype.itemsize == 2 else "float32",
                          "offset": int(tensor.data_offset), "bytes": int(tensor.n_bytes),
                          "sha256": hashlib.sha256(memoryview(tensor.data).cast("B")).hexdigest()})
    return inventory
