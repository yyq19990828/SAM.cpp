"""The versioned, raw-tensor interchange shared by the M1 tools and C++ driver."""

import hashlib
import json
import math
from pathlib import Path


SAM3_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
PAB_REVISION = "416186c501d060df7ca02989d49b38080f5f81f3"
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
REQUIRED_TENSORS = {
    "preprocessed_image": ([1, 3, 1008, 1008], "NCHW"),
    "vision_features_0": ([1, 256, 288, 288], "NCHW"),
    "vision_features_1": ([1, 256, 144, 144], "NCHW"),
    "vision_features_2": ([1, 256, 72, 72], "NCHW"),
    "text_features": ([32, 1, 256], "LNC"),
    "fusion_features": ([1, 256, 72, 72], "NCHW"),
    "pred_boxes": ([1, 200, 4], "NQC"),
    "presence_logits": ([1, 1], "NC"),
    "class_logits": ([1, 200, 1], "NQC"),
    "mask_logits": ([1, 200, 288, 288], "NQHW"),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_run_artifacts(build_dir, executable, model, expected_model_sha256):
    """Bind a validation batch to its model, sidecar, binary and build libraries."""
    paths = {Path(executable).absolute(), Path(model).absolute(),
             Path(model).with_suffix(Path(model).suffix + ".manifest.json").absolute()}
    for pattern in ("*.dylib", "*.so*", "*.dll"):
        paths.update(path.absolute() for path in Path(build_dir).rglob(pattern) if path.is_file())
    # Retain alias paths: repointing a library symlink must also change its hash.
    snapshot = {str(path): sha256_file(path) for path in sorted(paths)}
    if snapshot[str(Path(model).absolute())] != expected_model_sha256:
        raise ValueError("validation model changed after provenance inspection")
    return snapshot


def verify_run_artifacts(snapshot):
    for path, digest in snapshot.items():
        if sha256_file(path) != digest:
            raise ValueError(f"validation artifact changed during the batch: {path}")


def freeze_output_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("validation output directory is missing")
    return {path.relative_to(directory).as_posix(): sha256_file(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def verify_output_files(directory, snapshot):
    if freeze_output_files(directory) != snapshot:
        raise ValueError("validation output changed during analysis")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject_constant, object_pairs_hook=reject_duplicates)


def validate_case_manifest(manifest):
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("sam3_revision") != SAM3_REVISION):
        raise ValueError("unsupported case manifest or official source revision")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("case manifest is empty")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case manifest entries must be objects")
        case_id = case.get("id")
        if (not isinstance(case_id, str) or not case_id or case_id in seen
                or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in case_id)):
            raise ValueError(f"invalid or duplicate case ID: {case_id}")
        seen.add(case_id)
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            raise ValueError(f"{case_id}: missing prompt")
        if case.get("score_threshold") != 0.5:
            raise ValueError(f"{case_id}: M1 case threshold must be 0.5")
        digest = case.get("source_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{case_id}: source image hash must be frozen before export")
    return manifest


def load_case_manifest(path):
    return validate_case_manifest(read_json(path))


def artifact_path(directory, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f"invalid artifact filename: {name!r}")
    directory = Path(directory).resolve()
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f"artifact escapes bundle: {name}")
    return path


def read_array(directory, metadata, expected_dtype=None, require_hash=True):
    import numpy as np

    dtype = metadata.get("dtype")
    if dtype not in ("float32", "uint8") or (expected_dtype and dtype != expected_dtype):
        raise ValueError(f"unsupported artifact dtype: {dtype}")
    shape = metadata.get("shape")
    if not isinstance(shape, list) or not shape or any(type(x) is not int or x < 0 for x in shape):
        raise ValueError(f"invalid artifact shape: {shape}")
    path = artifact_path(directory, metadata.get("file"))
    item_size = 4 if dtype == "float32" else 1
    expected_bytes = math.prod(shape) * item_size
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} bytes, got {path.stat().st_size}")
    digest = metadata.get("sha256")
    if require_hash and (not isinstance(digest, str) or len(digest) != 64):
        raise ValueError(f"{path}: missing SHA-256")
    if digest is not None and sha256_file(path) != digest:
        raise ValueError(f"{path}: SHA-256 mismatch")
    array = np.memmap(path, dtype="<f4" if dtype == "float32" else "u1", mode="r", shape=tuple(shape))
    if dtype == "float32" and not np.isfinite(array).all():
        raise ValueError(f"{path}: non-finite values")
    if dtype == "uint8" and np.any(array > 1):
        raise ValueError(f"{path}: mask values must be 0 or 1")
    return array


def read_tensor_index(directory, require_hash=True):
    index = read_json(Path(directory) / "tensors.json")
    if index.get("schema_version") != 1 or index.get("byte_order") != "little":
        raise ValueError("unsupported tensor dump schema or byte order")
    token_ids = index.get("token_ids")
    if (not isinstance(token_ids, list) or len(token_ids) != 32
            or any(type(x) is not int or x < 0 or x >= 49408 for x in token_ids)):
        raise ValueError("tensor dump must contain 32 valid token IDs")
    tensors = index.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("missing tensor inventory")
    missing = REQUIRED_TENSORS.keys() - tensors.keys()
    if missing:
        raise ValueError(f"missing required tensors: {', '.join(sorted(missing))}")
    for name, (shape, layout) in REQUIRED_TENSORS.items():
        metadata = tensors[name]
        if metadata.get("shape") != shape or metadata.get("layout") != layout:
            raise ValueError(f"{name}: expected shape {shape} and layout {layout}")
        read_array(directory, metadata, "float32", require_hash)
    return index


def dump_array(directory, name, array, layout=None):
    import numpy as np

    directory = Path(directory)
    dtype = "uint8" if array.dtype == np.uint8 or array.dtype == bool else "float32"
    array = np.asarray(array, dtype="u1" if dtype == "uint8" else "<f4", order="C")
    if dtype == "float32" and not np.isfinite(array).all():
        raise ValueError(f"{name}: refusing to export non-finite values")
    path = directory / f"{name}.bin"
    with path.open("wb") as stream:
        array.tofile(stream)
    metadata = {"file": path.name, "dtype": dtype, "shape": list(array.shape), "sha256": sha256_file(path)}
    if layout is not None:
        metadata["layout"] = layout
    return metadata
