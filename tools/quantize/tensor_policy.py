"""Standard-library-only SAM image linear eligibility and exact-name validation."""

import json
from functools import lru_cache
from pathlib import Path
import re

from tools.quantize.weight_policy import MODULE_TENSOR_PREFIXES


MODULE_LINEAR_PATTERNS = {
    "vision": (re.compile(r"vit\.blocks\.[0-9]+\.(?:attn\.(?:qkv|proj)|mlp\.lin[12])\.weight\Z"),),
    "text": (re.compile(r"text\.blocks\.[0-9]+\.(?:attn\.(?:in_proj|out_proj)|mlp\.fc[12])\.weight\Z"),
             re.compile(r"text\.resizer\.weight\Z")),
    "fusion": (re.compile(r"fenc\.layers\.[0-9]+\.(?:sa\.in_proj_weight|sa\.out_proj\.weight|ca\.in_proj_weight|ca\.out_proj\.weight|linear[12]\.weight)\Z"),),
    "decoder": (
        re.compile(r"ddec\.layers\.[0-9]+\.(?:sa\.in_proj_weight|sa\.out_proj\.weight|ca\.in_proj_weight|ca\.out_proj\.weight|ca_text\.in_proj_weight|ca_text\.out_proj\.weight|linear[12]\.weight)\Z"),
        re.compile(r"ddec\.(?:bbox_embed\.layers\.[012]|presence_token_head\.layers\.[01]|ref_point_head\.layers\.[0-9]+)\.weight\Z"),
        re.compile(r"ddec\.boxRPB_embed_[xy]\.layers\.[0-9]+\.weight\Z"),
        re.compile(r"geom\.layers\.[0-9]+\.(?:sa\.in_proj_weight|sa\.out_proj\.weight|ca\.in_proj_weight|ca\.out_proj\.weight|linear[12]\.weight)\Z"),
        re.compile(r"geom\.(?:points_direct_project|boxes_direct_project|points_pool_project|points_pos_enc_project|boxes_pos_enc_project|final_proj)\.weight\Z"),
        re.compile(r"scoring\.(?:hs_proj|prompt_proj)\.weight\Z"),
        re.compile(r"scoring\.prompt_mlp\.layers\.[0-9]+\.weight\Z"),
        re.compile(r"seg\.cross_attend_prompt\.(?:in_proj_weight|out_proj\.weight)\Z"),
        re.compile(r"seg\.mask_predictor\.mask_embed\.layers\.[0-9]+\.weight\Z"),
    ),
}
MODULAR_LAYER_INDEX_RULES = tuple((re.compile(pattern), limit) for pattern, limit in (
    (r"^vit\.blocks\.([^.]+)\.", 32), (r"^text\.blocks\.([^.]+)\.", 24),
    (r"^fenc\.layers\.([^.]+)\.", 6), (r"^ddec\.layers\.([^.]+)\.", 6),
    (r"^ddec\.bbox_embed\.layers\.([^.]+)\.", 3), (r"^ddec\.presence_token_head\.layers\.([^.]+)\.", 3),
    (r"^ddec\.ref_point_head\.layers\.([^.]+)\.", 2), (r"^ddec\.boxRPB_embed_[xy]\.layers\.([^.]+)\.", 2),
    (r"^geom\.layers\.([^.]+)\.", 3), (r"^scoring\.prompt_mlp\.layers\.([^.]+)\.", 2),
    (r"^seg\.mask_predictor\.mask_embed\.layers\.([^.]+)\.", 3),
))
MODULAR_F32_ROW_WIDTHS = {"geom.points_direct_project.weight": 2,
                         "geom.boxes_direct_project.weight": 4,
                         "geom.boxes_pos_enc_project.weight": 258}
MODULAR_CANONICAL_VECTOR_F32 = {"ddec.presence_token_head.layers.2.weight"}
MAX_TENSOR_OVERRIDES = 348
MAX_TENSOR_POLICY_CSV_LENGTH = 65536


def quantization_module_for_tensor(name):
    if isinstance(name, str):
        return next((module for module, prefixes in MODULE_TENSOR_PREFIXES.items() if name.startswith(prefixes)), None)
    return None


def canonical_modular_layer_indices(name):
    for pattern, limit in MODULAR_LAYER_INDEX_RULES:
        match = pattern.match(name)
        if match is not None:
            token = match.group(1)
            if not re.fullmatch(r"(?:0|[1-9][0-9]*)", token) or int(token) >= limit:
                return False
    return True


def quantization_module_for_linear_weight(name):
    module = quantization_module_for_tensor(name)
    if (module is not None and canonical_modular_layer_indices(name)
            and any(pattern.fullmatch(name) for pattern in MODULE_LINEAR_PATTERNS[module])):
        return module
    return None


def overridable_tensor(name, dimensions):
    """Dimensions use GGML order. Overrides cannot expand the existing whitelist."""
    return (quantization_module_for_linear_weight(name) is not None and len(dimensions) == 2
            and all(type(value) is int and value > 0 for value in dimensions)
            and dimensions[0] % 32 == 0 and name not in MODULAR_F32_ROW_WIDTHS
            and name not in MODULAR_CANONICAL_VECTOR_F32
            and not re.fullmatch(r"ddec\.boxRPB_embed_[xy]\.layers\.0\.weight", name))


@lru_cache(maxsize=1)
def canonical_image_schema():
    path = Path(__file__).resolve().parents[1] / "convert/sam3_tensor_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))["tensors"]


def validate_tensor_precisions(value, expected=None):
    if not isinstance(value, dict) or not 1 <= len(value) <= MAX_TENSOR_OVERRIDES:
        raise ValueError(f"tensor_precisions requires 1..{MAX_TENSOR_OVERRIDES} exact tensor names")
    if expected is None:
        expected = canonical_image_schema()
    for name, precision in value.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_.]{1,127}", name)
                or name not in expected):
            raise ValueError(f"unknown exact tensor override: {name!r}")
        if not overridable_tensor(name, expected[name]):
            raise ValueError(f"tensor override cannot target a protected/ineligible tensor: {name}")
        if not isinstance(precision, str) or precision not in ("f32", "q8_0", "q6_k", "q5_k", "q4_k"):
            raise ValueError(f"{name}: tensor format must be f32/q8_0/q6_k/q5_k/q4_k; F16 is not implemented")
    return dict(sorted(value.items()))


def parse_tensor_precisions(csv):
    if not isinstance(csv, str) or not 1 <= len(csv) <= MAX_TENSOR_POLICY_CSV_LENGTH:
        raise ValueError("tensor precision CSV exceeds its bound")
    result = {}
    for field in csv.split(","):
        name, separator, precision = field.partition("=")
        if not separator or name in result:
            raise ValueError("duplicate or malformed tensor precision override")
        result[name] = precision
    resolved = validate_tensor_precisions(result)
    if csv != ",".join(f"{name}={precision}" for name, precision in resolved.items()):
        raise ValueError("tensor precision overrides must be sorted, canonical CSV")
    return resolved
