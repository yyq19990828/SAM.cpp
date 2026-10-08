"""Versioned SAM 3 precision policy, ranked payloads and spatial object parity.

This module never regrades a v1 receipt. Arithmetic and deployment qualification
must be supplied by independent, identity-bound evidence.
"""

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from tools.convert.sam3_artifacts import read_json, sha256_file


GATES_PATH = Path(__file__).resolve().parents[2] / "tests/data/sam3-precision-gates-v2.json"
GATES_SHA256 = "4cf06bdc609e5a143b2d74b42f394480bbe39e7510726ecedfe44fde3212a802"
STATUSES = frozenset({"PASS", "FAIL", "INCONCLUSIVE", "NOT_RUN", "NOT_APPLICABLE"})
QUANTIZED = frozenset({"q8_0", "q6_k", "q5_k", "q4_k"})
MODULES = ("vision", "text", "fusion", "decoder")


def load_gates(path=GATES_PATH):
    if sha256_file(path) != GATES_SHA256:
        raise ValueError("v2 gate identity differs from the frozen policy")
    value = read_json(path)
    if value.get("schema_version") != 2 or value.get("gate_set") != "sam3-precision-acceptance-v2":
        raise ValueError("unsupported precision gate version")
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def quality_profile(recipe, gates, incremental=False):
    """Resolve a full declared recipe without guessing from the smallest dtype."""
    if (recipe.get("schema_version") != 2 or recipe.get("task") != "image"
            or recipe.get("backend") not in ("cpu", "metal", "cuda")):
        raise ValueError("recipe requires an explicit v2 image/backend identity")
    weight, arithmetic, cache = (recipe.get(key) for key in ("weight_precision", "compute_mode", "feature_cache"))
    if weight not in {"f32", "f16", "bf16", *QUANTIZED}:
        raise ValueError("unsupported weight precision")
    if arithmetic not in ("f32", "f16", "bf16", "w8a8", "fp8-e4m3", "fp8-e5m2"):
        raise ValueError("unsupported compute precision")
    if cache not in ("f32", *gates["cache_profiles"]):
        raise ValueError("unsupported feature cache recipe")
    storage, modules = recipe.get("storage_profile"), recipe.get("quantization_modules")
    if weight in QUANTIZED:
        valid = {f"image-vision-linear-{weight}-v1": ["vision"],
                 f"image-full-linear-{weight}-v1": list(MODULES),
                 f"image-modules-linear-{weight}-v1": modules}
        if (storage not in valid or not isinstance(modules, list) or not modules
                or any(item not in MODULES for item in modules)
                or modules != [item for item in MODULES if item in modules]
                or modules != valid[storage] or arithmetic not in ("f32", "f16")):
            raise ValueError("unsupported quantized storage/modules/compute combination")
        name = weight
    else:
        if storage != "dense" or modules != []:
            raise ValueError("dense recipe must declare dense storage and no quantized modules")
        if arithmetic in ("w8a8", "fp8-e4m3", "fp8-e5m2"):
            if weight != "f32" or recipe["backend"] != "cuda":
                raise ValueError("activation prototypes require the original F32 CUDA recipe")
            if not recipe.get("activation_recipe_sha256") or not recipe.get("arithmetic_profile"):
                raise ValueError("activation quantization needs explicit scale/layer and arithmetic identity")
            if not re.fullmatch(r"[a-f0-9]{64}", recipe["activation_recipe_sha256"]):
                raise ValueError("invalid activation recipe hash")
            name = "w8a8" if arithmetic == "w8a8" else "fp8"
        else:
            if weight == "bf16" and arithmetic == "f16" or weight == "f16" and arithmetic == "bf16":
                raise ValueError("mixed F16/BF16 requires a separately declared policy")
            name = "bf16" if "bf16" in (weight, arithmetic) else "f16" if "f16" in (weight, arithmetic) else "f32"
    if recipe["backend"] != "cuda" and (arithmetic != "f32" or cache != "f32"):
        raise ValueError("this policy has no qualified non-CUDA compute/cache combination")
    profile = dict(gates["profiles"][name])
    if incremental:
        if cache == "f32":
            raise ValueError("F32 cache has no compressed-cache incremental gate")
        profile.update(gates["cache_profiles"][cache])
    return name, profile


def validate_rle(value, height, width):
    """Validate compressed runs before passing data to the native COCO decoder."""
    if (not isinstance(value, dict) or value.get("size") != [height, width]
            or not isinstance(value.get("counts"), str)):
        raise ValueError("RLE mask shape/counts differ from the output contract")
    encoded = value["counts"]
    if not encoded or len(encoded) > height * width * 2 + 16:
        raise ValueError("invalid RLE payload length")
    runs, position, total = [], 0, 0
    while position < len(encoded):
        number, shift = 0, 0
        while True:
            if position >= len(encoded) or shift >= 65:
                raise ValueError("truncated or overflowing RLE run")
            byte = ord(encoded[position]) - 48
            position += 1
            if not 0 <= byte <= 63:
                raise ValueError("invalid RLE character")
            number |= (byte & 31) << shift
            shift += 5
            if not byte & 32:
                if byte & 16:
                    number |= -1 << shift
                break
        if len(runs) > 2:
            number += runs[-2]
        total += number
        if number < 0 or total > height * width:
            raise ValueError("RLE runs exceed the mask bounds")
        runs.append(number)
    if total != height * width:
        raise ValueError("RLE runs do not cover the image")
    return {"size": [height, width], "counts": encoded.encode("ascii")}


def required_queries(scores, max_detections=100):
    scores = np.asarray(scores, dtype=np.float64)
    ranked = sorted(range(len(scores)), key=lambda query: (-scores[query], query))[:max_detections]
    return ranked, sorted(set(ranked) | set(np.flatnonzero(scores > 0.5).tolist()))


def validate_output(value):
    height, width = value.get("height"), value.get("width")
    if (value.get("schema_version") != 2 or type(height) is not int or type(width) is not int
            or min(height, width) <= 0 or height * width > 100_000_000):
        raise ValueError("invalid v2 image output dimensions/version")
    tokens = value.get("token_ids")
    if (not isinstance(value.get("prompt"), str) or not value["prompt"]
            or not isinstance(tokens, list) or len(tokens) != 32
            or any(type(t) is not int or not 0 <= t < 49408 for t in tokens)):
        raise ValueError("invalid prompt or token identity")
    scores, boxes = np.asarray(value.get("query_scores")), np.asarray(value.get("query_boxes"))
    if (scores.shape != (200,) or boxes.shape != (200, 4)
            or scores.dtype.kind not in "fi" or boxes.dtype.kind not in "fi"
            or not np.isfinite(scores).all() or not np.isfinite(boxes).all()
            or (scores < 0).any() or (scores > 1).any() or (boxes[:, 2:] < boxes[:, :2]).any()):
        raise ValueError("invalid finite SAM query scores/boxes")
    ranked, required = required_queries(scores)
    rows = value.get("masks")
    if (value.get("ranked_queries") != ranked or not isinstance(rows, list)
            or any(not isinstance(row, dict) or type(row.get("query_index")) is not int for row in rows)
            or [row["query_index"] for row in rows] != required):
        raise ValueError("ranked/deployed mask inventory differs from scores")
    masks = {row["query_index"]: validate_rle(row["mask"], height, width) for row in rows}
    return scores.astype(np.float64), boxes.astype(np.float64), masks


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def spatial_assignment(ious, high, box_errors=None, minimum=0.5):
    """Rectangular Hungarian assignment with lexicographic vector costs.

    Count priorities are integers, so arbitrary epsilon weights cannot trade a
    high-confidence match for better geometry. Tie order is lexicographic in
    the sorted query indices. No score-error or acceptance budget is used.
    """
    ious = np.asarray(ious, dtype=np.float64)
    if ious.ndim != 2 or not np.isfinite(ious).all() or (ious < 0).any() or (ious > 1).any():
        raise ValueError("invalid matching IoUs")
    n, columns = ious.shape
    if len(high) != n or not 0 <= minimum <= 1:
        raise ValueError("invalid matching priorities")
    if not n or not columns:
        return []
    boxes = np.zeros_like(ious) if box_errors is None else np.asarray(box_errors, dtype=np.float64)
    if boxes.shape != ious.shape or not np.isfinite(boxes).all() or (boxes < 0).any():
        raise ValueError("invalid matching box distances")
    zero = (0, 0, 0.0, 0.0, 0)
    m = columns + n
    cost = []
    for row in range(n):
        place = (columns + 1) ** (n - 1 - row)
        values = [(-int(bool(high[row])), -1, -float(ious[row, column]), float(boxes[row, column]), column * place)
                  if ious[row, column] >= minimum else (n + 1, 0, 0.0, 0.0, 0) for column in range(columns)]
        cost.append(values + [(0, 0, 0.0, 0.0, columns * place)] * n)
    u, v, p, way = [zero] * (n + 1), [zero] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        p[0], current = row, 0
        best, used = [None] * (m + 1), [False] * (m + 1)
        while True:
            used[current] = True
            active, delta, next_column = p[current], None, 0
            for column in range(1, m + 1):
                if used[column]:
                    continue
                reduced = _sub(_sub(cost[active - 1][column - 1], u[active]), v[column])
                if best[column] is None or reduced < best[column]:
                    best[column], way[column] = reduced, current
                if delta is None or best[column] < delta:
                    delta, next_column = best[column], column
            for column in range(m + 1):
                if used[column]:
                    u[p[column]], v[column] = _add(u[p[column]], delta), _sub(v[column], delta)
                elif best[column] is not None:
                    best[column] = _sub(best[column], delta)
            current = next_column
            if p[current] == 0:
                break
        while current:
            previous = way[current]
            p[current], current = p[previous], previous
    return sorted((p[column] - 1, column - 1) for column in range(1, columns + 1)
                  if p[column] and ious[p[column] - 1, column - 1] >= minimum)


def mask_ious(left, right):
    from pycocotools import mask as masks
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float64)
    result = np.asarray(masks.iou(left, right, [False] * len(right)), dtype=np.float64)
    empty = (masks.area(left) == 0)[:, None] & (masks.area(right) == 0)[None, :]
    result[empty] = 1.0
    return result


def compare_objects(reference, candidate, profile, common, ground_truth=()):
    from pycocotools import mask as masks
    rs, rb, rm = validate_output(reference)
    cs, cb, cm = validate_output(candidate)
    if any(reference[key] != candidate[key] for key in ("width", "height", "prompt", "token_ids")):
        raise ValueError("candidate input/prompt/token identity differs from reference")
    r_ids, c_ids = np.flatnonzero(rs > common["score_threshold"]), np.flatnonzero(cs > common["score_threshold"])
    r_masks, c_masks = [rm[int(q)] for q in r_ids], [cm[int(q)] for q in c_ids]
    dimensions = np.asarray([reference["width"], reference["height"]] * 2)
    distances = np.max(np.abs(rb[r_ids, None] - cb[c_ids][None, :]) / dimensions, axis=2)
    ious = mask_ious(r_masks, c_masks)
    high = rs[r_ids] >= common["high_confidence_min"]
    assignments = spatial_assignment(ious, high, distances, common["matching_iou_min"])
    matched_r, matched_c = {r for r, _ in assignments}, {c for _, c in assignments}
    pairs, bad = [], 0
    for r, c in assignments:
        rq, cq = int(r_ids[r]), int(c_ids[c])
        score_error, box_error = float(abs(rs[rq] - cs[cq])), float(distances[r, c])
        passed = (ious[r, c] >= profile["mask_iou_min"] and score_error <= profile["score_error_max"]
                  and box_error <= profile["box_fraction_max"])
        counted = bool(high[r] or cs[cq] >= common["high_confidence_min"])
        bad += counted and not passed
        pairs.append({"reference_query": rq, "candidate_query": cq, "mask_iou": float(ious[r, c]),
                      "score_error": score_error, "box_fraction": box_error, "counted": counted, "passed": bool(passed)})
    missing = [int(r_ids[r]) for r in range(len(r_ids)) if r not in matched_r and high[r]]
    extra = [int(c_ids[c]) for c in range(len(c_ids)) if c not in matched_c and cs[c_ids[c]] >= common["high_confidence_min"]]
    protected_count, protected_misses = 0, 0
    if ground_truth:
        gt = [validate_rle(row, reference["height"], reference["width"]) for row in ground_truth]
        protected = [r for r, q in enumerate(r_ids) if rs[q] >= common["protected_score_min"]
                     and int(masks.area(r_masks[r])) >= common["protected_area_min"]]
        original_gt = spatial_assignment(mask_ious([r_masks[r] for r in protected], gt),
                                         [True] * len(protected), minimum=common["protected_gt_iou_min"])
        protected_gt = [gt[column] for _, column in original_gt]
        kept = spatial_assignment(mask_ious(protected_gt, c_masks), [True] * len(protected_gt),
                                  minimum=common["protected_gt_iou_min"])
        protected_count, protected_misses = len(protected_gt), len(protected_gt) - len(kept)
    low, upper = common["gray_interval"]
    changes = [{"query": q, "reference_score": float(rs[q]), "candidate_score": float(cs[q]),
                "reference_selected": bool(rs[q] > 0.5), "candidate_selected": bool(cs[q] > 0.5),
                "reference_in_gray_interval": bool(low < rs[q] < upper)} for q in range(200)
               if (rs[q] > 0.5) != (cs[q] > 0.5)]
    return {"high_reference_objects": int(high.sum()), "bad_objects": int(bad) + len(missing) + len(extra),
            "missing_high_objects": len(missing), "extra_high_objects": len(extra),
            "missing_reference_queries": missing, "extra_candidate_queries": extra,
            "unmatched_reference_queries": [int(q) for r, q in enumerate(r_ids) if r not in matched_r],
            "unmatched_candidate_queries": [int(q) for c, q in enumerate(c_ids) if c not in matched_c],
            "protected_objects": protected_count, "protected_misses": protected_misses,
            "reference_detections": len(r_ids), "candidate_detections": len(c_ids),
            "matches": pairs, "fixed_query_selection_changes": changes}


def combine_statuses(statuses):
    if not statuses or any(value not in STATUSES for value in statuses):
        raise ValueError("invalid acceptance status inventory")
    for value in ("FAIL", "NOT_RUN", "INCONCLUSIVE"):
        if value in statuses:
            return value
    return "PASS" if "PASS" in statuses else "NOT_APPLICABLE"


def upper_gate(value, limit, name, upper_bound=None, required=True):
    if value is None:
        return {"name": name, "status": "INCONCLUSIVE", "value": value, "limit": limit, "upper_bound": upper_bound}
    if any(not math.isfinite(x) for x in (value, limit) if x is not None) or upper_bound is not None and not math.isfinite(upper_bound):
        raise ValueError("non-finite quality statistic")
    status = ("FAIL" if value > limit else "INCONCLUSIVE" if required and upper_bound is None
              or upper_bound is not None and upper_bound > limit else "PASS")
    return {"name": name, "status": status, "value": value, "limit": limit, "upper_bound": upper_bound}


def object_gates(comparisons, profile, common, strict=False):
    counts = {key: sum(row[key] for row in comparisons) for key in
              ("high_reference_objects", "bad_objects", "missing_high_objects", "extra_high_objects", "protected_objects", "protected_misses")}
    high = counts["high_reference_objects"]
    rows = []
    for key in ("bad_objects", "missing_high_objects", "extra_high_objects"):
        limit = 0 if strict else profile["bad_object_rate_max"] * (1 if key == "bad_objects" else common["unmatched_budget_fraction"])
        value = counts[key] / high if high else (counts[key] if strict else None)
        row = upper_gate(value, limit, key, required=False)
        if not strict and high < common["high_reference_objects_min"] and row["status"] != "FAIL":
            row["status"] = "INCONCLUSIVE"
        rows.append(row)
    rows.append(upper_gate(counts["protected_misses"], common["protected_misses_max"], "protected_misses", required=False))
    return {"status": combine_statuses([row["status"] for row in rows]), "counts": counts, "checks": rows}
