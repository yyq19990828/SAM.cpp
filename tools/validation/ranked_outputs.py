"""Policy-free ranked SAM output contracts and spatial mask matching."""

import hashlib
import json
import math
import numpy as np


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()

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
