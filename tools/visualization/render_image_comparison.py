#!/usr/bin/env python3
"""Render actual SAM 3 validation masks against an original-checkpoint reference."""

import argparse
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.sam3_artifacts import artifact_path, read_array, read_json, sha256_file, write_json


PALETTE = [(14, 165, 233), (249, 115, 22), (168, 85, 247), (34, 197, 94), (236, 72, 153)]
PANEL_WIDTH, IMAGE_HEIGHT = 480, 320
FONT = ImageFont.load_default(size=20)
SMALL_FONT = ImageFont.load_default(size=16)


def load_result(directory, case, hashes, result_digest):
    path = directory / "results.json"
    if sha256_file(path) != result_digest:
        raise ValueError(f"{path}: result differs from validation receipt")
    result = read_json(path)
    threshold = case["score_threshold"]
    if result.get("prompt") != case["prompt"] or result.get("score_threshold") != threshold:
        raise ValueError(f"{path}: prompt or threshold differs")
    masks = {}
    for detection in result["detections"]:
        query = detection["query_index"]
        if (type(query) is not int or query < 0 or query in masks or not math.isfinite(detection["score"])
                or not threshold < detection["score"] <= 1.0):
            raise ValueError(f"{path}: invalid selected detection")
        if len(detection["box"]) != 4 or not all(math.isfinite(x) for x in detection["box"]):
            raise ValueError(f"{path}: invalid box")
        metadata = dict(detection["mask"])
        metadata["sha256"] = hashes[metadata["file"]]
        mask = read_array(directory, metadata, "uint8")
        if list(mask.shape) != [result["height"], result["width"]]:
            raise ValueError(f"{path}: mask dimensions differ")
        masks[query] = np.asarray(mask, dtype=bool)
    return result, masks


def output_metrics(reference, actual, reference_masks, actual_masks):
    ref = {d["query_index"]: d for d in reference["detections"]}
    act = {d["query_index"]: d for d in actual["detections"]}
    pairs = []
    for query in sorted(ref.keys() & act.keys()):
        left, right = reference_masks[query], actual_masks[query]
        union = np.count_nonzero(left | right)
        box_fraction = np.abs(np.asarray(ref[query]["box"]) - act[query]["box"]) / np.asarray(
            [reference["width"], reference["height"], reference["width"], reference["height"]])
        pairs.append({"query_index": query,
                      "mask_iou": np.count_nonzero(left & right) / union if union else 1.0,
                      "score_absolute_error": abs(ref[query]["score"] - act[query]["score"]),
                      "box_dimension_fraction": float(np.max(box_fraction))})
    return {"objects": len(act), "selected_queries_match": ref.keys() == act.keys(),
            "detections": pairs, "minimum_mask_iou": min((p["mask_iou"] for p in pairs), default=None)}


def overlay(image, result, masks, colors):
    pixels = np.asarray(image).copy()
    for query, mask in sorted(masks.items()):
        color = colors[query]
        pixels[mask] = (pixels[mask] * 0.58 + np.asarray(color) * 0.42).astype(np.uint8)
        bitmap = Image.fromarray(mask.astype(np.uint8) * 255)
        edge = np.asarray(bitmap) != np.asarray(bitmap.filter(ImageFilter.MinFilter(5)))
        pixels[edge] = color
    rendered = Image.fromarray(pixels)
    draw = ImageDraw.Draw(rendered)
    for detection in result["detections"]:
        draw.rectangle(detection["box"], outline=colors[detection["query_index"]], width=3)
    return rendered


def differences(reference_masks, actual_masks, size):
    pixels = np.full((size[1], size[0], 3), 248, dtype=np.uint8)
    added = np.zeros((size[1], size[0]), dtype=bool)
    missing = added.copy()
    for query in reference_masks.keys() | actual_masks.keys():
        left = reference_masks.get(query, np.zeros_like(added))
        right = actual_masks.get(query, np.zeros_like(added))
        pixels[left | right] = (200, 208, 216)
        added |= right & ~left
        missing |= left & ~right
    # Highlight thin boundary differences; metrics use undilated original masks.
    for mask, color in [(missing, (37, 99, 235)), (added, (225, 29, 72))]:
        visible = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(7))) > 0
        pixels[visible] = color
    return Image.fromarray(pixels), int(np.count_nonzero(added)), int(np.count_nonzero(missing))


def panel(image, title, lines, result=None, colors=None):
    detections = (result or {}).get("detections", [])
    legend_rows = math.ceil(len(detections) / 3)
    canvas = Image.new("RGB", (PANEL_WIDTH, IMAGE_HEIGHT + 102 + legend_rows * 24), "white")
    thumbnail = ImageOps.contain(image, (PANEL_WIDTH, IMAGE_HEIGHT))
    left, top = (PANEL_WIDTH - thumbnail.width) // 2, 38 + (IMAGE_HEIGHT - thumbnail.height) // 2
    canvas.paste(thumbnail, (left, top))
    draw = ImageDraw.Draw(canvas)
    placed = []
    badge_font = ImageFont.load_default(size=12)
    for detection in detections:
        query, color = detection["query_index"], colors[detection["query_index"]]
        label = f"q{query}"
        text_box = draw.textbbox((0, 0), label, font=badge_font)
        width, height = text_box[2] + 8, 19
        x0, y0, x1, y1 = detection["box"]
        sx, sy = thumbnail.width / image.width, thumbnail.height / image.height
        options = [(left + x0 * sx, top + y0 * sy - height),
                   (left + x0 * sx, top + y1 * sy),
                   (left + x1 * sx - width, top + y0 * sy - height),
                   (left + x1 * sx - width, top + y1 * sy)]
        for x, y in options:
            x = max(left, min(x, left + thumbnail.width - width))
            y = max(top, min(y, top + thumbnail.height - height))
            rectangle = (x, y, x + width, y + height)
            if not any(x < r[2] and x + width > r[0] and y < r[3] and y + height > r[1] for r in placed):
                break
        placed.append(rectangle)
        draw.rectangle(rectangle, fill="white", outline=color, width=2)
        draw.text((x + 4, y + 3), label, fill=(15, 23, 42), font=badge_font)
    draw.text((10, 9), title, fill=(15, 23, 42), font=FONT)
    for row, line in enumerate(lines):
        draw.text((10, IMAGE_HEIGHT + 48 + row * 22), line, fill=(51, 65, 85), font=SMALL_FONT)
    for index, detection in enumerate(detections):
        x, y = 10 + index % 3 * 158, IMAGE_HEIGHT + 98 + index // 3 * 24
        color = colors[detection["query_index"]]
        draw.rectangle((x, y + 4, x + 9, y + 13), fill=color)
        draw.text((x + 15, y), f"q{detection['query_index']} {detection['score']:.3f}",
                  fill=(15, 23, 42), font=SMALL_FONT)
    return canvas


def grid(panels):
    columns = min(3, len(panels))
    width, height = panels[0].width, max(panel.height for panel in panels)
    canvas = Image.new("RGB", (columns * (width + 12) + 12,
                               math.ceil(len(panels) / columns) * (height + 12) + 12), (226, 232, 240))
    for index, image in enumerate(panels):
        padded = Image.new("RGB", (width, height), "white")
        padded.paste(image, (0, 0))
        canvas.paste(padded, (12 + index % columns * (width + 12), 12 + index // columns * (height + 12)))
    return canvas


def render(reference, comparisons, output):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    roots = [reference] + [Path(path).resolve() for _, path in comparisons]
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError("comparison output overlaps an input bundle")
    manifest = read_json(reference / "manifest.json")
    if not comparisons or not manifest.get("cases"):
        raise ValueError("reference and comparison cases must not be empty")
    if len({case["id"] for case in manifest["cases"]}) != len(manifest["cases"]):
        raise ValueError("duplicate reference case IDs")
    for case in manifest["cases"]:
        if (not case["id"] or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in case["id"])
                or not math.isfinite(case["score_threshold"]) or not 0 <= case["score_threshold"] <= 1):
            raise ValueError("invalid case ID or score threshold")
    reference_digest = sha256_file(reference / "manifest.json")
    reports = [read_json(root / "metrics.json") for root in roots[1:]]
    if any(report["reference_manifest_sha256"] != reference_digest for report in reports):
        raise ValueError("comparisons do not use the same original-checkpoint reference")
    evidence = {"reference_kind": manifest["reference_kind"], "backend": None, "cases": []}
    if manifest["reference_kind"] in ("official-checkpoint", "official-checkpoint-rethresholded"):
        reference_title, reference_description = "Meta FP32 reference", "original checkpoint"
    elif manifest["reference_kind"] == "supplementary-converted-weights":
        reference_title, reference_description = "Supplementary reference", "converted weights; diagnostic"
    else:
        reference_title, reference_description = "Reference", "diagnostic reference"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + "-", dir=output.parent) as temporary:
        destination = Path(temporary)
        for case in manifest["cases"]:
            ref_dir = artifact_path(reference, case["directory"])
            input_path = artifact_path(ref_dir, case["input"])
            if sha256_file(input_path) != case["input_sha256"]:
                raise ValueError("reference input changed")
            image = Image.open(input_path).convert("RGB")
            reference_result = read_json(ref_dir / "results.json")
            hashes = {d["mask"]["file"]: d["mask"]["sha256"] for d in reference_result["detections"]}
            ref_result, ref_masks = load_result(ref_dir, case, hashes, case["results_sha256"])
            rows, overlays, difference_panels = [], [], []
            colors = {query: PALETTE[index % len(PALETTE)] for index, query in enumerate(sorted(ref_masks))}
            for (label, _), root, report in zip(comparisons, roots[1:], reports):
                receipt = next(c for c in report["cases"] if c["id"] == case["id"])
                actual, masks = load_result(artifact_path(root, case["id"]), case,
                                            receipt["output_sha256"], receipt["output_sha256"]["results.json"])
                if image.size != (actual["width"], actual["height"]) or actual["backend"] != report["backend"]:
                    raise ValueError("comparison dimensions or backend differ")
                if evidence["backend"] not in (None, actual["backend"]):
                    raise ValueError("use a separate gallery for each backend")
                evidence["backend"] = actual["backend"]
                colors.update({query: PALETTE[query % len(PALETTE)] for query in masks if query not in colors})
                metrics = output_metrics(ref_result, actual, ref_masks, masks)
                iou = metrics["minimum_mask_iou"]
                detail = f"min mask IoU {iou:.6f}" if iou is not None else "mask IoU n/a (no matched objects)"
                overlays.append(panel(overlay(image, actual, masks, colors), label,
                                      [f"{len(masks)} selected objects | {actual['backend']}", detail], actual, colors))
                difference, added, missing = differences(ref_masks, masks, image.size)
                difference_panels.append(panel(difference, label,
                                                [f"added {added} px | missing {missing} px", "colors expanded for visibility"]))
                rows.append({"label": label, "precision": actual["precision"],
                             "storage_profile": actual["storage_profile"], **metrics,
                             "output_quality_passed": receipt.get("output_quality_passed", receipt["passed"]),
                             "candidate_confidences": [{"query_index": d["query_index"], "score": d["score"]}
                                                       for d in actual["detections"]],
                             "tensor_normalized_l2": {name: value["normalized_l2"]
                                                       for name, value in receipt["tensors"].items()}})
            prefix = case["id"]
            baseline = grid([panel(image, "Input", [f"prompt: {case['prompt']}", f"threshold: {case['score_threshold']}"]),
                  panel(overlay(image, ref_result, ref_masks, colors), reference_title,
                        [f"{len(ref_masks)} selected objects", reference_description], ref_result, colors)])
            baseline.save(destination / f"{prefix}-reference.jpg", quality=94)
            grid(overlays).save(destination / f"{prefix}-comparison.jpg", quality=94)
            grid(difference_panels).save(destination / f"{prefix}-differences.png")
            evidence["cases"].append({"id": prefix, "prompt": case["prompt"], "score_threshold": case["score_threshold"],
                                      "reference_objects": len(ref_masks),
                                      "comparisons": rows})
        write_json(destination / "comparison.json", evidence)
        destination.rename(output)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison", action="append", required=True, metavar="LABEL=DIRECTORY")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    comparisons = [value.split("=", 1) for value in args.comparison]
    if any(len(value) != 2 or not all(value) for value in comparisons):
        parser.error("each comparison must be LABEL=DIRECTORY")
    render(args.reference, comparisons, args.output)
    print(f"Rendered actual mask comparisons in {args.output}")


if __name__ == "__main__":
    main()
