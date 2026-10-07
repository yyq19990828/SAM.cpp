#!/usr/bin/env python3
"""Measure annotated COCO quality of archived prompted-output screening results."""

import argparse
import contextlib
import copy
import io
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np

from runtime_quantization import GATES_SHA256, load_gates, validate_output
from sam3_artifacts import artifact_path, read_json, sha256_file, verify_run_artifacts, write_json


def api_rle(rle):
    return {"size": list(rle["size"]), "counts": rle["counts"].encode("ascii") if isinstance(rle["counts"], str) else rle["counts"]}


def union_mask(rles, height, width):
    from pycocotools import mask as masks
    if not rles:
        return np.zeros((height, width), dtype=bool)
    if any(list(rle["size"]) != [height, width] for rle in rles):
        raise ValueError("RLE shape differs from COCO image dimensions")
    return masks.decode(masks.merge([api_rle(rle) for rle in rles])).astype(bool)


def prompted_ground_truth(coco, samples):
    """Keep exactly the requested image/category pairs, including absent classes."""
    from pycocotools.coco import COCO
    categories = {row["name"]: row["id"] for row in coco.dataset["categories"]}
    pairs, lookup = set(), {}
    for sample in samples:
        image_id = sample["coco_image_id"]
        if image_id not in coco.imgs:
            raise ValueError("selection image is absent from COCO annotations")
        positive, negative = set(sample["positive_category_ids"]), set(sample["negative_category_ids"])
        if positive & negative:
            raise ValueError("positive and negative prompt labels overlap")
        actual_categories = {row["category_id"] for row in coco.imgToAnns[image_id]}
        if not positive <= actual_categories or negative & actual_categories:
            raise ValueError("prompt labels disagree with frozen COCO annotations")
        prompted = set()
        for prompt in sample["prompts"]:
            if prompt not in categories:
                raise ValueError("prompt is not an exact COCO category name")
            category = categories[prompt]
            key = (image_id, category)
            if key in pairs:
                raise ValueError("duplicate prompted image/category pair")
            pairs.add(key)
            prompted.add(category)
            lookup[(sample["id"], prompt)] = key
        if prompted != positive | negative:
            raise ValueError("prompt inventory differs from labelled categories")
    image_ids = {image for image, _ in pairs}
    category_ids = {category for _, category in pairs}
    subset = COCO()
    subset.dataset = {"info": copy.deepcopy(coco.dataset.get("info", {})),
                      "images": [copy.deepcopy(coco.imgs[image]) for image in sorted(image_ids)],
                      "categories": [copy.deepcopy(coco.cats[category]) for category in sorted(category_ids)],
                      "annotations": [copy.deepcopy(row) for row in coco.dataset["annotations"]
                                      if (row["image_id"], row["category_id"]) in pairs]}
    subset.createIndex()
    return subset, lookup


def annotation_metrics(coco, outputs):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    predictions, pair_metrics, positives, negative_detections, negative_pairs = [], [], [], 0, 0
    for (image_id, category_id), output in sorted(outputs.items()):
        image = coco.imgs[image_id]
        height, width = image["height"], image["width"]
        if (output["height"], output["width"]) != (height, width):
            raise ValueError("output dimensions differ from frozen COCO annotations")
        scores, _ = validate_output(output, 0.5)
        rles = [row["mask"] for row in output["detections"]]
        pred_union = union_mask(rles, height, width)
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], catIds=[category_id]))
        foreground = union_mask([coco.annToRLE(row) for row in annotations if not row.get("iscrowd", 0)], height, width)
        crowd = union_mask([coco.annToRLE(row) for row in annotations if row.get("iscrowd", 0)], height, width)
        foreground &= ~crowd
        pred_valid = pred_union & ~crowd
        if foreground.any():
            intersection = int(np.count_nonzero(foreground & pred_valid))
            union = int(np.count_nonzero(foreground | pred_valid))
            iou = intersection / union
            positives.append(iou)
            kind = "positive"
        elif not annotations:
            iou = None
            kind = "negative"
            negative_pairs += 1
            negative_detections += len(rles)
        else:
            iou = None
            kind = "crowd-only-or-empty"
        pair_metrics.append({"image_id": image_id, "category_id": category_id, "kind": kind,
                             "positive_union_mask_iou": iou, "detections": len(rles),
                             "predicted_pixels": int(np.count_nonzero(pred_union))})
        for row in output["detections"]:
            predictions.append({"image_id": image_id, "category_id": category_id,
                                "segmentation": api_rle(row["mask"]), "score": float(scores[row["query_index"]])})
    if predictions:
        detected = coco.loadRes(predictions)
    else:
        # COCO.loadRes([]) assumes a first detection; represent a valid empty run explicitly.
        detected = COCO()
        detected.dataset = {"info": {}, "images": copy.deepcopy(coco.dataset["images"]),
                            "categories": copy.deepcopy(coco.dataset["categories"]), "annotations": []}
        detected.createIndex()
    evaluator = COCOeval(coco, detected, "segm")
    evaluator.params.imgIds = sorted(coco.imgs)
    evaluator.params.catIds = sorted(coco.cats)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = [None if value < 0 else float(value) for value in evaluator.stats]
    return {"mask_ap": stats[0], "mask_ap50": stats[1], "mask_ap75": stats[2],
            "mask_ap_small": stats[3], "mask_ap_medium": stats[4], "mask_ap_large": stats[5],
            "coco_statistics": stats, "positive_union_mask_miou": float(np.mean(positives)) if positives else None,
            "positive_pairs": len(positives), "negative_pairs": negative_pairs,
            "negative_detection_count": negative_detections,
            "negative_pairs_with_detections": sum(row["kind"] == "negative" and row["detections"] > 0 for row in pair_metrics),
            "excluded_crowd_only_or_empty_pairs": sum(row["kind"] == "crowd-only-or-empty" for row in pair_metrics),
            "pairs": pair_metrics}


def evaluate(args):
    from pycocotools.coco import COCO
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference environment")
    if args.output.exists():
        raise FileExistsError("annotated report output must be new")
    report_path = args.screening / "screening.json"
    report = read_json(report_path)
    if (report.get("schema_version") != 1 or report.get("complete") is not True
            or report.get("kind") != "sam3-vision-runtime-quantization-output-screen"
            or report.get("gates_sha256") != GATES_SHA256):
        raise ValueError("unsupported or incomplete output-screening artifact")
    if report["diagnostic_only"] and not args.allow_diagnostic:
        raise ValueError("partial diagnostic screening requires --allow-diagnostic")
    dataset_path = args.screening / "dataset.json"
    if sha256_file(dataset_path) != report["dataset_sha256"]:
        raise ValueError("archived dataset manifest changed")
    dataset = read_json(dataset_path)
    annotation_digest = sha256_file(args.annotations)
    if annotation_digest != dataset["provenance"]["annotations_sha256"]:
        raise ValueError("COCO annotation identity differs from the dataset split")
    gates = load_gates()
    source_files = [Path(__file__), Path(__file__).with_name("runtime_quantization.py"), Path(__file__).with_name("sam3_artifacts.py")]
    identities = {str(path.resolve()): sha256_file(path) for path in (report_path, dataset_path, args.annotations, *source_files)}
    original_rows = [row for row in report["results"] if row["mode"] == "original"]
    requested_ids = {row["sample_id"] for row in original_rows}
    if not original_rows or len(original_rows) != len(requested_ids):
        raise ValueError("missing or duplicate original-image rows")
    samples = [sample for sample in dataset["samples"] if sample["id"] in requested_ids]
    if len(samples) != len(requested_ids) or any(sample["split"] != "selection" for sample in samples):
        raise ValueError("screening rows must belong to the frozen selection split")
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        original_coco = COCO(str(args.annotations))
        coco, lookup = prompted_ground_truth(original_coco, samples)
        per_mode = {}
        for mode in ("original", *report["summary"]):
            selected = [row for row in report["results"] if row["mode"] == mode]
            if len(selected) != len(samples) or {row["sample_id"] for row in selected} != requested_ids:
                raise ValueError("candidate image inventory differs from original")
            outputs = {}
            for row in selected:
                for prompt in row["prompts"]:
                    key = lookup[(row["sample_id"], prompt["prompt"])]
                    path = artifact_path(args.screening, prompt["file"])
                    if key in outputs or sha256_file(path) != prompt["sha256"]:
                        raise ValueError("duplicate or changed output payload")
                    identities[str(path.resolve())] = prompt["sha256"]
                    outputs[key] = read_json(path)
                    if outputs[key]["prompt"] != prompt["prompt"]:
                        raise ValueError("payload prompt differs from its reference")
            if set(outputs) != set(lookup.values()):
                raise ValueError("candidate prompt inventory differs")
            print(f"Mode: {mode}")
            per_mode[mode] = annotation_metrics(coco, outputs)
    baseline = per_mode["original"]
    comparisons = {}
    for mode, metrics in per_mode.items():
        if mode == "original":
            continue
        if any(m[k] is None for m in (baseline, metrics) for k in ("mask_ap", "positive_union_mask_miou")):
            raise ValueError("annotated screen lacks valid AP or positive mIoU")
        ap_drop = baseline["mask_ap"] - metrics["mask_ap"]
        miou_drop = baseline["positive_union_mask_miou"] - metrics["positive_union_mask_miou"]
        comparisons[mode] = {"mask_ap_drop": ap_drop, "positive_union_mask_miou_drop": miou_drop,
                             "annotated_screen_budget_met": ap_drop <= gates["annotated_evaluation"]["mask_ap_drop_max"]
                             and miou_drop <= gates["annotated_evaluation"]["positive_union_mask_miou_drop_max"],
                             "original_output_screen_passed": report["summary"][mode]["output_screen_passed"]}
    verify_run_artifacts(identities)
    args.output.mkdir(parents=True)
    (args.output / "coco-eval.log").write_text(stream.getvalue())
    write_json(args.output / "metrics.json", {"schema_version": 1, "complete": True,
               "kind": "sam3-prompted-coco-selection-metrics", "diagnostic_only": report["diagnostic_only"],
               "full_model_qualification": False, "final_evaluation_completed": False,
               "scope": "Prompted image/category pairs from selection only; emitted masks above score 0.5; not full val2017 AP",
               "metric_definition": {"score_floor": 0.5, "ap": "COCO segm AP at IoU .50:.05:.95, maxDets=100, original crowd handling",
                                     "miou": "Macro mean over positive prompted pairs; union instances; exclude crowd pixels and crowd-only pairs",
                                     "negative_prompts": "Count emitted detections and pairs with detections separately"},
               "images": len(samples), "prompted_pairs": len(lookup), "gates_sha256": GATES_SHA256,
               "numpy": np.__version__, "pycocotools": version("pycocotools"),
               "artifact_sha256": identities, "modes": per_mode, "comparisons": comparisons})
    print(f"Completed annotated selection metrics: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screening", "annotations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    try:
        evaluate(args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        parser.exit(1, f"COCO screening evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
