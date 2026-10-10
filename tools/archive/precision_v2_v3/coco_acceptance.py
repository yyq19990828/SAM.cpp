"""Ranked COCO metrics and image-clustered paired precision acceptance.

COCO performs mask/GT matching once. Bootstrap repeats those independent image
records with exact stable score-tie ordering; it never samples queries or uses
the v1 threshold-truncated AP as a ranked-AP surrogate.
"""

import contextlib
import copy
import io

import numpy as np

from tools.validation.evaluate_coco_screening import union_mask
from tools.archive.precision_v2_v3.precision_acceptance import combine_statuses, object_gates, upper_gate, validate_output


class CocoAPCache:
    def __init__(self, evaluator):
        self.image_ids = [int(value) for value in evaluator.params.imgIds]
        self.category_ids = [int(value) for value in evaluator.params.catIds]
        self.area_names = list(evaluator.params.areaRngLbl)
        self.recall_thresholds = np.asarray(evaluator.params.recThrs)
        self.iou_thresholds = np.asarray(evaluator.params.iouThrs)
        self.records = {}
        image_count, area_count = len(self.image_ids), len(self.area_names)
        for category_index, category in enumerate(self.category_ids):
            for area_index, area in enumerate(self.area_names):
                records = evaluator.evalImgs[(category_index * area_count + area_index) * image_count:
                                             (category_index * area_count + area_index + 1) * image_count]
                scores, image_indices, true_positive, false_positive = [], [], [], []
                gt_counts = np.zeros(image_count, dtype=np.int64)
                for image_index, record in enumerate(records):
                    if record is None:
                        continue
                    gt_counts[image_index] = np.count_nonzero(np.logical_not(record["gtIgnore"]))
                    scores.extend(record["dtScores"])
                    image_indices.extend([image_index] * len(record["dtScores"]))
                    matched = np.asarray(record["dtMatches"]) != 0
                    visible = np.logical_not(record["dtIgnore"])
                    true_positive.append(matched & visible)
                    false_positive.append(~matched & visible)
                order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
                shape = (len(self.iou_thresholds), 0)
                tp = np.concatenate(true_positive, axis=1)[:, order] if true_positive else np.zeros(shape, bool)
                fp = np.concatenate(false_positive, axis=1)[:, order] if false_positive else np.zeros(shape, bool)
                # Trailing false positives cannot affect the interpolated AP.
                # The cutoff is global per category/area, never per image.
                any_tp = np.flatnonzero(tp.any(axis=0))
                ordered_scores = np.asarray(scores, dtype=np.float64)[order]
                # Retain the whole final score tie: a trailing false positive
                # in one copy can precede the true positive in another copy.
                length = int(np.searchsorted(-ordered_scores, -ordered_scores[any_tp[-1]], side="right")) if len(any_tp) else 0
                tp, fp = tp[:, :length], fp[:, :length]
                ordered_scores = ordered_scores[:length]
                images = np.asarray(image_indices, dtype=np.int64)[order][:length]
                boundaries = np.r_[0, np.flatnonzero((ordered_scores[1:] != ordered_scores[:-1]) |
                                                     (images[1:] != images[:-1])) + 1, length] if length else np.array([0])
                blocks = [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:]) if b - a > 1]
                self.records[(category, area)] = {"tp": tp, "fp": fp, "images": images,
                                                  "gt": gt_counts, "tie_blocks": blocks}

    def precision(self, category, area="all", counts=None):
        value = self.records[(category, area)]
        if counts is None:
            counts = np.ones(len(self.image_ids), dtype=np.int64)
        counts = np.asarray(counts)
        if counts.shape != (len(self.image_ids),) or counts.dtype.kind not in "iu" or (counts < 0).any():
            raise ValueError("bootstrap counts must be non-negative image multiplicities")
        denominator = int(value["gt"] @ counts)
        if not denominator:
            return None
        weights = counts[value["images"]]
        indices = np.repeat(np.arange(len(weights)), weights)
        if value["tie_blocks"]:
            offsets = np.r_[0, np.cumsum(weights)]
            for start, end in value["tie_blocks"]:
                repetitions = int(weights[start])
                if repetitions > 1:
                    # Repeated image copies each retain their original stable
                    # query order inside a same-score group.
                    indices[offsets[start]:offsets[end]] = np.tile(np.arange(start, end), repetitions)
        precision = np.zeros((len(self.iou_thresholds), len(self.recall_thresholds)), dtype=np.float64)
        if not len(indices):
            return precision
        tp = np.cumsum(value["tp"][:, indices], axis=1, dtype=np.float64)
        fp = np.cumsum(value["fp"][:, indices], axis=1, dtype=np.float64)
        for threshold in range(len(self.iou_thresholds)):
            recall = tp[threshold] / denominator
            raw_precision = tp[threshold] / (tp[threshold] + fp[threshold] + np.spacing(1))
            envelope = np.maximum.accumulate(raw_precision[::-1])[::-1]
            positions = np.searchsorted(recall, self.recall_thresholds, side="left")
            valid = positions < len(envelope)
            precision[threshold, valid] = envelope[positions[valid]]
        return precision

    def ap(self, area="all", counts=None, category=None):
        values = [self.precision(key, area, counts) for key in ([category] if category is not None else self.category_ids)]
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None

    def coverage(self):
        return {area: {"images": int(np.count_nonzero(sum((self.records[(category, area)]["gt"]
                                                          for category in self.category_ids), np.zeros(len(self.image_ids), dtype=np.int64)))),
                       "instances": sum(int(self.records[(category, area)]["gt"].sum()) for category in self.category_ids)}
                for area in self.area_names}


def evaluate_ranked(coco, outputs):
    """Return small per-pair records and reusable AP arrays; no dense mask archive."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    predictions, pairs = [], []
    for (image_id, category_id), output in sorted(outputs.items()):
        if image_id not in coco.imgs or category_id not in coco.cats:
            raise ValueError("output image/category is absent from the prompted ground truth")
        image = coco.imgs[image_id]
        if (output["width"], output["height"]) != (image["width"], image["height"]) or output["prompt"] != coco.cats[category_id]["name"]:
            raise ValueError("output identity differs from COCO input")
        scores, _, rles = validate_output(output)
        for query in output["ranked_queries"]:
            predictions.append({"image_id": image_id, "category_id": category_id,
                                "segmentation": rles[query], "score": float(scores[query])})
        deployed = [rles[int(q)] for q in np.flatnonzero(scores > 0.5)]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], catIds=[category_id]))
        height, width = image["height"], image["width"]
        pred_union = union_mask(deployed, height, width)
        foreground = union_mask([coco.annToRLE(row) for row in annotations if not row.get("iscrowd", 0)], height, width)
        crowd = union_mask([coco.annToRLE(row) for row in annotations if row.get("iscrowd", 0)], height, width)
        foreground &= ~crowd
        valid_prediction = pred_union & ~crowd
        if foreground.any():
            kind = "positive"
            miou = float(np.count_nonzero(foreground & valid_prediction) / np.count_nonzero(foreground | valid_prediction))
        else:
            kind, miou = ("negative" if not annotations else "crowd-only-or-empty"), None
        pairs.append({"image_id": image_id, "category_id": category_id, "kind": kind,
                      "positive_union_mask_iou": miou, "detections": len(deployed),
                      "noncrowd_instances": sum(not row.get("iscrowd", 0) for row in annotations)})
    with contextlib.redirect_stdout(io.StringIO()):
        if predictions:
            detected = coco.loadRes(predictions)
        else:
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
    cache = CocoAPCache(evaluator)
    primary = cache.ap()
    official = None if evaluator.stats[0] < 0 else float(evaluator.stats[0])
    if (primary is None) != (official is None) or primary is not None and abs(primary - official) > 1e-12:
        raise ValueError("cached ranked AP differs from the independent COCO accumulator")
    positives = [row["positive_union_mask_iou"] for row in pairs if row["kind"] == "positive"]
    metrics = {"mask_ap": primary, "positive_union_mask_miou": float(np.mean(positives)) if positives else None,
               "area_ap": {area: cache.ap(area) for area in cache.area_names}, "area_coverage": cache.coverage(),
               "category_ap": {str(category): cache.ap(category=category) for category in cache.category_ids},
               "category_coverage": {str(category): {"positive_pairs": sum(row["kind"] == "positive" and row["category_id"] == category for row in pairs),
                                                       "instances": int(cache.records[(category, "all")]["gt"].sum())}
                                      for category in cache.category_ids},
               "coco_statistics": [None if value < 0 else float(value) for value in evaluator.stats],
               "pairs": pairs}
    return metrics, cache


def miou_image_sums(records, image_ids):
    indices = {image: index for index, image in enumerate(image_ids)}
    sums, counts = np.zeros(len(image_ids)), np.zeros(len(image_ids), dtype=np.int64)
    for row in records:
        if row["kind"] == "positive":
            index = indices[row["image_id"]]
            sums[index] += row["positive_union_mask_iou"]
            counts[index] += 1
    return sums, counts


def paired_bootstrap(reference_cache, candidate_cache, reference_pairs, candidate_pairs, common, progress=None):
    if reference_cache.image_ids != candidate_cache.image_ids or reference_cache.category_ids != candidate_cache.category_ids:
        raise ValueError("paired AP caches have different image/category inventories")
    identity = lambda rows: [(row["image_id"], row["category_id"], row["kind"]) for row in rows]
    if identity(reference_pairs) != identity(candidate_pairs):
        raise ValueError("paired mIoU records differ in ground-truth identity")
    image_ids = reference_cache.image_ids
    rs, rn = miou_image_sums(reference_pairs, image_ids)
    cs, cn = miou_image_sums(candidate_pairs, image_ids)
    if not np.array_equal(rn, cn):
        raise ValueError("paired mIoU denominators differ")
    random = np.random.default_rng(common["bootstrap_seed"])
    ap_drops, miou_drops = [], []
    for repetition in range(common["bootstrap_repetitions"]):
        counts = np.bincount(random.integers(0, len(image_ids), len(image_ids)), minlength=len(image_ids))
        ref_ap, candidate_ap = reference_cache.ap(counts=counts), candidate_cache.ap(counts=counts)
        denominator = int(rn @ counts)
        if ref_ap is not None and candidate_ap is not None:
            ap_drops.append(ref_ap - candidate_ap)
        if denominator:
            miou_drops.append(float((rs - cs) @ counts / denominator))
        if progress and (repetition + 1) % 100 == 0:
            progress(repetition + 1, common["bootstrap_repetitions"])
    result = {"repetitions": common["bootstrap_repetitions"], "seed": common["bootstrap_seed"],
              "unit": "image-with-all-prompts", "quantile": common["bootstrap_quantile"], "method": common["bootstrap_method"]}
    for name, values in (("ap", ap_drops), ("miou", miou_drops)):
        result[name + "_valid_repetitions"] = len(values)
        result[name + "_upper_bound"] = (float(np.quantile(values, common["bootstrap_quantile"], method=common["bootstrap_method"]))
                                            if len(values) == common["bootstrap_repetitions"] else None)
    return result


def quality_gates(reference, candidate, comparisons, profile, common, confidence, image_count):
    rows = []
    for name, field in (("ap", "mask_ap"), ("miou", "positive_union_mask_miou")):
        drop = reference[field] - candidate[field] if reference[field] is not None and candidate[field] is not None else None
        rows.append(upper_gate(drop, profile[name + "_drop_max"], name + "_drop", confidence[name + "_upper_bound"]))
    for area in ("small", "medium", "large"):
        r, c = reference["area_ap"][area], candidate["area_ap"][area]
        row = upper_gate(r - c if r is not None and c is not None else None,
                         profile["ap_drop_max"] * common["area_ap_budget_multiplier"], "ap_" + area, required=False)
        coverage = reference["area_coverage"][area]
        if (coverage["images"] < common["area_images_min"] or coverage["instances"] < common["area_instances_min"]) and row["status"] != "FAIL":
            row["status"] = "INCONCLUSIVE"
        row["coverage"] = coverage
        rows.append(row)
    for category, coverage in reference["category_coverage"].items():
        r, c = reference["category_ap"][category], candidate["category_ap"][category]
        row = upper_gate(r - c if r is not None and c is not None else None,
                         max(profile["ap_drop_max"] * common["area_ap_budget_multiplier"], common["category_ap_budget_floor"]),
                         "category_ap_" + category, required=False)
        if coverage["positive_pairs"] < common["category_positive_pairs_min"] or coverage["instances"] < common["category_instances_min"]:
            row["status"] = "NOT_APPLICABLE"
        row["coverage"] = coverage
        rows.append(row)
    r_pairs = {(row["image_id"], row["category_id"]): row for row in reference["pairs"]}
    c_pairs = {(row["image_id"], row["category_id"]): row for row in candidate["pairs"]}
    if r_pairs.keys() != c_pairs.keys() or any(row["kind"] != c_pairs[key]["kind"] for key, row in r_pairs.items()):
        raise ValueError("quality pair inventories differ")
    negative = [key for key, row in r_pairs.items() if row["kind"] == "negative"]
    new = sum(r_pairs[key]["detections"] == 0 and c_pairs[key]["detections"] > 0 for key in negative)
    increase = sum(max(c_pairs[key]["detections"] - r_pairs[key]["detections"], 0) for key in negative)
    rows.append(upper_gate(new / len(negative) if negative else None, profile["new_negative_rate_max"], "new_negative_rate", required=False))
    rows.append(upper_gate(increase / len(negative) if negative else None,
                          profile["new_negative_rate_max"] * common["negative_count_budget_multiplier"], "extra_negative_count_rate", required=False))
    objects = object_gates(comparisons, profile, common)
    rows.extend(objects["checks"])
    rows.append({"name": "evaluation_images", "status": "PASS" if image_count >= common["evaluation_images_min"] else "INCONCLUSIVE",
                 "value": image_count, "minimum": common["evaluation_images_min"]})
    return {"status": combine_statuses([row["status"] for row in rows]), "checks": rows, "object_counts": objects["counts"],
            "negative_pairs": len(negative), "new_negative_pairs": new, "extra_negative_detections": increase,
            "reference_negative_detections": sum(r_pairs[key]["detections"] for key in negative),
            "candidate_negative_detections": sum(c_pairs[key]["detections"] for key in negative)}
