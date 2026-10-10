#!/usr/bin/env python3
"""Freeze fresh COCO evaluation images while retaining v2 development and reserve."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.sam3_artifacts import SAM3_REVISION, artifact_path, read_json, sha256_file
from tools.validation.prepare_coco_acceptance import (SPLITS, load_precision_dataset, rank,
                                                      stratified_images, validate_precision_dataset)


def make_followup_dataset(annotations, root, previous_path, regression_hashes, evaluation=1024, seed=20261009):
    if type(seed) is not int or not 0 <= seed < 2**32 or type(evaluation) is not int or evaluation <= 0:
        raise ValueError("invalid follow-up seed or evaluation size")
    annotation_hash = sha256_file(annotations)
    previous = load_precision_dataset(previous_path, root)
    if previous["provenance"]["annotations_sha256"] != annotation_hash:
        raise ValueError("previous precision dataset uses different COCO annotations")
    source = read_json(annotations)
    images = {row["id"]: row for row in source["images"]}
    categories = {row["id"]: row["name"] for row in source["categories"]}
    if len(images) != len(source["images"]) or len(categories) != len(source["categories"]):
        raise ValueError("duplicate COCO image/category IDs")
    presence, size_counts = defaultdict(set), defaultdict(Counter)
    for annotation in source["annotations"]:
        image, category = annotation["image_id"], annotation["category_id"]
        if image not in images or category not in categories:
            raise ValueError("invalid COCO annotation identity")
        presence[image].add(category)
        if not annotation.get("iscrowd", 0):
            area = annotation["area"]
            for size, low, high in (("small", 0, 32**2), ("medium", 32**2, 96**2), ("large", 96**2, float("inf"))):
                if low <= area <= high:
                    size_counts[image][size] += 1

    retained = {split: [] for split in SPLITS}
    old_ids, seen_hashes = set(), set(regression_hashes)
    for sample in previous["samples"]:
        image = sample["coco_image_id"]
        if image not in images or not presence[image]:
            raise ValueError("retained image is missing its COCO annotations")
        if sample["positive_category_ids"] != sorted(presence[image]):
            raise ValueError("retained prompt categories differ from COCO annotations")
        prompts = [categories[category] for category in sample["positive_category_ids"] + sample["negative_category_ids"]]
        if sample["prompts"] != prompts:
            raise ValueError("retained prompt text differs from COCO categories")
        old_ids.add(image)
        seen_hashes.add(sample["source_sha256"])
        if sample["split"] != "evaluation":
            retained[sample["split"]].append(sample)
    if len(retained["reserve"]) < 1024 and evaluation >= 1024:
        raise ValueError("follow-up must keep the unopened reserve intact")

    candidates, content_hashes = [], {}
    for image in sorted(set(images) - old_ids, key=lambda identifier: rank(seed, "followup-deduplicate", identifier)):
        if not presence[image] or image in previous["previously_used_image_ids"]:
            continue
        relative = (Path("val2017") / images[image]["file_name"]).as_posix()
        digest = sha256_file(artifact_path(root, relative))
        if digest in seen_hashes or digest in previous["previously_used_content_hashes"]:
            continue
        seen_hashes.add(digest)
        content_hashes[image] = (relative, digest)
        candidates.append(image)
    selected = stratified_images(candidates, evaluation, presence, size_counts, categories,
                                 seed, "followup-evaluation")
    for image in sorted(selected, key=lambda identifier: rank(seed, "followup-order", identifier)):
        positive = sorted(presence[image])
        absent = sorted(set(categories) - presence[image],
                        key=lambda category: rank(seed, "followup-negative", f"{image}:{category}"))
        if not absent:
            raise ValueError("no absent category for follow-up negative prompt")
        relative, digest = content_hashes[image]
        retained["evaluation"].append({"id": f"coco-{image:012d}", "coco_image_id": image,
                                       "split": "evaluation", "source_group": f"coco-image-{image}",
                                       "source_sha256": digest, "image": relative,
                                       "positive_category_ids": positive, "negative_category_ids": absent[:1],
                                       "prompts": [categories[category] for category in positive + absent[:1]]})
    samples = [sample for split in SPLITS for sample in retained[split]]
    summaries = {}
    for split in SPLITS:
        rows = retained[split]
        prompted = Counter(category for row in rows for category in row["positive_category_ids"])
        groups = Counter(size for row in rows for size in size_counts[row["coco_image_id"]])
        instances = sum((size_counts[row["coco_image_id"]] for row in rows), Counter())
        summaries[split] = {"images": len(rows), "prompts": sum(len(row["prompts"]) for row in rows),
                            "positive_prompt_categories": len(prompted),
                            "positive_pairs_per_category": dict(sorted(prompted.items())),
                            "images_per_area": dict(groups), "noncrowd_instances_per_area": dict(instances)}
    consumed = [sample for sample in previous["samples"] if sample["split"] != "reserve"]
    used_ids = sorted(set(previous["previously_used_image_ids"]) | {row["coco_image_id"] for row in consumed})
    used_hashes = sorted(set(previous["previously_used_content_hashes"]) |
                         {row["source_sha256"] for row in consumed})
    value = {"schema_version": 2, "kind": "sam3-precision-dataset-v2", "sam3_revision": SAM3_REVISION,
             "seed": seed, "samples": samples, "summary": summaries,
             "previously_used_image_ids": used_ids, "previously_used_content_hashes": used_hashes,
             "provenance": {"annotations_sha256": annotation_hash,
                            "previous_manifest_sha256": {str(previous_path.resolve()): sha256_file(previous_path)},
                            "eligible_unique_images": len(content_hashes) + len(previous["samples"]),
                            "unused_before_split": len(candidates),
                            "sampling": "New evaluation: annotation-only category/area coverage then seeded SHA-256 order",
                            "prompt_policy": "Inherited development/reserve prompts; new evaluation has every annotated category plus one seeded absent category",
                            "scope": "Independent follow-up COCO evaluation; old evaluation excluded; reserve remains unopened"}}
    validate_precision_dataset(value)
    if {row["coco_image_id"] for row in retained["evaluation"]} & old_ids:
        raise ValueError("used v2 image leaked into follow-up evaluation")
    if sha256_file(previous_path) != value["provenance"]["previous_manifest_sha256"][str(previous_path.resolve())]:
        raise ValueError("previous manifest changed during follow-up selection")
    if sha256_file(annotations) != annotation_hash:
        raise ValueError("annotations changed during follow-up selection")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "input-root", "previous", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evaluation", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("follow-up dataset output already exists")
        regression = read_json(Path(__file__).resolve().parents[2] / "tests/data/sam3-image-cases.json")
        value = make_followup_dataset(args.annotations, args.input_root, args.previous,
                                      {case["source_sha256"] for case in regression["cases"]},
                                      args.evaluation, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print({split: {key: value["summary"][split][key] for key in ("images", "prompts")}
               for split in SPLITS})
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"COCO follow-up preparation failed: {error}\n")


if __name__ == "__main__":
    main()
