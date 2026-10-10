#!/usr/bin/env python3
"""Select and materialize an unused COCO train2017 precision holdout."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from urllib.request import urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert.sam3_artifacts import SAM3_REVISION, read_json, sha256_file, write_json
from tools.validation.prepare_coco_acceptance import (SPLITS, load_precision_dataset, rank,
                                                      stratified_images, validate_precision_dataset)


def annotation_index(source):
    images = {row["id"]: row for row in source["images"]}
    categories = {row["id"]: row["name"] for row in source["categories"]}
    if len(images) != len(source["images"]) or len(categories) != len(source["categories"]):
        raise ValueError("duplicate COCO image or category ID")
    presence, sizes = defaultdict(set), defaultdict(Counter)
    for row in source["annotations"]:
        image, category = row["image_id"], row["category_id"]
        if image not in images or category not in categories:
            raise ValueError("invalid COCO annotation reference")
        presence[image].add(category)
        if not row.get("iscrowd", 0):
            area = row["area"]
            for name, low, high in (("small", 0, 32**2), ("medium", 32**2, 96**2),
                                    ("large", 96**2, float("inf"))):
                if low <= area <= high:
                    sizes[image][name] += 1
    return images, categories, presence, sizes


def excluded_dataset(previous_path, val_path, train_path, excluded_path):
    if excluded_path is None:
        return None
    excluded = load_precision_dataset(excluded_path)
    provenance = excluded["provenance"]
    if (provenance.get("val_annotations_sha256") != sha256_file(val_path)
            or provenance.get("train_annotations_sha256") != sha256_file(train_path)
            or sha256_file(previous_path) not in provenance.get("previous_manifest_sha256", {}).values()):
        raise ValueError("excluded dataset is not bound to the prior holdout sources")
    return excluded


def select(previous_path, val_path, train_path, count, seed, excluded_path=None):
    previous = load_precision_dataset(previous_path)
    if previous["provenance"]["annotations_sha256"] != sha256_file(val_path):
        raise ValueError("previous dataset is not bound to these val2017 annotations")
    excluded = excluded_dataset(previous_path, val_path, train_path, excluded_path)
    val, train = read_json(val_path), read_json(train_path)
    val_images, _, _, _ = annotation_index(val)
    images, categories, presence, sizes = annotation_index(train)
    if val["categories"] != train["categories"] or set(val_images) & set(images):
        raise ValueError("train/val COCO categories or image identities disagree")
    prior_ids = set(previous["previously_used_image_ids"]) | {row["coco_image_id"] for row in previous["samples"]}
    if excluded is not None:
        prior_ids.update(excluded["previously_used_image_ids"])
        prior_ids.update(row["coco_image_id"] for row in excluded["samples"])
    eligible = [image for image in images if presence[image] and image not in prior_ids]
    if len(eligible) < count:
        raise ValueError("insufficient fresh annotated train2017 images")
    ordered = sorted(eligible, key=lambda image: rank(seed, "fresh-train-pool", image))
    pool = set(ordered[:max(8192, count * 4)])
    for category in categories:
        positives = [image for image in ordered if category in presence[image]][:64]
        pool.update(positives)
    chosen = stratified_images(list(pool), count, presence, sizes, categories, seed, "fresh-train-holdout")
    chosen.sort(key=lambda image: rank(seed, "fresh-train-order", image))
    return {"schema_version": 1, "kind": "sam3-fresh-coco-train2017-selection",
            "seed": seed, "count": count, "sam3_revision": SAM3_REVISION,
            "previous_sha256": sha256_file(previous_path), "val_annotations_sha256": sha256_file(val_path),
            "train_annotations_sha256": sha256_file(train_path),
            "excluded_dataset_sha256": sha256_file(excluded_path) if excluded_path is not None else None,
            "images": [{"id": image, "file_name": images[image]["file_name"],
                        "positive_category_ids": sorted(presence[image])} for image in chosen]}


def download(selection_path, root, workers):
    selection = read_json(selection_path)
    if selection.get("kind") != "sam3-fresh-coco-train2017-selection":
        raise ValueError("unsupported holdout selection")
    target = root / "train2017"
    target.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        name = row["file_name"]
        if name != f"{row['id']:012d}.jpg":
            raise ValueError("unexpected train2017 file name")
        path = target / name
        if not path.exists():
            url = "https://s3.amazonaws.com/images.cocodataset.org/train2017/" + name
            temporary = target / (name + ".part")
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=40) as response, temporary.open("wb") as output:
                        if response.status != 200:
                            raise OSError(f"HTTP {response.status}: {url}")
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                    if temporary.stat().st_size < 100 or temporary.open("rb").read(2) != b"\xff\xd8":
                        raise ValueError(f"invalid JPEG download: {name}")
                    temporary.replace(path)
                    break
                except (OSError, ValueError):
                    temporary.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
        return {"id": row["id"], "file": "train2017/" + name, "sha256": sha256_file(path)}

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, row): row["id"] for row in selection["images"]}
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 100 == 0:
                print(f"downloaded {len(rows)}/{len(futures)}", flush=True)
    order = {row["id"]: index for index, row in enumerate(selection["images"])}
    rows.sort(key=lambda row: order[row["id"]])
    receipt = root / "downloads.json"
    if receipt.exists():
        raise FileExistsError("holdout download receipt already exists")
    write_json(receipt, {"complete": True, "selection_sha256": sha256_file(selection_path), "images": rows})


def finalize(selection_path, previous_path, val_path, train_path, root, annotations_path, dataset_path,
             excluded_path=None):
    if annotations_path.exists() or dataset_path.exists():
        raise FileExistsError("fresh holdout outputs must not overwrite prior evidence")
    selection = read_json(selection_path)
    if (selection.get("kind") != "sam3-fresh-coco-train2017-selection"
            or selection["previous_sha256"] != sha256_file(previous_path)
            or selection["val_annotations_sha256"] != sha256_file(val_path)
            or selection["train_annotations_sha256"] != sha256_file(train_path)
            or selection.get("excluded_dataset_sha256") != (sha256_file(excluded_path) if excluded_path is not None else None)):
        raise ValueError("holdout selection source identity changed")
    excluded = excluded_dataset(previous_path, val_path, train_path, excluded_path)
    receipt = read_json(root / "downloads.json")
    if (receipt.get("complete") is not True or receipt.get("selection_sha256") != sha256_file(selection_path)
            or [row["id"] for row in receipt["images"]] != [row["id"] for row in selection["images"]]):
        raise ValueError("holdout downloads differ from the selection")
    previous = load_precision_dataset(previous_path, root)
    prior_hashes = set(previous["previously_used_content_hashes"]) | {row["source_sha256"] for row in previous["samples"]}
    if excluded is not None:
        prior_hashes.update(excluded["previously_used_content_hashes"])
        prior_hashes.update(row["source_sha256"] for row in excluded["samples"])
    downloaded = {row["id"]: row for row in receipt["images"]}
    for row in receipt["images"]:
        if row["sha256"] in prior_hashes or sha256_file(root / row["file"]) != row["sha256"]:
            raise ValueError("fresh train image duplicates or differs from retained evidence")
    if len({row["sha256"] for row in receipt["images"]}) != len(receipt["images"]):
        raise ValueError("duplicate train2017 image content")

    val, train = read_json(val_path), read_json(train_path)
    train_images, categories, train_presence, _ = annotation_index(train)
    selected_ids = {row["id"] for row in selection["images"]}
    if val["categories"] != train["categories"] or {row["id"] for row in val["images"]} & selected_ids:
        raise ValueError("merged COCO identity mismatch")
    train_annotations = [row for row in train["annotations"] if row["image_id"] in selected_ids]
    if {row["id"] for row in val["annotations"]} & {row["id"] for row in train_annotations}:
        raise ValueError("COCO train/val annotation ID collision")
    merged = {**val, "images": val["images"] + [train_images[row["id"]] for row in selection["images"]],
              "annotations": val["annotations"] + train_annotations,
              "precision_provenance": {"val_annotations_sha256": sha256_file(val_path),
                                       "train_annotations_sha256": sha256_file(train_path),
                                       "selection_sha256": sha256_file(selection_path)}}
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(annotations_path, merged)

    retained = {split: [row for row in previous["samples"] if row["split"] == split]
                for split in SPLITS if split != "evaluation"}
    evaluation = []
    for row in selection["images"]:
        image = row["id"]
        positive = sorted(train_presence[image])
        if positive != row["positive_category_ids"]:
            raise ValueError("train2017 annotation presence differs from frozen selection")
        absent = sorted(set(categories) - set(positive),
                        key=lambda category: rank(selection["seed"], "fresh-train-negative", f"{image}:{category}"))
        evaluation.append({"id": f"coco-{image:012d}", "coco_image_id": image,
                           "split": "evaluation", "source_group": f"coco-image-{image}",
                           "source_sha256": downloaded[image]["sha256"], "image": downloaded[image]["file"],
                           "positive_category_ids": positive, "negative_category_ids": absent[:1],
                           "prompts": [categories[category] for category in positive + absent[:1]]})
    retained["evaluation"] = evaluation
    samples = [row for split in SPLITS for row in retained[split]]
    _, _, presence, sizes = annotation_index(merged)
    summary = {}
    for split in SPLITS:
        rows = retained[split]
        prompted = Counter(category for row in rows for category in row["positive_category_ids"])
        area = Counter(size for row in rows for size in sizes[row["coco_image_id"]])
        instances = sum((sizes[row["coco_image_id"]] for row in rows), Counter())
        summary[split] = {"images": len(rows), "prompts": sum(len(row["prompts"]) for row in rows),
                          "positive_prompt_categories": len(prompted),
                          "positive_pairs_per_category": dict(sorted(prompted.items())),
                          "images_per_area": dict(area), "noncrowd_instances_per_area": dict(instances)}
    consumed = [row for row in previous["samples"] if row["split"] != "reserve"]
    excluded_consumed = ([row for row in excluded["samples"] if row["split"] != "reserve"]
                         if excluded is not None else [])
    dataset = {"schema_version": 2, "kind": "sam3-precision-dataset-v2", "sam3_revision": SAM3_REVISION,
               "seed": selection["seed"], "samples": samples, "summary": summary,
               "previously_used_image_ids": sorted(set(previous["previously_used_image_ids"]) |
                                                   (set(excluded["previously_used_image_ids"]) if excluded else set()) |
                                                   {row["coco_image_id"] for row in consumed + excluded_consumed}),
               "previously_used_content_hashes": sorted(set(previous["previously_used_content_hashes"]) |
                                                         (set(excluded["previously_used_content_hashes"]) if excluded else set()) |
                                                         {row["source_sha256"] for row in consumed + excluded_consumed}),
               "provenance": {"annotations_sha256": sha256_file(annotations_path),
                              "previous_manifest_sha256": {str(previous_path.resolve()): sha256_file(previous_path),
                                                           **({str(excluded_path.resolve()): sha256_file(excluded_path)}
                                                              if excluded_path is not None else {})},
                              "val_annotations_sha256": sha256_file(val_path),
                              "train_annotations_sha256": sha256_file(train_path),
                              "selection_sha256": sha256_file(selection_path),
                              "downloads_sha256": sha256_file(root / "downloads.json"),
                              "sampling": "Unused train2017 annotation-only category/area coverage with seeded order",
                              "prompt_policy": "Retained development/reserve prompts; train holdout every annotated category plus one seeded absent category",
                              "scope": "Fresh paired-precision train2017 holdout; original val2017 reserve unopened"}}
    validate_precision_dataset(dataset)
    if len(evaluation) != selection["count"] or len(retained["reserve"]) < 1024:
        raise ValueError("fresh evaluation or preserved reserve is incomplete")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(dataset_path, dataset)
    print({split: {key: summary[split][key] for key in ("images", "prompts")} for split in SPLITS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "finalize"):
        child = sub.add_parser(name)
        for field in ("previous", "val-annotations", "train-annotations"):
            child.add_argument("--" + field, type=Path, required=True)
        child.add_argument("--selection", type=Path, required=True)
        child.add_argument("--exclude-dataset", type=Path)
    choose = sub.choices["select"]
    choose.add_argument("--evaluation", type=int, default=1024)
    choose.add_argument("--seed", type=int, default=20261009)
    finish = sub.choices["finalize"]
    for field in ("input-root", "output-annotations", "output-dataset"):
        finish.add_argument("--" + field, type=Path, required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--selection", type=Path, required=True)
    fetch.add_argument("--input-root", type=Path, required=True)
    fetch.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    try:
        if args.command == "select":
            if args.selection.exists() or args.evaluation < 1024:
                raise ValueError("new holdout selection must be fresh and at least 1024 images")
            write_json(args.selection, select(args.previous, args.val_annotations, args.train_annotations,
                                              args.evaluation, args.seed, args.exclude_dataset))
        elif args.command == "download":
            if not 1 <= args.workers <= 32:
                raise ValueError("download workers must be within 1..32")
            download(args.selection, args.input_root, args.workers)
        else:
            finalize(args.selection, args.previous, args.val_annotations, args.train_annotations,
                     args.input_root, args.output_annotations, args.output_dataset, args.exclude_dataset)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"fresh COCO holdout preparation failed: {error}\n")


if __name__ == "__main__":
    main()
