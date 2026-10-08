#!/usr/bin/env python3
"""Freeze an image-disjoint COCO v2 development/evaluation/reserve manifest."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re

from tools.quantize.calibration import validate_dataset
from tools.convert.sam3_artifacts import SAM3_REVISION, artifact_path, read_json, sha256_file, write_json


SPLITS = ("calibration", "development", "evaluation", "reserve")


def rank(seed, purpose, identifier):
    return hashlib.sha256(f"{seed}:{purpose}:{identifier}".encode()).digest()


def stratified_images(candidates, count, presence, size_counts, categories, seed, purpose):
    """Greedy annotation-only coverage followed by seeded hash sampling."""
    if count > len(candidates) or count <= 0:
        raise ValueError("not enough unused unique images for requested split")
    available = Counter(category for image in candidates for category in presence[image])
    targets = {category: min(20, available[category]) for category in categories}
    observed, group_images, group_instances = Counter(), Counter(), Counter()
    remaining = sorted(candidates, key=lambda image: rank(seed, purpose, image))
    selected = []
    while len(selected) < count and remaining:
        def gain(image):
            category_gain = sum(1 / targets[c] for c in presence[image] if observed[c] < targets[c])
            area_gain = sum((group_images[size] < 50) / 50 + min(max(200 - group_instances[size], 0), n) / 200
                            for size, n in size_counts[image].items())
            return category_gain + area_gain
        gains = [gain(image) for image in remaining]
        best = max(range(len(remaining)), key=lambda i: gains[i])
        if gains[best] == 0:
            selected.extend(remaining[:count - len(selected)])
            break
        image = remaining.pop(best)
        selected.append(image)
        observed.update(presence[image])
        group_images.update(size_counts[image].keys())
        group_instances.update(size_counts[image])
    return selected


def validate_precision_dataset(value):
    if (value.get("schema_version") != 2 or value.get("kind") != "sam3-precision-dataset-v2"
            or value.get("sam3_revision") != SAM3_REVISION or type(value.get("seed")) is not int
            or not 0 <= value["seed"] < 2**32):
        raise ValueError("invalid precision dataset identity")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("empty precision dataset")
    seen = {key: set() for key in ("id", "coco_image_id", "source_sha256", "source_group", "image")}
    previous_ids = set(value.get("previously_used_image_ids", []))
    previous_hashes = set(value.get("previously_used_content_hashes", []))
    for sample in samples:
        if sample.get("split") not in SPLITS:
            raise ValueError("unknown precision dataset split")
        if not isinstance(sample.get("id"), str) or not re.fullmatch(r"coco-\d{12}", sample["id"]):
            raise ValueError("invalid COCO sample ID")
        if type(sample.get("coco_image_id")) is not int or sample["id"] != f"coco-{sample['coco_image_id']:012d}":
            raise ValueError("COCO sample ID disagrees with annotation identity")
        if not isinstance(sample.get("source_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", sample["source_sha256"]):
            raise ValueError("invalid image hash")
        for key, values in seen.items():
            field = sample.get(key)
            if not isinstance(field, (int, str)) or field in values:
                raise ValueError("duplicate image identity/content across precision splits")
            values.add(field)
        relative = sample["image"]
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("image path is not bounded")
        positive, negative, prompts = (sample.get(key) for key in ("positive_category_ids", "negative_category_ids", "prompts"))
        if (not isinstance(positive, list) or not positive or positive != sorted(set(positive))
                or any(type(category) is not int for category in positive)
                or not isinstance(negative, list) or len(negative) != 1 or type(negative[0]) is not int
                or set(positive) & set(negative) or not isinstance(prompts, list)
                or len(prompts) != len(positive) + 1 or len(set(prompts)) != len(prompts)
                or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)):
            raise ValueError("invalid complete prompted category inventory")
        if sample["split"] in ("evaluation", "reserve") and (sample["coco_image_id"] in previous_ids or sample["source_sha256"] in previous_hashes):
            raise ValueError("previously used image leaked into a new held-out split")
    if set(sample["split"] for sample in samples) != set(SPLITS):
        raise ValueError("precision dataset needs all four distinct roles")
    return value


def load_precision_dataset(path, root=None):
    value = validate_precision_dataset(read_json(path))
    if root is not None:
        for sample in value["samples"]:
            if sha256_file(artifact_path(root, sample["image"])) != sample["source_sha256"]:
                raise ValueError(f"{sample['id']}: changed image content")
    return value


def make_dataset(annotations, root, previous_manifests, regression_hashes, evaluation=1024, reserve=1024,
                 seed=20261008, additional_used_ids=(), additional_used_hashes=()):
    if type(seed) is not int or not 0 <= seed < 2**32 or not previous_manifests:
        raise ValueError("a valid seed and previous usage manifests are required")
    annotation_hash = sha256_file(annotations)
    source = read_json(annotations)
    images = {row["id"]: row for row in source["images"]}
    categories = {row["id"]: row["name"] for row in source["categories"]}
    if len(images) != len(source["images"]) or len(categories) != len(source["categories"]):
        raise ValueError("duplicate COCO image/category IDs")
    presence, size_counts = defaultdict(set), defaultdict(Counter)
    for annotation in source["annotations"]:
        image, category = annotation["image_id"], annotation["category_id"]
        if image not in images or category not in categories:
            raise ValueError("COCO annotation identity is invalid")
        presence[image].add(category)
        if not annotation.get("iscrowd", 0):
            area = annotation["area"]
            for size, low, high in (("small", 0, 32**2), ("medium", 32**2, 96**2), ("large", 96**2, 1e10)):
                if low <= area <= high:
                    size_counts[image][size] += 1
    previous, used_ids, used_hashes, identities = {}, set(additional_used_ids), set(additional_used_hashes) | set(regression_hashes), {}
    for path in previous_manifests:
        value = validate_dataset(read_json(path))
        if value["provenance"]["annotations_sha256"] != annotation_hash:
            raise ValueError("previous manifest annotations differ from this COCO split")
        identities[str(path.resolve())] = sha256_file(path)
        for sample in value["samples"]:
            image = sample["coco_image_id"]
            if image in previous and previous[image]["source_sha256"] != sample["source_sha256"]:
                raise ValueError("previous image has conflicting content identity")
            previous[image] = sample
            used_ids.add(image)
            used_hashes.add(sample["source_sha256"])
    details, seen_hashes = {}, set()
    for image in sorted(images, key=lambda identifier: (identifier not in previous, rank(seed, "deduplicate", identifier))):
        relative = (Path("val2017") / images[image]["file_name"]).as_posix()
        digest = sha256_file(artifact_path(root, relative))
        if image in previous and digest != previous[image]["source_sha256"]:
            raise ValueError("previous input image changed")
        if not presence[image] or digest in regression_hashes or digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        details[image] = (relative, digest)
    unused = [image for image, (_, digest) in details.items() if image not in used_ids and digest not in used_hashes]
    if len(unused) < evaluation + reserve:
        raise ValueError("insufficient unused images after the complete content/usage exclusion")
    selected = stratified_images(unused, evaluation, presence, size_counts, categories, seed, "evaluation")
    rest = set(unused) - set(selected)
    held = stratified_images(rest, reserve, presence, size_counts, categories, seed, "reserve")
    roles = {image: "calibration" if sample["split"] == "calibration" else "development" for image, sample in previous.items()}
    roles.update({image: "evaluation" for image in selected})
    roles.update({image: "reserve" for image in held})
    samples = []
    for split in SPLITS:
        for image in sorted((image for image, role in roles.items() if role == split), key=lambda i: rank(seed, split, i)):
            if image not in details:
                raise ValueError("previous used image was excluded or content-duplicated; audit the usage ledger")
            positive = sorted(presence[image])
            absent = sorted(set(categories) - presence[image], key=lambda c: rank(seed, "negative", f"{image}:{c}"))
            if not absent:
                raise ValueError("no absent class is available for the fixed negative prompt")
            relative, digest = details[image]
            samples.append({"id": f"coco-{image:012d}", "coco_image_id": image, "split": split,
                            "source_group": f"coco-image-{image}", "source_sha256": digest, "image": relative,
                            "positive_category_ids": positive, "negative_category_ids": absent[:1],
                            "prompts": [categories[c] for c in positive + absent[:1]]})
    summaries = {}
    for split in SPLITS:
        rows = [row for row in samples if row["split"] == split]
        prompted = Counter(category for row in rows for category in row["positive_category_ids"])
        groups = Counter(size for row in rows for size in size_counts[row["coco_image_id"]])
        instances = sum((size_counts[row["coco_image_id"]] for row in rows), Counter())
        summaries[split] = {"images": len(rows), "prompts": sum(len(row["prompts"]) for row in rows),
                            "positive_prompt_categories": len(prompted), "positive_pairs_per_category": dict(sorted(prompted.items())),
                            "images_per_area": dict(groups), "noncrowd_instances_per_area": dict(instances)}
    value = {"schema_version": 2, "kind": "sam3-precision-dataset-v2", "sam3_revision": SAM3_REVISION,
             "seed": seed, "samples": samples, "summary": summaries,
             "previously_used_image_ids": sorted(used_ids), "previously_used_content_hashes": sorted(used_hashes),
             "provenance": {"annotations_sha256": annotation_hash, "previous_manifest_sha256": identities,
                            "eligible_unique_images": len(details), "unused_before_split": len(unused),
                            "sampling": "Annotation-only category/area coverage, then seeded SHA-256 order; deduplicate by content",
                            "prompt_policy": "Every annotated category plus one seeded unannotated category",
                            "scope": "Custom prompted COCO split; not full val2017 AP; reserve may not be inferred on"}}
    for path, expected in identities.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("usage manifest changed during split creation")
    if sha256_file(annotations) != annotation_hash:
        raise ValueError("COCO annotations changed during split creation")
    return validate_precision_dataset(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "input-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous", type=Path, action="append", required=True)
    parser.add_argument("--extra-used-ids", type=Path, help="JSON list of additionally consumed COCO image IDs")
    parser.add_argument("--evaluation", type=int, default=1024)
    parser.add_argument("--reserve", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("precision dataset output already exists")
        regression = read_json(Path(__file__).resolve().parents[2] / "tests/data/sam3-image-cases.json")
        value = make_dataset(args.annotations, args.input_root, args.previous,
                             {case["source_sha256"] for case in regression["cases"]}, args.evaluation, args.reserve,
                             args.seed, read_json(args.extra_used_ids) if args.extra_used_ids else ())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json
        with args.output.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print({split: {key: summary[key] for key in ("images", "prompts", "positive_prompt_categories")}
               for split, summary in value["summary"].items()})
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"COCO v2 preparation failed: {error}\n")


if __name__ == "__main__":
    main()
