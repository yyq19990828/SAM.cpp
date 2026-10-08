#!/usr/bin/env python3
"""Freeze disjoint SAM calibration/selection/evaluation subsets of a local COCO split."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import sys

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.quantize.calibration import validate_dataset
from tools.convert.sam3_artifacts import SAM3_REVISION, artifact_path, read_json, sha256_file


def make_dataset(annotations, root, image_directory, counts=(256, 128, 512), seed=20261007):
    if len(counts) != 3 or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("all three split sizes must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    source_hash = sha256_file(annotations)
    source = read_json(annotations)
    categories = {item["id"]: item["name"] for item in source["categories"]}
    if len(categories) != len(source["categories"]) or not categories:
        raise ValueError("COCO categories must have unique IDs")
    images = {item["id"]: item for item in source["images"]}
    if len(images) != len(source["images"]):
        raise ValueError("COCO images must have unique IDs")
    presence, sizes = defaultdict(set), defaultdict(set)
    for annotation in source["annotations"]:
        image_id, category = annotation["image_id"], annotation["category_id"]
        if image_id not in images or category not in categories:
            raise ValueError("COCO annotation refers to an unknown image or category")
        presence[image_id].add(category)
        area = annotation["area"]
        sizes[image_id].add("small" if area < 32**2 else "medium" if area < 96**2 else "large")
    regression = read_json(Path(__file__).resolve().parents[2] / "tests/data/sam3-image-cases.json")
    excluded = {case["source_sha256"] for case in regression["cases"]}

    def rank(identifier, purpose):
        return hashlib.sha256(f"{seed}:{purpose}:{identifier}".encode()).digest()

    # Stable hash sampling is independent of annotation file ordering. Keep one
    # representative of duplicate content and reject missing files up front.
    candidates, seen = [], set()
    for identifier in sorted(images, key=lambda item: rank(item, "image")):
        image = images[identifier]
        relative = (Path(image_directory) / image["file_name"]).as_posix()
        path = artifact_path(root, relative)
        digest = sha256_file(path)
        if digest in seen or digest in excluded or not presence[identifier]:
            continue
        seen.add(digest)
        candidates.append((identifier, relative, digest))
    if len(candidates) < sum(counts):
        raise ValueError("not enough unique annotated images for the requested splits")
    samples, summaries = [], {}
    offset = 0
    for split, count in zip(("calibration", "selection", "evaluation"), counts):
        class_counts, size_counts = Counter(), Counter()
        for identifier, relative, digest in candidates[offset:offset + count]:
            present = sorted(presence[identifier], key=lambda item: rank(f"{identifier}:{item}", "positive"))
            absent = sorted(set(categories) - presence[identifier], key=lambda item: rank(f"{identifier}:{item}", "negative"))
            # Up to three annotated concepts and one unannotated COCO class.
            # Annotation absence is not an oracle: quality reports must still
            # compare against the original model and original annotations.
            positive = present[:3]
            negative = absent[:1]
            prompts = [categories[item] for item in positive + negative]
            samples.append({"id": f"coco-{identifier:012d}", "split": split,
                            "source_group": f"coco-image-{identifier}", "source_sha256": digest,
                            "image": relative, "prompts": prompts,
                            "coco_image_id": identifier,
                            "positive_category_ids": positive, "negative_category_ids": negative})
            class_counts.update(presence[identifier])
            size_counts.update(sizes[identifier])
        summaries[split] = {"images": count, "prompts": sum(len(s["prompts"]) for s in samples if s["split"] == split),
                            "annotated_categories": len(class_counts),
                            "images_per_category": {str(key): value for key, value in sorted(class_counts.items())},
                            "images_per_size_group": dict(sorted(size_counts.items()))}
        offset += count
    if sha256_file(annotations) != source_hash:
        raise ValueError("COCO annotations changed while preparing the manifest")
    value = {"schema_version": 1, "kind": "sam3-calibration-dataset", "sam3_revision": SAM3_REVISION,
             "seed": seed, "samples": samples,
             "provenance": {"dataset": "COCO", "annotations_sha256": source_hash,
                            "annotations_name": annotations.name, "image_directory": image_directory,
                            "sampling": "seeded SHA-256 image order; sequential disjoint subsets; content deduplicated",
                            "scope": "custom subsets, not the complete official COCO evaluation split",
                            "eligible_unique_images": len(candidates)}, "summary": summaries}
    return validate_dataset(value, excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--image-directory", default="val2017")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration", type=int, default=256)
    parser.add_argument("--selection", type=int, default=128)
    parser.add_argument("--evaluation", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("dataset output already exists")
        value = make_dataset(args.annotations, args.input_root, args.image_directory,
                             (args.calibration, args.selection, args.evaluation), args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create prevents accidentally replacing a frozen data split.
        import json
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")
        print({split: {key: summary[key] for key in ("images", "prompts", "annotated_categories")}
               for split, summary in value["summary"].items()})
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"COCO preparation failed: {error}\n")


if __name__ == "__main__":
    main()
