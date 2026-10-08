#!/usr/bin/env python3
"""Select eight annotation-defined development workloads before benchmarking."""

import argparse
from collections import Counter
from pathlib import Path
import sys

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.validation.precision_acceptance import GATES_SHA256, load_gates
from tools.maintenance.precision_artifacts import load_inputs
from tools.benchmark.precision_performance import validate_cases
from tools.convert.sam3_artifacts import artifact_path, read_json, sha256_file


def prepare(dataset_path, inputs, annotations, output):
    if output.exists():
        raise FileExistsError("performance case manifest already exists")
    dataset, samples, input_rows = load_inputs(inputs, dataset_path, "development")
    if sha256_file(annotations) != dataset["provenance"]["annotations_sha256"]:
        raise ValueError("performance annotations differ from the frozen dataset")
    coco = read_json(annotations)
    images = {row["id"]: row for row in coco["images"]}
    counts, small, large = Counter(), Counter(), Counter()
    for row in coco["annotations"]:
        if not row.get("iscrowd", 0):
            counts[row["image_id"]] += 1
            small[row["image_id"]] += row["area"] < 32**2
            large[row["image_id"]] += row["area"] > 96**2
    pixels = {row["id"]: images[row["coco_image_id"]]["width"] * images[row["coco_image_id"]]["height"] for row in samples}
    middle = sorted(pixels.values())[len(pixels) // 2]
    selectors = [
        ("source-small", lambda row: pixels[row["id"]]),
        ("source-middle", lambda row: abs(pixels[row["id"]] - middle)),
        ("source-large", lambda row: -pixels[row["id"]]),
        ("instances-few", lambda row: counts[row["coco_image_id"]]),
        ("instances-many", lambda row: -counts[row["coco_image_id"]]),
        ("objects-small", lambda row: -small[row["coco_image_id"]]),
        ("objects-large", lambda row: -large[row["coco_image_id"]]),
        ("negative", lambda row: -counts[row["coco_image_id"]]),
    ]
    selected, cases = set(), []
    artifacts = {str(path.resolve()): sha256_file(path) for path in (dataset_path, inputs / "manifest.json", annotations)}
    for coverage, key in selectors:
        candidates = [row for row in samples if row["id"] not in selected and counts[row["coco_image_id"]] > 0]
        if not candidates:
            raise ValueError("insufficient distinct development performance images")
        sample = min(candidates, key=lambda row: (key(row), row["id"]))
        selected.add(sample["id"])
        image = artifact_path(inputs, input_rows[sample["id"]]["file"])
        digest = input_rows[sample["id"]]["sha256"]
        artifacts[str(image.resolve())] = digest
        # First positive category and the frozen unannotated category; timings
        # never select the prompt producing the fastest or smallest output.
        prompt, alternate = sample["prompts"][0], sample["prompts"][-1]
        if coverage == "negative":
            prompt, alternate = alternate, prompt
        cases.append({"id": "perf-" + coverage, "sample_id": sample["id"], "coverage": coverage,
                      "image": str(image.resolve()), "image_sha256": digest, "prompt": prompt, "alternate": alternate,
                      "source_pixels": pixels[sample["id"]], "noncrowd_instances": counts[sample["coco_image_id"]],
                      "small_instances": small[sample["coco_image_id"]], "large_instances": large[sample["coco_image_id"]]})
    value = {"schema_version": 2, "kind": "sam3-precision-performance-cases-v2", "gates_sha256": GATES_SHA256,
             "dataset_sha256": sha256_file(dataset_path), "cases": cases, "artifact_sha256": artifacts,
             "selection": "Annotation-only, deterministic extrema/median; unique development images",
             "latency_scope": "Decoded RGB input through preprocessing, model execution and deployed postprocessing; model load is reported separately and disk decoding is excluded",
             "workloads": ["full_image", "changed_prompt", "repeated_result"]}
    validate_cases(value, dataset, sha256_file(dataset_path), load_gates())
    output.parent.mkdir(parents=True, exist_ok=True)
    import json
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Frozen {len(cases)} performance cases: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "inputs", "annotations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.dataset, args.inputs, args.annotations, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Performance preparation failed: {error}\n")


if __name__ == "__main__":
    main()
