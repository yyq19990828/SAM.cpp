#!/usr/bin/env python3
"""Normalize one allowed v2 image phase once for every inference backend."""

import argparse
from pathlib import Path

from precision_artifacts import phase_samples
from prepare_coco_acceptance import load_precision_dataset
from sam3_artifacts import artifact_path, sha256_file, write_json


def prepare(dataset_path, input_root, output, phase):
    from PIL import Image, __version__ as pillow_version
    if output.exists():
        raise FileExistsError("normalized precision inputs must be a new directory")
    dataset = load_precision_dataset(dataset_path, input_root)
    samples = phase_samples(dataset, phase)
    identities = {str(dataset_path.resolve()): sha256_file(dataset_path), str(Path(__file__).resolve()): sha256_file(Path(__file__))}
    output.mkdir(parents=True)
    rows = []
    for sample in samples:
        source = artifact_path(input_root, sample["image"])
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        path = output / (sample["id"] + ".png")
        image.save(path)
        if sha256_file(source) != sample["source_sha256"]:
            raise ValueError("source image changed during normalization")
        rows.append({"sample_id": sample["id"], "source_sha256": sample["source_sha256"],
                     "file": path.name, "sha256": sha256_file(path), "width": image.width, "height": image.height})
    from sam3_artifacts import verify_run_artifacts
    verify_run_artifacts(identities)
    write_json(output / "manifest.json", {"schema_version": 2, "kind": "sam3-precision-inputs-v2", "complete": True,
                                          "phase": phase, "dataset_sha256": identities[str(dataset_path.resolve())],
                                          "pillow": pillow_version, "images": rows, "artifact_sha256": identities})
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "input-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    args = parser.parse_args()
    try:
        print(f"Normalized {prepare(args.dataset, args.input_root, args.output, args.phase)} RGB images")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Precision input preparation failed: {error}\n")


if __name__ == "__main__":
    main()
