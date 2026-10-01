#!/usr/bin/env python3
"""Generate the defined M2 PNG recipes without running a model.

Input media stays external. Generated manifests explicitly require later
verification that the official oracle exercises every requested scenario.
"""

import argparse
import importlib.metadata
from pathlib import Path

from PIL import Image

from sam3_artifacts import read_json, sha256_file, write_json


CASES = Path(__file__).resolve().parents[1] / "tests/data/sam3-video-cases.json"


def recipe_frame(case_id, source, frame):
    if case_id == "negative":
        return source.copy()
    if case_id == "entry":
        image = Image.new("RGB", (1800, 1200), (127, 127, 127))
        truck = source.resize((900, 600), Image.Resampling.BICUBIC)
        image.paste(truck, (0, 300))
        image.paste(truck, (1800 - min(max(frame - 15, 0) * 60, 900), 300))
        return image
    if case_id == "motion":
        image = Image.new("RGB", source.size, (127, 127, 127))
        image.paste(source, (4 * min(frame, 48 - frame) - 48, 0))
        return image
    if case_id == "occlusion":
        image = source.copy()
        if 20 <= frame <= 23:
            image.paste((127, 127, 127), (64, 256, 1744, 920))
        return image
    if case_id == "hotstart-removal":
        return source.copy() if frame < 2 else Image.new("RGB", source.size, (127, 127, 127))
    raise ValueError(f"undefined video frame recipe: {case_id}")


def generate(input_root, output):
    frozen = read_json(CASES)
    if importlib.metadata.version("pillow") != frozen["pillow_version"]:
        raise ValueError("video recipes require pinned Pillow 11.2.1")
    input_root = input_root.resolve()
    for case in frozen["cases"]:
        if sha256_file(input_root / case["source"]) != case["source_sha256"]:
            raise ValueError(f"source image hash differs for {case['id']}")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "sam3_revision": frozen["sam3_revision"],
                "task": "text_video", "complete": False, "eligible_for_milestone": False,
                "reference_behavior_verified": False, "cases_sha256": sha256_file(CASES),
                "generator_sha256": sha256_file(__file__), "pillow_version": frozen["pillow_version"],
                "cases": []}
    write_json(output / "manifest.json", manifest)
    for case in frozen["cases"]:
        directory = output / case["id"]
        directory.mkdir()
        frames = []
        with Image.open(input_root / case["source"]) as original:
            source = original.convert("RGB")
            for frame in range(case["frames"]):
                image = recipe_frame(case["id"], source, frame)
                path = directory / f"{frame:06d}.png"
                image.save(path)
                frames.append({"index": frame, "file": str(path.relative_to(output)),
                               "sha256": sha256_file(path), "width": image.width, "height": image.height})
        manifest["cases"].append({**case, "frames_manifest": frames})
    manifest["complete"] = True
    write_json(output / "manifest.json", manifest)
    print(f"Generated {sum(case['frames'] for case in frozen['cases'])} PNG frames; oracle behavior remains unverified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        generate(args.input_root, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"video case generation failed: {error}\n")


if __name__ == "__main__":
    main()
