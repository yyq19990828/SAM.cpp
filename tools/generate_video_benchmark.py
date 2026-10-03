#!/usr/bin/env python3
"""Generate the pinned 64-frame one/four-object workloads; oracle qualification is separate."""

import argparse
import importlib.metadata
from pathlib import Path

from sam3_artifacts import sha256_file, write_json

SOURCE_SHA256 = "941715e721c8864324a1425b445ea4dde0498b995c45ddce0141a58971c6ff99"
WORKLOADS = (("one-object", [(500, 330, False, 0)]),
             ("four-object", [(40, 30, False, 0), (960, 30, True, 8),
                              (40, 630, True, 16), (960, 630, False, 24)]))


def generate(source, output):
    from PIL import Image, ImageOps

    source, output = Path(source).resolve(), Path(output).resolve()
    if sha256_file(source) != SOURCE_SHA256 or importlib.metadata.version("Pillow") != "11.2.1":
        raise ValueError("benchmark requires the pinned truck.jpg and Pillow 11.2.1")
    output.mkdir(parents=True, exist_ok=False)
    with Image.open(source) as image:
        original = image.convert("RGB").resize((800, 533), Image.Resampling.BICUBIC)
    mirrored = ImageOps.mirror(original)
    manifest = {"schema_version": 1, "kind": "independent-m2-performance-fixtures", "complete": False,
                "eligible_for_milestone": False, "original_behavior_verified": False,
                "source": str(source), "source_sha256": SOURCE_SHA256, "pillow_version": "11.2.1",
                "generator_sha256": sha256_file(__file__), "width": 1800, "height": 1200,
                "frames": 64, "warmup_frames": 16, "measured_frames": 48, "prompt": "truck", "max_objects": 8,
                "recipe": {"canvas_rgb": [127, 127, 127], "tile_size": [800, 533],
                           "motion": "dx=2*(16-abs(((frame+phase)%32)-16)-8); dy=8-abs(((frame+phase)%16)-8)",
                           "purpose": "Separate persistent objects; the frozen numerical corpus is unchanged."},
                "workloads": []}
    write_json(output / "fixture-manifest.json", manifest)
    for identifier, positions in WORKLOADS:
        directory = output / identifier
        directory.mkdir()
        hashes = []
        for frame in range(64):
            image = Image.new("RGB", (1800, 1200), (127, 127, 127))
            for x, y, flip, phase in positions:
                dx = 2 * (16 - abs(((frame + phase) % 32) - 16) - 8)
                dy = 8 - abs(((frame + phase) % 16) - 8)
                image.paste(mirrored if flip else original, (x + dx, y + dy))
            target = directory / f"{frame:06d}.png"
            image.save(target, compress_level=1)
            hashes.append(sha256_file(target))
        manifest["workloads"].append({"id": identifier, "expected_objects": len(positions), "directory": identifier,
                                      "positions_xy_mirror_phase": positions, "input_sha256": hashes})
        write_json(output / "fixture-manifest.json", manifest)
    manifest["complete"] = True
    write_json(output / "fixture-manifest.json", manifest)
    return output / "fixture-manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(generate(args.image, args.output))
    except (OSError, ValueError) as error:
        parser.exit(1, f"fixture generation failed: {error}\n")


if __name__ == "__main__":
    main()
