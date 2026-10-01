#!/usr/bin/env python3
"""Copy pinned Meta sources and apply the five recorded CPU-image import/cache fixes.

The original checkout stays untouched. This does not load, download, or alter weights.
The exporter separately records its unfused FP32 MLP adaptation.
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from export_reference import validate_source
from sam3_artifacts import SAM3_REVISION, sha256_file, write_json


PATCHES = [
    {
        "file": "sam3/model_builder.py",
        "source_sha256": "d71d6d3e485ec3eae48bbc2ba676f401b5853d65c4195a91d077b04da38121c2",
        "adapted_sha256": "175ac9205362652585984dac1fcdcd16ff63519ca1049dd78c24bf85013a8e1a",
        "removed_imports": [
            "from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor\n",
            "from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor\n",
            "from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity\n",
            "from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU\n",
            "from sam3.model.video_tracking_multiplex import VideoTrackingDynamicMultiplex\n",
        ],
        "reason": "Remove eager imports of unused interactive/video code requiring Triton; defer type annotations. Image inference statements unchanged.",
    },
    {
        "file": "sam3/model/sam3_image.py",
        "source_sha256": "2c20de90fb0b76b01dd437385eb7226f8c3cd4757a4c80cc8dec943dcf46a4a0",
        "adapted_sha256": "798818a657bce10d317548630078fc600c41cc33ee4f22bb00ab1e922e276844",
        "removed_imports": ["from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor\n"],
        "reason": "Remove eager imports of unused interactive/video code requiring Triton; defer type annotations. Image inference statements unchanged.",
    },
    {
        "file": "sam3/model/position_encoding.py",
        "source_sha256": "6ec4e053bad5d17035f9ea8f60a032f60c51ba5ce5a5a174b4b2fe0de62749f6",
        "adapted_sha256": "c9bed3b5d30a5fa3a0a0b5af7926dee9bf69d242be230e9e8c1c39b634b3bb24",
        "replace": ['tensors = torch.zeros((1, 1) + size, device="cuda")', 'tensors = torch.zeros((1, 1) + size, device="cpu")'],
        "reason": "CPU-only positional encoding cache precomputation; values use identical FP32 formulas.",
    },
    {
        "file": "sam3/model/decoder.py",
        "source_sha256": "3a2bfb30f0f4b5405a0c9a27dc21aed71aa6691032a56fa8d96b3f9af3a15b96",
        "adapted_sha256": "b77dd0fba5ad14b6b5807906519110e5f21b4c5422ff7152a330a0d6841f8066",
        "replace": ['feat_size, feat_size, device="cuda"', 'feat_size, feat_size, device="cpu"'],
        "reason": "CPU-only box relative-position coordinate cache precomputation; values use identical FP32 formulas.",
    },
    {
        "file": "sam3/model/geometry_encoders.py",
        "source_sha256": "02c50b9ba5caf1c51c472a78a98b5b943cba6b480a1157485eb74c59c08d7680",
        "adapted_sha256": "59d4bf97542b03429d678d7aa6bfa3f9edf725a4091da5540cf33925cb29186a",
        "replace": ["scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)",
                    "scale = scale.to(device=boxes_xyxy.device, non_blocking=True)"],
        "reason": "Remove GPU pinned-memory hint from CPU box scale transfer; identical FP32 values and tensor device.",
    },
]


def patched_text(text, patch):
    if hashlib.sha256(text.encode()).hexdigest() != patch["source_sha256"]:
        raise ValueError(f"pinned source hash differs: {patch['file']}")
    if "removed_imports" in patch:
        for statement in patch["removed_imports"]:
            if text.count(statement) != 1:
                raise ValueError(f"expected exactly one import statement in {patch['file']}")
            text = text.replace(statement, "")
        text = "from __future__ import annotations\n\n" + text
    else:
        original, replacement = patch["replace"]
        if text.count(original) != 1:
            raise ValueError(f"expected exactly one device cache statement in {patch['file']}")
        text = text.replace(original, replacement)
    if hashlib.sha256(text.encode()).hexdigest() != patch["adapted_sha256"]:
        raise ValueError(f"CPU adaptation hash differs: {patch['file']}")
    return text


def prepare(source, output):
    source, _, _ = validate_source(source)
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("reference source and runtime output must not overlap")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime source: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sam3-cpu-source-", dir=output.parent))
    try:
        shutil.copytree(source / "sam3", staging / "sam3", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(source / "LICENSE", staging / "LICENSE")
        for patch in PATCHES:
            path = staging / patch["file"]
            path.write_text(patched_text(path.read_text(encoding="utf-8"), patch), encoding="utf-8")
        write_json(staging / "adaptations.json", {"source": str(source), "source_revision": SAM3_REVISION,
                                                 "adapted_source": str(output), "preparer_sha256": sha256_file(__file__),
                                                 "changes": PATCHES})
        validate_source(source, staging)
        os.rename(staging, output)
        print(f"Prepared pinned CPU-image reference source: {output}")
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        prepare(args.source, args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"reference source preparation failed: {error}\n")


if __name__ == "__main__":
    main()
