#!/usr/bin/env python3
"""Copy pinned Meta sources and apply recorded FP32 image/video adaptations.

The original checkout stays untouched. This does not load, download, or alter weights.
The exporter separately records its unfused FP32 MLP adaptation.
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from tools.validation.export_reference import validate_source
from tools.convert.sam3_artifacts import SAM3_REVISION, sha256_file, write_json


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

VIDEO_SOURCE_HASHES = {
    "sam3/model/sam3_tracker_utils.py": "dc5fdeba2d4416f273394a9bd4450dff050608db6009a4546ca68adbaa24a640",
    "sam3/model/sam3_tracker_base.py": "b2b52409c002e1590262375aa794f8ab67e7476f42f8fee41a76de0c14aa62e2",
    "sam3/model/sam3_tracking_predictor.py": "145df76a5c045605d4d15360456015adb4f987c4b18e82f4f4876a63e6a32345",
    "sam3/model/sam3_video_base.py": "7ffae0a8c15f17814ce438078b6a00cc052be21ad16c8a59205df859c3df52c4",
    "sam3/model/sam3_video_inference.py": "a502cd76a845292aaf022f64667c176bad919a1bd0cfd9b542d2aa58a7d561fe",
    "sam3/perflib/connected_components.py": "f09bb5b905a12e0aeb6e7f1096af839015230f007dc47c43a1de7893a355dbd1",
}


def image_patches(device="cpu"):
    if device not in ("cpu", "cuda"):
        raise ValueError("unsupported reference device")
    if device == "cpu":
        return PATCHES
    # These caches are plain tensors rather than registered buffers, so moving
    # the model cannot move caches precomputed on CPU. Keep the pinned CUDA
    # precomputation for both image and video CUDA references.
    cpu_cache_files = {"sam3/model/position_encoding.py", "sam3/model/decoder.py"}
    return [patch for patch in PATCHES if patch["file"] not in cpu_cache_files]


def video_adaptations(source, device="cpu"):
    if device not in ("cpu", "cuda"):
        raise ValueError("unsupported reference device")
    changes = []
    for patch in image_patches(device):
        text = (source / patch["file"]).read_text(encoding="utf-8")
        if patch["file"] == "sam3/model_builder.py":
            if hashlib.sha256(text.encode()).hexdigest() != patch["source_sha256"]:
                raise ValueError("pinned video builder source hash differs")
            for statement in patch["removed_imports"]:
                if "sam3_tracking_predictor" not in statement and "sam3_video_inference" not in statement:
                    if text.count(statement) != 1:
                        raise ValueError("pinned video builder import differs")
                    text = text.replace(statement, "")
            text = "from __future__ import annotations\n\n" + text
        else:
            text = patched_text(text, patch)
        changes.append((patch["file"], text,
                        "CPU cache/import adaptation; retain original single-rank video inference and tracker."))
    for filename, digest in VIDEO_SOURCE_HASHES.items():
        text = (source / filename).read_text(encoding="utf-8")
        if hashlib.sha256(text.encode()).hexdigest() != digest:
            raise ValueError(f"pinned video source hash differs: {filename}")
        replacements = []
        if filename.endswith("sam3_tracker_utils.py"):
            replacements = [("from sam3.model.edt import edt_triton\n", ""),
                            ("    fn_mask_dt = edt_triton(padded_fn_masks)",
                             "    from sam3.model.edt import edt_triton\n\n    fn_mask_dt = edt_triton(padded_fn_masks)")]
        if filename.endswith("sam3_tracking_predictor.py"):
            replacements = [("torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16)",
                             f"torch.autocast(device_type=\"{device}\", enabled=False)"),
                            ("torch.device(\"cuda\")", "self.device")]
        if filename.endswith("sam3_video_inference.py") and device == "cuda":
            original = '@torch.autocast(device_type="cuda", dtype=torch.bfloat16)'
            if text.count(original) != 2:
                raise ValueError("expected two pinned video autocast decorators")
            text = text.replace(original, '@torch.autocast(device_type="cuda", enabled=False)')
        if filename.endswith("sam3_video_base.py"):
            replacements = [(f'sam3_image_out["tracker_backbone_fpn_{i}"]',
                             f'sam3_image_out["tracker_backbone_fpn_{i}"].float()') for i in range(3)]
        if filename.endswith("connected_components.py"):
            replacements = [("    batch_size = input_tensor.shape[0]\n",
                             "    batch_size = input_tensor.shape[0]\n"
                             "    if batch_size == 0:\n"
                             "        empty = torch.zeros(out_shape, dtype=torch.int64, device=input_tensor.device)\n"
                             "        return empty, empty.clone()\n")]
        for original, replacement in replacements:
            if text.count(original) != 1:
                raise ValueError(f"expected one CPU-video adaptation site: {filename}: {original}")
            text = text.replace(original, replacement)
        text = text.replace(".pin_memory()", "")
        text = text.replace(".cuda(non_blocking=True)", ".to(device=self.device, non_blocking=True)")
        text = text.replace(".cuda()", ".to(device=self.device)")
        if filename.endswith("sam3_tracker_base.py"):
            text = text.replace('prev["maskmem_features"].to(device=self.device, non_blocking=True)',
                                'prev["maskmem_features"].to(device=self.device, non_blocking=True).float()')
        reason = "CPU-only transfers and FP32 consumers; retain explicit BF16 storage and original temporal rules. Defer unused interactive Triton import."
        if device == "cuda":
            reason = "Device-aware transfers and FP32 consumers/autocast; retain explicit BF16 storage and original temporal rules. Defer unused interactive Triton import."
        changes.append((filename, text, reason))
    return [{"file": filename, "source_sha256": sha256_file(source / filename),
             "adapted_sha256": hashlib.sha256(text.encode()).hexdigest(), "reason": reason,
             "text": text} for filename, text, reason in changes]


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


def prepare(source, output, task="image", device="cpu"):
    if task not in ("image", "video"):
        raise ValueError("unsupported reference task")
    if device not in ("cpu", "cuda"):
        raise ValueError("unsupported reference device")
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
        changes = video_adaptations(source, device) if task == "video" else image_patches(device)
        for patch in changes:
            path = staging / patch["file"]
            text = patch["text"] if task == "video" else patched_text(path.read_text(encoding="utf-8"), patch)
            path.write_text(text, encoding="utf-8")
        write_json(staging / "adaptations.json", {"source": str(source), "source_revision": SAM3_REVISION,
                                                 "adapted_source": str(output), "preparer_sha256": sha256_file(__file__),
                                                 "task": task,
                                                 "device": device,
                                                 "changes": [{key: value for key, value in patch.items() if key != "text"}
                                                             for patch in changes]})
        validate_source(source, staging)
        os.rename(staging, output)
        print(f"Prepared pinned {device.upper()}-{task} reference source: {output}")
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=("image", "video"), default="image")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.task, args.device)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"reference source preparation failed: {error}\n")


if __name__ == "__main__":
    main()
