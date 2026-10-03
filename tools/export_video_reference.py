#!/usr/bin/env python3
"""Export the pinned Meta forward video oracle with explicit F16/BF16 storage.

Original modules retain association, lifecycle and postprocessing rules. The
isolated CPU source adaptations and unfused FP32 arithmetic are recorded.
Diagnostic subsets never satisfy the complete M2 acceptance corpus.
"""

import argparse
import importlib.metadata
import inspect
import os
from pathlib import Path
import platform
import shutil
import sys
import time

from export_reference import install_unfused_fp32, validate_source
from prepare_reference_source import video_adaptations
from sam3_artifacts import (BPE_SHA256, SAM3_REVISION, artifact_path, dump_array,
                           read_json, sha256_file, write_json)

OFFICIAL_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
CASES = Path(__file__).resolve().parents[1] / "tests/data/sam3-video-cases.json"


def selected_memory_trace(tracker, outputs, frame, count):
    from sam3.model.sam3_tracker_utils import select_closest_cond_frames

    selected, unselected = select_closest_cond_frames(frame, outputs["cond_frame_outputs"], 4,
                                                     keep_first_cond_frame=False)
    valid = tracker.frame_filter(outputs, False, frame, count, 1)
    spatial = [[index, 0] for index in selected]
    pointers = [[index, frame - index] for index in selected]
    available = outputs["non_cond_frame_outputs"].keys() | unselected.keys()
    for position in range(1, 7):
        relative = 7 - position
        if relative <= len(valid) and valid[-relative] in available:
            spatial.append([valid[-relative], position])
    for relative in range(1, min(count, 16)):
        if relative >= len(valid):
            break
        if valid[-relative] in available:
            pointers.append([valid[-relative], relative])
    return spatial, pointers


class Capture:
    """Capture only the current/next grounding result and requested stage frames."""

    def __init__(self, model):
        self.model = model
        self.frame = 0
        self.ids = []
        self.selected = set()
        self.tensors = {}
        self.grounding = {}
        self.groups = []
        self.propagation = []
        self._install()

    def store(self, name, value):
        if self.frame in self.selected:
            self.tensors[name] = value.detach().cpu().float().clone()

    def objects(self, name, value, ids=None):
        ids = self.ids if ids is None else ids
        if value.size(0) != len(ids):
            raise ValueError(f"{name}: original object batch does not match recorded IDs")
        for index, obj_id in enumerate(ids):
            self.store(f"object.{obj_id}.{name}", value[index:index + 1])

    def record_propagation(self, ious):
        if ious.size(0) != len(self.ids):
            raise ValueError("propagation candidate batch does not match recorded IDs")
        for index, obj_id in enumerate(self.ids):
            selected = int(ious[index].argmax()) + 1
            self.propagation.append({"id": int(obj_id), "iou": self.candidate_iou[index].tolist(),
                                     "mask_index": selected, "pointer_index": selected})

    def _install(self):
        original = self.model.detector.forward_grounding

        def grounding(*args, **kwargs):
            find_input = kwargs.get("find_input", args[1] if len(args) > 1 else None)
            frame = int(find_input.img_ids[0])
            result = original(*args, **kwargs)
            if frame in self.selected:
                backbone = result["prev_encoder_out"]["backbone_out"]
                self.grounding[frame] = {
                    "fusion_features": result["encoder_hidden_states"].permute(1, 2, 0).reshape(1, 256, 72, 72).detach(),
                    "presence_logits": result["presence_logit_dec"].detach().clone(),
                    "mask_logits": result["pred_masks"].detach(),
                    "text_features": backbone["language_features"][:, find_input.text_ids].detach(),
                    **{f"detector_neck_fpn{i}": value.detach() for i, value in enumerate(backbone["backbone_fpn"])},
                    "class_logits": self.raw_class.detach().clone(),
                }
            return result

        def scoring(module, args, value):
            self.raw_class = value[-1]

        self.model.detector.dot_prod_scoring.register_forward_hook(scoring)
        self.model.detector.forward_grounding = grounding
        original_video = self.model.detector.forward_video_grounding_multigpu

        def video(*args, **kwargs):
            result = original_video(*args, **kwargs)
            index = kwargs["frame_idx"]
            if index in self.selected:
                for name, value in self.grounding.pop(index).items():
                    self.store(name, value)
                for level in range(3):
                    self.store(f"tracker_neck_fpn{level}", result[0][f"tracker_backbone_fpn_{level}"])
            return result

        self.model.detector.forward_video_grounding_multigpu = video
        tracker = self.model.tracker
        original_inference = tracker._run_single_frame_inference

        def inference(*args, **kwargs):
            bound = inspect.signature(original_inference).bind(*args, **kwargs).arguments
            state = bound["inference_state"]
            if bound["batch_size"] != 1 or bound["mask_inputs"] is None:
                self.ids = list(state["obj_ids"])
            result = original_inference(*args, **kwargs)
            if bound["mask_inputs"] is None:
                self.objects("propagated_mask_logits", result[0]["pred_masks"])
            self.objects("pointer", result[0]["obj_ptr"])
            self.objects("object_logit", result[0]["object_score_logits"])
            return result

        tracker._run_single_frame_inference = inference
        original_add = tracker.add_new_mask

        def add(*args, **kwargs):
            self.ids = [int(kwargs["obj_id"])]
            return original_add(*args, **kwargs)

        tracker.add_new_mask = add
        original_condition = tracker._prepare_memory_conditioned_features

        def condition(*args, **kwargs):
            result = original_condition(*args, **kwargs)
            self.objects("conditioned_features", result)
            self.objects("propagated_conditioned_features", result)
            return result

        tracker._prepare_memory_conditioned_features = condition
        original_heads = tracker._forward_sam_heads

        def heads(*args, **kwargs):
            # The mask-input initialization bypasses conditioned features.
            if kwargs.get("mask_inputs") is not None:
                self.objects("conditioned_features", kwargs["backbone_features"])
            result = original_heads(*args, **kwargs)
            if kwargs.get("mask_inputs") is None:
                if not kwargs.get("multimask_output"):
                    raise ValueError("forward video propagation requires the pinned multimask path")
                self.record_propagation(result[2])
            return result

        tracker._forward_sam_heads = heads
        original_masks = tracker.sam_mask_decoder.predict_masks

        def masks(*args, **kwargs):
            result = original_masks(*args, **kwargs)
            self.candidate_iou = result[1].detach().cpu().float().clone()
            self.objects("decoder_masks", result[0])
            self.objects("decoder_iou", result[1])
            self.objects("decoder_object_logit", result[3])
            return result

        tracker.sam_mask_decoder.predict_masks = masks
        original_memory = tracker._run_memory_encoder

        def memory(*args, **kwargs):
            bound = inspect.signature(original_memory).bind(*args, **kwargs).arguments
            self.ids = list(bound["inference_state"]["obj_ids"])
            return original_memory(*args, **kwargs)

        tracker._run_memory_encoder = memory
        original_encode = tracker._encode_new_memory

        def encode(*args, **kwargs):
            result = original_encode(*args, **kwargs)
            self.objects("memory_features", result[0])
            return result

        tracker._encode_new_memory = encode
        tracker.maskmem_backbone.mask_downsampler.encoder[0].register_forward_pre_hook(
            lambda module, args: self.objects("memory_mask", args[0]))


def verify_behavior(case_id, results, traces):
    visible = [{obj["id"] for obj in frame["objects"]} for frame in results]
    if case_id == "negative":
        return not any(visible)
    if case_id == "motion":
        return bool(visible) and bool(set.intersection(*visible))
    if case_id == "entry":
        first = visible[0]
        late = set().union(*visible[16:]) - first
        births = {birth["id"] for trace in traces[16:] for birth in trace["births"]}
        return bool(first) and bool(late & births & visible[-1]) and bool(set.intersection(*visible))
    if case_id == "occlusion":
        return bool(visible[19] & visible[24] & visible[-1])
    if case_id == "hotstart-removal":
        removed = set().union(*(set(trace["removed"]) for trace in traces))
        births = {birth["id"] for trace in traces[:2] for birth in trace["births"]}
        return bool(removed & births) and not any((removed & ids for ids in visible))
    raise ValueError("unknown frozen video case")


def export(args):
    import numpy as np
    import torch
    from PIL import Image

    exporter_digest = sha256_file(__file__)

    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference environment")
    if sha256_file(args.checkpoint) != OFFICIAL_SHA256:
        raise ValueError("video oracle requires the pinned original checkpoint")
    source, runtime, adaptations = validate_source(args.sam3_source, args.sam3_runtime_source)
    expected = [{key: value for key, value in change.items() if key != "text"} for change in video_adaptations(source)]
    if adaptations != expected:
        raise ValueError("CPU video source differs from the recorded preparer's adaptations")
    bpe = args.bpe or source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if sha256_file(bpe) != BPE_SHA256:
        raise ValueError("video tokenizer asset differs from the pinned official BPE")
    frozen = read_json(args.cases)
    if frozen != read_json(CASES):
        raise ValueError("video cases differ from the frozen M2 recipes/gates")
    generated = read_json(args.frames / "manifest.json")
    if not generated["complete"] or generated["cases_sha256"] != sha256_file(args.cases):
        raise ValueError("generated frame recipes are incomplete or changed")
    if generated["pillow_version"] != "11.2.1" or importlib.metadata.version("pillow") != "11.2.1":
        raise ValueError("video reference requires pinned Pillow 11.2.1")
    wanted = [case for case in generated["cases"] if not args.case or case["id"] == args.case]
    if not wanted or len({case["id"] for case in generated["cases"]}) != len(frozen["cases"]):
        raise ValueError("video input case inventory is invalid")
    for case in wanted:
        expected_case = next(value for value in frozen["cases"] if value["id"] == case["id"])
        if any(case.get(key) != value for key, value in expected_case.items()) or len(case["frames_manifest"]) != case["frames"]:
            raise ValueError("video input recipe metadata differs")
        for index, frame in enumerate(case["frames_manifest"]):
            if frame["index"] != index or sha256_file(artifact_path(args.frames, frame["file"])) != frame["sha256"]:
                raise ValueError("video input frame index/hash differs")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", {"schema_version": 1, "complete": False, "eligible_for_milestone": False})
    shutil.copy2(args.cases, args.output / "cases.json")
    os.environ.update(USE_PERFLIB="0", HF_HUB_OFFLINE="1", RANK="0", WORLD_SIZE="1")
    sys.path.insert(0, str(runtime))
    from sam3.model_builder import build_sam3_video_model
    from sam3.model.sam3_video_inference import Sam3VideoInference

    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    replacement = install_unfused_fp32()
    model = build_sam3_video_model(checkpoint_path=str(args.checkpoint), bpe_path=str(bpe),
                                  device="cpu", load_from_HF=False, compile=False)
    model.max_num_objects = args.max_objects
    capture = Capture(model)
    cases = []
    with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
        for case in wanted:
            count = min(case["frames"], args.max_frames) if args.max_frames else case["frames"]
            directory = args.output / case["id"]
            for subdirectory in ("inputs", "trace", "tensors"):
                (directory / subdirectory).mkdir(parents=True, exist_ok=True)
            images = []
            for frame in case["frames_manifest"][:count]:
                path = artifact_path(args.frames, frame["file"])
                shutil.copy2(path, directory / "inputs" / f"{frame['index']:06d}.png")
                with Image.open(path) as original:
                    images.append(original.convert("RGB"))
            capture.selected = {0, 1, 16, count - 1}
            capture.grounding.clear()
            state = model.init_state(images, offload_video_to_cpu=True)
            state["text_prompt"] = case["prompt"]
            state["input_batch"].find_text_batch[0] = case["prompt"]
            tokens = model.detector.backbone.language_backbone.tokenizer([case["prompt"]], context_length=32)[0].tolist()
            traces, outputs, delayed, removed = [], [], [], set()
            previous_ids = []
            started = time.perf_counter()
            for frame in range(count):
                capture.frame = frame; capture.tensors = {}; capture.groups = []; capture.propagation = []
                for tracker_state in state["tracker_inference_states"]:
                    spatial, pointers = selected_memory_trace(model.tracker, tracker_state["output_dict"], frame, count)
                    capture.groups.append({"birth": min(tracker_state["output_dict"]["cond_frame_outputs"]),
                                           "ids": [int(value) for value in tracker_state["obj_ids"]], "spatial": spatial, "pointers": pointers})
                raw = Sam3VideoInference._run_single_frame_inference(model, state, frame, False)
                current_ids = state["tracker_metadata"]["obj_ids_all_gpu"].tolist()
                new_ids = [value for value in current_ids if value not in previous_ids]
                newly_removed = [value for value in previous_ids if value not in current_ids]
                removed.update(newly_removed)
                trace = {"frame_index": frame, "groups": capture.groups,
                         "births": [{"id": int(value)} for value in new_ids], "removed": newly_removed,
                         "propagation": capture.propagation}
                traces.append(trace); write_json(directory / "trace" / f"{frame:06d}.json", trace)
                previous_ids = current_ids
                if frame in capture.selected:
                    capture.store("preprocessed_image", state["input_batch"].img_batch[frame:frame + 1])
                    destination = directory / "tensors" / f"{frame:06d}"
                    destination.mkdir()
                    inventory = {name: dump_array(destination, name, value.numpy()) for name, value in capture.tensors.items()}
                    write_json(destination / "tensors.json", {"schema_version": 1, "byte_order": "little", "tensors": inventory})
                delayed.append((frame, raw))
                emit_count = len(delayed) if frame == count - 1 else (1 if len(delayed) >= 15 else 0)
                for _ in range(emit_count):
                    output_frame, pending = delayed.pop(0)
                    result = model._postprocess_output(state, pending, removed_obj_ids=removed,
                                                       suppressed_obj_ids=pending["suppressed_obj_ids"], unconfirmed_obj_ids=[])
                    target = directory / f"{output_frame:06d}"; target.mkdir()
                    objects = []
                    for index, obj_id in enumerate(result["out_obj_ids"]):
                        x, y, w, h = result["out_boxes_xywh"][index]
                        width, height = state["orig_width"], state["orig_height"]
                        objects.append({"id": int(obj_id), "score": float(result["out_probs"][index]),
                                        "box": [float(x * width), float(y * height), float((x + w) * width), float((y + h) * height)],
                                        "mask": dump_array(target, f"object-{obj_id}", result["out_binary_masks"][index].astype(np.uint8))})
                    record = {"schema_version": 1, "frame_index": output_frame, "emitted_after_frame": frame, "objects": objects}
                    outputs.append(record); write_json(target / "results.json", record)
                print(f"{case['id']}: {frame + 1}/{count}, objects {len(current_ids)}, emitted {len(outputs)}", flush=True)
            full = count == case["frames"]
            behavior = verify_behavior(case["id"], outputs, traces) if full else False
            record = {**{key: value for key, value in case.items() if key != "frames_manifest"},
                      "frames": count, "directory": case["id"], "width": state["orig_width"], "height": state["orig_height"],
                      "token_ids": tokens, "reference_behavior_verified": behavior,
                      "inference_seconds": time.perf_counter() - started,
                      "files": {path.relative_to(directory).as_posix(): sha256_file(path) for path in directory.rglob("*") if path.is_file()}}
            cases.append(record)
    eligible = not args.case and not args.max_frames and all(case["reference_behavior_verified"] for case in cases)
    packages = {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()}
    manifest = {"schema_version": 1, "task": "text_video", "architecture": "sam3", "sam3_revision": SAM3_REVISION,
                "complete": True, "eligible_for_milestone": eligible,
                "reference_kind": "official-checkpoint", "checkpoint": {"sha256": OFFICIAL_SHA256},
                "bpe": {"sha256": BPE_SHA256}, "cases_manifest": {"file": "cases.json", "sha256": sha256_file(args.cases)},
                "max_objects": args.max_objects,
                "oracle": {"variant": "official-video-fp32-explicit-storage", "device": "cpu", "precision": "float32",
                           "compile": False, "autocast": False, "tf32": False, "threads": args.threads,
                           "input_storage": "float16", "tracker_transport": "bfloat16", "memory_storage": "bfloat16",
                           "runtime_adaptations": adaptations, "fp32_adaptation": replacement, "packages": packages,
                           "exporter_sha256": exporter_digest, "python": sys.version, "platform": platform.platform(),
                           "forward_profile": "one fixed text prompt; each frame processed once; 15-frame delayed base outputs"},
                "cases": cases}
    write_json(args.output / "manifest.json", manifest)
    print(f"Exported {len(cases)} cases; complete video acceptance eligible: {eligible}")
    return 0 if eligible or args.case or args.max_frames else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sam3-source", required=True, type=Path)
    parser.add_argument("--sam3-runtime-source", required=True, type=Path)
    parser.add_argument("--bpe", type=Path)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--frames", type=Path, default=Path("models/video-cases"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--case", choices=("motion", "entry", "occlusion", "hotstart-removal", "negative"))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.threads <= 0 or args.max_objects <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("thread/object/frame limits must be positive")
    try:
        parser.exit(export(args))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f"video reference export failed: {error}\n")


if __name__ == "__main__":
    main()
