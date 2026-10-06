#!/usr/bin/env python3
"""Qualify a benchmark prefix using original Meta, or reuse a byte-identical sealed parent."""

import argparse
import copy
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
import time

from export_reference import configure_cuda_oracle, install_unfused_fp32, validate_source
from prepare_reference_source import video_adaptations
from sam3_artifacts import (BPE_SHA256, SAM3_REVISION, artifact_path, read_json,
                           sha256_file, write_json, verify_run_artifacts)
from generate_video_benchmark import SOURCE_SHA256
from benchmark_video import validate_qualification_records

CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"


def fixture_inputs(path, workload_id):
    path = Path(path).resolve()
    fixture = read_json(path)
    if (fixture.get("complete") is not True or fixture.get("frames") != 64
            or fixture.get("warmup_frames") != 16 or fixture.get("measured_frames") != 48
            or fixture.get("prompt") != "truck" or fixture.get("max_objects") != 8
            or fixture.get("source_sha256") != SOURCE_SHA256):
        raise ValueError("fixture does not match the pinned 64/16/48 protocol")
    workloads = [item for item in fixture["workloads"] if item["id"] == workload_id]
    if len(workloads) != 1:
        raise ValueError("fixture workload is absent or duplicated")
    workload = workloads[0]
    expected = 1 if workload_id == "one-object" else 4
    if workload["expected_objects"] != expected or len(workload["input_sha256"]) != 64:
        raise ValueError("fixture object/frame counts differ")
    frames = artifact_path(path.parent, workload["directory"])
    inputs = {str(frames / f"{i:06d}.png"): digest for i, digest in enumerate(workload["input_sha256"])}
    if len(list(frames.glob("*.png"))) != 64:
        raise ValueError("fixture PNG inventory differs")
    verify_run_artifacts(inputs)
    return fixture, workload, inputs


def reuse_qualification(fixture_path, workload_id, parent_path, parent_fixture_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("qualification output already exists")
    fixture, workload, inputs = fixture_inputs(fixture_path, workload_id)
    old_fixture, old_workload, old_inputs = fixture_inputs(parent_fixture_path, workload_id)
    parent = read_json(parent_path)
    if (any(parent.get(key) is not True for key in ("complete", "passed", "qualified_for_performance"))
            or parent.get("reference_kind") != "official-original-performance-prefix"
            or parent.get("checkpoint_sha256") != CHECKPOINT_SHA256 or parent.get("sam3_revision") != SAM3_REVISION
            or parent.get("declared_frame_count") != 64 or parent.get("prefix_frames") != 17
            or parent.get("workload") != workload_id or parent.get("expected_objects") != workload["expected_objects"]
            or len(parent.get("records", [])) != 17
            or parent.get("fixture_manifest_sha256") != sha256_file(parent_fixture_path)):
        raise ValueError("parent is not a complete matching original-module qualification")
    for key in ("width", "height", "frames", "prompt", "max_objects", "pillow_version"):
        if fixture[key] != old_fixture[key]:
            raise ValueError(f"qualification fixture property changed: {key}")
    for key in ("positions_xy_mirror_phase", "input_sha256"):
        if workload[key] != old_workload[key]:
            raise ValueError(f"qualification workload changed: {key}")
    if {Path(p).name: h for p, h in old_inputs.items()} != {Path(p).name: h for p, h in parent["input_sha256"].items()}:
        raise ValueError("parent qualification PNG hashes differ")
    validate_qualification_records(parent, workload["expected_objects"])
    verify_run_artifacts(parent["source_sha256"])
    report = copy.deepcopy(parent)
    report["fixture_manifest_sha256"] = sha256_file(fixture_path)
    report["input_sha256"] = inputs
    report["source_sha256"].update({str(Path(p).resolve()): sha256_file(p)
                                   for p in (__file__, Path(__file__).with_name("benchmark_video.py"), fixture_path, parent_path, parent_fixture_path)})
    report["reused_qualification"] = {"parent": str(Path(parent_path).resolve()),
                                      "parent_sha256": sha256_file(parent_path),
                                      "scope": "Same 64 PNG bytes, dimensions, prompt, object positions and declared-length prefix; no new oracle execution."}
    verify_run_artifacts(report["source_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        import json
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return output


def qualify(args):
    os.environ.update(USE_PERFLIB="0", HF_HUB_OFFLINE="1", RANK="0", WORLD_SIZE="1")
    fixture_path, receipt = args.fixture_manifest.resolve(), args.output.resolve()
    if receipt.exists():
        raise FileExistsError("qualification output already exists")
    fixture, workload, inputs = fixture_inputs(fixture_path, args.workload)
    root = fixture_path.parent
    source, runtime = args.sam3_source.resolve(), args.sam3_runtime_source.resolve()
    _, _, adaptations = validate_source(source, runtime)
    if adaptations != [{key: value for key, value in change.items() if key != "text"}
                       for change in video_adaptations(source, args.device)]:
        raise ValueError("runtime adaptations differ from the pinned device/video preparation")
    checkpoint, bpe = args.checkpoint.resolve(), args.bpe.resolve()
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256 or sha256_file(bpe) != BPE_SHA256:
        raise ValueError("original checkpoint/tokenizer identity differs")
    source_files = {str(path): sha256_file(path) for path in [Path(__file__).resolve(), fixture_path, checkpoint, bpe,
                    runtime / "adaptations.json", *source.rglob("*.py"), *runtime.rglob("*.py"),
                    *[Path(__file__).parent / name for name in ("export_reference.py", "prepare_reference_source.py",
                                                               "sam3_artifacts.py", "requirements.lock", "benchmark_video.py", "generate_video_benchmark.py")]]}
    report = {"schema_version": 1, "complete": False, "passed": False, "qualified_for_performance": False,
              "eligible_for_milestone": False, "reference_kind": "official-original-performance-prefix",
              "sam3_revision": SAM3_REVISION, "workload": args.workload, "expected_objects": workload["expected_objects"],
              "declared_frame_count": 64, "prefix_frames": 17, "max_objects": 8,
              "fixture_manifest_sha256": sha256_file(fixture_path), "checkpoint_sha256": checkpoint_hash,
              "source_sha256": source_files, "input_sha256": inputs, "runtime_adaptations": adaptations,
              "records": [], "emitted": [], "failures": []}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x") as stream:
        import json
        stream.write(json.dumps(report, indent=2) + "\n")
    sys.path.insert(0, str(runtime))
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_video_model
    from sam3.model.sam3_video_inference import Sam3VideoInference
    torch.set_num_threads(4); torch.set_default_dtype(torch.float32); torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report["oracle"] = {"device": args.device, "precision": "float32", "autocast": False, "tf32": False,
                        **configure_cuda_oracle(torch, args.device)}
    report["unfused_fp32_adaptation"] = install_unfused_fp32()
    report["packages"] = {name: importlib.metadata.version(name) for name in ("torch","numpy","Pillow")}
    model = build_sam3_video_model(checkpoint_path=str(checkpoint), bpe_path=str(bpe), device=args.device, load_from_HF=False, compile=False)
    model.max_num_objects = 8
    images = []
    for path in inputs:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    expected = workload["expected_objects"]
    started = time.perf_counter()
    try:
        with torch.inference_mode(), torch.autocast(device_type=args.device, enabled=False):
            state = model.init_state(images, offload_video_to_cpu=True)
            state["text_prompt"] = fixture["prompt"]; state["input_batch"].find_text_batch[0] = fixture["prompt"]
            previous = set(); initial = None; quadrants = {}; delayed = []; removed = set()
            def observed(raw):
                value = model._postprocess_output(state, raw, removed_obj_ids=removed,
                                                 suppressed_obj_ids=raw["suppressed_obj_ids"], unconfirmed_obj_ids=[])
                result = []
                for index, obj_id in enumerate(value["out_obj_ids"]):
                    mask = np.asarray(value["out_binary_masks"][index], dtype=np.uint8)
                    x,y,w,h = [float(v) for v in value["out_boxes_xywh"][index]]
                    result.append({"id": int(obj_id), "score": float(value["out_probs"][index]), "box_xywh_normalized": [x,y,w,h],
                                   "area": int(mask.sum()), "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                                   "quadrant": int(x+w/2 >= .5) + 2*int(y+h/2 >= .5)})
                return result
            for frame in range(17):
                raw = Sam3VideoInference._run_single_frame_inference(model, state, frame, False)
                active = set(int(value) for value in state["tracker_metadata"]["obj_ids_all_gpu"].tolist())
                births, deaths = sorted(active-previous), sorted(previous-active)
                removed.update(deaths)
                visible = observed(raw)
                row = {"frame_index": frame, "active_ids": sorted(active), "births": births, "removed": deaths,
                       "visible": visible, "elapsed_seconds": time.perf_counter()-started}
                report["records"].append(row)
                if initial is None:
                    initial = active
                    quadrants = {item["id"]: item["quadrant"] for item in visible}
                if len(active) != expected or len(visible) != expected or active != {item["id"] for item in visible}:
                    raise ValueError(f"frame {frame}: expected {expected} complete objects, active={sorted(active)}, visible={[item['id'] for item in visible]}")
                if active != initial or (frame > 0 and (births or deaths)) or any(item["area"] == 0 for item in visible):
                    raise ValueError(f"frame {frame}: object continuity/nonempty mask failed")
                if expected == 4 and (set(quadrants.values()) != {0,1,2,3} or any(quadrants[item["id"]] != item["quadrant"] for item in visible)):
                    raise ValueError(f"frame {frame}: one persistent object per quadrant was not established")
                delayed.append((frame,raw))
                if len(delayed) >= 15:
                    output_frame, pending = delayed.pop(0)
                    emitted = observed(pending)
                    if {item["id"] for item in emitted} != initial:
                        raise ValueError(f"delayed frame {output_frame}: IDs differ")
                    report["emitted"].append({"frame_index": output_frame, "emitted_after_frame": frame, "objects": emitted})
                previous = active
                write_json(receipt, report)
                print(args.workload, frame, "active", sorted(active), "emitted", len(report["emitted"]), flush=True)
        validate_qualification_records(report, expected)
        verify_run_artifacts(inputs); verify_run_artifacts(source_files)
        report.update(complete=True, passed=True, qualified_for_performance=True,
                      pending_frames_at_prefix_end=len(delayed), first_output_after_frame=report["emitted"][0]["emitted_after_frame"],
                      scope="17-frame prefix of a 64-frame initialized sequence; no artificial final drain. Full 64-frame actual benchmark counts/IDs remain required.")
    except (OSError,ValueError,RuntimeError,AssertionError) as error:
        report["failures"].append(str(error))
        write_json(receipt, report)
        raise
    write_json(receipt, report)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--workload", choices=("one-object", "four-object"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--parent-fixture", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for name in ("sam3-source", "sam3-runtime-source", "checkpoint", "bpe"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    try:
        if args.reuse_from:
            if args.parent_fixture is None:
                parser.error("--reuse-from requires --parent-fixture")
            print(reuse_qualification(args.fixture_manifest, args.workload, args.reuse_from, args.parent_fixture, args.output))
        else:
            if any(getattr(args, name) is None for name in ("sam3_source", "sam3_runtime_source", "checkpoint", "bpe")):
                parser.error("fresh qualification requires source/runtime/checkpoint/bpe")
            print(qualify(args))
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration) as error:
        parser.exit(1, f"qualification failed: {error}\n")


if __name__ == "__main__":
    main()
