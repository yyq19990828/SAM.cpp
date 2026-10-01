#!/usr/bin/env python3
"""Export pinned official SAM 3 image results and true FP32 pipeline tensors."""

import argparse
import importlib.metadata
import inspect
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from convert_sam3 import unused_tracker_key
from sam3_artifacts import (BPE_SHA256, REQUIRED_TENSORS, SAM3_REVISION,
                           artifact_path, dump_array, load_case_manifest, read_json, read_tensor_index,
                           sha256_file, write_json)


def validate_source(source, runtime_source=None):
    source = Path(source).resolve()
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    if revision != SAM3_REVISION:
        raise ValueError(f"official source revision must be {SAM3_REVISION}; got {revision}")
    dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--", "sam3"],
                           check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("pinned official sam3 package has local modifications")
    runtime_source = Path(runtime_source).resolve() if runtime_source else source
    changes = []
    if runtime_source != source:
        adaptation = read_json(runtime_source / "adaptations.json")
        if adaptation.get("source_revision") != SAM3_REVISION:
            raise ValueError("CPU runtime adaptations do not identify the pinned source")
        changes = adaptation["changes"]
        declared = {item["file"]: item for item in changes}
        if len(declared) != len(changes):
            raise ValueError("duplicate runtime adaptation entries")
        original_files = {path.relative_to(source).as_posix() for path in (source / "sam3").rglob("*.py")}
        runtime_files = {path.relative_to(runtime_source).as_posix() for path in (runtime_source / "sam3").rglob("*.py")}
        if original_files != runtime_files:
            raise ValueError("adapted runtime must retain the complete original Python package")
        for filename in original_files:
            original_hash = sha256_file(source / filename)
            runtime_hash = sha256_file(runtime_source / filename)
            if filename in declared:
                item = declared[filename]
                if original_hash != item["source_sha256"] or runtime_hash != item["adapted_sha256"]:
                    raise ValueError(f"runtime adaptation hash mismatch: {filename}")
            elif original_hash != runtime_hash:
                raise ValueError(f"unrecorded runtime modification: {filename}")
        if set(declared) - original_files:
            raise ValueError("adaptation manifest refers to a missing package file")
    return source, runtime_source, changes


def install_unfused_fp32():
    import torch
    import torch.nn.functional as functional
    import sam3.model.vitdet as vitdet

    def unfused_addmm_act(activation, linear, mat1):
        if torch.is_grad_enabled() or mat1.dtype != torch.float32 or linear.weight.dtype != torch.float32:
            raise ValueError("FP32 oracle requires inference mode and FP32 MLP inputs/weights")
        result = functional.linear(mat1, linear.weight, linear.bias)
        if activation in (functional.gelu, torch.nn.GELU):
            return functional.gelu(result)
        if activation in (functional.relu, torch.nn.ReLU):
            return functional.relu(result)
        raise ValueError(f"unsupported fused activation: {activation}")

    vitdet.addmm_act = unfused_addmm_act
    return {"target": "sam3.model.vitdet.addmm_act", "replacement": inspect.getsource(unfused_addmm_act),
            "reason": "Pinned fused.addmm_act unconditionally casts to BF16. Use the identical linear/activation formulas in FP32."}


def prepare_image(case, root):
    from PIL import Image

    source = artifact_path(root, case["image"])
    if sha256_file(source) != case["source_sha256"]:
        raise ValueError(f"{case['id']}: source image SHA-256 mismatch")
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    transform = case.get("transform")
    if transform:
        if set(transform) != {"crop_xyxy", "resize_wh", "resampling"} or transform["resampling"] != "Pillow.Resampling.BILINEAR":
            raise ValueError("only the frozen crop/Pillow bilinear resize transform is supported")
        crop = transform["crop_xyxy"]
        resize = transform["resize_wh"]
        if (len(crop) != 4 or any(type(x) is not int for x in crop)
                or not 0 <= crop[0] < crop[2] <= image.width
                or not 0 <= crop[1] < crop[3] <= image.height
                or len(resize) != 2 or any(type(x) is not int or x <= 0 for x in resize)):
            raise ValueError("invalid frozen crop/resize bounds")
        image = image.crop(tuple(crop)).resize(tuple(resize), Image.Resampling.BILINEAR)
    return image


def load_detector_checkpoint(model, state_dict):
    import torch

    if (not isinstance(state_dict, dict)
            or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                   for key, value in state_dict.items())):
        raise ValueError("official checkpoint does not contain a tensor state dictionary")
    schema = read_json(Path(__file__).with_name("sam3_tensor_schema.json"))
    if schema.get("schema_version") != 1 or schema.get("sam3_revision") != SAM3_REVISION:
        raise ValueError("unsupported detector tensor schema")
    detector = {key.removeprefix("detector."): value for key, value in state_dict.items()
                if key.startswith("detector.")}
    if not detector:
        raise ValueError("official checkpoint has no detector tensors")
    skipped = []
    for key in list(detector):
        if not key.startswith("backbone.vision_backbone.sam2_convs."):
            continue
        tensor = detector[key]
        dimensions = list(reversed(tensor.shape))
        while len(dimensions) > 1 and dimensions[-1] == 1:
            dimensions.pop()
        source_key = "detector." + key
        if (not tensor.is_floating_point() or tensor.is_complex()
                or schema["unused_tracker_tensors"].get(unused_tracker_key(source_key)) != dimensions):
            raise ValueError(f"unrecognized or incompatible unused tracker tensor: {source_key}")
        skipped.append({"name": source_key, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "reason": "Unused interactive SAM 2 neck; enable_inst_interactivity=False."})
        del detector[key]
    missing, unexpected = model.load_state_dict(detector, strict=False)
    if missing or unexpected:
        raise ValueError(f"official checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return skipped


def export(args):
    import numpy as np
    import torch

    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use an isolated Python virtual environment for reference export")
    if not args.sam3_source:
        raise ValueError("set SAM3_SOURCE_DIR or pass --sam3-source with the pinned official checkout")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"original official checkpoint is missing: {args.checkpoint}")
    reference_kind = "official-checkpoint"
    supplementary_weights = None
    if args.supplementary_weights_manifest:
        supplementary_weights = read_json(args.supplementary_weights_manifest)
        if (supplementary_weights.get("schema_version") != 1
                or supplementary_weights["restored_checkpoint"]["sha256"] != sha256_file(args.checkpoint)
                or supplementary_weights["container"]["precision"] not in ("f32", "f16")
                or not supplementary_weights["container"].get("hf_revision")
                or not supplementary_weights.get("reconstruction")):
            raise ValueError("incomplete or mismatched supplementary weight provenance")
        reference_kind = "supplementary-converted-weights"
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reference directory: {output}")
    source, runtime_source, adaptations = validate_source(args.sam3_source, args.sam3_runtime_source)
    sys.path.insert(0, str(runtime_source))
    os.environ["HF_HUB_OFFLINE"] = "1"
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe = args.bpe or source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if sha256_file(bpe) != BPE_SHA256:
        raise ValueError("BPE asset does not match the pinned official SHA-256")
    manifest = load_case_manifest(args.cases)
    root = args.input_root.resolve()
    # Check all sources before allocating the model or starting expensive inference.
    for case in manifest["cases"]:
        prepare_image(case, root)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    replacement = install_unfused_fp32()
    model = build_sam3_image_model(bpe_path=str(bpe), device=args.device, eval_mode=True,
                                  checkpoint_path=None, load_from_HF=False,
                                  enable_inst_interactivity=False, compile=False)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and isinstance(state_dict.get("model"), dict):
        state_dict = state_dict["model"]
    skipped_detector_tensors = load_detector_checkpoint(model, state_dict)
    del state_dict
    # Module.float converts real floating buffers while preserving complex RoPE phases.
    model = model.float().to(device=args.device).eval()
    processor = Sam3Processor(model, device=args.device, confidence_threshold=0.5)
    tokenizer = model.backbone.language_backbone.tokenizer
    captured = {}
    original_image = model.backbone.forward_image
    original_grounding = model.forward_grounding

    def capture_image(image):
        captured["preprocessed_image"] = image.detach()
        return original_image(image)

    def capture_grounding(*arguments, **keywords):
        result = original_grounding(*arguments, **keywords)
        captured["grounding"] = result
        return result

    model.backbone.forward_image = capture_image
    model.forward_grounding = capture_grounding
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sam3-reference-", dir=output.parent))
    try:
        shutil.copyfile(args.cases, staging / "cases.json")
        bundle_cases = []
        previous_image_key = None
        state = None
        with torch.inference_mode(), torch.autocast(device_type=args.device, enabled=False):
            for case in manifest["cases"]:
                print(f"Exporting {case['id']}", flush=True)
                image = prepare_image(case, root)
                case_directory = staging / case["id"]
                case_directory.mkdir()
                image.save(case_directory / "input.ppm")
                image_key = (case["image"], repr(case.get("transform")))
                start = time.perf_counter()
                cache_hit = image_key == previous_image_key
                if not cache_hit:
                    state = processor.set_image(image)
                    previous_image_key = image_key
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(case["prompt"], state)
                elapsed = time.perf_counter() - start
                raw = captured["grounding"]
                fusion = raw["encoder_hidden_states"]
                if list(fusion.shape) != [72 * 72, 1, 256]:
                    raise ValueError(f"fusion encoder memory has unsupported shape: {list(fusion.shape)}")
                tensors = {"preprocessed_image": captured["preprocessed_image"],
                           **{f"vision_features_{index}": feature for index, feature in enumerate(state["backbone_out"]["backbone_fpn"])},
                           "text_features": state["backbone_out"]["language_features"],
                           "fusion_features": fusion.permute(1, 2, 0).reshape(1, 256, 72, 72),
                           "pred_boxes": raw["pred_boxes"], "presence_logits": raw["presence_logit_dec"],
                           "class_logits": raw["pred_logits"], "mask_logits": raw["pred_masks"]}
                inventory = {}
                for name, (shape, layout) in REQUIRED_TENSORS.items():
                    tensor = tensors[name]
                    if tensor.dtype != torch.float32 or list(tensor.shape) != shape:
                        raise ValueError(f"{name}: expected true FP32 {shape}, got {tensor.dtype} {list(tensor.shape)}")
                    inventory[name] = dump_array(case_directory, name, tensor.detach().cpu().contiguous().numpy(), layout)
                tokens = tokenizer([case["prompt"]], context_length=32)[0].tolist()
                write_json(case_directory / "tensors.json", {"schema_version": 1, "byte_order": "little",
                                                           "token_ids": tokens, "tensors": inventory})
                query_scores = (raw["pred_logits"].sigmoid() * raw["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
                query_indices = torch.nonzero(query_scores[0] > 0.5).flatten().cpu().tolist()
                detections = []
                for index, query in enumerate(query_indices):
                    mask = state["masks"][index, 0].detach().cpu().numpy().astype(np.uint8)
                    detections.append({"query_index": query, "score": float(state["scores"][index]),
                                       "box": state["boxes"][index].cpu().tolist(),
                                       "mask": dump_array(case_directory, f"mask-{query}", mask)})
                write_json(case_directory / "results.json", {"schema_version": 1, "width": image.width,
                                                           "height": image.height, "prompt": case["prompt"],
                                                           "score_threshold": 0.5, "detections": detections})
                read_tensor_index(case_directory)
                bundle_cases.append({**case, "directory": case["id"], "input": "input.ppm",
                                     "input_sha256": sha256_file(case_directory / "input.ppm"),
                                     "tensors_sha256": sha256_file(case_directory / "tensors.json"),
                                     "results_sha256": sha256_file(case_directory / "results.json"),
                                     "inference_seconds": elapsed, "image_cache_hit": cache_hit})
        packages = {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()}
        bundle = {"schema_version": 1, "architecture": "sam3", "sam3_revision": SAM3_REVISION,
                  "reference_kind": reference_kind, "eligible_for_milestone": reference_kind == "official-checkpoint",
                  "checkpoint": {"file": args.checkpoint.name, "sha256": sha256_file(args.checkpoint)},
                  "bpe": {"sha256": sha256_file(bpe)},
                  "cases_manifest": {"file": "cases.json", "sha256": sha256_file(staging / "cases.json")},
                  "oracle": {"variant": "official-unfused-fp32" if reference_kind == "official-checkpoint" else "supplementary-converted-weights-unfused-fp32",
                             "source_graph": "pinned Meta SAM 3", "device": args.device, "precision": "float32",
                             "compile": False, "autocast": False, "tf32": False,
                             "deterministic_algorithms": True, "threads": args.threads,
                             "unused_detector_tensors": skipped_detector_tensors,
                             "runtime_adaptations": adaptations, "fp32_adaptation": replacement,
                             "fused_source_sha256": sha256_file(source / "sam3/perflib/fused.py"),
                             "vitdet_source_sha256": sha256_file(source / "sam3/model/vitdet.py"),
                             "exporter_sha256": sha256_file(__file__), "python": sys.version,
                             "platform": platform.platform(), "packages": packages},
                  "cases": bundle_cases}
        if supplementary_weights is not None:
            bundle["supplementary_weights"] = supplementary_weights
        write_json(staging / "manifest.json", bundle)
        os.rename(staging, output)
        print(f"Exported {len(bundle_cases)} {reference_kind} unfused-FP32 cases to {output}")
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam3-source", default=os.environ.get("SAM3_SOURCE_DIR"), type=Path)
    parser.add_argument("--sam3-runtime-source", type=Path)
    parser.add_argument("--bpe", type=Path)
    parser.add_argument("--supplementary-weights-manifest", type=Path,
                        help="Label a restored public container checkpoint as supplementary, never official-checkpoint acceptance")
    parser.add_argument("--input-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    try:
        export(args)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"reference export failed: {error}\n")


if __name__ == "__main__":
    main()
