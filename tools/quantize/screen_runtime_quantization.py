#!/usr/bin/env python3
"""Screen vision-linear or image-cache quantization on held-out COCO selection images."""

import argparse
import inspect
import os
from pathlib import Path
import shutil
import sys
import time

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.quantize.calibration import load_dataset
from tools.quantize.cache_quantization import CACHE_MODES, CacheQuantization, CudaCacheEncoder
from tools.convert.convert_sam3 import rename_key
from tools.validation.export_reference import configure_cuda_oracle, install_unfused_fp32, load_detector_checkpoint, validate_source
from tools.quantize.runtime_quantization import (GATES_PATH, GATES_SHA256, LAYER_FAMILIES, MODES, SCALE_MODES,
                                  LinearQuantization, channel_scale, compare_outputs, encode_mask,
                                  load_gates, select_layers, validate_output)
from tools.convert.sam3_artifacts import BPE_SHA256, artifact_path, read_json, sha256_file, verify_run_artifacts, write_json
from tools.convert.sam3_gguf import quantization_module_for_linear_weight
from tools.quantize.study_activation_quantization import calibration_inputs, smooth_scale


def capture_output(state, raw, prompt, tokenizer, torch):
    from sam3.model.box_ops import box_cxcywh_to_xyxy
    width, height = state["original_width"], state["original_height"]
    scores = (raw["pred_logits"].sigmoid() * raw["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
    if scores.shape != (1, 200) or raw["pred_boxes"].shape != (1, 200, 4):
        raise ValueError("SAM 3 query contract differs")
    indices = torch.nonzero(scores[0] > 0.5).flatten().cpu().tolist()
    if len(indices) != len(state["masks"]):
        raise ValueError("processor selected mask inventory differs")
    dimensions = torch.tensor([width, height, width, height], device=scores.device)
    boxes = box_cxcywh_to_xyxy(raw["pred_boxes"])[0] * dimensions
    if not torch.isfinite(raw["pred_masks"]).all():
        raise ValueError("non-finite model mask logits")
    detections = []
    for row, query in enumerate(indices):
        if not torch.equal(state["boxes"][row], boxes[query]) or abs(float(state["scores"][row] - scores[0, query])) > 1e-6:
            raise ValueError("processor score/box differs from original raw outputs")
        detections.append({"query_index": query, "mask": encode_mask(state["masks"][row, 0].cpu().numpy())})
    result = {"schema_version": 1, "width": width, "height": height, "prompt": prompt,
              "token_ids": tokenizer([prompt], context_length=32)[0].tolist(),
              "query_scores": scores[0].cpu().tolist(), "query_boxes": boxes.cpu().tolist(),
              "detections": detections}
    validate_output(result, 0.5)
    return result


def feature_error(actual, reference, np):
    actual = actual.detach().cpu().numpy()
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("vision feature shape or finiteness differs")
    a, r = actual.reshape(-1), reference.reshape(-1)
    squared, norm, maximum = 0.0, 0.0, 0.0
    for start in range(0, len(a), 1024 * 1024):
        x, y = a[start:start + 1024 * 1024].astype(np.float64), r[start:start + 1024 * 1024].astype(np.float64)
        difference = x - y
        squared += float(np.square(difference).sum())
        norm += float(np.square(y).sum())
        maximum = max(maximum, float(np.abs(difference).max()))
    return {"relative_l2": float(np.sqrt(squared / norm)) if norm > 1e-30 else None, "max_abs": maximum}


def screen(args):
    import numpy as np
    import torch
    from PIL import Image, __version__ as pillow_version
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference environment")
    if args.output.exists():
        raise FileExistsError("screening output must be new")
    gates = load_gates()
    tools_root = Path(__file__).resolve().parents[1]
    names = (Path(__file__), Path(__file__).with_name("runtime_quantization.py"),
             Path(__file__).with_name("cache_quantization.py"), Path(__file__).with_name("calibration.py"),
             tools_root / "validation/export_reference.py",
             Path(__file__).with_name("study_activation_quantization.py"),
             tools_root / "convert/convert_sam3.py", tools_root / "convert/sam3_gguf.py",
             tools_root / "convert/sam3_artifacts.py")
    sources = {str(path.resolve()): sha256_file(path) for path in names}
    sources[str(GATES_PATH)] = GATES_SHA256
    cache_encoder = None
    if args.target == "image-cache" and "cache-q8_0" in args.modes:
        if args.cache_preflight is None:
            raise ValueError("Q8_0 cache requires an exact native-codec preflight report")
        preflight = read_json(args.cache_preflight)
        if not preflight.get("complete") or not preflight.get("passed"):
            raise ValueError("cache codec preflight did not pass")
        helper = Path(__file__).with_name("cache_quantization.py").resolve()
        if preflight["artifact_sha256"].get(str(helper)) != sha256_file(helper):
            raise ValueError("cache numerical source differs from its preflight")
        sources[str(args.cache_preflight.resolve())] = sha256_file(args.cache_preflight)
        if args.device == "cuda":
            if args.cache_bridge is None:
                raise ValueError("CUDA Q8_0 cache requires a validated native bridge")
            library = args.cache_bridge.resolve(strict=True)
            bridge_source = Path(__file__).with_name("ggml_cuda_cache_bridge.cu").resolve()
            for path in (library, bridge_source):
                digest = sha256_file(path)
                if preflight["artifact_sha256"].get(str(path)) != digest:
                    raise ValueError("native CUDA cache bridge differs from its preflight")
                sources[str(path)] = digest
            cache_encoder = CudaCacheEncoder(library)
    manifest, statistics, reservoir = calibration_inputs(args.calibration)
    calibration_digest = sha256_file(args.calibration / "manifest.json")
    del reservoir
    dataset_digest = sha256_file(args.dataset)
    if dataset_digest != manifest["dataset"]["sha256"]:
        raise ValueError("selection dataset differs from calibration split manifest")
    dataset = load_dataset(args.dataset, args.input_root)
    samples = [sample for sample in dataset["samples"] if sample["split"] == "selection"]
    if args.sample_ids is not None:
        if not args.diagnostic or args.limit is not None:
            raise ValueError("sample IDs require --diagnostic without --limit")
        requested = set(args.sample_ids)
        if len(requested) != len(args.sample_ids) or not requested <= {sample["id"] for sample in samples}:
            raise ValueError("sample IDs must be unique members of selection")
        samples = [sample for sample in samples if sample["id"] in requested]
    if args.limit is not None:
        if not args.diagnostic:
            raise ValueError("partial selection requires --diagnostic")
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("no selection images")
    checkpoint_digest = sha256_file(args.checkpoint)
    if checkpoint_digest != manifest["checkpoint_sha256"]:
        raise ValueError("original checkpoint differs from calibration")
    source, runtime_source, adaptations = validate_source(args.sam3_source, args.sam3_runtime_source)
    bpe = args.bpe or source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if sha256_file(bpe) != BPE_SHA256:
        raise ValueError("BPE differs from the original model")
    sys.path.insert(0, str(runtime_source))
    os.environ["HF_HUB_OFFLINE"], os.environ["USE_PERFLIB"] = "1", "0"
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import sam3.model.vitdet as vitdet
    torch.set_num_threads(args.threads)
    torch.manual_seed(dataset["seed"])
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    oracle = configure_cuda_oracle(torch, args.device)
    original_addmm = vitdet.addmm_act
    adaptation = install_unfused_fp32()
    model = build_sam3_image_model(bpe_path=str(bpe), device=args.device, eval_mode=True,
                                  checkpoint_path=None, load_from_HF=False, enable_inst_interactivity=False, compile=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    load_detector_checkpoint(model, state)
    del state
    model = model.float().to(device=args.device).eval()
    modules = {}
    for path, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            name, _ = rename_key("detector." + path + ".weight")
            if name and quantization_module_for_linear_weight(name) == "vision":
                modules[name] = module
    if set(modules) != set(statistics) or len(modules) != 128:
        raise ValueError("model/calibration vision linear inventory differs")
    scales, selected = {}, []
    if args.target == "vision-linear":
        scales = {name: channel_scale(smooth_scale(row["activation"]["absmax"], row["weight_input_absmax"], args.alpha), args.scale_mode)
                  for name, row in statistics.items()}
        selected = select_layers(modules, args.layer_family, args.blocks)
        control = LinearQuantization(modules, scales, torch, selected=selected,
                                     exact_power_of_two=args.scale_mode in ("power-of-two", "identity"))
    else:
        control = CacheQuantization(model.backbone, torch, levels=args.cache_levels, cuda_encoder=cache_encoder)
    captured = {}
    original_grounding = model.forward_grounding

    def capture_grounding(*arguments, **keywords):
        result = original_grounding(*arguments, **keywords)
        captured["raw"] = result
        return result

    def controlled_addmm(activation, linear, values):
        # Calling the module covers the functional-linear bypass in fused MLP.
        result = linear(values)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(result, approximate="none")
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(result)
        raise ValueError("unsupported fused MLP activation")

    model.forward_grounding, vitdet.addmm_act = capture_grounding, controlled_addmm
    processor = Sam3Processor(model, device=args.device, confidence_threshold=0.5)
    tokenizer = model.backbone.language_backbone.tokenizer
    args.output.mkdir(parents=True)
    shutil.copyfile(args.dataset, args.output / "dataset.json")
    shutil.copyfile(GATES_PATH, args.output / "gates.json")
    scale_digest = None
    if scales:
        np.savez(args.output / "channel-scales.npz", **scales)
        scale_digest = sha256_file(args.output / "channel-scales.npz")
    results = []
    try:
        with torch.inference_mode(), torch.autocast(device_type=args.device, enabled=False):
            for index, sample in enumerate(samples, 1):
                path = artifact_path(args.input_root, sample["image"])
                if sha256_file(path) != sample["source_sha256"]:
                    raise ValueError("selection image changed")
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                reference, features = {}, None
                for mode in ("original", *args.modes):
                    started = time.monotonic()
                    control.set_mode(mode)
                    gate_mode = ("reparameterized" if mode == "cache-f32" else "w8a8-token") if args.target == "image-cache" else mode
                    state = processor.set_image(image)
                    current_features = state["backbone_out"]["backbone_fpn"]
                    if mode == "original":
                        features = [feature.detach().cpu().numpy() for feature in current_features]
                        errors = []
                    else:
                        if len(current_features) != len(features):
                            raise ValueError("vision feature inventory differs")
                        errors = [feature_error(actual, expected, np) for actual, expected in zip(current_features, features)]
                    destination = args.output / sample["id"] / mode
                    destination.mkdir(parents=True)
                    prompts = []
                    for prompt_index, prompt in enumerate(sample["prompts"]):
                        processor.reset_all_prompts(state)
                        state = processor.set_text_prompt(prompt, state)
                        output = capture_output(state, captured.pop("raw"), prompt, tokenizer, torch)
                        output_file = destination / f"prompt-{prompt_index}.json"
                        write_json(output_file, output)
                        row = {"prompt": prompt, "file": str(output_file.relative_to(args.output)), "sha256": sha256_file(output_file)}
                        if mode == "original":
                            reference[prompt] = output
                        else:
                            row["comparison"] = compare_outputs(reference[prompt], output, gate_mode, gates)
                        prompts.append(row)
                    if any(count != 1 for count in control.calls.values()):
                        raise ValueError("each controlled vision boundary must execute once per image/mode")
                    if gate_mode == "reparameterized":
                        profile = gates["reparameterization"]
                        feature_passed = all((error["relative_l2"] <= profile["vision_feature_relative_l2_max"]
                                              if error["relative_l2"] is not None else error["max_abs"] <= profile["zero_norm_max_abs"])
                                             for error in errors)
                    else:
                        feature_passed = None
                    passed = (feature_passed is not False and all(row["comparison"]["output_screen_passed"] for row in prompts)) if mode != "original" else None
                    row = {"sample_id": sample["id"], "mode": mode, "image_sha256": sample["source_sha256"],
                           ("vision_linear_calls" if args.target == "vision-linear" else "cache_feature_calls"): control.calls.copy(), "vision_feature_errors": errors,
                           "feature_equivalence_passed": feature_passed, "output_screen_passed": passed, "prompts": prompts}
                    if args.target == "image-cache":
                        row["cache_payloads"] = control.accounting.copy()
                    write_json(destination / "result.json", row)
                    results.append(row)
                    print(f"{index}/{len(samples)} {sample['id']} {mode}: screen={passed}, {time.monotonic() - started:.2f}s", flush=True)
                    del state, current_features
                write_json(args.output / "progress.json", {"complete": False, "images_finished": index, "images_requested": len(samples)})
                if sha256_file(path) != sample["source_sha256"]:
                    raise ValueError("selection image changed during evaluation")
        verify_run_artifacts(sources)
        if scales and sha256_file(args.output / "channel-scales.npz") != scale_digest:
            raise ValueError("recorded channel scales changed")
        if sha256_file(args.checkpoint) != checkpoint_digest or sha256_file(args.dataset) != dataset_digest:
            raise ValueError("checkpoint or dataset changed")
        final_calibration, _, _ = calibration_inputs(args.calibration)
        if (final_calibration["calibration_id"] != manifest["calibration_id"]
                or sha256_file(args.calibration / "manifest.json") != calibration_digest):
            raise ValueError("calibration changed during screening")
        load_dataset(args.dataset, args.input_root)
        validate_source(source, runtime_source)
        summary = {mode: {"images": sum(row["mode"] == mode for row in results),
                          "images_output_screen_passed": sum(row["mode"] == mode and row["output_screen_passed"] is True for row in results),
                          "output_screen_passed": all(row["output_screen_passed"] for row in results if row["mode"] == mode)} for mode in args.modes}
        write_json(args.output / "screening.json", {"schema_version": 1, "complete": True,
                   "diagnostic_only": args.diagnostic, "kind": "sam3-vision-runtime-quantization-output-screen",
                   "full_model_qualification": False, "annotated_evaluation_completed": False,
                   "scope": f"Original F32 versus numerical {args.target} variants on selection only; no runtime-performance claim",
                   "quantization_target": args.target,
                   "checkpoint_sha256": checkpoint_digest, "dataset_sha256": dataset_digest,
                   "calibration_id": manifest["calibration_id"], "calibration_manifest_sha256": calibration_digest,
                   "gates_sha256": GATES_SHA256, "alpha": args.alpha if scales else None,
                   "scale_mode": args.scale_mode if scales else None, "layer_family": args.layer_family if scales else None,
                   "selected_layers": selected, "unselected_linear_arithmetic": "original-f32",
                   "channel_scales": {"file": "channel-scales.npz", "sha256": scale_digest} if scales else None,
                   "cache_levels": args.cache_levels if args.target == "image-cache" else None,
                   "arithmetic": ("FPN-only cache F16 or Q8_0; channel blocks of 32 with F16 scales; native CUDA fast-math or CPU AVX2 encoder; F32 decode before consumers"
                                  if args.target == "image-cache" else "Symmetric per-output W8 / per-token A8; F32 division and nearest-even; torch._int_mm INT32 dots for W8A8, then F32 scales/bias"),
                   "oracle": {**oracle, "adaptations": adaptations, "unfused_f32": adaptation,
                              "mlp_override": inspect.getsource(controlled_addmm), "device": args.device,
                              "precision": "float32", "autocast": False, "tf32": False, "compile": False,
                              "torch": torch.__version__, "numpy": np.__version__, "pillow": pillow_version},
                   "source_sha256": sources, "summary": summary, "results": results})
        write_json(args.output / "progress.json", {"complete": True, "images_finished": len(samples), "images_requested": len(samples)})
        print(f"Completed output screening: {args.output}", flush=True)
    finally:
        control.close()
        vitdet.addmm_act = original_addmm
        model.forward_grounding = original_grounding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "calibration", "dataset", "input-root", "sam3-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sam3-runtime-source", type=Path)
    parser.add_argument("--bpe", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--target", choices=("vision-linear", "image-cache"), default="vision-linear")
    parser.add_argument("--cache-levels", nargs="+", type=int, choices=range(3), default=[0, 1, 2])
    parser.add_argument("--cache-bridge", type=Path)
    parser.add_argument("--cache-preflight", type=Path)
    parser.add_argument("--scale-mode", choices=SCALE_MODES, default="calibrated")
    parser.add_argument("--layer-family", choices=LAYER_FAMILIES, default="all")
    parser.add_argument("--blocks", nargs="+", type=int)
    parser.add_argument("--modes", nargs="+", choices=(*MODES[1:], *CACHE_MODES[1:]))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-ids", nargs="+")
    args = parser.parse_args()
    allowed_modes = MODES[1:] if args.target == "vision-linear" else CACHE_MODES[1:]
    args.modes = args.modes or list(allowed_modes)
    if not set(args.modes) <= set(allowed_modes) or len(set(args.cache_levels)) != len(args.cache_levels):
        parser.error("modes must belong to the selected target and cache levels must be unique")
    if not 0 <= args.alpha <= 1 or args.threads <= 0 or (args.limit is not None and args.limit <= 0) or len(set(args.modes)) != len(args.modes):
        parser.error("alpha, threads, limit or modes are invalid")
    try:
        screen(args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        parser.exit(1, f"runtime quantization screening failed: {error}\n")


if __name__ == "__main__":
    main()
