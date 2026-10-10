"""Shared original/native ranked exporters; no quality thresholds or campaigns."""

import os
import subprocess
import sys
import time

from tools.validation.ranked_outputs import required_queries, validate_output
from tools.maintenance.artifact_snapshot import packages, runtime_environment
from tools.quantize.runtime_quantization import encode_mask
from tools.convert.sam3_artifacts import BPE_SHA256, artifact_path, read_json, sha256_file, write_json
from tools.quantize.quantization_config import resolve_model_configuration


def capture_ranked(state, raw, prompt, tokenizer, torch):
    from sam3.model.box_ops import box_cxcywh_to_xyxy
    from sam3.model.data_misc import interpolate
    width, height = state["original_width"], state["original_height"]
    scores = (raw["pred_logits"].sigmoid() * raw["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
    if (scores.shape != (1, 200) or raw["pred_boxes"].shape != (1, 200, 4)
            or raw["pred_masks"].ndim != 4 or raw["pred_masks"].shape[:2] != (1, 200)
            or not torch.isfinite(raw["pred_masks"]).all()):
        raise ValueError("original ranked query shape or finiteness differs")
    boxes = box_cxcywh_to_xyxy(raw["pred_boxes"])[0] * torch.tensor([width, height, width, height], device=scores.device)
    score_values = scores[0].cpu().tolist()
    ranked, required = required_queries(score_values)
    masks = []
    for start in range(0, len(required), 16):
        indices = required[start:start + 16]
        logits = raw["pred_masks"][0, indices].unsqueeze(1)
        binary = (interpolate(logits, (height, width), mode="bilinear", align_corners=False).sigmoid() > 0.5).cpu().numpy()
        masks.extend({"query_index": query, "mask": encode_mask(binary[index, 0])} for index, query in enumerate(indices))
    value = {"schema_version": 2, "width": width, "height": height, "prompt": prompt,
             "token_ids": tokenizer([prompt], context_length=32)[0].tolist(), "query_scores": score_values,
             "query_boxes": boxes.cpu().tolist(), "ranked_queries": ranked, "masks": masks}
    validate_output(value)
    return value

def oracle_recipe(args):
    from tools.validation.export_reference import validate_source
    source, runtime_source, adaptations = validate_source(args.sam3_source, args.sam3_runtime_source)
    if sha256_file(args.bpe) != BPE_SHA256:
        raise ValueError("original BPE identity differs")
    return {"schema_version": 2, "task": "image", "engine": "original", "backend": "cuda",
            "weight_precision": "f32", "storage_profile": "dense", "quantization_modules": [],
            "compute_mode": "f32", "feature_cache": "f32", "threads": 4,
            "activation": "backend-selected",
            "quantization_configuration": resolve_model_configuration(args, {"precision": "f32"}),
            "environment": runtime_environment("cuda"),
            "checkpoint_sha256": sha256_file(args.checkpoint), "bpe_sha256": BPE_SHA256,
            "reference_kind": "official-checkpoint", "sam3_source": str(source), "runtime_source": str(runtime_source),
            "source_adaptations": adaptations, "packages": packages()}

def export_original(args, samples, input_rows, identities, recipe):
    import torch
    from PIL import Image
    from tools.validation.export_reference import configure_cuda_oracle, install_unfused_fp32, load_detector_checkpoint, validate_source
    source, runtime_source, _ = validate_source(args.sam3_source, args.sam3_runtime_source)
    sys.path.insert(0, str(runtime_source))
    os.environ["HF_HUB_OFFLINE"], os.environ["USE_PERFLIB"] = "1", "0"
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import sam3.model.vitdet as vitdet
    torch.set_num_threads(4)
    torch.manual_seed(20261007)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    oracle = configure_cuda_oracle(torch, "cuda")
    adaptation = install_unfused_fp32()
    model = build_sam3_image_model(bpe_path=str(args.bpe), device="cuda", eval_mode=True,
                                  checkpoint_path=None, load_from_HF=False, enable_inst_interactivity=False, compile=False)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(weights, dict) and isinstance(weights.get("model"), dict):
        weights = weights["model"]
    load_detector_checkpoint(model, weights)
    del weights
    model = model.float().to(device="cuda").eval()
    captured = {}
    original_grounding, original_addmm = model.forward_grounding, vitdet.addmm_act

    def capture(*arguments, **keywords):
        value = original_grounding(*arguments, **keywords)
        captured["raw"] = value
        return value

    def controlled_addmm(activation, linear, values):
        result = linear(values)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(result, approximate="none")
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(result)
        raise ValueError("unsupported original MLP activation")

    model.forward_grounding, vitdet.addmm_act = capture, controlled_addmm
    processor = Sam3Processor(model, device="cuda", confidence_threshold=1.0)
    tokenizer = model.backbone.language_backbone.tokenizer
    rows = []
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            for index, sample in enumerate(samples, 1):
                started = time.monotonic()
                path = artifact_path(args.inputs, input_rows[sample["id"]]["file"])
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                state = processor.set_image(image)
                for prompt_index, prompt in enumerate(sample["prompts"]):
                    processor.reset_all_prompts(state)
                    state = processor.set_text_prompt(prompt, state)
                    payload = capture_ranked(state, captured.pop("raw"), prompt, tokenizer, torch)
                    name = f"{sample['id']}-p{prompt_index}.json"
                    write_json(args.output / name, payload)
                    rows.append({"sample_id": sample["id"], "prompt_index": prompt_index, "prompt": prompt,
                                 "file": name, "sha256": sha256_file(args.output / name)})
                del state
                write_json(args.output / "progress.json", {"complete": False, "images_finished": index, "images_requested": len(samples)})
                print(f"original {index}/{len(samples)} {sample['id']} {time.monotonic() - started:.2f}s", flush=True)
    finally:
        model.forward_grounding, vitdet.addmm_act = original_grounding, original_addmm
    validate_source(source, runtime_source)
    return rows, {"oracle": {**oracle, "precision": "float32", "tf32": False, "autocast": False, "compile": False,
                             "fp32_adaptation": adaptation}, "reference_kind": "official-checkpoint"}

def export_native(args, samples, input_rows, identities, recipe):
    from tools.benchmark.precision_reporting import expected_runtime_profile, native_weight_policy_matches
    table = args.output / "cases.tsv"
    entries = []
    for sample in samples:
        image = artifact_path(args.inputs, input_rows[sample["id"]]["file"])
        for index, prompt in enumerate(sample["prompts"]):
            if any(character in prompt or character in str(image) for character in ("\t", "\n", "\r")):
                raise ValueError("native TSV cannot represent this prompt/path")
            entries.append(f"{sample['id']}-p{index}\t{image}\t{prompt}\n")
    table.write_text("".join(entries))
    identities[str(table.resolve())] = sha256_file(table)
    raw = args.output / "native"
    with (args.output / "native.log").open("x") as log:
        command = ["rtk", "proxy", str(args.binary.resolve()), str(args.model.resolve()), str(table.resolve()),
                   args.backend, args.cache, args.compute, str(raw.resolve())]
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = read_json(raw / "run.json")
    if (receipt.get("complete") is not True or receipt.get("cases") != len(entries)
            or receipt.get("backend") != args.backend or receipt.get("feature_cache") != args.cache
            or receipt.get("cuda_compute") != args.compute):
        raise ValueError("native export did not complete the declared recipe")
    identities[str((raw / "run.json").resolve())] = sha256_file(raw / "run.json")
    rows, device_names, arithmetic = [], set(), set()
    for sample in samples:
        for index, prompt in enumerate(sample["prompts"]):
            path = raw / f"{sample['id']}-p{index}.json"
            value = read_json(path)
            validate_output(value)
            if (value["prompt"] != prompt or value.get("precision") != recipe["weight_precision"]
                    or (value.get("storage_profile") or "dense") != recipe["storage_profile"]
                    or value.get("quantization_modules") != recipe["quantization_modules"]
                    or not native_weight_policy_matches(value, recipe)
                    or value.get("arithmetic_profile") != expected_runtime_profile(recipe)
                    or (args.backend == "cuda" and value.get("cuda_device") != recipe.get("cuda_device", 0))
                    or value.get("feature_cache") != args.cache or value.get("cuda_compute") != args.compute):
                raise ValueError("native payload model/compute/cache identity differs")
            stats = value["runtime"]
            if args.backend == "cuda" and (stats.get("cuda_nodes", 0) <= 0 or any(stats.get(key, 0) for key in ("cpu_nodes", "metal_nodes", "blas_nodes"))):
                raise ValueError("native export did not retain strict CUDA placement")
            device_names.add(value["device_name"])
            arithmetic.add(value["arithmetic_profile"])
            rows.append({"sample_id": sample["id"], "prompt_index": index, "prompt": prompt,
                         "file": str(path.relative_to(args.output)), "sha256": sha256_file(path)})
    if len(device_names) != 1 or len(arithmetic) != 1:
        raise ValueError("native backend identity changed within a run")
    return rows, {"device_name": device_names.pop(), "arithmetic_profile": arithmetic.pop(), "reference_kind": "converted-model"}
