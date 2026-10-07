#!/usr/bin/env python3
"""Export original-FP32 SAM 3 vision-linear calibration inputs and statistics."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from calibration import ChannelStats, layer_seed, load_dataset
from convert_sam3 import rename_key
from export_reference import (configure_cuda_oracle, install_unfused_fp32,
                              load_detector_checkpoint, validate_source)
from sam3_artifacts import (BPE_SHA256, SAM3_REVISION, artifact_path, read_json, sha256_file,
                            verify_run_artifacts, write_json)
from sam3_gguf import quantization_module_for_linear_weight


class VisionCollector:
    def __init__(self, model, torch, seed, sample_rows, chunk_rows):
        self.torch, self.chunk_rows = torch, chunk_rows
        self.names, self.stats, self.modules = {}, {}, {}
        self.handles, self.calls = [], {}
        for module_name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            name, _ = rename_key("detector." + module_name + ".weight")
            if name is None or quantization_module_for_linear_weight(name) != "vision":
                continue
            self.names[id(module)] = name
            self.modules[name] = module
            self.stats[name] = ChannelStats(module.in_features, sample_rows, layer_seed(seed, name))
            self.calls[name] = 0
            self.handles.append(module.register_forward_pre_hook(self._hook))
        schema = read_json(Path(__file__).with_name("sam3_tensor_schema.json"))
        expected = {name for name in schema["tensors"] if quantization_module_for_linear_weight(name) == "vision"}
        if set(self.stats) != expected:
            self.close()
            raise ValueError("reference modules do not cover the canonical vision linear inventory")

    def _hook(self, module, arguments):
        self.observe(module, arguments[0])

    def observe(self, module, values):
        name = self.names.get(id(module))
        if name is None:
            return
        if values.dtype != self.torch.float32 or values.shape[-1] != self.stats[name].channels:
            raise ValueError(f"{name}: calibration requires original F32 inputs on the last channel axis")
        rows = values.detach().reshape(-1, values.shape[-1])
        for offset in range(0, rows.shape[0], self.chunk_rows):
            chunk = rows[offset:offset + self.chunk_rows].to(device="cpu").numpy()
            self.stats[name].update(chunk)
        self.calls[name] += 1

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def export(args):
    import numpy as np
    import torch
    from PIL import Image, __version__ as pillow_version

    if sys.prefix == sys.base_prefix:
        raise RuntimeError("use the isolated reference Python environment")
    if not args.sam3_source:
        raise ValueError("provide the pinned --sam3-source checkout")
    if args.output.exists():
        raise FileExistsError("calibration output already exists")
    if not args.checkpoint.is_file():
        raise FileNotFoundError("original checkpoint is missing")
    source_files = ("export_calibration.py", "calibration.py", "export_reference.py", "convert_sam3.py",
                    "sam3_gguf.py", "sam3_artifacts.py", "sam3_tensor_schema.json")
    source_snapshot = {str(Path(__file__).with_name(name).resolve()): sha256_file(Path(__file__).with_name(name))
                       for name in source_files}
    dataset_hash = sha256_file(args.dataset)
    dataset = load_dataset(args.dataset, args.input_root, args.diagnostic)
    calibration = [sample for sample in dataset["samples"] if sample["split"] == "calibration"]
    if args.limit is not None:
        if not args.diagnostic:
            raise ValueError("--limit requires --diagnostic; partial data cannot produce a calibration profile")
        calibration = calibration[:args.limit]
    source, runtime_source, adaptations = validate_source(args.sam3_source, args.sam3_runtime_source)
    bpe = args.bpe or source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if sha256_file(bpe) != BPE_SHA256:
        raise ValueError("BPE identity differs from the pinned model")
    checkpoint_hash = sha256_file(args.checkpoint)
    sys.path.insert(0, str(runtime_source))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["USE_PERFLIB"] = "0"
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
    fp32_addmm = vitdet.addmm_act
    collector = None
    staging = None
    try:
        model = build_sam3_image_model(bpe_path=str(bpe), device=args.device, eval_mode=True,
                                      checkpoint_path=None, load_from_HF=False,
                                      enable_inst_interactivity=False, compile=False)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        load_detector_checkpoint(model, state)
        del state
        model = model.float().to(device=args.device).eval()
        processor = Sam3Processor(model, device=args.device, confidence_threshold=0.5)
        collector = VisionCollector(model, torch, dataset["seed"], args.sample_rows, args.chunk_rows)

        def observed_addmm(activation, linear, mat1):
            # The official fused MLP calls functional.linear, bypassing the
            # nn.Linear hooks. Observe that input once, before the F32 oracle.
            collector.observe(linear, mat1)
            return fp32_addmm(activation, linear, mat1)

        vitdet.addmm_act = observed_addmm
        counts = []
        with torch.inference_mode(), torch.autocast(device_type=args.device, enabled=False):
            for index, sample in enumerate(calibration, 1):
                started = time.monotonic()
                print(f"Calibrating {index}/{len(calibration)}: {sample['id']}", flush=True)
                before = collector.calls.copy()
                image_path = artifact_path(args.input_root, sample["image"])
                if sha256_file(image_path) != sample["source_sha256"]:
                    raise ValueError("calibration input changed after preflight")
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                state = processor.set_image(image)
                for prompt in sample["prompts"]:
                    processor.reset_all_prompts(state)
                    state = processor.set_text_prompt(prompt, state)
                calls = {name: collector.calls[name] - before[name] for name in collector.calls}
                if any(count != 1 for count in calls.values()):
                    raise ValueError("each vision linear must be observed exactly once per image")
                counts.append({"id": sample["id"], "vision_linear_calls": sum(calls.values()),
                               "prompts": len(sample["prompts"])})
                print(f"Completed {sample['id']} in {time.monotonic() - started:.2f}s", flush=True)
        if sha256_file(args.dataset) != dataset_hash or sha256_file(args.checkpoint) != checkpoint_hash:
            raise ValueError("dataset or checkpoint changed during calibration")
        load_dataset(args.dataset, args.input_root, args.diagnostic)
        validate_source(source, runtime_source)
        verify_run_artifacts(source_snapshot)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".sam3-calibration-", dir=args.output.parent))
        shutil.copyfile(args.dataset, staging / "dataset.json")
        np.savez(staging / "samples.npz", **{name: stats.samples for name, stats in collector.stats.items()})
        layers = {}
        for name, stats in collector.stats.items():
            module = collector.modules[name]
            weights = module.weight.detach().cpu().numpy()
            layers[name] = {"activation": stats.summary(), "calls": collector.calls[name],
                            "input_channel_axis": -1, "weight_layout": "output,input",
                            "weight_input_absmax": np.abs(weights).max(axis=0).tolist(),
                            "weight_output_absmax": np.abs(weights).max(axis=1).tolist()}
        write_json(staging / "statistics.json", {"schema_version": 1, "layers": layers})
        manifest = {"schema_version": 1, "kind": "sam3-vision-activation-calibration",
                    "sam3_revision": SAM3_REVISION, "diagnostic_only": args.diagnostic,
                    "reference_kind": "official-checkpoint", "checkpoint_sha256": checkpoint_hash,
                    "bpe_sha256": BPE_SHA256, "seed": dataset["seed"],
                    "scope": "128 vision linear inputs; no quantized model or quality acceptance implied",
                    "observed_samples": counts, "sample_rows_per_layer": args.sample_rows,
                    "chunk_rows": args.chunk_rows,
                    "statistics": {"file": "statistics.json", "sha256": sha256_file(staging / "statistics.json")},
                    "samples": {"file": "samples.npz", "sha256": sha256_file(staging / "samples.npz"),
                                "dtype": "float32", "layout": "token,channel", "selection": "uniform row priority reservoir"},
                    "dataset": {"file": "dataset.json", "sha256": sha256_file(staging / "dataset.json")},
                    "oracle": {**oracle, "precision": "float32", "autocast": False, "tf32": False,
                               "compile": False, "device": args.device, "torch": torch.__version__,
                               "numpy": np.__version__, "pillow": pillow_version, "python": sys.version,
                               "adaptations": adaptations, "unfused_fp32": adaptation,
                               "observer": inspect.getsource(observed_addmm)},
                    "source_sha256": {Path(path).name: digest for path, digest in source_snapshot.items()}}
        manifest["calibration_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
        write_json(staging / "manifest.json", manifest)
        if args.output.exists():
            raise FileExistsError("calibration output appeared during export")
        staging.rename(args.output)
        staging = None
        print(f"Exported {len(layers)} layers from {len(counts)} images to {args.output}")
    finally:
        if collector is not None:
            collector.close()
        vitdet.addmm_act = original_addmm
        if staging is not None:
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "input-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--sam3-source", type=Path, default=os.environ.get("SAM3_SOURCE_DIR"))
    parser.add_argument("--sam3-runtime-source", type=Path)
    parser.add_argument("--bpe", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--diagnostic", action="store_true", help="Smoke test only; not usable as a calibrated model profile")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if min(args.threads, args.sample_rows, args.chunk_rows, args.limit or 1) <= 0 or args.limit == 0:
        parser.error("threads, sample rows, chunk rows and limit must be positive")
    try:
        export(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"calibration export failed: {error}\n")


if __name__ == "__main__":
    main()
