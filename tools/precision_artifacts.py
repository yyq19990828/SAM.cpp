"""Shared v2 input/export identities and campaign boundaries."""

from importlib.metadata import version
from pathlib import Path
import csv
import io
import os
import platform
import re
import shutil
import subprocess

from precision_acceptance import GATES_PATH, GATES_SHA256, canonical_hash, load_gates, quality_profile
from prepare_coco_acceptance import load_precision_dataset
from sam3_artifacts import artifact_path, read_json, sha256_file, verify_run_artifacts


def runtime_environment(backend):
    """Bind the host and visible device inventory, without volatile utilization."""
    value = {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
             "cpu": platform.processor(), "cpu_count": os.cpu_count()}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        value["cpu"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                             if line.startswith("model name")), value["cpu"])
    if backend == "cuda":
        fields = ("index", "uuid", "name", "driver_version", "compute_cap", "memory.total")
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                                check=True, text=True, capture_output=True)
        rows = list(csv.reader(io.StringIO(result.stdout)))
        if not rows or any(len(row) != len(fields) for row in rows):
            raise ValueError("cannot establish the CUDA device inventory")
        value["cuda_devices"] = [dict(zip(fields, [field.strip() for field in row])) for row in rows]
        value["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return value


def phase_samples(dataset, phase):
    if phase == "development":
        return [row for row in dataset["samples"] if row["split"] in ("calibration", "development")]
    if phase == "evaluation":
        return [row for row in dataset["samples"] if row["split"] == "evaluation"]
    raise ValueError("reserve images cannot be used for inference in this campaign")


def load_inputs(directory, dataset_path, phase):
    dataset = load_precision_dataset(dataset_path)
    value = read_json(directory / "manifest.json")
    if (value.get("schema_version") != 2 or value.get("kind") != "sam3-precision-inputs-v2"
            or value.get("complete") is not True or value.get("phase") != phase
            or value.get("dataset_sha256") != sha256_file(dataset_path)):
        raise ValueError("normalized inputs differ from the requested dataset/phase")
    samples = phase_samples(dataset, phase)
    if [row["sample_id"] for row in value["images"]] != [row["id"] for row in samples]:
        raise ValueError("normalized input inventory is incomplete or reordered")
    for sample, row in zip(samples, value["images"]):
        if row["source_sha256"] != sample["source_sha256"] or sha256_file(artifact_path(directory, row["file"])) != row["sha256"]:
            raise ValueError("normalized image/source identity changed")
    return dataset, samples, {row["sample_id"]: row for row in value["images"]}


def repository_root():
    return Path(__file__).resolve().parents[1]


def source_snapshot():
    # Freeze shared contracts as well as the entry point. Future mutations are
    # allowed only before a new campaign, not while its batches are running.
    # Compiled src/ implementations and their build files belong to the same
    # identity as the remaining tools and headers.
    root = repository_root()
    paths = {GATES_PATH, root / "CMakeLists.txt", root / "examples/CMakeLists.txt", root / "tests/CMakeLists.txt"}
    for pattern in ("tools/*.py", "tools/*.cpp", "tools/*.hpp", "tools/*.cu", "tools/*.json", "include/**/*.hpp",
                    "src/**/*.cpp", "src/**/*.hpp", "src/**/*.cu", "src/**/CMakeLists.txt",
                    "examples/*.cpp", "examples/*.hpp", "tests/data/*.json", "cmake/**/*.cmake",
                    "cmake/**/*.patch", "tools/requirements*.lock"):
        paths.update(root.glob(pattern))
    return {str(path.resolve()): sha256_file(path) for path in sorted(paths)}


def archive_sources(directory, identities):
    root = repository_root()
    archived = {}
    for name, expected in identities.items():
        path = Path(name)
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.parts[0] not in ("include", "src", "tools", "cmake", "tests", "examples") and relative.as_posix() != "CMakeLists.txt":
            continue
        if path.suffix not in (".py", ".hpp", ".cpp", ".cu", ".cmake", ".patch", ".json", ".txt", ".lock"):
            continue
        destination = directory / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if sha256_file(destination) != expected:
            raise ValueError("producer source changed while archiving")
        archived[name] = destination.relative_to(directory).as_posix()
    return archived


def verify_export_artifacts(directory, manifest):
    for name, digest in manifest["artifact_sha256"].items():
        archived = manifest.get("archived_sources", {}).get(name)
        path = artifact_path(directory, archived) if archived is not None else Path(name)
        if sha256_file(path) != digest:
            raise ValueError(f"export artifact changed: {name}")


def native_snapshot(binary, model):
    manifest = model.with_suffix(model.suffix + ".manifest.json")
    paths = {binary.resolve(), model.resolve(), manifest.resolve()}
    linked = subprocess.run(["ldd", str(binary)], check=True, text=True, capture_output=True).stdout
    if "not found" in linked:
        raise ValueError("native export has unresolved linked libraries")
    for line in linked.splitlines():
        match = re.search(r"(?:=>\s+)?(/\S+)\s+\(", line)
        if match:
            paths.add(Path(match.group(1)))
    return {str(path): sha256_file(path) for path in sorted(paths)}


def native_recipe(binary, model, backend, compute, cache):
    manifest = read_json(model.with_suffix(model.suffix + ".manifest.json"))
    if manifest.get("task") != "image":
        raise ValueError("precision export requires a converted image model")
    precision = manifest["precision"]
    storage = manifest.get("storage_profile") or "dense"
    modules = manifest.get("quantization_modules", ["vision"] if storage.startswith("image-vision-") else [])
    recipe = {"schema_version": 2, "task": "image", "engine": "native", "backend": backend,
              "weight_precision": precision, "storage_profile": storage, "quantization_modules": modules,
              "compute_mode": compute, "feature_cache": cache, "threads": 4,
              "environment": runtime_environment(backend),
              "checkpoint_sha256": manifest["checkpoint"]["sha256"], "model_sha256": sha256_file(model),
              "binary_sha256": sha256_file(binary), "conversion_manifest_sha256": sha256_file(model.with_suffix(model.suffix + ".manifest.json"))}
    if recipe["model_sha256"] != manifest["output"]["sha256"]:
        raise ValueError("model bytes differ from their original conversion manifest")
    quality_profile(recipe, load_gates())
    return recipe


def campaign_check(path, dataset_path, recipe, phase, partial=False):
    if phase != "evaluation":
        if path is not None:
            raise ValueError("final campaign files cannot be used for development selection")
        return None
    if path is None or partial:
        raise ValueError("final evaluation requires a frozen campaign and the complete split")
    campaign = read_json(path)
    if (campaign.get("schema_version") != 2 or campaign.get("kind") != "sam3-precision-campaign-v2"
            or campaign.get("frozen_before_evaluation") is not True or campaign.get("gates_sha256") != GATES_SHA256
            or campaign.get("dataset_sha256") != sha256_file(dataset_path)):
        raise ValueError("invalid frozen precision campaign")
    if canonical_hash(recipe) not in campaign.get("recipe_sha256", []):
        raise ValueError("recipe was not frozen in the final candidate list")
    verify_run_artifacts(campaign["artifact_sha256"])
    current = source_snapshot()
    if any(campaign["artifact_sha256"].get(path) != digest for path, digest in current.items()):
        raise ValueError("campaign did not freeze every current evaluator/export source")
    return sha256_file(path)


def claim_evaluation(path, recipe, output):
    """Record an exclusive recipe attempt before inference; retain failed attempts."""
    ledger = path.parent / ".precision-attempts" / sha256_file(path)
    ledger.mkdir(parents=True, exist_ok=True)
    claim = ledger / (canonical_hash(recipe) + ".json")
    import json
    from datetime import datetime, timezone
    with claim.open("x") as stream:
        json.dump({"campaign_sha256": sha256_file(path), "recipe_sha256": canonical_hash(recipe),
                   "output": str(output.resolve()), "started_at": datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        stream.write("\n")
    return claim


def packages():
    return {name: version(name) for name in ("numpy", "pillow", "pycocotools", "torch")}
