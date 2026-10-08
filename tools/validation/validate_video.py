#!/usr/bin/env python3
"""Enforce original-weight M2 video gates, fixed birth-ID mapping and traces."""

import argparse
import math
from pathlib import Path
import sys
import subprocess

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.validation.export_video_reference import CASES, OFFICIAL_SHA256
from tools.convert.sam3_artifacts import (BPE_SHA256, SAM3_REVISION, artifact_path, read_array, read_json, sha256_file, write_json,
                           freeze_run_artifacts, verify_run_artifacts, freeze_output_files, verify_output_files,
                           validate_cuda_oracle_provenance)
from tools.convert.sam3_gguf import HYBRID_PROFILE, inspect_tensors, read_gguf, tensor_schema, validate_metadata
from tools.validation.validate_image import CUDA_F16_ARITHMETIC_PROFILE, mask_iou, tensor_error, validate_cuda_compute_mode

MODEL_HASHES = {
    "f32": "02513232afca5ba8c174b66c7fc839c67b32590df4b53bdd6df5089a6a546844",
    "f16": "9c9bc86c81d11a041db10a46d3d1e8ecaa1cbcf6fad683b00901f641746bf32c",
    "hybrid": "3975b4b1a10b962c6fad5022e2798baa6266aad7dbcc39210b459537208e1a05",
}


def check_provenance(model_path, reference_path, allow_diagnostic=False):
    reference = read_json(reference_path / "manifest.json")
    frozen = read_json(CASES)
    oracle = reference.get("oracle", {})
    validate_cuda_oracle_provenance(oracle)
    if (reference.get("schema_version") != 1 or reference.get("task") != "text_video"
            or reference.get("complete") is not True or reference.get("sam3_revision") != SAM3_REVISION
            or reference.get("reference_kind") != "official-checkpoint"
            or reference.get("checkpoint", {}).get("sha256") != OFFICIAL_SHA256
            or reference.get("bpe", {}).get("sha256") != BPE_SHA256):
        raise ValueError("reference is incomplete or not the pinned original video oracle")
    if (oracle.get("variant") != "official-video-fp32-explicit-storage" or oracle.get("precision") != "float32"
            or any(oracle.get(key) is not False for key in ("compile", "autocast", "tf32"))
            or oracle.get("input_storage") != "float16" or oracle.get("tracker_transport") != "bfloat16"
            or oracle.get("memory_storage") != "bfloat16" or not oracle.get("packages")
            or not oracle.get("runtime_adaptations") or not oracle.get("fp32_adaptation")):
        raise ValueError("video oracle precision/storage adaptations are unrecorded")
    definition = artifact_path(reference_path, reference["cases_manifest"]["file"])
    if sha256_file(definition) != reference["cases_manifest"]["sha256"] or read_json(definition) != frozen:
        raise ValueError("video reference recipes/tolerances differ from the frozen corpus")
    cases = reference.get("cases", [])
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("video reference case IDs are empty or duplicated")
    expected = {case["id"]: case for case in frozen["cases"]}
    if not allow_diagnostic:
        if reference.get("eligible_for_milestone") is not True or set(case["id"] for case in cases) != set(expected):
            raise ValueError("complete video acceptance requires every verified official case")
    for case in cases:
        if case["id"] not in expected or case.get("prompt") != expected[case["id"]]["prompt"]:
            raise ValueError("video reference contains an unknown case or prompt")
        if (type(case.get("frames")) is not int or not 1 <= case["frames"] <= expected[case["id"]]["frames"]
                or any(type(case.get(key)) is not int or case[key] <= 0 for key in ("width", "height"))
                or not isinstance(case.get("files"), dict) or not case["files"]):
            raise ValueError("video reference dimensions/frame count/file inventory is invalid")
        if not allow_diagnostic and (case["frames"] != expected[case["id"]]["frames"]
                                     or case["reference_behavior_verified"] is not True):
            raise ValueError("required video reference behavior is missing")
        directory = artifact_path(reference_path, case["directory"])
        for filename, digest in case["files"].items():
            if sha256_file(artifact_path(directory, filename)) != digest:
                raise ValueError(f"{case['id']}: reference file hash mismatch: {filename}")
    model = read_json(model_path.with_suffix(model_path.suffix + ".manifest.json"))
    precision = model.get("precision")
    if precision == "hybrid" and model.get("storage_profile") != HYBRID_PROFILE:
        raise ValueError("hybrid model storage profile is missing or incompatible")
    if (model.get("sam_schema_version") != 2 or model.get("task") != "video" or precision not in MODEL_HASHES
            or model.get("sam3_revision") != SAM3_REVISION
            or model.get("checkpoint", {}).get("sha256") != OFFICIAL_SHA256
            or model.get("bpe", {}).get("sha256") != BPE_SHA256
            or model.get("container_format") != "gguf" or model.get("container_version") != 3):
        raise ValueError("model is not the pinned full SAM 3 video GGUF")
    digest = sha256_file(model_path)
    if digest != MODEL_HASHES[precision] or digest != model["output"]["sha256"] or model_path.stat().st_size != model["output"]["bytes"]:
        raise ValueError("full model identity differs from the verified original conversion")
    reader = read_gguf(model_path)
    validate_metadata(reader, precision, OFFICIAL_SHA256, "video")
    expected_tensors = tensor_schema(read_json(Path(__file__).resolve().parents[2] / "tools/convert/sam3_tensor_schema.json"), "video")
    recorded = {tensor["name"]: tensor for tensor in model["tensors"]}
    if len(recorded) != len(model["tensors"]) or set(recorded) != set(expected_tensors):
        raise ValueError("full model sidecar inventory is duplicated or incomplete")
    actual = inspect_tensors(reader, precision, expected_tensors, {name: value["shape"] for name, value in recorded.items()})
    for tensor in actual:
        if any(tensor[key] != recorded[tensor["name"]].get(key)
               for key in ("shape", "ggml_shape", "dtype", "offset", "bytes", "sha256")):
            raise ValueError("full model tensor payload/sidecar differs")
    return precision, reference, frozen["acceptance"]


def read_objects(directory, frame, width, height, require_hash):
    result = read_json(directory / "results.json")
    if (result.get("schema_version") != 1 or result.get("frame_index") != frame
            or not isinstance(result.get("objects"), list)):
        raise ValueError("video frame result has incorrect schema/index")
    objects = {}
    for obj in result.get("objects", []):
        identifier, score, box = obj.get("id"), obj.get("score"), obj.get("box")
        if (type(identifier) is not int or identifier < 0 or identifier in objects
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                or not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or box[0] > box[2] or box[1] > box[3] or obj["mask"]["shape"] != [height, width]):
            raise ValueError("video object identity/score/box/mask is malformed")
        read_array(directory, obj["mask"], "uint8", require_hash)
        objects[identifier] = obj
    return result, objects


def compare_case(reference, actual, case, precision, backend, gates, max_objects, cuda_device=0, cuda_compute="f32"):
    import numpy as np

    gate_precision = "f16" if precision == "hybrid" else precision
    validate_cuda_compute_mode(backend, cuda_compute)
    fast_compute = cuda_compute == "f16"
    output_gate_precision = "f16" if fast_compute else gate_precision
    manifest = read_json(actual / "manifest.json")
    expected_arithmetic = CUDA_F16_ARITHMETIC_PROFILE if fast_compute else ""
    if manifest.get("arithmetic_profile", "") != expected_arithmetic:
        raise ValueError("C++ video arithmetic profile differs from the requested compute mode")
    if manifest.get("storage_profile", "") != (HYBRID_PROFILE if precision == "hybrid" else ""):
        raise ValueError("C++ video storage profile differs from the converted artifact")
    if (manifest.get("complete") is not True or manifest.get("task") != "text_video"
            or manifest.get("frame_count") != case["frames"] or manifest.get("precision") != precision
            or manifest.get("backend") != backend or manifest.get("max_objects") != max_objects
            or manifest.get("prompt") != case["prompt"] or manifest.get("token_ids") != case["token_ids"]
            or manifest.get("width") != case["width"] or manifest.get("height") != case["height"]):
        raise ValueError("C++ video run is incomplete or has different inputs/backend/tokens")
    stats = manifest["stats"]
    if backend == "cuda":
        if (stats["runtime"].get("cuda_nodes", 0) <= 0 or
                any(stats["runtime"].get(f"{other}_nodes", 0) for other in ("cpu", "metal", "blas")) or
                manifest.get("cuda_device") != cuda_device or not manifest.get("device_name")):
            raise ValueError("video CUDA graph has absent selected-GPU work or compute fallback")
    if stats["accepted_frames"] != case["frames"] or stats["emitted_frames"] != case["frames"] or stats["pending_frames"] != 0:
        raise ValueError("C++ video did not accept/emit/drain every declared frame")
    if stats["pending_high_water"] > 15 or stats["retained_records"] > 27 * stats["active_objects"]:
        raise ValueError("video output/state retention bound exceeded")
    if backend == "metal" and (stats["runtime"]["metal_nodes"] <= 0 or stats["runtime"]["cpu_nodes"] != 0):
        raise ValueError("video Metal graph has absent GPU work or CPU fallback")
    mapping, failures, tensors, objects, tensor_failures, candidate_diagnostics = {}, [], {}, [], [], []
    for frame in range(case["frames"]):
        expected_trace = read_json(reference / "trace" / f"{frame:06d}.json")
        actual_trace = read_json(actual / "trace" / f"{frame:06d}.json")
        if actual_trace.get("frame_index") != frame:
            raise ValueError("video trace frame order differs")
        a_births, r_births = actual_trace["births"], expected_trace["births"]
        if len(a_births) != len(r_births):
            raise ValueError(f"frame {frame}: birth count differs; fixed ID mapping cannot be established")
        for a, r in zip(a_births, r_births):
            if a["id"] in mapping or r["id"] in mapping.values():
                raise ValueError("video recycled or duplicated an object ID")
            mapping[a["id"]] = r["id"]
        # Mapping is fixed once at birth and never rematched by masks per frame.
        if sorted(mapping[value] for value in actual_trace["removed"]) != sorted(expected_trace["removed"]):
            failures.append(f"frame {frame}: retirement/lifecycle differs")
        normalized_groups = [{**group, "ids": [mapping[value] for value in group["ids"]]} for group in actual_trace["groups"]]
        if normalized_groups != expected_trace["groups"]:
            failures.append(f"frame {frame}: logical groups, selected memory or pointer order differs")
        if "propagation" in expected_trace:
            if fast_compute:
                for item in actual_trace.get("propagation", []):
                    scores = item.get("iou")
                    if (not isinstance(scores, list) or len(scores) != 4
                            or any(type(value) not in (int, float) or not math.isfinite(value) for value in scores)
                            or any(type(item.get(key)) is not int or item[key] not in (1, 2, 3)
                                   for key in ("mask_index", "pointer_index"))):
                        raise ValueError(f"frame {frame}: invalid propagation candidate scores/indices")
                    best = max(range(1, 4), key=lambda index: scores[index])
                    if item["mask_index"] != best or item["pointer_index"] != best:
                        failures.append(f"frame {frame}: mask/pointer selection disagrees with candidate IoU ranking")
            chosen = [{"id": mapping[item["id"]], "mask_index": item["mask_index"], "pointer_index": item["pointer_index"]}
                      for item in actual_trace.get("propagation", [])]
            expected_chosen = [{key: item[key] for key in ("id", "mask_index", "pointer_index")}
                               for item in expected_trace["propagation"]]
            if chosen != expected_chosen:
                if fast_compute:
                    candidate_diagnostics.append({"frame": frame, "actual": chosen, "reference": expected_chosen})
                    if [item["id"] for item in chosen] != [item["id"] for item in expected_chosen]:
                        failures.append(f"frame {frame}: propagation object IDs/order differ")
                else:
                    failures.append(f"frame {frame}: propagated mask/pointer candidate differs")
        a_directory, r_directory = actual / f"{frame:06d}", reference / f"{frame:06d}"
        a_result, a_objects = read_objects(a_directory, frame, case["width"], case["height"], False)
        r_result, r_objects = read_objects(r_directory, frame, case["width"], case["height"], True)
        if a_result["emitted_after_frame"] != r_result["emitted_after_frame"]:
            failures.append(f"frame {frame}: output delay/drain differs")
        normalized = {mapping[identifier]: obj for identifier, obj in a_objects.items()}
        if set(normalized) != set(r_objects):
            failures.append(f"frame {frame}: visible object IDs differ")
        for identifier, r_obj in r_objects.items():
            if identifier not in normalized:
                continue
            a_obj = normalized[identifier]
            iou = mask_iou(read_array(a_directory, a_obj["mask"], "uint8", False),
                           read_array(r_directory, r_obj["mask"], "uint8", True))
            score_error = abs(a_obj["score"] - r_obj["score"])
            box_error = float(np.max(np.abs(np.array(a_obj["box"]) - r_obj["box"]) /
                                     [case["width"], case["height"], case["width"], case["height"]]))
            passed = (score_error <= gates["score_max_abs"] and box_error <= gates["box_dimension_fraction"]
                      and (r_obj["score"] < gates["high_score_min"] or iou >= gates[f"mask_iou_{output_gate_precision}"]))
            objects.append({"frame": frame, "id": identifier, "mask_iou": iou, "score_error": score_error,
                            "box_dimension_fraction": box_error, "passed": passed,
                            "threshold_adjacent": gates["low_score_max"] < r_obj["score"] < gates["high_score_min"]})
            if not passed:
                failures.append(f"frame {frame}, object {identifier}: mask/score/box gate exceeded")
        sample = read_json(actual / "trace" / f"{frame:06d}-stats.json")
        if sample["retained_records"] > 27 * sample["active_objects"] or sample["pending_frames"] > 14:
            failures.append(f"frame {frame}: retained state or delayed-output bound exceeded")
        if backend == "metal" and sample["runtime"]["cpu_nodes"] != 0:
            failures.append(f"frame {frame}: CPU graph fallback")
        if backend == "cuda" and (sample["runtime"].get("cuda_nodes", 0) <= 0 or
                any(sample["runtime"].get(f"{other}_nodes", 0) for other in ("cpu", "metal", "blas"))):
            failures.append(f"frame {frame}: missing CUDA compute or graph fallback")
        if frame not in (0, 1, 16, case["frames"] - 1):
            continue
        a_path, r_path = actual / "tensors" / f"{frame:06d}", reference / "tensors" / f"{frame:06d}"
        a_index, r_index = read_json(a_path / "tensors.json"), read_json(r_path / "tensors.json")
        for index in (a_index, r_index):
            if (index.get("schema_version") != 1 or index.get("byte_order") != "little"
                    or not isinstance(index.get("tensors"), dict) or not index["tensors"]):
                raise ValueError("video tensor dump has incorrect schema/byte order or no stages")
        normalized_tensors = {}
        for name, metadata in a_index["tensors"].items():
            if name.startswith("object."):
                _, identifier, suffix = name.split(".", 2)
                name = f"object.{mapping[int(identifier)]}.{suffix}"
            normalized_tensors[name] = metadata
        if set(normalized_tensors) != set(r_index["tensors"]):
            raise ValueError(f"frame {frame}: stage tensor inventory differs")
        for name, metadata in r_index["tensors"].items():
            metrics = tensor_error(read_array(a_path, normalized_tensors[name], "float32", False),
                                   read_array(r_path, metadata, "float32", True))
            if name == "preprocessed_image":
                passed = metrics["maximum_absolute_error"] <= gates["preprocessed_max_abs"]
            elif metrics["normalized_l2"] is None:
                passed = metrics["maximum_absolute_error"] <= gates["zero_norm_max_abs"]
            else:
                passed = metrics["normalized_l2"] <= gates[f"normalized_l2_{gate_precision}"]
            tensors[f"{frame}:{name}"] = {**metrics, "passed": passed}
            if not passed:
                failure = f"frame {frame}, {name}: tensor tolerance exceeded"
                tensor_failures.append(failure)
                if not fast_compute or name == "preprocessed_image":
                    failures.append(failure)
    return {"passed": not failures, "id": case["id"], "failures": failures, "tensors": tensors,
            "objects": objects, "id_mapping": mapping, "stats": stats,
            **({"cuda_compute": cuda_compute, "runtime_arithmetic_profile": expected_arithmetic,
                "output_quality_passed": not failures, "tensor_fidelity_passed": not tensor_failures,
                "tensor_fidelity_failures": tensor_failures, "tensor_fidelity_is_release_gate": False,
                "internal_candidate_diagnostics": candidate_diagnostics,
                "internal_candidate_equivalence_passed": not candidate_diagnostics,
                "internal_candidate_equivalence_is_release_gate": False,
                "output_gate_precision": "f16"} if fast_compute else {})}


def validate(args):
    cuda_compute = getattr(args, "cuda_compute", None) or "f32"
    validate_cuda_compute_mode(args.backend, cuda_compute)
    precision, reference, gates = check_provenance(args.model, args.reference, args.allow_diagnostic)
    executable = args.build_dir / "examples/sam_video"
    if not executable.is_file():
        raise FileNotFoundError("sam_video executable is missing")
    args.output.mkdir(parents=True, exist_ok=False)
    binary_hash = sha256_file(executable)
    artifacts = freeze_run_artifacts(args.build_dir, executable, args.model, MODEL_HASHES[precision])
    report = {"schema_version": 1, "passed": False, "task": "text_video", "backend": args.backend,
              "precision": precision, "eligible_for_milestone": reference["eligible_for_milestone"] and not args.allow_diagnostic,
              "model_sha256": MODEL_HASHES[precision], "checkpoint_sha256": OFFICIAL_SHA256,
              "reference_manifest_sha256": sha256_file(args.reference / "manifest.json"),
              "binary_sha256": binary_hash, "run_artifact_sha256": artifacts, "gates": gates, "cases": []}
    if args.backend == "cuda":
        report["cuda_device"] = getattr(args, "cuda_device", None) or 0
        report["cuda_compute"] = cuda_compute
    if cuda_compute == "f16":
        report.update(acceptance_mode="cuda-f16-output-quality-v1", tensor_fidelity_is_release_gate=False,
                      internal_candidate_equivalence_is_release_gate=False,
                      output_gate_precision="f16", runtime_arithmetic_profile=CUDA_F16_ARITHMETIC_PROFILE)
    for case in reference["cases"]:
        directory = artifact_path(args.reference, case["directory"])
        actual = args.output / case["id"]
        command = [str(executable.resolve()), "--model", str(args.model.resolve()), "--frames", str(directory / "inputs"),
                   "--text", case["prompt"], "--backend", args.backend, "--threads", str(args.threads),
                   "--max-objects", str(reference["max_objects"]), "--dump-tensors", "--output", str(actual)]
        if args.backend == "cuda":
            command.extend(["--cuda-device", str(getattr(args, "cuda_device", None) or 0)])
            command.extend(["--cuda-compute", cuda_compute])
        try:
            verify_run_artifacts(artifacts)
            print(f"Validating {case['id']} ({args.backend}, {precision})", flush=True)
            with (args.output / f"{case['id']}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
            if result.returncode:
                raise RuntimeError(f"sam_video exited {result.returncode}; see {case['id']}.log")
            verify_run_artifacts(artifacts)
            output_files = freeze_output_files(actual)
            metrics = compare_case(directory, actual, case, precision, args.backend, gates, reference["max_objects"],
                                   getattr(args, "cuda_device", None) or 0, cuda_compute)
            verify_output_files(actual, output_files)
            metrics["output_sha256"] = output_files
            verify_run_artifacts(artifacts)
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
            metrics = {"id": case["id"], "passed": False, "failures": [str(error)]}
        report["cases"].append(metrics)
        write_json(args.output / "metrics.json", report)
    report["passed"] = all(case["passed"] for case in report["cases"])
    if cuda_compute == "f16":
        report.update(output_quality_passed=report["passed"], tensor_fidelity_passed=all(
            case.get("tensor_fidelity_passed") is True for case in report["cases"]),
            internal_candidate_equivalence_passed=all(
                case.get("internal_candidate_equivalence_passed") is True for case in report["cases"]))
    write_json(args.output / "metrics.json", report)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output / 'metrics.json'}")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), required=True)
    parser.add_argument("--cuda-device", type=int, help="Index among CUDA-visible devices (default: 0)")
    parser.add_argument("--cuda-compute", choices=("f32", "f16"), help="CUDA arithmetic; f16 uses final-output quality gates")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-diagnostic", action="store_true", help="Compare a labeled subset without claiming full M2 acceptance")
    args = parser.parse_args()
    if args.cuda_device is not None and (args.backend != "cuda" or args.cuda_device < 0):
        parser.error("--cuda-device requires backend cuda and a nonnegative index")
    if args.cuda_compute is not None and args.backend != "cuda":
        parser.error("--cuda-compute requires backend cuda")
    if args.threads <= 0 or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("threads and timeout must be positive and finite")
    try:
        parser.exit(validate(args))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f"video validation failed: {error}\n")


if __name__ == "__main__":
    main()
