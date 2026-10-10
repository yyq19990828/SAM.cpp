"""Exercise quality-free performance reports and precision evidence on CPU fixtures."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from tools.benchmark.performance_measurement import run_process
from tools.benchmark.precision_reporting import (conversion_manifest, expected_runtime_profile,
                                                precision_description, validate_configuration, weight_inventory)
from tools.benchmark.quantization_benchmark import main
from tools.benchmark.quantization_performance import load_run, run, summarize
from tools.convert.sam3_artifacts import read_json, sha256_file, write_json


def native_model(root, name, precision):
    model = root / (name + ".gguf")
    model.write_bytes(name.encode())
    write_json(model.with_suffix(".gguf.manifest.json"), {
        "task": "image", "precision": precision, "checkpoint": {"sha256": "a" * 64},
        "output": {"sha256": sha256_file(model)},
        "storage_profile": "dense" if precision == "f32" else "image-modules-linear-q4_k-v1",
        "quantization_modules": [] if precision == "f32" else ["text", "decoder"],
        "tensors": [{"name": "matrix", "dtype": precision if precision.startswith("q") else "float32",
                     "bytes": 144, "quantization_reason": "selected-linear"},
                    {"name": "bias", "dtype": "float32", "bytes": 16, "quantization_reason": "protected-f32"}]})
    return model


def process_record(recipe, case, kind, protocol, pid, ratio=1):
    return {"schema_version": 2, "complete": True, "backend": recipe["backend"], "kind": kind,
            "feature_cache": recipe["feature_cache"], "cuda_compute": recipe["compute_mode"], "threads": 4,
            "precision": recipe["weight_precision"], "storage_profile": recipe["storage_profile"],
            "quantization_modules": recipe["quantization_modules"], "arithmetic_profile": expected_runtime_profile(recipe),
            "prompt": case["prompt"], "alternate": case["alternate"], **protocol[kind],
            "process_id": pid, "memory_sampler_enabled": kind == "memory", "rss_peak_bytes": int(10000 * ratio),
            "timing_ms": {name: [100 * ratio] * protocol[kind]["iterations"]
                          for name in ("full_image", "changed_prompt", "repeated_result")},
            "runtime": {"cpu_nodes": 1}, "other_compute_pids": [], "observer_errors": [], "diagnostic_only": False}


class QuantizationPerformanceChecks(unittest.TestCase):
    def fixture(self, root, ratios=(2, 2, 2), contaminated=False):
        Image.new("RGB", (4, 3), "white").save(root / "image.png")
        cases = root / "application.json"
        write_json(cases, {"schema_version": 1, "samples": [
            {"id": "camera", "image": "image.png", "prompts": ["person", "forklift"]}]})
        binary = root / "binary"
        binary.write_bytes(b"synthetic measurement producer")
        args = argparse.Namespace(cases=cases, input_root=root, binary=binary,
                                  baseline_model=native_model(root, "baseline", "f32"),
                                  candidate_model=native_model(root, "candidate", "q4_k"), output=root / "run",
                                  backend="cpu", activation="backend-selected", baseline_compute="f32", candidate_compute="f32",
                                  baseline_cache="f32", candidate_cache="f32", pairs=3, warmups=0, iterations=3,
                                  memory_iterations=1, case=None, limit=1, timeout=1, benefit_min_percent=5)
        calls = []

        def measure(binary, model, recipe, case, kind, output, **options):
            pair = int(output.name.split("-")[-2])
            ratio = ratios[pair] if model == args.candidate_model.resolve() else 1
            row = process_record(recipe, case, kind, options["protocol"], 100 + len(calls), ratio)
            if contaminated:
                row["observer_errors"] = ["synthetic observer failure"]
            calls.append((kind, pair, model.name))
            return row

        with patch("tools.benchmark.quantization_performance.source_snapshot", return_value={}), \
                patch("tools.benchmark.quantization_performance.native_snapshot", return_value={}), \
                patch("tools.benchmark.quantization_performance.run_process", side_effect=measure), \
                patch("tools.archive.precision_v2_v3.precision_acceptance.load_gates", side_effect=AssertionError("quality policy used")), \
                patch("tools.archive.precision_v2_v3.benchmark_precision.accepted_quality", side_effect=AssertionError("quality prerequisite used")), \
                redirect_stdout(io.StringIO()):
            report = run(args)
        return args, report, calls

    def test_slowdown_without_quality_report_completes_and_offline_summary_needs_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, report, calls = self.fixture(root)
            self.assertEqual(report["report_validity"], "VALID")
            self.assertFalse(report["quality"]["required_for_performance"])
            self.assertEqual(report["quality"]["status"], "NOT_MEASURED")
            self.assertEqual(report["summary"]["workloads"]["full_image"]["ratios"]["p50_ratio"], 2)
            self.assertEqual(report["benefit_tags"]["workloads"]["full_image"]["latency"], "NOT_DEMONSTRATED")
            self.assertEqual(len(calls), 12)
            self.assertEqual([row[2] for row in calls[:6]], ["baseline.gguf", "candidate.gguf", "candidate.gguf",
                                                            "baseline.gguf", "baseline.gguf", "candidate.gguf"])
            self.assertNotIn("quality_tier", read_json(args.output / "run.json")["recipes"]["candidate"])
            for model in (args.baseline_model, args.candidate_model, args.binary):
                model.unlink()
            before = sha256_file(args.output / "run.json")
            self.assertEqual(main(["summarize-performance", "--run", str(args.output / "run.json"),
                                   "--output", str(root / "offline")]), 0)
            self.assertEqual(before, sha256_file(args.output / "run.json"))
            self.assertEqual(read_json(root / "offline/report.json")["summary"], report["summary"])

    def test_benefit_instability_and_contamination_are_separate_from_report_validity(self):
        for ratios, contaminated, status in (((.8, .8, .8), False, "CONSISTENT_REDUCTION"),
                                              ((.8, .98, .8), False, "UNSTABLE"),
                                              ((.8, .8, .8), True, "INCONCLUSIVE")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                _, report, _ = self.fixture(Path(directory), ratios, contaminated)
                self.assertEqual(report["report_validity"], "VALID")
                self.assertEqual(report["benefit_tags"]["workloads"]["changed_prompt"]["latency"], status)
                self.assertFalse(report["benefit_tags"]["affects_exit_code"])
                self.assertNotIn("gpu-memory", report["benefit_tags"]["workloads"]["full_image"])

    def test_altered_missing_and_invalid_measurements_cannot_publish_summary(self):
        for change in ("image", "hash", "missing", "nan", "sampler", "pid", "precision", "order"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, _, _ = self.fixture(root)
                path = args.output / "run.json"
                value = read_json(path)
                item = value["records"][0]
                record_path = args.output / item["file"]
                row = read_json(record_path)
                if change == "image":
                    (args.output / value["cases"][0]["image"]).write_bytes(b"changed")
                elif change == "hash":
                    record_path.write_bytes(b"changed")
                elif change == "missing":
                    value["records"].pop()
                elif change == "order":
                    value["records"].reverse()
                else:
                    if change == "nan":
                        row["timing_ms"]["full_image"][0] = float("nan")
                    elif change == "sampler":
                        row["memory_sampler_enabled"] = True
                    elif change == "pid":
                        other = read_json(args.output / value["records"][1]["file"])
                        row["process_id"] = other["process_id"]
                    else:
                        row["cuda_compute"] = "f16"
                    record_path.write_text(json.dumps(row))  # Deliberately allow malformed non-finite input.
                    item["sha256"] = sha256_file(record_path)
                write_json(path, value)
                with self.assertRaises((ValueError, OSError)):
                    summarize(argparse.Namespace(run=path, output=root / "bad-summary", benefit_min_percent=5))
                self.assertFalse((root / "bad-summary/report.json").exists())

    def test_unsupported_activation_and_non_cuda_precision_fail_before_launch(self):
        for backend, compute, cache, activation in (("cpu", "f16", "f32", "backend-selected"),
                                                   ("metal", "f32", "mixed-q8_0", "backend-selected"),
                                                   ("cuda", "f16", "f32", "int8")):
            with self.subTest(backend=backend, activation=activation), self.assertRaises(ValueError):
                validate_configuration(backend, compute, cache, activation)
        with tempfile.TemporaryDirectory() as directory, patch("tools.benchmark.precision_reporting.conversion_manifest") as weights:
            with self.assertRaises(SystemExit) as error:
                main(["inspect-precision", "--model", "unused.gguf", "--output", str(Path(directory) / "unused.json"), "--activation", "fp8"])
            self.assertEqual(error.exception.code, 1)
            weights.assert_not_called()

    def test_precision_evidence_distinguishes_weights_hints_and_unobserved_kernel_math(self):
        with tempfile.TemporaryDirectory() as directory:
            model = native_model(Path(directory), "candidate", "q4_k")
            manifest, _, _ = conversion_manifest(model)
            inventory = weight_inventory(manifest)
            self.assertEqual(set(inventory["types"]), {"q4_k", "float32"})
            recipe = {"engine": "native", "backend": "cuda", "weight_precision": "q4_k", "storage_profile": manifest["storage_profile"],
                      "quantization_modules": ["text", "decoder"], "compute_mode": "f16", "feature_cache": "mixed-q8_0"}
            description = precision_description({"recipe": recipe, "weight_inventory": inventory,
                                                 "arithmetic_profile": "ggml-quantized-cuda-f16-v1"})
            self.assertEqual([row["storage_type"] for row in description["cache"]["levels"]], ["q8_0", "q8_0", "f32"])
            self.assertFalse(description["activation"]["independent_setting_supported"])
            self.assertFalse(description["compute"]["universal_operand_dtype_guarantee"])
            self.assertEqual(description["execution_evidence"]["runtime_policy_evidence"], "producer-reported")
            self.assertEqual(description["execution_evidence"]["kernel_internal_arithmetic"], "NOT_COLLECTED")
            cpu_recipe = {**recipe, "backend": "cpu", "weight_precision": "f16", "storage_profile": "dense",
                          "quantization_modules": [], "compute_mode": "f32", "feature_cache": "f32"}
            cpu = precision_description({"recipe": cpu_recipe})
            self.assertIn("promoted to F32", cpu["compute"]["source_resolved_path"])
            self.assertEqual(cpu["execution_evidence"]["runtime_policy_evidence"], "NOT_COLLECTED")
            model.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                conversion_manifest(model)

    def test_direct_cpu_child_observation_uses_declared_counts_and_rejects_policy_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "probe"
            binary.write_text(f"#!{sys.executable}\n" + '''import json, pathlib, resource, sys
model, image, prompt, alternate, backend, cache, compute, kind, output, warmups, iterations = sys.argv[1:]
path = pathlib.Path(output)
path.mkdir()
value = {"complete": True, "backend": backend, "feature_cache": cache, "cuda_compute": compute,
         "precision": "f32", "storage_profile": "", "quantization_modules": [], "arithmetic_profile": "",
         "kind": kind, "prompt": prompt, "alternate": alternate, "threads": 4,
         "warmups": int(warmups), "iterations": int(iterations), "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
(path / "result.json").write_text(json.dumps(value))
''')
            binary.chmod(0o755)
            recipe = {"backend": "cpu", "feature_cache": "f32", "compute_mode": "f32", "weight_precision": "f32",
                      "storage_profile": "dense", "quantization_modules": [], "threads": 4}
            case = {"image": "unused.png", "prompt": "a", "alternate": "b"}
            protocol = {"memory": {"warmups": 0, "iterations": 1}}
            value = run_process(binary, root / "unused.gguf", recipe, case, "memory", root / "child", protocol=protocol, timeout=5)
            self.assertTrue(value["memory_sampler_enabled"])
            self.assertGreater(value["process_id"], 0)
            self.assertTrue((root / "child/memory-samples.json").exists())
            changed = {**recipe, "weight_precision": "q4_k"}
            with self.assertRaisesRegex(ValueError, "identity"):
                run_process(binary, root / "unused.gguf", changed, case, "memory", root / "mismatch", protocol=protocol, timeout=5)


if __name__ == "__main__":
    unittest.main()
