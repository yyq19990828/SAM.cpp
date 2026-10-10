"""CPU native policies and report boundaries; only bounded native matrix execution."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.tools.test_quantization_config import configuration, application_fixture
from tests.tools.test_quantization_performance import native_model, process_record
from tests.tools.test_tensor_quantization_tools import tensor_configuration
from tools.benchmark.execution_cost import run as cost_run, validate_cost, validate_native_cpu_cost
from tools.benchmark.precision_reporting import (native_recipe, expected_runtime_profile, precision_description,
                                                native_compute_policy_matches)
from tools.benchmark.quantization_benchmark import main
from tools.convert.sam3_artifacts import read_json, write_json
from tools.maintenance.artifact_snapshot import native_snapshot, runtime_environment
from tools.quantize.quantization_config import normalize_config, configuration_receipt, load_config, validate_context


ROOT = Path(__file__).resolve().parents[2]


class CpuNativeQuantizationChecks(unittest.TestCase):
    def test_native_cpu_requires_explicit_backend_and_quantized_weights(self):
        for precision in ("q8_0", "q6_k", "q5_k", "q4_k"):
            config = normalize_config(configuration(precision, "cpu", "native-quantized", "f32"))
            validate_context(config, "public-api")
            self.assertEqual(config["activation"]["mode"], "backend-selected")
        for schema in (2, 3):
            value = tensor_configuration()
            value["compute"]["mode"] = "native-quantized"
            if schema == 2:
                value["schema_version"] = 2
                del value["weights"]["tensor_precisions"]
            self.assertEqual(normalize_config(value)["schema_version"], schema)
        invalid = [configuration(p, "cpu", "native-quantized", "f32") for p in ("f32", "f16")]
        invalid += [configuration("q4_k", b, "native-quantized", "f32") for b in ("cuda", "metal", "auto")]
        invalid += [configuration("q4_k", "cpu", "native-quantized", "mixed-q8_0"),
                    {**configuration("q4_k", "cpu", "native-quantized", "f32"), "activation": {"mode": "int8"}}]
        all_f32 = tensor_configuration()
        all_f32["compute"]["mode"] = "native-quantized"
        all_f32["weights"]["module_precisions"] = dict.fromkeys(("vision", "text", "fusion", "decoder"), "f32")
        all_f32["weights"]["tensor_precisions"] = {"text.resizer.weight": "f32"}
        invalid.append(all_f32)
        for value in invalid:
            with self.subTest(config=value), self.assertRaises(ValueError):
                normalize_config(value)

    def test_precision_report_separates_f32_graph_activations_from_internal_rhs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"stub identity, no inference")
            model = native_model(root, "quantized", "q4_k")
            args = argparse.Namespace(model=model, binary=binary, backend="cpu", compute="native-quantized", cache="f32")
            recipe = native_recipe(args, collect_environment=False)
            self.assertEqual(expected_runtime_profile(recipe), "ggml-quantized-cpu-native-v1")
            report = precision_description({"recipe": recipe, "arithmetic_profile": expected_runtime_profile(recipe)})
            rhs = report["activation"]["cpu_native_rhs"]
            self.assertEqual((rhs["graph_input"], rhs["q8_0_weights"], rhs["q6_k_q5_k_q4_k_weights"]), ("f32", "q8_0", "q8_k"))
            self.assertIn("not runtime kernel capture", rhs["evidence"])
            self.assertFalse(report["compute"]["universal_operand_dtype_guarantee"])
            self.assertEqual(report["kernel_precision_evidence"], "NOT_COLLECTED")
            old = deepcopy(recipe)
            old["compute_mode"] = "f32"
            del old["quantization_configuration"]
            self.assertEqual(expected_runtime_profile(old), "ggml-quantized-weights-f32-v1")
            self.assertIn("cast to F32", precision_description({"recipe": old})["compute"]["source_resolved_path"])
            self.assertNotEqual(recipe["quantization_configuration"]["resolved_sha256"],
                                configuration_receipt(configuration("q4_k", "cpu", "f32", "f32"))["resolved_sha256"])
            args.model = native_model(root, "dense", "f32")
            with self.assertRaisesRegex(ValueError, "quantized weights"):
                native_recipe(args, collect_environment=False)

    def test_compute_receipts_do_not_mislabel_cpu_as_cuda_or_accept_legacy_native_claims(self):
        recipe = {"backend": "cpu", "compute_mode": "native-quantized"}
        correct = {"compute_mode": "native-quantized", "cpu_compute": "native-quantized", "cuda_compute": "f32"}
        self.assertTrue(native_compute_policy_matches(correct, recipe))
        for changed in ({**correct, "cuda_compute": "native-quantized"}, {**correct, "cpu_compute": "f32"},
                        {"cuda_compute": "native-quantized"}, {"compute_mode": "native-quantized"}):
            self.assertFalse(native_compute_policy_matches(changed, recipe))
        self.assertTrue(native_compute_policy_matches({"cuda_compute": "f32"}, {"backend": "cpu", "compute_mode": "f32"}))
        self.assertTrue(native_compute_policy_matches({"cuda_compute": "f16"}, {"backend": "cuda", "compute_mode": "f16"}))

    def test_paired_performance_configurations_keep_cpu_compute_identity_without_quality_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, binary = application_fixture(root)
            models = [native_model(root, name, "q4_k") for name in ("baseline", "candidate")]
            configs = [root / "baseline.json", root / "candidate.json"]
            for path, compute in zip(configs, ("f32", "native-quantized")):
                write_json(path, configuration("q4_k", "cpu", compute, "f32"))
            seen = []
            def producer(binary, model, recipe, case, kind, output, **options):
                seen.append(recipe["compute_mode"])
                row = process_record(recipe, case, kind, options["protocol"], 100 + len(seen), 1)
                row.update(compute_mode=recipe["compute_mode"], cpu_compute=recipe["compute_mode"], cuda_compute="f32")
                return row
            arguments = ["performance", "--cases", str(cases), "--input-root", str(root), "--binary", str(binary),
                         "--baseline-model", str(models[0]), "--candidate-model", str(models[1]),
                         "--baseline-config", str(configs[0]), "--candidate-config", str(configs[1]),
                         "--output", str(root / "run"), "--pairs", "1", "--warmups", "0", "--iterations", "2"]
            with patch("tools.benchmark.quantization_performance.source_snapshot", return_value={}), \
                    patch("tools.benchmark.quantization_performance.native_snapshot", return_value={}), \
                    patch("tools.benchmark.quantization_performance.runtime_environment", return_value={"fixture": "cpu"}), \
                    patch("tools.benchmark.quantization_performance.run_process", side_effect=producer), redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            self.assertEqual(set(seen), {"f32", "native-quantized"})
            report = read_json(root / "run/report.json")
            self.assertFalse(report["quality"]["required_for_performance"])
            self.assertEqual(report["recipes"]["candidate"]["compute_mode"], "native-quantized")

    def test_native_cpu_observed_costs_and_stub_wrapper_reject_wrong_rhs_and_policy(self):
        validator = Path(os.environ.get("SAM_TEST_NATIVE_CPU_VALIDATOR", ROOT / "build/cpu-precision-tools/tests/test_cpu_quantized_matmul"))
        if not validator.is_file():
            self.skipTest("build CPU test_cpu_quantized_matmul first")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cost_path = root / "cost.json"
            subprocess.run([str(validator), "--cost-report", str(cost_path)], check=True, capture_output=True, text=True, timeout=60)
            cost = read_json(cost_path)
            validate_cost(cost)
            validate_native_cpu_cost(cost)
            self.assertNotIn("quantized_to_f32", cost["categories"])
            self.assertEqual(cost["categories"]["matmul"]["calls"], 1)
            for key, value in (("backend", "BLAS"), ("cpu_rhs_dot_type", "q8_0"), ("output_type", "f16")):
                changed = deepcopy(cost)
                node = next(row for row in changed["nodes"] if row["category"] == "matmul")
                node[key] = value
                with self.assertRaises(ValueError):
                    validate_native_cpu_cost(changed)
            # Only the small graph above was really executed. This fake full
            # model producer tests argv/receipt validation, not SAM inference.
            model = native_model(root, "candidate", "q4_k")
            image = root / "image.png"
            image.write_bytes(b"stub producer does not decode this image")
            config = root / "native.json"
            write_json(config, configuration("q4_k", "cpu", "native-quantized", "f32"))
            args = argparse.Namespace(model=model, binary=validator, image=image, text="person", backend="cpu",
                                      compute="native-quantized", cache="f32", quantization_configuration=load_config(config),
                                      output=root / "wrapped", timeout=60)
            raw = {"kind": "sam-execution-cost-run-v1", "complete": True, "diagnostic_only": True,
                   "performance_comparable": False, "model": str(model.resolve()), "image": str(image.resolve()),
                   "prompt": "person", "threads": 4, "precision": "q4_k", "storage_profile": "image-modules-linear-q4_k-v1",
                   "quantization_modules": ["text", "decoder"], "arithmetic_profile": "ggml-quantized-cpu-native-v1",
                   "compute_mode": "native-quantized", "cpu_compute": "native-quantized", "cuda_compute": "f32",
                   "execution_cost": cost, "model_load_wall_ms": 0, "rss_peak_bytes": 0,
                   "runtime": {"cpu_nodes": 1, "cuda_nodes": 0, "metal_nodes": 0,
                               "host_upload_bytes": cost["transfer_totals"]["upload"]["bytes"],
                               "host_download_bytes": cost["transfer_totals"]["download"]["bytes"]}}
            def producer(command, **kwargs):
                self.assertEqual(command[command.index("--cpu-compute") + 1], "native-quantized")
                destination = Path(command[command.index("--output") + 1])
                destination.mkdir()
                write_json(destination / "execution-cost.json", raw)
                return subprocess.CompletedProcess(command, 0, "stub full-model producer\n", "")
            identities = native_snapshot(validator, model)
            environment = runtime_environment("cpu")
            with patch("tools.maintenance.artifact_snapshot.native_snapshot", return_value=identities), \
                    patch("tools.maintenance.artifact_snapshot.runtime_environment", return_value=environment), \
                    patch("tools.benchmark.execution_cost.subprocess.run", side_effect=producer), redirect_stdout(io.StringIO()):
                report = cost_run(args)
                self.assertFalse(report["performance_comparable"])
                args.output = root / "wrong-policy"
                raw["cpu_compute"] = "f32"
                with self.assertRaisesRegex(ValueError, "differs"):
                    cost_run(args)
                self.assertFalse((args.output / "report.json").exists())
