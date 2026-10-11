"""Numerical contract tests; native execution is verified separately on CUDA."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import patch
import subprocess

import numpy as np

from tools.quantize.native_activation import encode_reference, check_payload, decode_payload, supported_values
from tools.quantize.quantization_config import normalize_config, configuration_hash, resolve_performance_arguments
from tools.convert.sam3_artifacts import write_json
from tools.benchmark.precision_reporting import native_compute_policy_matches, expected_runtime_profile


class NativeActivationChecks(unittest.TestCase):
    def test_offline_bundle_revalidates_actual_activation_after_rehash(self):
        from tests.tools.test_quantization_benchmark import bundle, recipe
        from tests.tools.test_precision_acceptance import output_fixture
        from tools.benchmark.quantization_benchmark import load_bundle
        from tools.convert.sam3_artifacts import read_json, sha256_file
        declared = recipe(precision="q8_0")
        declared.update(storage_profile="image-vision-linear-q8_0-v1", quantization_modules=[], activation="ggml-q8-rhs")
        payload = output_fixture()
        payload.update(compute_mode="f32", cuda_compute="f32", cpu_compute="f32", activation="ggml-q8-rhs",
            activation_policy="ggml-q8-rhs", arithmetic_profile="ggml-quantized-cuda-q8-rhs-v1",
            runtime={"vision_encodes": 1, "activation_requests": 128, "activation_mmq_launches": 128,
                     "activation_payload_peak_bytes": 576, "activation_allocation_peak_bytes": 576,
                     "cuda_nodes": 128, "cpu_nodes": 0, "metal_nodes": 0, "blas_nodes": 0})
        with tempfile.TemporaryDirectory() as directory:
            root = bundle(Path(directory) / "run", payload, declared)
            load_bundle(root)
            payload["runtime"]["activation_mmq_launches"] = 127
            output = root / "frame-one-p0.json"
            write_json(output, payload)
            manifest = read_json(root / "manifest.json")
            manifest["outputs"][0]["sha256"] = sha256_file(output)
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "actual execution"): load_bundle(root)

    def test_f32_cost_wrapper_rejects_wrong_activation_execution(self):
        from tools.benchmark.execution_cost import run
        recipe = {"task": "image", "engine": "native", "backend": "cuda", "compute_mode": "f32",
                  "feature_cache": "f32", "weight_precision": "q8_0", "activation": "ggml-q8-rhs",
                  "storage_profile": "image-vision-linear-q8_0-v1", "quantization_modules": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image"; image.write_bytes(b"report fixture")
            args = SimpleNamespace(output=root / "out", backend="cuda", cache="f32", compute="f32", timeout=10,
                binary=root / "probe", model=root / "model", image=image, text="object")
            raw = {"kind": "sam-execution-cost-run-v1", "complete": True, "diagnostic_only": True, "performance_comparable": False,
                   "model": str(args.model.resolve()), "image": str(image.resolve()), "prompt": "object", "threads": 4,
                   "precision": "q8_0", "storage_profile": recipe["storage_profile"], "quantization_modules": [],
                   "arithmetic_profile": "ggml-quantized-cuda-q8-rhs-v1", "compute_mode": "f32", "cpu_compute": "f32",
                   "cuda_compute": "f32", "activation": "backend-selected"}
            def producer(command, **kwargs):
                self.assertEqual(command[command.index("--activation") + 1], "ggml-q8-rhs")
                path = Path(command[command.index("--output") + 1]); path.mkdir()
                write_json(path / "execution-cost.json", raw)
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch("tools.benchmark.execution_cost.native_recipe", return_value=recipe), \
                 patch("tools.maintenance.artifact_snapshot.native_snapshot", return_value={}), \
                 patch("tools.benchmark.execution_cost.subprocess.run", side_effect=producer), \
                 self.assertRaisesRegex(ValueError, "differs"):
                run(args)
            self.assertFalse((args.output / "report.json").exists())
    def test_versioned_configuration_and_supported_combinations(self):
        path = Path(__file__).resolve().parents[2] / "docs/configs/quantization/image-vision-q8-rhs-cuda.json"
        import json
        value = json.loads(path.read_text())
        config = normalize_config(value)
        self.assertEqual(normalize_config(config), config)
        invalid = []
        for schema in (1, 2, 3, 4): invalid.append({**value, "schema_version": schema})
        for backend in ("cpu", "metal"): invalid.append({**value, "backend": backend})
        for axis, mode in (("compute", "f16"), ("cache", "f16"), ("activation", "backend-selected")):
            invalid.append({**value, axis: {"mode": mode}})
        for precision, modules in (("q4_k", ["vision"]), ("q8_0", ["text"]), ("f32", [])):
            invalid.append({**value, "weights": {"precision": precision, "modules": modules}})
        for bad in invalid:
            with self.subTest(value=bad), self.assertRaises(ValueError): normalize_config(bad)

    def test_paired_configs_bind_separate_activation_identity(self):
        import json
        path = Path(__file__).resolve().parents[2] / "docs/configs/quantization/image-vision-q8-rhs-cuda.json"
        candidate = json.loads(path.read_text())
        baseline = deepcopy(candidate)
        baseline["schema_version"] = 1
        baseline["activation"]["mode"] = "backend-selected"
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "before.json", Path(directory) / "after.json"
            write_json(before, baseline); write_json(after, candidate)
            args = SimpleNamespace(baseline_config=before, candidate_config=after)
            resolve_performance_arguments(args)
            self.assertEqual(args.baseline_activation, "backend-selected")
            self.assertEqual(args.candidate_activation, "ggml-q8-rhs")
            self.assertNotEqual(configuration_hash(normalize_config(baseline)), configuration_hash(normalize_config(candidate)))

    def test_requested_mode_cannot_accept_default_or_fabricated_execution(self):
        recipe = {"backend": "cuda", "compute_mode": "f32", "feature_cache": "f32", "weight_precision": "q8_0", "activation": "ggml-q8-rhs"}
        value = {"compute_mode": "f32", "cuda_compute": "f32", "cpu_compute": "f32", "activation": "ggml-q8-rhs",
                 "activation_policy": "ggml-q8-rhs", "runtime": {"vision_encodes": 1, "activation_requests": 128,
                 "activation_mmq_launches": 128, "activation_payload_peak_bytes": 576, "activation_allocation_peak_bytes": 576}}
        self.assertTrue(native_compute_policy_matches(value, recipe))
        self.assertEqual(expected_runtime_profile(recipe), "ggml-quantized-cuda-q8-rhs-v1")
        for key, bad in (("activation", "backend-selected"), ("activation_policy", "backend-selected")):
            self.assertFalse(native_compute_policy_matches({**value, key: bad}, recipe))
        for key, bad in (("activation_requests", 0), ("activation_mmq_launches", 127), ("activation_payload_peak_bytes", 0),
                         ("activation_allocation_peak_bytes", 575)):
            changed = deepcopy(value); changed["runtime"][key] = bad
            self.assertFalse(native_compute_policy_matches(changed, recipe))
    def test_zero_and_exact_representable_blocks(self):
        source = np.zeros((3, 256), dtype="<f4")
        source[0] = np.tile(np.arange(-16, 16, dtype="<f4"), 8)
        source[0, ::32] = 127.
        source[2] = -source[0] * .5
        packed = encode_reference(source)
        self.assertTrue(check_payload(source, packed)["passed"])
        np.testing.assert_array_equal(decode_payload(packed, *source.shape), source)

    def test_tail_padding_multiple_rows_and_strided_source(self):
        source = (np.arange(3 * 128, dtype=np.float32).reshape(3, 128) % 53 - 26) * .013
        source = source[:, ::2]
        self.assertFalse(source.flags.c_contiguous)
        payload = encode_reference(source)
        report = check_payload(source, payload)
        self.assertEqual(report["padded_width"], 512)
        self.assertTrue(report["passed"])
        self.assertEqual(decode_payload(payload, *source.shape).shape, source.shape)

    def test_rounding_ties_and_signs(self):
        source = np.zeros((1, 32), dtype="<f4")
        source[0, :7] = [127., .5, -.5, 1.5, -1.5, 2.5, -2.5]
        decoded = decode_payload(encode_reference(source), 1, 32)
        np.testing.assert_array_equal(decoded[0, :7], [127., 1., -1., 2., -2., 3., -3.])

    def test_dynamic_range_fails_closed(self):
        for number in (np.nan, np.inf, -np.inf, 1.e31, 1.e-31, np.finfo(np.float32).tiny, 1.e100):
            with self.subTest(number=number), self.assertRaises(ValueError):
                supported_values(np.full((1, 32), number))
        for number in (0., 1.e-30, -1.e-30, 1.e30, -1.e30):
            source = np.full((1, 32), number, dtype="<f4")
            self.assertTrue(check_payload(source, encode_reference(source))["passed"])

    def test_corrupt_payload_cannot_pass(self):
        source = np.ones((2, 32), dtype="<f4")
        payload = bytearray(encode_reference(source))
        payload[16] ^= 127
        self.assertFalse(check_payload(source, bytes(payload))["passed"])
        with self.assertRaises(ValueError): check_payload(source, bytes(payload[:-1]))
        payload = bytearray(encode_reference(source))
        payload[:4] = np.array([np.inf], dtype="<f4").tobytes()
        with self.assertRaises(ValueError): decode_payload(bytes(payload), *source.shape)

    def test_invalid_shapes(self):
        for source in (np.zeros(32), np.zeros((0, 32)), np.zeros((1, 31)), np.zeros((1, 0))):
            with self.assertRaises(ValueError): encode_reference(source)


if __name__ == "__main__": unittest.main()
