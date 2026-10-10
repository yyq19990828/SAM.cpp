"""Four-axis configuration boundaries and tool propagation without GPU inference."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from tools.benchmark.precision_reporting import native_recipe
from tools.benchmark.quantization_benchmark import main as benchmark_main
from tools.convert import convert_sam3
from tools.convert.sam3_artifacts import SAM3_REVISION, read_json, sha256_file, write_json
from tools.quantize.quantization_config import (configuration_hash, configuration_receipt, conversion_options,
                                               load_config, main, normalize_config, validate_context,
                                               validate_recipe_configuration, validate_receipt)
from tests.tools.test_precision_acceptance import output_fixture
from tests.tools.test_quantization_performance import native_model, process_record


ROOT = Path(__file__).resolve().parents[2]


def configuration(precision="q4_k", backend="cuda", compute="f16", cache="mixed-q8_0", modules=None):
    weights = {"precision": precision}
    if precision.startswith("q"):
        weights["modules"] = ["text", "decoder"] if modules is None else modules
    return {"schema_version": 1, "kind": "sam-quantization-config", "task": "image", "backend": backend,
            "weights": weights, "activation": {"mode": "backend-selected"},
            "compute": {"mode": compute}, "cache": {"mode": cache}}


def application_fixture(root):
    Image.new("RGB", (64, 48), "white").save(root / "image.png")
    cases = root / "cases.json"
    write_json(cases, {"schema_version": 1, "samples": [
        {"id": "camera", "image": "image.png", "prompts": ["object", "other object"]}]})
    binary = root / "binary"
    binary.write_bytes(b"stub measurement executable")
    return cases, binary


class QuantizationConfigChecks(unittest.TestCase):
    def test_all_custom_module_subsets_and_formats_resolve_without_collapsing_profiles(self):
        modules = ("vision", "text", "fusion", "decoder")
        for precision in ("q8_0", "q6_k", "q5_k", "q4_k"):
            for count in range(1, 5):
                for selected in itertools.combinations(modules, count):
                    config = normalize_config(configuration(precision, modules=list(reversed(selected))))
                    self.assertEqual(config["weights"]["modules"], list(selected))
                    self.assertEqual(config["weights"]["storage_profile"], f"image-modules-linear-{precision}-v1")
                    self.assertEqual(normalize_config(config), config)
            for prefix, expected in (("image-vision-linear-", ["vision"]),
                                     ("image-linear-", ["vision", "text"]), ("image-full-linear-", list(modules))):
                value = configuration(precision)
                value["weights"] = {"precision": precision, "storage_profile": f"{prefix}{precision}-v1"}
                config = normalize_config(value)
                self.assertEqual(config["weights"]["modules"], expected)
                self.assertEqual(normalize_config(config), config)
                options = conversion_options(configuration_receipt(config))
                self.assertIsNone(options["quantize_modules"])
                self.assertEqual(options["storage_profile"], value["weights"]["storage_profile"])

    def test_unknown_missing_mixed_activation_and_backend_combinations_are_rejected(self):
        invalid = []
        for axis in ("weights", "activation", "compute", "cache"):
            value = configuration()
            del value[axis]
            invalid.append(value)
        for key, value in (("schema_version", True), ("task", "video"), ("backend", "auto"), ("unknown", 1)):
            invalid.append({**configuration(), key: value})
        for weights in ({"precision": "q4_k", "module_precisions": {"vision": "q4_k", "text": "q8_0"}},
                        {"precision": "q4_k"},
                        {"precision": "q4_k", "modules": {"vision": "q4_k"}},
                        {"precision": "q4_k", "modules": []},
                        {"precision": "q4_k", "modules": ["text", "text"]},
                        {"precision": "f16", "modules": ["vision"]},
                        {"precision": "q4_k", "storage_profile": "image-vision-linear-q8_0-v1"}):
            invalid.append({**configuration(), "weights": weights})
        for mode in ("int8", "fp8", "f16"):
            invalid.append({**configuration(), "activation": {"mode": mode}})
        invalid.extend([configuration(backend="cpu"), configuration(backend="metal"),
                        {**configuration(), "compute": {"mode": "int8"}},
                        {**configuration(), "cache": {"mode": "q4_k"}}])
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_config(value)

    def test_entrypoint_context_does_not_turn_probe_settings_into_public_support(self):
        reduced = normalize_config(configuration())
        validate_context(reduced, "conversion")
        validate_context(reduced, "benchmark")
        with self.assertRaisesRegex(ValueError, "private benchmark"):
            validate_context(reduced, "public-api")
        metal = normalize_config(configuration("f16", "metal", "f32", "f32"))
        validate_context(metal, "inspect")
        with self.assertRaisesRegex(ValueError, "Metal"):
            validate_context(metal, "benchmark")
        with self.assertRaises(ValueError):
            validate_context(reduced, "original-reference")
        validate_context(normalize_config(configuration("f32", "cuda", "f32", "f32")), "original-reference")

    def test_json_identity_is_canonical_and_duplicate_fields_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one, two = root / "one.json", root / "two.json"
            write_json(one, configuration(modules=["decoder", "text"]))
            write_json(two, configuration(modules=["text", "decoder"]))
            first, second = load_config(one), load_config(two)
            self.assertEqual(first["resolved_sha256"], second["resolved_sha256"])
            self.assertNotEqual(first["source"]["sha256"], second["source"]["sha256"])
            changed = deepcopy(first)
            changed["config"]["cache"]["mode"] = "f32"
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_receipt(changed)
            one.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_config(one)

    def test_examples_and_read_only_entrypoint_work_without_python_model_dependencies(self):
        for path in (ROOT / "docs/configs/quantization").glob("*.json"):
            receipt = load_config(path)
            self.assertEqual(receipt["resolved_sha256"], configuration_hash(receipt["config"]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "validated.json"
            # -S removes site packages: this path must not import torch/numpy/gguf.
            result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "tools/quantize/quantization_config.py"),
                                     "validate", "--config", str(ROOT / "docs/configs/quantization/image-full-q4-cuda.json"),
                                     "--output", str(output)], cwd=directory, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(read_json(output)["inference_executed"])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["capabilities", "--output", str(output)])

    def test_converter_uses_the_same_weight_selection_and_rejects_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            write_json(path, configuration("q8_0", modules=["decoder", "vision"]))
            arguments = ["--checkpoint", "source.pt", "--bpe", "bpe.gz", "--output", "output.gguf",
                         "--quantization-config", str(path)]
            with patch.object(convert_sam3, "convert") as convert:
                convert_sam3.main(arguments)
                self.assertEqual(convert.call_args.args, (Path("source.pt"), Path("bpe.gz"), "q8_0",
                                                         Path("output.gguf"), "image", None, None, ["vision", "decoder"]))
                self.assertEqual(convert.call_args.kwargs["configuration"]["source"]["sha256"], sha256_file(path))
            for flags in (["--precision", "q8_0"], ["--task", "image"], ["--quantize-modules", "vision"]):
                with patch.object(convert_sam3, "convert") as convert, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    convert_sam3.main(arguments + flags)
                convert.assert_not_called()

    def test_tiny_conversion_and_inspection_keep_runtime_requests_separate_from_applied_weights(self):
        import torch
        from tools.convert.sam3_gguf import bytes_to_unicode
        base = list(bytes_to_unicode().values())
        merges = [(left, right) for left in base for right in base][:48894]
        vocab = base + [token + "</w>" for token in base] + [left + right for left, right in merges]
        vocab += ["<start_of_text>", "<end_of_text>"]
        schema = {"schema_version": 1, "sam3_revision": SAM3_REVISION,
                  "tensors": {"vit.blocks.0.attn.qkv.weight": [32, 2], "ddec.norm.bias": [2]}, "unused_tracker_tensors": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bpe, output = root / "source.pt", root / "bpe.gz", root / "q8.gguf"
            torch.save({"detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight": torch.ones((2, 32)),
                        "detector.transformer.decoder.norm.bias": torch.tensor([.25, -.5])}, source)
            bpe.write_bytes(b"fixture BPE; tokenizer is mocked")
            path = root / "config.json"
            write_json(path, configuration("q8_0", modules=["vision"]))
            with patch.object(convert_sam3, "load_tokenizer", return_value=(vocab, merges)), \
                    patch("tools.convert.sam3_artifacts.read_json", return_value=schema), redirect_stdout(io.StringIO()):
                convert_sam3.main(["--checkpoint", str(source), "--bpe", str(bpe), "--output", str(output),
                                   "--quantization-config", str(path)])
            manifest = read_json(output.with_suffix(".gguf.manifest.json"))
            self.assertEqual(manifest["quantization_modules"], ["vision"])
            self.assertEqual(manifest["configuration_application"]["compute"], "runtime-request-only")
            self.assertFalse(manifest["configuration_application"]["runtime_execution_performed"])
            self.assertEqual(manifest["weight_policy_sha256"], sha256_file(ROOT / "tools/quantize/weight_policy.py"))
            self.assertEqual(manifest["quantization_config_resolver_sha256"], sha256_file(ROOT / "tools/quantize/quantization_config.py"))
            description = root / "description.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark_main(["inspect-precision", "--model", str(output), "--output", str(description),
                                                 "--quantization-config", str(path)]), 0)
            report = read_json(description)
            self.assertFalse(report["inference_executed"])
            self.assertEqual(report["precision"]["configuration"], manifest["quantization_configuration"])
            self.assertEqual(report["precision"]["kernel_precision_evidence"], "NOT_COLLECTED")
            self.assertEqual(report["precision"]["requested"]["compute"], "f16")

    def test_model_mismatch_and_schema3_module_reporting_fail_before_host_probes(self):
        import argparse
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = native_model(root, "candidate", "q4_k")
            args = argparse.Namespace(model=model, binary=root / "binary", backend="cuda", compute="f16", cache="mixed-q8_0",
                                      activation="backend-selected", quantization_configuration=configuration_receipt(configuration("q8_0")))
            with patch("tools.maintenance.artifact_snapshot.runtime_environment") as host, self.assertRaisesRegex(ValueError, "weights differ"):
                native_recipe(args)
            host.assert_not_called()
            manifest = read_json(model.with_suffix(".gguf.manifest.json"))
            manifest["storage_profile"] = "image-vision-linear-q4_k-v1"
            del manifest["quantization_modules"]
            write_json(model.with_suffix(".gguf.manifest.json"), manifest)
            del args.quantization_configuration
            args.binary.write_bytes(b"probe")
            recipe = native_recipe(args, collect_environment=False)
            self.assertEqual(recipe["quantization_modules"], [])  # Wire metadata belongs to schema 4 only.
            self.assertEqual(recipe["quantization_configuration"]["config"]["weights"]["modules"], ["vision"])
            validate_recipe_configuration(recipe)

    def test_performance_cli_preserves_independent_configurations_and_offline_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, binary = application_fixture(root)
            baseline, candidate = native_model(root, "baseline", "f32"), native_model(root, "candidate", "q4_k")
            paths = [root / "baseline.json", root / "candidate.json"]
            for path, precision in zip(paths, ("f32", "q4_k")):
                write_json(path, configuration(precision, "cpu", "f32", "f32"))
            arguments = ["performance", "--cases", str(cases), "--input-root", str(root), "--binary", str(binary),
                         "--baseline-model", str(baseline), "--candidate-model", str(candidate),
                         "--baseline-config", str(paths[0]), "--candidate-config", str(paths[1]),
                         "--output", str(root / "run"), "--pairs", "1", "--warmups", "0", "--iterations", "2"]
            calls = []

            def measure(binary, model, recipe, case, kind, output, **options):
                calls.append(recipe)
                return process_record(recipe, case, kind, options["protocol"], 100 + len(calls), 1)

            with patch("tools.benchmark.quantization_performance.source_snapshot", return_value={}), \
                    patch("tools.benchmark.quantization_performance.native_snapshot", return_value={}), \
                    patch("tools.benchmark.quantization_performance.runtime_environment", return_value={"fixture": "cpu"}), \
                    patch("tools.benchmark.quantization_performance.run_process", side_effect=measure), redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark_main(arguments), 0)
            report = read_json(root / "run/report.json")
            self.assertEqual(len(calls), 4)
            self.assertEqual(report["recipes"]["candidate"]["quantization_configuration"], load_config(paths[1]))
            self.assertFalse(report["quality"]["required_for_performance"])
            for path in paths:
                path.unlink()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark_main(["summarize-performance", "--run", str(root / "run/run.json"),
                                                 "--output", str(root / "offline")]), 0)

    def test_invalid_file_combinations_cannot_start_export_inspection_or_performance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, other = root / "config.json", root / "other.json"
            write_json(path, configuration())
            write_json(other, configuration("f32", "cpu", "f32", "f32"))
            inspect = ["inspect-precision", "--model", "unused.gguf", "--output", str(root / "inspect"), "--quantization-config", str(path)]
            performance = ["performance", "--cases", "unused.json", "--input-root", str(root), "--binary", "unused",
                           "--baseline-model", "a.gguf", "--candidate-model", "b.gguf", "--output", str(root / "run")]
            invalid = [inspect + ["--compute", "f16"],
                       performance + ["--candidate-config", str(path)],
                       performance + ["--baseline-config", str(path), "--candidate-config", str(other)],
                       performance + ["--baseline-config", str(path), "--candidate-config", str(path), "--backend", "cuda"]]
            for arguments in invalid:
                with patch("tools.benchmark.quantization_benchmark.export") as export, \
                        patch("tools.benchmark.precision_reporting.inspect") as inspect_call, \
                        patch("tools.benchmark.quantization_performance.run_process") as child, \
                        redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    benchmark_main(arguments)
                export.assert_not_called()
                inspect_call.assert_not_called()
                child.assert_not_called()

    def test_candidate_model_mismatch_fails_before_environment_queries_or_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, binary = application_fixture(root)
            baseline, candidate = native_model(root, "baseline", "f32"), native_model(root, "candidate", "q4_k")
            before, after = root / "before.json", root / "after.json"
            write_json(before, configuration("f32", "cuda", "f32", "f32"))
            write_json(after, configuration("q8_0"))
            with patch("tools.benchmark.quantization_performance.runtime_environment") as environment, \
                    patch("tools.benchmark.quantization_performance.run_process") as child, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark_main(["performance", "--cases", str(cases), "--input-root", str(root), "--binary", str(binary),
                                "--baseline-model", str(baseline), "--candidate-model", str(candidate),
                                "--baseline-config", str(before), "--candidate-config", str(after), "--output", str(root / "run")])
            environment.assert_not_called()
            child.assert_not_called()
            self.assertFalse((root / "run").exists())

    def test_export_records_the_same_config_and_rejects_semantically_changed_recipes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, binary = application_fixture(root)
            model = native_model(root, "candidate", "q4_k")
            path = root / "config.json"
            write_json(path, configuration("q4_k", "cpu", "f32", "f32"))

            def produce(args, samples, inputs, identities, recipe):
                rows = []
                for index, prompt in enumerate(samples[0]["prompts"]):
                    output = args.output / f"output-{index}.json"
                    write_json(output, output_fixture(prompt=prompt))
                    rows.append({"sample_id": "camera", "prompt_index": index, "prompt": prompt,
                                 "file": output.name, "sha256": sha256_file(output)})
                return rows, {}

            with patch("tools.maintenance.artifact_snapshot.source_snapshot", return_value={}), \
                    patch("tools.maintenance.artifact_snapshot.native_snapshot", return_value={}), \
                    patch("tools.maintenance.artifact_snapshot.runtime_environment", return_value={"fixture": "cpu"}), \
                    patch("tools.validation.export_ranked_outputs.export_native", side_effect=produce), redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark_main(["export", "--engine", "native", "--cases", str(cases), "--input-root", str(root),
                                                 "--binary", str(binary), "--model", str(model), "--output", str(root / "export"),
                                                 "--quantization-config", str(path)]), 0)
            recipe = read_json(root / "export/recipe.json")
            self.assertEqual(recipe["quantization_configuration"], load_config(path))
            changed = deepcopy(recipe)
            changed["quantization_configuration"]["config"]["weights"]["modules"] = ["vision"]
            changed["quantization_configuration"]["resolved_sha256"] = configuration_hash(changed["quantization_configuration"]["config"])
            with self.assertRaisesRegex(ValueError, "weights differ"):
                validate_recipe_configuration(changed)
            def alter_config(*arguments):
                rows, metadata = produce(*arguments)
                path.write_text("{}")
                return rows, metadata

            with patch("tools.maintenance.artifact_snapshot.source_snapshot", return_value={}), \
                    patch("tools.maintenance.artifact_snapshot.native_snapshot", return_value={}), \
                    patch("tools.maintenance.artifact_snapshot.runtime_environment", return_value={"fixture": "cpu"}), \
                    patch("tools.validation.export_ranked_outputs.export_native", side_effect=alter_config), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark_main(["export", "--engine", "native", "--cases", str(cases), "--input-root", str(root),
                                "--binary", str(binary), "--model", str(model), "--output", str(root / "changed-config"),
                                "--quantization-config", str(path)])
            self.assertFalse((root / "changed-config/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
