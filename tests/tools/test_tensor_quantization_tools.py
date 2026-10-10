"""Exact tensor policies, allocation previews and observed CPU arithmetic costs."""

from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import gguf

from tests.tools.test_mixed_quantization_tools import create_fixture, mixed_configuration
from tests.tools.test_modular_quantization_tools import _Field, _MetadataReader, _MetadataWriter
from tools.benchmark.precision_reporting import native_recipe, precision_description, mixed_weight_fields
from tools.benchmark.execution_cost import validate_cost, run as cost_run
from tools.convert import convert_sam3, conversion_plan
from tools.convert.sam3_artifacts import read_json, sha256_file, write_json
from tools.convert.sam3_gguf import read_gguf, validate_metadata, write_metadata
from tools.quantize.quantization_config import load_config, normalize_config, validate_recipe_configuration
from tools.quantize.tensor_policy import parse_tensor_precisions, validate_tensor_precisions
from tools.quantize.weight_policy import mixed_quantization_profile, QUANTIZATION_MODULES


ROOT = Path(__file__).resolve().parents[2]


def tensor_configuration():
    value = mixed_configuration(overrides={"text": "q8_0", "fusion": "q6_k", "decoder": "q5_k"})
    value["schema_version"] = 3
    value["weights"]["tensor_precisions"] = {"vit.blocks.0.attn.qkv.weight": "q6_k", "text.resizer.weight": "q4_k",
                                            "fenc.layers.0.linear1.weight": "q5_k", "ddec.layers.0.linear1.weight": "f32"}
    return value


class TensorQuantizationChecks(unittest.TestCase):
    def test_exact_overrides_precede_modules_and_bind_versioned_identity(self):
        value = normalize_config(tensor_configuration())
        weights = value["weights"]
        profile = mixed_quantization_profile({key: weights[key] for key in (
            "base_precision", "module_precisions", "tensor_precisions", "policy_sha256")})
        self.assertEqual(profile["schema_version"], 6)
        self.assertEqual(profile["policy_sha256"], "c290f907480a3b3cadc5304989e2d4463ba0e8ca1e414a18bcf63bcfca17f471")
        self.assertEqual(parse_tensor_precisions(profile["tensor_precisions_csv"]), weights["tensor_precisions"])
        schema = read_json(ROOT / "tools/convert/sam3_tensor_schema.json")["tensors"]
        rows = conversion_plan.allocation_rows(schema, "mixed", mixed_policy=mixed_weight_fields(weights))
        indexed = {row["name"]: row for row in rows}
        for name, precision in weights["tensor_precisions"].items():
            self.assertEqual(indexed[name]["resolved_dtype"], precision)
            self.assertEqual(indexed[name]["selector"], "tensor:" + name)
        self.assertEqual(indexed["vit.blocks.0.mlp.lin2.weight"]["resolved_dtype"], "q8_0")
        self.assertEqual(indexed["vit.blocks.0.mlp.lin2.weight"]["quantization_reason"], "q8-row-fallback")
        self.assertEqual(indexed["text.token_embed.weight"]["resolved_dtype"], "f32")
        reordered = deepcopy(value)
        reordered["weights"]["tensor_precisions"] = dict(reversed(list(weights["tensor_precisions"].items())))
        self.assertEqual(normalize_config(reordered), value)
        for name in weights["tensor_precisions"]:
            changed = deepcopy(tensor_configuration())
            changed["weights"]["tensor_precisions"][name] = "q8_0"
            self.assertNotEqual(normalize_config(changed)["weights"]["policy_sha256"], weights["policy_sha256"])

    def test_unsupported_protected_unknown_duplicate_and_unmatched_rules_fail(self):
        for overrides in ({}, [], {"vit.blocks.*.attn.qkv.weight": "f32"}, {"vit.blocks.32.attn.qkv.weight": "f32"},
                          {"vit.blocks.01.attn.qkv.weight": "f32"}, {"text.token_embed.weight": "f32"},
                          {"geom.points_direct_project.weight": "q8_0"}, {"ddec.presence_token_head.layers.2.weight": "f32"},
                          {"vit.blocks.0.attn.qkv.weight": "f16"}, {"vit.blocks.0.attn.qkv.weight": 8}):
            changed = tensor_configuration()
            changed["weights"]["tensor_precisions"] = overrides
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                normalize_config(changed)
        for csv in ("text.resizer.weight=q4_k,text.resizer.weight=f32", "text.resizer.weight=q4_k,", "",
                    "vit.blocks.0.attn.qkv.weight=q6_k,text.resizer.weight=q4_k"):
            with self.subTest(csv=csv), self.assertRaises(ValueError):
                parse_tensor_precisions(csv)
        with self.assertRaises(ValueError):
            validate_tensor_precisions({"text.resizer.weight": "f32"}, {})
        changed = tensor_configuration()
        changed["schema_version"] = 2
        with self.assertRaises(ValueError):
            normalize_config(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            text = json.dumps(tensor_configuration()).replace(
                '"text.resizer.weight": "q4_k"', '"text.resizer.weight": "q4_k", "text.resizer.weight": "f32"')
            path.write_text(text)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_q_override_can_enable_an_otherwise_f32_module(self):
        value = tensor_configuration()
        value["weights"]["module_precisions"] = {module: "f32" for module in QUANTIZATION_MODULES}
        value["weights"]["tensor_precisions"] = {"text.resizer.weight": "q8_0"}
        weights = normalize_config(value)["weights"]
        self.assertEqual(mixed_quantization_profile(mixed_weight_fields(weights))["modules"], ["text"])
        schema = read_json(ROOT / "tools/convert/sam3_tensor_schema.json")["tensors"]
        rows = conversion_plan.allocation_rows(schema, "mixed", mixed_policy=mixed_weight_fields(weights))
        self.assertEqual([row["name"] for row in rows if row["resolved_dtype"] != "f32"], ["text.resizer.weight"])

    def test_schema_6_metadata_is_canonical_typed_and_rejects_schema_5_reinterpretation(self):
        weights = normalize_config(tensor_configuration())["weights"]
        writer = _MetadataWriter()
        with patch("tools.convert.sam3_gguf.validate_tokenizer"):
            write_metadata(writer, "mixed", "a" * 64, [], [], mixed_policy=mixed_weight_fields(weights))
            self.assertEqual(validate_metadata(_MetadataReader(writer.fields), "mixed", mixed_policy=mixed_weight_fields(weights))[0], "mixed")
            for name, field in (("sam.schema_version", _Field([gguf.GGUFValueType.UINT32], 5)),
                                ("sam.storage_profile", _Field([gguf.GGUFValueType.STRING], "image-mixed-linear-v1")),
                                ("sam.quantization.tensor_precisions", _Field([gguf.GGUFValueType.UINT32], 1)),
                                ("sam.quantization.tensor_precisions", _Field([gguf.GGUFValueType.STRING], "")),
                                ("sam.quantization.policy_sha256", _Field([gguf.GGUFValueType.STRING], "0" * 64))):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    validate_metadata(_MetadataReader({**writer.fields, name: field}))
            fields = {name: field for name, field in writer.fields.items() if name != "sam.quantization.tensor_precisions"}
            with self.assertRaises(ValueError):
                validate_metadata(_MetadataReader(fields))

    def test_schema_only_preview_requires_no_checkpoint_encoder_or_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preview.json"
            with patch.object(convert_sam3, "quantize_native_rows", side_effect=AssertionError("must not encode")), \
                    patch.object(convert_sam3, "quantizer_identity", side_effect=AssertionError("must not load helper")), \
                    redirect_stdout(io.StringIO()):
                convert_sam3.main(["--dry-run", "--quantization-config", str(ROOT / "docs/configs/quantization/image-tensor-mixed-cpu.json"),
                                   "--output", str(output)])
            result = read_json(output)
            self.assertEqual(result["summary"]["tensor_count"], 1133)
            self.assertIsNone(result["gguf_size"]["exact_bytes"])
            self.assertGreater(result["gguf_size"]["upper_bound_bytes"], result["gguf_size"]["lower_bound_bytes"])
            self.assertEqual(result["checkpoint_preflight"], "NOT_REQUESTED")
            self.assertFalse(result["payload_generated"] or result["inference_executed"])
            self.assertEqual(list(Path(directory).iterdir()), [output])
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                convert_sam3.main(["--dry-run", "--precision", "q8_0", "--output", str(output)])

    def test_native_encoder_requirement_follows_resolved_tensors(self):
        value = tensor_configuration()
        value["weights"]["module_precisions"] = {module: "f32" for module in QUANTIZATION_MODULES}
        value["weights"]["tensor_precisions"] = {"vit.blocks.0.mlp.lin2.weight": "q4_k"}
        with tempfile.TemporaryDirectory() as directory:
            source, bpe, output, config, schema, tokenizer, _, _ = create_fixture(Path(directory), value)
            arguments = ["--checkpoint", str(source), "--bpe", str(bpe), "--quantization-config", str(config), "--output", str(output)]
            with patch.object(convert_sam3, "load_tokenizer", return_value=tokenizer), \
                    patch("tools.convert.sam3_artifacts.read_json", return_value=schema), \
                    patch.object(convert_sam3, "quantizer_identity", side_effect=AssertionError("Q8 fallback needs no native encoder")), \
                    redirect_stdout(io.StringIO()):
                convert_sam3.main(arguments)
            manifest = read_json(output.with_suffix(".gguf.manifest.json"))
            self.assertEqual(Counter(row["dtype"] for row in manifest["tensors"]), {"float32": 9, "q8_0": 1})
            self.assertFalse(manifest["conversion_plan"]["requires_native_encoder_for_conversion"])
        value["weights"]["tensor_precisions"] = {"text.resizer.weight": "q4_k"}
        with tempfile.TemporaryDirectory() as directory:
            source, bpe, output, config, schema, tokenizer, receipt, _ = create_fixture(Path(directory), value)
            with patch("tools.convert.sam3_artifacts.read_json", return_value=schema), self.assertRaisesRegex(ValueError, "--quantizer"):
                convert_sam3.convert(source, bpe, "mixed", output, storage_profile=receipt["config"]["weights"]["storage_profile"],
                                     configuration=receipt)
            self.assertFalse(output.exists())

    def test_preview_matches_real_single_pass_conversion_and_cpu_costs(self):
        helper = Path(os.environ.get("SAM_TEST_QUANTIZER", ROOT / "build/cpu-precision-tools/examples/sam_quantize_rows"))
        validator = Path(os.environ.get("SAM_TEST_MIXED_VALIDATOR", ROOT / "build/cpu-precision-tools/tests/test_mixed_quantization"))
        if not helper.is_file() or not validator.is_file():
            self.skipTest("build CPU sam_quantize_rows/test_mixed_quantization first")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bpe, output, config, schema, tokenizer, receipt, _ = create_fixture(root, tensor_configuration())
            arguments = ["--checkpoint", str(source), "--bpe", str(bpe), "--quantization-config", str(config)]
            preview_path = root / "preview.json"
            with patch.object(convert_sam3, "load_tokenizer", return_value=tokenizer), \
                    patch.object(conversion_plan, "read_json", return_value=schema), redirect_stdout(io.StringIO()):
                convert_sam3.main([*arguments, "--dry-run", "--output", str(preview_path)])
            preview = read_json(preview_path)
            self.assertFalse(output.exists())
            self.assertIn("values NOT_SCANNED", preview["checkpoint_preflight"]["validation"])
            with patch.object(convert_sam3, "load_tokenizer", return_value=tokenizer), \
                    patch("tools.convert.sam3_artifacts.read_json", return_value=schema), \
                    patch.object(convert_sam3, "quantize_native_rows", wraps=convert_sam3.quantize_native_rows) as encoder, \
                    redirect_stdout(io.StringIO()):
                convert_sam3.main([*arguments, "--quantizer", str(helper.resolve()), "--output", str(output)])
            manifest = read_json(output.with_suffix(".gguf.manifest.json"))
            self.assertEqual(encoder.call_count, 3)
            self.assertEqual(manifest["sam_schema_version"], 6)
            self.assertEqual(manifest["output"]["bytes"], preview["gguf_size"]["exact_bytes"])
            self.assertEqual(manifest["conversion_plan"]["summary"], preview["summary"])
            self.assertEqual(Counter(row["dtype"] for row in manifest["tensors"]),
                             {"float32": 5, "q4_k": 1, "q8_0": 2, "q6_k": 1, "q5_k": 1})
            rows = {row["name"]: row for row in preview["tensors"]}
            for row in manifest["tensors"]:
                for name in ("bytes", "requested_dtype", "resolved_dtype", "selector", "quantization_reason"):
                    self.assertEqual(row[name], rows[row["name"]][name])
            cost_path = root / "cpu-cost.json"
            executed = subprocess.run([str(validator), "--fixture", str(output), "--cost-report", str(cost_path)],
                                      check=True, capture_output=True, text=True, timeout=60)
            self.assertIn("maximum_absolute_error=", executed.stdout)
            cost = read_json(cost_path)
            validate_cost(cost)
            self.assertFalse(cost["performance_comparable"])
            self.assertEqual(cost["categories"]["matmul"]["calls"], 10)
            self.assertEqual(cost["categories"]["quantized_to_f32"]["calls"], 5)
            self.assertEqual({row["backend"] for row in cost["nodes"]}, {"CPU"})
            self.assertEqual(cost["kernel_internal_rhs_packing"], "NOT_COLLECTED")
            self.assertGreater(cost["transfer_totals"]["upload"]["bytes"], 0)
            self.assertGreaterEqual(cost["graph_compute_wall_ms"], sum(row["wall_ms"] for row in cost["nodes"]))
            for changed in ({**cost, "performance_comparable": True}, {**cost, "backend": "cuda"},
                            {**cost, "kernel_internal_rhs_packing": "f32"}, {**cost, "graph_compute_wall_ms": -1}):
                with self.assertRaises(ValueError):
                    validate_cost(changed)
            changed = deepcopy(cost)
            changed["categories"]["matmul"]["calls"] += 1
            with self.assertRaises(ValueError):
                validate_cost(changed)
            changed = deepcopy(cost)
            changed["transfer_totals"]["upload"]["bytes"] += 1
            with self.assertRaises(ValueError):
                validate_cost(changed)
            # Bind the exact tensor policy into runtime/precision receipts.
            import argparse
            recipe = native_recipe(argparse.Namespace(model=output, binary=validator, backend="cpu", compute="f32", cache="f32",
                                                      quantization_configuration=receipt), collect_environment=False)
            validate_recipe_configuration(recipe)
            described = precision_description({"recipe": recipe})
            self.assertEqual(described["weights"]["tensor_precisions"], manifest["tensor_precisions"])
            # Exercise the report/child boundary with a stub producer. Actual
            # arithmetic/timings above come from the native CPU fixture, not this stub.
            image = root / "stub-image.png"
            image.write_bytes(b"stub child does not decode pixels")
            args = argparse.Namespace(model=output, binary=validator, image=image, text="person", backend="cpu",
                                      compute="f32", cache="f32", quantization_configuration=receipt,
                                      output=root / "cost-wrapper", timeout=60)
            raw = {"kind": "sam-execution-cost-run-v1", "complete": True, "diagnostic_only": True,
                   "performance_comparable": False, "model": str(output.resolve()), "image": str(image.resolve()),
                   "prompt": "person", "threads": 4, "precision": "mixed", "storage_profile": manifest["storage_profile"],
                   "arithmetic_profile": "ggml-quantized-weights-f32-v1", "quantization_modules": manifest["quantization_modules"],
                   **mixed_weight_fields(manifest), "execution_cost": cost, "model_load_wall_ms": 0, "rss_peak_bytes": 0,
                   "runtime": {"cpu_nodes": 15, "cuda_nodes": 0, "metal_nodes": 0,
                               "host_upload_bytes": cost["transfer_totals"]["upload"]["bytes"],
                               "host_download_bytes": cost["transfer_totals"]["download"]["bytes"]}}
            def producer(command, **kwargs):
                self.assertEqual(command[command.index("--backend") + 1], "cpu")
                destination = Path(command[command.index("--output") + 1])
                destination.mkdir()
                write_json(destination / "execution-cost.json", raw)
                return subprocess.CompletedProcess(command, 0, "stub producer\n", "")
            with patch("tools.maintenance.artifact_snapshot.native_snapshot", return_value={str(output.resolve()): sha256_file(output)}), \
                    patch("tools.benchmark.execution_cost.subprocess.run", side_effect=producer), redirect_stdout(io.StringIO()):
                report = cost_run(args)
                self.assertFalse(report["performance_comparable"])
                self.assertEqual(report["recipe"]["tensor_precisions"], manifest["tensor_precisions"])
                self.assertEqual(report["image_sha256"], sha256_file(image))
                args.output = root / "bad-cost-wrapper"
                raw["policy_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "differs"):
                    cost_run(args)
                self.assertFalse((args.output / "report.json").exists())
            changed = deepcopy(recipe)
            del changed["tensor_precisions"]
            with self.assertRaises(ValueError):
                validate_recipe_configuration(changed)
            reader = read_gguf(output)
            self.assertEqual(validate_metadata(reader, "mixed", mixed_policy=mixed_weight_fields(receipt["config"]["weights"]))[0], "mixed")
            del reader
