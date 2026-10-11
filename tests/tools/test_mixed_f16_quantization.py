"""Versioned F16/Q allocation, transactional conversion and native interoperability."""

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import gguf
import numpy as np

from tests.tools.test_mixed_quantization_tools import create_fixture, convert_fixture, mixed_configuration
from tests.tools.test_modular_quantization_tools import _Field, _MetadataReader, _MetadataWriter
from tools.convert import convert_sam3
from tools.convert.sam3_artifacts import read_json
from tools.convert.sam3_gguf import quantized_tensor_type, read_gguf, validate_metadata, write_metadata
from tools.quantize.quantization_config import normalize_config, model_weights
from tools.quantize.weight_policy import F16_MIXED_STORAGE_PROFILE
from tools.benchmark.precision_reporting import native_weight_policy_matches, expected_allocated_weight_bytes


def f16_configuration():
    value = mixed_configuration("q8_0", {"vision": "f16", "fusion": "f32", "decoder": "f32"})
    value["schema_version"] = 4
    value["weights"]["tensor_precisions"] = {"text.resizer.weight": "f16"}
    return value


class MixedF16QuantizationChecks(unittest.TestCase):
    def test_versioned_identity_precedence_and_protection(self):
        value = f16_configuration()
        resolved = normalize_config(value)
        self.assertEqual(resolved, normalize_config(resolved))
        encoded = ("sam3:image-mixed-f16-linear-v1\nbase=q8_0\nvision=f16\ntext=q8_0\n"
                   "fusion=f32\ndecoder=f32\ntensor:text.resizer.weight=f16\n")
        self.assertEqual(resolved["weights"]["policy_sha256"], hashlib.sha256(encoded.encode()).hexdigest())
        policy = {key: val for key, val in resolved["weights"].items() if key not in ("precision", "storage_profile")}
        for name in ("vit.blocks.0.attn.qkv.weight", "vit.blocks.0.mlp.lin2.weight", "text.resizer.weight"):
            self.assertEqual(quantized_tensor_type(name, [2, 4736 if "lin2" in name else 256], "mixed", mixed_policy=policy),
                             gguf.GGMLQuantizationType.F16)
        self.assertIsNone(quantized_tensor_type("text.token_embed.weight", [2, 256], "mixed", mixed_policy=policy))
        for schema in (2, 3):
            changed = deepcopy(value); changed["schema_version"] = schema
            with self.assertRaises(ValueError): normalize_config(changed)
        for changes in ({"tensor_precisions": {}}, {"module_precisions": []},
                        {"storage_profile": "image-tensor-mixed-linear-v1"}, {"policy_sha256": "0" * 64},
                        {"tensor_precisions": {"text.token_embed.weight": "f16"}},
                        {"tensor_precisions": {"geom.points_direct_project.weight": "f16"}}):
            changed = deepcopy(value); changed["weights"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): normalize_config(changed)
        floating = deepcopy(value)
        floating["weights"]["module_precisions"] = dict.fromkeys(("vision", "text", "fusion", "decoder"), "f16")
        floating["compute"]["mode"] = "native-quantized"
        with self.assertRaisesRegex(ValueError, "quantized weights"): normalize_config(floating)

    def test_schema7_metadata_rejects_cross_version_and_corrupt_policies(self):
        weights = normalize_config(f16_configuration())["weights"]
        policy = {key: val for key, val in weights.items() if key not in ("precision", "storage_profile")}
        writer = _MetadataWriter()
        with patch("tools.convert.sam3_gguf.validate_tokenizer"):
            write_metadata(writer, "mixed", "a" * 64, [], [], mixed_policy=policy)
            self.assertEqual(validate_metadata(_MetadataReader(writer.fields), "mixed")[0], "mixed")
            for key, field in (("sam.schema_version", _Field([gguf.GGUFValueType.UINT32], 6)),
                               ("sam.storage_profile", _Field([gguf.GGUFValueType.STRING], "image-mixed-linear-v1")),
                               ("sam.quantization.policy_sha256", _Field([gguf.GGUFValueType.STRING], "0" * 64)),
                               ("sam.quantization.tensor_precisions", _Field([gguf.GGUFValueType.STRING], ""))):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_metadata(_MetadataReader({**writer.fields, key: field}))
        self.assertEqual(writer.fields["sam.schema_version"].contents(), 7)

    def test_real_f16_bytes_plan_and_manifest_agree_without_native_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            output, receipt, state = convert_fixture(Path(directory), f16_configuration())
            manifest = read_json(output.with_suffix(".gguf.manifest.json"))
            self.assertEqual(manifest["sam_schema_version"], 7)
            self.assertEqual(manifest["storage_profile"], F16_MIXED_STORAGE_PROFILE)
            self.assertEqual(Counter(row["dtype"] for row in manifest["tensors"]), {"float16": 3, "q8_0": 1, "float32": 6})
            self.assertEqual(model_weights(manifest), receipt["config"]["weights"])
            self.assertEqual(manifest["output"]["bytes"], manifest["conversion_plan"]["gguf_size"]["exact_bytes"])
            self.assertFalse(manifest["conversion_plan"]["requires_native_encoder_for_conversion"])
            from tools.benchmark.precision_reporting import weight_inventory
            recipe = {"weight_precision": "mixed", "storage_profile": F16_MIXED_STORAGE_PROFILE, "backend": "cpu",
                      "weight_inventory": weight_inventory(manifest),
                      **{key: manifest[key] for key in ("base_precision", "module_precisions", "tensor_precisions", "policy_sha256")}}
            raw = {**recipe, "allocated_weight_bytes": expected_allocated_weight_bytes(recipe)}
            self.assertTrue(native_weight_policy_matches(raw, recipe))
            self.assertFalse(native_weight_policy_matches({**raw, "allocated_weight_bytes": raw["allocated_weight_bytes"] - 2}, recipe))
            self.assertFalse(native_weight_policy_matches({key: val for key, val in raw.items() if key != "allocated_weight_bytes"}, recipe))
            reader = read_gguf(output)
            for tensor in reader.tensors:
                if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                    np.testing.assert_array_equal(tensor.data.reshape(-1), state[tensor.name].numpy().astype(np.float16).reshape(-1))
            self.assertFalse(list(Path(directory).glob(".sam3-*")))

    def test_nonfinite_overflow_and_reduced_source_are_transactionally_rejected(self):
        import torch
        for bad in (float("inf"), float("nan"), 70000.0, "reduced-source"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                source, bpe, output, config, schema, tokenizer, _, _ = create_fixture(Path(directory), f16_configuration())
                state = torch.load(source, weights_only=True)
                key = next(name for name in state if name.endswith("attn.qkv.weight"))
                if bad == "reduced-source": state[key] = state[key].half()
                else: state[key][0, 0] = bad
                torch.save(state, source)
                with patch.object(convert_sam3, "load_tokenizer", return_value=tokenizer), \
                        patch("tools.convert.sam3_artifacts.read_json", return_value=schema), redirect_stdout(io.StringIO()), \
                        self.assertRaises((ValueError, SystemExit)):
                    convert_sam3.main(["--checkpoint", str(source), "--bpe", str(bpe), "--output", str(output),
                                       "--quantization-config", str(config)])
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix(".gguf.manifest.json").exists())
                self.assertFalse(list(Path(directory).glob(".sam3-*")))

    def test_native_cpu_promotes_f16_and_retains_q_in_converted_fixture(self):
        validator = Path(os.environ.get("SAM_TEST_MIXED_VALIDATOR", "build/cpu-precision-tools/tests/test_mixed_quantization"))
        if not validator.is_file(): self.skipTest("build test_mixed_quantization or set SAM_TEST_MIXED_VALIDATOR")
        with tempfile.TemporaryDirectory() as directory:
            output, _, _ = convert_fixture(Path(directory), f16_configuration())
            completed = subprocess.run(["rtk", "proxy", str(validator.resolve()), "--fixture", str(output)],
                                       capture_output=True, text=True, check=True)
            self.assertIn("mixed CPU tensors=10", completed.stdout)

    def test_real_f16_and_k_formats_share_one_conversion_and_native_graph(self):
        helper = Path(os.environ.get("SAM_TEST_QUANTIZER", "build/cpu-precision-tools/examples/sam_quantize_rows"))
        validator = Path(os.environ.get("SAM_TEST_MIXED_VALIDATOR", "build/cpu-precision-tools/tests/test_mixed_quantization"))
        if not helper.is_file() or not validator.is_file(): self.skipTest("build the quantizer and mixed validator")
        config = f16_configuration()
        config["weights"].update(base_precision="q4_k", module_precisions={"vision": "f16", "text": "q8_0", "fusion": "q6_k", "decoder": "q5_k"})
        config["weights"]["tensor_precisions"]["vit.blocks.0.attn.qkv.weight"] = "q4_k"
        with tempfile.TemporaryDirectory() as directory:
            output, _, _ = convert_fixture(Path(directory), config, helper.resolve())
            manifest = read_json(output.with_suffix(".gguf.manifest.json"))
            self.assertEqual(Counter(row["dtype"] for row in manifest["tensors"]),
                             {"float16": 2, "q4_k": 1, "q8_0": 1, "q6_k": 1, "q5_k": 1, "float32": 4})
            completed = subprocess.run(["rtk", "proxy", str(validator.resolve()), "--fixture", str(output)],
                                       capture_output=True, text=True, check=True)
            self.assertIn("mixed CPU tensors=10", completed.stdout)
