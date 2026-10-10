"""Module mixed policies, real CPU conversion and native fixture interoperability."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import io
import itertools
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import gguf
import numpy as np

from tests.tools.test_modular_quantization_tools import _Field, _MetadataReader, _MetadataWriter
from tests.tools.test_quantization_performance import process_record
from tools.benchmark.precision_reporting import expected_runtime_profile, native_recipe
from tools.benchmark.quantization_benchmark import main as benchmark_main
from tools.benchmark.quantization_performance import validate_record
from tools.convert import convert_sam3
from tools.convert.sam3_artifacts import SAM3_REVISION, read_json, sha256_file, write_json
from tools.convert.sam3_gguf import (bytes_to_unicode, canonical_shape, inspect_tensors, quantized_tensor_type,
                                    read_gguf, validate_metadata, write_metadata)
from tools.quantize.quantization_config import (configuration_hash, configuration_receipt, conversion_options,
                                               load_config, normalize_config, validate_recipe_configuration)
from tools.quantize.weight_policy import (MIXED_STORAGE_PROFILE, QUANTIZATION_MODULES, QUANTIZATION_PROFILES,
                                         mixed_quantization_policy, parse_mixed_module_precisions)
from tools.validation.ranked_outputs import canonical_hash


ROOT = Path(__file__).resolve().parents[2]


def mixed_configuration(base="q4_k", overrides=None):
    return {"schema_version": 2, "kind": "sam-quantization-config", "task": "image", "backend": "cpu",
            "weights": {"precision": "mixed", "base_precision": base, "module_precisions": overrides or {}},
            "activation": {"mode": "backend-selected"}, "compute": {"mode": "f32"}, "cache": {"mode": "f32"}}


def create_fixture(root, config):
    """Create a tiny original F32 checkpoint; substitute only SAM shapes/tokenizer source."""
    import torch
    shapes = {"vit.blocks.0.attn.qkv.weight": (2, 256), "vit.blocks.0.mlp.lin2.weight": (2, 4736),
              "text.blocks.0.attn.in_proj.weight": (2, 256), "text.resizer.weight": (2, 256),
              "fenc.layers.0.linear1.weight": (2, 256), "ddec.layers.0.linear1.weight": (2, 256),
              "geom.points_direct_project.weight": (2, 2), "ddec.presence_token_head.layers.2.weight": (1, 256),
              "text.token_embed.weight": (2, 256), "ddec.norm.bias": (2,)}
    state = {name: torch.linspace(-.37, .51, int(np.prod(shape)), dtype=torch.float32).reshape(shape)
             for name, shape in shapes.items()}
    schema = {"schema_version": 1, "sam3_revision": SAM3_REVISION,
              "tensors": {name: list(reversed(canonical_shape(shape))) for name, shape in shapes.items()},
              "unused_tracker_tensors": {}}
    source, bpe, output, path = (root / name for name in ("source.pt", "bpe.gz", "mixed.gguf", "config.json"))
    def source_name(name):
        for target, original in (("vit.", "detector.backbone.vision_backbone.trunk."),
                                 ("text.blocks.", "detector.backbone.language_backbone.encoder.transformer.resblocks."),
                                 ("text.resizer.", "detector.backbone.language_backbone.resizer."),
                                 ("text.token_embed.", "detector.backbone.language_backbone.encoder.token_embedding."),
                                 ("fenc.layers.", "detector.transformer.encoder.layers."),
                                 ("ddec.", "detector.transformer.decoder."),
                                 ("geom.", "detector.geometry_encoder.")):
            if name.startswith(target):
                name = original + name[len(target):]
                return name.replace(".attn.in_proj.weight", ".attn.in_proj_weight").replace(".mlp.lin2.", ".mlp.fc2.")
        raise ValueError("unmapped fixture source")
    torch.save({source_name(name): tensor for name, tensor in state.items()}, source)
    bpe.write_bytes(b"fixture; tokenizer source is substituted")
    write_json(path, config)
    receipt = load_config(path)
    base = list(bytes_to_unicode().values())
    merges = [(left, right) for left in base for right in base][:48894]
    vocab = base + [token + "</w>" for token in base] + [left + right for left, right in merges]
    vocab += ["<start_of_text>", "<end_of_text>"]
    return source, bpe, output, path, schema, (vocab, merges), receipt, state


def convert_fixture(root, config, quantizer=None):
    source, bpe, output, path, schema, tokenizer, receipt, state = create_fixture(root, config)
    arguments = ["--checkpoint", str(source), "--bpe", str(bpe), "--output", str(output), "--quantization-config", str(path)]
    if quantizer is not None:
        arguments.extend(["--quantizer", str(quantizer)])
    with patch.object(convert_sam3, "load_tokenizer", return_value=tokenizer), \
            patch("tools.convert.sam3_artifacts.read_json", return_value=schema), redirect_stdout(io.StringIO()):
        convert_sam3.main(arguments)
    return output, receipt, state


class MixedQuantizationChecks(unittest.TestCase):
    def test_module_combinations_resolve_canonically_and_bind_base_and_protections(self):
        for base in QUANTIZATION_PROFILES:
            for formats in itertools.product(("f32", *QUANTIZATION_PROFILES), repeat=4):
                config = mixed_configuration(base, dict(zip(reversed(QUANTIZATION_MODULES), reversed(formats))))
                result = normalize_config(config)
                self.assertEqual(list(result["weights"]["module_precisions"]), list(QUANTIZATION_MODULES))
                self.assertEqual(result, normalize_config(result))
                policy = mixed_quantization_policy(base, dict(zip(QUANTIZATION_MODULES, formats)))
                self.assertEqual(result["weights"]["policy_sha256"], policy["policy_sha256"])
        policy = mixed_quantization_policy("q4_k", {"text": "q8_0", "fusion": "q6_k", "decoder": "q5_k"})
        self.assertEqual(policy["policy_sha256"], "2e892b88037662acaab934e1cc229225f719b282055197cdcf0e5a619537da9f")
        self.assertEqual(normalize_config(mixed_configuration())["weights"]["module_precisions"],
                         {module: "q4_k" for module in QUANTIZATION_MODULES})
        resolved = normalize_config(mixed_configuration())
        same = deepcopy(resolved)
        same["weights"]["module_precisions"] = dict(reversed(list(same["weights"]["module_precisions"].items())))
        self.assertEqual(configuration_hash(resolved), configuration_hash(normalize_config(same)))

    def test_invalid_mixed_schema_fields_and_precision_conflicts_are_rejected(self):
        invalid = []
        for changes in ({"base_precision": "f32"}, {"base_precision": "f16"}, {"precision": "q4_k"},
                        {"module_precisions": []}, {"module_precisions": {"tracker": "q8_0"}},
                        {"module_precisions": {"text": "f16"}}, {"module_precisions": {"text": 8}},
                        {"tensor_precisions": {}}, {"modules": ["vision"]},
                        {"storage_profile": "image-full-linear-q4_k-v1"}, {"policy_sha256": "0" * 64}):
            value = mixed_configuration()
            value["weights"].update(changes)
            invalid.append(value)
        invalid.append({**mixed_configuration(), "schema_version": 1})
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_config(value)
        for csv in ("", "vision=q4_k", "vision=q4_k,text=q8_0,fusion=q6_k,decoder=q5_k,", "x" * 257,
                    "text=q8_0,vision=q4_k,fusion=q6_k,decoder=q5_k", "vision=q4_k,text=q8_0,fusion=q6_k,decoder=f16"):
            with self.subTest(csv=csv), self.assertRaises(ValueError):
                parse_mixed_module_precisions(csv)

    def test_real_sam_inventory_obeys_module_formats_and_mandatory_f32_rules(self):
        schema = read_json(ROOT / "tools/convert/sam3_tensor_schema.json")["tensors"]
        policy = mixed_quantization_policy("q4_k", {"text": "q8_0", "fusion": "q6_k", "decoder": "q5_k"})
        counts = Counter()
        for name, dimensions in schema.items():
            qtype = quantized_tensor_type(name, list(reversed(dimensions)), "mixed", mixed_policy=policy)
            counts[qtype.name if qtype is not None else "F32"] += 1
        self.assertEqual(counts, {"F32": 785, "Q4_K": 96, "Q8_0": 129, "Q6_K": 36, "Q5_K": 87})
        for precision in ("f32", *QUANTIZATION_PROFILES):
            for module in QUANTIZATION_MODULES:
                policy = mixed_quantization_policy("q4_k", {key: precision if key == module else "f32" for key in QUANTIZATION_MODULES})
                for name, dimensions in schema.items():
                    qtype = quantized_tensor_type(name, list(reversed(dimensions)), "mixed", mixed_policy=policy)
                    if qtype is not None:
                        self.assertNotEqual(precision, "f32")
                        self.assertTrue(name.startswith({"vision": ("vit.",), "text": ("text.",),
                                                         "fusion": ("fenc.",), "decoder": ("ddec.", "geom.", "scoring.", "seg.")}[module]))

    def test_metadata_types_canonical_policy_hash_and_legacy_schema_boundary(self):
        policy = mixed_quantization_policy("q4_k", {"text": "q8_0", "decoder": "f32"})
        writer = _MetadataWriter()
        with patch("tools.convert.sam3_gguf.validate_tokenizer"):
            write_metadata(writer, "mixed", "a" * 64, [], [], mixed_policy=policy)
            self.assertEqual(validate_metadata(_MetadataReader(writer.fields), "mixed", mixed_policy=policy)[0], "mixed")
            for key, field in (("sam.quantization.base_precision", _Field([gguf.GGUFValueType.STRING], "f16")),
                               ("sam.quantization.module_precisions", _Field([gguf.GGUFValueType.UINT32], 1)),
                               ("sam.quantization.policy_sha256", _Field([gguf.GGUFValueType.STRING], "0" * 64)),
                               ("general.file_type", _Field([gguf.GGUFValueType.UINT32], 7)),
                               ("sam.quantization.modules", _Field([gguf.GGUFValueType.STRING], "vision")),
                               ("sam.schema_version", _Field([gguf.GGUFValueType.UINT32], 4))):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_metadata(_MetadataReader({**writer.fields, key: field}))
            for key in ("sam.quantization.base_precision", "sam.quantization.module_precisions", "sam.quantization.policy_sha256"):
                fields = {name: field for name, field in writer.fields.items() if name != key}
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    validate_metadata(_MetadataReader(fields))
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_metadata(_MetadataReader(writer.fields), mixed_policy=mixed_quantization_policy("q4_k", {}))
        self.assertEqual(writer.fields["sam.schema_version"].contents(), 5)
        self.assertNotIn("sam.quantization.modules", writer.fields)

    def test_tiny_q8_f32_conversion_inspection_and_runtime_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, receipt, _ = convert_fixture(root, mixed_configuration("q8_0", {"fusion": "f32", "decoder": "f32"}))
            manifest = read_json(output.with_suffix(".gguf.manifest.json"))
            self.assertEqual(manifest["sam_schema_version"], 5)
            self.assertEqual(manifest["quantization_modules"], ["vision", "text"])
            self.assertEqual(manifest["module_precisions"], receipt["config"]["weights"]["module_precisions"])
            self.assertEqual(Counter(row["dtype"] for row in manifest["tensors"]), {"q8_0": 4, "float32": 6})
            self.assertEqual(manifest["configuration_application"]["compute"], "runtime-request-only")
            for row in manifest["tensors"]:
                self.assertIn("selector", row)
                self.assertIn("requested_dtype", row)
                self.assertIn("resolved_dtype", row)
                self.assertIn("quantization_reason", row)
            description = root / "description.json"
            with redirect_stdout(io.StringIO()):
                benchmark_main(["inspect-precision", "--model", str(output), "--quantization-config", str(root / "config.json"),
                                "--output", str(description)])
            report = read_json(description)
            self.assertFalse(report["inference_executed"])
            self.assertEqual(report["precision"]["weights"]["policy_sha256"], manifest["policy_sha256"])
            self.assertEqual(report["precision"]["kernel_precision_evidence"], "NOT_COLLECTED")
            recipe = native_recipe(argparse.Namespace(model=output, binary=output, backend="cpu", compute="f32", cache="f32",
                                                      quantization_configuration=receipt), collect_environment=False)
            validate_recipe_configuration(recipe)
            altered = deepcopy(recipe)
            altered["module_precisions"]["text"] = "f32"
            with self.assertRaises(ValueError):
                validate_recipe_configuration(altered)
            case = {"prompt": "person", "alternate": "forklift"}
            protocol = {"latency": {"warmups": 0, "iterations": 2}}
            row = process_record(recipe, case, "latency", protocol, 1)
            row.update({key: recipe[key] for key in ("base_precision", "module_precisions", "policy_sha256")})
            row.update(recipe_sha256=canonical_hash(recipe), case_sha256=canonical_hash(case), protocol_sha256=canonical_hash(protocol))
            validate_record(row, recipe, case, protocol)
            for key in ("base_precision", "module_precisions", "policy_sha256"):
                changed = deepcopy(row)
                del changed[key]
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    validate_record(changed, recipe, case, protocol)

    def test_native_k_conversion_interoperates_with_independent_cpu_execution(self):
        helper = Path(os.environ.get("SAM_TEST_QUANTIZER", ROOT / "build/cpu-precision-tools/examples/sam_quantize_rows"))
        validator = Path(os.environ.get("SAM_TEST_MIXED_VALIDATOR", ROOT / "build/cpu-precision-tools/tests/test_mixed_quantization"))
        if not helper.is_file() or not validator.is_file():
            self.skipTest("build CPU sam_quantize_rows/test_mixed_quantization or set SAM_TEST_QUANTIZER/SAM_TEST_MIXED_VALIDATOR")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, receipt, state = convert_fixture(root, mixed_configuration(overrides={"text": "q8_0", "fusion": "q6_k", "decoder": "q5_k"}), helper.resolve())
            manifest = read_json(output.with_suffix(".gguf.manifest.json"))
            self.assertEqual(Counter(row["dtype"] for row in manifest["tensors"]),
                             {"float32": 4, "q4_k": 1, "q8_0": 3, "q6_k": 1, "q5_k": 1})
            self.assertEqual(manifest["quantizer"]["implementation"], "ggml_quantize_chunk")
            self.assertEqual(manifest["quantizer"]["helper"]["sha256"], sha256_file(helper))
            reader = read_gguf(output)
            for tensor in reader.tensors:
                original = state[tensor.name].numpy().reshape(-1)
                decoded = (gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(-1)
                           if tensor.tensor_type.name.startswith("Q") else tensor.data.reshape(-1))
                self.assertLessEqual(float(np.max(np.abs(decoded - original))), .06)
            schema = {tensor.name: list(map(int, tensor.shape)) for tensor in reader.tensors}
            policy = {key: receipt["config"]["weights"][key] for key in ("base_precision", "module_precisions", "policy_sha256")}
            inspect_tensors(reader, "mixed", schema, storage_profile=MIXED_STORAGE_PROFILE, mixed_policy=policy)
            wrong = mixed_quantization_policy("q8_0", {})
            with self.assertRaisesRegex(ValueError, "dtype"):
                inspect_tensors(reader, "mixed", schema, mixed_policy=wrong)
            completed = subprocess.run(["rtk", "proxy", str(validator.resolve()), "--fixture", str(output)],
                                       capture_output=True, text=True, check=True)
            self.assertIn("mixed CPU tensors=10", completed.stdout)
            self.assertFalse(list(root.glob(".sam3-*")))
            # A Q8 base must still request a native encoder and report the K
            # fallback when vision is explicitly overridden to a K format.
            other = root / "q8-base"
            other.mkdir()
            other_output, _, _ = convert_fixture(other, mixed_configuration("q8_0", {"vision": "q4_k", "text": "q6_k", "fusion": "q5_k"}), helper.resolve())
            other_manifest = read_json(other_output.with_suffix(".gguf.manifest.json"))
            self.assertEqual(other_manifest["quantization"]["gguf_file_type"], 7)
            self.assertEqual(other_manifest["options"]["row_block_fallback"], "Q8_0")
            self.assertEqual(other_manifest["quantizer"]["implementation"], "ggml_quantize_chunk")
            subprocess.run(["rtk", "proxy", str(validator.resolve()), "--fixture", str(other_output)],
                           check=True, capture_output=True)

    def test_all_f32_overrides_keep_mixed_policy_but_do_not_claim_quantized_arithmetic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, receipt, _ = convert_fixture(root, mixed_configuration(overrides={module: "f32" for module in QUANTIZATION_MODULES}))
            manifest = read_json(output.with_suffix(".gguf.manifest.json"))
            self.assertEqual(manifest["precision"], "mixed")
            self.assertEqual(manifest["quantization_modules"], [])
            self.assertEqual(manifest["quantizer"]["implementation"], "none-f32-only-policy")
            self.assertFalse(manifest["options"]["quantized_image_linear_weights"])
            self.assertIsNone(manifest["options"]["row_block_fallback"])
            recipe = native_recipe(argparse.Namespace(model=output, binary=output, backend="cpu", compute="f32", cache="f32",
                                                      quantization_configuration=receipt), collect_environment=False)
            self.assertEqual(expected_runtime_profile(recipe), "")
            validator = ROOT / "build/cpu-precision-tools/tests/test_mixed_quantization"
            if validator.is_file():
                subprocess.run(["rtk", "proxy", str(validator), "--fixture", str(output)], check=True, capture_output=True)

    def test_mixed_conversion_rejects_reduced_source_and_missing_native_encoder(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = configuration_receipt(mixed_configuration())
            options = conversion_options(receipt)
            with self.assertRaisesRegex(ValueError, "quantizer"):
                convert_sam3.convert(root / "missing.pt", root / "missing.bpe", options["precision"], root / "never.gguf",
                                     storage_profile=options["storage_profile"], configuration=receipt)
            torch.save({"detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight": torch.ones(2, 256, dtype=torch.float16)}, root / "source.pt")
            receipt = configuration_receipt(mixed_configuration("q8_0"))
            with patch.object(convert_sam3, "load_tokenizer", return_value=([], [])), \
                    self.assertRaisesRegex(ValueError, "original F32"):
                convert_sam3.convert(root / "source.pt", root / "missing.bpe", "mixed", root / "never.gguf", configuration=receipt)
            self.assertFalse((root / "never.gguf").exists())
            self.assertFalse(list(root.glob(".sam3-*")))


if __name__ == "__main__":
    unittest.main()
