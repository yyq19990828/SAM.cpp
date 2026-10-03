"""Focused schema-4 SAM 3 modular quantization tooling checks."""

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gguf

from convert_sam3 import convert
from sam3_artifacts import read_json
from sam3_gguf import (QUANTIZATION_MODULES, canonical_quantization_modules,
                       quantization_module_for_linear_weight, quantization_module_for_tensor,
                       quantization_profile, quantization_reason_for_tensor,
                       quantized_tensor_type, validate_metadata, write_metadata)


class _Field:
    def __init__(self, types, contents):
        self.types = types
        self._contents = contents

    def contents(self):
        return self._contents


class _MetadataWriter:
    def __init__(self):
        self.fields = {"general.architecture": _Field([gguf.GGUFValueType.STRING], "sam3")}

    def add_custom_alignment(self, value):
        self.fields["general.alignment"] = _Field([gguf.GGUFValueType.UINT32], value)

    def add_file_type(self, value):
        self.fields["general.file_type"] = _Field([gguf.GGUFValueType.UINT32], value)

    def add_string(self, key, value):
        self.fields[key] = _Field([gguf.GGUFValueType.STRING], value)

    def add_uint32(self, key, value):
        self.fields[key] = _Field([gguf.GGUFValueType.UINT32], value)

    def add_key_value(self, key, value, value_type, sub_type=None):
        types = [value_type, sub_type] if value_type == gguf.GGUFValueType.ARRAY else [value_type]
        self.fields[key] = _Field(types, value)

    def add_token_list(self, tokens):
        self.fields["tokenizer.ggml.tokens"] = _Field([gguf.GGUFValueType.ARRAY,
                                                       gguf.GGUFValueType.STRING], tokens)

    def add_token_merges(self, merges):
        self.fields["tokenizer.ggml.merges"] = _Field([gguf.GGUFValueType.ARRAY,
                                                       gguf.GGUFValueType.STRING], merges)


class _MetadataReader:
    def __init__(self, fields):
        self.fields = fields

    def get_field(self, key):
        return self.fields.get(key)


class ModularQuantizationChecks(unittest.TestCase):
    @staticmethod
    def schema_tensors():
        return read_json(Path(__file__).with_name("sam3_tensor_schema.json"))["tensors"]

    @staticmethod
    def qtype_name(value):
        return None if value is None else value.name

    def test_module_mapping_and_exact_linear_tensor_inventory(self):
        schema = self.schema_tensors()
        expected = {
            "vision": (128, 444_596_224),
            "text": (97, 302_252_032),
            "fusion": (36, 9_437_184),
            "decoder": (87, 18_027_520),
        }
        counts = {}
        for module in QUANTIZATION_MODULES:
            selected = []
            for name, ggml_shape in schema.items():
                source_shape = list(reversed(ggml_shape))
                qtype = quantized_tensor_type(name, source_shape, "q8_0",
                                              quantization_modules=[module])
                if qtype is not None:
                    selected.append((name, math.prod(ggml_shape)))
            counts[module] = (len(selected), sum(params for _, params in selected))
            self.assertEqual(counts[module], expected[module])
            self.assertTrue(all(quantization_module_for_tensor(name) == module
                                for name, _ in selected))

        self.assertEqual(sum(count for count, _ in counts.values()), 348)
        self.assertEqual(sum(params for _, params in counts.values()), 774_312_960)
        self.assertEqual(quantization_module_for_tensor("neck.det.blocks.0.weight"), "vision")
        self.assertEqual(quantization_module_for_tensor("fenc.layers.0.sa.in_proj_weight"), "fusion")
        self.assertEqual(quantization_module_for_tensor("geom.layers.0.sa.in_proj_weight"), "decoder")
        self.assertIsNone(quantization_module_for_tensor("unknown.weight"))

    def test_module_csv_canonicalization_and_profile_identity(self):
        self.assertEqual(canonical_quantization_modules("decoder,vision"), ["vision", "decoder"])
        self.assertEqual(canonical_quantization_modules("decoder, text"), ["text", "decoder"])
        for invalid in ("", ",vision", "vision,", "vision,vision", "vision,tracker", []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                canonical_quantization_modules(invalid)

        custom = quantization_profile("q6_k", quantization_modules="decoder,vision")
        self.assertEqual(custom["schema_version"], 4)
        self.assertEqual(custom["storage_profile"], "image-modules-linear-q6_k-v1")
        self.assertEqual(custom["modules"], ["vision", "decoder"])
        self.assertEqual(custom["modules_csv"], "vision,decoder")
        self.assertEqual(custom["profile_status"], "diagnostic")

        custom_all = quantization_profile("q8_0", quantization_modules=QUANTIZATION_MODULES)
        self.assertEqual(custom_all["storage_profile"], "image-modules-linear-q8_0-v1")
        self.assertEqual(custom_all["profile_status"], "diagnostic")
        full = quantization_profile("q8_0", "image-full-linear-q8_0-v1")
        self.assertEqual(full["schema_version"], 4)
        self.assertEqual(full["modules"], list(QUANTIZATION_MODULES))
        self.assertEqual(full["profile_status"], "candidate")
        self.assertEqual(quantization_profile("q8_0")["schema_version"], 3)
        self.assertIsNone(quantization_profile("q8_0")["modules"])

        with self.assertRaisesRegex(ValueError, "all four"):
            quantization_profile("q8_0", "image-full-linear-q8_0-v1", "vision")
        with self.assertRaisesRegex(ValueError, "requires quantization modules"):
            quantization_profile("q8_0", "image-modules-linear-q8_0-v1")

    def test_overlong_csv_is_rejected_before_split(self):
        with self.assertRaisesRegex(ValueError, "CSV is too long"):
            canonical_quantization_modules("vision," * 1000)
        with self.assertRaisesRegex(ValueError, "short"):
            canonical_quantization_modules([" " * 100_000])
        # Bounded surrounding whitespace remains accepted and normalized.
        self.assertEqual(canonical_quantization_modules(" vision , text "), ["vision", "text"])

    def test_large_duplicate_list_reads_only_five_items(self):
        class CountedList(list):
            def __init__(self, values):
                super().__init__(values)
                self.yielded = 0

            def __iter__(self):
                for value in super().__iter__():
                    self.yielded += 1
                    yield value

        values = CountedList(["vision"] * 100_000)
        with self.assertRaisesRegex(ValueError, "at most 4"):
            canonical_quantization_modules(values)
        self.assertEqual(values.yielded, 5)

    def test_infinite_generator_is_bounded_to_five_items(self):
        yielded = 0

        def repeated_modules():
            nonlocal yielded
            while True:
                yielded += 1
                yield "vision"

        with self.assertRaisesRegex(ValueError, "at most 4"):
            canonical_quantization_modules(repeated_modules())
        self.assertEqual(yielded, 5)

    def test_full_and_custom_module_profiles_quantize_only_selected_matrices(self):
        schema = self.schema_tensors()
        profile_names = {
            "q8_0": (gguf.GGMLQuantizationType.Q8_0, None),
            "q6_k": (gguf.GGMLQuantizationType.Q6_K, gguf.GGMLQuantizationType.Q8_0),
            "q5_k": (gguf.GGMLQuantizationType.Q5_K, gguf.GGMLQuantizationType.Q8_0),
            "q4_k": (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q8_0),
        }
        expected_types = {
            "q8_0": (348, 348, 0),
            "q6_k": (348, 316, 32),
            "q5_k": (348, 316, 32),
            "q4_k": (348, 316, 32),
        }
        for precision, (main_type, fallback_type) in profile_names.items():
            with self.subTest(precision=precision):
                profile = quantization_profile(precision, f"image-full-linear-{precision}-v1")
                actual = []
                for name, ggml_shape in schema.items():
                    qtype = quantized_tensor_type(name, list(reversed(ggml_shape)), precision,
                                                  profile["storage_profile"], profile["modules"])
                    if qtype is not None:
                        actual.append(qtype)
                expected_total, expected_main, expected_fallback = expected_types[precision]
                self.assertEqual(len(actual), expected_total)
                self.assertEqual(sum(qtype == main_type for qtype in actual), expected_main)
                if fallback_type is not None:
                    self.assertEqual(sum(qtype == fallback_type for qtype in actual), expected_fallback)

                custom = quantization_profile(precision, quantization_modules="vision,fusion")
                custom_types = [
                    quantized_tensor_type(name, list(reversed(shape)), precision,
                                          custom["storage_profile"], custom["modules"])
                    for name, shape in schema.items()
                ]
                self.assertEqual(sum(qtype is not None for qtype in custom_types), 164)
                self.assertEqual(sum(qtype is not None for (name, _), qtype
                                     in zip(schema.items(), custom_types)
                                     if name.startswith("text.")), 0)

    def test_float_exceptions_and_row_fallback_are_explicit(self):
        all_modules = list(QUANTIZATION_MODULES)
        self.assertIsNone(quantized_tensor_type("text.token_embed.weight", (49_408, 1_024), "q8_0",
                                                quantization_modules=all_modules))
        self.assertIsNone(quantized_tensor_type("ddec.query_embed.weight", (200, 256), "q8_0",
                                                quantization_modules=all_modules))
        self.assertIsNone(quantized_tensor_type("ddec.reference_points.weight", (200, 4), "q8_0",
                                                quantization_modules=all_modules))
        self.assertIsNone(quantized_tensor_type("geom.label_embed.weight", (2, 256), "q8_0",
                                                quantization_modules=all_modules))
        self.assertIsNone(quantized_tensor_type("vit.patch_embed.weight", (1, 3, 14, 14, 1_024), "q8_0",
                                                quantization_modules=all_modules))
        for precision in ("q8_0", "q6_k", "q5_k", "q4_k"):
            with self.subTest(precision=precision):
                self.assertIsNone(quantized_tensor_type("geom.points_direct_project.weight", (256, 2),
                                                        precision, quantization_modules=all_modules))
                self.assertIsNone(quantized_tensor_type("geom.boxes_direct_project.weight", (256, 4),
                                                        precision, quantization_modules=all_modules))
                self.assertIsNone(quantized_tensor_type("geom.boxes_pos_enc_project.weight", (256, 258),
                                                        precision, quantization_modules=all_modules))
                self.assertIsNone(quantized_tensor_type("ddec.boxRPB_embed_x.layers.0.weight", (256, 2),
                                                        precision, quantization_modules=all_modules))
        for precision in ("q6_k", "q5_k", "q4_k"):
            self.assertEqual(quantized_tensor_type("vit.blocks.0.mlp.lin2.weight", (1_024, 4_736),
                                                   precision, quantization_modules=all_modules),
                             gguf.GGMLQuantizationType.Q8_0)
            self.assertIsNone(quantized_tensor_type("vit.blocks.0.attn.qkv.weight", (3_072, 1_000),
                                                    precision, quantization_modules=all_modules))

        presence_name = "ddec.presence_token_head.layers.2.weight"
        self.assertEqual(quantization_module_for_tensor(presence_name), "decoder")
        self.assertIsNone(quantization_module_for_linear_weight(presence_name))
        self.assertIsNone(quantized_tensor_type(presence_name, (1, 256), "q8_0",
                                                quantization_modules=all_modules))
        self.assertEqual(quantization_reason_for_tensor(presence_name, (1, 256), "q8_0",
                                                        "image-modules-linear-q8_0-v1", all_modules),
                         "canonical-one-dimensional")
        self.assertEqual(quantization_reason_for_tensor("geom.points_direct_project.weight", (256, 2),
                                                        "q6_k", "image-modules-linear-q6_k-v1", all_modules),
                         "row-block-ineligible")

    def test_module_indices_are_ascii_canonical_and_within_layer_bounds(self):
        all_modules = list(QUANTIZATION_MODULES)
        invalid_names = (
            "vit.blocks.01.attn.qkv.weight", "vit.blocks.32.attn.qkv.weight",
            "vit.blocks.٠.attn.qkv.weight", "text.blocks.24.attn.in_proj.weight",
            "fenc.layers.6.sa.in_proj_weight", "ddec.layers.6.sa.in_proj_weight",
            "geom.layers.3.sa.in_proj_weight", "scoring.prompt_mlp.layers.2.weight",
            "seg.mask_predictor.mask_embed.layers.03.weight",
        )
        for name in invalid_names:
            with self.subTest(name=name):
                self.assertIsNone(quantization_module_for_linear_weight(name))
                self.assertIsNone(quantized_tensor_type(name, (3_072, 1_024), "q8_0",
                                                        quantization_modules=all_modules))

    def test_schema4_module_metadata_is_canonical_and_validated(self):
        class Writer:
            def __init__(self):
                self.fields = {"general.architecture": _Field([gguf.GGUFValueType.STRING], "sam3")}

            def add_custom_alignment(self, value):
                self.fields["general.alignment"] = _Field([gguf.GGUFValueType.UINT32], value)

            def add_file_type(self, value):
                self.fields["general.file_type"] = _Field([gguf.GGUFValueType.UINT32], value)

            def add_string(self, key, value):
                self.fields[key] = _Field([gguf.GGUFValueType.STRING], value)

            def add_uint32(self, key, value):
                self.fields[key] = _Field([gguf.GGUFValueType.UINT32], value)

            def add_key_value(self, key, value, value_type, sub_type=None):
                types = [value_type, sub_type] if value_type == gguf.GGUFValueType.ARRAY else [value_type]
                self.fields[key] = _Field(types, value)

            def add_token_list(self, values):
                self.fields["tokenizer.ggml.tokens"] = _Field([gguf.GGUFValueType.ARRAY,
                                                               gguf.GGUFValueType.STRING], values)

            def add_token_merges(self, values):
                self.fields["tokenizer.ggml.merges"] = _Field([gguf.GGUFValueType.ARRAY,
                                                               gguf.GGUFValueType.STRING], values)

        class Reader:
            def __init__(self, fields):
                self.fields = fields

            def get_field(self, key):
                return self.fields.get(key)

        modules = ["vision", "fusion"]
        writer = Writer()
        with patch("sam3_gguf.validate_tokenizer"):
            write_metadata(writer, "q6_k", "a" * 64, [], [], storage_profile="image-modules-linear-q6_k-v1",
                           quantization_modules=modules)
            actual, tokens, merges = validate_metadata(Reader(writer.fields), "q6_k", "a" * 64,
                                                       "image", "image-modules-linear-q6_k-v1", modules)
        self.assertEqual(actual, "q6_k")
        self.assertEqual(tokens, [])
        self.assertEqual(merges, [])
        self.assertEqual(writer.fields["sam.schema_version"].contents(), 4)
        self.assertEqual(writer.fields["sam.quantization.modules"].types, [gguf.GGUFValueType.STRING])
        self.assertEqual(writer.fields["sam.quantization.modules"].contents(), "vision,fusion")
        self.assertEqual(writer.fields["sam.storage_profile"].contents(), "image-modules-linear-q6_k-v1")
        self.assertNotIn("sam.quantization.module_version", writer.fields)
        self.assertNotIn("sam.quantization.preset", writer.fields)

        full = Writer()
        with patch("sam3_gguf.validate_tokenizer"):
            write_metadata(full, "q8_0", "b" * 64, [], [], storage_profile="image-full-linear-q8_0-v1",
                           quantization_modules=list(QUANTIZATION_MODULES))
            validate_metadata(Reader(full.fields), "q8_0", "b" * 64, "image",
                              "image-full-linear-q8_0-v1", list(QUANTIZATION_MODULES))
        self.assertEqual(full.fields["sam.quantization.modules"].contents(), "vision,text,fusion,decoder")

        noncanonical = dict(writer.fields)
        noncanonical["sam.quantization.modules"] = _Field([gguf.GGUFValueType.STRING], "fusion,vision")
        with patch("sam3_gguf.validate_tokenizer"), self.assertRaisesRegex(ValueError, "canonical order"):
            validate_metadata(Reader(noncanonical), "q6_k", "a" * 64, "image",
                              "image-modules-linear-q6_k-v1", modules)

        wrong_type = dict(writer.fields)
        wrong_type["sam.quantization.modules"] = _Field([gguf.GGUFValueType.UINT32], 1)
        with patch("sam3_gguf.validate_tokenizer"), self.assertRaisesRegex(ValueError, "incompatible type"):
            validate_metadata(Reader(wrong_type), "q6_k", "a" * 64, "image",
                              "image-modules-linear-q6_k-v1", modules)

        wrong_file_type = dict(writer.fields)
        wrong_file_type["general.file_type"] = _Field([gguf.GGUFValueType.UINT32], 7)
        with patch("sam3_gguf.validate_tokenizer"), self.assertRaisesRegex(ValueError, "file type"):
            validate_metadata(Reader(wrong_file_type), "q6_k", "a" * 64, "image",
                              "image-modules-linear-q6_k-v1", modules)

    def test_cli_selector_conflicts_fail_before_checkpoint_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "never-created.gguf"
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                convert("missing.pt", "missing.bpe", "q8_0", output,
                        storage_profile="image-full-linear-q8_0-v1", quantize_modules="vision")
            with self.assertRaisesRegex(ValueError, "requires a quantized image precision"):
                convert("missing.pt", "missing.bpe", "f32", output, quantize_modules="vision")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
