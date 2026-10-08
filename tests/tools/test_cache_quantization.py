"""Cache layout, numerical coding, validation and alias behavior."""

import unittest

import numpy as np

from tools.quantize.cache_quantization import CacheQuantization, PackedFeature, decode_feature, encode_feature


class CacheQuantizationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)

    def test_channel_rows_and_noncontiguous_roundtrip(self):
        torch = self.torch
        values = torch.arange(2 * 64 * 3 * 5, dtype=torch.float32).reshape(2, 64, 3, 5).transpose(2, 3)
        for mode in ("cache-f32", "cache-f16"):
            packed = encode_feature(values, mode, torch)
            np.testing.assert_array_equal(packed.values[0, 1, 2].numpy(), values[0, :, 1, 2].numpy())
            self.assertTrue(torch.equal(decode_feature(packed, torch), values))
            self.assertEqual(packed.payload_bytes, values.numel() * (4 if mode == "cache-f32" else 2))

    def test_q8_block_layout_rounding_and_half_scales(self):
        torch = self.torch
        data = np.zeros((1, 64, 2, 1), np.float32)
        data[0, :8, 0, 0] = [127, 0.5, 1.5, -0.5, -1.5, np.nextafter(np.float32(0.5), np.float32(0)), -127, 0]
        data[0, 32:, 0, 0] = np.linspace(-0.3, 0.6, 32, dtype=np.float32)
        packed = encode_feature(torch.from_numpy(data), "cache-q8_0", torch)
        self.assertEqual(packed.values[0, :8].tolist(), [127, 1, 2, -1, -2, 0, -127, 0])
        np.testing.assert_array_equal(packed.scales[2:].numpy(), np.zeros((2, 1), np.float16))
        rows = data.transpose(0, 2, 3, 1).reshape(-1, 32)
        maximum = np.abs(rows).max(axis=1, keepdims=True)
        scale = maximum / np.float32(127)
        inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale != 0)
        scaled = rows * inverse
        encoded = (np.sign(scaled) * np.floor(np.abs(scaled).astype(np.float64) + 0.5)).astype(np.int8)
        np.testing.assert_array_equal(packed.values.numpy(), encoded)
        expected = (encoded.astype(np.float32) * scale.astype(np.float16).astype(np.float32)).reshape(1, 2, 1, 64).transpose(0, 3, 1, 2)
        np.testing.assert_array_equal(decode_feature(packed, torch).numpy(), expected)
        self.assertEqual(packed.payload_bytes, data.size // 32 * 34)

    def test_invalid_and_overflowing_cache_payloads(self):
        torch = self.torch
        for values, mode in ((torch.zeros(1, 31, 1, 1), "cache-q8_0"),
                             (torch.full((1, 32, 1, 1), float("nan")), "cache-f16"),
                             (torch.full((1, 32, 1, 1), 1e6), "cache-f16"),
                             (torch.full((1, 32, 1, 1), 1e8), "cache-q8_0"),
                             (torch.full((1, 32, 1, 1), 1e-38), "cache-q8_0")):
            with self.assertRaises(ValueError):
                encode_feature(values, mode, torch)
        for scale in (-1.0, float("nan"), float("inf")):
            bad = PackedFeature("cache-q8_0", (1, 32, 1, 1), torch.zeros(1, 32, dtype=torch.int8), torch.tensor([[scale]], dtype=torch.float16))
            with self.assertRaises(ValueError):
                decode_feature(bad, torch)
        bad = PackedFeature("cache-f16", (1, 32, 1, 1), torch.zeros(1, 1, 1, 31, dtype=torch.float16))
        with self.assertRaises(ValueError):
            decode_feature(bad, torch)

    def test_cpu_avx2_rounding_and_reciprocal_contract(self):
        torch = self.torch
        data = np.random.default_rng(39).normal(size=(1, 64, 7, 1)).astype(np.float32)
        data[0, :5, 0, 0] = [127, 0.5, -0.5, 1.5, -1.5]
        packed = encode_feature(torch.from_numpy(data), "cache-q8_0", torch, q8_backend="cpu-avx2")
        self.assertEqual(packed.values[0, :5].tolist(), [127, 0, 0, 2, -2])
        rows = data.transpose(0, 2, 3, 1).reshape(-1, 32)
        amax = np.abs(rows).max(axis=1, keepdims=True)
        expected = np.rint(rows * (np.float32(127) / amax)).astype(np.int8)
        np.testing.assert_array_equal(packed.values.numpy(), expected)
        with self.assertRaisesRegex(ValueError, "bridge"):
            encode_feature(torch.from_numpy(data), "cache-q8_0", torch, q8_backend="cuda")

    def test_fpn_aliases_exceptions_accounting_and_restore(self):
        torch = self.torch
        features = [torch.full((1, 32, i + 1, 1), 1.0003, dtype=torch.float32) for i in range(3)]
        positions, tracker = object(), object()

        class Backbone:
            def forward_image(self):
                return {"backbone_fpn": features, "vision_features": features[-1],
                        "vision_pos_enc": positions, "sam2_backbone_out": tracker}

        backbone = Backbone()
        control = CacheQuantization(backbone, torch, levels=(0, 2))
        try:
            control.set_mode("cache-f16")
            with torch.inference_mode():
                output = backbone.forward_image()
            self.assertIs(output["vision_features"], output["backbone_fpn"][-1])
            self.assertIs(output["backbone_fpn"][1], features[1])
            self.assertTrue(torch.equal(output["backbone_fpn"][0], features[0].half().float()))
            self.assertIs(output["vision_pos_enc"], positions)
            self.assertIs(output["sam2_backbone_out"], tracker)
            self.assertEqual(control.calls, {"fpn.0": 1, "fpn.1": 1, "fpn.2": 1})
            self.assertEqual([r["packed_payload_bytes"] for r in control.accounting], [64, 256, 192])
            self.assertEqual(float(features[0][0, 0, 0, 0]), float(np.float32(1.0003)))
        finally:
            control.close()
        self.assertNotIn("forward_image", backbone.__dict__)
        with self.assertRaisesRegex(ValueError, "levels"):
            CacheQuantization(backbone, torch, levels=(0, 0))


if __name__ == "__main__":
    unittest.main()
