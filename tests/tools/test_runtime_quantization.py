"""Integer arithmetic, reversible reparameterization and semantic output gates."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.quantize.runtime_quantization import (GATES_PATH, LinearQuantization, compare_outputs, encode_mask,
                                  channel_scale, int8_rows, load_gates, rle_iou, select_layers,
                                  validate_output)


class RuntimeQuantizationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)

    def test_zero_rows_and_nearest_even(self):
        torch = self.torch
        with torch.inference_mode():
            q, s = int8_rows(torch.tensor([[0.5, 1.5, 2.5, -1.5, 127], [0, 0, 0, 0, 0]]), torch)
            self.assertEqual(q.tolist(), [[0, 2, 2, -2, 127], [0, 0, 0, 0, 0]])
            self.assertEqual(s.tolist(), [[1], [1]])
            with self.assertRaisesRegex(ValueError, "finite"):
                int8_rows(torch.tensor([[float("nan")]]), torch)

    def test_exact_integer_dots_and_reparameterization(self):
        torch = self.torch
        module = torch.nn.Linear(96, 48)
        rng = np.random.default_rng(97)
        w = rng.normal(0, 0.1, (48, 96)).astype(np.float32)
        bias = rng.normal(0, 0.2, 48).astype(np.float32)
        x = rng.normal(0, 1, (3, 11, 96)).astype(np.float32)
        s = np.exp2(np.linspace(-2, 2, 96)).astype(np.float32)
        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(w))
            module.bias.copy_(torch.from_numpy(bias))
        original_weight = module.weight.detach().clone()
        control = LinearQuantization({"test": module}, {"test": s}, torch)
        try:
            with torch.inference_mode():
                values = torch.from_numpy(x)
                baseline = module(values)
                control.set_mode("reparameterized")
                np.testing.assert_allclose(module(values).numpy(), baseline.numpy(), atol=2e-6, rtol=1e-5)
                ws, xs = w * s, x.reshape(-1, 96) / s

                def encode(v):
                    scale = np.abs(v).max(axis=1, keepdims=True) / np.float32(127)
                    return np.clip(np.rint(v / scale), -127, 127).astype(np.int32), scale

                qw, sw = encode(ws)
                qa, sa = encode(xs)
                exact = qa.astype(np.int64) @ qw.astype(np.int64).T
                for mode, expected in (
                        ("w8a8-token", exact.astype(np.float32) * sw.T * sa + bias),
                        ("weight-only", (xs @ qw.astype(np.float32).T) * sw.T + bias),
                        ("activation-only", (qa.astype(np.float32) @ ws.T) * sa + bias)):
                    control.set_mode(mode)
                    actual = module(values).numpy().reshape(33, 48)
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
                    self.assertEqual(control.calls, {"test": 1})
        finally:
            control.close()
        self.assertTrue(torch.equal(module.weight, original_weight))
        self.assertNotIn("forward", module.__dict__)

    def test_cuda_scale_uses_float_division_at_rounding_boundary(self):
        torch = self.torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        values = np.array([[0.3027027, 0.15135135, 0]], np.float32)
        scale = np.abs(values).max(axis=1, keepdims=True) / np.float32(127)
        expected = np.rint(values / scale).astype(np.int8)
        with torch.inference_mode():
            actual, actual_scale = int8_rows(torch.from_numpy(values).cuda(), torch)
        np.testing.assert_array_equal(actual_scale.cpu().numpy(), scale)
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)

    def test_invalid_parameters_do_not_replace_forward(self):
        torch = self.torch
        module = torch.nn.Linear(32, 16)
        for bad in (np.zeros(32, np.float32), np.full(32, np.nan, np.float32), np.ones(31, np.float32)):
            with self.assertRaisesRegex(ValueError, "contract"):
                LinearQuantization({"test": module}, {"test": bad}, torch)
            self.assertNotIn("forward", module.__dict__)
        control = LinearQuantization({"test": module}, {"test": np.full(32, 0.01, np.float32)}, torch)
        try:
            control.set_mode("w8a8-token")
            with torch.inference_mode(), self.assertRaisesRegex(ValueError, "overflowed"):
                module(torch.full((17, 32), 1e38))
        finally:
            control.close()

    def test_channel_scale_modes_and_normal_float_bounds(self):
        values = np.array([0.24, 0.7, 1.0, 3.1, 8.2], np.float32)
        np.testing.assert_array_equal(channel_scale(values, "power-of-two"), [0.25, 0.5, 1, 4, 8])
        np.testing.assert_array_equal(channel_scale(values, "calibrated"), values)
        np.testing.assert_array_equal(channel_scale(values, "identity"), np.ones(5))
        bounds = channel_scale([np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).max], "power-of-two")
        np.testing.assert_array_equal(bounds, np.exp2(np.array([-126, 127], np.float64)).astype(np.float32))
        for invalid in ([0], [-1], [np.inf], [np.nan], [[1, 2]]):
            with self.assertRaisesRegex(ValueError, "positive vector"):
                channel_scale(invalid, "power-of-two")

    def test_layer_family_and_block_selection(self):
        names = [f"vit.blocks.{block}.{kind}.weight" for block in (0, 1, 31)
                 for kind in ("attn.qkv", "attn.proj", "mlp.lin1", "mlp.lin2")]
        self.assertEqual(select_layers(names, "mlp", [1]), ["vit.blocks.1.mlp.lin1.weight", "vit.blocks.1.mlp.lin2.weight"])
        self.assertEqual(len(select_layers(names, "attention")), 6)
        self.assertEqual(select_layers(names, "mlp-input", [31]), ["vit.blocks.31.mlp.lin1.weight"])
        for family, blocks in (("unknown", [0]), ("all", [0, 0]), ("all", [-1]), ("all", [32]), ("all", []), ("all", [2])):
            with self.assertRaises(ValueError):
                select_layers(names, family, blocks)
        with self.assertRaisesRegex(ValueError, "linear name"):
            select_layers(["text.layers.0.weight"])

    def test_mixed_layers_preserve_unselected_arithmetic_and_restore(self):
        torch = self.torch
        modules = {name: torch.nn.Linear(32, 16) for name in ("selected", "preserved")}
        values = torch.randn(17, 32)
        with torch.inference_mode():
            original = {name: module(values).clone() for name, module in modules.items()}
        for invalid in ([], ["missing"]):
            with self.assertRaisesRegex(ValueError, "inventory"):
                LinearQuantization(modules, {name: np.ones(32) for name in modules}, torch, selected=invalid)
            self.assertTrue(all("forward" not in module.__dict__ for module in modules.values()))
        control = LinearQuantization(modules, {name: np.ones(32) for name in modules}, torch, selected=["selected"])
        try:
            control.set_mode("w8a8-token")
            with torch.inference_mode():
                selected = modules["selected"](values)
                preserved = modules["preserved"](values)
            self.assertTrue(torch.equal(preserved, original["preserved"]))
            self.assertFalse(torch.equal(selected, original["selected"]))
            self.assertEqual(control.calls, {"selected": 1, "preserved": 1})
            self.assertEqual(set(control.packed), {"selected"})
        finally:
            control.close()
        self.assertTrue(all("forward" not in module.__dict__ for module in modules.values()))

    def test_power_of_two_reparameterization_is_exact_for_normal_operands(self):
        torch = self.torch
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            module = torch.nn.Linear(96, 48, device=device)
            values = torch.randn(3, 11, 96, device=device)
            scales = np.exp2(np.arange(96) % 9 - 4).astype(np.float32)
            with torch.inference_mode():
                original = module(values).clone()
            control = LinearQuantization({"test": module}, {"test": scales}, torch, exact_power_of_two=True)
            try:
                control.set_mode("reparameterized")
                with torch.inference_mode():
                    actual = module(values)
                self.assertTrue(torch.equal(actual, original), device)
            finally:
                control.close()

    def test_exact_scale_rejects_operand_underflow(self):
        torch = self.torch
        module = torch.nn.Linear(1, 1)
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            LinearQuantization({"test": module}, {"test": [1.5]}, torch, exact_power_of_two=True)
        self.assertNotIn("forward", module.__dict__)
        tiny = np.nextafter(np.float32(0), np.float32(1))
        for input_value, weight, scale in ((tiny, 1.0, 2.0), (1.0, tiny, 0.5)):
            with torch.no_grad():
                module.weight.fill_(float(weight))
            control = LinearQuantization({"test": module}, {"test": [scale]}, torch, exact_power_of_two=True)
            try:
                control.set_mode("reparameterized")
                with torch.inference_mode(), self.assertRaisesRegex(ValueError, "underflow"):
                    module(torch.tensor([[float(input_value)]]))
            finally:
                control.close()

    @staticmethod
    def output():
        scores = [0.0] * 200
        scores[0], scores[1], scores[2] = 0.9, 0.49, 0.2
        mask = np.zeros((8, 10), np.uint8)
        mask[2:6, 3:8] = 1
        return {"width": 10, "height": 8, "prompt": "person", "token_ids": [1, 2],
                "query_scores": scores, "query_boxes": [[3, 2, 8, 6] for _ in range(200)],
                "detections": [{"query_index": 0, "mask": encode_mask(mask)}]}

    def test_missing_false_positive_and_threshold_adjacent_queries(self):
        reference, gates = self.output(), load_gates()
        actual = copy.deepcopy(reference)
        actual["query_scores"][0] = 0.3
        actual["detections"] = []
        self.assertFalse(compare_outputs(reference, actual, "w8a8-token", gates)["output_screen_passed"])
        actual = copy.deepcopy(reference)
        actual["query_scores"][2] = 0.7
        actual["detections"].append({"query_index": 2, "mask": actual["detections"][0]["mask"]})
        self.assertFalse(compare_outputs(reference, actual, "w8a8-token", gates)["output_screen_passed"])
        actual = copy.deepcopy(reference)
        actual["query_scores"][1] = 0.51
        actual["detections"].append({"query_index": 1, "mask": actual["detections"][0]["mask"]})
        report = compare_outputs(reference, actual, "w8a8-token", gates)
        self.assertTrue(report["output_screen_passed"])
        self.assertEqual(report["threshold_adjacent"][0]["query"], 1)

    def test_independent_mask_score_and_box_limits(self):
        reference, gates = self.output(), load_gates()
        for field in ("mask", "score", "box"):
            actual = copy.deepcopy(reference)
            if field == "mask":
                actual["detections"][0]["mask"] = encode_mask(np.zeros((8, 10), np.uint8))
            elif field == "score":
                actual["query_scores"][0] -= 0.03
            else:
                actual["query_boxes"][0][0] += 0.2
            with self.subTest(field=field):
                self.assertFalse(compare_outputs(reference, actual, "w8a8-token", gates)["output_screen_passed"])
        actual = copy.deepcopy(reference)
        actual["query_scores"][0] -= 0.01
        self.assertTrue(compare_outputs(reference, actual, "w8a8-token", gates)["output_screen_passed"])
        self.assertFalse(compare_outputs(reference, actual, "reparameterized", gates)["output_screen_passed"])

    def test_output_integrity_and_empty_masks(self):
        reference, gates = self.output(), load_gates()
        bad = copy.deepcopy(reference)
        bad["query_scores"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "query"):
            validate_output(bad, 0.5)
        bad = copy.deepcopy(reference)
        bad["detections"] = []
        with self.assertRaisesRegex(ValueError, "selected masks"):
            validate_output(bad, 0.5)
        bad = copy.deepcopy(reference)
        bad["token_ids"] = [3]
        with self.assertRaisesRegex(ValueError, "token"):
            compare_outputs(reference, bad, "w8a8-token", gates)
        empty = encode_mask(np.zeros((7, 11), np.uint8))
        self.assertEqual(rle_iou(empty, empty), 1)
        self.assertEqual(rle_iou(reference["detections"][0]["mask"], reference["detections"][0]["mask"]), 1)

    def test_changed_gates_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gates.json"
            path.write_bytes(GATES_PATH.read_bytes().replace(b'"mask_iou_min": 0.96', b'"mask_iou_min": 0.90'))
            with self.assertRaisesRegex(ValueError, "frozen"):
                load_gates(path)


if __name__ == "__main__":
    unittest.main()
