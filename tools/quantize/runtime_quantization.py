"""Numerical-only vision-linear variants and original-output screening gates."""

from pathlib import Path
import re

import numpy as np

from tools.convert.sam3_artifacts import read_json, sha256_file


GATES_PATH = Path(__file__).resolve().parents[2] / "tests/data/sam3-runtime-quantization-gates.json"
GATES_SHA256 = "0acc04eae4d37b0b014ad9f169e2dcae14bb674de90600fc0273ed8c842a4d21"
MODES = ("original", "reparameterized", "weight-only", "activation-only", "w8a8-token")
SCALE_MODES = ("calibrated", "power-of-two", "identity")
LAYER_FAMILIES = ("all", "mlp", "attention", "mlp-input", "mlp-output")


def load_gates(path=GATES_PATH):
    if sha256_file(path) != GATES_SHA256:
        raise ValueError("runtime quantization gates differ from the pre-run frozen version")
    return read_json(path)


def int8_rows(values, torch):
    if values.ndim != 2 or values.dtype != torch.float32 or not torch.isfinite(values).all():
        raise ValueError("INT8 rows require finite F32 matrix values")
    maximum = values.abs().amax(dim=1, keepdim=True)
    # A Python scalar divisor takes PyTorch's reciprocal-multiply CUDA path,
    # which can change a nearest-even tie. Match the probe's F32 division.
    scale = (maximum / torch.full_like(maximum, 127)).clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.where(maximum == 0, torch.ones_like(scale), scale)
    encoded = (values / scale).round().clamp(-127, 127).to(torch.int8)
    return encoded, scale


def channel_scale(values, mode):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("channel scales must be a finite positive vector")
    if mode == "identity":
        return np.ones_like(values)
    if mode == "calibrated":
        return values.copy()
    if mode != "power-of-two":
        raise ValueError("unsupported channel scale mode")
    # Clamp to normal, exactly representable F32 powers. The transformed
    # operands are checked separately; a finite scale alone cannot prove that.
    exponents = np.clip(np.rint(np.log2(values.astype(np.float64))), -126, 127)
    return np.exp2(exponents).astype(np.float32)


def select_layers(names, family="all", blocks=None):
    if family not in LAYER_FAMILIES:
        raise ValueError("unsupported vision layer family")
    blocks = list(range(32)) if blocks is None else list(blocks)
    if not blocks or len(set(blocks)) != len(blocks) or any(type(b) is not int or not 0 <= b < 32 for b in blocks):
        raise ValueError("vision block selection must contain unique indices in [0, 31]")
    selected = []
    for name in names:
        match = re.fullmatch(r"vit\.blocks\.(\d+)\.(attn\.(qkv|proj)|mlp\.(lin1|lin2))\.weight", name)
        if match is None or not 0 <= int(match.group(1)) < 32:
            raise ValueError("unsupported vision linear name")
        kind = match.group(2)
        if int(match.group(1)) in blocks and (family == "all"
                or family == "mlp" and kind.startswith("mlp.")
                or family == "attention" and kind.startswith("attn.")
                or family == "mlp-input" and kind == "mlp.lin1"
                or family == "mlp-output" and kind == "mlp.lin2"):
            selected.append(name)
    if not selected:
        raise ValueError("vision layer selection is empty")
    return sorted(selected)


class LinearQuantization:
    """Keep original weights intact; switch only the explicitly supplied linears."""

    def __init__(self, modules, scales, torch, selected=None, exact_power_of_two=False):
        self.torch, self.modules = torch, dict(modules)
        self.scales, self.original_forwards, self.packed, self.calls = {}, {}, {}, {}
        self.mode = "original"
        if set(modules) != set(scales):
            raise ValueError("linear/scale inventory differs")
        self.selected = set(modules) if selected is None else set(selected)
        if not self.selected or not self.selected <= set(modules):
            raise ValueError("selected linear inventory is empty or unknown")
        self.exact_power_of_two = exact_power_of_two
        # Validate every parameter before replacing any forward method.
        for name, module in self.modules.items():
            scale = torch.as_tensor(scales[name], dtype=torch.float32, device=module.weight.device)
            if (module.weight.dtype != torch.float32 or scale.shape != (module.in_features,)
                    or not torch.isfinite(scale).all() or (scale <= 0).any()
                    or module.in_features > 16384 or module.in_features <= 0):
                raise ValueError(f"{name}: invalid F32 weight/channel-scale contract")
            self.scales[name] = scale
            if exact_power_of_two:
                mantissa, _ = torch.frexp(scale)
                if (mantissa != 0.5).any():
                    raise ValueError("exact scaling requires power-of-two channel scales")
        for name, module in self.modules.items():
            self.original_forwards[name] = (module.forward, module.__dict__.get("forward"))
            self.calls[name] = 0

            def forward(values, name=name):
                return self.linear(name, values)

            module.forward = forward

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError("unsupported numerical study mode")
        self.mode = mode
        self.calls = dict.fromkeys(self.modules, 0)

    def linear(self, name, values):
        torch, module = self.torch, self.modules[name]
        if torch.is_grad_enabled() or values.dtype != torch.float32 or values.shape[-1] != module.in_features:
            raise ValueError("linear study requires inference-mode F32 input with matching channels")
        self.calls[name] += 1
        if self.mode == "original" or name not in self.selected:
            return self.original_forwards[name][0](values)
        scale = self.scales[name]
        x = values.reshape(-1, module.in_features) / scale
        if not torch.isfinite(x).all():
            raise ValueError("activation channel transformation overflowed")
        if self.exact_power_of_two and not torch.equal(x * scale, values.reshape(-1, module.in_features)):
            raise ValueError("activation channel transformation lost information through underflow")
        if self.mode == "reparameterized":
            weight = module.weight * scale
            if not torch.isfinite(weight).all():
                raise ValueError("weight channel transformation overflowed")
            if self.exact_power_of_two and not torch.equal(weight / scale, module.weight):
                raise ValueError("weight channel transformation lost information through underflow")
            # Preserve Linear's dimensional dispatch and fused bias. Flattening
            # the input and adding bias after matmul changes the F32 operation.
            result = torch.nn.functional.linear(x.reshape(values.shape), weight, module.bias)
            if not torch.isfinite(result).all():
                raise ValueError("non-finite linear study result")
            return result
        if self.mode in ("weight-only", "w8a8-token"):
            if name not in self.packed:
                weight = module.weight * scale
                if self.exact_power_of_two and not torch.equal(weight / scale, module.weight):
                    raise ValueError("weight channel transformation lost information through underflow or overflow")
                self.packed[name] = int8_rows(weight, torch)
            qw, sw = self.packed[name]
        if self.mode in ("activation-only", "w8a8-token"):
            qa, sa = int8_rows(x, torch)
        if self.mode == "w8a8-token":
            # K <= 16384 and |q| <= 127 bound each exact dot below INT32_MAX.
            result = torch._int_mm(qa, qw.T).to(torch.float32)
            result = (result * sw[:, 0]) * sa
        elif self.mode == "weight-only":
            result = (x @ qw.to(torch.float32).T) * sw[:, 0]
        else:
            weight = module.weight * scale
            if not torch.isfinite(weight).all():
                raise ValueError("weight channel transformation overflowed")
            if self.exact_power_of_two and not torch.equal(weight / scale, module.weight):
                raise ValueError("weight channel transformation lost information through underflow")
            result = (qa.to(torch.float32) @ weight.T) * sa
        if module.bias is not None:
            result = result + module.bias
        if not torch.isfinite(result).all():
            raise ValueError("non-finite linear study result")
        return result.reshape(*values.shape[:-1], module.out_features)

    def close(self):
        for name, module in self.modules.items():
            _, instance_forward = self.original_forwards[name]
            if instance_forward is None:
                del module.forward
            else:
                module.forward = instance_forward
        self.packed.clear()


def encode_mask(mask):
    from pycocotools import mask as masks
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError("output mask must be a binary plane")
    encoded = masks.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": list(encoded["size"]), "counts": encoded["counts"].decode("ascii")}


def rle_iou(actual, reference):
    from pycocotools import mask as masks
    if actual["size"] != reference["size"]:
        raise ValueError("output mask dimensions differ")
    a = {"size": actual["size"], "counts": actual["counts"].encode("ascii")}
    b = {"size": reference["size"], "counts": reference["counts"].encode("ascii")}
    if int(masks.area(a)) == 0 and int(masks.area(b)) == 0:
        return 1.0
    return float(masks.iou([a], [b], [False])[0, 0])


def validate_output(result, threshold):
    width, height = result.get("width"), result.get("height")
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("invalid image dimensions")
    scores = np.asarray(result["query_scores"], dtype=np.float64)
    boxes = np.asarray(result["query_boxes"], dtype=np.float64)
    if (scores.shape != (200,) or boxes.shape != (200, 4) or not np.isfinite(scores).all()
            or not np.isfinite(boxes).all() or (scores < 0).any() or (scores > 1).any()
            or (boxes[:, 2:] < boxes[:, :2]).any()):
        raise ValueError("invalid SAM query outputs")
    selected = np.flatnonzero(scores > threshold).tolist()
    if selected != [row["query_index"] for row in result["detections"]]:
        raise ValueError("selected masks disagree with query scores")
    for row in result["detections"]:
        if row["mask"]["size"] != [height, width]:
            raise ValueError("mask shape disagrees with image")
    return scores, boxes


def compare_outputs(reference, actual, mode, gates):
    if mode not in MODES[1:]:
        raise ValueError("comparison requires a candidate mode")
    if any(reference[key] != actual[key] for key in ("width", "height", "prompt", "token_ids")):
        raise ValueError("input dimensions, text or token IDs differ")
    common = gates["common"]
    threshold = common["score_threshold"]
    rs, rb = validate_output(reference, threshold)
    cs, cb = validate_output(actual, threshold)
    profile = gates["reparameterization" if mode == "reparameterized" else "quantized"]
    original = {row["query_index"]: row for row in reference["detections"]}
    candidate = {row["query_index"]: row for row in actual["detections"]}
    failures, detections, adjacent = [], [], []
    dimensions = np.asarray([reference["width"], reference["height"]] * 2)
    for index, score in enumerate(rs):
        if score >= common["high_confidence_reference_score_min"]:
            if index not in candidate:
                failures.append({"query": index, "reason": "high-confidence reference detection missing"})
                continue
            iou = rle_iou(candidate[index]["mask"], original[index]["mask"])
            score_error = float(abs(cs[index] - score))
            box_error = float(np.max(np.abs(cb[index] - rb[index]) / dimensions))
            passed = (iou >= profile["mask_iou_min"] and score_error <= profile["score_absolute_error_max"]
                      and box_error <= profile["box_dimension_fraction_max"])
            detections.append({"query": index, "mask_iou": iou, "score_absolute_error": score_error,
                               "box_dimension_fraction": box_error, "passed": passed})
            if not passed:
                failures.append({"query": index, "reason": "mask, score or box limit exceeded"})
        elif score <= common["low_confidence_reference_score_max"] and index in candidate:
            failures.append({"query": index, "reason": "low-confidence reference query became selected"})
        else:
            low, high = common["threshold_adjacent_interval"]
            if low < score < high:
                adjacent.append({"query": index, "reference_score": float(score), "actual_score": float(cs[index]),
                                 "reference_selected": index in original, "actual_selected": index in candidate})
    return {"output_screen_passed": not failures, "detections": detections,
            "threshold_adjacent": adjacent, "failures": failures,
            "all_query_score_max_abs": float(np.max(np.abs(cs - rs))),
            "reference_detections": len(original), "actual_detections": len(candidate)}
