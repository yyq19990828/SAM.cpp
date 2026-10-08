"""Numerical image-cache codecs; payload layout follows GGML channel-first rows."""

from dataclasses import dataclass
import ctypes
from pathlib import Path


CACHE_MODES = ("original", "cache-f32", "cache-f16", "cache-q8_0")


@dataclass
class PackedFeature:
    mode: str
    shape: tuple
    values: object
    scales: object = None

    @property
    def payload_bytes(self):
        size = self.values.numel() * self.values.element_size()
        if self.scales is not None:
            size += self.scales.numel() * self.scales.element_size()
        return size


class CudaCacheEncoder:
    """Use the pinned CUDA routine inside the numerical oracle, not a new profile."""

    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.library = ctypes.CDLL(str(self.path))
        self.library.sam_cache_bridge_abi.argtypes = []
        self.library.sam_cache_bridge_abi.restype = ctypes.c_int
        if self.library.sam_cache_bridge_abi() != 1:
            raise ValueError("unsupported diagnostic cache bridge ABI")
        self.library.sam_cache_encode_q8_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        self.library.sam_cache_encode_q8_0.restype = ctypes.c_int

    def encode(self, rows, torch):
        if rows.device.type != "cuda" or rows.dtype != torch.float32 or not rows.is_contiguous():
            raise ValueError("CUDA cache bridge requires contiguous device F32 rows")
        if rows.numel() == 0 or rows.numel() % 32 or rows.numel() > 2**26:
            raise ValueError("CUDA cache bridge input exceeds its block bounds")
        with torch.cuda.device(rows.device):
            packed = torch.empty((rows.numel() // 32, 34), dtype=torch.uint8, device=rows.device)
            result = self.library.sam_cache_encode_q8_0(rows.data_ptr(), packed.data_ptr(), rows.numel(),
                                                       torch.cuda.current_stream(rows.device).cuda_stream)
            if result:
                raise RuntimeError(f"CUDA cache bridge launch failed: {result}")
            return packed[:, 2:].contiguous().view(torch.int8), packed[:, :2].contiguous().view(torch.float16)


def encode_feature(values, mode, torch, q8_backend="reference", cuda_encoder=None):
    if (mode not in CACHE_MODES[1:] or values.ndim != 4 or values.dtype != torch.float32
            or min(values.shape) <= 0 or not torch.isfinite(values).all()):
        raise ValueError("cache codec requires a finite nonempty F32 NCHW tensor and supported mode")
    shape = tuple(values.shape)
    # C is ne[0] in SAM's GGML neck outputs, so blocks never join different pixels.
    rows = values.permute(0, 2, 3, 1).contiguous()
    if mode in ("cache-f32", "cache-f16"):
        packed = rows.to(torch.float32 if mode == "cache-f32" else torch.float16)
        if not torch.isfinite(packed).all():
            raise ValueError("cache payload overflowed")
        return PackedFeature(mode, shape, packed)
    if shape[1] % 32:
        raise ValueError("Q8_0 cache requires channels divisible by 32")
    blocks = rows.reshape(-1, 32)
    if q8_backend == "cuda":
        if cuda_encoder is None:
            raise ValueError("Q8_0 CUDA screening requires the validated native cache bridge")
        encoded, stored_scale = cuda_encoder.encode(rows, torch)
        if not torch.isfinite(stored_scale).all() or (stored_scale < 0).any() or (encoded == -128).any():
            raise ValueError("native CUDA cache encoding overflowed")
        return PackedFeature(mode, shape, encoded, stored_scale)
    if q8_backend not in ("reference", "cpu-avx2"):
        raise ValueError("unsupported Q8_0 numerical encoder")
    maximum = blocks.abs().amax(dim=1, keepdim=True)
    # The scalar reference and the optimized AVX2 backend differ in both
    # reciprocal ordering and rounding. CUDA's fast-math encoder is bridged
    # above rather than approximated with PyTorch arithmetic.
    scale = maximum / torch.full_like(maximum, 127)
    inverse = (torch.full_like(maximum, 127) / maximum if q8_backend == "cpu-avx2"
               else torch.ones_like(scale) / scale)
    inverse = torch.where(maximum != 0, inverse, torch.zeros_like(inverse))
    stored_scale = scale.to(torch.float16)
    if not torch.isfinite(inverse).all() or not torch.isfinite(stored_scale).all():
        raise ValueError("Q8_0 cache scale or reciprocal overflowed")
    scaled = blocks * inverse
    magnitude = scaled.abs()
    whole = magnitude.floor()
    rounded = (scaled.round() if q8_backend == "cpu-avx2"
               else (whole + ((magnitude - whole) >= 0.5).to(torch.float32)) * scaled.sign())
    if not torch.isfinite(rounded).all() or (rounded.abs() > 127).any():
        raise ValueError("Q8_0 cache integer is outside its signed range")
    return PackedFeature(mode, shape, rounded.to(torch.int8), stored_scale)


def decode_feature(packed, torch):
    if not isinstance(packed, PackedFeature) or packed.mode not in CACHE_MODES[1:]:
        raise ValueError("unsupported cache payload")
    if len(packed.shape) != 4 or any(type(size) is not int or size <= 0 for size in packed.shape):
        raise ValueError("invalid cache tensor dimensions")
    batch, channels, height, width = packed.shape
    if packed.mode == "cache-q8_0":
        count = batch * channels * height * width // 32
        if (channels % 32 or packed.values.dtype != torch.int8 or packed.values.shape != (count, 32)
                or packed.scales is None or packed.scales.dtype != torch.float16
                or packed.scales.shape != (count, 1) or (packed.scales < 0).any()
                or packed.values.device != packed.scales.device or (packed.values == -128).any()):
            raise ValueError("invalid Q8_0 cache payload")
        decoded = packed.values.to(torch.float32) * packed.scales.to(torch.float32)
    else:
        expected = torch.float32 if packed.mode == "cache-f32" else torch.float16
        if packed.values.dtype != expected or packed.values.shape != (batch, height, width, channels) or packed.scales is not None:
            raise ValueError("invalid floating cache payload")
        decoded = packed.values.to(torch.float32)
    if not torch.isfinite(decoded).all():
        raise ValueError("cache payload decodes to non-finite values")
    return decoded.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


class CacheQuantization:
    """Replace only FPN cache boundaries in a numerical SAM image experiment."""

    def __init__(self, backbone, torch, levels=(0, 1, 2), cuda_encoder=None):
        self.levels = tuple(levels)
        if not self.levels or len(set(self.levels)) != len(self.levels) or any(type(i) is not int or i not in (0, 1, 2) for i in self.levels):
            raise ValueError("cache levels must be unique indices in [0, 2]")
        self.backbone, self.torch = backbone, torch
        self.cuda_encoder = cuda_encoder
        self.original_forward = backbone.forward_image
        self.instance_forward = backbone.__dict__.get("forward_image")
        self.set_mode("original")
        backbone.forward_image = self.forward_image

    def set_mode(self, mode):
        if mode not in CACHE_MODES:
            raise ValueError("unsupported image cache mode")
        self.mode, self.calls, self.accounting = mode, dict.fromkeys(("fpn.0", "fpn.1", "fpn.2"), 0), []

    def forward_image(self, *arguments, **keywords):
        if self.torch.is_grad_enabled():
            raise ValueError("cache study requires inference mode")
        result = self.original_forward(*arguments, **keywords)
        features = result["backbone_fpn"]
        if len(features) != 3 or result["vision_features"] is not features[-1]:
            raise ValueError("SAM FPN alias or level inventory differs")
        replacement = []
        for index, feature in enumerate(features):
            self.calls[f"fpn.{index}"] += 1
            packed_bytes = original_bytes = feature.numel() * feature.element_size()
            if self.mode != "original" and index in self.levels:
                packed = encode_feature(feature, self.mode, self.torch,
                                        q8_backend="cuda" if feature.device.type == "cuda" else "cpu-avx2",
                                        cuda_encoder=self.cuda_encoder)
                replacement.append(decode_feature(packed, self.torch))
                packed_bytes = packed.payload_bytes
                del packed
            else:
                replacement.append(feature)
            self.accounting.append({"level": index, "shape_nchw": list(feature.shape),
                                    "original_payload_bytes": original_bytes, "packed_payload_bytes": packed_bytes})
        # vision_features and the last FPN entry are the same semantic tensor.
        # Replace both aliases, leaving positional and SAM2/tracker features alone.
        return {**result, "backbone_fpn": replacement, "vision_features": replacement[-1]}

    def close(self):
        if self.instance_forward is None:
            del self.backbone.forward_image
        else:
            self.backbone.forward_image = self.instance_forward
