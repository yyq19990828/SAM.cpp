"""Independent contract for the first scoped CUDA Q8 RHS policy.

This is an offline numerical reference, not a selectable native runtime mode.
The initial forced path uses Q8_0 weights and MMQ D4 RHS staging. MMVQ's half
scale/original-sum contract is separate and is not implied by this contract.
"""

import numpy as np

from tools.validation.verify_q8_mmq_rhs_staging import check_blocks


MODE = "ggml-q8-rhs"
CONTRACT = "ggml-cuda-q8-mmq-d4-v1"
BLOCK = 32
TILE_VALUES = 128
TILE_BYTES = 144
ROW_PADDING = 512
MIN_NONZERO_BLOCK_MAX = np.float32(1.e-30)
MAX_BLOCK_MAX = np.float32(1.e30)


def supported_values(values):
    """Reject unsafe native reciprocal ranges instead of silently narrowing.

    Every nonzero 32-element block must have max(abs(x)) in [1e-30,1e30].
    Zero blocks are supported. Bounds are F32 values; all values must be finite.
    Width is positive, 32 aligned; outer rows may be strided in this reference.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(values, dtype="<f4")
    if array.ndim != 2 or not array.shape[0] or not array.shape[1] or array.shape[1] % BLOCK:
        raise ValueError("native Q8 RHS needs nonempty 32-aligned rows")
    if not np.isfinite(array).all():
        raise ValueError("native Q8 RHS requires finite F32 inputs")
    maximum = np.max(np.abs(array.reshape(array.shape[0], -1, BLOCK)), axis=2)
    if ((maximum > MAX_BLOCK_MAX) | ((maximum != 0) & (maximum < MIN_NONZERO_BLOCK_MAX))).any():
        raise ValueError("native Q8 RHS block maximum outside supported [1e-30,1e30] or zero range")
    return array


def encode_reference(values):
    """Independent exact-target round-away reference, with native MMQ layout.

    CUDA fast reciprocals may differ at ties; acceptance uses check_payload's
    pinned four-F32-ulp scale and adjacent-code-at-tie rules, not byte equality
    to this ideal target. Layout is [K/128 tile, row, 4 F32 scales + 128 int8].
    """
    array = supported_values(values)
    rows, width = array.shape
    padded = (width + ROW_PADDING - 1) // ROW_PADDING * ROW_PADDING
    source = np.zeros((rows, padded), dtype="<f4")
    source[:, :width] = array
    blocks = source.reshape(rows, -1, BLOCK).astype(np.float64)
    maximum = np.max(np.abs(blocks), axis=2)
    scale = (maximum / 127.).astype("<f4")
    target = np.divide(blocks * 127., maximum[:, :, None], out=np.zeros_like(blocks), where=maximum[:, :, None] != 0)
    codes = np.copysign(np.floor(np.abs(target) + .5), target).astype(np.int8)
    payload = np.zeros((padded // TILE_VALUES, rows, TILE_BYTES), dtype=np.uint8)
    payload[:, :, :16] = scale.reshape(rows, -1, 4).transpose(1, 0, 2).copy().view(np.uint8).reshape(-1, rows, 16)
    payload[:, :, 16:] = codes.reshape(rows, -1, TILE_VALUES).transpose(1, 0, 2).view(np.uint8)
    return payload.tobytes()


def check_payload(values, payload):
    array = supported_values(values)
    rows, width = array.shape
    padded = (width + ROW_PADDING - 1) // ROW_PADDING * ROW_PADDING
    expected = padded // TILE_VALUES * rows * TILE_BYTES
    if len(payload) != expected:
        raise ValueError("native MMQ Q8 RHS payload size differs")
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(-1, rows, TILE_BYTES)
    source = np.zeros((rows, padded), dtype="<f4")
    source[:, :width] = array
    total = {key: 0 for key in ("blocks", "subblocks", "scale_ulp_allowances", "integer_tie_allowances", "scale_errors", "integer_errors")}
    for tile, begin in enumerate(range(0, padded, TILE_VALUES)):
        part = check_blocks(source[:, begin:begin + TILE_VALUES], packed[tile])
        for key in total:
            total[key] += part[key]
    total.update(contract=CONTRACT, rows=rows, width=width, padded_width=padded, payload_bytes=expected,
                 passed=not (total["scale_errors"] or total["integer_errors"]))
    return total


def decode_payload(payload, rows, width):
    if type(rows) is not int or type(width) is not int or rows <= 0 or width <= 0 or width % BLOCK:
        raise ValueError("native MMQ decode needs positive 32-aligned dimensions")
    padded = (width + ROW_PADDING - 1) // ROW_PADDING * ROW_PADDING
    if len(payload) != padded // TILE_VALUES * rows * TILE_BYTES:
        raise ValueError("native MMQ decode payload size differs")
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(-1, rows, TILE_BYTES)
    scales = packed[:, :, :16].copy().view("<f4").reshape(-1, rows, 4)
    codes = packed[:, :, 16:].view(np.int8).reshape(-1, rows, 4, BLOCK)
    if not np.isfinite(scales).all() or (scales < 0).any() or (codes == -128).any():
        raise ValueError("invalid native MMQ Q8 RHS scales/codes")
    values = (scales[:, :, :, None] * codes).transpose(1, 0, 2, 3).reshape(rows, padded)
    return values[:, :width]
