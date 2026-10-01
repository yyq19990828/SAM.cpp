# GGML Metal precision and window patch

`ggml-precise-metal.patch` supplies local precision corrections and native window
partition/restoration against official
[GGML 0.25.3](https://github.com/ggml-org/ggml/commit/353b63b439f27ab2cc19dac97ab1681ba6d2d084)
commit `353b63b439f27ab2cc19dac97ab1681ba6d2d084` (MIT; upstream license retained).
The shared checkout is read-only; SAM applies this patch to a build-local copy.
Patch SHA-256: `0a0b80dd15c2a8b5a05d148a31e9e53f8df4d0555852ef301da336a9d97b7c48`.

## Precision contract

- Dense `GGML_OP_MUL_MAT` with explicit `GGML_PREC_F32`, F32/F16 left operand,
  and F32 right operand selects float staging and `simdgroup_float8x8` operands.
  Default precision, quantized operands, and expert-ID operations keep upstream
  behavior. The regular kernel's 64x32 left and 32x32 right tiles require
  12,288 bytes of threadgroup memory; its right-tile offset follows the left
  staging type. Tail output reuse fits this allocation. The tensor API variant
  stages only the left tile and sizes that allocation for its actual type.
- `GGML_OP_FLASH_ATTN_EXT` with explicit `GGML_PREC_F32`, F32 K/V, and equal
  32- or 64-dimensional K/V heads selects a matrix attention specialization with
  float Query storage. Q, K, V, softmax intermediates, and output accumulation
  remain float. Query-sized offsets and host threadgroup allocation match the
  float layout (7,168 bytes for head 32; 8,192 bytes for head 64).
- The same attention predicate bypasses both short-query and sparse-hint vector
  dispatch because those upstream kernels narrow Query to half. Masks remain
  authoritative when a sparse hint is present. Default attention, F16/quantized
  K/V, and other head dimensions preserve upstream behavior. There is no new
  precision guarantee for wider heads: float Query would require more than
  M4 Pro's 32 KiB threadgroup limit at heads 512/576.
- Upstream Metal accepts attention only when K/V types match. SAM's graph-side
  convolution precision handling remains separate from this patch.

## Window operator contract

`GGML_OP_WIN_PART` and `GGML_OP_WIN_UNPART` execute natively on Metal for
contiguous F32 inputs/outputs with fewer than `2^31` output elements. A flat
thread grid copies each channel value without arithmetic; edge windows receive
zero padding, and restoration crops the padding. Tail threads check the output
count before accessing memory. GGML skips empty tensors before dispatch.

The host argument structs, shader byte offsets and dispatch use the same
contiguous layouts as GGML's CPU operators. Other types or strides remain
unsupported by these Metal kernels. They are adapted from the MIT-licensed
[PABannier GGML revision `7a466633`](https://github.com/PABannier/ggml/commit/7a466633195d336e08dd5fec8b8ab42a2c5cb5e3).
The kernels reuse `kernels/misc.metal`, which already participates in embedded
and standalone Metal library builds; function discovery handles routing.

## Provenance

SHA-256 of the exact touched files before and after applying the patch:

| File | Base | Patched |
| --- | --- | --- |
| `src/ggml-metal/ggml-metal-device.cpp` | `c37fecc8c847140b31e8ffb0223394821b8bda7111d504b2757183add1eb926e` | `b4879d6dcee59d1938be042424cd10c1ce671d60a61199b4cb3c2a6cfed9176f` |
| `src/ggml-metal/ggml-metal-device.h` | `4147922b64c1395fe5f5be71f3bbe4a3094a18761f8858fa7c413bd468137bf5` | `ff760be068bb74c5430bbba60c4997ded85b56167345fc38ebbee02bb6f83e13` |
| `src/ggml-metal/ggml-metal-device.m` | `0ed00ba6b14a93177ab655712ebcd74a5d8164a88bee833f883d76b7d372695e` | `c807303e1e0af2a1a06343e6bfdca8a38dd4b4d9f0e9abaedb14df36bbbce185` |
| `src/ggml-metal/ggml-metal-impl.h` | `d0273e36d76772968fe50aa043f898be7139d7dbd9ab81ee2037848d5b572327` | `ddc150f2d4945e20b17758a3bd35f83c965907d2274959b04c77aa6903c4e9d4` |
| `src/ggml-metal/ggml-metal-ops.cpp` | `95f7770a76a7eebb2df455a9e67ea3a4e54c23ff1df23fe111e89b0481cf6c6e` | `18681e7b9b58e63a68fa9f3944cdc5ef6855b6fc8befa38fc330b3555bca76e1` |
| `src/ggml-metal/ggml-metal-ops.h` | `fa1d2fb8b3c039eb7aedbf4ff8e28dc2494a5a483d7617da33705be9cdacc40b` | `83fc575a2924fe5917f1c3d161243ead3917e5986594e37c562bc7440bcba23e` |
| `src/ggml-metal/kernels/fa.metal` | `fa1405fbfbd9740d861db22ff8b1324925b801e971604a65a31e9ac4081df838` | `33ca59b532fae2d534818969df90faef46c93050904b5313165616dbac1099ac` |
| `src/ggml-metal/kernels/misc.metal` | `040429b8fcba1a301c26131fb22c389bf34a60b992762fcec45cf66c012487be` | `4595e34a8f6388923d665a43e1a76a4945f3dd39a11061449226d7a28e531364` |
| `src/ggml-metal/kernels/mul_mm.metal` | `4244a2199826bae50294ff471886fd3bded1159997cd33357062b4c26efc5c35` | `947db05673df8d3c9b125bfbd184a880c895651e9b78dbbe1c4eeb87c02b05dd` |

Apply from a separate copy of the pinned source with `git apply
/path/to/ggml-precise-metal.patch`. No Python is required for consumer builds.
A fresh copy of these official files passed `git apply --check`; applying the
patch produced all nine recorded hashes. The complete original source tree
hash is `5fc1277d1894e92a0b1a812c3ac88bb46564a2cccd9873e02ad7f8533346b48c`; the patched tree
hash is `5816fc926f890714e8a8fa853c428550387f0653794a19537a0ba0b420987c8a`. CMake verifies
these full-tree hashes, the fixed patch hash and patch applicability, including
source archives without Git metadata. It repairs generated-source drift and
rejects overlapping source/output paths before removal.

## Hardware verification

Validated on Apple M4 Pro, actual GGML backend `MTL0`, AppleClang
`21.0.0` (`clang-2100.3.34.2`) for GGML and Homebrew Clang `23.1.2` for the standalone
probe, with `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON`, Release.
The standalone probe allocates all operands and outputs directly on the selected
backend, calls `ggml_backend_graph_compute`, and reads the results. Pipeline logs
confirm precise dense and float-Query matrix attention dispatch. The probe and
complete output are ignored under `build/upstream-migration/metal-probe.cpp`
and `build/upstream-migration/metal-probe-extended.log`; the maintained regression
is registered with SAM's CTest suite.

A direct Metal device query confirms `maxThreadgroupMemoryLength=32768`.
The M4 Pro reports `has_tensor=false` and excludes the tensor API shader branch.
Its allocation change is reviewed, but neither compilation nor arithmetic
execution of that branch is verified here; the standalone Metal compiler is
also unavailable on this host.

For K/M/N=256/256/32 filled with `1.0003f`, double-accumulated expected F32
output is `256.153656`. CPU outputs `256.153625`, default Metal `256`, and
precise Metal `256.153137` (maximum absolute error `0.000518799`, relative
error `2.03e-6`). With stored F16 left values rounded to 1 and the same F32
right values, expected output is `256.076813`, default Metal `256`, and
precise Metal `256.077332` (maximum error `0.000518799`).

Varied inputs exercise input/output tails and a larger dense shape:

| K/M/N | F32/F32 maximum error | F16/F32 maximum error |
| --- | --- | --- |
| 257/65/33 | `1.15e-5` | `1.14e-5` |
| 768/1024/197 | `2.58e-5` | `2.41e-5` |

The dense probe allows `3e-6 * max(1, max_abs_reference)` for float summation
and reduction order; the constant case rejects half staging by more than two
orders of magnitude. This arithmetic bound does not change model gates.

F32 attention was checked at both heads 32/64 with 1, 7, 19, 20, and 33
queries and 128/130 keys. Per-query values vary below half's resolution, so the
multi-query cases check preservation of Query detail and the shared layout.
The 130-key cases exercise KV padding; the 33-query cases exercise a final query
block. A masked 33-query case supplies a sparse hint and still selects precise
matrix attention. Maximum precise error across these cases is `3.58e-7`, against
`1.86e-4` with default precision; the probe bound is `3e-6`.

For head 64, one query, and 128 keys, Q=`1.0003f`, K=+/-0.0625,
V=+/-1, and scale=0.125 give expected `tanh(0.5 * double(1.0003f))`
=`0.462235123`. Default attention outputs `0.462117165` and precise attention
`0.462235332`. The SAM empty-geometry cross-attention shape (head 32,
one query, 5,184 keys) outputs `0.244989067` against expected `0.244989172`;
default attention outputs `0.244918659`.

The maintained `tests/test_window.cpp` regression compares native Metal output
with CPU output exactly for rectangular/padded inputs, one-pixel inputs,
multiple channels, window size one, partial final threadgroups and SAM's
C1024/W72/H72/window24 shape. It checks handwritten channel/window ordering
and padding goldens, plus exact restoration of the original input. The direct
`ggml_backend_graph_compute` call cannot assign window nodes to a CPU fallback.
On Apple M4 Pro with AppleClang 21.0.0, the isolated window probe passed all six
cases and logged `kernel_win_part_f32` and `kernel_win_unpart_f32` on `MTL0`.
Its frozen patch metadata and output are in ignored
`build/perf-repair/metal-window-metadata.json` and `metal-window-probe.log`.
The caller-owned-GGML consumer reuses this same maintained regression.

These are direct arithmetic and layout probes. Full-model frozen numerical gates and
performance are recorded separately in the SAM milestone reports.
