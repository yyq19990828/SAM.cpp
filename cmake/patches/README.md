# GGML precision and window patches

Both patches target official [GGML v0.26.0](https://github.com/ggml-org/ggml/commit/d7cb574130e6f01ad25b3289685489200febcd74)
commit `d7cb574130e6f01ad25b3289685489200febcd74` (MIT; upstream license retained).
SAM prepares and patches a separate build-local copy; the supplied source checkout stays read-only.

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

The v0.26.0 attention code is split into dtype-specific libraries. Query layout
changes live in `kernels/fa_common.metal`; only `kernels/fa_f32.metal` adds the
head-32/64 precise specializations. Host allocation accounts for float Query
storage. The upstream few-row MMA path is retained: its generic F32/F16 kernels
use float matrix operands, and the precision regression covers its dispatch boundaries.

## CUDA contract

For NVIDIA CUDA, dense F32/F16-by-F32 matrix operations with explicit
`GGML_PREC_F32` use cuBLAS pedantic F32 arithmetic. This bypasses reduced-precision
custom matrix kernels and prevents `GGML_CUDA_CUBLAS_COMPUTE_TYPE` from overriding
the explicit precision request. An additional explicit F16 source-1 hint selects
F16 dense operands with F32 accumulation and output. SAM exposes this as opt-in
`CudaComputeMode::F16`; it does not follow the weight storage type automatically.
Quantized operands retain the upstream packed-weight dispatch rules.

Explicit F32 attention with F32 Q/K/V and equal head dimensions 32 or 64 uses
pedantic F32 cuBLAS products and an F32 masked softmax. Up to eight heads share
a strided-batched cuBLAS call and at most 1024 queries per tile. The head group
is capped by a 32 MiB score budget, or one head when that head alone exceeds the
budget; grouped-query attention keeps a single-head group. Products write directly
to the output strides without a separate output-copy kernel. Scratch grows with
the key count and stays bounded independently of the total query count. Q/K/V may have
outer strides; masks support broadcast or per-head/per-batch F16 additive values.
The complete mask/key range remains authoritative when sparse hints are present.
Fully masked rows produce zero. Attention sinks, ALiBi, softcap and other head
profiles are outside this specialization. HIP and MUSA retain upstream attention
and matrix dispatch; they are not validated by these CUDA checks.

Native window partition/restoration supports contiguous F32 inputs/outputs with
one image batch. Kernels preserve channel/window order, pad edge windows with
zero, crop restoration and guard tail threads using 64-bit indices.

Contiguous-plane concat batches the fourth dimension into one kernel launch.
The kernel preserves actual source/destination plane strides and loops over
planes beyond the CUDA grid-y limit. It copies 1/2/4/8-byte elements and packed
quantized blocks without arithmetic. The noncontiguous and dimension-3 paths
retain upstream behavior; the CUDA regression compares every output byte across
all axes, padded planes and more than 65,535 planes.

The SAM CUDA driver checks required matrix and head-32/64 attention precision at
initialization, including with caller-owned targets. Explicit CUDA graphs require
every compute node on the selected device; CPU remains available for input copies.
The opt-in F16 compute mode uses the pinned fused CUDA attention for unmasked
heads 64 and 256, including batched tracker memory attention. Masked attention and head 32 retain the precise path, preserving zero
output for fully masked rows. Its startup matrix probe checks operand
rounding and rejects F16 accumulator/output overflow. The CUDA driver also
supplies a 512-query memory-attention tile and direct output assembly policy for default compute;
CPU and Metal retain their existing policies. CUDA graph capture stays off. The previous revision's hardware checks and performance evidence remain in the
[CUDA implementation](../../docs/plans/20261006-215722-cuda-backend.md) and
[precision/dataflow](../../docs/plans/20261007-094949-cuda-precision-and-dataflow-optimization.md)
plans. They are not validation of v0.26.0.

Non-contiguous scalar copies into a dense destination use precomputed source
index divisors when the element count fits `INT32_MAX`, retaining 64-bit byte
strides. Exact dense transpositions of two contiguous dimension groups reuse
the upstream tiled transpose kernel, including outer batches. Other layouts,
large index ranges and grids outside the tiled kernel's limits retain generic
copy dispatch. This changes data movement only, without changing arithmetic,
weight formats or tensor representations. `tests/runtime/ggml/test_cuda_copy.cpp` checks
logical order, raw same-type bits, conversions, padding and layout boundaries.


## Source verification

CMake verifies every source file, the patch hashes and the resulting tree.
Unmodified archives, verified Metal-only archives and verified combined archives
are accepted; caller-owned GGML targets are used directly and must meet the same
operator and precision contracts. Generated-source drift is repaired, and
source/output overlap or output symlinks are rejected before replacement.

- Metal patch SHA-256: `86e7140e59a8eaa8c8d83eabe3102f199a83527cbb7be9c81c8aba05001f4efb`.
- CUDA patch SHA-256: `b1ce5629bba572c0cf5d488ca4b11362927e1be3ac6271a1ff573faf729f88ab`.
- Original source-tree SHA-256: `43c54450bdc1ad5d3a5dd16d69507fa2b740d2f3283cb80f9034e9d669c79fa1`.
- Metal-only source-tree SHA-256: `026988224220506e00cfab5477f1a11c5e926e4a090eb03575d4120d29e536f4`.
- Combined source-tree SHA-256: `37f787b8a432f2399e9ee50270f8d025378c605ae23f5efaae6a6ae9148fd5cb`.

The touched-file SHA-256 values below are generated from the pinned original
and patched trees. A dash denotes a file added by the patch.

| File | Base | Patched |
| --- | --- | --- |
| `src/ggml-metal/ggml-metal-device.cpp` | `c91ad5f2d2770e7d1d77fb65384b38e87105695487759b1d52ae259a30636866` | `11cac660b37aac9e092ea37c83182c46cc66b9117561fb102c48d8fd376958ae` |
| `src/ggml-metal/ggml-metal-device.h` | `2ed9ef4d671222d895e312a0feb16ec640c08a0bfa500f4b09a1cc6733e088b6` | `c672a6cca575b1915a4137596e71e6ccf6dd413fa3e559b4f1b473f1430d6447` |
| `src/ggml-metal/ggml-metal-device.m` | `3f520d505e99ece2eac65607dd7e9e670c93ca77e32b5d31c766359f57204cdd` | `e24a10f76822d8f8e57c817dad4ba2ea08e842c2aa8b79ab3c3a7a60a59ae71d` |
| `src/ggml-metal/ggml-metal-impl.h` | `43817b7b3d6edcb9aaef517773e629d37422691720e266b090670f733d977271` | `b5daed8290aceb2054befc2200df07b650316a18580d59a773e6311063b1e4b9` |
| `src/ggml-metal/ggml-metal-ops.cpp` | `40114ecae3071b40f8ea308d87d9141e78a94b31907c6beaebe9e51bea819dbb` | `1907f8d1607da365053ed9be3095e4376e90c254cd215dd8876a11ec3af368be` |
| `src/ggml-metal/ggml-metal-ops.h` | `fa1d2fb8b3c039eb7aedbf4ff8e28dc2494a5a483d7617da33705be9cdacc40b` | `83fc575a2924fe5917f1c3d161243ead3917e5986594e37c562bc7440bcba23e` |
| `src/ggml-metal/kernels/fa_common.metal` | `ec44dd8f84580201fe63eb3ea0342039519740d4c4bffa03aff1e7a361a288d8` | `92746679caf4113a6d0324b3ff7805da4d3aae6240acf2e849b717430f2758b9` |
| `src/ggml-metal/kernels/fa_f32.metal` | `b6a84cc876f4e7b6f25ea0f99c179bb6105aebbf518084d9654d47a8ef656020` | `e1f90583841a9806925b4bb88b41072f949195143eca27450710c099026bfc7b` |
| `src/ggml-metal/kernels/misc.metal` | `8169ba408031f6d953c2f0ced2ad9c8ee8152a22260526efc5fe3e243c1c74ea` | `6e3bc055e08325ceefdc16cb6abd9b6cc3739f03d2e039092acb9669ae6b7eae` |
| `src/ggml-metal/kernels/mul_mm.metal` | `b25889dec680cfebe5ab22315a5a682eb377e0a0df5a0c01446a28282d9d943b` | `082f2428467ae197c46765081bb3ec3cd10be5357916bd0e1673b4e1ec7437d4` |
| `src/ggml-cuda/concat.cu` | `3cba4d2a26e6a70970ae9b77017f449e41d48e9b1297ba64966e18c6146e53db` | `9eca3e1c5f5ae55c01f6b718e81d002e6432064c8668fc1e82c4b1403bc45b7e` |
| `src/ggml-cuda/cpy.cu` | `d0bee0ec1fb79f6fdbe1d39719c2a595e0bd6f51bbaea6536284848f80637341` | `306911de61305c4dd27d4a9f998bb480ff7d4d15acdded807b8abec5f33f166a` |
| `src/ggml-cuda/fattn.cu` | `8cd53030107e6956288c2cc1f3e2dc245ef2332b0c39bc23cacadbda3a997aa9` | `e3bc0ddbfb6c56e806be478ce869df42e32386aca043f3bc20141473b4e8da24` |
| `src/ggml-cuda/ggml-cuda.cu` | `1315f06baff63da74e085b7ec01974474dac1f8ec1074af13eb70faba4751961` | `8a2911a6650fc9d039404ac6fd3aa5a0da9caf1ebefb7191e72975de75709193` |
| `src/ggml-cuda/sam-precise.cu` | — | `a602054915b0b9cc648de9aacafe1e8be220690e231e03d7f9c7488c778210c5` |
| `src/ggml-cuda/sam-precise.cuh` | — | `e35a1fc7e554e072cb9dc761d1ecdf6d0c878c97da3e1a8dfe67edb717d83080` |

## Validation status

The [v0.26.0 upgrade plan](../../docs/plans/20261009-012531-ggml-v0260-upgrade.md)
records source preparation, Linux CPU/CUDA checks, model regressions and limitations.
Metal was ported to the split kernel layout, but v0.26.0 has not been compiled or
executed on Apple hardware in this upgrade. Earlier M4 Pro checks do not certify
this version; the tensor API branch also remains unverified.
