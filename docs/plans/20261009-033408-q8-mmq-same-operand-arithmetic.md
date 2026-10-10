# Native Q8_0 CUDA MMQ same-operand arithmetic

Created: 2026-10-09 03:34:08, Asia/Shanghai.

## Scope

Advance the frozen v2 arithmetic prerequisite for CUDA Q8_0 matrix
multiplication at image-token batch sizes. The prior Q8_1 oracle covers MMVQ's
32-value/half-scale staging, while MMQ uses a distinct 128-value layout with
four F32 scales for Q8_0 weights. Check that layout and native MMQ output
against independently reconstructed packed operands. This is a component
check, not a new model-quality or deployment result.

## Approach and steps

1. Confirm the pinned GGML v0.26.0 Q8_0 MMQ staging ABI, row transposition,
   scale and rounding semantics. Identify representative SAM and adversarial
   matrix dimensions that select MMQ on RTX 4090, with a profiler trace.
2. Add an opt-in diagnostic that calls GGML's exported MMQ staging routine and
   saves its native bytes. Build an independent Python verifier for all scales,
   signed values, row mapping and bounded tie behavior.
3. Reuse the opt-in raw Q8 linear-probe output. For the same F32 RHS, calculate
   every output dot from the validated MMQ staging and packed Q8_0 weights,
   using signed INT64 block products and F64 scaling. Compare against the
   unchanged F32 same-operand relative-L2 limit of 2e-5, with a near-zero
   absolute check and worst element/row details.
4. Bind source, input, binary, library, profiler and receipt identities in
   fresh ignored local evidence. Keep prior receipts and v2 gates immutable;
   distinguish an MMQ component PASS/FAIL from full-recipe arithmetic status.

## Verification

Run focused adversarial verifier tests, native CUDA probes, the isolated Python
tool suite and quantization CTest. Check documentation links and whitespace,
then remove duplicate outputs and disposable profiling intermediates while
preserving the verified build and evidence.

## Results

On the pinned GGML v0.26.0 CUDA build and RTX 4090 (SM 8.9), the opt-in
`sam_cuda_q8_mmq_rhs_probe` called GGML's exported
`quantize_mmq_q8_1_cuda` with Q8_0 weights selected. The independent
[MMQ staging verifier](../../tools/validation/verify_q8_mmq_rhs_staging.py)
checked the 128-value D4 layout, four F32 scales, signed integers,
block-major/token-minor transposition and 512-value row padding. It passed
220,032/220,032 native blocks (880,128 32-value subblocks) with zero scale or
integer errors. CUDA fast divide/reciprocal placed 202,255 scales at another
F32 result within the explicit four-ulp allowance; no integer tie allowance
was needed. The staging diagnostic calls the same exported function as MMQ,
but does not capture MMQ's private scratch buffer.

The opt-in `sam_ggml_linear_probe --dump-q8-operands` exported Q8_0 weights,
the F32 RHS and raw CUDA output after timing. The independent
[same-operand verifier](../../tools/validation/verify_q8_mmq_dots.py)
recomputed all 720,145/720,145 output dots from signed INT64 block products
and F64 decoded scales after validating the MMQ staging bytes. Seven cases
passed: QKV, attention projection, MLP `lin1` and `lin2` at N=64; a 17-row
K=4736 tail at N=5184; zero input; and signed rounding boundaries at the
probe's maximum K=16384 with N=65. The worst nonzero whole-output relative
L2 was 7.38e-8, below the unchanged 2e-5 F32 same-operand limit. The zero
case produced exact zero output. A QKV Nsight Systems trace recorded both
`quantize_mmq_q8_1` and `mul_mat_q` kernels, confirming native MMQ execution.

The SAM-shaped input matrices derive from real block-0 calibration probes,
but their three source sample rows repeat across each N=64/5184 run. The
N=5184 case covers batch tiling with synthetic 17-row weights; it does not
represent a full image's diverse activations. Same-operand MMQ checks for
other weight formats and all
model-level arithmetic/quality/performance combinations remain separate.

The immutable local summary is `build/q8-mmq-20261009/summary-v2.json`
(SHA-256 `568dfca5f305bb2e62b416b162c3461e642cf91cad1ce1699f2ec8dc9fbfbb0f`).
Its bound inputs, receipts, sources, binaries, linked libraries and profiler
trace are in the ignored `build/q8-mmq-20261009/` directory and are available
only in this validation workspace. The earlier CMake source version bound by
the MMVQ summary was preserved at
`build/q8-rhs-staging-20261009/source-archive/tools/CMakeLists.txt`; the
earlier receipt itself was not changed. This selected Q8_0 MMQ component is
**PASS**, while full-recipe arithmetic remains **NOT_RUN**. Frozen v2
thresholds and historical model-quality/performance reports were unchanged.

The isolated Python suite passed 167/167 checks, and CPU/CUDA quantization
CTest passed 2/2. Documentation and final whitespace checks followed the
documentation update. Duplicate diagnostic output and the profiler's SQLite
intermediate were removed after evidence binding.
