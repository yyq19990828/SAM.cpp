# Native Q8_1 right-hand staging oracle

Created: 2026-10-09 03:23:03, Asia/Shanghai.

## Scope

Advance the frozen v2 same-operand arithmetic requirement for native CUDA quantized matrix multiplication. Existing Q-weight tests use decoded weights against the original F32 right operand with a broad tolerance; the actual MMVQ path stages that operand as Q8_1. Independently check the CUDA Q8_1 staging bytes before attempting a tighter raw-dot comparison. This does not certify MMQ, every quantized weight type, or a complete model recipe.

## Approach and steps

1. Confirm the pinned GGML v0.26.0 MMVQ staging function and `block_q8_1` layout on the current CUDA build. Export its bytes through an opt-in diagnostic probe, without modifying GGML or the runtime.
2. Build an independent host verifier for block maximum, rounding, half scale/sum, signed values, layout and decoded operand. State narrowly any necessary CUDA fast-math or reduction allowance, preserving exact stored-byte and decoded-bit checks.
3. Cover zero, rounding boundaries, outliers, tails and representative SAM K/row shapes on RTX 4090. Bind source, input, binary and linked-library identities in fresh local evidence.
4. If the staging bytes pass, add an opt-in post-timing raw-dot export to the existing GGML Q8 linear probe. Compare its native matrix output with an independent dot over the exact Q8_0 weight bytes and independently validated Q8_1 staging bytes. Include MMVQ-sized real SAM shapes and adversarial boundaries; report relative L2 and worst elements without changing the existing frozen gate.
5. Keep frozen v2 receipts and thresholds unchanged. A component result does not certify MMQ or a complete model recipe.

## Verification

Rebuild the diagnostic CUDA target; run focused adversarial tests, native runs and the appropriate tool/CTest checks. Check `git diff --check` and remove disposable intermediates while preserving the verified build, usable models and required evidence.

## Results

On the pinned GGML v0.26.0 CUDA build and RTX 4090 (SM 8.9), the opt-in
`sam_cuda_q8_rhs_probe` called GGML's exported `quantize_row_q8_1_cuda` with
Q8_0 weights selected. The independent
[Q8_1 verifier](../../tools/validation/verify_q8_rhs_staging.py) passed
94,691/94,691 native staging blocks across zero, rounding-boundary,
QKV, MLP `lin2`, tail-width and F16-scale-boundary fixtures. It found zero
scale, integer or original-input-sum errors. One scale and 70 integer results
used the explicitly bounded adjacent-value allowance at rounding boundaries;
the stored sum field matched the F32 CUDA reduction tree exactly. The
diagnostic invokes the same exported GGML function used by MMVQ, but it does
not capture MMVQ's private scratch buffer.

The opt-in `sam_ggml_linear_probe --dump-q8-operands` saved packed Q8_0
weights, F32 RHS input and post-timing raw CUDA output. The independent
[same-operand dot verifier](../../tools/validation/verify_q8_mmvq_dots.py)
recomputed every dot from signed INT64 block products and F64 decoded scales,
after independently checking Q8_1 staging. Five N=3 cases passed:
QKV (M=3072, K=1024), attention projection (1024, 1024), MLP `lin1`
(4736, 1024), MLP `lin2` (1024, 4736), and a 17-row K=4736 tail. This
checks 29,619/29,619 output dots; worst whole-output relative L2 was
5.97e-8 versus the existing 2e-5 same-operand limit. A QKV Nsight Systems
trace recorded both `quantize_q8_1` and `mul_mat_vec_q` CUDA kernels,
confirming that run selected MMVQ. The earlier broad GGML Q-weight check
compares against the original F32 RHS, so its tolerance and purpose remain
unchanged.

The immutable local summary is
`build/q8-rhs-staging-20261009/summary.json` (SHA-256
`24186be77fb1d0e92fb03e5bcc34b4767945b320d4679de121e143dd60624a10`).
Its bound input, receipt, source, binary, library and profiler files are in
the same ignored `build/q8-rhs-staging-20261009/` directory and are available
only in this validation workspace. The component result is **PASS**;
MMQ, the full model recipe, other quantized weight types and deployment
qualification remain **NOT_RUN**. No frozen v2 threshold or receipt changed.

The final isolated Python tool suite passed 164/164 checks, and CPU/CUDA
quantization CTest passed 2/2. The ordinary GGML Q8 probe still ran without
the dump flag; the flag was rejected for F16 mode before creating an output
directory. Documentation tests and final whitespace verification followed
the documentation update.
