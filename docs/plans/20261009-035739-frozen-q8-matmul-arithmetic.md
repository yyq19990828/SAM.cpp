# Frozen v2 CUDA Q8 matrix arithmetic

Created: 2026-10-09 03:57:39, Asia/Shanghai.

## Scope

Check the Q8_0 CUDA MMVQ and MMQ arithmetic against independently reconstructed
Q8_0/Q8_1 packed operands while linking the exact GGML v0.25.3 CUDA library
used by the frozen v2 image recipe. Existing v0.26.0 probe evidence does not
automatically transfer across library versions. Preserve frozen v2 gates,
recipes and outputs. This is a bounded matrix-component check, not a new
whole-recipe or failed-Q8-quality PASS.

## Approach and steps

1. Build isolated opt-in probes from current source against the retained
   v0.25.3 headers and *existing* frozen CUDA shared library, without
   rebuilding or modifying any frozen binary/library. Confirm dynamic
   resolution, ABI and SHA-256 identities.
2. Reuse frozen, hash-bound SAM-shaped linear inputs for N=3 MMVQ and N=64
   MMQ plus zero/tie/tail/max-K boundaries. Export the native packed weight,
   right operand, staged Q8_1 bytes and raw CUDA dots.
3. Independently verify Q8_1 staging and each exact-operand dot with the
   existing frozen 2e-5 F32 output limit and zero-norm rule. Confirm actual
   MMVQ/MMQ kernel selection with available GPU profiling evidence.
4. Bind all inputs, source, binaries, libraries and outputs in immutable local
   receipts. Document measured scope, test tools and clean intermediates.

## Verification

Any current-source/old-ABI mismatch or different CUDA library hash stops the
claim. Retain all failed checks. Use the isolated Python reference environment
for independent verifiers, then run affected tool tests, docs checks and
`git diff --check`.

## Results

The isolated linear and MMVQ/MMQ staging probes were compiled against the
retained v0.25.3 GGML headers and linked to the *existing* frozen v2
`libggml-cuda.so.0` (SHA-256
`1e4f954340ea2c30cb60deb49f4ccda6780993b2e09cb81bc8105aa053c925ef`).
All three new binaries and the frozen v2 image binary dynamically resolve
that exact library. No frozen build or library was rebuilt. The frozen
vision-Q8 and full-Q8 development exports also bind the same image binary and
library; their recipe SHA-256 values are
`09d743a2a840ce81b2f16484e9e37847368c57aa8efea46845acd8e447cbc593`
and `ab9a7513a2e52e60e19b0a22a0d21f4162662d80136df141034aef110fc4cc90`.

All 12 selected cases passed independent Q8_1 staging and exact packed-operand
dot checks: five N=3 MMVQ cases checked 29,619 dots, and seven MMQ cases
checked 720,145 dots. Together they cover 749,764 dots and 221,208 staged
blocks. The worst nonzero whole-output relative L2 was 7.38e-8, below the
unchanged 2e-5 F32 same-operand limit. One case exercised zero-norm output;
the N=65 case covered K=16,384 and rounding-tie inputs. The MMQ scale verifier
used its existing bounded 4-ULP fast-math allowance 202,255 times; integer
rounding-tie allowances were zero. The N=64 and tail fixtures repeat three
sample activation rows, so the coverage is not a claim about every model
activation. Nsight Systems recorded `quantize_q8_1`/`mul_mat_vec_q` for N=3 and
`quantize_mmq_q8_1`/`mul_mat_q` for N=64.

The [read-only reconciler](../../tools/validation/reconcile_frozen_q8_matmul.py)
re-executes both independent dot verifiers, checks their original operand
hashes, all probe/source/library identities, the two Nsight traces, and the
frozen exports and failure reports. Its bound receipt is
`build/frozen-q8-matmul-20261009/summary-v4.json` (SHA-256
`5e8c037e8940991396c0fdd971551310211942b1b0a7c4317034c8257846cf79`,
127 bound files, available only in this validation workspace). Substituting
the v0.26.0 image binary was rejected before receipt creation.

This establishes a **PASS for the selected Q8_0 CUDA matrix components on the
frozen library**. Both vision-Q8 and full-Q8 frozen development quality remain
**FAIL** under their unchanged object-rate gate; the F32/mixed-Q8 cache recipe
does not use Q8 weight matrix multiplication. No full-model Q8 arithmetic or
deployment PASS follows from these component results. Other quantized weight
formats and untested shapes remain outside this receipt.

The frozen build's focused `sam_quantization_cuda` CTest passed 1/1. The
isolated Python tool suite passed 169/169, documentation/link checks passed
89 documents, all 127 bound summary files rehashed correctly, and
`git diff --check` passed. Redundant intermediate aggregate JSON files were
removed after the final receipt was fixed; original frozen evidence and
the 12 per-case receipts remain unchanged.
