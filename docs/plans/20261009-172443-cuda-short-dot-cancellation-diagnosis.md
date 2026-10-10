# CUDA short F32 dot cancellation diagnosis

Created: 2026-10-09 17:24:43, Asia/Shanghai.

## Scope

Diagnose the frozen corrected candidate's `small-whitespace` graph-7
`ddec.presence_token_head.layers.2.weight` F32 dot failure without editing
that candidate, its gate, model, quality population or immutable receipts.
Determine whether a general short, single-output F32 dot with F64 products
and accumulation is a viable next implementation candidate. This diagnostic
does not qualify a new recipe.

## Approach and steps

1. Rehash the retained 256-element raw operands and observed output against
   the existing failure receipt. Confirm GGML's explicit-F32 path selects
   cuBLAS pedantic arithmetic and capture the current GPU/compiler identity.
2. Build an isolated CUDA diagnostic under Git-ignored `build/`. Compare
   pedantic cuBLAS and a bounded single-block F64 reduction on the captured
   operands and independent synthetic cancellation, signed, zero and tail
   inputs. Use an independent host exact-product reference and unchanged
   `2e-5` relative / conditional `1e-6` absolute rules.
3. Record numerical results, kernel timing and source/binary/raw hashes.
   If the general kernel succeeds, specify its dispatch bounds and a separate
   new candidate identity, tests and full requalification requirements.
   Do not promote the frozen candidate or revise a threshold post hoc.

## Verification

Compile and run on the RTX 4090 while remote desktop is stopped. Retain raw
diagnostic inputs and result under a fresh ignored directory; run the docs
checker and `git diff --check`. No production source is changed in this
diagnostic cycle. Preserve original checkpoints, usable GGUF, the current
verified build and corrected-candidate validation archive.

## Results

The captured operand hashes matched the frozen receipt. Replaying the
1×1×256 dot through standalone cuBLAS `CUBLAS_COMPUTE_32F_PEDANTIC` on the
RTX 4090 returned `-0.0043205022811889648`, bit-identical to the model's
failed node. Thus the observer did not cause the mismatch and the kernel was
already on the pedantic F32 path.

The single-block F64-product/F64-sum probe returned
`-0.004320603795349598`, only `9.277371232352394e-11` absolute and
`2.147239569049425e-8` relative error against the independent exact-product
Decimal reference. It passes the original `2e-5` relative rule without a
threshold change. Zero, signed-scaled, and deterministic cancellation inputs
at K=32, 256, 257 and 1,024 also passed the same frozen gate. The original
captured case remains a real F32 pedantic failure, not a revised PASS.

The microprobe's 1,000-launch event timings on this GPU were about 2.72 μs
per pedantic cuBLAS launch and 2.25 μs per F64 kernel launch at K=256. This
is a dispatch microbenchmark only; it does not establish model latency,
memory or a deployable performance label.

The ignored diagnostic is `build/f32-short-dot-20261009/`: `probe.cu`,
`run_probe.py`, copied raw operands and `results.json` (SHA-256
`a38e757f778b82432a5bfa41242e9223cfe9286f4bf8efdc67e5966ec7f324bc`).
The compiled probe SHA-256 is
`142b13bb79148cd37cdbe5e930395ab0e6b2bb030ee6ba654bc0394c09099cc3`;
CUDA toolkit version is 13.3. The current candidate patch, build, model,
thresholds and archived receipts remain unchanged. The documentation checker
passed 126 documents and `git diff --check` passed.

A next candidate can dispatch a general explicit-F32, contiguous, single-
output, short dot to this bounded kernel, with cuBLAS retained outside that
predicate. It needs a new patch/source/binary identity, an active Release
regression over both dispatch boundaries, full graph and image-quality
requalification on a fresh holdout, and independent end-to-end performance.
