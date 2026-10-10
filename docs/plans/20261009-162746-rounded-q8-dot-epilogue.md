# Corrected-library Q8 dot and F32 epilogue acceptance

Created: 2026-10-09 16:27:46, Asia/Shanghai.

## Scope

Re-run independently verified Q8_0 same-operand dots and F32 bias/GELU
epilogues against the corrected pinned CUDA library, rather than transferring
the old GGML revision's arithmetic receipt. Keep the current model/campaign,
source snapshot and old immutable results unchanged. Use the two already
predeclared real-model `mlp-lin1` probe inputs: MMVQ M=4,736/N=3/K=1,024
(SHA-256 `1839a77491804feb1dce6ae2cd0e56dab6449052872426723a4ac11d7d0b1c25`)
and MMQ M=4,736/N=64/K=1,024 (SHA-256
`cfbd4d14ba0422cffd3f272b3c88682e7aabd0fe44a890aaee1b6343a3226c49`).
These are component fixtures, not proof of all image-session inputs.

## Approach and steps

1. Bind the corrected build's `sam_ggml_linear_probe` and RHS staging probes
   through `ldd` and hashes to the corrected CUDA library. Run both inputs
   in new ignored directories with `ggml-q8 --dump-q8-operands` and retain
   Q8_0 weights, F32 RHS, Q8_1 staging, F32 raw dots and final outputs.
2. Apply the unchanged MMVQ/MMQ independent staging and F64 packed-dot
   verifiers under the 2e-5 relative-L2 / 1e-6 zero-norm gate. Independently
   recompute the bias and exact GELU from each raw dot output with the
   existing F32 epilogue oracle. Reject nonfinite operands and outputs.
3. Rehash source, binaries, library, input, raw outputs and verifier scripts
   in an aggregate. Keep whole-recipe arithmetic and deployment `NOT_RUN`
   until graph/session and input-domain prerequisites are closed.

## Verification

Run via `rtk proxy` in `.venv-reference` with remote desktop stopped and no
competing GPU process. Retain all raw new evidence. Run the corrected CUDA
regression, documentation check and `git diff --check` after the cycle.

## Results

The corrected-build probe ran both predeclared inputs with raw Q8 operands
enabled. The corrected native MMVQ/MMQ RHS staging probes produced fresh
Q8_1 payloads. The independent packed-operand F64 dot verifier passed
**317,312** dots (14,208 MMVQ and 303,104 MMQ) with worst whole-output
relative L2 `5.848915638e-8`; both per-row maximum relative L2 values
were below the unchanged 2e-5 limit. An independently recomputed F32
bias/precise-GELU epilogue passed the same **317,312** outputs, worst
relative L2 `2.837382945e-7`. The native CUDA reports recorded no CPU
compute nodes. The component report binds exact input hashes, raw packed
operands, staging, dot and final output files, all verifier scripts,
probe binaries, the frozen campaign and corrected CUDA library. Its ignored
`rounded-q8-dot-epilogue-20261009/epilogue-reconciled.json` SHA-256 is
`e5dbd04ca26c9438736176997dd5e28d9e60e65c16acbb98b4a2abdde0c14586`.
Raw evidence is retained under
`/mnt/SSD1/samcpp-validation/q8-mmq-rounded-candidate-20261009/` in this
validation workspace and is unavailable in a clean checkout. This probe
uses the diagnostic linear graph's CUDA F16 compute-mode setting; it
establishes the Q8 packed-dot and F32 epilogue component arithmetic on
those operands, not the full text-image recipe. Whole-recipe arithmetic
and deployment remain `NOT_RUN`.
