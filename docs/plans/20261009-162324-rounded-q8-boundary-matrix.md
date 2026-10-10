# Corrected-library CUDA Q8 native boundary matrix

Created: 2026-10-09 16:23:24, Asia/Shanghai.

## Scope

Exercise the corrected, pinned CUDA GGML library's native Q8_1 RHS staging
on an explicitly fixed boundary matrix. Preserve the frozen campaign,
source, model, existing receipts and numerical gates. This is component
evidence for the whole-recipe arithmetic prerequisite, not a substitute for
the still-missing multi-input/session full-graph audit. Use no final or
reserve images.

## Approach and steps

1. Generate these deterministic F32 fixtures before running either native
   probe: MMVQ 4×128 with zero blocks, separate signed ±65,504 endpoints,
   ±2,000 outliers, ±30,000 pairs and signed half-step ties; MMVQ 1×16,384
   with zero/dense blocks, first/last-block endpoints and ties; and a
   negative MMVQ 1×32 block with two +40,000 values whose original-input
   sum overflows F16. MMQ 65×160 includes zero/tail padding, ±3e38,
   signed outliers and ties; MMQ 65×16,384 exercises the maximum accepted
   K with zero/dense blocks, endpoint signs and ties. Fixed deterministic
   small-value patterns and exact indices are encoded in the generator
   source and manifest, both hashed before probe execution.
2. Run the **existing corrected-build native probes** against those inputs,
   require that `ldd` resolves the corrected CUDA library, and apply the
   unchanged independent MMVQ/MMQ packed-byte verifiers. The overflowing
   F16-sum fixture must be rejected by the independent verifier; it is an
   expected negative, not a passing in-domain input. Record zero, endpoint,
   finite-scale/sum, integer, padding, K and tie outcomes separately.
3. Independently rehash every new input, native output, source, binary,
   verifier, library and manifest into a read-only receipt. Do not promote
   whole-recipe status unless the distinct epilogue and session/full-graph
   prerequisites also pass.

## Verification

Use `rtk proxy` and the isolated `.venv-reference` environment; keep the
remote-desktop user service inactive and check GPU occupancy. Retain raw
new payloads and receipts under the Git-ignored evidence volume. Run the
relevant CUDA regression, documentation/link check and `git diff --check`
after the cycle. Clean only disposable intermediates.

## Results

The input generator froze all five fixtures before native execution; the
ignored `inputs/manifest.json` SHA-256 is
`40ed201cf34521d2244e4ea0f71575ea00a337da57ca5253e12e22ad0beaf142`.
Both corrected-build probes resolved the corrected CUDA library SHA-256
`d50d7f7c44c964d775dba8e89cb12b4a9b93e2d21b3191a4e374ec594a076228`.
The existing independent verifiers were rerun by a separate read-only
reconciler after their initial pass. Both valid MMVQ fixtures passed **528**
Q8_1 blocks with zero scale, integer or F16-sum errors. Both valid MMQ
fixtures passed **8,580** D4 blocks / **34,320** subblocks with zero scale
or integer errors. The 16,384-wide maximum-K MMVQ/MMQ inputs passed, as
did zero blocks, signed ±127 endpoints, half-step ties, the 160-wide MMQ
padding tail and ±3e38 finite extremes. Four stored MMQ scales used the
unchanged four-F32-ulp allowance; no integer tie allowance was needed.

The separate 1×32 MMVQ input produced a native F16 original-input sum of
`+inf` (`0x7c00`), and the independent verifier rejected that payload as
required. This negative case is **EXPECTED_REJECTION**, not a passing
in-domain input or evidence that the public inference API rejects that
activation. The aggregate rehashed every input/native payload, probe,
verifier, source generator, manifest, library and binary. Its ignored
`rounded-q8-boundary-20261009/boundary-reconciled.json` SHA-256 is
`d28b28ec5afdaa6f75017db1df28118a572ff3d24ab029d1dc5043005813ec01`.
All raw evidence is retained under
`/mnt/SSD1/samcpp-validation/q8-mmq-rounded-candidate-20261009/` in this
validation workspace and is not distributed with a clean checkout.
Component status is `PASS`; whole-recipe arithmetic and deployment remain
`NOT_RUN`.
