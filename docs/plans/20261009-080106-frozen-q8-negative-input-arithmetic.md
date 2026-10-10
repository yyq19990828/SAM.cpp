# Frozen Q8 arithmetic on an independent development prompt

Created: 2026-10-09 08:01:06, Asia/Shanghai.

## Scope

Extend the exact frozen CUDA Q8_0 same-operand matrix arithmetic evidence
beyond the one `tv` image. Use the already-frozen development performance
case `perf-negative`: COCO `000000303566`, negative prompt `vase` (`p6`).
Keep the 1,024-image reserve unopened and the exposed final evaluation
unused for tuning. Preserve the campaign, model, binary, GGML library,
thresholds and quality verdict. This second development input is additional
component evidence, not campaign-wide arithmetic or performance acceptance.

## Approach and steps

1. Bind the case ID, prompt, image hash and output to the frozen dataset,
   performance-case manifest, native development export and exact Q8 recipe.
   Check disk space before using the existing isolated Q8 capture sidecar.
2. Capture all active Q8 matrix operations in one fresh process; verify
   each captured GGUF weight slice, Q8_1 RHS staging and CUDA dot output
   against an independent same-operand F64 reference under the unchanged
   2e-5 relative-L2/1e-6 zero-norm rule. Require final masks, boxes and
   ranking to agree with the unobserved frozen export.
3. Reconcile count, shape, model/library/campaign identities, all payload
   hashes and final output. Retain the complete receipt and required
   evidence. Do not promote whole-recipe arithmetic or deployment on two
   development inputs.

## Verification

Run the frozen executable and independent verifier in the isolated
reference environment, the relevant tool suite, docs checker and
`git diff --check`. Clean only obsolete intermediate files after the
completed cycle; preserve bound payloads and the current build.

## Results

The frozen `perf-negative` development case was bound to the original image,
`vase` prompt, `p6` native output, campaign, model and probe binary. The
case-binding receipt is `build/q8-integrated-arithmetic-20261009/negative-case-bound.json`
(SHA-256 `a4f430c90b01f573702326ce4ec500c63bd90be5d45b7dfe04846b1b12065382`).
The frozen binary completed inference under the existing capture sidecar. Its
final payload, apart from runtime measurements, exactly matched the prior
unobserved native development export.

Independent verification passed all 327 active Q8_0 matrix nodes, checking
their exact GGUF weight slices, MMVQ/MMQ Q8_1 RHS staging and 161,445,056
same-operand F64 reference dots under the unchanged numerical gate. The worst
relative L2 was `9.324993300199227e-08`. The per-node receipt is
`build/q8-integrated-arithmetic-20261009/negative-capture-verification.json`
(SHA-256 `c0933b447b8b885e82ad92aa36c59bec441a44afbda025f5ce9234ec5ff4d645`).
The first development case had the same node/dot counts and a worst relative
L2 of `1.1302755267997587e-07`.

The two-input reconciliation independently rehashed all 3,270 bound per-node
payload files, both receipts and their frozen model, executable and CUDA
library identities. All 327 Q8_0 weight payloads were unchanged between
inputs; the RHS activation and dot-output payloads changed for 312 nodes.
The reconciliation is
`build/q8-integrated-arithmetic-20261009/negative-two-input-reconciled.json`
(SHA-256 `46e3af2dd5abb0f7823c6dd4c2516b8f1747413e243e7c5a1516cc59f144a19b`).
These `build/` receipts and raw payloads are Git-ignored local evidence and
are available only in the validation workspace. The campaign, thresholds,
quality decision and 1,024-image reserve were not changed or used for tuning.
Whole-recipe arithmetic, exclusive-GPU performance and deployment remain
`NOT_RUN`; the remote-desktop CUDA context still prevents an exclusive run.

Verification completed with 170/170 tool tests passing, the bilingual/local
link checker passing on 104 documents, and `git diff --check` passing. The
post-run cleanup review found no disposable new intermediate: the raw capture,
staging payloads, observer output and receipts are all required to recheck
this result. The current verified build and prior bound evidence were retained.
