# Frozen F32 matrix arithmetic on an independent negative prompt

Created: 2026-10-09 08:22:58, Asia/Shanghai.

## Scope

Extend the exact frozen CUDA custom-Q8 recipe's real-operand arithmetic
evidence to all F32 matrix nodes on the predeclared `perf-negative`
calibration/development input (`coco-000000303566-p6`, `vase`). Q8 matrix
arithmetic has already passed for this input. This is a second full matrix
partition, not yet a second complete active-graph audit. Keep the frozen
model, binary, CUDA library, quality result and 2e-5/1e-6 arithmetic limit.
Do not use the final evaluation or unopened reserve for tuning.

## Approach and steps

1. Bind the prior negative-case receipt, image/prompt, frozen native output,
   campaign and eight-case Q8 reconciliation. Check space on the separate
   evidence volume and use a fresh directory reached through Git-ignored
   `build/q8-integrated-arithmetic-20261009/many-external/`.
2. Reuse the unchanged F32 capture sidecar on the exact frozen binary. Check
   all F32/F32/F32 matrix node operands and outputs against an independent
   same-operand F64 computation; enforce the original numerical gate, finite
   values and named GGUF weight parity. Compare final masks, boxes, ranking
   and scores with the unobserved native development export.
3. Rehash every retained F32 payload, reconcile node order/shape and frozen
   identities with both the earlier F32 input and this input's 327 Q8 nodes,
   and distinguish full matrix coverage from whole-graph arithmetic.

## Verification

Run the exact binary, isolated reference verifier and read-only reconciliation.
Run the relevant tool suite, docs checker and `git diff --check`; preserve
the current verified build and all raw evidence needed for reproduction.
Review new intermediates for cleanup after validation.

## Results

The frozen binary captured all 192 F32 matrix nodes on the bound negative
development prompt. An independent F64 same-operand verifier passed
1,897,738,953 dots under the unchanged gate; the worst tensor relative L2
was `3.4587719756585384e-06`. All 141 named F32 GGUF weight operands
matched their stored bytes. The callback-run final inference payload matched
the unobserved native export exactly, including scores, masks, boxes and
ranking. Its receipt is
`build/q8-integrated-arithmetic-20261009/f32-negative-verification.json`
(SHA-256 `2f99d997466fa5121950d90ae05c88acfaf86f91c55bd395e163abe750181e72`).

The independently generated negative-input graph inventory (SHA-256
`9c5fbe00fdb8b992d9a426859afcb2adc2d8b33d4787669a0cb14250eba2196f`)
contains the same four graph partition sizes as the first input and exactly
519 matrix nodes: 327 Q8_0/F32/F32 plus 192 F32/F32/F32. A read-only
reconciliation matched every node's name, type, order, shape and batch count
to the two frozen-binary ledgers, and rehashed 2,787 raw payload files from
this and the first F32 case plus this input's Q8 case. Relative to the
first input, 24 F32 left operands, 176 right operands and 186 dot outputs
changed. The combined receipt is
`build/q8-integrated-arithmetic-20261009/negative-all-matmul-reconciled.json`
(SHA-256 `4b31a5975b5a182c25d42da8438494509486f3c4b49886131c28d46f7d32df32`).
The 17 GiB full raw capture is under the Git-ignored local
`build/q8-integrated-arithmetic-20261009/many-external/f32-negative-capture`
path, backed by `/mnt/SSD1` and available only in this validation workspace.
The follow-on [eight-input matrix audit](20261009-082627-eight-input-f32-matmul-arithmetic.md)
reuses this immutable receipt. This is a second complete matrix partition,
not a second all-operator graph audit. Whole-recipe arithmetic,
exclusive-GPU performance and deployment remain `NOT_RUN`.

The completed validation cycle passed the isolated Python tool suite
(170/170), the bilingual/local-link documentation check (109 documents)
and `git diff --check`. Its raw capture and receipts remain necessary
validation evidence; no disposable intermediate was retained.
