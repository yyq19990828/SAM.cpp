# Frozen-recipe F32 attention arithmetic on real operands

Created: 2026-10-09 07:08:27, Asia/Shanghai.

## Scope

Extend the one-case frozen CUDA arithmetic audit from all 519 matrix nodes to
the 87 `FLASH_ATTN_EXT` nodes identified in the same graph inventory. This is
operator-component evidence, not a change to the frozen quality campaign,
whole-recipe arithmetic gate, model or binary. Formal performance remains
unavailable on the shared RTX 4090.

## Approach and steps

1. Inventory all attention shapes, input types, mask shapes, precision flags
   and scale parameters. Verify that the frozen SAM-specific CUDA F32 attention
   path is the path selected for these graph nodes.
2. Add a build-only `LD_PRELOAD` sidecar to the exact frozen image probe binary.
   Capture each node's real Q, K, V, optional mask and raw output with logical
   strides and complete graph order; make output directories new and fail
   closed on unsupported layouts.
3. On a small trial, independently recompute attention in F64 from the saved
   operands: QK score, scale, F16 mask, stable softmax and weighted V. Then
   process all 87 nodes in bounded query tiles. Apply the existing 2e-5 F32
   tensor relative-L2 rule and 1e-6 zero-norm absolute rule without changing
   either. Record per-node errors and any nonfinite values.
4. Reconcile graph order and all artifact hashes against the prior matrix
   reports, campaign, model, binary, GGML library and final output. Preserve
   failed captures or checks for diagnosis. Keep attention-component and
   whole-recipe statuses separate.

## Verification

Compile with warnings as errors; compare no-preload and callback-only controls
for any final-output drift. Run the isolated tool tests, documentation link
checker and whitespace check. Remove trial artifacts after final evidence is
retained.

## Results

The frozen graph inventory has exactly 87 `FLASH_ATTN_EXT` nodes across its
four partitions, with counts 32, 6, 24 and 25 in graph order. All
Q/K/V and outputs are F32; 37 nodes use F16 masks and 50 are unmasked. The
frozen SAM CUDA library selects its explicit F32 attention path for these
shapes and precision flags. The same frozen image binary, GGML CUDA library,
campaign, GGUF and development case from the matrix audit were used without
modification.

The two-node trial passed 95,551,488 F64-recomputed QK score pairs. The
complete capture then passed **87/87 attention nodes**, covering
4,359,948,024 score pairs and 188,527,104 output values. The worst
tensor-relative L2 was `7.15875809898605e-07` under the unchanged `2e-5`
limit. The full report is
`build/q8-integrated-arithmetic-20261009/attention-full-verification.json`
(SHA-256 `a6b462d4ea5144cb6424c3d196121d7e78300d194971091e31b3da69d2bbef19`).
Its 2.8 GB of logical Q/K/V, mask and output captures are retained under
`build/q8-integrated-arithmetic-20261009/attention-full/` and available only
in this Git-ignored validation workspace.

A read-only reconciliation matched all 87 captures to their source and output
types, shapes, strides and graph order in the independent inventory. It
rehashed all 385 attention payloads, checked the frozen binary and loaded
CUDA library, bound the campaign/model and preceding 519-matmul receipt, and
confirmed that the full observed inference payload is exactly equal to the
original unobserved export after runtime metadata is removed. The receipt is
`build/q8-integrated-arithmetic-20261009/attention-reconciled.json` (SHA-256
`b9164f068ec3ec0a7869c9013bd0a3e561b8663c1e5c9e79889862bc27d8f614`).

The first trial inadvertently selected the different `mixed-q8_0` **feature
cache mode** and yielded a different graph/output; it was discarded before
verification and immediately rerun with the frozen `f32` cache argument. The
correct trial and full capture have the original graph sizes and exact final
payload. This argument mismatch was diagnostic setup error, not evidence of a
callback-induced numerical issue. Superseded trial dumps were removed.

This establishes attention-component arithmetic for one text-image case in
addition to all 519 matrix nodes. The many remaining non-matrix graph
operators and unobserved inputs still lack equivalent independent checks.
Whole-recipe arithmetic, exclusive-GPU performance and deployment remain
`NOT_RUN`; no frozen threshold or final quality verdict changed.

Validation: the sidecar compiled with `-Wall -Wextra -Werror`; the independent
F64 verifier and read-only reconciliation passed; the isolated tool suite
passed 170 tests; the documentation checker passed 96 documents; and
`git diff --check` passed. The GPU still shows the remote-desktop compute PID,
so the formal performance runner's exclusivity precondition is unmet.
