# Frozen normalization and activation arithmetic on real operands

Created: 2026-10-09 07:14:34, Asia/Shanghai.

## Scope

Continue the exact frozen CUDA Q8 image recipe's one-case arithmetic audit
after its 519 matrix and 87 attention nodes. Cover all 177 `NORM` and 140
`UNARY` nodes seen in the independent graph inventory. Preserve the frozen
campaign, GGUF, image binary, CUDA library and all thresholds. This remains
operator-component evidence; other graph ops and formal exclusive-GPU
performance are separate requirements.

## Approach and steps

1. Confirm each normalization/activation node's source and output types,
   shape, strides, epsilon or unary function and graph order. Record the
   actual F32 source and result immediately after each node using a build-only
   preload sidecar on the exact frozen image binary. Use a small trial before
   the full capture and reject unexpected layouts or functions.
2. Independently recompute the saved operands in F64: normalize each GGML
   row over dimension 0 with the recorded epsilon, and evaluate the recorded
   activation function pointwise. Apply the existing `2e-5` tensor-relative
   L2 limit and `1e-6` zero-norm absolute rule without adjustment; reject
   nonfinite operands and outputs. Preserve any failure for investigation.
3. Reconcile every case with the earlier graph inventory and receipts for
   matrices and attention. Rehash retained operands, outputs, source,
   sidecar, binary, library, campaign, model and final inference payload.
   Keep whole-recipe arithmetic status `NOT_RUN` until every required graph
   operation and input scope is covered.

## Verification

Compile with C++17 warnings as errors, recheck payload parity and the
quality/performance identities, run the isolated tool tests, documentation
checker and whitespace check. Remove superseded trial captures and retain
the complete immutable evidence under ignored `build/`.

## Results

The full graph contains 177 `NORM` and 140 `UNARY` nodes. The latter are
57 `GELU_ERF`, 52 `RELU`, 12 `SGN`, 12 `ABS` and 7 `SIGMOID`. All their inputs
and outputs are F32. The exact frozen image binary, Q8 GGUF, GGML CUDA
library and development text-image case were reused with the `f32` feature
cache and compute modes.

The initial sidecar captured input and output *after* each node. For an
in-place GELU, these bytes were identical, producing an apparent 0.604
relative-L2 failure against the F64 oracle. This was a capture-timing error,
not a kernel arithmetic result. The corrected sidecar requests a callback at
every graph node so that each predecessor finishes, saves the selected source
**before** its node executes, then saves its output after execution. A
four-node trial including three normalizations and one GELU passed. The
earlier failed trial payloads and report are retained for diagnosis, while
the successful trial dump was removed after full validation.

The full capture passed **317/317 nodes** and 1,311,625,440 independently
recomputed output values. The worst tensor-relative L2 was
`5.107398020268101e-07`, below the unchanged `2e-5` limit. There were no
zero-norm reference cases. Its receipt is
`build/q8-integrated-arithmetic-20261009/norm-unary-full-verification.json`
(SHA-256 `ed49c49add31f353d4a829a9130d88f372575689b5715d53ea91395c89fe4d49`).
The retained 9.8 GB of input/output files reside under
`build/q8-integrated-arithmetic-20261009/norm-unary-full/`, available only in
this Git-ignored validation workspace.

A separate read-only reconciliation matched every node's order, operation,
source/output type, shape and strides with the independent graph inventory;
rehashed all 634 captured payloads; and bound the frozen binary, library,
campaign, model and prior matrix/attention receipts. The receipt is
`build/q8-integrated-arithmetic-20261009/norm-unary-reconciled.json` (SHA-256
`fa2014672a7da89a1e739f8c0a193641f2e313d90e6eafd3eeb0ba965894f525`).
The all-node callback changes only final `query_scores`, by at most
`1.1900000007614153e-07`; masks, boxes, ranking and feature cache match
the unobserved frozen export. This synchronization effect is diagnostic,
and does not change the final quality result or numerical gate.

The one-case component audit now covers all 519 matrix, 87 attention, 177
normalization and 140 unary activation nodes. Other elementwise, layout,
convolution and graph operations, along with other prompts/images, remain
unverified at this level. Whole-recipe arithmetic, exclusive-GPU performance
and deployment remain `NOT_RUN`.

Validation: the final sidecar compiled with C++17 `-Wall -Wextra -Werror`;
the F64 verifier and read-only reconciliation passed; the isolated tool
suite passed 170 tests; the documentation checker passed 97 documents; and
`git diff --check` passed. The remote-desktop compute PID still occupies the
GPU, so formal performance remains unavailable in this session.
