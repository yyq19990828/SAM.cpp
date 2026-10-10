# Frozen binary elementwise arithmetic on real operands

Created: 2026-10-09 07:22:13, Asia/Shanghai.

## Scope

Extend the exact frozen CUDA Q8 image recipe's one-case arithmetic audit to
all 955 `ADD`, 448 `MUL` and 106 `SUB` graph nodes. Prior independent checks
cover 519 matrix, 87 attention, 177 normalization and 140 unary nodes. Keep
the campaign, GGUF, exact binary/library, final-quality verdict and 2e-5
F32 relative-L2/1e-6 zero-norm limits unchanged. This is component evidence;
whole-recipe arithmetic and exclusive-GPU performance are separate gates.

## Approach and steps

1. Use the retained graph inventory to confirm all elementwise source/output
   types, source-0 identity to output shape and source-1 broadcasting. The
   1,509 outputs occupy roughly 15.96 GiB; the union of their unique sources
   and outputs is roughly 29.35 GiB, compared with about 38 GiB if each
   operation repeats its inputs. Confirm adequate free space before capture.
2. Add a build-only preload sidecar to the exact frozen image binary. Request
   every node's evaluation callback to synchronize before a selected op.
   Save each logical F32 tensor at first appearance, before an in-place op
   can overwrite it. For repeated uses of the same tensor, compare every
   current byte to the retained first capture, failing closed if it changed.
   Record each elementwise node's graph order, op, shapes, strides and three
   tensor IDs. Test a bounded trial before the full capture.
3. Independently recompute every output from the retained F32 inputs in F64
   with GGML's broadcast semantics, in bounded chunks. Reject nonfinite
   values, apply the unchanged relative-L2 and zero-norm limits, and report
   F32 bitwise mismatch counts separately as diagnostics. Compare final
   inference payload to the frozen unobserved export.
4. Reconcile the node ledger with the graph inventory and prior matrix,
   attention and normalization/activation receipts. Rehash all retained
   operands/results and bind the sidecar source, binary, CUDA library,
   campaign, model and verifier. Keep any failing evidence and no whole-recipe
   PASS unless all required operators and input scope are covered.

## Verification

Compile with C++17 warnings as errors, run the isolated tool suite,
documentation checker and whitespace check. Clean superseded successful
trials after the full evidence is immutable; preserve the current verified
build, GGUF, manifests and any failures.

## Results

The independent graph inventory identified exactly 955 `ADD`, 448 `MUL`
and 106 `SUB` nodes in the frozen development text-image inference. Every
input and output is F32. Source 0 has the output shape; source 1 is either
that shape or uses dimension-1 broadcasting. The capture sidecar compiled
with C++17 `-Wall -Wextra -Werror` and used the exact frozen binary and
GGML CUDA library with the unchanged custom Q8 GGUF and `f32` cache/compute
modes.

A four-node trial (`ADD`, `MUL`, `ADD`, `MUL`) passed 21,233,664 F64-recomputed
values with zero F32 bitwise mismatches. The full run then captured all
**1,509/1,509** nodes. It stored 3,357 distinct logical tensors in roughly
29 GiB instead of duplicating each input for each operation; on every reused
tensor it reread the current CUDA bytes and compared them byte-for-byte to
the first retained copy. No reuse mismatch occurred. The retained files are
under `build/q8-integrated-arithmetic-20261009/elementwise-full/`, available
only in this Git-ignored validation workspace.

The independent verifier recomputed 4,284,831,745 output values in F64 and
passed the unchanged `2e-5` relative-L2/`1e-6` zero-norm rule for every
node. The worst tensor-relative L2 was `5.201046712757726e-08`. Every
observed F32 output was also bitwise identical to the correctly rounded F64
reference: **zero mismatches**. The report is
`build/q8-integrated-arithmetic-20261009/elementwise-full-verification.json`
(SHA-256 `f1a90e5f1b7b49ef9716b3ea3b925f8c71cb0dbf85ec39daf2212a641af50afa`).

A read-only reconciliation matched each operation, graph order, operand
type/shape/stride and deduplicated tensor ID to the separate graph inventory,
rehashed all 3,357 files, and bound prior matrix, attention and
normalization/activation receipts with the frozen binary, CUDA library,
campaign and model. Its receipt is
`build/q8-integrated-arithmetic-20261009/elementwise-reconciled.json`
(SHA-256 `4ad94742faf021543c457e9bc7ea59926e76f3eddd5a34c7986b373a3bfc2ff5`).
The callback changed only final `query_scores`, by at most
`1.1900000007614153e-07`; masks, boxes, ranking and feature cache match
the original unobserved export. This is diagnostic synchronization drift,
not a frozen-gate adjustment.

The one-case component audit now covers 519 matrix, 87 attention, 317
normalization/activation and 1,509 elementwise arithmetic nodes. Convolution,
softmax, layout, other graph operators and other inputs remain to be checked.
Whole-recipe arithmetic, exclusive-GPU performance and deployment remain
`NOT_RUN`.

Validation: the final sidecar compiled with `-Wall -Wextra -Werror`, the F64
verifier and read-only 3,357-file reconciliation passed, the isolated tool
suite passed 170 tests, the documentation checker passed 98 documents and
`git diff --check` passed. The successful trial artifacts were removed after
the full evidence was bound.
