# Frozen-recipe F32 matrix arithmetic on real operands

Created: 2026-10-09 06:59:04, Asia/Shanghai.

## Scope

Extend the exact frozen CUDA Q8 recipe's arithmetic evidence from its 327
active Q8_0 matrix nodes to the 192 remaining F32 `MUL_MAT` nodes in the same
development text-image inference. This is an operator-component audit toward
the full recipe. Keep the final quality campaign, models, binaries, GGML
libraries and all gates immutable. The only GPU remains shared with the
remote-desktop service, so this work is numerical, not performance evidence.

## Approach and steps

1. Freeze an inventory of all 192 F32/F32/F32 matmuls from the retained graph
   snapshot, including shapes, strides and estimated I/O volume. Check that
   no other `MUL_MAT` type escapes the Q8 and F32 partition.
2. Use a build-only `LD_PRELOAD` diagnostic sidecar on the **exact frozen
   image probe binary** to save both actual F32 input operands and the raw
   F32 output immediately after each F32 matrix node. Support noncontiguous
   views by copying logical rows, fail closed on unexpected batch dimensions
   or storage, and retain a complete ordered node ledger.
3. Independently recompute every output with F64 matrix multiplication in
   bounded chunks. Apply the unchanged 2e-5 relative-L2 F32 limit and 1e-6
   zero-norm absolute rule, report worst element/row and nonfinite values,
   and compare final inference payload with the original frozen export.
4. Reconcile every node with the separate graph inventory and bind the
   source, sidecar, operands, binary, library, campaign, model, input and
   verifier hashes. Retain failures; distinguish F32 matmul coverage from
   remaining attention, normalization, convolution, activation and other
   graph operations. Do not promote whole-recipe arithmetic prematurely.

## Verification

Use a small trial before the full capture; validate exact weights where they
come from the GGUF. Recheck source/binary/library identities and all output
hashes, plus compiler warnings, relevant tests, documentation links and
`git diff --check`. Preserve the verified full capture; delete superseded
trial dumps and other disposable intermediates.

## Results

The graph inventory for the frozen development case contains 6,076 nodes in
four partitions, with exactly 519 `MUL_MAT` nodes: 327 Q8_0/F32/F32 and 192
F32/F32/F32. No third matrix-input type appeared. The exact image probe binary
(SHA-256 `a2980263d888d21835b3644032311deabe4124d678d2042041b7c8794dd72a53`)
used the frozen GGML CUDA library (SHA-256
`1e4f954340ea2c30cb60deb49f4ccda6780993b2e09cb81bc8105aa053c925ef`).
The final quality campaign and custom GGUF remained at SHA-256
`0d7e79b83440b8b643ea89a06f65740e90aee4e904bece87c97bb152724d9a6d`
and `f854afae5a2eb2c388ff2dd85804a4327b4525b45a09c1fdc9b5f5619c93142e`.

The three-node trial passed 26,542,080 F64-recomputed dots. The complete
capture then passed **192/192 F32 nodes**, including 12 batched nodes, with
1,897,738,953 recomputed dots. The worst tensor relative L2 was
`2.5515898179008188e-06`, under the unchanged `2e-5` threshold. All 141
named F32 GGUF matrix operands matched their GGUF bytes; the other 51 matrix
operands are generated or unnamed views, checked against their captured
values rather than a named stored weight. The F32 report is
`build/q8-integrated-arithmetic-20261009/f32-full-verification.json` (SHA-256
`e69220a8841b7e0bf5dec2dc1af7d79befb2a65ffc13b6125d0d43ed5fec23f4`).
Its full operands and raw outputs are in the 17 GB
`build/q8-integrated-arithmetic-20261009/f32-full/` directory. These ignored
evidence files are available only in the validation workspace.

A separate read-only reconciliation matched every F32 and Q8 record to graph
order, source type, name, dimensions and batch size, rehashed 2,211 operand
and output files, and rechecked the frozen binary, CUDA library, campaign,
model, original output, final quality report and prior Q8 reconciliation.
Together the two checks cover **all 519 matrix nodes** and 2,059,184,009
same-operand dots in this one text-image inference. The combined report is
`build/q8-integrated-arithmetic-20261009/f32-reconciled-v2.json` (SHA-256
`266941b19ab1322f376acccccfcef92165f3875097c73639c61cfdd2b58fced7`).
The graph snapshot was independently produced by the existing diagnostic
profiler; the reconciliation checks graph correspondence, not a second
capture of F32 operands.

The F32 callback-run output differs from the preexisting unobserved export
only in `query_scores`, by at most `1.1900000007614153e-07`; masks, boxes and
ranked queries are identical. A fresh run with no preload exactly matched the
unobserved payload. A callback-only run with zero tensor reads produced the
same score change as the three-node and full captures. This isolates the
change to callback synchronization rather than saved-tensor reads. It does
not make the observed output byte-identical to the unobserved export.
The no-preload control and zero-read callback outputs are retained under
`build/q8-integrated-arithmetic-20261009/f32-control/` and `f32-observed-zero/`
(SHA-256 `94869a544637374f11a9901b2997fc4e57538290c886684db4286dee92e2bc13`
and `4035f5c0bd1aed83dd441dba48840904af1ee1f6e2a23bad2778710541a58d30`
for their result JSON files). The three-node trial dumps and superseded draft
reconciliation were removed after the full verification.

The Q8 and F32 matrix components pass for this one case. `FLASH_ATTN_EXT`,
normalization, convolution, elementwise and other non-matrix graph operations
have not received equivalent independent checks. Formal whole-recipe
arithmetic, exclusive-GPU performance and deployment remain `NOT_RUN`.
The RTX 4090 still hosts the remote-desktop compute process, so the formal
performance runner correctly refuses that shared session.

Validation: the sidecar compiled with `-Wall -Wextra -Werror`; the read-only
reconciler passed its full rehash; `tools/test_tools.py` passed 170 tests in the
isolated reference environment; the documentation checker passed 95 documents;
and `git diff --check` passed. Unneeded trial dumps and bytecode were cleaned.
