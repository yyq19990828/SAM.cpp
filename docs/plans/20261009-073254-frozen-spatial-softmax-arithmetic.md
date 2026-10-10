# Frozen spatial, softmax and group-normalization arithmetic

Created: 2026-10-09 07:32:54, Asia/Shanghai.

## Scope

Continue the one-case exact frozen CUDA Q8 recipe audit after all matrix,
attention, row-normalization/unary and ADD/MUL/SUB nodes passed. Target every
remaining `IM2COL` (10), `WIN_PART` (28), `WIN_UNPART` (28), `UPSCALE` (2),
`GROUP_NORM` (2) and `SOFT_MAX` (6) node. Preserve all frozen campaign,
binary, GGUF, CUDA library, quality and arithmetic limits. The GPU remains
shared, so this is numerical evidence rather than performance evidence.

## Approach and steps

1. Inventory source/output shapes, strides, types and op parameters. Capture
   the exact logical F32 operands before each selected node and its raw output
   afterward using an isolated preload sidecar on the frozen binary. Request
   all-node callback synchronization because some operators may reuse input
   storage. Fail closed on unsupported modes, masks, interpolation or storage.
2. Trial a small prefix, then capture every target node with a size preflight
   against the remaining disk space. Recompute layout-only `IM2COL`, windows
   and nearest upscale by independent index formulas, requiring exact F32
   bytes. Recompute group normalization and masked softmax in F64 and apply
   the unchanged 2e-5 tensor-relative-L2/1e-6 zero-norm rule. Record exact
   mismatch counts separately for all operators.
3. Reconcile graph order, source/output identity, parameters and all retained
   hashes with the independent graph snapshot, earlier component receipts,
   exact binary/library, campaign, model and final inference payload. Retain
   failures for diagnosis and do not promote whole-recipe arithmetic until
   all required operators and input scope have evidence.

## Verification

Compile C++17 with warnings as errors; run the isolated tool suite,
documentation checker and whitespace check. Remove superseded successful
trial files while preserving complete evidence and usable builds/models.

## Results

The corrected sidecar compiled with C++17 warnings as errors. A three-node
trial passed exact-byte references before the full capture. The first two
diagnostic trials stopped at `GROUP_NORM`: the sidecar treated the second
`int32_t` parameter slot as a byte offset and falsely read epsilon as zero.
A separately linked parameter probe confirmed the frozen library stored
`8` groups and F32 epsilon `9.99999975e-06`. Correcting only the sidecar
read also corrected the analogous softmax `max_bias` read. No frozen binary,
library, GGUF, campaign or threshold changed.

All 76 target nodes passed the full same-operand check on the frozen
`coco-000000001503-p0`/`tv` image. The 10 `IM2COL`, 28 `WIN_PART`, 28
`WIN_UNPART` and two nearest-neighbor `UPSCALE` nodes yielded 874,955,520
exact F32 output values with zero bitwise mismatches. The two `GROUP_NORM`
and six masked `SOFT_MAX` nodes yielded 76,557,312 independently F64-recomputed
values under the unchanged 2e-5 relative-L2 limit; worst relative L2 was
2.389e-6. For these reductions, 63,971,888 F32 bit patterns differ from the
F64-rounded reference, so passing is a numerical-limit result, not exact-byte
parity. In total, 951,512,832 values were checked. All 158 captured operand
and output files were hashed and matched to node order, types, shapes and
strides in the frozen graph inventory. The callback changed `query_scores`
by at most 1.19e-7; final masks, boxes and ranking matched the unobserved
export. Quality remains `PASS`; whole-recipe arithmetic, exclusive-GPU
performance and deployment remain `NOT_RUN`.

The retained `build/q8-integrated-arithmetic-20261009/` receipt paths below
are Git-ignored local evidence, available only in this workspace:

- `spatial-full-verification.json`: SHA-256
  `41f5fcfa249d3ad7b0aa8b3300d7d3bb0e80f7370f8fef80f9951870d63b1c2f`.
- `spatial-reconciled.json`: SHA-256
  `2a400c63ee1e732e8af7a93755c9d3ab4437e7636b677dfae0675c9b1d68b501`.
- `spatial-full/nodes.tsv`: SHA-256
  `5135ed8803ada9ebceb28bf9707047290d61fdf9be1998e28b8d1aae59b39832`.
- `capture_spatial.cpp` and `libcapture_spatial.so`: SHA-256
  `b437c17b022a086b68e777af474abe0a42cd8ba0a1d311becf0b1228b2f04b45`
  and `ca76241f8dc4bb87ea007c5e25fec9fa24bb62e2879bb25aa310532a3ea2121b`.
- `verify_spatial.py`: SHA-256
  `04597f7f0a1309c4cfa9fe9d281451d54dd10afcfd080caa700580f22943f91f`.

The frozen campaign SHA-256 remains
`0d7e79b83440b8b643ea89a06f65740e90aee4e904bece87c97bb152724d9a6d`;
the exact frozen probe and GGML CUDA library hashes are bound in the
reconciliation receipt. The CUDA host still has a non-campaign desktop GPU
client, so the formal exclusive-GPU performance gate cannot run in this
session.

The isolated Python tool suite passed 170 tests; the documentation checker
passed 99 documents; `git diff --check` passed. The successful prefix trial's
duplicate tensor dumps and output were removed after full reconciliation.
