# Frozen pointwise and repeat arithmetic

Created: 2026-10-09 07:41:21, Asia/Shanghai.

## Scope

Continue the exact frozen CUDA Q8 single-image arithmetic audit after 76
spatial, normalization and softmax nodes passed. Target all 24 `SIN`, 24
`COS`, 24 `LOG`, 67 `SCALE`, 19 `CLAMP`, 72 `REPEAT` and 12 `DUP` nodes
(242 active nodes). Preserve the campaign, binary, CUDA library, diagnostic
GGUF, data split and all numerical limits. This component check is not a
whole-recipe or performance result.

## Approach and steps

1. Inspect operation parameters, input/output types, shapes and strides in
   the exact graph. Build an isolated observer that captures logical operands
   before nodes and outputs afterward. Synchronize each graph node to ensure
   any aliased input is captured before mutation. Bound the capture by a disk
   preflight and trial a small prefix.
2. Recompute trigonometric, logarithmic and scalar arithmetic in F64 using
   exact captured F32 operands, applying the existing 2e-5 relative-L2 and
   1e-6 zero-norm limits. Check `CLAMP`, `REPEAT` and `DUP` against independent
   F32 indexing formulas with exact byte parity. Reject unsupported op
   parameter modes and nonfinite reference mismatches.
3. Reconcile the complete node list and tensor descriptors with the frozen
   graph inventory, hash every retained payload and bind the observer,
   verifier and output to previous component evidence. Compare the final
   inference result with the unobserved frozen export.

## Verification

Compile C++17 with warnings as errors. Run the independent reference,
reconciliation, isolated Python tool suite, documentation checker and
`git diff --check`. Remove duplicate trial files after full validation;
preserve required immutable evidence and current usable models/builds.

## Results

The C++17 sidecar compiled with warnings as errors and its three-node
`REPEAT` trial matched 37,158,912 F32 values bit for bit. The full capture
covered all 242 targeted nodes on frozen
`coco-000000001503-p0`/`tv`. All 72 `REPEAT`, 19 `CLAMP` and 12 `DUP`
nodes passed 314,261,432 exact-byte comparisons. The 24 `SIN`, 24 `COS`,
24 `LOG` and 67 `SCALE` nodes passed 2,359,400 F64-recomputed values under
the unchanged 2e-5 relative-L2 limit. Worst relative L2 was 4.781e-7.
The transcendental nodes had 649,834 F32 bitwise mismatches to F64-rounded
results; `SCALE` had none. In total, 316,620,832 values were checked.

The reconciliation matched all 242 records and 484 retained F32 payloads
to the frozen graph's node order, types, shapes and strides and rehashed
each file. `SCALE` parameters include nonzero bias in the actual graph;
the oracle used the captured scale and bias. Final masks, boxes and query
ranking matched the unobserved frozen export; synchronization changed
`query_scores` by at most 1.19e-7. The image quality result remains `PASS`.
Whole-recipe arithmetic, exclusive-GPU performance and deployment remain
`NOT_RUN`.

The following Git-ignored receipt paths are available only in this
workspace under `build/q8-integrated-arithmetic-20261009/`:

- `pointwise-full-verification.json`: SHA-256
  `9a0aa2234f09b24da8c7ec5d1e135d884f53d99eca46f34a06ac16f3c0b2519f`.
- `pointwise-reconciled.json`: SHA-256
  `deca42426e73c64ea1487bb482638ba99e8fd9b3fd050f43b306981901b983a6`.
- `pointwise-full/nodes.tsv`: SHA-256
  `2aca5fa1746d23097d8c8d00bd8da76621de4b58ed50a65e0a7c7afd63ed87bc`.
- `capture_pointwise.cpp` and `libcapture_pointwise.so`: SHA-256
  `3727bf8b46277d32af5cc68b1f95bc4c25562a4e9b2f5e005c8fa56b0a73bbef`
  and `4e40a53c32bd4ffc4f853a904b6d48aca154c399eff1e73e9dd5d74c7223a485`.
- `verify_pointwise.py`: SHA-256
  `f028c0976d4d4d76d9f53134f7d178f5d5413606c90a0cb20febe18a163ed9e9`.

The frozen campaign, binary, CUDA library, diagnostic model and prior
component receipts are bound by the reconciliation. The GPU remains shared
with a non-campaign desktop client, so the exclusive performance gate
cannot run in this session.

The isolated Python tool suite passed 170 tests; the documentation checker
passed 100 documents; `git diff --check` passed. The successful prefix
trial's duplicate tensor dumps and output were removed after full
reconciliation.
