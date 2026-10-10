# Frozen concatenation arithmetic

Created: 2026-10-09 07:45:12, Asia/Shanghai.

## Scope

Continue the exact frozen CUDA Q8 single-image arithmetic audit after
pointwise and repeat nodes passed. Target all 132 active F32 `CONCAT` nodes,
preserving the frozen campaign, model, binary, CUDA library, source image,
quality split and arithmetic gates. GPU sharing still excludes formal
performance measurements.

## Approach and steps

1. Inspect `CONCAT` dimensions and both source descriptors in the frozen
   graph. Build a separate preload observer that captures logical F32
   sources before each node and its output afterward, with all-node
   synchronization and a disk-space preflight. Trial a small prefix.
2. Compute NumPy concatenation on the captured operands along the exact
   GGML axis; require F32 bitwise parity for every output value. Reject
   unsupported types, axes, strides or input shapes.
3. Reconcile node order, parameters, descriptors and every retained hash
   with the graph inventory and prior component receipts; compare final
   inference fields with the unobserved frozen export. Do not infer
   whole-recipe or deployment PASS from this component.

## Verification

Compile C++17 with warnings as errors. Run the reference and reconciler,
isolated Python tool suite, documentation checker and `git diff --check`.
Remove duplicate successful trial data while retaining complete evidence.

## Results

The C++17 sidecar compiled with warnings as errors. A three-node prefix
trial passed 15,925,248 exact F32 comparisons. The full capture of all
132 active `CONCAT` nodes on frozen `coco-000000001503-p0`/`tv` passed
392,507,904 bitwise comparisons with zero mismatches. All 396 input and
output files were rehashed and matched to node order, concat axis, types,
shapes and strides in the frozen graph inventory. Callback synchronization
changed `query_scores` by at most 1.19e-7; masks, boxes and ranking matched
the unobserved export. Quality remains `PASS`; whole-recipe arithmetic,
exclusive-GPU performance and deployment remain `NOT_RUN`.

The following Git-ignored receipt paths are available only in this workspace
under `build/q8-integrated-arithmetic-20261009/`:

- `concat-full-verification.json`: SHA-256
  `435c69b74c2ea9cee08b16db4923a805e764f30dfe2c7e96f80f244521065ddb`.
- `concat-reconciled.json`: SHA-256
  `4b2e9ebf3fc043b131ee9a2f1750c314aa812830c8fde8328939d988e20e5410`.
- `concat-full/nodes.tsv`: SHA-256
  `9bba0c660c8c5f5069d38de5f316609e5f552f98017e44edb7f1224004f5928e`.
- `capture_concat.cpp` and `libcapture_concat.so`: SHA-256
  `8b0375c12f5629191a274297c19e5f2acaf15f272cb8712f821901aa6fb1fb0c`
  and `cf2eb195b6e5eaaa3c9a212e9917725b14f4c6653e6a6991672a098606dd79d0`.
- `verify_concat.py`: SHA-256
  `0f854838843c403168b5c37c673c0f9f2bf2d896eee5bf18f9d01d3cc7f0d2c0`.

The exact campaign, binary, CUDA library, model and previous components are
bound by the reconciliation. The GPU remains shared with a non-campaign
desktop client, so formal exclusive-GPU performance cannot run in this
session.

The isolated Python tool suite passed 170 tests; the documentation checker
passed 101 documents; `git diff --check` passed. Successful prefix-trial
duplicates were removed after full reconciliation.
