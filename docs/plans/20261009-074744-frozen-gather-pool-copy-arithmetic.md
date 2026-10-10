# Frozen gather, pooling and F16 copy arithmetic

Created: 2026-10-09 07:47:44, Asia/Shanghai.

## Scope

Continue the exact frozen CUDA Q8 single-image arithmetic audit after all
132 active concatenations passed. Target the one F32 `GET_ROWS` lookup,
one F32 `POOL_1D` average and 13 F32-to-F16 `CPY` nodes. Preserve the
frozen model, binary, CUDA library, campaign, quality split and arithmetic
limits. This is diagnostic component evidence, not a whole-recipe or
performance gate.

## Approach and steps

1. Build an isolated observer for the actual F32, I32 and F16 tensor types.
   Capture logical operands before each selected node and its output after;
   handle `CPY`'s self-referential destination without reading its
   pre-operation contents. Record pool parameters, descriptors and graph
   order. Trial a small prefix before full capture.
2. Recompute `GET_ROWS` indexing and F32-to-F16 nearest-even conversion
   independently, requiring exact output bits. Recompute average pooling
   from captured F32 operands in F64 and apply the unchanged 2e-5
   relative-L2/1e-6 zero-norm rule. Reject invalid indices, shapes,
   parameters and unsupported pool modes.
3. Reconcile graph identity, complete node coverage, hashes, previous
   component receipts and final inference payload. Preserve failures for
   diagnosis; do not promote whole-recipe arithmetic until all remaining
   compute and metadata operations are accounted for.

## Verification

Compile C++17 with warnings as errors; run independent reference and
reconciliation, isolated Python tool tests, documentation checker and
`git diff --check`. Remove superseded successful trial data.

## Results

The C++17 observer compiled with warnings as errors. Its three-node
`GET_ROWS`/`CPY` trial passed 2,769,920 exact comparisons. The complete
capture of all 15 targeted nodes on frozen `coco-000000001503-p0`/`tv`
passed 9,931,440 checked values. The `GET_ROWS` and 13 F32-to-F16 `CPY`
nodes matched 9,931,184 output bits exactly, including expected F16
overflow to infinity for large finite source values. The one average
`POOL_1D` node passed 256 F64-recomputed values at relative L2
6.315e-8 under the unchanged 2e-5 limit. Its 149 mismatches against
F64-rounded F32 output are reported separately from the numerical pass.

The reconciliation matched all 15 records and 31 retained operand/output
files to the frozen graph's node order, source self-reference, types,
shapes and strides, then rehashed each file. Callback synchronization
changed `query_scores` by at most 1.19e-7; final masks, boxes and ranking
matched the unobserved export. Quality remains `PASS`; whole-recipe
arithmetic, exclusive-GPU performance and deployment remain `NOT_RUN`.

The following Git-ignored receipt paths are available only in this
workspace under `build/q8-integrated-arithmetic-20261009/`:

- `gather-pool-copy-full-verification.json`: SHA-256
  `0c0581d4c4f689b64d45f757dee68a4f1fa1d1d561cd95382b99f3719b40973c`.
- `gather-pool-copy-reconciled.json`: SHA-256
  `afdc0b203799a0db6dd8dc7a9f6cb017435a24d83f054924a82c85a14ba13fc1`.
- `gather-pool-copy-full/nodes.tsv`: SHA-256
  `427b123100a412b47ae3388655cd14faa0bf7ab6c2198c80d6279d5745942a9a`.
- `capture_gather_pool_copy.cpp` and `libcapture_gather_pool_copy.so`:
  SHA-256 `17b1c7106319be8e4615fe144887efce465d54f6acb93d8089f523b5ea5a3441`
  and `7501a4699df104ba33ca8d8331efc4d80a409d0beb5c658e4744345aa648dc53`.
- `verify_gather_pool_copy.py`: SHA-256
  `f6f5c26524fc06642027a9f25083cfb3671f2b89dbd0b6720eb37c646f8b3159`.

The exact campaign, binary, CUDA library, model and previous component
receipts are bound by the reconciliation. The GPU still has a non-campaign
desktop client, so the formal exclusive-GPU performance gate cannot run
in this session.

The isolated Python tool suite passed 170 tests; the documentation checker
passed 102 documents; `git diff --check` passed. Successful prefix-trial
duplicates were removed after full reconciliation.
