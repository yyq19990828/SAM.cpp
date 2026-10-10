# Corrected-library changed-prompt full graph arithmetic

Created: 2026-10-09 16:33:01, Asia/Shanghai.

## Scope

Extend the corrected-library one-image active-graph audit to the cached
image's first **changed prompt** in the already predeclared seven-call
`ImageSession`: development image `coco-000000001503`, `tv` then
`teddy bear`. The changed call executes session graph indices 4 and 5,
with the image encoding reused. Capture only those two graphs to stay within
the 58 GiB evidence-volume headroom while retaining every prior raw receipt.
Do not change the frozen campaign, production source, model, binary, CUDA
library, thresholds, quality or performance evidence.

## Approach and steps

1. Make fresh **diagnostic-only** copies of the existing F32 and nonmatrix
   capture sidecars, adding a strict graph-index filter for session indices
   4 and 5. The filter must call the original compute function without an
   observer for every other graph. Keep the session's seven input rows and
   uninstrumented expected outputs byte-bound to the prior session receipt.
2. Run one sidecar at a time. Independently verify every captured F32
   matrix, attention, normalization/unary, elementwise, spatial/reduction,
   pointwise/repeat, concat, gather/pool/F16-copy and CONT node under the
   unchanged arithmetic or exact-bit rules. Rehash retained raw evidence
   and compare all seven non-runtime outputs with the previously unobserved
   corrected-library session export. The already completed Q8 session
   verification covers Q8 matrix nodes on these calls.
3. Reconcile all compute and metadata nodes of changed-prompt graphs 4/5
   against the corrected session graph inventory, with no gaps or duplicate
   assignments. Keep whole-recipe arithmetic and deployment `NOT_RUN` until
   remaining predeclared prompts, input boundaries and full-path contracts
   have equally direct evidence.

## Verification

Use `rtk proxy`, `.venv-reference`, an inactive remote-desktop service and
exclusive GPU. Pin sidecar source/so, corrected binary/library/model, input
TSV, prior session/Q8 receipts and raw files in the new read-only results.
Monitor SSD space before each large group and retain successful raw evidence.
Run selected CUDA regression, tool suite, docs checker and `git diff --check`
after the cycle.

## Results

The nine filtered observers captured only session graphs 4 and 5. Independent
Q8, F32 and nonmatrix verifiers passed all 1,900 compute nodes: 296 Q8
matrices, 54 F32 matrices and 1,550 remaining operations. The compute
reconciliation is `PASS` (`rounded-changed-prompt-20261009/compute-reconciled.json`,
SHA-256 `c4f4c04461c3efc1905c00bb0d58b271d7f16a8fb4ee3bb6d136c92c70cf0c81`).
All retained raw operands and outputs live under the same external SSD
validation root and are not distributed with the repository.

A separate observer captured allocated addresses without an evaluation
callback. Its direct audit passed 1,059 metadata aliases (502 reshapes, 30
transposes, 212 permutes and 315 views), 653 leaf allocations and all buffer
bounds. Graphs 4 and 5 contain 847 and 2,112 nodes respectively. Its seven
exported non-runtime outputs match both fresh and earlier uninstrumented
sessions exactly. The metadata audit is `PASS`
(`rounded-changed-prompt-20261009/metadata/metadata-audit.json`, SHA-256
`bed53bc10e23d124f174fae706a330d60b38e3dbae9a2043103896758b0d27b7`).
The combined component-coverage receipt is `PASS`
(`rounded-changed-prompt-20261009/changed-graph-reconciled.json`, SHA-256
`71c11735594ccd2a0b4c31d0e5243e701157fe60a6772321291e62641212d9bd`).

One F32 compute observer perturbed only `query_scores` on the changed and
repeated prompts, by at most `1.1600000001353583e-9`; all other exported
fields matched. The independent uninstrumented repeat was exact for all
seven calls. This limits the claim to independently checked active-graph
components, not exact output parity under every compute observer. Whole-recipe
arithmetic and deployment remain `NOT_RUN`.
