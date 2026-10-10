# Frozen active-graph arithmetic on a second input

Created: 2026-10-09 08:36:53, Asia/Shanghai.

## Scope

Extend the complete active-graph arithmetic audit beyond the initial `tv`
development input to the predeclared `perf-negative` `vase` input. Both matrix
types already pass on all eight predeclared calibration/development inputs;
this second input's independent graph inventory contains the same 3,461
compute-node layout as the first. Keep the frozen model, image binary, CUDA
library, quality decision and numeric/bitwise gates unchanged. Use no final
evaluation or reserve inference for tuning. Exclusive-GPU performance remains
a separate gate.

## Approach and steps

1. Bind the negative-input dataset/recipe, full Q8/F32 matrix reconciliation,
   fresh graph inventory and original native output. Reuse the unchanged
   build-only observer libraries through fresh external evidence directories.
2. In graph order, independently check the remaining attention, normalization
   and activation, elementwise, spatial/reduction, pointwise/repeat, concat,
   gather/pooling/F16 copy, and contiguous-copy groups against captured
   operands. Verify metadata aliases, ranges and leaves separately. Reuse
   established F64 numerical and bitwise equality rules for each group.
3. Reconcile every group to the independent negative-input graph inventory,
   rehash retained raw/ledger evidence and check final inference behavior.
   Only then mark this second **input's** active-graph arithmetic PASS. Do not
   infer a campaign-wide arithmetic or deployment PASS from two inputs.

## Verification

Use the exact frozen binary and prior isolated reference environment, fresh
capture outputs, read-only verifiers and final graph reconciliation. Run the
relevant tool suite, docs checker and `git diff --check` after completed
cycles. Store required raw evidence on the separate mounted volume through
the Git-ignored `build/` path; clean only disposable intermediates.

## Results

The second input's independent frozen graph contained the same 3,461 CUDA
compute nodes, 2,615 metadata nodes and 1,155 leaves as the first, in four
graph partitions. The existing Q8/F32 matrix receipts covered all 519 matrix
nodes. Fresh observer captures and independent checks passed the remaining
87 attention, 317 normalization/activation, 1,509 elementwise, 76 spatial
and reduction, 242 pointwise/repeat, 132 concat, 15 gather/pool/F16-copy,
and 564 contiguous-copy nodes. The elementwise group had zero F32 bitwise
mismatches across 4,284,831,745 values; concat had zero across 392,507,904
values; CONT copied 8,156,709,344 bytes without a mismatch. The other groups
met their previously frozen numerical or exact-bit gates. Each group's
ledger, node order, operands, shapes and retained payload hashes matched
the independent graph inventory. All observer runs preserved the native
final masks, boxes, ranking and scores for this negative prompt after
excluding runtime measurements.

The metadata audit checked alias identity, shape/stride relationships,
buffer bounds and leaves. The aggregate receipt
`build/q8-integrated-arithmetic-20261009/negative-active-graph-reconciled.json`
has SHA-256 `48d5522b1fea65cf0c3b46768b6122576b2dcad3a8133c4d274959e3d9f94a6a`.
Its component receipts are `attention-negative-reconciled.json`
(`cc577c03b35e5fc2cf4e0a0b5a18668d0a5a1a9f096289936dacf84ec971c785`),
`norm-unary-negative-reconciled.json`
(`2e61de37148e1c81897d4015a5a6cb59a4d174bb2dc3b69db3c799fcfd3fffef`),
`elementwise-negative-reconciled.json`
(`78673bcd887228d5277239eb5746a2e212e1e76df634cd8e83364a5649e2c511`),
`spatial-negative-reconciled.json`
(`db6884693bd626cf55efca388d1b59d885eee85288992e9572c6f7cb22b5e1f5`),
`pointwise-negative-reconciled.json`
(`28a7e039765309cdef68e089d5fd95588080ba24c822fec61147746d425c41f8`),
`concat-negative-reconciled.json`
(`ac9215a7d15920928e17e0bca1191485763ad0306c0575228617dbbe12f2b528`),
`gather-pool-copy-negative-reconciled.json`
(`a258de7b8828d4243328b2826616c8d7c7f4ec4f491b980db8298435a9c158a7`),
`cont-negative-reconciled.json`
(`ae60f9f821127306d4965763b7a53d71c9858f032f7ac495a6c931aa8ebe1ce5`),
and `metadata-negative-audit.json`
(`c420946ca704b2980671e0df551c020cb5eb88b90c767659caddde9f486989a1`).
All paths are under `build/q8-integrated-arithmetic-20261009/` unless noted;
raw captures live under its Git-ignored `many-external/` link to `/mnt/SSD1`
and are available only in this validation workspace. Retain these immutable
receipts and raw captures for reproduction.

The single-input active-graph arithmetic status is `PASS` for both the first
`tv` development case and this independent `vase` negative case. Six other
predeclared inputs currently have complete matrix-component checks, not
complete active-graph checks. Campaign-wide arithmetic, exclusive-GPU
performance and deployment remain `NOT_RUN`; no acceptance threshold or
frozen model artifact changed.

The completed validation cycle passed the isolated Python tool suite
(170/170), the bilingual/local-link documentation check (109 documents)
and `git diff --check`. The exact frozen image binary, CUDA library,
campaign and model hashes still match their previously bound identities.
Raw captures and receipts remain necessary validation evidence on the
separate volume; no disposable intermediate was retained.
