# Corrected-library nonmatrix arithmetic on a full image graph

Created: 2026-10-09 15:50:11, Asia/Shanghai.

## Scope

Directly audit the corrected GGML CUDA library's nonmatrix graph operations
on the already predeclared `perf-source-small` development image. The old
full-graph receipts came from a different GGML revision; exact graph
metadata alone cannot transfer their numeric verdicts. Preserve the frozen
campaign, model, quality and performance artifacts, and never use final
evaluation or unopened reserve inputs. The first target is one fully
checked current-library graph; broader image/session coverage remains a
separate scope decision.

## Approach and steps

1. Recompile existing diagnostic capture sidecars against the corrected
   prepared GGML headers, in fresh ignored directories. Bind each sidecar,
   native binary, model, exact case TSV, image and prior uninstrumented
   output. Check the SSD before large captures; run groups sequentially.
2. Independently verify all active nodes in attention, normalization/unary,
   elementwise, spatial/reduction, pointwise/repeat, concat, gather/pool/F16
   copy and contiguous-copy groups under the original numerical/exact-bit
   rules. Rehash raw captured operands/outputs and compare observed final
   payloads. Do not loosen thresholds or skip failed nodes.
3. Reconcile all groups against the corrected-library 3,461-node graph
   inventory, the separately verified 327 Q8 and 192 F32 matrix nodes,
   metadata/leaf relationships, backend placement and source identities.
   Promote only a precisely scoped current-library graph arithmetic status
   if every compute and metadata node has direct evidence.
4. Evaluate a replayable bounded-storage procedure for any additional
   image/session cases. Keep whole-recipe arithmetic and deployment
   `NOT_RUN` until their full prerequisites are independently covered.

## Verification

Run via `rtk proxy` in the isolated reference environment with remote
desktop inactive and no competing GPU compute process. Retain successful
raw evidence and immutable receipts on the SSD; clean disposable trial
intermediates only. Run relevant tool tests, documentation/link check and
`git diff --check` after the cycle.

## Results

The corrected-library `perf-source-small` attention capture completed all
87 active `FLASH_ATTN_EXT` nodes. Its non-runtime final output matched the
previous uninstrumented corrected-library export. Independent F64 attention
reference checks passed 4,359,948,024 score pairs and 188,527,104 output
values; worst node relative L2 was `7.158758099e-7`. A separate audit
rehashed every raw attention operand/output, matched each node's graph
position, Q/K/V/mask/output shapes and CUDA placement against the corrected
inventory, and passed. The ignored per-node verification SHA-256 is
`62ce699e1a1e7f3d4831670a973be387fcedbfba00cb5c9fc8f8e2f93b5682c4`;
the ignored `attention/reconciled.json` SHA-256 is
`2bd744e4db0211c7df06a03f8446915c579d2138c6d4c310d9cd279bea899ab3`.
About 2.8 GiB of raw attention capture is retained. At that stage the
remaining nonmatrix groups were pending; this component result alone did
not promote whole-recipe arithmetic.

The corrected-library normalization/unary capture then passed all **317**
active `NORM`/`UNARY` nodes, checking 1,311,625,440 values. Its worst node
relative L2 was `5.107398020e-7`. Independent reconciliation matched
all node shapes, output strides and CUDA placement to the current graph
inventory and rehashed the raw payloads. The ignored per-node report
SHA-256 is
`98cf58121d98c25a6e10a9d5c904573368dc21331d7b74826518a67719fad613`;
the ignored `norm-unary/reconciled.json` SHA-256 is
`8bfb30772ae531980fafe2f8aad749de2257812c7206066d47d64daebc11d4cf`.
About 9.8 GiB of raw capture is retained. The result remains a component
PASS while other nonmatrix groups are pending.

The corrected-library elementwise capture completed **1,509** ADD/MUL/SUB
nodes, using about 29 GiB of raw source/output tensors. The independent
oracle checked 4,284,831,745 values under the unchanged numerical gate;
worst node relative L2 was `5.201666634e-8`. An independent read-only
reconciliation matched every node's operation, source/output shapes,
output stride and CUDA placement to the current graph inventory and
rehashed 3,357 raw tensor files. The ignored per-node report SHA-256 is
`3e246f70e4d3bdd7550e1d10436d5eb9ab99defc9e1b47ec3e03252b271eb33c`;
the ignored `elementwise/reconciled.json` SHA-256 is
`96a380de9c7da9bd08fcff33ab9c57b94d56d3ef44fd1b07d7436501b8390117`.
At that stage the SSD had about 68 GiB free, with spatial, pointwise,
concat, gather/pool and CONT groups still pending.

The corrected-library spatial/reduction capture passed all **76** active
IM2COL, window-partition/unpartition, upscale, group-norm and softmax
nodes. The independent verifier checked 951,512,832 values, applying the
original exact-bit rule to lossless operations and the unchanged numerical
rule to normalization/softmax. The separate reconciliation matched current
graph positions, source/output shapes and CUDA placement and rehashed all
raw operands/outputs. The ignored per-node report SHA-256 is
`0486910058da3c3109867039fefa50246379beddbf494dd2ad05e7ac6d0d2695`;
the ignored `spatial/reconciled.json` SHA-256 is
`221ec73153d5c11ca6a81e3d5a1fff61fb2c5cdb5f9098e0dac6d87fa5058cb8`.
Pointwise/repeat, concat, gather/pool and CONT were the final pending groups
after the spatial audit.

The remaining groups passed on the same corrected-library development input:
**242** pointwise/repeat nodes (316,620,832 checked values), **132** concat
nodes (392,507,904 exact-bit values), and **15** gather/pool/F16-copy nodes
(9,931,440 values). Independent read-only reconciliation matched each group
to the current graph inventory and rehashed its raw evidence. Their ignored
verification/reconciliation SHA-256 pairs are, respectively:

- pointwise: `5fd1897231491327ff5ba14f2cb20f804c5c6aff4b9b443a1f780c7d59ff5f61` / `fdd93a343f392505b91166a5920e45f815e649b564a1707cb9eccc5ad5fdeaf7`;
- concat: `01d1a28229f9fe786c9b76657a3e6e367f636c30b1ba4151f0b6d53b5d97d88f` / `231197dc8f990b848da556119fd47648f1acc3c272321651ce4c1b1d5c041572`;
- gather/pool/F16 copy: `65438db724e8ab1189a33ba8bc91de835d83f1b2c6a0b27a3a2ec1629cc0533c` / `ff33dcf8f6e10d70b441ac010dc4666b81b16a1833329ea6ae902aa39c733714`.

All **564** current-library CONT nodes compared logical source and output
bytes in the CUDA callback: 8,156,709,344 bytes checked, zero mismatches;
three nodes additionally retained raw source/output trials. The verifier
bound the exact graph positions, tensor types/shapes/strides, final output,
binary, model and CUDA library; an independent audit rehashed all trial and
identity files. The ignored verification/reconciliation SHA-256 pair is
`36dd799df1e3f94c2bdb4b287869cfbef13800c256d2c16a0a1fed0d205fd742` /
`32593ad9c3ee6289dbf5b75b88fc3a407235d0032422142b5ca199c4e1cadca8`.

A fresh current-library `sam_profile_graph` inventory of the same byte-identical
development image and model passed the metadata alias/shape/span audit for
**2,615** metadata nodes and **1,155** leaf tensors. Its 6,076 graph nodes
matched the corrected-library session inventory exactly; the metadata audit
SHA-256 is `6e85bd6ffe87205e4fae1763bd5f9ba85008c8987110878ada99ee64fbb80c18`.
Finally, a disjoint cross-group reconciliation assigned every one of the
**3,461** CUDA compute nodes to the Q8, F32 or nonmatrix receipts without gaps
or duplicates. The one-case aggregate is a **PASS** for that active graph;
its ignored `rounded-nonmatrix/one-case-active-graph-reconciled.json` SHA-256
is `f5c37f6b3f69a72b3764186c7a95e276f13f21ddd98103fcd0f1cd05df7daccd`.
All ignored evidence above is available in this validation workspace under
`/mnt/SSD1/samcpp-validation/q8-mmq-rounded-candidate-20261009/` and is not
distributed with a clean checkout. The original campaign and production
sources were not edited during these captures. Multi-input whole-recipe
arithmetic and deployment remain `NOT_RUN`.

The documentation/link check passed all 120 repository Markdown documents;
`git diff --check` passed. The remote-desktop user service remains inactive
and no competing GPU compute process is present. About 58 GiB remains free
on the SSD after retaining the current-library raw captures. A second
whole-graph raw capture of comparable size does not fit there alongside the
immutable existing evidence, so additional input coverage requires a
predeclared bounded-storage capture/verification procedure or more evidence
storage. No claim is made for multi-input whole-recipe arithmetic.
