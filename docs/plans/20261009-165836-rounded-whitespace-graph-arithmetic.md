# Corrected-library whitespace-prompt active-graph arithmetic

Created: 2026-10-09 16:58:36, Asia/Shanghai.

## Scope

Independently verify the corrected-library cached-image whitespace-prompt
path (`small-whitespace`, session graphs 6/7), which follows the exact
repeat cache hit. Its graph structure is the same class as the changed
prompt, but its node inputs differ. Keep the frozen model, campaign,
binary, library and acceptance thresholds unchanged. Capture only graphs
6/7 and retain all existing evidence.

## Approach and steps

1. Make new diagnostic copies of the nine reviewed node observers with a
   strict graph-6/7 filter. Run the predeclared seven-call session once per
   observer. Record source, shared-library, raw-payload, graph-inventory
   and unobserved-output identities.
2. Reuse the unchanged independent F32 and nonmatrix node oracles against
   the fresh raw operands. Reconcile all Q8/F32/nonmatrix compute nodes
   without duplicate or missing assignments. Audit all metadata aliases
   and leaf bounds directly in a separate lightweight graph observer.
3. Issue a scoped active-graph component receipt only after exact topology
   and raw identity checks. Retain whole-recipe arithmetic and deployment
   `NOT_RUN` until the other prompt/image boundaries are covered.

## Verification

Use the corrected verified CUDA build, inactive remote desktop and
exclusive GPU. Preserve immutable receipts and model files. Run selected
CUDA CTest, the isolated Python tool suite if source changes, docs checker
and `git diff --check`. Keep external raw evidence on the SSD; clean only
unneeded intermediate files after the cycle.

## Results

The nine filtered observers captured only session graphs 6 and 7. All seven
exports from each observer match the fresh uninstrumented corrected-library
session in every non-runtime field. The earlier Q8 session receipt covers
296 Q8 matrix nodes on these graphs. Fresh independent checks passed 1,550
nonmatrix nodes and 53 of 54 F32 matrix nodes. The direct allocation audit
passed 1,059 metadata aliases and 653 leaves. Graphs 6/7 have 847/2,112
nodes. No CPU, Metal or BLAS compute nodes appeared.

One F32 node **fails** the unchanged `2e-5` tensor-relative-L2 rule:
`ddec.presence_token_head.layers.2.weight`, graph 7, F32 ordinal 49,
M=1, N=1, K=256. The native output is `-0.0043205022811889648`; an
independent exact-product Decimal sum gives
`-0.004320603702575885607384265085784136317670345306396484375`.
Absolute error is `1.0142138692076363e-7`, relative error
`2.3473892516524385e-5`, and the sum-of-absolute-products cancellation
condition is approximately 1,263. `math.fsum` agrees with the Decimal
reference. The `1e-6` absolute fallback in this frozen gate applies only
to zero-norm references, so it does not turn this result into a pass.
The failure is not an observer-output mismatch or an altered GGUF weight.

The retained F32 node receipt is `FAIL`
(`rounded-whitespace-prompt-20261009/f32/verification.json`, SHA-256
`663562dec7b81a08e170ff85227154b6a4d0e678b71956341dfc31d4c1d49958`);
the separate high-precision reproduction is `FAIL`
(`rounded-whitespace-prompt-20261009/f32/singleton-audit.json`, SHA-256
`8c08e21467eeb977b9a07b516be3a4e2cddb64020ef9a8aaa1d59b9ce99efcb8`).
The compute reconciliation checks all 1,900 nodes, recording 1,899 passes
and one failure (`rounded-whitespace-prompt-20261009/compute-reconciled.json`,
SHA-256 `c195adde6b7c463565e0ec700d6c2fb0f61701e651da8615c86f6877ef2719cc`).
The separate direct metadata audit is `PASS`
(`rounded-whitespace-prompt-20261009/metadata/metadata-audit.json`, SHA-256
`5b36185b20828bd10721a3229ed19ab1adda29b505c23de38e946da083a96b79`).
The combined active-graph component status is **`FAIL`**, not partial PASS
(`rounded-whitespace-prompt-20261009/whitespace-graph-reconciled.json`,
SHA-256 `ad094309fb6e8b7ae6d62d4a2f963e7ca986939ec1a801bbdb101df65355bc64`).
All raw operands, observers and receipts remain in external SSD validation
storage and are not distributed with the repository. The corrected Q8
candidate's whole-recipe arithmetic and deployment remain `NOT_RUN`; do
not relax a frozen threshold after observing this failure. A new numerical
implementation or separately predeclared gate revision would need its own
candidate identity and complete validation.
