# Corrected-library repeated-prompt cache-hit audit

Created: 2026-10-09 16:51:31, Asia/Shanghai.

## Scope

Determine whether the first changed prompt (`small-alternate`) and its
immediately repeated identical prompt (`small-repeat`) execute equivalent
graphs in the same seven-call corrected-library Q8 session. The initial
graph-index hypothesis was that graphs 4/5 and 6/7 belonged to these calls.
Reuse the completed graph-4/5 independent arithmetic and direct-alias
receipts only if corresponding runtime node operands and results prove
identical. The model, library, executable, input cases and prior evidence
stay fixed.

## Approach and steps

1. Add a diagnostic-only scheduler observer that records SHA-256 of every
   logical row of every compute node's operands before execution and output
   after execution on graphs 4/5/6/7. Bind node order, operation, type,
   shape and strides; hash no padding bytes. Keep only compact hashes, not
   duplicate tensor dumps.
2. Check the graph-to-call mapping from cumulative native CUDA node counts
   before comparing node digests. Independently compare native session
   outputs, backend placement, model/binary/library/input identities and
   the already retained direct arithmetic/alias receipts.
3. If the repeat executes no graph, issue a narrowly scoped zero-graph
   cache-hit receipt. Otherwise, compare corresponding graph digests before
   any arithmetic transfer. Preserve whole-recipe arithmetic and deployment
   as `NOT_RUN` pending the remaining input and prompt boundaries.

## Verification

Run on the exclusive CUDA device while the remote-desktop service remains
inactive. Record the observer's source, shared library, captured hashes,
session exports and reconciliation in external validation storage. Run docs
and whitespace checks after writing results. Retain immutable evidence.

## Results

The graph-index hypothesis was false: cumulative CUDA nodes are 5,361 after
`small-alternate` and still 5,361 after `small-repeat` in the original,
fresh uninstrumented and hash-observed runs. The repeated call executes
**zero** CUDA, CPU, Metal and BLAS compute nodes and no scheduler graph.
Its entire non-runtime export matches the preceding changed
prompt in each run. Graphs 6/7 belong to the subsequent whitespace prompt;
511/511 and 1,238/1,389 compute-node hash rows differ from graphs 4/5, respectively,
as expected
for a different token input. No graph-4/5-to-6/7 arithmetic transfer is
claimed.

The corrected-library zero-graph cache-hit audit is `PASS`
(`rounded-repeat-prompt-20261009/repeat-cache-hit-audit.json`, SHA-256
`323bff689c2fda3ddbc815450d189572300cb28ebbbaa5bf30592d5324fa0e2b`).
The SHA-256 node manifest, observer source/library, seven observed outputs,
original and fresh native outputs, model, binary, CUDA library and preceding
graph-component receipts are hash-bound there. External validation storage
holds the raw manifest and outputs; they are not distributed with the
repository. The hash observer changed only `query_scores` on the changed and
repeated prompts, by at most `1.1600000001353583e-9`; it preserved equality
*between those two calls*. Whole-recipe arithmetic and deployment remain
`NOT_RUN`.
