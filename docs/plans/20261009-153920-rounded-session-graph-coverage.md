# Corrected-library image-session graph coverage

Created: 2026-10-09 15:39:20, Asia/Shanghai.

## Scope

Identify which non-Q8 graph operations, shapes, types and layouts are newly
active on the corrected CUDA library's changed-prompt, repeated-prompt and
tokenizer-boundary image-session paths. Use the same predeclared seven
calibration/development calls as the Q8 session arithmetic audit. Preserve
all frozen quality, performance and arithmetic receipts. This is coverage
analysis, not a substitute for independent same-operand arithmetic.

## Approach and steps

1. Build a diagnostic GGML scheduler sidecar outside the frozen source tree
   that records each graph node's operation, type, shape, strides, source
   types/shapes and selected backend without copying tensor payloads. Bind
   the sidecar source and binary to the corrected prepared GGML headers and
   native image probe. Capture the seven calls in one process using the
   already frozen session case TSV and compare non-runtime outputs and
   cumulative CUDA/vision counters to the prior uninstrumented run.
2. Reconcile the lightweight inventory against the corrected single-prompt
   graphs and previously audited old-library operator classes. Enumerate
   any genuinely new operation/type/shape signatures and source paths.
   Preserve graph order and metadata hashes.
3. Use the findings to choose the minimum still-required independent
   non-Q8 arithmetic capture. Keep whole-recipe/deployment statuses
   `NOT_RUN` until those checks pass directly; do not infer a pass solely
   because the graph signatures match.

## Verification

Run with `rtk proxy`, remote desktop inactive and no other CUDA compute
process. Store the new sidecar, inventory and receipts in a fresh ignored
SSD directory, retaining immutable existing evidence. Run relevant tests,
documentation/link check and `git diff --check` after changes.

## Results

The sidecar built against the corrected prepared GGML headers captured the
same seven development-only calls in one process without copying tensor
payloads. All seven non-runtime outputs matched their previous
uninstrumented corrected-library outputs. Their CUDA-node deltas were
`3,461`, `1,900`, `0`, `1,900`, `1,900`, `3,461`, `1,900`; the repeated prompt
added no graph. The inventory contains 16 graphs and occupies about 2.3 MB.

An independent reconciliation compared every graph node against the four
graph classes in the older library's fully audited single-prompt inventory.
For all **23,988 nodes**, operation, type, shape, stride, tensor name, source
types/shapes and assigned backend matched exactly in graph order. No new
graph signature appeared for changed, repeated, whitespace or long prompts,
or for the larger image. The scheduler assigns 126 metadata-only `RESHAPE`
or `PERMUTE` nodes to CPU across these graphs; the runtime recorded zero CPU
compute nodes and all substantive compute stayed on CUDA. The ignored
`session-graph-reconciled.json` SHA-256 is
`23410be37e9a3e66b4bacdc6ba70569e3d1306535cffebc33c99b1289584ae96`.
It binds the current sidecar source/binary, new inventory and outputs, old
audited inventory, corrected probe/library/model and passing Q8 session
receipt by SHA-256.

This closes the *graph coverage* question for the seven selected calls. It
does not independently recompute non-Q8 output arithmetic on their new
operands, and it does not prove every accepted image/text value is covered.
Whole-recipe arithmetic and deployment stay **NOT_RUN** pending the
requirement-by-requirement arithmetic scope audit or direct additional
checks. Existing raw evidence remains immutable; remote desktop was
inactive and the GPU was empty after the run.

A source-provenance check sharpened the remaining gap: the older full-graph
arithmetic library is prepared from GGML revision
`353b63b439f27ab2cc19dac97ab1681ba6d2d084`, whereas the corrected
candidate is prepared from `d7cb574130e6f01ad25b3289685489200febcd74`.
Many GGML CUDA files differ between those revisions. Exact graph metadata
therefore **cannot transfer the old library's non-Q8 numerical receipts** to
the corrected library. Direct non-Q8 same-operand checks are required; the
large older corpus cannot be treated as a passing current-library audit.
