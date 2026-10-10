# Integrated nonvision Q8 arithmetic capture feasibility

Created: 2026-10-09 06:44:26, Asia/Shanghai.

## Scope

Move the quality-passing SAM 3 CUDA `text,fusion,decoder` Q8_0 recipe from
selected synthetic matrix arithmetic toward a whole-recipe same-operand
audit. Preserve the frozen evaluation campaign, candidate GGUF, gates,
reference outputs and CUDA libraries. An observed diagnostic execution must
not be presented as the already frozen production binary, and its output
must agree with that binary before it supports a recipe claim.

## Approach and steps

1. Inspect the exact frozen GGML v0.25.3 scheduler callback ABI and the
   retained image recipe/binary identities. Use an existing linked graph
   profiler on a frozen development image to inventory every Q8_0 matrix
   operation and identify graph partitioning, backend placement and operand
   types. Record a bounded inventory before attempting large tensor dumps.
2. If the inventory is complete, build an **isolated, diagnostic-only**
   `LD_PRELOAD` sidecar outside the frozen source tree. It interposes the
   scheduler's graph-compute call in the exact frozen image executable,
   attaches GGML's per-node evaluation callback, and saves packed Q8_0
   weights, F32 RHS and F32 raw dots immediately after each matrix node.
   Bind sidecar source, frozen binary/library, recipe, case and outputs.
   Never read all intermediate nodes after the graph finishes because the
   scheduler may have reused their storage.
3. Independently reconstruct Q8_1 RHS staging and each dot using the existing
   packed-operand oracle at the unchanged 2e-5 F32 limit and zero-norm rule.
   Compare the observed diagnostic final masks/boxes/scores bytewise with the
   frozen candidate on the same input. Audit complete quantized-node
   coverage; non-Q8 operations require their own evidence before claiming a
   full-recipe arithmetic PASS.
4. Retain failures and limitations. Recheck the formal performance runner's
   GPU exclusivity prerequisite without weakening it. Do not start its
   measurement until no other compute PID is present.

## Verification

Check the original campaign and model hashes, dynamic GGML library SHA-256,
source/binary provenance, every captured tensor's dimensions and storage,
independent verifier results, complete node counts, final-output parity,
relevant tests, documentation links and `git diff --check`. Keep evidence in
ignored `build/` directories and remove disposable profiling intermediates.

## Results

The retained `sam_profile_graph` executable links the exact frozen GGML
v0.25.3 CUDA library (SHA-256
`1e4f954340ea2c30cb60deb49f4ccda6780993b2e09cb81bc8105aa053c925ef`).
On frozen development case `coco-000000001503-p0` (`tv`), it recorded four
graphs with 2,910, 207, 847 and 2,112 nodes. Their 327 active Q8_0
`MUL_MAT` nodes use F32 RHS and F32 output; the only other Q8_0 consumers
are 111 `VIEW` nodes. The 327 matrix nodes reference 218 of the candidate
GGUF's 220 Q8_0 tensors. `geom.points_pool_project.weight` and
`geom.points_pos_enc_project.weight` are dormant on this text-image path.

An isolated `LD_PRELOAD` sidecar then interposed the graph-compute call of
the **exact frozen `sam_precision_image_probe` binary** (SHA-256
`a2980263d888d21835b3644032311deabe4124d678d2042041b7c8794dd72a53`)
without rebuilding it or the GGML library. The scheduler callback saved the
Q8_0 weight bytes, F32 RHS and raw F32 dot output immediately after each of
those 327 nodes, including strided RHS views. The sidecar changed execution
observation and timing, not the selected recipe or inference payload: its
scores, boxes, ranked masks, token IDs and other non-runtime output fields
were exactly equal to the pre-existing frozen development export for this
case. Neither run used the unopened reserve or the new final-evaluation
images.

An independent Python oracle checked each captured weight as bytes within
its named tensor in the frozen GGUF. It independently checked the staged
Q8_1 scale/integer layout, then decoded the **actual packed operands** and
recomputed each F32 dot in F64, comparing native output at the unchanged
2e-5 relative-L2 limit (1e-6 absolute for zero-norm cases). All 327 nodes
passed: 28 MMVQ and 299 MMQ, covering 161,445,056 output dots and
2,612,156 RHS staging blocks. The worst node-relative L2 was `1.1303e-7`;
no node had a zero-norm reference. The full local receipt is
`build/q8-integrated-arithmetic-20261009/full-verification.json` (SHA-256
`c81cfcbae9002388f8e6bcd0cbc4583afb74312974223ac0b2b9247c6f5673d3`).

A read-only reconciliation rehashed the 1,635 per-node operand/staging files,
matched every node in graph order and shape to the independently collected
graph inventory, confirmed the 218/220 active-weight coverage and two dormant
point weights, and rechecked the frozen campaign, exact binary/library/model
and independent final image-quality PASS. Its receipt is
`build/q8-integrated-arithmetic-20261009/reconciled.json` (SHA-256
`00a8d74f3bf097838cbfc925d874f7c74e3d2df34d57128df11c4bde009c545d`).
The underlying artifacts under `build/` are Git-ignored and available only
in this validation workspace. The capture is diagnostic and its timing is
not performance evidence.

This yields **PASS for every active Q8_0 matrix multiplication in one real
text-image inference** of the exact frozen recipe. The two dormant point
weights, other prompts/images, F32 graph operators and epilogues are not
independently covered by this one capture. The whole-recipe arithmetic gate
therefore remains **NOT_RUN**; exclusive-GPU performance and deployment also
remain **NOT_RUN**. No arithmetic or quality threshold was changed after the
final evaluation.

The sidecar source passed C++17 `-Wall -Wextra -Werror` syntax checks. The
read-only receipt's bound identities still rehashed after disposable trial
captures were removed. The repository's bilingual documentation/link check
passed 94 documents and `git diff --check` found no whitespace errors.
