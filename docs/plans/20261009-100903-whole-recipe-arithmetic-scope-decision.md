# Whole-recipe arithmetic scope decision for the frozen custom Q8 allocation

Created: 2026-10-09 10:09:03, Asia/Shanghai.

## Scope

Audit whether the completed eight-input CUDA active-graph arithmetic
receipt is sufficient to promote the frozen custom-Q8 recipe's
`arithmetic_status` from `NOT_RUN` to `PASS`. Treat the frozen v2 gate plan
as the policy, not the new receipt's status field. Preserve all campaign,
quality, model, evaluator and threshold identities. Do not use final
evaluation or unopened reserve inputs for tuning.

## Approach and steps

1. Enumerate the arithmetic prerequisites and boundary cases in the frozen
   gate plan, including input/tokenizer, tensor inventory, layout, finite
   values, bounds, backend placement, GGUF Q8 block encoding/decoding,
   same-operand arithmetic and result provenance.
2. Map each requirement to an authoritative current receipt, source-level
   implementation contract and test. Check whether the evidence covers the
   whole recipe and its declared input domain rather than only eight
   representative performance cases. Identify any missing or indirect
   proof without weakening thresholds or relabeling component results.
3. If coverage is complete, create a new read-only reconciliation with
   independent identity checks and a narrowly justified PASS. Otherwise
   keep `arithmetic_status: NOT_RUN` and state the specific remaining
   experiment or proof needed. Leave exclusive-GPU performance and
   deployment decisions separate.

## Verification

Inspect the frozen policy, model/schema, campaign, component receipts,
integration tests and current code. Recheck hashes where scope claims
depend on immutable evidence. Run relevant tests, documentation links and
`git diff --check` after any resulting implementation or documentation edit.

## Results

The eight-case aggregate is intact: `build/q8-integrated-arithmetic-20261009/eight-full-graph-reconciled.json`
has SHA-256 `b13de109bd53413f7814410192c4ff29fbdfe5ee7c494206f982531d6bc1ee51`.
It reports `PASS` for the eight predeclared calibration/development
performance inputs, not for arbitrary text-image calls. Across those inputs
it reconciles 27,688 compute-node instances, 20,920 metadata-node instances,
9,240 leaves and 16,473,472,072 same-operand matrix dots, and rehashes
34,392 raw payloads plus 515 identity files. These ignored `build/` receipts
and captures are available only in this validation workspace. The candidate
model, binary, CUDA library, campaign and quality-receipt hashes remain the
frozen ones; the independent final image-quality decision is `PASS`.

| Frozen prerequisite | Current evidence | Scope decision |
| --- | --- | --- |
| Input, tokenizer and host output contracts | `sam_contracts` exercises official tokenizer goldens, fixed 32-token padding, image stride/resize/rounding boundaries, detection thresholds and nonfinite output rejection. Seven official-reference cases have identical token IDs and preprocessed F32 image tensors; the 1,024-image final-quality export matches sample and prompt identities. | Contract tests and observed inputs pass; they do not independently recompute every host transform on every accepted input. |
| GGUF inventory, type, layout, bounded access and placement | The schema-4 model declares 220 Q8_0 text/fusion/decoder linears and 913 F32 tensors; `sam_contracts` rejects malformed profile/type/shape/offset cases. The frozen CUDA graph inventory records no CPU-tail compute nodes; per-node Q8 weight bytes match their named GGUF slices. | Passes for the frozen model and eight observed graphs. The two geometry point weights are dormant in the current text-only image task, not missing active nodes. |
| Q8 storage and same-operand kernel arithmetic | The frozen-library MMVQ/MMQ probes cover zero, tails, signed ties and K=16,384; the 14 custom-shape probes cover every actual Q8 K. In the eight real-input graphs, all active Q8_0 matrix nodes pass independent Q8_1 staging and F64 packed-operand dots under the unchanged 2e-5/1e-6 rules. | Passes on measured modes and operands. The general policy's saturation and overflow boundaries are not separately demonstrated for this exact frozen candidate/library; verifier rejection of malformed bytes is not an executed native boundary. |
| Complete graph arithmetic and provenance | All compute and metadata node classes in each of the eight captured active graphs have independent numerical or exact-bit checks; final payloads agree with unobserved native exports. The aggregate binds graph order, tensor shape/type/stride, observers, source, binary, library and raw hashes. | Passes for those eight single-prompt executions. Observer output parity proves observation did not change those outputs, not universal arithmetic correctness. |
| Full declared image-session path | The frozen recipe is CUDA F32 compute/F32 cache with text-prompted image segmentation. `ImageSession` also has changed-prompt and repeated-prompt/cache-hit paths, and accepts other valid token contents and image dimensions before fixed 1008 preprocessing. Existing session tests check cache behavior, while final quality spans 3,833 prompts. | No independent full-graph arithmetic reconciliation yet binds the changed-prompt/cache path or a predeclared set of input extremes to the frozen candidate. Eight performance inputs alone cannot certify every declared path. |

**Decision:** retain `whole_recipe_arithmetic_status: NOT_RUN`. The eight-case
`performance_input_arithmetic_status: PASS` is valid and substantial but is
not a substitute for the frozen whole-recipe prerequisite. Closing the gap
requires a *predeclared* CUDA boundary/path matrix for this exact recipe:
exercise changed/repeated prompts and image-size extremes, prove graph
coverage or independently check any new nodes, and run native Q8
saturation/overflow/epilogue boundary fixtures under the frozen library.
Check token IDs, preprocessing, finite values, postprocessing and backend
placement at the same boundaries. Bind these results to the existing model,
campaign and source identities before deciding the whole-recipe state;
do not retune against final-evaluation images or infer coverage from the
unopened reserve. The exclusive-GPU performance gate remains independent.

The existing verified CUDA build passed all 29 CTests during this audit.
No executable, model, threshold or immutable receipt changed.
