# Frozen F32 matrix arithmetic across all eight performance inputs

Created: 2026-10-09 08:26:27, Asia/Shanghai.

## Scope

Extend real-operand F32 matrix arithmetic for the exact frozen custom-Q8
CUDA recipe from the initial small-source and negative-prompt cases to the
six remaining predeclared calibration/development performance inputs. All
eight already passed Q8 matrix-component checks. Preserve the model,
binary, CUDA library, original-quality verdict and 2e-5/1e-6 arithmetic
gate. Do not use the final evaluation or reserve for tuning. This is
matrix-component coverage, not whole-graph or formal performance acceptance.

## Approach and steps

1. Reuse the previously bound case IDs, prompts, image hashes and unobserved
   Q8 native outputs. For each fresh case, capture all F32/F32/F32 matrix
   inputs and outputs through the existing preload sidecar on the frozen
   binary. Retain raw evidence on the separate mounted volume through a
   Git-ignored `build/` symlink.
2. Independently recompute each node's output from exact captured operands
   in F64, check GGUF named F32 weights, finite values, unchanged numerical
   limits and final masks, boxes, ranking and scores. Generate an independent
   graph inventory for each input with the frozen profiler binary.
3. Reconcile graph order, types and dimensions against all 192 F32 and 327 Q8
   nodes per input. Rehash every source/operand/output receipt, compare the
   eight input signatures and report the exact matrix-component scope.
   Preserve failures and leave whole-recipe arithmetic `NOT_RUN` until all
   active graph operators across the required inputs are verified.

## Verification

Run the frozen capture/profiler binaries, isolated reference verifiers and
read-only aggregate reconciliation. Run the relevant tool suite, docs checker
and `git diff --check`. Clean only disposable intermediates after the cycle;
retain the verified build, model, raw captures and bound evidence.

## Results

All six remaining predeclared performance inputs passed independent F32
same-operand verification: 192/192 active nodes and 1,897,738,953 F64
recomputed dots per input. The per-input receipt SHA-256 hashes are:

| Performance input | Receipt SHA-256 |
| --- | --- |
| `perf-source-middle` | `385e039899c34105839285c72c22a1a89665eff62f07a4fe4ad5fb192e7a15a1` |
| `perf-source-large` | `5e0b2c6164861cc1dd10826c0f466796966919537007fb5f15152efd9d0d91f7` |
| `perf-instances-few` | `d463e26076ddfa5670375b7b2bb2dc4ab85bd6b410151785a6dddf394fccd3c9` |
| `perf-instances-many` | `e55633921fefcd6318d31a92f4cd6cef9a5bfec0f5ca9019d201e2d0cd3f75d7` |
| `perf-objects-small` | `e674cb5a46e7351565d520840bde10de4142286db930a46e5bd685581a456314` |
| `perf-objects-large` | `3b6691ea875a89ba5590e7e86ba5be1661d2a7572ff27f6f078e7f39307d4fc7` |

Across all eight predeclared calibration/development inputs, independent
graph inventories each found exactly 327 Q8_0/F32/F32 and 192 F32/F32/F32
matrix nodes in four equal-size graph partitions. Per-node receipts pass the
unchanged arithmetic gate for all **4,152 matrix instances**, comprising
1,291,560,448 Q8 and 15,181,911,624 F32 same-operand dots. The worst F32
tensor relative L2 is `3.5238818156513417e-06`, under the `2e-5` limit.
All 141 named F32 GGUF weights per input matched their stored bytes and
remained byte-identical between inputs; the F32 RHS hash signatures were
distinct across all eight. Final masks, boxes and ranking matched each
prior unobserved native export. The first input's diagnostic callback
changed scores by at most `1.19e-7`; the other seven final inference
payloads matched exactly after removing runtime measurements.

The aggregate reconciliation rehashed all 17,688 bound per-node Q8/F32
payload files and matched each ledger to its independent graph inventory.
Its receipt is `build/q8-integrated-arithmetic-20261009/eight-all-matmul-reconciled.json`
(SHA-256 `48f1288e76054075113cb5e1b56da9778efffbe63a005ecebcfc0b15badc8bcd`).
Full raw evidence and staging occupy about 125 GiB under Git-ignored
`build/q8-integrated-arithmetic-20261009/many-external/`, backed by
`/mnt/SSD1`; these local files are available only in the validation
workspace. The repository volume retained about 1.3 GiB free. No frozen
model, campaign, threshold, evaluation input or reserve input was changed
or used for tuning. This is complete active **matrix** coverage across eight
inputs; non-matrix graph operations still need multi-input validation.
Whole-recipe arithmetic, exclusive-GPU performance and deployment remain
`NOT_RUN`.

Later workspace cleanup (2026-10-09): the historical top-level F32 capture
payloads on `/mnt/SSD1` were deleted after the independent verification and
reconciliation receipts had been recorded and checked as `PASS`. The unchanged
receipts remain under `build/q8-integrated-arithmetic-20261009/`, but their
removed raw payloads can no longer be rehashed locally. The current rounded
candidate's separate archive was preserved.

The completed validation cycle passed the isolated Python tool suite
(170/170), the bilingual/local-link documentation check (109 documents)
and `git diff --check`. Raw captures, graph inventories and receipts remain
necessary validation evidence; no disposable intermediate was retained.
