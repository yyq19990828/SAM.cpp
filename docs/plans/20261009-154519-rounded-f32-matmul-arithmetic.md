# Corrected-library F32 matrix arithmetic on a development image

Created: 2026-10-09 15:45:19, Asia/Shanghai.

## Scope

Directly verify the non-Q8 F32 matrix nodes on the corrected GGML CUDA
library for the predeclared `perf-source-small` development case. Older
full-graph arithmetic used a different GGML revision and cannot be
transferred by graph metadata alone. Preserve the frozen final quality,
performance, model and campaign identities, and do not use evaluation or
reserve inputs. This is a bounded F32 matrix-component check, not a
whole-recipe verdict.

## Approach and steps

1. Recompile the existing F32 matrix capture sidecar against the corrected
   prepared headers in a fresh ignored evidence directory. Bind the exact
   case TSV, normalized image, prompt, native binary, GGUF and CUDA library.
2. Capture all active F32 matrix operands and outputs from one native CUDA
   run. Compare the non-runtime final output with the previous uninstrumented
   corrected-library development export. Independently recompute every
   same-operand F64 dot and check named GGUF weights under the unchanged
   `2e-5` relative-L2 and `1e-6` zero-norm gates, including worst-row/error
   reporting.
3. Rehash the raw captures and report identities, cross-check node counts
   and signatures against the corrected session graph inventory, and retain
   raw evidence. Do not infer results for other inputs or nonmatrix nodes.

## Verification

Use `rtk proxy`, the isolated reference environment, an inactive remote
desktop and an otherwise idle GPU. Keep raw evidence and immutable receipts
on the SSD; preserve existing builds/models and clean only disposable
intermediates. Run applicable tests, documentation check and
`git diff --check` after changes.

## Results

The sidecar was rebuilt against the corrected prepared GGML headers, and
the exact frozen `perf-source-small` normalized development input was run
with the corrected CUDA image probe. The run captured all **192** active F32
matrix nodes, about 17 GiB of raw operands and outputs, and produced a
non-runtime final payload identical to the previous uninstrumented
corrected-library development export. Runtime recorded 3,461 CUDA compute
nodes and no CPU/Metal/BLAS compute node.

Every captured F32 node passed independent F64 same-operand dot
recomputation under the unchanged `2e-5`/`1e-6` gates. The verifier checked
1,897,738,953 dots; worst node relative L2 was `2.553692823e-6`, and the
worst reported row relative L2 was `5.696617295e-6`. At 141 nodes with
named F32 GGUF weights, the captured left operand also matched the exact
GGUF bytes. The ignored per-node verification SHA-256 is
`2663ad1f9703a4830a0c5db92699de066ca598e297a5598233b0026eadcac3b8`.

An independent read-only reconciliation rehashed all 576 raw files and 14
identity files, checked all 192 matrix positions and operand/output shapes
against the corrected-library session graph inventory, and confirmed the
passing numerical totals. Its ignored
`perf-source-small-f32-reconciled.json` SHA-256 is
`22121a87e9576dbdf74c2ebc7b3de4d46ebdbc39b93c4b5de7fcbd3d9ce416eb`.
The evidence volume retains about 109 GiB free. Remote desktop remained
inactive and no GPU compute process was left running. This is a bounded
current-library F32 matrix **component PASS**; nonmatrix operators and
other accepted inputs have not been promoted to whole-recipe arithmetic.
