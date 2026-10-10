# Nonvision Q8 frozen-library shape arithmetic

Created: 2026-10-09 06:36:51, Asia/Shanghai.

## Scope

Extend the exact-packed-operand Q8_0 CUDA MMVQ/MMQ arithmetic evidence to
matrix dimensions selected from the newly quality-passing SAM 3
`text,fusion,decoder` allocation. Its 220 Q8_0 matrices have K dimensions
256, 512, 1,024, 2,048 and 4,096; the prior frozen-library oracle included
K=1,024 but not the other four actual K values. Keep the GGML v0.25.3 library,
quality campaign, gates and source files unchanged. This is additional
component evidence, not a whole-recipe arithmetic or deployment claim.

## Approach and steps

1. Recheck the custom GGUF tensor inventory and choose actual `(M,K)` shape
   representatives, including narrow decoder outputs and text/fusion layers.
2. Construct deterministic finite F32 probe operands for those shapes with
   N=3 and N=64 so the frozen v2 binary selects MMVQ and MMQ respectively.
   Reuse the isolated probes already linked to the exact library in the
   quality campaign; do not rebuild or edit that library, probes or sources.
3. Export native Q8_0 weights, native Q8_1 RHS staging and raw CUDA dots.
   Independently verify staging bytes and every same-operand dot against the
   unchanged 2e-5 F32 relative-L2 arithmetic limit and zero-norm rule.
   Confirm selected kernels where a profiler trace is available.
4. Bind fixture, binary, library, source, result and independent verifier
   identities in new ignored evidence. Retain failures; distinguish tested
   shapes from all model activations and graph operations. Run appropriate
   tests and clean only disposable intermediates.

## Verification

Compare all binary and GGML CUDA library hashes against the earlier frozen
v2 receipt and new quality campaign before any arithmetic claim. Use fresh
output directories, isolated reference Python for the verifiers, then run
relevant tests, documentation/link checks and `git diff --check`. The exact
candidate model and final image-quality PASS remain untouched.

## Results

The exact custom-GGUF conversion manifest identifies 220 Q8_0 linear
matrices. Seven selected tensor shapes cover every actual K value (256, 512,
1,024, 2,048 and 4,096), a fusion attention projection, text resizer/MLP,
decoder heads and 4/8-row narrow outputs. Each shape used separate N=3 and
N=64 fixtures with deterministic **synthetic F32 weights and activations**;
the latter have 64 distinct rows. These fixtures test actual matrix *shapes*,
not original model weights or observed model activations. The fixture manifest
is `build/q8-nonvision-shapes-20261009/inputs.json` (SHA-256
`3d6f2b0b758010fe982e41063a0d45e5563cc8eed6d01c2a95249253568f609e`).

All 14 native CUDA probes and independent packed-operand verifications passed
the unchanged 2e-5 F32 relative-L2 limit: 7 MMVQ cases checked 7,716 dots
and 792 RHS blocks; 7 MMQ cases checked 164,608 dots and 4,608 blocks. The
combined 172,324 dots and 5,400 staged blocks had worst whole-output
relative L2 `1.165e-7`. The linked GGML v0.25.3 CUDA library has SHA-256
`1e4f954340ea2c30cb60deb49f4ccda6780993b2e09cb81bc8105aa053c925ef`,
identical to the previous frozen arithmetic receipt and the image campaign's
library. No frozen binary, library, model, source or quality gate was changed.
The per-case source, operand, binary, verifier and library bindings are in
`build/q8-nonvision-shapes-20261009/summary.json` (SHA-256
`753b9e075cae1fcc92029486fdf8efce8e99d1e16f577f52e337ea193f7c5064`).

Nsight traces for the narrow `(M,N,K)=(4,64,256)` and wide
`(1024,64,4096)` cases contain `quantize_mmq_q8_1` and `mul_mat_q`; the
wide `(1024,3,4096)` trace contains `quantize_q8_1` and `mul_mat_vec_q`.
An independent read-only audit rechecked all 14 receipts and their hashed
operands, parsed the three traces, and linked the component summary to the
new campaign's image-quality PASS. Its receipt is
`build/q8-nonvision-shapes-20261009/audit.json` (SHA-256
`29dc794e73e44f57852eb4dda42c19b689b4cbc572731b5035ec408f2a660dbc`).
All paths under `build/` are Git-ignored local validation evidence. The
superseded Nsight SQLite extracts were removed; original profiler reports,
input fixtures, probe outputs and audit remain.

This closes the selected **component shape-arithmetic** gap from the
[follow-up evaluation](20261009-043541-q8-independent-followup.md), not the
whole-recipe arithmetic gate. Real model activations, every one of the 220
matrices, other operators and exclusive-GPU performance remain unqualified.
The component status is **PASS**; whole-recipe arithmetic and performance are
still **NOT_RUN**.

The isolated Python tool suite passed 170/170 checks after this evidence was
bound. Documentation/link checks passed 92 documents and `git diff --check`
found no whitespace errors. No source rebuild was needed for these frozen
probe runs.
