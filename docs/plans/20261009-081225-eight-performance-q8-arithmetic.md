# Frozen Q8 arithmetic across all eight predeclared performance inputs

Created: 2026-10-09 08:12:25, Asia/Shanghai.

## Scope

Extend the exact frozen CUDA Q8_0 matrix arithmetic audit from three inputs
to all eight predeclared performance inputs from the calibration/development
splits. Use the same frozen
model, binary, GGML CUDA library, 2e-5/1e-6 gate and native development
outputs. This checks input diversity for the Q8 component only; it does not
open the independent evaluation or reserve, qualify the whole graph on all
inputs, or replace exclusive-GPU performance acceptance.

## Approach and steps

1. Bind each remaining performance case to its calibration/development image, prompt
   index, image/output hash, recipe and frozen dataset. Use distinct fresh
   ignored evidence directories on the separately mounted volume, reached
   through `build/q8-integrated-arithmetic-20261009/many-external/`.
2. Run the unchanged frozen binary with the unchanged capture sidecar once per
   case. Independently verify every active Q8 matrix node against its exact
   GGUF weight bytes, fresh Q8_1 RHS staging and F64 same-operand dots, and
   compare final inference payloads with prior unobserved exports.
3. Reconcile all eight per-case receipts and raw artifact hashes, graph
   shapes, stable weights and input-dependent activations. Preserve raw
   evidence and record any failure without relaxing the frozen gate.

## Verification

Confirm the prior three-case receipt before extending. Run each verifier in
the isolated reference environment, a final read-only reconciliation, the
relevant tool suite, docs checker and `git diff --check`. Review storage and
delete only disposable intermediates; retain the current build, model and
all evidence needed to reproduce the conclusions.

## Results

The five remaining predeclared cases completed with the exact frozen binary
and independent per-node verifier. Each passed 327 active Q8_0 matrix nodes,
161,445,056 same-operand F64 dots, Q8_1 RHS staging and exact final inference
payload parity with its earlier unobserved export. The cases and per-case
receipt SHA-256 hashes are:

| Performance input | Split | Receipt SHA-256 |
| --- | --- | --- |
| `perf-source-middle` | development | `378d0674abb57a555d61e07d903170bef178a43e832926d58d8d93197023285a` |
| `perf-source-large` | development | `55ce575aa77ea2df4c01d9e1fe888a391ced5b2cfc5c9d69d6e0b9526d318b5f` |
| `perf-instances-few` | calibration | `7b37473fb93708abab3cf2b6eff4846d248615eb6b67cc68484499dc8d1d4a57` |
| `perf-objects-small` | development | `e75f774cda65ef359015f3b604d5669ed0b6d324ae8f2f86b1c3a33dafa15dfb` |
| `perf-objects-large` | development | `7240b56553bddd7540f0fe34eb78acea8ad49fd57b5f4974871ca9c4ca823037` |

Together with the three prior cases, all eight predeclared performance
inputs passed 2,616 active Q8 matmul instances and 1,291,560,448 exact-operand
dots. The worst relative L2 across inputs was `1.3390837367721728e-07`,
far below the unchanged 2e-5 limit. Final reconciliation independently
rehashed all 13,080 per-node payload files, checked stable Q8 weight bytes,
matched node order and shapes, and found eight distinct RHS-input signatures.
Relative to the initial small-source case, each other input changed RHS
activations and dot outputs at 312 nodes. The receipt is
`build/q8-integrated-arithmetic-20261009/eight-q8-reconciled.json` (SHA-256
`f9f1f43298fb83ba204ba01a8caf2cdb7ef84919df0ef6df66fef01ffdeff824`).

The complete raw captures and staging outputs occupy about 13 GiB through
`build/q8-integrated-arithmetic-20261009/many-external/`; that Git-ignored
path is backed by `/mnt/SSD1` and available only in this validation workspace.
The repository volume retained about 1.3 GiB free; no preexisting evidence
was removed. Only seven of these inputs belong to development; the
predeclared `perf-instances-few` input belongs to calibration. No evaluation
or reserve inference was used. The frozen campaign and numerical gate are
unchanged. This is Q8 matrix-component evidence across eight inputs; the
whole-recipe arithmetic, exclusive-GPU performance and deployment gates
remain `NOT_RUN`.
The completed validation cycle passed 170/170 tool tests, the bilingual/local
link checker on 106 documents, and `git diff --check`. Cleanup review found
no disposable new intermediate: each raw capture, staging file, observer
output and receipt is bound for reproduction. The remote-desktop process
remained an NVIDIA compute PID after the arithmetic runs, so no formal
exclusive-GPU performance process was started.
