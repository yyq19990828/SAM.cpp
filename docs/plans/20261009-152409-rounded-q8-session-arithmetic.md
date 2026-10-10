# Corrected-library Q8 image-session arithmetic

Created: 2026-10-09 15:24:09, Asia/Shanghai.

## Scope

Check the exact corrected-library CUDA Q8_0 matrix arithmetic exercised by
one predeclared seven-call image session: fresh image, changed prompt,
repeated prompt/cache hit, whitespace, long tokenizer boundary, large image
and changed prompt on the large image. Preserve the frozen campaign, model,
binary and existing output receipts. Use only calibration/development inputs,
never final evaluation or unopened reserve images. This checks the Q8
component and session path coverage, not full-graph arithmetic by itself.

## Approach and steps

1. Bind the seven already selected calls to byte-identical normalized images
   in the fresh development dataset, exact prompt text and the previous
   corrected-library uninstrumented session outputs before GPU inference.
   Check identities and free evidence capacity.
2. Run all seven calls in one process with the corrected native image probe
   and a sidecar compiled against the corrected GGML headers. Capture every
   active Q8 matrix node's exact weight, F32 RHS and F32 dot; preserve raw
   data. Check graph/node counts and that the repeated prompt adds no new
   CUDA compute.
3. Compare each observed non-runtime output to the existing uninstrumented
   corrected-library session output. For every captured Q8 node,
   independently check GGUF weight bytes, native MMVQ/MMQ RHS staging and
   same-operand F64 dots under the unchanged `2e-5` relative-L2 and `1e-6`
   zero-norm gates. Rehash all raw and identity files in a final receipt.
4. Keep whole-recipe arithmetic and deployment `NOT_RUN` unless the
   remaining non-Q8 full graph and host/session requirements also acquire
   direct evidence.

## Verification

Run agent commands via `rtk proxy`, with remote desktop inactive and no
competing GPU compute process. Use fresh Git-ignored directories on the SSD;
retain current verified builds, GGUF, raw captures and immutable reports.
Run applicable tests, documentation/link check and `git diff --check` after
the cycle; clean only disposable intermediates.

## Results

Before GPU inference, the seven exact predeclared prompts were rebound to
byte-identical normalized development images from the fresh dataset and
their previous uninstrumented corrected-library outputs. The ignored
`bound-session.json` SHA-256 is
`b49daf890c99854a10822c82c4322f0beafc59a28c21dcc40a7702d344d2e2b8`.
The corrected image probe, GGUF and CUDA library remained the identities
recorded in the [eight-input matrix audit](20261009-150104-rounded-q8-arithmetic-recapture.md).
No final evaluation or reserve input was used.

One instrumented process completed all seven calls. Every non-runtime
result exactly matched its previously uninstrumented corrected-library
counterpart. The cumulative CUDA-node increments were `3,461`, `1,900`,
`0`, `1,900`, `1,900`, `3,461` and `1,900`; vision encoding occurred only
on the first small and first large image. The repeated prompt added no new
CUDA graph or Q8 capture. The sidecar recorded 16 graphs and 1,838 active
Q8 matrix nodes.

The independent verifier checked all captured GGUF weight bytes, native
MMVQ/MMQ RHS staging and same-operand F64 dots; every node passed the
unchanged gate. The per-node report SHA-256 is
`d9aa93a169b6082e06bddc0eb768007327cf5907491c04924a9eaf4af672f174`.
A separate read-only audit rehashed 9,190 raw files and 34 identity files,
reconciled graph/Q8-node counts, and confirmed 936,772,736 checked dots,
15,173,800 staged blocks and worst relative L2 `1.3342969910e-7`.
Its ignored `session-q8-reconciled.json` SHA-256 is
`defd48dac7720dffeedb5e9f01286cdb0a99934525d91d04b04cac6f7449f1bd`.
All 240 observed Q8 weight/shape/mode signatures were already present in
the first independently checked performance graph; there were zero new Q8
signatures. This binds the Q8 component across changed, repeated, blank and
long-prompt paths, but does not check every non-Q8 compute node or the full
accepted-input domain. Whole-recipe arithmetic and deployment remain
**NOT_RUN**.

Raw capture, staging data and receipts occupy about 12 GiB on the SSD and
remain available unchanged. After this cycle about 125 GiB was free. The
remote desktop stayed inactive and the GPU had no lingering compute
process.
