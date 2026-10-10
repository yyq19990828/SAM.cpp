# Corrected-library Q8 arithmetic recapture

Created: 2026-10-09 15:01:04, Asia/Shanghai.

## Scope

Extend the corrected MMQ-scale CUDA library's arithmetic evidence from the
extreme boundary and one representative RHS to the eight already predeclared
calibration/development performance inputs. Keep the frozen fresh campaign,
model, native binaries, source, acceptance gates, final-quality and formal
performance evidence unchanged. This is a Q8 matrix-component audit; it does
not by itself qualify full-graph or whole-recipe arithmetic.

## Approach and steps

1. Rehash the corrected image probe, Q8 staging probes, GGUF and CUDA library;
   confirm the remote desktop stays inactive and the GPU has no competing
   compute process. Use only the frozen eight performance cases and their
   calibration/development normalized images, never final evaluation or the
   unopened reserve.
2. Rebuild or rebind the existing GGML graph capture sidecar to the corrected
   prepared headers, and run the corrected native image probe for each case
   with its frozen prompt. Preserve each raw Q8 weight, F32 RHS and F32 dot
   capture in a fresh ignored evidence directory on the separate SSD.
3. For every active Q8_0 matrix node, check the captured weight bytes against
   the exact GGUF, independently stage its RHS with the corrected native MMVQ
   or MMQ path, validate packed scales/integers, and recompute the same-operand
   F64 dot under the unchanged `2e-5` relative-L2 and `1e-6` zero-norm gates.
   Compare non-runtime final payloads to the corresponding corrected-library
   development export. Reconcile all eight case identities, node counts and
   raw file hashes without altering the old receipts.
4. Assess whether a later full-graph/session audit can keep independently
   checkable evidence within available capacity. Do not promote arithmetic or
   deployment status on the basis of this component audit alone.

## Verification

Use `rtk proxy` and the isolated reference environment. Store code and raw
captures under Git-ignored `build/` on the SSD; preserve immutable old and
new receipts. Run appropriate tool tests, documentation/link check and
`git diff --check`. Clean only disposable trial intermediates, preserving
current verified builds, original GGUF, successful captures and receipts.

## Results

The exact existing eight-case performance selection was bound before these
diagnostic GPU runs (ignored `bound-cases.json` SHA-256
`d3c9a759e9a3da40b7c0fbc917a976755be74106eea30b0fefb6f983f7305c6e`).
The corrected CUDA image probe, GGML CUDA library and unchanged diagnostic
GGUF SHA-256 values were respectively
`f17ad495ec0f21223751378b93dab3baf521929b8396c5fc57b0bb6e9e2454b0`,
`d50d7f7c44c964d775dba8e89cb12b4a9b93e2d21b3191a4e374ec594a076228`
and `f854afae5a2eb2c388ff2dd85804a4327b4525b45a09c1fdc9b5f5619c93142e`.
The capture sidecar was recompiled against the corrected prepared GGML
headers; its shared-library SHA-256 is
`764ce43f802d1011cca5368b9497270e84ad2ef7378d4f2fdd7a0801bf90af8b`.
All runtime and verifier scripts for this audit are Git-ignored under
`build/q8-integrated-arithmetic-20261009/` or the new SSD evidence directory.

All eight cases completed their own 327-node capture and independent
same-operand verification against their exact GGUF weights and corrected
native RHS staging. The per-case reports passed their unchanged gates and
matched each case's corrected-library development output outside runtime
fields. A separate read-only aggregate rehashed 13,080 raw files and 54
identity files, checked stable weights at 327 nodes, eight distinct RHS
signatures, and 2,616 node instances covering 1,291,560,448 F64 reference
dots and 20,897,248 staged blocks. Worst relative L2 was
`1.3447641567e-7`, below the `2e-5` gate. The ignored
`rounded-q8-arithmetic/eight-q8-reconciled.json` SHA-256 is
`7e19f7c147dbeba1cfeac60309a71f83cb405ee92fb81a2c2953af26015cee03`.
The raw captures, staging payloads, per-case reports and aggregate are
retained unmodified on the SSD; this audit uses about 17 GiB. No final
evaluation or reserve image was used. The remote desktop remained inactive
and the GPU was empty between cases. This is a **Q8 matrix component PASS**,
not whole-recipe arithmetic or deployment qualification.

The SSD has about 137 GiB free after retaining these receipts. The older
eight-input full-graph raw corpus uses about 487 GB. Representative F32
payload samples from its largest elementwise and normalization groups
compressed only to about 91–92% of their original size with fast zlib, so
simple lossless compression is unlikely to fit an equally complete second
raw corpus on the remaining volume. A later full-graph audit needs either
additional evidence capacity or a replayable bounded capture/verification
scheme whose compact receipts are independently checked before disposable
raw intermediates are removed. Do not infer a full-graph PASS from this
component result.

The completed cycle passed the isolated Python tool suite **171/171**, the
bilingual/local-link checker on **116** documents and `git diff --check`.
The previously verified corrected CUDA build remains unchanged. The
captured raw artifacts and verified build are retained; no new disposable
compiler or download cache was created.
