# CUDA Q8 MMQ extreme-scale diagnosis

Created: 2026-10-09 11:19:44, Asia/Shanghai.

## Scope

Diagnose the predeclared finite ±3e38 MMQ staging failure without altering
the frozen SAM 3 candidate, pinned GGML source/library, quality/performance
gates or retained evidence. Determine whether the `-use_fast_math` division
path can explain nonfinite scales and whether a controlled F32 division can
preserve finite scales for this synthetic boundary. This is a diagnostic,
not a new accepted recipe or a replacement for the failed receipt.

## Approach and steps

1. Record the exact frozen CUDA source expression and compiler flags. Build
   a small isolated CUDA kernel on the tested GPU with the same fast-math
   setting, comparing `127.0f / amax` and explicitly rounded F32 division
   for zero, ordinary, 1e30, 1e35, 1e37, 1e38 and 3e38 magnitudes. Retain
   raw outputs, executable/source hashes and a read-only receipt in ignored
   `build/` output. Do not feed a result into the frozen aggregate.
2. If the small kernel confirms the mechanism, evaluate the narrowly scoped
   next-candidate change and its need for new source identity, boundary,
   arithmetic, quality and exclusive performance validation. Do not patch
   the pinned library in place or relabel the current candidate.
3. In an isolated copy of `quantize.cu`, replace only the MMQ inverse-scale
   division with explicit round-to-nearest F32 division. Compile that unit
   with the frozen GGML headers/CUDA flags and link a fresh diagnostic probe
   against the existing runtime. Check the original extreme input and a
   representative passing MMQ fixture with the independent staging
   validator. Treat this as a trial unit, not a qualified library or model.

## Verification

Check compilation/output determinism and finite/zero cases against F64
reference calculations. Run documentation/link and whitespace checks after
editing this plan; keep the remote-desktop service stopped and check GPU
occupancy before the diagnostic. Clean disposable compiler objects while
retaining source, executable and receipt needed to interpret the result.

## Results

The RTX 4090 had no other NVIDIA compute process, and the remote-desktop
service remained inactive. The frozen `quantize.cu` MMQ staging expression
is `d_inv = 127.0f / amax`, followed by `d = 1.0f / d_inv`; the frozen CUDA
build uses `-use_fast_math`. An isolated `sm_89` kernel built with the same
fast-math setting tested zero, one and magnitudes 1e30 through 3e38. Its
PTX contains `div.approx.ftz.f32` and `rcp.approx.ftz.f32` for the ordinary
expression, versus `div.rn.ftz.f32` for explicit `__fdiv_rn`. At the two
largest F32 inputs (approximately 1e38 and 3e38), the ordinary expression
returned inverse zero and scale `+inf`, whereas the rounded division
returned finite inverse and scale within 1e-6 relative error of F64.
Both paths produced the expected zero-block behavior. The ignored,
hash-bound receipt `build/q8-integrated-arithmetic-20261009/q8-mmq-division-diagnostic.json`
has SHA-256 `a6bd40f886d3e64f5d45570add4f6e99802154623c769e5a2f006667db3bd0e8`.
This reproduces the mechanism on the same GPU and compiler setting. It is
not a disassembly of the frozen GGML kernel itself.

An isolated copy of frozen `quantize.cu` was changed at that **single**
expression to `__fdiv_rn(127.0f, amax)`, compiled with the frozen GGML
headers and CUDA fast-math setting, and linked into a fresh MMQ probe. It
did not modify the frozen source, library or model. With the original
predeclared 65 x 160 finite ±3e38 input, the independent MMQ staging
validator passed 260 blocks and 1,040 subblocks, with zero scale/integer
errors. The same trial passed 512 blocks and 2,048 subblocks from the
retained `mlp-lin1` real-model RHS. Relative to that RHS's frozen native
payload, all signed integer bytes were identical and 448 F32 scales
changed by at most `3.7252902985e-9`. The ignored,
hash-bound `build/q8-integrated-arithmetic-20261009/candidate-mmq-division-trial.json`
has SHA-256 `c19989f2dfbd09c87a8824e08ad6f7c74c314edf5b7c424544011227f488ad5b`.

This identifies a narrow candidate repair, but the trial object is not a
new integrated GGML library. The frozen predeclared boundary remains
`FAIL`; its whole-recipe arithmetic and deployment statuses remain
`NOT_RUN`. Promoting a corrected implementation requires a separately
pinned source/library identity, direct boundary and full-graph arithmetic
checks, fresh quality evaluation and exclusive-GPU performance under the
same predeclared gate policy. No current result can be relabeled using this
trial.

The selected CUDA CTest suite passed 29/29, the isolated reference tool
suite passed 170/170, the bilingual documentation/link checker passed 114
documents, and `git diff --check` passed during this diagnostic cycle.
