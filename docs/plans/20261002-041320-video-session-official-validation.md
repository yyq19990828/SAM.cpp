# Complete the SAM 3 forward video session and official validator

Created: 2026-10-02 04:13:20 Asia/Shanghai.
Status: video integration implemented and validated; complete M2 acceptance remains blocked by recorded corpus/numerical failures.
Baseline: `2ddee1b` plus the retained local image-validation documentation and
video FP16 score-projection fix from the preceding validation batch.

## Scope

Complete the remaining implementation in the
[M2 plan](20261001-115321-sam3-text-video-tracking.md): `VideoSession`, tracker
stage execution, forward association/lifecycle, delayed owned results, video CLI,
and official video reference export/validation. Its API, precision boundaries,
temporal policy, retention bounds, fixtures and frozen numerical gates remain
authoritative. The image prerequisite and converted original full files are
already verified in the [local validation record](20261002-032709-local-model-meta-validation.md).
Preserve all previous changes and artifacts; no commit or push is included.

## Approach and steps

1. Trace the pinned Meta single-rank detector/tracker flow and existing GGML
   stage interfaces. Prepare an isolated, hash-recorded CPU video source copy
   with FP32 arithmetic and explicit F16/BF16 transport/storage boundaries.
   Add only required CPU connected-component dependencies to the reference lock.
2. Export original-weight stage tensors and temporal/output traces. Verify fixture
   behavior before making any claim about the video corpus. Reuse original
   detector/tracker modules; do not substitute a community tracker or per-frame
   detector matching for memory-based propagation.
3. Connect prompt preparation, conditioned memory, SAM decoding, object pointers
   and memory encoding to shared GGML graphs. Preserve logical birth groups and
   group-quality semantics while executing object tensor work serially.
4. Port the selected Meta forward lifecycle, reconditioning, capacity and overlap
   rules, integrate the verified retention selectors, and drain delayed outputs.
   Add the small task contract and public facade using the existing image pattern.
5. Add `sam_video` for contiguous PNG directories, exclusive output publication,
   completion receipts, masks/IDs, statistics and optional tensor dumps. Implement
   `export_video_reference.py` and `validate_video.py` with complete provenance,
   fixed ID mapping, stable-case decisions and unchanged numerical gates.
6. Run behavior/numerical regressions, independent/repeated headers, two-TU and
   downstream checks on CPU and Metal, plus Python tooling. Compare real original
   tracker stages before the full sequence matrix. Diagnose the first divergent
   stage rather than narrowing weights or weakening tolerances.
7. Run the original video corpus and image regressions where applicable. Record
   real placement, session/reset/lifetime behavior and retained-state bounds.
   Publish only demonstrated support; distinguish implemented tools, stage
   acceptance, full sequence acceptance and performance evidence.

## Verification and results

Use fresh unsynchronized build/output directories under `/private/tmp` and keep
receipts in ignored `build/video-validation/20261002-041320/`, avoiding the old
generated Metal cache's observed file-read blocker. Keep dependency revisions,
the verified GGML precision patch, image weights, references and gates unchanged.
No new C++ dependency, generic backend, model family or live-stream API is needed.

## Implemented integration

The public text-video contract/facade, ordered finite input, failure/reset
recovery and owned delayed results are implemented. The SAM 3 adapter now
executes the original conditioned memory, SAM mask decoder, object pointers and
memory encoder, preserving logical birth groups and explicit BF16 state. The
session retains bounded history and prunes retired metadata; text is encoded
once, while both image necks share a single trunk per frame.

The temporal port preserves the inspected branch/order details: admission uses
all post-NMS >0.5 detections when no tracks exist, and >=0.7 for later births;
track matching precedes ambiguity clearing; periodic correction tests the raw
tracker logit >0.8; correction updates records but emits pre-correction masks.
Normal batch IoU [B] and object logit [B,1] broadcasting differs from consolidated
[B,1] records, so group-quality computation retains that distinction.

`sam_video` reuses host PNG decoding and shared raw-tensor publication. It writes
exclusive output, an incomplete receipt before work, per-frame binary masks/IDs,
selection/lifecycle traces and stats, and optional 0/1/16/last-stage snapshots.
Only complete final draining marks success. `export_video_reference.py` calls
pinned original modules, verifies every input hash and scenario behavior, and
records checkpoint/source/runtime/package/storage identities. `validate_video.py`
validates the full original conversion, fixed birth-ID mapping, every output,
frozen stage/score/box/mask gates, exact temporal traces, retained-state limits
and Metal graph placement. Subsets never establish full milestone acceptance.

The isolated CPU source records all device/import/precision adaptations. It
preserves original explicit BF16 memory and tracker transport, promotes consumers
to FP32, disables persistent CUDA autocast and keeps temporal rules unchanged.
The original CPU connected-component fallback failed on a zero-object batch;
the prepared copy now returns correctly shaped empty int64 labels/counts.
This adaptation is hashed and documented; the original checkout remains clean.

## Current verified results

The original two-frame motion diagnostic passes FP32/CPU, FP32/Metal, FP16/CPU
and FP16/Metal. All stage inventories, tokens, temporal selections, IDs, masks,
scores, boxes and final draining match within unchanged gates. The initial Metal
run exposed ten CPU nodes: SAM decoder head-16 flash attention has no matching
pinned Metal kernel. Shared F32 matmul/softmax resolves this without changing the
GGML revision/patch; the repeated diagnostic has zero CPU graph nodes. A small
independent multi-head computation checks head-16 softmax/layout numerics.

Full 48-frame motion comparison on FP16/Metal passes: minimum mask IoU
0.9999874882, maximum stage normalized L2 0.0044535036 (<0.02), exact memory/pointer
order and lifecycle, 48 accepted/emitted frames, one text encode, 48 trunk encodes,
zero CPU nodes, 26 maximum retained records and 15 maximum in-flight pending
results. After ordinary emission the queue has <=14 entries and ends empty.
The compute-buffer high-water mark stays 1,245,268,288 bytes. These are acceptance
statistics from concurrent validation, not a video performance benchmark.

CPU and Metal CTest pass 11/11; downstream consumers pass 3/3 each, including
independent/repeated headers and two-TU linkage. Python tools pass 15/15,
including fail-closed provenance, fixed video birth mapping, invalid output and
Metal fallback rejection. Real checkpoint-backed sessions pass FP32/CPU,
FP32/Metal and FP16/Metal for model lifetime, interleaved independent sessions,
owned results, reset and repeatability. The extended Metal check also rejects
duplicate/skipped/negative indices, changed resolution and mid-sequence prompts
without running a new graph. The fake-weight failure check covers reset recovery
and an INT32_MAX declared count without preallocating sequence state.

The fresh image regression matrix passes 28/28 original cases across all four
precision/backend configurations. The full official exporter and FP16/Metal CLI
processed all 216 frames. Each CLI completion, immutable binary hash and command
is retained. The original reference is complete but ineligible for M2:

| Case | Frames | Required official behavior | FP16/Metal differential gates | Maximum tensor normalized L2 |
| --- | ---: | --- | --- | ---: |
| Motion | 48 | Verified | Pass | 0.004454 |
| Entry | 64 | Failed: no late birth | Fail: memory/recurrence tensors | 0.109836 |
| Occlusion | 64 | Verified | Pass | 0.003476 |
| Hotstart removal | 24 | Failed: no object retired | Pass against the actual oracle | 0.003476 |
| Negative | 16 | Verified | Pass | 0.004667 |

Every output-frame mask/score/box gate passes, including entry's minimum mask
IoU 0.993724 (>0.95). Exact temporal traces and zero CPU fallback pass all five
cases. Retained records peak at 26/25/27/15/0 per case; the queue peaks at 15 and
drains completely. These successes do not override the entry tensor failure or
the two missing official scenario behaviors.

Entry first exceeds the FP16 tensor limit at frame 16 memory encoding:
normalized L2 0.049065 (>0.02); frame 63 conditioned features reach 0.109836.
FP32/Metal rerunning the same first 17 frames passes every frame-16 stage;
memory normalized L2 is 0.000052734 (<0.001). The original and FP16 propagated
decoder both choose candidate 1 at frame 16: original IoUs 0.992402/0.981659/0.992297,
FP16 0.992298/0.977411/0.992248. Thus candidate-rank flipping is ruled out for this
first mismatch. The precise propagated-mask/storage/recurrence cause remains
open; no weights, numerical limits or policy thresholds were changed to hide it.
Temporary diagnostic printing was removed and the clean binary hash matches the
verified final smoke executable.

The original entry recipe
produced only ID 0 for all 64 official frames, so it does not exercise the required
late birth. A separate original-weight diagnostic with two separated 800x533
tiles and a mirrored second tile entering at frame 2 produces IDs 0 and 1 while
retaining ID 0. The original hotstart recipe retains ID 0 through all 24 frames,
so its removal/delayed-suppression requirement is also unverified. The stock
forward policy counts only nonempty unmatched predictions toward retirement;
disappearing input alone does not guarantee removal. The exploratory entry result
is not a replacement frozen corpus or
full acceptance. The original recipe and its failed behavior evidence are
preserved; any fixture revision needs its own recorded proof before acceptance.

The validator's default full-acceptance guard rejects the complete but ineligible
reference before running model comparisons. A labeled diagnostic comparison
retains all failures; it cannot mark this corpus accepted. Final timing includes
preprocessing in total frame time and accounts for seed/correction decoding in
tracker time without counting it again in memory time. Targeted CPU/Metal smoke,
real Metal session and CTest checks pass after that instrumentation update;
the 216-frame numerical receipts archive the earlier immutable executable and
remain acceptance diagnostics, not benchmark measurements.

Receipts are in ignored `build/video-validation/20261002-041320/`; original
references and C++ outputs are under `/private/tmp/sam-video-20261002-041320/`.
The full CPU video matrix and controlled 64-frame one/four-object benchmarks
remain pending. The next acceptance work must verify corrected scenario recipes
against the original oracle, resolve the first entry tensor divergence, and then
complete the matrix/performance runs. Complete M2 support requires all corpus/matrix/behavior and
performance gates; implementation or a passing motion diagnostic cannot replace
them. No commit or push is included.

The [subsequent numerical/fixture record](20261002-100625-video-numerics-fixtures.md)
preserves this earlier corpus, exposes pre-correction propagation and actual
memory inputs, and verifies revised full-length entry/hotstart behaviors with
the original checkpoint. Its new receipts are separate from this run. The
earlier entry numerical stress failure is still retained rather than relabeled
as accepted by changing recipes.
