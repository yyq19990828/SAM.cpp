# Precision contract and remaining follow-ups

Created: 2026-10-03T12:01:51.632646+08:00.
Baseline: local main `56a4cde`; clean at start. Documentation only.
Status: complete.

## Scope and approach

Clarify existing GGUF weight storage, backend resident weights/arithmetic,
FP16 video preprocessing and BF16 feature/memory boundaries in MODEL_ZOO and
BENCHMARK. Record prioritized follow-ups from completed acceptance evidence.
Do not change runtime, weights, thresholds, measurements or commit/push state.

## Steps and verification

1. Read the current converter, backend loader, graph precision patch, video
   preprocessing, feature transport and memory encoder/reader.
2. Add an authoritative precision map to MODEL_ZOO and link the benchmark
   conditions to it. Clarify image versus video applicability and physical
   buffer type versus rounding semantics.
3. Check source-backed statements, document links and `git diff --check`.
   Reuse completed numerical/performance evidence; no model tests are needed.

## Prioritized remaining work

| Priority | Item | Evidence / next action |
| --- | --- | --- |
| P1 | Throughput and first-output latency | Hybrid Metal medians 7.833/13.356 s per frame for 1/4 objects; CPU 51.102/81.644 s. First output follows frame14 (108.750/158.408 s observed on Metal). Profile visual/detector stages separately before changing kernels, object batching or host transfers. Preserve numerical and temporal gates. |
| P1 | Durable reusable acceptance bundle | Reference/build/raw output dependencies include ignored build/ and /private/tmp paths. Preserve a small versioned baseline index and durable private raw artifacts before treating them as reusable after cleanup/reboot. Keep weights/private media out of Git. |
| P2 | Portable benchmark qualification tooling | Current one/four fixture and original-prefix qualification scripts are in the local ignored receipt bundle. Promote the minimum source-only recipes/qualification commands to normal tools when clean-clone reproduction is required; generated media and weights remain external. |
| P2 | Explicit F16 video limitation | Metal entry23/24 exact candidate mismatch is reproduced inside Meta using identical rounded weights. Retain diagnostic status; the deferred CPU216 run is optional and cannot repair this rounding boundary. Reconsider storage allocation only if a smaller accepted model is required. |
| P2 | Automated quick checks | No tracked .github workflow currently exists. Add weight-free CPU/header/tool checks if remote CI is wanted; keep heavy model acceptance manual/optional and reuse matching baselines. |
| P3 | Expanded validation/support scope | Current performance/long-session evidence is finite and from M4 Pro/macOS. Longer continuous workloads, other hardware/OS, CUDA, quantized formats, reverse/interactive video and SAM3.1/other model families need separate scope and matching-hardware acceptance. |

The default F32/hybrid CPU/Metal M2 acceptance is complete; these are next
work items or explicit limits, not a new failure in the supported matrix.
Local main is ahead of origin/main by two commits; remote publication remains
an explicit delivery action, separate from runtime defects.

## Results

MODEL_ZOO now separates on-disk/resident weights and video stage precision.
BENCHMARK labels weight storage explicitly and records CPU/Metal arithmetic,
FP16 normalization and BF16 feature/memory boundaries. The changelog notes this
clarification. Source readback confirms BF16 memory records, FP32 pointer vectors,
BF16-rounded feature vectors and FP16-rounding preprocessing.

Checks pass: video metadata retains bf16/bf16/f16-normalize declarations;
image metadata has no tracker BF16 declaration; all measured HTML table cells
and every GGUF artifact size/hash row are unchanged; all local doc link targets
exist. `git diff --check` passes. No runtime/model/measurement file was changed,
no model test rerun, and no commit or push was performed in this documentation task.

## User-document cleanup archive: architecture details (2026-10-04)

This appendix preserves implementation-specific precision and local acceptance
details removed from the user-facing architecture guide. The current public
support summary and quantization status live in their dedicated docs; these
values describe the completed local runs below.

### Image and quantized acceptance notes

- On the fixed seven-case corpus, the four vision-only schema-3 profiles pass
  output quality across CPU/BLAS, native CPU and Metal (84/84). Raw tensor
  fidelity is independent: 0/84 cells pass its full gate; 500/840
  tensor-by-case comparisons pass and 340 fail. This does not establish
  dataset-wide accuracy. The serial 18-cell performance matrix measured lower
  process peak RSS for all 12 quantized cells than same-backend F32; quantized
  Metal RSS exceeded Metal F16, and the medians did not establish a speedup.
- The full image graph executes 3,332 FP32 or 3,342 FP16 nodes on Metal in six
  partitions with no CPU graph fallback. Host preprocessing/postprocessing
  still run on CPU.
- Explicit FP32 Metal retained F32 weight storage and passed the seven original
  image cases plus the real-checkpoint session lifetime/cache checks locally
  on 2026-10-02. Auto FP32 continued to select CPU.
- Native Metal `WIN_PART` and `WIN_UNPART` handle contiguous F32 tensors,
  including padded windows. Direct GPU checks and full-model results were kept
  with the window/performance validation records.

### Implementation notes removed from the concise architecture overview

The video adapter shares the ViT trunk between detector and tracker necks and
rounds tracker feature transport through BF16. Its ordinary image path retains
the existing preprocessing and detector. Memory attention tiles 128 queries
at a time while retaining all spatial/pointer keys in each softmax. Forward
retention keeps hotstart group history; after group membership is fixed it
protects an eight-frame correction window, four conditioning records and 15
older eligible records, and preserves the number of pruned conditioning
records for the official ordering policy. This is forward-only behavior, not a
reverse/editable-session policy.

`tracking/session.hpp` integrates logical birth groups, quality recomputation
after hotstart removal, association, periodic reconditioning, overlap
suppression and a 15-frame delayed queue. Tensor work runs serially per object;
group membership and original batch-quality broadcasting semantics stay
intact. `tracking/execution.hpp` connects conditioned memory, temporal pointers,
SAM decoding and BF16 memory encoding to shared graphs. SAM decoder head-16
cross-attention uses F32 GGML matmul/softmax because the pinned Metal flash
implementation does not support that head size; head-32 self-attention keeps
the precise flash path.

The public video facade retains the model and owns prompt/temporal state and
returned results. It enforces fixed finite forward input and requires reset
after execution failure. `sam_video` decodes host PNGs, publishes exclusive
output and writes an incomplete/complete receipt. The source adapter, official
exporter and differential validator record F16/BF16 boundaries and retain
Meta's association/lifecycle modules. The original CPU component fallback
requires an empty-batch adaptation when no object is born. Snapshots preserve
propagated conditioned features and selected masks before periodic correction,
plus the normalized mask passed to memory encoding, so correction cannot
overwrite recurrence diagnostics.

Weight-free tests compare resized bytes and F16 normalization against Pillow
11.2.1, attention against an independent full-key computation, and 2,024 memory
selections against hash-verified pinned Meta functions. Checkpoint-stage,
temporal, session and backend-placement evidence is retained in the separate
video validation records.

### Precision detail moved from the SAM 3 model guide

GGUF precision labels describe stored weights, not every activation or video
state value. Image F32 uses F32 weights; image F16 stores selected multi-
dimensional weights in F16 and retains designated vectors/embeddings/constants
in F32. CPU promotes rounded F16 weights to F32 in memory; Metal keeps stored
half weights and stages the specified dense operations in F32. The video-only
`visual-tracker-f32-v1` profile restores original F32 values for the shared
visual trunk and tracker while retaining the mixed detector/text allocation.
This profile adds storage fidelity; it does not change temporal precision
boundaries.

| Video stage/state | Rounding/storage boundary | Runtime representation |
| --- | --- | --- |
| RGB normalization | FP16 rounding after scale, center and normalize | FP32 output/upload buffer contains FP16-rounded values |
| Tracker-neck feature transport | BF16 rounding | FP32 feature vectors contain BF16-rounded values; no native BF16 graph kernel is implied |
| Retained mask-memory features | BF16 storage | `ggml_bf16_t` records expanded to FP32 for memory attention |
| Object pointers and host mask logits | FP32 | FP32 vectors; returned binary masks are uint8 |

Therefore F32 video denotes F32 weights with explicit FP16 input and BF16
transport/memory boundaries, not an end-to-end FP32 sequence. Hybrid changes
weight allocation, not these boundaries. The original video reference also
used FP32 arithmetic with explicit FP16 inputs/BF16 storage, with autocast and
TF32 disabled.
