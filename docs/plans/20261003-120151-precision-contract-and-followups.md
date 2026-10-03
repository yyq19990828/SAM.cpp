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
