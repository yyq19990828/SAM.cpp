# M4 Pro: ViT projection and CPU BLAS measurements

[中文](m4-pro-blas-vit-20261003_zh.md) · [Current summary](../../BENCHMARK.md)

Measured on 2026-10-03, Apple M4 Pro (14 CPU/20 GPU cores, 48 GiB), macOS 27.0
(26A428), Release AppleClang 21.0.0 and pinned GGML 0.25.3
`353b63b439f27ab2cc19dac97ab1681ba6d2d084` with the existing precision/window patch.
Implementation base is main 56a4cde plus uncommitted changes; exact source/binary/
library/input identities are bound by receipts, rather than inherited from HEAD.

Before commit, the BLAS counter moved to the end of the public stats aggregate
for source compatibility, and archive verification gained complete inventory/size
checks. The index records this metadata-only delta and focused CPU/Metal parity;
the measurements below remain bound to their immutable pre-fix binaries.

Original checkpoint SHA-256:
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
Video uses hybrid `visual-tracker-f32-v1`, GGUF SHA-256
`3975b4b1a10b962c6fad5022e2798baa6266aad7dbcc39210b459537208e1a05`.
Image uses the schema1 F32/F16 files identified in [Model Zoo](../../MODEL_ZOO.md).
Weight labels do not describe the whole pipeline; the existing F16 normalization
and BF16 feature/memory boundaries are unchanged.

The historical CPU build compiled BLAS but its scheduler omitted that ACCEL
device; native CPU already used Accelerate vector primitives. The new
CPU execution retains weight ownership and schedules an available registry BLAS accelerator
before native CPU for eligible matrices. `blas_nodes` is a subset of `cpu_nodes`.
ViT shared-weight channel positions are folded into GEMM columns. Metal keeps its
selected/fallback order and has zero CPU graph fallback in accepted runs.
GGML/BLAS receives four-thread requests; processes request
`VECLIB_MAXIMUM_THREADS=4` before startup. Accelerate manages SGEMM threads itself.

## Image: one warmup and five complete calls

Original 1800×1200 `truck.jpg`, prompt `truck`, threshold 0.5, model input 1008×1008.
Loading, decoding, file writing and repeated-result-cache calls are excluded from
warmed latency. Peak RSS comes from the whole `/usr/bin/time -l` child process.
Seconds / decimal GB:

| Weight storage | CPU median / peak RSS | Metal median / peak RSS |
| --- | ---: | ---: |
| Mixed F16/F32 | 7.338 / 4.918 | 5.354 / 2.661 |
| F32 | 7.316 / 4.922 | 5.364 / 4.248 |

## Video: 64 frames, 16 warmup, 48 measured

Independent qualified one/four-object fixtures, 1800×1200, prompt `truck`,
max_objects=8, no tensor dumps, final drain included. p95 uses linear interpolation
at `(n-1)*0.95`. Peak RSS includes loading and final drain.

| Objects | CPU median / p95 / peak RSS | Metal median / p95 / peak RSS |
| ---: | ---: | ---: |
| 1 | 9.155 / 9.352 / 5.212 | 7.550 / 7.595 / 4.193 |
| 4 | 13.887 / 14.555 / 5.356 | 13.166 / 13.370 / 4.340 |

Frame 0 appears after processing frame 14. First output-file observations, sampled
about once per second, include load/decode/file work: CPU 130.681/178.184 s;
Metal 103.536/155.065 s (1/4 objects). The temporal policy is unchanged.

### Stage medians, seconds

| Backend | Objects | Image encoding | Detector pipeline | Tracker | Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpu | 1 | 6.448 | 1.105 | 1.480 | 0.075 |
| cpu | 4 | 6.481 | 1.109 | 5.932 | 0.309 |
| metal | 1 | 5.047 | 0.624 | 1.793 | 0.039 |
| metal | 4 | 5.043 | 0.624 | 7.277 | 0.161 |

Image encoding includes ViT, necks, geometry, transfers and allocation. Detector
pipeline includes prompt preparation, fusion, detection and masks. Video encodes
text once per session; replacing an image encodes text again, included in image
`inference_ms`. Independent stage medians cannot reconstruct the full-frame median.
Metal four-object tracking remains the largest measured stage.

All video cells retain constant weight/compute high-water after warmup: CPU
3447558112/1020660736 bytes; Metal 2763224992/1245268288 bytes. Retained memory is
bounded in this protocol by 27 records per object: 1/4 objects reach 27/108 records,
17943552/71774208 bytes. Allocation counters are not added to process RSS.
Asynchronous current-RSS samples are retained with frame tags in each receipt.
Finite observations do not prove unlimited-stream RSS bounds or real-time throughput.

## Conditions, acceptance and preserved failures

Video cells ran serially 17:09:17–17:54:31 Asia/Shanghai; four fresh image cells
followed the same serial policy. All power endpoints were AC/100%, no thermal or
performance warning was recorded, and live power-log audit found no system sleep
inside a measured cell. Task-scoped caffeinate prevented idle sleep. No other
SAM/Meta inference or compilation overlapped measurements; desktop services stayed
active and clocks were not forced. Source/model/binary/library/input/output hashes
were verified after completion: 373 unique identities and 104 output trees.

Fresh acceptance passes 864 video frames (F32/hybrid × CPU/Metal, 216 each), 70 image
cases, six real short sessions and two 128-push interleaved long checks. Long positive
outputs/history match their accepted standalone entry64 exactly; negative state
stays empty, model lifetime/owned outputs/session isolation pass. Peak long-process
RSS is 5.555 GB CPU and 4.595 GB Metal; weight/compute high-water stays constant after
warmup. These are finite checks, not a universal memory bound.

The extra v2 image benchmark assertion mistakenly compared stb-decoded JPEG with
Pillow-exported reference PPM. 163934/6480000 RGB channel values differ (max 3).
Same-JPEG single and repeated output are exactly equal. V3 preserves the exact
comparison using an identical-JPEG single-inference bridge; the bridge is not a
new original-Meta oracle. The failed v2 receipt and pre-inference v1 counter
correction remain archived. Official numerical/candidate gates were not relaxed.
Explicit F16 video remains diagnostic, including its historical entry23/24 failure;
full CPU F16 remains deferred.

The [historical record](m4-pro-20261003.md) retains previous measurements. Relative
to its same-protocol CPU video rows, observed speed ratios are 5.58×/5.88×.
The batches were not interleaved A/B runs, so small Metal differences need caution.
Follow [validation](../validation.md) to regenerate fixtures, qualify or explicitly
reuse unchanged original-Meta prefixes, and run only matching accepted binaries.

Raw receipts: `build/p1-p2/20261003/reaccept/`; sealed summary: `final-evidence.json`.
The [new baseline index](../validation-baselines/blas-vit-m4pro-20261003.json) binds
14664 files/43333663701 logical bytes in a private APFS clone bundle,
with unchanged original model/reference resources in the parent archive.
Byte preservation does not grant automatic acceptance to changed executables;
recorded paths/RPATHs may need restoration. Remote quick CI has not run. No new
commit or push was performed.
