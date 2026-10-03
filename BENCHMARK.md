# Benchmarks

[中文](BENCHMARK_zh.md) · [Models and precision](MODEL_ZOO.md#precision-contract)

Apple M4 Pro (14 CPU / 20 GPU cores, 48 GiB), macOS 27.0, Release
AppleClang 21.0.0, GGML 0.25.3 with the pinned precision/window patch.
CPU requests four GGML/BLAS threads; processes request `VECLIB_MAXIMUM_THREADS=4`.
Accelerate manages SGEMM threading. Times are seconds; RSS uses decimal GB.
**Labels describe GGUF weight storage, not end-to-end arithmetic.**

## Image

One 1800×1200 truck image, 1008×1008 model input, prompt `truck`.
One warmup plus five full-image samples; load/decode/file writes excluded.

| Weight storage | CPU median / peak RSS | Metal median / peak RSS |
| --- | ---: | ---: |
| Mixed F16/F32 | 7.338 / 4.918 | 5.354 / 2.661 |
| F32 | 7.316 / 4.922 | 5.364 / 4.248 |

All four configurations pass the seven-case image reference suite.

## Video: 64-frame protocol

Default hybrid `visual-tracker-f32-v1`, 64 frames at 1800×1200, `truck`,
`max_objects=8`; 16 warmup + 48 measured, final drain included, no tensor dumps.
All cells ran serially on AC power with no system sleep or overlapping inference.

| Objects | CPU median / p95 / peak RSS | Metal median / p95 / peak RSS |
| ---: | ---: | ---: |
| 1 | 9.155 / 9.352 / 5.212 | 7.550 / 7.595 / 4.193 |
| 4 | 13.887 / 14.555 / 5.356 | 13.166 / 13.370 / 4.340 |

The official hotstart policy emits frame0 after processing frame14. Observed
first-output times: CPU 130.681/178.184 s; Metal 103.536/155.065 s (1/4 objects).
Finite measurements do not establish unlimited-stream memory bounds or real-time throughput.
F32/hybrid CPU/Metal numerical acceptance passes all five cases (216 frames each).
F16 video remains diagnostic and has no accepted performance row.

## Precision and profiling

Hybrid restores original FP32 visual/tracker weights; other weights remain
mixed F16/F32. CPU promotes stored half weights to F32; Metal retains mixed
weights and requests the specified F32 arithmetic. Video normalization rounds
to F16; tracker features round to BF16 in F32 buffers; mask-memory records are
BF16 and expanded to F32 for computation. Image inference has no temporal BF16
state. See the [full precision map](MODEL_ZOO.md#precision-contract).

Current CLI JSON exposes existing `runtime.image_ms` (ViT/necks/geometry) and
`runtime.inference_ms` (prompt/fusion/detection/masks), including transfers and
allocation. `text_ms` is the last text encode: once per video session; image
replacement encodes text again, already included in image `inference_ms`.
Current stage distributions are in the detail record; historical missing times
remain unavailable. `frame_ms`, `tracker_ms`, `memory_ms` retain their meanings.

## Reproduction and evidence

Use the [portable workflow](docs/validation.md) for fixture generation, original
Meta qualification/reuse, validation, profiling and private archive verification.
The [current M4 Pro record](docs/benchmarks/m4-pro-blas-vit-20261003.md) retains
stage samples, RSS, conditions, hashes and validation scope. The
[historical record](docs/benchmarks/m4-pro-20261003.md) preserves earlier measurements.
The [baseline index](docs/validation-baselines/blas-vit-m4pro-20261003.json) identifies
accepted evidence; model/media/raw files stay outside Git.

Relative to the same-protocol historical CPU video rows, observed speedups are
5.58×/5.88× (1/4 objects). These batches were not interleaved A/B measurements.
Four-object Metal remains dominated by tracking (7.277 s versus 5.043 s encoding).
