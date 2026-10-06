# Performance

[中文](BENCHMARK_zh.md) · [Models and precision](MODEL_ZOO.md)

This page contains the latest complete measurements for the validated model
configurations. SAM 3 is the current adapter; additional models and platforms
will have their own tables when validated. Results apply to the stated workload
and hardware. Weight labels describe storage; computation and state follow the
model/backend precision policy.

## macOS measurement environment

Apple M4 Pro, 14 CPU cores, 20 GPU cores, 48 GiB memory; macOS 27.0;
Release AppleClang 21.0.0; patched GGML 0.25.3. All cells ran sequentially on
AC power. CPU requests four threads with `VECLIB_MAXIMUM_THREADS=4`;
Accelerate manages its own SGEMM threading. CPU/BLAS enables Accelerate,
native CPU disables BLAS/Accelerate, and Metal uses default execution settings.
All weights are GGUF exports of the official SAM 3 `sam3.pt` checkpoint.

## SAM 3 text-prompted image segmentation (macOS)

Measured: 2026-10-04, Asia/Shanghai. Ten weight configurations across three
backends; each passed original-reference output-quality checks.

Input: 1800 × 1200 RGB PPM, prompt `truck`, score threshold 0.5, model input
1008 × 1008. Each process runs one cold call and five uncached full-image calls
with image replacement. The table reports their warm median; cache replays are
excluded. Latency includes image/text encoding and segmentation, and excludes
model loading, decoding and file output. Peak RSS covers the whole process,
including startup and the cold call. GB is decimal (`10^9` bytes).

Each backend cell shows **inference seconds / peak RSS GB**.

| SAM 3 image weights | GGUF, GB | CPU/BLAS, s / GB | Native CPU, s / GB | Metal, s / GB |
| --- | ---: | ---: | ---: | ---: |
| F32 | 3.371 | 7.060 / 5.023 | 39.128 / 5.045 | 2.707 / 4.364 |
| Mixed F16/F32 | 1.798 | 6.911 / 5.070 | 39.121 / 5.027 | 2.669 / 2.787 |
| Vision Q8_0 | 2.065 | 6.980 / 3.967 | 39.349 / 3.927 | 2.615 / 3.054 |
| Full-component Q8_0 | 1.097 | 6.991 / 3.003 | 39.357 / 2.985 | 2.611 / 2.079 |
| Vision Q6_K | 1.995 | 6.956 / 3.895 | 39.444 / 3.857 | 2.640 / 2.985 |
| Full-component Q6_K | 0.947 | 6.995 / 2.873 | 39.434 / 2.830 | 2.626 / 1.929 |
| Vision Q5_K | 1.957 | 6.954 / 3.875 | 39.378 / 3.815 | 2.640 / 2.946 |
| Full-component Q5_K | 0.864 | 7.028 / 2.772 | 39.358 / 2.751 | 2.644 / 1.851 |
| Vision Q4_K | 1.920 | 6.966 / 3.864 | 39.322 / 3.785 | 2.613 / 2.909 |
| Full-component Q4_K | 0.787 | 7.086 / 2.715 | 39.356 / 2.674 | 2.671 / 1.775 |

Vision presets quantize eligible vision encoder linear weights. Full-component
presets also cover text, fusion and decoder linears. Embeddings, convolutions,
biases, normalization and designated small weights retain F32. CPU keeps
quantized weights compressed and uses temporary F32 matrices for operations.
See [quantized model use](docs/quantization.md) for precision and scope.

## SAM 3 forward text-prompted video tracking (macOS)

Measured: 2026-10-05, Asia/Shanghai. F32 and hybrid
`visual-tracker-f32-v1` video weights on CPU/BLAS and Metal, following
original-reference qualification for each model/backend pair.

Each cell processes a fixed 64-frame sequence of 1800 × 1200 RGB PNGs,
with prompt `truck`, one or four persistent objects, and an eight-object limit.
Model input is 1008 × 1008. The first 16 frames are warmup; statistics use the
remaining 48 frames. P95 uses linear interpolation at `(n-1)*0.95`.

Per-frame latency covers preprocessing, frame encoding, detection, tracking,
mask/result preparation and internal transfers/allocations inside
`VideoSession::push_frame`. Model loading, external PNG decoding and file output
are excluded. The existing 14-frame output delay is separate from processing
latency; the final output drain is included in the measured samples. Peak RSS
covers the whole process, including loading and file output. Units are seconds
per frame and decimal GB.

| Video weights | GGUF, GB | Backend | Objects | Median, s/frame | P95, s/frame | Peak RSS, GB |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| F32 | 3.449 | CPU/BLAS | 1 | 8.852 | 9.023 | 5.397 |
| F32 | 3.449 | CPU/BLAS | 4 | 13.503 | 14.573 | 5.480 |
| F32 | 3.449 | Metal | 1 | 4.224 | 4.298 | 4.682 |
| F32 | 3.449 | Metal | 4 | 7.914 | 8.160 | 4.795 |
| Hybrid | 2.765 | CPU/BLAS | 1 | 8.838 | 9.063 | 5.415 |
| Hybrid | 2.765 | CPU/BLAS | 4 | 13.463 | 14.249 | 5.390 |
| Hybrid | 2.765 | Metal | 1 | 4.246 | 4.368 | 3.989 |
| Hybrid | 2.765 | Metal | 4 | 7.967 | 8.179 | 4.087 |

## Linux CUDA measurements

Measured: 2026-10-07, Asia/Shanghai. Intel Core i7-12700KF, 20 logical CPUs,
32.606 GB OS-visible RAM; NVIDIA RTX 4090, 24,564 MiB VRAM, 500 W power limit;
Linux x86_64, kernel 7.0.0-31, GCC 15.2, NVCC 13.3.73 and driver 610.57.04.
Release builds use patched GGML 0.25.3. CPU/BLAS uses OpenBLAS 0.3.32 (pthread)
and requests four threads; CUDA disables BLAS and CUDA graph capture. All cells
run sequentially with the desktop active. These are separate measurements from
the macOS tables above.

### Text-prompted image segmentation

The same 1800 × 1200 RGB PPM, `truck` prompt and 1008 × 1008 model input use one
cold call, five uncached full-image calls and five separate cache replays.
The table reports the median and P95 of the five full-image calls, excluding
model loading, external decoding, file output and cache replays. Every listed
configuration passed all seven original-reference image cases before timing.
CPU/BLAS cells show seconds / peak RSS GB; a dash means this cell was not timed.

CUDA process RSS and GPU memory are separate. RSS covers the whole process;
GPU memory is sampled with `nvidia-smi` about once per second for the inference
process, including its context and pools. Sampling may miss transient peaks.
GB is decimal; P95 uses linear interpolation at `(n-1)*0.95`.

| Image weights | GGUF, GB | CPU/BLAS, s / RSS GB | CUDA median, s | CUDA P95, s | CUDA peak RSS, GB | CUDA sampled GPU, GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 | 3.371 | 16.981 / 4.674 | 0.837 | 0.839 | 0.901 | 4.758 |
| Mixed F16/F32 | 1.798 | 17.066 / 4.675 | 0.851 | 0.854 | 0.944 | 3.152 |
| Vision Q8_0 | 2.065 | — | 0.735 | 0.741 | 0.957 | 3.483 |
| Full-component Q8_0 | 1.097 | — | 0.739 | 0.741 | 0.972 | 2.517 |
| Vision Q6_K | 1.995 | — | 0.737 | 0.743 | 0.964 | 3.414 |
| Full-component Q6_K | 0.947 | — | 0.737 | 0.739 | 0.979 | 2.366 |
| Vision Q5_K | 1.957 | — | 0.732 | 0.739 | 0.964 | 3.376 |
| Full-component Q5_K | 0.864 | — | 0.737 | 0.738 | 0.978 | 2.284 |
| Vision Q4_K | 1.920 | — | 0.731 | 0.740 | 0.963 | 3.339 |
| Full-component Q4_K | 0.787 | — | 0.736 | 0.737 | 0.978 | 2.252 |

Quantized CUDA uses native packed-weight kernels with RHS Q8_1 staging and its
own arithmetic profile. Qualification measures final output quality; it does
not imply F32 tensor equivalence. See [quantized model use](docs/quantization.md).

### Forward text-prompted video tracking

F32 and hybrid `visual-tracker-f32-v1` each passed the complete five-case,
216-frame original-reference corpus before timing. Each cell uses the fixed
64-frame 1800 × 1200 RGB PNG workload, `truck` prompt, one or four persistent
objects and an eight-object limit. The first 16 frames are warmup; the remaining
48 frames provide the median and P95.

Per-frame latency covers `VideoSession::push_frame`, including preprocessing,
encoding, detection, tracking, masks and internal transfers/allocations. Loading,
external PNG decoding and file output are excluded. The existing 14-frame output
delay is separate; the final drain is included in the samples. RSS and sampled
GPU memory use the process-wide definitions above.

| Video weights | GGUF, GB | Objects | Median, s/frame | P95, s/frame | Peak RSS, GB | Sampled GPU, GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 | 3.449 | 1 | 1.177 | 1.254 | 1.163 | 4.895 |
| F32 | 3.449 | 4 | 2.005 | 2.198 | 1.311 | 4.895 |
| Hybrid | 2.765 | 1 | 1.176 | 1.254 | 1.161 | 4.222 |
| Hybrid | 2.765 | 4 | 2.008 | 2.217 | 1.302 | 4.222 |

## Measure your workload

Use a Release build and fix decoded pixels, model, backend, thread request and
power conditions. The image CLI's `--repeat` reports full-image calls and cache
hits separately. Video fixture generation, original-reference comparison and
complete-video timing commands are in [model verification](docs/validation.md).

Runtime weight/compute buffers describe allocations; process RSS describes
resident memory. Do not add them together. Stage medians cannot be added to
reconstruct the total median. Application inputs need their own quality and
performance checks.

Measurement receipts, qualification details and historical records are retained
in the [macOS performance measurement plan](docs/plans/20261004-214547-latest-complete-model-performance-records.md)
and the [CUDA implementation plan](docs/plans/20261006-215722-cuda-backend.md).
