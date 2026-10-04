# Performance

[中文](BENCHMARK_zh.md) · [Models and precision](MODEL_ZOO.md)

Measurements dated 2026-10-04 describe the currently implemented SAM 3 adapter
on the configuration below. Other models and platforms will have separate
result tables as they are added. Weight precision names storage; computation
and state formats follow the model/backend precision policy.

## SAM 3 text-prompted image segmentation

Environment: Apple M4 Pro, 14 CPU cores, 20 GPU cores, 48 GiB memory; macOS 27.0;
Release AppleClang 21.0.0; patched GGML 0.25.3. CPU requests four threads and
`VECLIB_MAXIMUM_THREADS=4`; Accelerate manages its own SGEMM threading. Native
CPU disables BLAS and Accelerate. Measurements ran sequentially on AC power.

Weights are image GGUF exports of the official SAM 3 `sam3.pt` checkpoint.

Input: 1800 × 1200 RGB PPM, prompt `truck`, threshold 0.5, model input
1008 × 1008. One warmup, then five full-image calls with image replacement.
The table reports median latency and whole-process peak RSS, in seconds and
decimal GB. Latency excludes model loading, decoding and file output; RSS
includes loading and final mask output. Result-cache hits are timed separately
and excluded from these medians.

| SAM 3 weights | CPU/BLAS time / peak RSS | Native CPU time / peak RSS | Metal time / peak RSS |
| --- | ---: | ---: | ---: |
| F32 | 7.321 / 5.049 | 39.598 / 5.025 | 5.401 / 4.363 |
| Mixed F16/F32 | 7.352 / 5.060 | 39.500 / 5.028 | 5.408 / 2.787 |
| Vision Q8_0 | 7.410 / 3.964 | 39.597 / 3.953 | 5.363 / 3.054 |
| Vision Q6_K | 7.379 / 3.899 | 39.625 / 3.855 | 5.388 / 2.984 |
| Vision Q5_K | 7.402 / 3.873 | 39.600 / 3.820 | 5.417 / 2.944 |
| Vision Q4_K | 7.435 / 3.825 | 39.695 / 3.785 | 5.381 / 2.910 |
| Full-component Q8_0 | 7.383 / 3.015 | 39.610 / 2.981 | 5.360 / 2.081 |
| Full-component Q6_K | 7.407 / 2.853 | 39.660 / 2.835 | 5.392 / 1.929 |
| Full-component Q5_K | 7.427 / 2.770 | 39.668 / 2.773 | 5.420 / 1.850 |
| Full-component Q4_K | 7.431 / 2.698 | 39.754 / 2.672 | 5.386 / 1.770 |

Vision presets quantize eligible ViT linear weights. Full-component presets
also cover eligible text, fusion and decoder linears; embeddings, convolutions,
biases, normalization and designated small weights retain F32. See
[quantized model use](docs/quantization.md) for exact profile scope.

Quantization lowers peak RSS relative to F32 in this workload, while warmed
latency remains similar. CPU promotes F16 weights to F32 and uses temporary
F32 matrix weights for quantized operations; its memory use therefore does not
track GGUF size directly. Metal F16 uses less memory than the vision-quantized
models here. These five-call samples do not establish a general speedup or
accuracy on other inputs.

## SAM 3 tracker propagation

F32 and hybrid video exports of the same checkpoint use the same configuration
to compare the previous execution pipeline with graph
reuse, shared frame features and compatible-object batching. This real-weight
synthetic fixture has six spatial memories and sixteen pointers per object.
It measures memory conditioning and mask decoding, excluding frame encoding,
detection, external fixture generation, model loading and file output. Memory
and pointer preparation inside each tracker call remains included. These are
tracker-stage timings; complete video-frame latency must be measured separately.

Each cell uses one baseline/updated/baseline process sequence (A1/B/A2), with
a cold call followed by five matched warm calls in each process. Baseline time
is the median of the five matched A1/A2 averages; updated time is the B median.
RSS is the whole-process peak, including setup and the cold call. Units are
seconds and decimal GB. The four-object RSS column shows A1/A2 → B.

| Backend | Weights | 1 object, before → after | 4 objects, before → after | 4-object peak RSS, before → after |
| --- | --- | ---: | ---: | ---: |
| CPU/BLAS | F32 | 1.079 → 1.067 | 4.318 → 4.239 | 4.241/4.255 → 4.242 |
| CPU/BLAS | Hybrid | 1.078 → 1.059 | 4.337 → 4.243 | 4.240/4.242 → 4.258 |
| Native CPU | F32 | 7.203 → 7.199 | 28.736 → 29.079 | 4.255/4.248 → 4.187 |
| Native CPU | Hybrid | 7.215 → 7.160 | 28.880 → 28.587 | 4.245/4.255 → 4.194 |
| Metal | F32 | 0.960 → 0.946 | 3.849 → 3.119 | 4.571/4.576 → 4.233 |
| Metal | Hybrid | 0.959 → 0.945 | 3.850 → 3.116 | 3.881/3.876 → 3.541 |

Metal four-object propagation latency fell by about 19% for both F32 and hybrid,
with lower process peak RSS and identical outputs in this comparison. CPU
results changed by only a few percent and do not establish an acceleration
benefit. The small sample count and this fixture's memory shape limit the claim
to the measured workload. The existing F32/hybrid precision and temporal
output policy are preserved.

## Measure your workload

Use a Release build and fix decoded pixels, model, backend, thread request and
power conditions. The image CLI's `--repeat` reports full-image calls and cache
hits separately. For video generation, original-reference comparison and
complete-video timing commands, see [model verification](docs/validation.md).

Runtime weight/compute buffers describe allocations; process RSS describes
resident process memory. Do not add the two measurements together. Stage
medians also cannot be added to reconstruct the total median.

The [implementation and measurement plan](docs/plans/20261004-040240-sam3-pipeline-execution-and-unified-benchmarks.md) retains experimental
procedures, complete stage/boundary results and machine-readable evidence.
Earlier full-video measurements remain in the
[previous profiling plan](docs/plans/20261003-134534-visual-encoding-profile-and-optimization.md#english-profiling-record).
