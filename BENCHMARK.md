# Performance

[中文](BENCHMARK_zh.md) · [Models and precision](MODEL_ZOO.md)

Measurements are specific to the model, input, hardware, backend, and software
configuration below. They describe the currently implemented SAM 3 adapter;
other models and platforms will have separate result tables as they are added.
Weight precision names storage, not every computation or state format.

## SAM 3 image segmentation

Environment: Apple M4 Pro, 14 CPU cores, 20 GPU cores, 48 GiB memory; macOS 27.0;
Release AppleClang 21.0.0; patched GGML 0.25.3. CPU requests four threads and
`VECLIB_MAXIMUM_THREADS=4`; Accelerate manages its own SGEMM threading.

Input: 1800 × 1200 RGB PPM, prompt `truck`, threshold 0.5, model input
1008 × 1008. One warmup, then five full-image calls with image replacement.
Latency excludes model loading, decoding, and file output. Peak RSS covers the
whole process, including loading and final mask output. Units are seconds and
decimal GB. Measurements ran sequentially on AC power.

| SAM 3 weights | CPU/BLAS time / peak RSS | Native CPU time / peak RSS | Metal time / peak RSS |
| --- | ---: | ---: | ---: |
| F32 | 7.353 / 4.928 | 38.857 / 4.833 | 5.373 / 4.245 |
| Mixed F16/F32 | 7.375 / 4.954 | 39.007 / 4.901 | 5.349 / 2.661 |
| Vision Q8_0 | 7.352 / 3.751 | 39.234 / 3.543 | 5.322 / 2.937 |
| Vision Q6_K | 7.348 / 3.678 | 38.851 / 3.676 | 5.348 / 2.868 |
| Vision Q5_K | 7.364 / 3.641 | 39.230 / 3.645 | 5.370 / 2.829 |
| Vision Q4_K | 7.360 / 3.606 | 39.180 / 3.413 | 5.339 / 2.793 |

Vision quantization reduces peak RSS relative to F32 in this workload, with
similar warmed latency. Metal F16 uses less memory than the vision-quantized
models. CPU F16 is promoted to F32, so its resident memory remains comparable
to F32. See [quantized model use](docs/quantization.md) for profile selection.
These small, sequential samples do not establish a general speedup or accuracy
on other inputs.

## SAM 3 video tracking

The same hardware and software configuration, with hybrid
`visual-tracker-f32-v1` weights. Input is 64 frames at 1800 × 1200, prompt
`truck`, `max_objects=8`; 16 warmup frames and 48 measured frames. Final drain
is included, with no tensor dumps. Each process runs sequentially on AC power.

| Objects | CPU/BLAS frame median / p95 / peak RSS | Metal frame median / p95 / peak RSS |
| ---: | ---: | ---: |
| 1 | 9.155 / 9.352 / 5.212 | 7.550 / 7.595 / 4.193 |
| 4 | 13.887 / 14.555 / 5.356 | 13.166 / 13.370 / 4.340 |

The adapter buffers output during hotstart. Measured time to the first output
was 130.681/178.184 seconds on CPU and 103.536/155.065 seconds on Metal
(one/four objects), including loading and startup. Steady-state frame time is
not startup latency or a real-time guarantee. Longer-running streams need
application-specific memory and behavior checks.

## Measure your workload

Use a Release build and the same decoded pixels, model, backend, thread request,
and power conditions for comparisons. The image CLI's `--repeat` reports full
image calls and cache hits separately. For video generation, reference
comparison, and timing commands, see [model verification](docs/validation.md).

Runtime weight/compute buffers describe allocations; process RSS describes
resident process memory. Do not add the two measurements together. Stage
medians also cannot be added to reconstruct the total median.

The corresponding plans retain [image measurement details](docs/plans/20261003-230244-quantized-image-performance-acceptance.md#english-performance-record)
and [video profiling details](docs/plans/20261003-134534-visual-encoding-profile-and-optimization.md#english-profiling-record).
