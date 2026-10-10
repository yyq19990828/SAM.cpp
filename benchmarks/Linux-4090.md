# Linux · NVIDIA RTX 4090 performance

[中文](Linux-4090_zh.md) · [Performance index](../BENCHMARK.md) · [Models and precision](../MODEL_ZOO.md)

These existing SAM 3 measurements apply to the stated hardware and workload.
Weight labels describe storage; computation and state follow the model/backend
precision policy.

All weights are GGUF exports of the official SAM 3 `sam3.pt` checkpoint.

## Linux CUDA measurements

Measured: 2026-10-07, Asia/Shanghai. Intel Core i7-12700KF, 20 logical CPUs,
32.606 GB OS-visible RAM; NVIDIA RTX 4090, 24,564 MiB VRAM, 500 W power limit;
Linux x86_64, kernel 7.0.0-31, GCC 15.2, NVCC 13.3.73 and driver 610.57.04.
Release builds use patched GGML 0.25.3. CPU/BLAS uses OpenBLAS 0.3.32 (pthread)
and requests four threads; CUDA uses cuBLAS with the CPU BLAS backend and CUDA
graph capture disabled. All cells
run sequentially with the desktop active. These are separate measurements from
the [macOS M4 Pro measurements](MacOS-m4pro.md).

## Text-prompted image segmentation

The same 1800 × 1200 RGB PPM, `truck` prompt and 1008 × 1008 model input use one
cold call, five uncached full-image calls and five separate cache replays.
The table reports the median and P95 of the five full-image calls, excluding
model loading, external decoding, file output and cache replays. Every listed
configuration passed all seven original-reference image cases before timing.
CPU/BLAS cells show seconds / peak RSS GB; a dash means this cell was not timed.

CUDA process RSS and GPU memory are separate. RSS covers the timed process.
Image GPU memory comes from separate runs of the same workload and binaries,
sampled with `nvidia-smi` about every 50 ms, including context and pools.
Video GPU memory is sampled about once per second during timing. Sampling may
miss transient peaks; these figures are lower bounds for the true process peak.
GB is decimal; P95 uses linear interpolation at `(n-1)*0.95`.

CUDA below uses the default `--cuda-compute f32`; the two CPU/BLAS cells
retain their previous measurements. F16 weight storage does not select F16 math.

| Image weights | GGUF, GB | CPU/BLAS, s / RSS GB | CUDA median, s | CUDA P95, s | CUDA peak RSS, GB | CUDA sampled GPU, GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 | 3.371 | 16.981 / 4.674 | 0.505 | 0.508 | 0.901 | 4.777 |
| Mixed F16/F32 | 1.798 | 17.066 / 4.675 | 0.523 | 0.527 | 0.943 | 3.207 |
| Vision Q8_0 | 2.065 | — | 0.391 | 0.400 | 0.924 | 3.536 |
| Full-component Q8_0 | 1.097 | — | 0.401 | 0.405 | 0.938 | 2.569 |
| Vision Q6_K | 1.995 | — | 0.392 | 0.402 | 0.931 | 3.467 |
| Full-component Q6_K | 0.947 | — | 0.403 | 0.406 | 0.945 | 2.420 |
| Vision Q5_K | 1.957 | — | 0.388 | 0.399 | 0.930 | 3.429 |
| Full-component Q5_K | 0.864 | — | 0.400 | 0.402 | 0.944 | 2.336 |
| Vision Q4_K | 1.920 | — | 0.389 | 0.399 | 0.930 | 3.391 |
| Full-component Q4_K | 0.787 | — | 0.398 | 0.401 | 0.944 | 2.259 |

Quantized CUDA uses native packed-weight kernels with RHS Q8_1 staging and its
own arithmetic profile. Qualification measures final output quality; it does
not imply F32 tensor equivalence. See [quantized model use](../docs/quantization.md).

The following opt-in `--cuda-compute f16` results include direct F16 convolution
columns. Each row passed all seven original-reference cases and retained bit-exact
diagnostic tensors and masks relative to the previous F16 execution path. Dense
matrices use F16 inputs with F32 accumulation/output; quantized weights retain
their packed kernels. This does not imply equivalence to default F32 compute.
These three rows use three independent processes with 20 complete warm calls
each (60 samples); GPU memory is the maximum of three separate 50 ms sampled
runs. Other image configurations' earlier F16 timings are retained in the
[implementation record](../docs/plans/20261007-143003-activation-and-runtime-quantization.md).

| Image weights | F16 compute median, s | P95, s | Sampled GPU, GB |
| --- | ---: | ---: | ---: |
| Mixed F16/F32 | 0.356 | 0.359 | 2.974 |
| Full-component Q8_0 | 0.336 | 0.341 | 2.227 |
| Full-component Q4_K | 0.330 | 0.339 | 1.917 |

CUDA accelerates lossless layout copies with simplified indexing and tiled transposes.
It selects strategies by storage and compute mode: all modes batch concat
launches; quantized images in default compute combine fusion, detection and mask
stages to reduce internal host transfers; F16 compute uses fused head-256 tracker
attention. F16 image convolutions avoid full F32 expanded columns and their
duplicate conversion buffer. Quantized weights retain native packed matrix kernels. Q8_0 and Q4_K
latencies remain close, with smaller weight memory as the main benefit of lower
bit storage. Joint prediction and fast compute change temporary-buffer needs;
compare latency and GPU memory together.

## Forward text-prompted video tracking

F32 and hybrid `visual-tracker-f32-v1`, in both F32 and F16 compute modes, each
passed the complete five-case, 216-frame original-reference corpus before timing.
F16 compute uses the final tracking-quality gates. Its measurements below include
direct F16 convolution columns in shared image encoding/detection; emitted outputs
and diagnostic tensors also match the previous F16 path bit for bit.
Each cell uses the fixed
64-frame 1800 × 1200 RGB PNG workload, `truck` prompt, one or four persistent
objects and an eight-object limit. The first 16 frames are warmup; the remaining
48 frames provide the median and P95.

Per-frame latency covers `VideoSession::push_frame`, including preprocessing,
encoding, detection, tracking, masks and internal transfers/allocations. Loading,
external PNG decoding and file output are excluded. The existing 14-frame output
delay is separate; the final drain is included in the samples. RSS and sampled
GPU memory use the process-wide definitions above.

| Video weights | GGUF, GB | CUDA compute | Objects | Median, s/frame | P95, s/frame | Peak RSS, GB | Sampled GPU, GB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| F32 | 3.449 | F32 | 1 | 0.771 | 0.844 | 1.164 | 4.914 |
| F32 | 3.449 | F32 | 4 | 1.340 | 1.499 | 1.280 | 4.914 |
| Hybrid | 2.765 | F32 | 1 | 0.765 | 0.836 | 1.136 | 4.228 |
| Hybrid | 2.765 | F32 | 4 | 1.330 | 1.509 | 1.273 | 4.228 |
| F32 | 3.449 | F16 | 1 | 0.580 | 0.650 | 1.356 | 4.744 |
| F32 | 3.449 | F16 | 4 | 1.029 | 1.218 | 1.507 | 4.744 |
| Hybrid | 2.765 | F16 | 1 | 0.581 | 0.657 | 1.356 | 4.056 |
| Hybrid | 2.765 | F16 | 4 | 1.045 | 1.207 | 1.514 | 4.056 |

See the [performance index](../BENCHMARK.md#measure-your-workload) for shared measurement guidance and reproduction entry points.
