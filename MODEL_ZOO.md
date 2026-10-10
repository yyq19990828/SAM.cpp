# Models and precision

[中文](MODEL_ZOO_zh.md) · [Performance](BENCHMARK.md)

SAM.cpp is designed to host multiple SAM model adapters and composed segmentation
pipelines across platforms. This catalog separates available implementations
from future integration work.

## Available now

| Model / task | Weight choices | Current backends | Recommended starting point |
| --- | --- | --- | --- |
| SAM 3 text image segmentation | F32, mixed F16/F32 | CPU, Metal, CUDA | F16 on GPU; F32 for a reference configuration |
| SAM 3 text-prompted image segmentation (quantized weights) | Vision Q8_0, Q6_K, Q5_K, Q4_K | CPU, Metal, CUDA | Q8_0 for a conservative quantized configuration |
| SAM 3 text-prompted image segmentation (full-component mixed quantized weights) | Full-component linear Q8_0, Q6_K, Q5_K, Q4_K | CPU, Metal, CUDA | Full preset for compression, followed by application-data checks |
| SAM 3 forward video tracking | F32, hybrid | CPU, Metal, CUDA | Hybrid `visual-tracker-f32-v1` |

Vision quantization covers only ViT attention projections and MLP linear
weights. The text encoder, fusion, detection and mask heads, and remaining
weights stay F32. The task remains text-prompted image segmentation.
Full presets also cover target text, fusion, detection and mask-related decoder
linears, totaling 348 matrices. Embeddings, convolutions, biases, normalization
and explicit small-weight exceptions stay F32; activation precision keeps its
existing policy.

Users can select local quantization with options such as
`--quantize-modules vision,text`. Repository numerical acceptance focuses on
the fixed vision and full presets; custom combinations need separate checks.
See the [quantization guide](docs/quantization.md).

The [module mixed configuration](docs/quantization-config.md#module-mixed-weights)
also assigns F32/Q8_0/Q6_K/Q5_K/Q4_K independently per module. Small CPU
conversion/arithmetic fixtures are validated; complete-model quality/performance
and mixed CUDA/Metal execution remain unmeasured.

[Visual examples](docs/visual-examples.md) compare CPU and Metal outputs for
each image weight configuration using the same images, prompts and a detection
threshold of 0.2. Quantized acceptance uses final output quality; tensor errors
are reported separately for reference.

The SAM 3 implementation remains experimental. Its validation uses a small
reference corpus, so check the quality of your application's own inputs.
F16 video, legacy `image-linear-*` and custom `image-modules-linear-*` profiles
remain diagnostic by default; validate application data before integration.
One exact [text/fusion/decoder Q8_0 CUDA image recipe](docs/quantization.md)
has a scoped GPU-memory acceptance on RTX 4090. Separate experimental mixed-Q8_0
cache-tool recipes with F32 or those Q8_0 weights pass full-image/changed-prompt
latency gates; public loading retains its existing cache precision. Quantized video,
reverse tracking, and interactive video prompts are not available.

CPU is the portable execution path. Platform validation covers macOS CPU/Metal
and Linux x86_64 CUDA on an RTX 4090: F32/F16 image, the eight vision/full image
quantization presets, and F32/hybrid video. F32/F16 image also passed on the same
Linux CPU with OpenBLAS. Other CPU/GPU hardware and operating systems require
their own builds and numerical checks. Metal is an Apple-specific backend, not
a requirement for the library's model interfaces.

## Model and pipeline roadmap

| Family / pipeline | Intended tasks | Status |
| --- | --- | --- |
| Other SAM generations, including SAM 2/2.1 | Point/box image prompts and memory-based video tracking | Additional adapters |
| SAM 3.1 | Its own model/task contract | Additional adapter |
| GroundingSAM | Text detection composed with a SAM segmentation adapter | Composed pipeline |
| DART-style pipelines | Reuse detection stages without mandatory mask decoding | Extension point |

These entries describe the repository's direction; they do not advertise
implemented support. New adapters own their tensor schema, tokenizer, transforms,
and temporal rules. [Architecture](docs/architecture.md) explains the shared
interfaces and model/backend extension boundaries.

## Precision contract

Weight labels describe storage. Computation and state can use different formats.
The following policies belong to the current SAM 3 adapter and backends.

| Weight label | Disk / Metal / CUDA resident weights | CPU resident weights |
| --- | --- | --- |
| F32 | Original F32 | F32 |
| F16 | Mixed F16/F32 | Stored F16 promoted to F32 |
| Hybrid | F32 visual/tracker weights, mixed detector/text | Remaining F16 promoted to F32 |
| Vision / full quantized | Packed Q8/K selected linears, other weights F32 | Packed weights retained; temporary F32 matrix weights |

Promotion preserves rounded values, not the original F32 values. Quantized
Metal and CUDA execution use native kernels with separate arithmetic profiles;
CUDA's validated MMVQ/MMQ paths use RHS Q8_1 staging. CPU keeps compressed weights
resident and performs matrix operations with temporary F32 weights. `ModelInfo::precision`,
`storage_profile`, and `arithmetic_profile` describe the loaded configuration.
See the [quantization guide](docs/quantization.md) for exact profile names.

CUDA compute precision is selected separately with `--cuda-compute f32|f16`
(`BackendOptions::cuda_compute` in C++). The default preserves F32 dense inputs
and attention. Opt-in F16 compute rounds dense operands to F16 while accumulating
and returning F32, and enables supported fused reduced-input attention. It uses
`ggml-cuda-f16-v1`; quantized weights keep their packed kernels and use
`ggml-quantized-cuda-f16-v1`. The storage label and GGUF file do not change.
F16 compute acceptance uses the existing F16 final mask/score/box gates, or the
existing quantized output gates, while recording intermediate tensor errors
separately. Video ID, lifecycle, output-delay and state bounds remain required.

SAM 3 video retains these state boundaries for F32, F16, and hybrid storage:

| Boundary | Representation |
| --- | --- |
| Normalization | F16 rounding, with an F32 output buffer |
| Tracker-neck features | BF16 rounding in F32 vectors |
| Mask-memory records | BF16 storage, expanded to F32 for attention |
| Object pointers / host mask logits | F32; final binary masks use uint8 |

An F32 video file does not imply an entirely F32 pipeline. Image inference has
no tracker BF16 state. These policies should not be assumed for future model
adapters.

## Download and use

The current adapter uses the original [facebook/sam3](https://huggingface.co/facebook/sam3)
checkpoint and its tokenizer assets. Model access and license acceptance are
separate from dependency licensing. See [licenses](THIRD_PARTY_NOTICES.md).

Follow the [SAM 3 download and conversion guide](docs/models/sam3-details.md).
It covers the pinned source revision, authentication, image/video conversion,
and the Python tooling environment. C++ inference does not require Python.
Use new output paths; the converter refuses to overwrite existing files.

[README](README.md) shows image/video integration and CLI examples.
[Model verification](docs/validation.md) explains how to compare your converted
model with a reference. [Performance](BENCHMARK.md) lists measurements by model,
hardware, and backend.
