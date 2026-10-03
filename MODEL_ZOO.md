# Models and precision

[中文](MODEL_ZOO_zh.md) · [Performance](BENCHMARK.md)

SAM.cpp is designed to host multiple SAM model adapters and composed segmentation
pipelines across platforms. This catalog separates available implementations
from future integration work.

## Available now

| Model / task | Weight choices | Current backends | Recommended starting point |
| --- | --- | --- | --- |
| SAM 3 text image segmentation | F32, mixed F16/F32 | CPU, Metal | F16 on Metal; F32 for a reference configuration |
| SAM 3 text-prompted image segmentation (quantized weights) | Vision Q8_0, Q6_K, Q5_K, Q4_K | CPU, Metal | Q8_0 for a conservative quantized configuration |
| SAM 3 text-prompted image segmentation (full-component mixed quantized weights) | Full-component linear Q8_0, Q6_K, Q5_K, Q4_K | CPU, Metal | Full preset for compression, followed by application-data checks |
| SAM 3 forward video tracking | F32, hybrid | CPU, Metal | Hybrid `visual-tracker-f32-v1` |

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

[Visual examples](docs/visual-examples.md) compare CPU and Metal outputs for
each image weight configuration using the same images, prompts and a detection
threshold of 0.2. Quantized acceptance uses final output quality; tensor errors
are reported separately for reference.

The SAM 3 implementation remains experimental. Its validation uses a small
reference corpus, so check the quality of your application's own inputs.
F16 video, legacy `image-linear-*` and custom `image-modules-linear-*` profiles
remain diagnostic; validate application data before integration. Quantized video,
reverse tracking, and interactive video prompts are not available.

CPU is the portable execution path. Current platform validation covers macOS
CPU and Metal; Linux, Windows, other CPU hardware, and additional accelerators
require their own builds and numerical checks. Metal is an Apple-specific backend,
not a requirement for the library's model interfaces.

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

| Weight label | Disk / Metal resident weights | CPU resident weights |
| --- | --- | --- |
| F32 | Original F32 | F32 |
| F16 | Mixed F16/F32 | Stored F16 promoted to F32 |
| Hybrid | F32 visual/tracker weights, mixed detector/text | Remaining F16 promoted to F32 |
| Vision quantized | Packed Q8/K vision linears, other weights F32 | Packed weights retained; temporary F32 matrix weights |

Promotion preserves rounded values, not the original F32 values. Quantized
Metal execution uses native kernels; CPU keeps compressed weights resident and
performs matrix operations with temporary F32 weights. `ModelInfo::precision`,
`storage_profile`, and `arithmetic_profile` describe the loaded configuration.
See the [quantization guide](docs/quantization.md) for exact profile names.

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
