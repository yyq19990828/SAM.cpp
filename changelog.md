# Changelog

User-visible changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Implementation plans and experimental results are retained in `docs/plans/`.
No releases have been published.

## Unreleased

### Added

- Header-only C++17 SAM model integration with owned results, reusable sessions, and the `sam::sam` CMake target. The current adapter provides SAM 3 text image segmentation and forward video tracking on CPU and Metal.
- GGUF v3 conversion and loading for image F32/F16, full video F32/F16/hybrid, and vision-only image Q8_0/Q6_K/Q5_K/Q4_K profiles. Quantized video is not available.
- `sam_image` and `sam_video` command-line tools with masks, boxes, scores, persistent video IDs, runtime statistics, and optional diagnostic tensor exports.
- Image, video, session, header/linkage, and conversion checks, plus original-model reference and performance tools.
- Model/backend extension boundaries for additional SAM variants and composed segmentation pipelines.
- Bilingual visual image examples comparing F32, mixed F16/F32 and all four vision quantization profiles on CPU and Metal, with a 0.2 detection threshold and mask difference views.
- SAM 3 image weight quantization by vision, text, fusion and decoder components, with schema-4 full/custom profiles, per-tensor storage explanations and runtime component reporting. Biases, normalization, embeddings, convolutions and explicit small-weight exceptions retain F32.
- SAM 3 tracker graph caches and shared workspaces, current-frame feature residency, and compatible-object propagation batches with bounded memory and serial fallback.

### Changed

- Video conversion defaults to `visual-tracker-f32-v1` hybrid weights; image conversion requires an explicit precision. F16 video remains diagnostic.
- CPU execution uses registered BLAS when available. Quantized weights remain compressed in memory while matrix operations use temporary F32 weights.
- Explicit F32 Metal execution and native Metal window operations are available. Quantized Metal execution rejects CPU compute fallback.
- Public facades delegate to model adapters; backend modules own device initialization, storage policy, and accounting. Private implementation headers use `sam::internal` and unique include guards.
- Dependency preparation pins GGML 0.25.3, checks supplied source/patch identity, and leaves caller checkouts untouched.
- User documentation focuses on integration, supported configurations, and concise performance tables. Experiment history lives in the corresponding plans.
- Quantized model acceptance uses final segmentation quality, with intermediate tensor errors reported separately for reference.

### Fixed

- Preserve positional `RuntimeStats` aggregate initialization when adding BLAS statistics.
- Preserve canonical video object IDs and required F32 tensor shapes during conversion and diagnostic output.
- Keep required detector attention and window operations on Metal, and preserve activation precision for CPU/Metal matrix operations.
- Handle empty object batches in reference tools and reject overlapping source/output paths before writing or cleaning generated data.
- Bind reference validation to the actual model, executable, backend libraries, and output files, and reject incomplete or altered archives.

### Removed

- Legacy custom `.ggml` loading and incomplete-tokenizer repair. Reconvert original checkpoints to GGUF; renaming an old file does not convert it.

### Security

- Validate metadata, model schemas, tensor ranges, case IDs, and generated paths before use. Reject unsafe symbolic links and overlapping dependency paths.
