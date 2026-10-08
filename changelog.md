# Changelog

User-visible changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Implementation plans and experimental results are retained in `docs/plans/`.
No releases have been published.

## Unreleased

### Added

- Experimental v2 precision acceptance tools with separate quality budgets, one-to-one object matching, ranked COCO mask AP, paired confidence bounds, independent data splits and frozen performance comparisons. Native F16 codec bit patterns and the seven fixed image cases have separate checks. Results distinguish failure, insufficient evidence and untested configurations; existing model support and v1 receipts retain their original scope.
- Experimental image-feature cache tools and typed host storage, with native GGML F16/Q8_0 codec checks and a mixed Q8_0 recipe that preserves the F32 detection feature. Public model loading continues to use the existing cache precision.
- Original-model COCO screening for calibrated vision linears, with separate weight/activation ablations, frozen per-object gates, prompted mask AP, positive-mask mIoU and negative-prompt counts. These tools do not qualify a runtime W8A8 profile.
- Opt-in CUDA linear quantization probes with independently checked INT8 arithmetic, calibrated channel scaling, explicit temporary-memory accounting, and separate GGML comparisons. FP8 probes remain diagnostic and require their numerical gates to pass.
- Internal graph profiling with alias-aware live allocation spans, tensor shapes/types and backend arena sizes, plus bounded original-FP32 vision calibration statistics, reproducible disjoint COCO data manifests, and offline W8A8 channel-scaling studies.
- Document a [staged runtime quantization roadmap](docs/plans/20261007-143003-activation-and-runtime-quantization.md) with separate quality, latency and memory criteria. Calibrated W8A8/FP8, quantized attention and additional video-state compression remain planned.
- Explicit CUDA F16 compute mode with F32 dense accumulation/output, separate arithmetic profiles, and final-output quality validation independent of weight storage.
- CUDA backend selection with a visible-device index, device identity, CUDA node statistics and strict GPU compute placement. SAM 3 F32/F16 image, eight vision/full quantized image presets and F32/hybrid video have original-model qualification on Linux x86_64 RTX 4090.
- Pinned GGML CUDA corrections for F32 dense/attention precision and native window operations, with required-GPU CTest checks.
- CUDA native quantized image arithmetic with a distinct runtime profile, RHS Q8_1 staging and separate output-quality and tensor-diagnostic records.
- Header-only C++17 SAM model integration with owned results, reusable sessions, and the `sam::sam` CMake target. The current adapter provides SAM 3 text image segmentation and forward video tracking on CPU, Metal and CUDA.
- GGUF v3 conversion and loading for image F32/F16, full video F32/F16/hybrid, and vision-only image Q8_0/Q6_K/Q5_K/Q4_K profiles. Quantized video is not available.
- `sam_image` and `sam_video` command-line tools with masks, boxes, scores, persistent video IDs, runtime statistics, and optional diagnostic tensor exports.
- Image, video, session, header/linkage, and conversion checks, plus original-model reference and performance tools.
- Model/backend extension boundaries for additional SAM variants and composed segmentation pipelines.
- Bilingual visual image examples comparing F32, mixed F16/F32 and all four vision quantization profiles on CPU and Metal, with a 0.2 detection threshold and mask difference views.
- SAM 3 image weight quantization by vision, text, fusion and decoder components, with schema-4 full/custom profiles, per-tensor storage explanations and runtime component reporting. Biases, normalization, embeddings, convolutions and explicit small-weight exceptions retain F32.
- SAM 3 tracker graph caches and shared workspaces, current-frame feature residency, and compatible-object propagation batches with bounded memory and serial fallback.

### Changed

- Reduce CUDA F16 convolution workspace and scratch memory in image segmentation and shared video encoding/detection by producing F16 expanded columns directly for the existing F16-input/F32-output arithmetic.
- Focus routine performance tables on GPU and CPU/BLAS; retain native CPU measurements in the historical record and keep native CPU correctness coverage.
- Reduce CUDA image/video latency with batched cuBLAS attention products, bounded query tiles and direct output assembly, preserving default numerical qualification and existing GGUF conversion receipts.
- Batch CUDA concat launches across tensor planes; combine quantized-image prediction stages in default compute mode to reduce host transfers; use fused head-256 tracker memory attention in the opt-in F16 mode.
- Accelerate CUDA layout copies with bounded source indexing and tiled transposes, preserving compute precision and existing tensor storage.
- Runtime backend node counters count compute operations, excluding metadata-only views, reshapes, permutations and transposes.
- Video conversion defaults to `visual-tracker-f32-v1` hybrid weights; image conversion requires an explicit precision. F16 video remains diagnostic.
- CPU execution uses registered BLAS when available. Quantized weights remain compressed in memory while matrix operations use temporary F32 weights.
- Explicit F32 Metal execution and native Metal window operations are available. Quantized Metal execution rejects CPU compute fallback.
- Public facades delegate to model adapters; backend modules own device initialization, storage policy, and accounting. Private implementation headers use `sam::internal` and unique include guards.
- Dependency preparation pins GGML 0.25.3, checks supplied source/patch identity, and leaves caller checkouts untouched.
- User documentation focuses on integration, supported configurations, and concise performance tables. Experiment history lives in the corresponding plans.
- Bilingual performance pages list complete current model measurements; historical results and optimization comparisons remain in implementation plans.
- Validation provenance records identify retired historical payloads and retained original references.
- Quantized model acceptance uses final segmentation quality, with intermediate tensor errors reported separately for reference.
- SAM 3 vision rotary embeddings use contiguous channel vectors to reduce CPU/BLAS and Metal image inference latency while preserving weight and arithmetic precision.

### Fixed

- Install the pinned COCO dependency in Quick checks so mask and precision acceptance tests can run in CI.
- Preserve the original Linear bias operation in numerical reparameterization studies; add explicit mixed-layer selection and exactly representable channel-scale experiments without changing runtime defaults.
- Bind CUDA validation and video benchmark receipts to their compute mode, and report the selected quantization tensor gate identity correctly.
- Replace broken links to unavailable historical performance receipts with archive path records, preserving their original hashes.
- Keep shared video workspaces within their serial memory budget when graph stages grow device and host arenas in opposite directions.
- Preserve CUDA position-cache placement when preparing original-model reference sources.
- Make image antialias preprocessing match the original float resize path across compilers, including byte-rounding boundaries.
- Preserve mask resize rounding at video initialization thresholds across host compilers.
- Accept and record native quantizers linked to the pinned combined Metal/CUDA GGML patches.
- Preserve positional `RuntimeStats` aggregate initialization when adding BLAS statistics.
- Preserve canonical video object IDs and required F32 tensor shapes during conversion and diagnostic output.
- Keep required detector attention and window operations on Metal, and preserve activation precision for CPU/Metal matrix operations.
- Handle empty object batches in reference tools and reject overlapping source/output paths before writing or cleaning generated data.
- Bind reference validation to the actual model, executable, backend libraries, and output files, and reject incomplete or altered archives.

### Removed

- Legacy custom `.ggml` loading and incomplete-tokenizer repair. Reconvert original checkpoints to GGUF; renaming an old file does not convert it.

### Security

- Validate metadata, model schemas, tensor ranges, case IDs, and generated paths before use. Reject unsafe symbolic links and overlapping dependency paths.
