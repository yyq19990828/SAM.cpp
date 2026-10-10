# Changelog

User-visible changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Implementation plans and experimental results are retained in `docs/plans/`.
No releases have been published.

## Unreleased

### Added

- Quality-independent application performance benchmarks with paired latency and separate process-memory measurements, small-case selection, configurable iteration counts, offline summaries and advisory benefit tags. Slowdowns or missing quality reports do not block valid measurements.
- Precision inspection and consistent quality/performance descriptions distinguish stored tensor types, requested compute/cache policies, runtime identifiers and uncollected kernel arithmetic. Opt-in graph diagnostics expose operand/output types and precision hints without treating them as kernel traces.

- Advisory application-image quantization benchmarks against original-checkpoint or native-F32 outputs, with custom image/prompt exports, one-to-one mask agreement, per-case missing/added reference objects and optional user limits that do not veto valid reports. Completed ranked v2/v3 exports are reusable without regrading historical receipts; production accuracy and performance are explicitly separate from reference agreement.
- Installable relocatable SDK package: `SAM_ENABLE_INSTALL` (standalone default) adds standard `install(TARGETS)`/`install(EXPORT)` rules with a CMake package config and version file. Static and shared builds both produce a consumable `find_package(sam CONFIG)` package; SAM-prepared GGML is installed and exported with it, a caller-owned GGML must resolve through its own package, and an unexportable build-tree GGML is rejected with an explicit message. Installed shared libraries locate their GGML siblings through an `$ORIGIN` run path. `tests/install_consumer` verifies the package from an isolated build directory and after moving the install prefix.
- `SAM_BUILD_TOOLS` option (standalone default `ON`, embedded `OFF`) builds the standalone conversion, validation and profiling tools independently of `SAM_BUILD_EXAMPLES`, which now owns only the `sam_image` and `sam_video` applications.
- Compiled-library source identity: `source_snapshot()` and `archive_sources()` now recursively cover `src/` implementations and build files, with behavior tests proving that changing a new implementation changes its identity, that an archive verifies after the live sources are removed, and that archive tampering fails.
- Independent CUDA Q8_0 feature-cache codec verification and frozen-library reconciliation, checking native block scales, signed values, layout and decoded bits against their F32 inputs. Arithmetic evidence remains separate from model quality and measured benefits.
- Hash-bound stage and original-weight comparators for the CUDA F32 reference configuration, preserving declared layout transformations and keeping intermediate differences separate from final-output acceptance.
- Opt-in CUDA Q8_1 right-operand staging and Q8_0 MMVQ/MMQ raw-dot probes, with independent packed-byte and same-operand checks bound to the measured GGML library. Kernel boundary, model quality and complete recipe conclusions remain separately scoped.
- The separately frozen short-dot CUDA Q8 image recipe passes complete seven-call graph arithmetic, fresh 1,024-image final quality and uncontaminated paired performance. Its scoped deployment result earns a GPU-memory label for full-image and changed-prompt inference on RTX 4090; the 10% latency gate and repeated-result memory label do not pass. Earlier candidate receipts keep their original statuses.
- Separate experimental mixed-Q8_0 image-cache recipes with F32 or custom text/fusion/decoder Q8_0 weights pass complete graph arithmetic, absolute/incremental final quality and paired latency gates on RTX 4090. Acceptance covers full-image and changed-prompt latency only; memory gates do not pass, and public loading keeps its existing cache precision.
- An independent COCO follow-up selector that keeps the original development and unopened reserve images, excludes every exposed v2 evaluation image by ID and content, and freezes fresh evaluation images before inference.
- A COCO train2017 precision holdout preparer that selects unused images from annotations before inference, verifies downloaded content, and preserves the existing development and reserve splits for a new paired evaluation.
- Opt-in W8A8 CUDA probe dumps of packed operands and raw INT32 dots, with a signed INT64 verifier that checks every output position and rejects malformed layouts or payloads. This closes the probe's sampled-dot arithmetic gap without adding an integrated W8A8 model profile.
- Opt-in FP8 CUDA probe algorithm selection and packed-operand dumps, with independent E4M3 sampled-dot verification. FP8 remains diagnostic and does not enable a native model profile.
- Experimental image-feature cache tools and typed host storage, with native GGML F16/Q8_0 codec checks and a mixed Q8_0 recipe that preserves the F32 detection feature. Public model loading continues to use the existing cache precision.
- Original-model COCO screening for calibrated vision linears, with separate weight/activation ablations, frozen per-object gates, prompted mask AP, positive-mask mIoU and negative-prompt counts. These tools do not qualify a runtime W8A8 profile.
- Opt-in CUDA linear quantization probes with independently checked INT8 arithmetic, calibrated channel scaling, explicit temporary-memory accounting, and separate GGML comparisons. FP8 probes remain diagnostic and require their numerical gates to pass.
- Internal graph profiling with alias-aware live allocation spans, tensor shapes/types and backend arena sizes, plus bounded original-FP32 vision calibration statistics, reproducible disjoint COCO data manifests, and offline W8A8 channel-scaling studies.
- Document a [staged runtime quantization roadmap](docs/plans/20261007-143003-activation-and-runtime-quantization.md) with separate quality, latency and memory criteria. Calibrated W8A8/FP8, quantized attention and additional video-state compression remain planned.
- Explicit CUDA F16 compute mode with F32 dense accumulation/output, separate arithmetic profiles, and final-output quality validation independent of weight storage.
- CUDA backend selection with a visible-device index, device identity, CUDA node statistics and strict GPU compute placement. SAM 3 F32/F16 image, eight vision/full quantized image presets and F32/hybrid video have original-model qualification on Linux x86_64 RTX 4090.
- Pinned GGML CUDA corrections for F32 dense/attention precision and native window operations, with required-GPU CTest checks.
- CUDA native quantized image arithmetic with a distinct runtime profile, RHS Q8_1 staging and separate output-quality and tensor-diagnostic records.
- Compiled C++17 SAM model integration with owned results, reusable sessions, and the `sam::sam` CMake target. The current adapter provides SAM 3 text image segmentation and forward video tracking on CPU, Metal and CUDA.
- GGUF v3 conversion and loading for image F32/F16, full video F32/F16/hybrid, and vision-only image Q8_0/Q6_K/Q5_K/Q4_K profiles. Quantized video is not available.
- `sam_image` and `sam_video` command-line tools with masks, boxes, scores, persistent video IDs, runtime statistics, and optional diagnostic tensor exports.
- Image, video, session, header/linkage, and conversion checks, plus original-model reference and performance tools.
- Model/backend extension boundaries for additional SAM variants and composed segmentation pipelines.
- Bilingual visual image examples comparing F32, mixed F16/F32 and all four vision quantization profiles on CPU and Metal, with a 0.2 detection threshold and mask difference views.
- SAM 3 image weight quantization by vision, text, fusion and decoder components, with schema-4 full/custom profiles, per-tensor storage explanations and runtime component reporting. Biases, normalization, embeddings, convolutions and explicit small-weight exceptions retain F32.
- SAM 3 tracker graph caches and shared workspaces, current-frame feature residency, and compatible-object propagation batches with bounded memory and serial fallback.

### Changed

- Archive v2/v3 quality tiers, hard budgets and campaign-dependent qualification commands; remove their active grouped and flat entry points. Current quantization selection uses independent reference agreement and performance reports. Historical policy JSON bytes and result receipts are unchanged.

- Organize bilingual performance measurements under `benchmarks/` by operating system and hardware (`MacOS-m4pro` and `Linux-4090`), keeping the root performance pages as indexes and retaining existing measurement values and historical links.
- Keep the English and Chinese quantization guides focused on conversion, precision limits and scoped results; retain experimental history and diagnostic evidence in the linked plans.
- Group the repository by ownership: the CLI applications moved to `apps/{image,video}/`, decoding support to `support/image_io/`, vendored STB headers to `third_party/stb/`, tests to `tests/{api,models,runtime/ggml,integration,tools}/`, and the Python tools to `tools/{convert,quantize,validation,benchmark,visualization,maintenance}/`. Historical flat tool module paths remain as forwarding entry points, and the `${build}/examples/` CLI and probe output paths are unchanged. SDK-only builds no longer compile decoding, STB, Python or test code.
- Extend compiled-library source identity and archives to nested tools, `apps/`, `support/`, the compiled `third_party/stb/` headers and all related CMake files, with migration coverage tests for nested tool changes and offline archive verification.
- Move the private GGML runtime and its CPU, Metal and CUDA drivers from `include/sam/internal/runtime/` into `src/runtime/ggml/` with path-derived include guards, completing the private source relocation. The runtime keeps its device discovery, storage/arithmetic policies, workspace and statistics and does not include model headers; tests and probes follow through `sam_private`.
- Move the private task contracts, input validation, GGUF reader and SAM 3 adapter from `include/sam/internal/` into `src/{contracts,common,io,models/sam3}/`, with the tracking sources renamed to `src/models/sam3/video/`. Internal tests and probes reach them through the non-installed `sam_private` target; public headers, model behavior and the installed interface are unchanged.
- `sam::sam` is now a real compiled library instead of an INTERFACE target. The public `Model`, `ImageSession` and `VideoSession` wrappers and the explicit model factory build in `src/`; public headers depend only on the standard library, public values and forward declarations, and no longer expose GGML or private implementation headers. The library is static by default, honors a parent `BUILD_SHARED_LIBS`, marks public symbols through a generated export header, and propagates the required GGML link dependency to final consumers. Internal tests, private header checks and experimental probes now use the non-installed `sam_private` support target.
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
- Upgrade the pinned GGML dependency to v0.26.0 with ported Metal/CUDA precision patches and verified source identities. New conversions record the new quantizer identity; existing GGUF models retain compatibility with their recorded v0.25.3 provenance. Metal validation on Apple hardware remains pending for this upgrade.
- User documentation focuses on integration, supported configurations, and concise performance tables. Experiment history lives in the corresponding plans.
- Bilingual performance pages list complete current model measurements; historical results and optimization comparisons remain in implementation plans.
- Validation provenance records identify retired historical payloads and retained original references.
- Quantized model acceptance uses final segmentation quality, with intermediate tensor errors reported separately for reference.
- SAM 3 vision rotary embeddings use contiguous channel vectors to reduce CPU/BLAS and Metal image inference latency while preserving weight and arithmetic precision.

### Fixed

- Accept the exact current three-patch GGML build identity for Q6_K/Q5_K/Q4_K conversion, retaining rejection of unknown quantizer builds and compatibility with historical conversion manifests.
- Release completed prompt tensors and the previous image state in the original-model reference exporter, reducing transient GPU memory when switching images while preserving same-image feature reuse.
- Preserve source-override compatibility with the previously verified combined GGML archive, upgrading its CUDA corrections in a build-local copy without modifying the supplied source.
- Use bounded double-precision accumulation for explicit-F32 short CUDA dots, repairing a cancellation failure in the custom Q8 recipe's decoder presence head without changing the arithmetic gate.
- CUDA Q8_0 MMQ staging now uses correctly rounded F32 inverse-scale division, avoiding nonfinite stored scales for the tested finite extreme inputs.
- Make documentation checks independent of private build/model artifacts: retain their paths as archive records, reject links to Git-ignored targets, and check all repository Markdown documents.
- Restore direct execution of grouped Python commands from any working directory, including reference-export child processes, while preserving flat compatibility entry points and module invocation.
- Resolve the fully static SDK's OpenMP runtime dependencies for consumers that enable only C++, without requiring a C compiler in the consuming project.
- Build conversion tools with caller-provided `ggml::ggml` packages and recognize namespaced CUDA backend targets in tools and tests.
- Include compiled implementation and build sources in video benchmark receipts, and reject runs whose implementation changes during measurement.
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

- Retire the temporary compiled-library migration PRD and technical specification; the historical task plan now points to the retained architecture and migration plan.
- Legacy custom `.ggml` loading and incomplete-tokenizer repair. Reconvert original checkpoints to GGUF; renaming an old file does not convert it.

### Security

- Validate metadata, model schemas, tensor ranges, case IDs, and generated paths before use. Reject unsafe symbolic links and overlapping dependency paths.
