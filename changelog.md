# Changelog

Notable project changes are recorded here by release, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
No releases have been published.

## Unreleased

### Added

- Complete M2 original-reference acceptance for F32 and hybrid on CPU/Metal: four five-case/216-frame cells, 56 fresh image regressions plus 14 sealed hybrid cases, six real short session checks and two 64+64-frame interleaved session/lifetime checks. Explicit F16 remains diagnostic: Metal fails entry candidate selection at frames 23/24, reproduced by exact-F16 original Meta modules; the optional full CPU diagnostic is deferred.
- A 64-frame video benchmark runner with 16 warmup/48 measured frames, one/four-object qualification, per-frame/stage samples, first-output/final-drain accounting, source/artifact/output guards, current/peak RSS and allocation checks. All four hybrid CPU/Metal workload cells pass; [measurements and limits](BENCHMARK.md#video-64-frame-protocol) retain the raw evidence.
- Optional `test_video_long_session` CMake/CTest target for two interleaved 64-frame sessions, model/session lifetime, owned outputs, independent caches and bounded state. Both backends pass exact comparison with their accepted standalone positive sequence and empty negative results.

- Experimental schema-2 full SAM 3 video GGUF conversion/loading with 1,464 tensors, named tracker/storage/preprocessing metadata, exclusive output publication and task/profile information. Image schema 1 remains the default.
- Header-only forward `VideoSession` with one cached text prompt, ordered finite RGB input, persistent IDs, owned delayed results, failure/reset recovery and bounded memory. Logical birth groups retain the pinned Meta association, reconditioning, overlap and hotstart lifecycle rules while sharing one visual trunk per frame.
- `sam_video` for contiguous PNG sequences with exclusive output, per-frame masks/IDs, lifecycle/memory traces, stage tensor dumps and explicit incomplete/complete receipts.
- Original-checkpoint video reference export and differential validation, preserving F16 input/BF16 storage boundaries, checking the five required scenario behaviors, fixed birth-ID mapping, frozen numerical gates and Metal graph placement. Diagnostic subsets remain ineligible for complete video acceptance.
- Propagated video mask and normalized memory-input snapshots on both the C++ session and original reference exporter. Periodic correction no longer hides the propagated mask's error; older video references need re-exporting for the additional observations.
- Memory-input snapshots reuse the consumed mask buffer after encoding, avoiding a 5 MiB per-object/frame diagnostic copy. Final-source original two-frame CPU/FP32 and Metal/FP16 checks verify snapshot ownership and numerical parity.
- Per-frame video candidate IoU/mask/pointer-token traces before correction, plus an explicit `sam_video --dump-all-tensors` option for finding the first precision/recurrence divergence. Default tensor dumps remain sparse.
- Explicit video-only `visual-tracker-f32-v1` hybrid GGUF conversion/loading, retaining original FP32 visual/tracker values and mixed detector/text payloads. Strict profile/type declarations and CLI/model reporting identify its actual mixed storage. All payloads, the 17-frame original stress sequence, the five-case/216-frame original video corpus and seven image cases pass on CPU and Metal under unchanged mixed gates. Its completed support and performance scope is recorded in the [M2 acceptance plan](docs/plans/20261002-182848-m2-complete-acceptance.md); explicit F16's failed evidence is retained.
- Separate no-dump, three-frame hybrid CPU/Metal latency and memory samples with raw per-frame times, final-drain scope, power conditions and immutable artifact/output receipts. These historical samples remain separate from the completed 64-frame protocol.
- Original/mixed-module and exact-input replay evidence locates the retained FP16 stress discrepancy at conditioned-feature perturbations crossing hard candidate selection and amplifying pointer/memory state. Existing FP32/Metal passes all 17 original-prefix frames and choices; unverified precision policies are not added and the FP16 failure remains explicit.
- Earlier original 216-frame video export and FP16/Metal comparison: motion, occlusion, hotstart and negative numerical gates pass with zero CPU graph fallback; entry memory/recurrence fails. The earlier entry/hotstart recipes miss their required original behaviors. Their failed receipts remain preserved separately from the revised fixtures.
- Tracker mask/memory execution, a shared ViT trunk for both necks, explicit BF16 transport rounding, and precise 256-wide full-key attention with 128-query tiles.
- Pillow 11.2.1 RGB bicubic/F16 preprocessing goldens, full official tensor-inventory checks, and forward memory retention checked against 2,024 weight-free pinned Meta selector cases.
- Five deterministic M2 frame recipes and `generate_video_cases.py`, recording PNG hashes and incomplete/complete generation separately from unverified oracle behavior. Original media remains external.
- Revised entry/hotstart recipes: a separated mirrored vehicle enters at frame 16; body-only occlusion preserves nonempty unmatched tracks for retirement. Entry verification now requires the late birth and original ID continuity throughout, with boundary/mirroring regressions and original-model behavior checked separately.
- Revised entry (64 frames) and hotstart (24 frames) pass original behavior checks. Earlier FP16/Metal subset receipts remain historical; the later complete-corpus candidate check rejects entry frames 23/24. F32 and hybrid pass the complete revised corpus on both backends.

- [FP32 CPU/Metal validation plan](docs/plans/20261001-121633-fp32-cpu-metal-validation.md) defining explicit FP32 Metal enablement, unchanged Auto selection, official four-configuration acceptance and comparable image benchmarks before M2. Local original-weight acceptance and the four benchmark cells are complete.
- [M2 video-tracking plan](docs/plans/20261001-115321-sam3-text-video-tracking.md) defining full video GGUF weights, buffered text-driven tracking, bounded temporal state, CPU/Metal execution and original-reference acceptance; the [local video record](docs/plans/20261002-041320-video-session-official-validation.md) distinguishes implementation from complete corpus/matrix and performance acceptance.
- [SAM GGUF schema 1](docs/gguf.md), a bounded common GGUF reader, and official `gguf` Python serialization with streaming tensor writes and source/output provenance.
- [Benchmark matrix](BENCHMARK.md) with model rows and hardware/backend headers, recorded software versions and reproduction steps; [model catalog](MODEL_ZOO.md) with pinned HF/GitHub sources, verified conversion commands and explicit GGUF support status.
- [Contributor guidelines](AGENTS.md) for the planned header-only SAM library, focused testing, extensibility for multiple backends and model architectures, implementation plans, and changelog maintenance.
- [First implementation plan](docs/plans/20260930-192519-sam3-text-image-baseline.md) defining SAM 3 text image segmentation on CPU/Metal, integration and validation requirements, and follow-on video/SAM 3.1 milestones.
- [Compute-precision correction plan](docs/plans/20260930-205540-ggml-fp16-compute-precision.md) documenting the measured FP16 backend errors and their verification requirements.
- Experimental header-only C++17 SAM 3 image segmentation with a reusable model, per-image sessions, text/token prompts, owned masks/boxes/scores, and CPU/Metal execution through the pinned GGML dependency.
- The `sam::sam` CMake integration target, a downstream consumer example, and a CLI that writes masks and reports inference/cache timing and backend execution statistics.
- Exact FP16-to-FP32 CPU loading and a verified [Metal precision patch](cmake/patches/README.md), applied to isolated dependency copies. All seven original-checkpoint cases now pass tensor and detection checks on each supported precision/backend combination; earlier supplementary evidence remains separately labeled.
- Strict GGUF v3 checkpoint loading, FP32/FP16 conversion, provenance manifests, reproducible CPU reference preparation, and tensor/mask differential validation. Supplementary community-weight checks are explicitly separate from original-checkpoint acceptance.
- Contract, graph, two-translation-unit, output-handling, and Python tooling regressions, plus retained dependency licenses and an integration guide.
- [Architecture guide](docs/architecture.md) and [refactoring plan](docs/plans/20260930-221000-model-layering-header-guards.md) describing extension boundaries for SAM 2/2.1, composed GroundingSAM pipelines, and DART-style detection. These extensions are not yet implemented.
- Independent/repeated header compilation, a GGML-free task-contract check, and a session regression that releases the last public model handle before inference.
- [Backend modularization plan](docs/plans/20260930-224223-internal-backend-modules.md) and CUDA extension guidance, with driver-specific initialization, precision/storage policies, and execution accounting.
- [Official GGML migration plan](docs/plans/20260930-231832-upstream-ggml-0253.md), with direct GPU arithmetic checks and numerical regressions for convolution precision and attention boundaries.
- Native Metal window partition/restoration, including zero padding, cropping, and partial threadgroups. Direct GPU regressions cover six shapes, including the SAM vision shape, and run in caller-owned GGML consumers.
- Individual full-image/stage and cache timing samples in CLI JSON, retaining existing median fields for compatible benchmark consumers.
- Verified original SAM 3 GGUF conversion and unfused FP32 reference acceptance: 21/21 cases across FP32/CPU, FP16/CPU and FP16/Metal pass unchanged numerical gates, with high-confidence mask IoU 1.0. All 210 tensor snapshots and 18 output masks match the prior custom-container runs byte-for-byte. Checkpoint and reference hashes are recorded in the [GGUF acceptance plan](docs/plans/20261001-020507-gguf-conversion-loading.md); [earlier evidence](docs/plans/20261001-002850-metal-window-cpu-performance-official-weights.md) retains its original format identities.
- Local 2026-10-02 original-weight image acceptance passes all four FP32/FP16 CPU/Metal configurations (28/28 cases), with high-confidence mask IoU 1.0, unchanged gates and zero CPU graph fallback on Metal. FP32 Metal also passes real-checkpoint session lifetime/cache checks. [Fresh four-cell benchmarks](BENCHMARK.md) preserve raw samples and power/thermal conditions separately from historical measurements.
- Original-checkpoint schema-2 FP32/FP16 conversions with all 1,464 tensor payloads verified independently and all 1,133 image-subset tensors preserved exactly. Local Meta selector/preprocessing reproduction and full-profile image checks remain distinct from pending video tracking acceptance.

### Changed

- Video conversion now defaults to hybrid when `--precision` is omitted, following the verified original-reference precision boundary. Image conversion continues to require explicit precision; explicit F16/F32 and the Python conversion API retain their payload policy. A CLI dispatch regression covers all branches without model inference.

- Explicit FP32 Metal requests now initialize Metal and retain F32 weights; Auto FP32 continues choosing CPU. The arithmetic probe and unavailable-device errors remain enforced. This path now passes official local image acceptance without changing numerical gates.
- Image provenance validation also accepts the image subset of complete schema-2 video files, validating the complete full-file inventory and sidecar before running the image comparison.

- Model conversion and loading now use GGUF v3 with named image parameters and the complete tokenizer. Public model/session APIs and FP32/FP16 arithmetic policies remain unchanged; use `.gguf` output paths when converting original checkpoints.
- Public model/session facades now delegate to a SAM 3 adapter, with model-specific schemas, tokenization, transforms, caches, and graphs isolated under `internal/models/sam3/`. Existing application calls remain unchanged.
- SAM 3 fusion, detection, and mask execution are separate internal stages; detection can be reused without running mask decoding. GGML runtime code no longer initializes SAM 3-specific tensors.
- Project-owned headers use unique include guards instead of `#pragma once`. Public value types and task contracts depend only on the standard library; default-constructed model information leaves its architecture unset until a loader identifies it.
- Contributor guidance now includes model namespaces, header guards, authenticated Hugging Face downloads, and separate original-weight versus supplementary validation evidence.
- Renamed the private `sam/detail/` include tree and `sam::detail` namespace to `sam/internal/` and `sam::internal` to make the implementation boundary explicit. Public headers and APIs are unchanged; direct users of old private paths must update their includes.
- Split GGML resources, CPU/Metal drivers, backend selection, and graph execution into separate modules. Model loading consumes the selected driver's FP16 storage policy, and graph statistics identify actual owned backend handles instead of treating unknown devices as CPU. CUDA remains an unimplemented extension.
- Replaced the PABannier GGML fork with official GGML 0.25.3 at `353b63b439f27ab2cc19dac97ab1681ba6d2d084`. The local Metal patch retains precise dense multiplication and float Query attention for SAM's 32/64-dimensional heads; public SAM APIs remain unchanged. Native window operators now eliminate the migration's 56 CPU fallbacks and reduce full-image graph partitions from 118 to 6.
- Corrected the CPU performance interpretation: matched, interleaved old/new GGML runs measured 61.36/62.21 seconds, rather than reproducing the historical 13.6% difference. The historical cause remains unknown; no unsupported CPU kernel fix was applied. See the [diagnosis and validation plan](docs/plans/20261001-002850-metal-window-cpu-performance-official-weights.md).
- Adapted precision requests to `ggml_prec_set_acc` and constructed convolutions with explicit FP32 `im2col`, preserving activation precision across the GGML upgrade. Default upstream convolution helpers otherwise narrow these inputs to FP16.
- Dependency preparation now verifies complete original/patched source trees and the patch itself, including source archives without Git metadata. It repairs generated-source drift while leaving supplied checkouts untouched; custom GGML implementations remain available through caller-owned targets.

### Fixed

- Reject noncanonical object IDs in video diagnostic tensor names, preventing repeated alias reads from retaining duplicate tensor snapshots. Canonical names and inference behavior are unchanged.
- Bind image/video validation batches to model, sidecar, executable and selected-build library hashes, and freeze completed output files before comparison. Changed artifacts or outputs now fail validation; a real replacement regression reproduces the earlier false-acceptance path.
- Serialize original NumPy object IDs as ordinary integers in video propagation traces. A real-ID/tied-candidate/batch-boundary regression reproduces the JSON failure and verifies the corrected output.

- Keep SAM decoder head-16 cross attention on Metal using shared FP32 matmul/softmax; the pinned flash kernel otherwise causes CPU fallback. The numerical/layout regression compares an independent full-key computation.
- Handle empty object batches in the isolated original CPU connected-component fallback, retaining original temporal semantics and recording the adaptation hash.
- Preserve the original SAM 3 video object-score head's `[1,256]` final weight in F32, matching its canonical GGML vector storage. FP16 full-model conversion now succeeds without changing image tensors or singleton-shape rejection; the converter regression reproduces the prior failure and verifies exact payload preservation.
- Reject checkpoint singleton shapes whose FP16 storage policy would produce a GGUF file rejected by the runtime, before publishing either output file.
- Reject overlapping source/output paths before preparing the Python reference copy, avoiding recursive self-copy and source-directory changes.
- Reject overlapping dependency source/output paths before cleanup, including temporary-copy paths. Skip reading obsolete generated copies when changed inputs already require replacement, avoiding failures on unreadable stale cache files.
- Official image-reference export now accepts the checkpoint's known unused interactive-neck tensors only after schema/type validation and records each exclusion. Missing image weights and other unexpected detector tensors still fail; a regression checks both accepted output and rejection boundaries.

### Removed

- Experimental custom `.ggml` container loading and incomplete-tokenizer repair. Existing files are preserved but must be reconverted from the original `.pt` checkpoint; changing their extension is insufficient.

### Security

- Validate custom reference case IDs before creating output paths, and reject symbolic links at generated dependency destinations before cleanup. These checks prevent validation logs or dependency repair from affecting files outside their intended locations.
