# First Implementation Plan: SAM 3 Text Image Segmentation

Created: 2026-09-30 19:25:19 Asia/Shanghai

Status: Experimental image implementation available; original-checkpoint numerical acceptance passed on 2026-10-01 for all three supported precision/backend combinations. See the follow-on acceptance record below. Implementation authorized on 2026-09-30; delegated workers use gpt-6.1-sol with xhigh reasoning.

## 1. Outcome and Scope

Deliver a usable C++17 header-only SAM layer that another CMake project can include through `sam::sam`. The first milestone accepts a decoded RGB image and a text prompt, then returns matching masks, boxes, and scores with verified CPU and Metal execution.

The product direction remains text-driven image segmentation and video tracking with SAM 3 and SAM 3.1. This plan makes the first image milestone implementation-ready and defines the boundaries needed for the next two milestones. A working image library must remain useful if video work stops.

Included: SAM 3 image concept segmentation, FP32 CPU reference alignment, FP16 CPU/Metal execution, image-feature reuse, a small image CLI, checkpoint conversion, meaningful tests, and a downstream integration example.

Deferred: video sessions, SAM 3.1 Object Multiplex, interactive point/box prompts, quantization, GGUF support, other model families, other GPU backends, bindings, GUI, and real-time guarantees. Each needs a separate implementation plan.

Working language assumption: validate plain English prompts first. Full Unicode cleaning parity is a later tokenizer task unless the user expands this milestone. Provide a token-ID entry point so host applications can supply official tokenization without changing inference code.

## 2. Verified Starting Point

- The repository contains contributor guidance, a changelog, and the completed workflow-documentation plan. There is no inference implementation, build system, or commit history.
- The local machine is an Apple M4 Pro with 48 GiB unified memory. CMake 4.4.3, Ninja, Homebrew Clang, and Apple command-line Clang are available.
- `xcrun --find metal` fails on this machine. The selected GGML revision supports embedding Metal source and compiling it through the runtime; use that route. A successful Metal model run is still required.
- No checkpoint has been downloaded, no dependency has been built, and no inference or throughput result has been measured during planning.

Use these immutable source baselines:

| Source | Revision | Role |
| --- | --- | --- |
| [Meta SAM 3](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40) | `2345a4ad109ac29c569da749c91d84f10dc08c40` | Behavioral reference and checkpoint mapping |
| [PABannier/sam3.cpp](https://github.com/PABannier/sam3.cpp/tree/416186c501d060df7ca02989d49b38080f5f81f3) | `416186c501d060df7ca02989d49b38080f5f81f3` | Reuse selected SAM 3 graph/converter code |
| [PABannier/ggml](https://github.com/PABannier/ggml/tree/331b9cba52b23d895bc4ad218c007eb5e667540f) | `331b9cba52b23d895bc4ad218c007eb5e667540f` | Initially supported CPU/Metal dependency |
| [ropoctl/sam3cpp](https://github.com/ropoctl/sam3cpp/tree/802f33e3c1d4b892063d069f56816b8b899abcb5) | `802f33e3c1d4b892063d069f56816b8b899abcb5` | Reference for graph decomposition; not a second runtime dependency |

The PABannier converter emits its own version-3 `.ggml` container, not GGUF. Its text preprocessing approximates the official tokenizer. Its image postprocessing adds NMS, whereas the pinned official image processor filters scores without that extra step. Port selectively and compare against Meta outputs.

## 3. Dependency and Packaging Decisions

1. Reuse the inspected SAM 3 graph implementation instead of rewriting matrix/attention kernels. Exclude other model families, GUI, video decoding, tracking heuristics, and upstream debug globals from the first port.
2. Keep all SAM definitions in headers with inline non-template functions and self-contained includes. Do not require an implementation macro or produce a SAM binary library.
3. Use CMake 3.20+ and an `INTERFACE` target named `sam::sam`, requiring C++17 and propagating GGML includes/linkage.
4. Reuse an existing `ggml` target when supplied by the host; otherwise use CMake FetchContent at the pinned revision. Support CMake's `FETCHCONTENT_SOURCE_DIR_GGML` override for an existing local checkout. API/operator checks must reject incompatible dependencies instead of adding a second conflicting GGML copy.
5. Use GGML's backend registry and scheduler through one internal adapter. The model graph must not contain separate CPU and Metal implementations. Set `GGML_METAL_EMBED_LIBRARY=ON` for the standalone Apple build; retain CPU-only builds through `GGML_METAL=OFF`.
6. Keep codecs outside the library. The example may compile the pinned upstream stb implementation in one `.cpp`; users passing RGB memory need neither stb nor Python.
7. Preserve source copyright/license notices and record provenance in `THIRD_PARTY_NOTICES.md`. Keep GGML and adapted C++ notices separate from the [SAM model license](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/LICENSE).

The dependency pin is a compatibility boundary, not a promise to support arbitrary GGML versions. If an embedding application requires an incompatible GGML revision, port and validate the required operations before declaring that combination supported.

## 4. Architecture and Ownership

```text
Host RGB buffer --> preprocessing --> vision encoder --> cached image features
Host text/token IDs --> tokenizer --> text encoder -------+
                                                         |
cached image features --> fusion + detector + mask head --+--> postprocessing
                                                                  |
                                                       masks / boxes / scores

ModelDefinition --> weight validation and graph construction
GgmlRuntime -----> backend selection, buffers, scheduling, execution
ImageSession ---> image cache, prompt cache, per-call working state
```

Keep three explicit extension boundaries:

| Boundary | First implementation | Concrete future variation | Cost/control |
| --- | --- | --- | --- |
| `GgmlRuntime` | CPU and Metal selection, allocation, execution | Another GGML backend | One adapter; do not duplicate GGML's plugin system |
| `ModelDefinition` | SAM 3 tensor schema, preprocessing, graphs | SAM 3.1 neck/memory/decoder changes | One internal interface is acceptable; expose only implemented capabilities |
| Weight reader | Pinned SAM 3 container version 3 | Another container/model schema | Produce validated named tensors; keep file parsing out of graph code |

Model weights are immutable after loading. Sessions own image features, tokenizer caches, and temporary buffers. A session retains the model state it needs. Serialize execution for sessions sharing one model initially; independent model instances remain separate. Document this concurrency ceiling and mark the implementation with a `ponytail:` comment naming independent execution contexts as the upgrade path.

The same session is not concurrently callable. Replacing an image invalidates its dependent caches. Additional prompts on the same image must not rerun the vision encoder. Cache the last prompt only; a general cache manager is outside this milestone.

## 5. Public Contract

Use namespace `sam`; do not expose GGML tensor pointers as public results.

| Surface | Required behavior |
| --- | --- |
| `ImageView` | Borrowed RGB8 pointer, buffer byte count, width, height, and row stride; validate bounds/overflow before reading; consume during image preparation |
| `BackendOptions` | `Auto` by default, or explicit `Cpu`/`Metal`; four CPU threads by default, configurable |
| `Model::load` | Validate and load an external checkpoint; expose resolved backend and supported architecture/capabilities |
| `ImageSession::set_image` | Prepare and cache one image's visual features; retain no borrowed input pointer after return |
| `ImageSession::segment_text` | Accept a supported plain English prompt and score threshold, default 0.5 |
| `ImageSession::segment_tokens` | Accept the model's 32 token IDs with validated vocabulary range and padding convention |
| Result | Owned detections containing source-image XYXY boxes, scores, and source-resolution binary masks with values 0/1; zero matches is an empty result |

Distinguish failed execution from successful empty results. Use standard exceptions: invalid arguments for input-contract violations and runtime errors for model-file/backend/execution failures. Explicit Metal selection must fail clearly when unavailable; `Auto` may select CPU during initialization and must report the resolved backend. Do not silently retry a failed model graph on another backend.

The initial supported combinations are FP32/CPU and FP16/CPU or Metal. `Auto` selects CPU for FP32 and prefers available Metal for FP16. Explicit FP32/Metal is outside the verified matrix and must report an unsupported combination.

Plain-text support covers printable ASCII plus ASCII whitespace, case folding, contractions, punctuation, and digits. Reject unsupported Unicode or HTML/entity-encoded input with a clear message; callers can use token IDs for those cases. Match the official 32-token truncation/EOT behavior. Full Unicode support must implement the official cleaning contract rather than silently treating every non-ASCII byte as a letter.

## 6. Numerical Pipeline and Weights

- Use the pinned official SAM 3 image model with instance interactivity disabled as the oracle.
- Resize RGB input directly to 1008 x 1008 and normalize with channel mean/std 0.5. Match interpolation, antialiasing, rounding, layout, and coordinate restoration; do not substitute letterboxing.
- Preserve the visual backbone, feature neck, text encoder, empty geometry/prompt path, fusion encoder, detector, presence scoring, and segmentation head needed for text inference.
- Compute confidence from class and presence probabilities. Follow official filtering and mask-logit resizing/thresholding. Do not carry upstream's extra image NMS into the parity path.
- Preserve original query indices in validation artifacts so tensor/mask comparisons do not depend on detection ordering. These indices are not video object IDs.
- Read only the defined version-3 schema and support FP32/FP16 initially. Validate magic/version, tensor names, dimensions, byte ranges, duplicates, required tensors, and tokenizer contents before graph allocation. Recognize unused tracker tensors explicitly; unknown incompatible model layouts fail.
- Adapt the pinned converter into `tools/convert_sam3.py`. Generate tokenizer vocabulary/merges from the official BPE asset, preserve the required FP32 tensors, and emit a manifest with source/output SHA-256, model revision, conversion options, and tensor inventory. Publish converted files only after successful validation.

Obtain the original checkpoint through the user's authorized [Meta model access](https://huggingface.co/facebook/sam3), or accept an existing local checkpoint. Store it under ignored `models/`. Credentials stay in the user's credential store; runtime inference performs no downloads.

Conversion/reference tools run in an isolated Python environment. The reference exporter calls the pinned model in evaluation mode, disables compilation/autocast and any TF32 acceleration for the FP32 oracle, and records package versions and the reference device. The official image builder exposes a CPU path, so try that path first. If its full dependency chain cannot run locally, export on a supported official environment and consume the recorded artifacts on the Mac. Missing weights or reference outputs block numerical acceptance; synthetic tests alone cannot complete this milestone.

## 7. Files and Implementation Sequence

This milestone is expected to touch more than eight files. Keep the split at module boundaries; it does not create services.

| Area | Planned files |
| --- | --- |
| Build/dependencies | `CMakeLists.txt`, `cmake/ggml.cmake`, `.gitignore`, `THIRD_PARTY_NOTICES.md`, retained license files |
| Public headers, under `include/sam/` | `sam.hpp`, `types.hpp`, `model.hpp`, `image_session.hpp` |
| Internal headers, under `include/sam/detail/` | `ggml_runtime.hpp`, `weights.hpp`, `tokenizer.hpp`, `image_ops.hpp`, `model_definition.hpp`, `models/sam3.hpp` |
| Example/tools | `examples/CMakeLists.txt`, `examples/image.cpp`, `tools/convert_sam3.py`, `tools/export_reference.py`, `tools/validate_image.py`, `tools/requirements.lock` |
| Tests, under `tests/` | `CMakeLists.txt`, `test_contracts.cpp`, `test_image.cpp`, `test_header_only_main.cpp`, `header_only_other.cpp`, `consumer/CMakeLists.txt`, `consumer/main.cpp`, `data/sam3-image-cases.json` |
| Documentation | `README.md`, this plan, `changelog.md` |

Implementation steps inside one usable image milestone:

- [x] Add the pinned dependency integration, header-only target, ignored artifact directories, notices, and consumer build check.
- [x] Add the validated weight reader/converter and runtime ownership. Verify missing/truncated/incompatible files before model execution.
- [x] Port preprocessing, tokenizer, vision/text encoders, detector, and mask generation. Establish FP32 CPU alignment stage by stage using supplementary same-weight references; original-checkpoint acceptance was verified separately on 2026-10-01.
- [x] Connect the same graphs to Metal with the embedded shader route. Check actual operator placement, numerical agreement, and cleanup on failure against the supplementary same-weight corpus.
- [x] Finish image caching, the CLI, owned outputs, fixtures, and differential validation.
- [x] Run the supplementary acceptance matrix, record benchmarks and limitations, update README/changelog, and mark results in this plan.
- [x] Obtain the original authorized checkpoint and complete conversion/reference acceptance on all three precision/backend combinations (2026-10-01).

The CLI executable is `sam_image`, with `--model`, `--image`, `--text`, `--backend auto|cpu|metal`, `--threads`, `--score-threshold`, `--output`, and `--repeat`. It writes metadata to `results.json` and per-instance PNG masks to a new output directory, refusing to overwrite an existing result directory.

## 8. Testing and Acceptance

Use CTest plus small test executables with explicit failure checks that remain active in Release builds. Do not rely on `assert` when `NDEBUG` disables it. No coverage quota and no tests that merely repeat getters or graph-construction code.

| Layer | Required evidence |
| --- | --- |
| Fast contract checks | Invalid/truncated model data, image stride/size overflow, invalid prompt/token inputs, and successful empty results |
| Header integration | Two translation units include/call the library; a separate CMake consumer links only `sam::sam` on CPU and Metal builds |
| Preprocessing/tokenizer | Golden token IDs match exactly; normalized resize error is at most one source-byte step (2/255); padding/truncation and non-square images are covered |
| Model differential checks | Compare vision features, text features, boxes, presence/class logits, and low-resolution mask logits to pinned official outputs |
| Session behavior | Repeated prompts reuse image encoding; changed images invalidate caches; alternating sessions do not exchange cached state |
| Metal | Large matrix/attention stages actually execute on Metal; report CPU graph partitions and transfers; initialization alone is not acceleration evidence |

Reference corpus: use the pinned upstream `assets/images/truck.jpg` with `truck`, `wheel`, and `purple elephant`; use `assets/images/groceries.jpg` with `fruit`, `bottle`, and `purple elephant`. Expected detections come from the oracle, not assumptions about those phrases. Add a deterministic crop/resize case and synthetic resize/layout fixtures. Record input hashes, provenance, prompt, reference version, thresholds, and tensor shapes in `tests/data/sam3-image-cases.json`. Fetch upstream media explicitly into ignored local fixture storage; commit only fixtures with documented redistribution rights.

Initial acceptance targets, fixed before comparing C++ results:

- Define normalized L2 as `norm(actual - reference) / norm(reference)`. Require at most 1e-3 for FP32 CPU and 2e-2 for FP16 CPU/Metal. When the reference norm is at most 1e-12, require maximum absolute error at most 1e-5 instead.
- For reference detections with score at least 0.6, mask IoU at least 0.98 for FP32 and 0.95 for FP16, score absolute error at most 0.02, and box-coordinate error at most 1% of the corresponding image dimension.
- Reference queries at or below 0.4 must not become detections at the 0.5 threshold. Report threshold-adjacent queries separately and still check their tensor errors.
- Compare CPU and Metal using the same converted FP16 weights. Freeze the fixture manifest before judging results; tolerance changes require a written explanation and review.
- After warm-up, repeated same-size inference must not accumulate live GGML buffers or retained session history. Report process peak RSS and backend buffer allocations separately; neither is a universal total-memory measurement.

These are proposed correctness gates, not measured achievements or guarantees for arbitrary data. Fix systematic mismatches at the responsible stage instead of broadly relaxing tolerances.

## 9. Verification Commands

These commands configure the implementation. `SAM_BUILD_TESTS` and `SAM_BUILD_EXAMPLES` default on for standalone builds and off when embedded. Use `SAM_REFERENCE_DIR` with `SAM_REFERENCE_MODEL` and `SAM_REFERENCE_BACKEND` to enable the optional reference suite; ordinary tests must not download checkpoints.

```sh
rtk proxy cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DSAM_BUILD_TESTS=ON -DSAM_BUILD_EXAMPLES=ON
rtk proxy cmake --build build/cpu
rtk proxy ctest --test-dir build/cpu --output-on-failure

rtk proxy cmake -S . -B build/metal -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DSAM_BUILD_TESTS=ON -DSAM_BUILD_EXAMPLES=ON
rtk proxy cmake --build build/metal
rtk proxy ctest --test-dir build/metal --output-on-failure

rtk proxy cmake -S tests/consumer -B build/consumer -DSAM_SOURCE_DIR=../..
rtk proxy cmake --build build/consumer
rtk git diff --check
```

The consumer resolves `SAM_SOURCE_DIR` relative to its source directory. It uses `add_subdirectory` and links only `sam::sam`.

The reference suite is mandatory for milestone acceptance. Enable it with `-DSAM_REFERENCE_DIR=<absolute-export-directory> -DSAM_REFERENCE_MODEL=<absolute-model-file> -DSAM_REFERENCE_BACKEND=cpu|metal`; set `Python3_EXECUTABLE` to the isolated reference interpreter. Missing required files are failures when enabled, not passing skips.

Run these tools after activating the isolated reference environment and obtaining the checkpoint, official BPE asset, and corpus inputs. Set `SAM3_SOURCE_DIR` to the pinned official checkout. On macOS, first run `tools/prepare_reference_source.py --source "$SAM3_SOURCE_DIR" --output build/reference-runtime/sam3-cpu` and pass `--sam3-runtime-source build/reference-runtime/sam3-cpu` to the exporter:

```sh
rtk proxy python3 tools/convert_sam3.py --checkpoint models/sam3.pt --bpe models/bpe_simple_vocab_16e6.txt.gz --precision f32 --output models/sam3-f32.ggml
rtk proxy python3 tools/convert_sam3.py --checkpoint models/sam3.pt --bpe models/bpe_simple_vocab_16e6.txt.gz --precision f16 --output models/sam3-f16.ggml
rtk proxy python3 tools/export_reference.py --checkpoint models/sam3.pt --cases tests/data/sam3-image-cases.json --device cpu --output models/reference/sam3-f32
rtk proxy python3 tools/validate_image.py --build-dir build/cpu --model models/sam3-f32.ggml --reference models/reference/sam3-f32 --backend cpu
rtk proxy python3 tools/validate_image.py --build-dir build/metal --model models/sam3-f16.ggml --reference models/reference/sam3-f32 --backend metal
rtk proxy build/metal/examples/sam_image --model models/sam3-f16.ggml --image models/fixtures/truck.jpg --text truck --backend metal --output build/truck-result --repeat 5
```

`validate_image.py` drives the `test_image` differential executable built by CMake, compares its tensor dumps with the reference bundle, writes a metrics report, and returns nonzero on failure. The FP16/CPU path must also pass the same reference suite before acceptance.

Benchmark one cold start and at least five warmed runs, separating complete image inference from prompts on cached image features. Report median latency, stage times, model hash, precision, backend, thread count, image dimensions, and memory. Establish a local baseline without promising an FPS target.

## 10. Follow-On Milestones

| Milestone | Independently useful delivery | Required extension and acceptance |
| --- | --- | --- |
| M1, this plan | SAM 3 text image segmentation on CPU/Metal | All gates above, header-only consumer integration, documented model compatibility |
| M2, separate plan | SAM 3 text-driven video tracking over host-supplied frames | `VideoSession` owns temporal memory and IDs; implement detection/tracker association and object lifecycle; validate entry, occlusion, disappearance, reappearance, and bounded retained history |
| M3, separate plan | SAM 3.1 Object Multiplex tracking | Add the actual model schema, neck/memory/decoder changes, and bucket assignment; test object counts around bucket boundaries (1, 16, 17), removal/reassignment, accuracy, latency, and memory |

M2 must preserve text-driven discovery of new objects, not merely propagate masks selected on the first frame. M3 must use the new checkpoint and tracking architecture; renaming SAM 3 output is not SAM 3.1 support. The [official release notes](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/RELEASE_SAM3p1.md) and [model builder](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model_builder.py) define that boundary.

## 11. Risks, Recovery, and Completion

| Risk | Response |
| --- | --- |
| GGML fork is incompatible with a host application's GGML | Support the pinned combination first; test a supplied target explicitly; do not hide duplicate runtime libraries |
| Approximate tokenizer/preprocessing changes masks | Exact token fixtures, stage comparisons, explicit language scope, and token-ID input |
| Missing offline Metal compiler or runtime shader failure | Use the inspected embedded-source build; retain CPU diagnosis; Metal acceptance stays incomplete until a real run passes |
| Checkpoint/reference access is unavailable | Continue build/contract work, report the missing prerequisite, and keep numerical acceptance incomplete |
| Header-only ODR or compile-time cost | Self-contained inline definitions, two-TU/consumer checks, and compile measurements before adding packaging variants |
| Video state is forced into image sessions | Keep mutable session ownership separate and implement video under its own plan |
| Performance is insufficient | Use measured stage/backend data to choose optimizations; quantization is a later change after numerical alignment |

The most fragile premise is that the pinned community graph implementation can reach the official numerical gates with bounded corrections. If it cannot, keep the selected API/build structure and replace failing graph modules against the pinned official reference. Do not declare parity from visually plausible masks.

Implementation adds source and ignored local artifacts; no service deployment or data migration is involved. Recovery is a scoped revert of implementation changes. Preserve the external original checkpoint and provenance; regenerated outputs are disposable.

Completion requires a working downstream consumer, passing fast and reference tests on CPU and Metal, recorded correctness/performance results, and accurate documentation/changelog entries. Local checks do not imply remote CI passed. Commit, push, or publish only when separately requested.

## Planning Record

- [x] Read repository constraints and verify the current working tree.
- [x] Inspect upstream code, dependency revisions, model format, tokenizer, and Metal build route.
- [x] Define M1 contracts, file ownership, implementation sequence, tests, and M2/M3 boundaries.
- [x] Implementation and supplementary model validation.
- [x] Original-checkpoint model validation (2026-10-01; follow-on record below).

The initial planning task wrote this document and its changelog entry. Filename, document structure, local links, whitespace, required-scope checks, pinned revision identifiers, and English punctuation checks passed. The user subsequently authorized implementation with GPT-6.1-sol subagents at xhigh reasoning.

## Implementation Record

Work began on 2026-09-30 with five delegated areas: graph/runtime, input/weight contracts, tooling, CMake/example integration, and reference execution. The coordinator owns cross-module verification and documentation. This record separates implemented behavior from milestone acceptance.

### Verified foundation

- C++17 Release builds succeeded on Apple M4 Pro, macOS arm64, AppleClang 21, with both CPU-only and embedded Metal configurations.
- All five CTest checks passed in both builds, including backend selection. Separate CPU and Metal consumers compiled and passed while reusing a caller-provided GGML target. The two-translation-unit test exercised header-only linking.
- Five Python tooling regressions passed in the isolated Python 3.12 environment. These checks do not constitute checkpoint conversion or model parity evidence.
- The actual Metal backend executed the image pipeline and produced an owned mask. The preliminary run recorded 6,642 non-view Metal nodes and zero CPU nodes across two complete forwards. A lower-level Metal ADD probe independently passed 44 executed cases.

### Corrections supported by execution

- Preserved FP32 in the 2x2 transposed convolution instead of implicitly casting its kernel to FP16.
- Initialized every repeated-name decoder zero input, not just the first matching tensor.
- Promoted FP16 reference-point activations before mixed-type coordinate matrix multiplication; the pinned CPU kernel otherwise rejected the graph.
- Copied independently clamped inverse-sigmoid branches because this GGML revision's clamp mutates its input. The endpoint regression remains active in Release.
- Selected the pinned Metal registry by `MTL`; its device is named `MTL0`.
- Added unsupported-operation diagnostics before the scheduler's native assertion.
- Recognized and reported the narrowly validated legacy BPE layout, restoring six missing merges and canonical special-token spellings. Newly converted files retain the complete tokenizer.

### Initial reference provenance (2026-09-30)

Original Meta checkpoint access was unavailable during initial implementation, so that stage made no original-checkpoint or full M1 acceptance claim. The later original-checkpoint record below supersedes that access limitation. Public containers from Hugging Face `PABannier/sam3.cpp`, revision `a3892b63b918e872671322e116982a8910f0ffb7`, were supplementary inputs:

| Container | SHA-256 |
| --- | --- |
| FP32 | `24ae10151d7b81b909b2eabba4f8642db5f50e8f931a3277d98a59c630879f7c` |
| FP16 | `1c8cef822a6f0f0908c8e7c51139c1061a842ebfbc88e8b0bb1d8e342ec50f8e` |

A supplementary image checkpoint was restored from the FP32 container with recorded mapping/substitutions and hashes. The pinned Meta graph produced all seven frozen reference cases and ten tensors per case. Its bundle explicitly records `reference_kind=supplementary-converted-weights` and `eligible_for_milestone=false`; default validation rejects this provenance for official acceptance.

`tools/prepare_reference_source.py` reproduces an isolated CPU source copy with hashed modifications for unused eager video imports, CPU positional caches, and an unnecessary pinned-memory hint. The exporter uses an explicitly recorded unfused FP32 MLP because the pinned fused helper forces BF16, and preserves complex RoPE buffers during float conversion. Neither the shared source checkout nor source containers are edited.

Initial numerical comparison passed text encoding and the high-confidence truck mask/score/box checks, but failed intermediate image tensors. The measured preprocessing discrepancy exposed intermediate uint8 rounding and float normalization differences. After correction, ten deterministic random/full/thin/padded input cases matched the pinned ARM Torchvision preprocessing bit for bit across 30,481,920 values. Reproducible evidence is under `build/reference-runtime/preprocess-parity/`.

### Initial supplementary acceptance matrix, before precision correction

The following results use this project's converter outputs and the complete frozen seven-case corpus, with no tolerance changes:

| Storage/backend | Cases passing every tensor/detection gate | Worst tensor normalized L2 | Minimum high-confidence mask IoU |
| --- | --- | --- | --- |
| FP32 / CPU | 7 / 7 | 0.000570083 | 1.0 |
| FP16 / CPU | 5 / 7 | 0.0347920 | 0.9999275 |
| FP16 / Metal | 2 / 7 | 0.0794922 | 0.9992634 |

Reports are `build/validation-cpu-f32/metrics.json`, `build/validation-cpu-f16/metrics.json`, and `build/validation-metal-f16/metrics.json`. The FP16 CPU failures concern raw boxes, class logits, or mask logits for the groceries fruit/bottle cases. Metal also exceeds raw-tensor gates for wheel and negative prompts. All six high-confidence detections meet mask, score, and box gates, and empty-result selection agrees. These facts do not override the failed tensor gates.

Converted FP32 SHA-256: `2e8d55036d71f2c46fb16bd6538086522ae84e70710208ef9ad3b7441e673c67`. Converted FP16 SHA-256: `ad3619fa725efce76bec6214a0aaa2d66f901859fbab00fdf2b711b0d4dcce7f`. Both conversion roundtrips independently checked all 1,133 tensor records: FP32 values are byte-identical to their source, and FP16 values exactly match the declared per-tensor rounding policy. The complete tokenizer is exact and needs no legacy repair.

The same converted FP16 file was also compared directly across CPU and Metal. Ten of 70 tensor comparisons exceed 0.02; the maximum is 0.0579019. All six high-confidence detections agree, with minimum IoU 0.9992634 and no threshold-selection differences. The reproducible artifact-only comparison is in `build/backend-agreement/`.

Further isolation proves that stored-weight rounding alone is insufficient to explain these failures: promoting those exact FP16 values into the pinned Meta FP32 runtime passes all seven frozen FP16 cases, with maximum normalized L2 0.0130666 and high-confidence mask IoU 1.0. The report is `build/reference-runtime/f16-weight-floor-metrics.json`. A small Metal matrix probe separately demonstrates activation rounding in the pinned SIMD-group half-operand kernel; setting `GGML_PREC_F32` does not change that kernel's result. Backend arithmetic needs further work before the FP16 matrix can be accepted.

The [compute-precision correction plan](20260930-205540-ggml-fp16-compute-precision.md) now covers CPU value promotion and an isolated, reproducible Metal dependency patch. The matrix and benchmark figures below describe the implementation before that correction; new results must be recorded separately.

### Initial session behavior and performance, before precision correction

Checkpoint-backed CPU and Metal checks passed changed-prompt image-cache reuse, repeated-result reuse, image replacement/restoration, and independence of two sessions sharing one model. Ten alternating prompts retained the same compute-buffer high-water mark. CPU peak RSS remained 3,230,351,360 bytes in the observed repetition window; the Metal run increased from 2,971,500,544 to 2,979,577,856 bytes. These are finite-run observations, not proof of a universal memory bound.

Benchmarks used Apple M4 Pro, 48 GiB RAM, macOS 27.0 (26A428), AppleClang 21, four CPU threads, the converted FP16 hash above, `truck.jpg` at 1800 x 1200, prompt `truck`, threshold 0.5, and five warmed complete-image runs. CPU and Metal benchmark processes ran sequentially after corpus inference; ordinary desktop activity remained present.

| Metric | CPU | Metal |
| --- | --- | --- |
| First pipeline call, including decode/load | 22.937 s | 7.032 s |
| Median warmed full-image inference, five runs | 22.414 s | 6.509 s |
| Median identical-prompt result-cache call | 7.549 ms | 7.579 ms |
| Peak process RSS | 3,024,060,416 bytes | 2,662,891,520 bytes |
| GGML weight buffer | 1,795,849,440 bytes | 1,795,849,440 bytes |
| Largest scheduled compute allocation | 615,776,256 bytes | 860,440,896 bytes |

Receipts: `build/benchmark-cpu-f16/results.json` and `build/benchmark-metal-f16/results.json`. First-call timing excludes process startup, and OS file/shader caches may be warm. Node/transfer counters are cumulative across six forwards; the Metal receipt records 19,926 executed Metal nodes and zero CPU graph nodes. Host preprocessing/postprocessing still run on CPU. Explicit host-copy bytes exclude internal scheduler transfers.

An independent Metal session run measured a new prompt on cached visual features at 617 ms and a ten-prompt median of 607 ms (`build/session-metal/results.json`). This is distinct from identical-prompt result-cache timing. These measurements precede the compute-precision correction below.

### Corrected supplementary acceptance matrix

The [precision correction](20260930-205540-ggml-fp16-compute-precision.md) preserves the external checkpoint format and all frozen acceptance targets. CPU allocation now promotes stored FP16 values exactly to FP32. Shared graph execution requests FP32 matrix/attention arithmetic; convolution inputs retain FP32 precision. The isolated GGML Metal patch honors those requests, including short-query attention.

| Storage/backend | Cases passing every tensor/detection gate | Worst tensor normalized L2 | Minimum high-confidence mask IoU |
| --- | --- | --- | --- |
| FP32 / CPU | 7 / 7 | 0.000570083 | 1.0 |
| FP16 / CPU | 7 / 7 | 0.0130565 | 1.0 |
| FP16 / Metal | 7 / 7 | 0.0130338 | 1.0 |

Receipts: `build/validation-precise-cpu-f32/metrics.json`, `build/validation-precise-cpu-f16/metrics.json`, and `build/validation-precise-metal-f16-patched/metrics.json`. The same converted FP16 file also passes all 70 direct CPU/Metal tensor comparisons: maximum normalized L2 0.000163574, minimum mask IoU 1.0, maximum score difference 3.58e-7, and zero selection differences (`build/backend-agreement-precise/agreement.json`). All receipts remain explicitly ineligible for original-checkpoint acceptance.

The corrected Metal run executes 3,342 non-view GPU nodes and zero CPU graph nodes per inference. Its packed weight buffer remains 1,795,849,440 bytes. CPU FP16 loading now allocates 3,369,375,008 bytes for promoted weights; the container remains FP16. Memory and latency from the earlier implementation must not be reused for this corrected path.

### Corrected performance and final checks

Using the same M4 Pro, four threads, FP16 checkpoint hash, 1800 x 1200 truck input,
threshold, and five warmed runs described above, the corrected CPU and Metal
benchmark processes ran sequentially:

| Metric | CPU | Metal |
| --- | --- | --- |
| First pipeline call, including decode/load | 62.233 s | 17.724 s |
| Median warmed full-image inference, five runs | 59.288 s | 6.836 s |
| Median identical-prompt result-cache call | 10.657 ms | 10.692 ms |
| Peak process RSS | 4,834,148,352 bytes | 2,627,977,216 bytes |
| GGML weight buffer | 3,369,375,008 bytes | 1,795,849,440 bytes |
| Largest scheduled compute allocation | 961,062,048 bytes | 1,245,268,288 bytes |

Receipts: `build/benchmark-precise-cpu-f16/results.json` and
`build/benchmark-precise-metal-f16/results.json`. The Metal first-call figure
includes 10.968 seconds of model/backend initialization; shader/cache state can
change startup timing. Across six forwards the corrected Metal receipt records
20,052 GPU nodes and zero CPU graph nodes. These measurements do not promise
real-time video or describe a universal total-memory footprint.

Final CPU and Metal builds pass all seven CTest checks. Independent consumers
using caller-owned, prepared GGML targets pass both linking and precision checks
(two tests per configuration). Five Python tooling regressions pass. All three
shared upstream checkouts remain clean; checkpoints and generated artifacts are
ignored. No commit, push, release, or remote CI run was performed.

The corrected Metal two-session check also passes image replacement/restoration,
changed prompts, result reuse, and shared-model isolation. Ten alternating
prompts have median cached-vision latency 711.777 ms; a first changed prompt took
785.367 ms. Compute-buffer high-water remains 1,245,268,288 bytes. Peak RSS rises
from 2,912,206,848 to 2,920,267,776 bytes over this finite window
(`build/session-precise-metal/results.json`). The corrected CPU benchmark also
executes six image replacements and five result-cache calls in one session.

Implementation and supplementary validation are complete. The remaining M1 gate
is obtaining the original authorized checkpoint and rerunning the recorded
conversion/reference workflow; none of the supplementary reports replaces it.

### Original-checkpoint acceptance (2026-10-01)

Authenticated download of `facebook/sam3` revision
`3c879f39826c281e95690f02c7821c4de09afae7` supplied the original `sam3.pt`, SHA-256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
Fresh FP32/FP16 conversion and the pinned Meta unfused-FP32 oracle retain this
provenance. The full frozen corpus was rerun without supplementary acceptance.
The 22 known unused interactive-neck tensors are schema-validated and recorded.

| Original-weight configuration | Cases | Maximum normalized tensor L2 | Minimum high-confidence mask IoU |
| --- | --- | --- | --- |
| FP32 / CPU | 7 / 7 | 0.000570083 | 1.0 |
| FP16 / CPU | 7 / 7 | 0.0130565 | 1.0 |
| FP16 / Metal | 7 / 7 | 0.0129848 | 1.0 |

All token, input, tensor, score, box, mask and low-confidence detection gates
are unchanged. Reports are in `build/perf-repair/official-validation-*/metrics.json`.
The final native-window Metal graph executes 3,342 GPU nodes, zero CPU graph
nodes and six graph partitions; the original-weight session ownership/cache
check also passes. Fresh CPU/Metal CTest passes 9/9 each, caller-owned consumers
3/3 each, and the isolated Python tooling suite 6/6.

The original-weight numerical acceptance gap is closed. Historical supplementary
receipts above retain their original provenance. See the
[follow-on plan](20261001-002850-metal-window-cpu-performance-official-weights.md)
for exact identities, CPU timing limitations and final performance records.
Video, SAM 3.1 and additional model/backend milestones remain separate work.
