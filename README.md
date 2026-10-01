# SAM.cpp

Header-only C++17 SAM 3 text-prompted image segmentation, using GGML on Apple CPU
and Metal. Applications provide RGB pixels and receive owned masks, boxes, and
scores. GGML and its backends are compiled dependencies; checkpoints remain
external files.

[Model sources and conversion](MODEL_ZOO.md) |
[Benchmarks by model, hardware and backend](BENCHMARK.md)

This is an experimental first implementation. All seven original-checkpoint
reference cases pass the frozen tensor and detection gates on FP32/CPU,
FP16/CPU, and FP16/Metal using GGUF. The authorized original checkpoint,
converted weights and pinned Meta reference have recorded hashes. All 210
tensor snapshots and 18 output masks match the previous container's accepted
results byte-for-byte. See the
[GGUF migration and acceptance plan](docs/plans/20261001-020507-gguf-conversion-loading.md).
The [first implementation plan](docs/plans/20260930-192519-sam3-text-image-baseline.md)
records the image milestone and later video/SAM 3.1 work.
The [GGML migration plan](docs/plans/20260930-231832-upstream-ggml-0253.md)
records dependency changes; the
[window and performance plan](docs/plans/20261001-002850-metal-window-cpu-performance-official-weights.md)
records native Metal windows, controlled CPU diagnosis and original-weight acceptance.

On the tested M4 Pro with four CPU threads, five warmed FP16 image runs had
median latency 60.59 seconds on CPU and 6.56 seconds on Metal, with peak process
RSS 4.94/2.67 GB, using the original checkpoint's GGUF conversion and patched
GGML 0.25.3.
See [BENCHMARK.md](BENCHMARK.md) for the result table, protocol and reproduction
commands, and the window/performance plan for the earlier controlled investigations.
This measurement batch does not establish a speedup from changing the container.
These are local single-image measurements, not video frame-rate guarantees.

## Code organization

Public value types and session interfaces are separate from model-specific
loading, tokenization, transforms, and graphs. `include/sam/internal/` contains
non-public implementation; SAM 3 lives under `models/sam3/`. Under
`internal/runtime/ggml/`, `backends/cpu.hpp` and `backends/metal.hpp` own device
initialization and precision/storage policy, while resource management and
graph execution remain shared. Detection and mask decoding are separate stages.
All project-owned headers use include guards.

See [architecture and extension guidance](docs/architecture.md) for the layer
boundaries and planned SAM 2/2.1, GroundingSAM, and DART integration points.
Those models and pipelines are not implemented by this refactor. CUDA is a
documented extension path; only CPU and Metal are currently supported.

## Build

Use CMake 3.20 or newer and a C++17 compiler. Apple builds require the Xcode
Command Line Tools and SDK. The first configure fetches the pinned GGML source,
copies it into the build directory, and applies the included
[Metal precision and window patch](cmake/patches/README.md). Git is required for preparation;
Python is not required to build or use the C++ library.

```sh
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure

cmake -S . -B build/metal -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build/metal --parallel
ctest --test-dir build/metal --output-on-failure
```

Metal source is embedded and compiled by the runtime. A separate offline Metal
compiler is not needed for this route. Use
`-DFETCHCONTENT_SOURCE_DIR_GGML=/absolute/path/to/pinned/ggml` to reuse a local
checkout without modifying it. The compatible dependency is official GGML 0.25.3,
fixed at [`353b63b439f27ab2cc19dac97ab1681ba6d2d084`](https://github.com/ggml-org/ggml/tree/353b63b439f27ab2cc19dac97ab1681ba6d2d084).
This revision also requires the Metal patch above. Use a clean source tree
with build output elsewhere. CMake verifies complete source and patch hashes,
including archive overrides, and records dependency provenance in its copy.
Generated dependency destination paths cannot be symbolic links.
Arbitrary GGML revisions are not a supported compatibility promise.

`SAM_BUILD_TESTS` and `SAM_BUILD_EXAMPLES` default to `ON` for standalone builds
and `OFF` when embedded. Ordinary tests need no checkpoint or network access
after dependency configuration.

## Integrate into an application

```cmake
add_subdirectory(external/SAM.cpp)
target_link_libraries(my_application PRIVATE sam::sam)
```

The `sam::sam` target is an `INTERFACE` library. It propagates C++17, include
directories, and GGML linkage. If the parent already defines a compatible `ggml`
target, SAM reuses it. A build-time API check catches incompatible scheduler
and bounded GGUF reader APIs. The caller must apply the precision patch or supply an equivalent
implementation. Metal initialization executes a small arithmetic probe and
rejects a backend that narrows the required FP32 matrix operands. Full model
compatibility still requires the reference suite.

```cpp
#include <sam/sam.hpp>

sam::Result segment_rgb(const std::vector<std::uint8_t>& rgb,
                        int width, int height) {
    auto model = sam::Model::load("models/sam3-f16.gguf",
                                 {sam::Backend::Auto, 4});
    sam::ImageSession session(model);
    session.set_image({rgb.data(), rgb.size(), width, height,
                       static_cast<std::size_t>(width) * 3});
    return session.segment_text("truck", 0.5f);
}
```

Load the model once and retain the session for repeated use. `set_image` consumes
borrowed, interleaved RGB8 data during the call and caches its visual features.
Set `row_stride` explicitly; padding is allowed. A new prompt reuses those
features. Repeating the same token sequence reuses the last raw prediction, so a
new score threshold only reruns postprocessing.

Each detection has an original-image XYXY box, a score, and a row-major
source-resolution mask containing bytes `0` or `1`. Boxes follow the reference
coordinates and are not clipped to image bounds. `query_index` identifies a
detector query for diagnostics; it is not a tracking ID. No matches produce an
empty `detections` vector. Input violations throw `std::invalid_argument`;
model/backend/execution failures throw `std::runtime_error`.

Sessions retain their model state and own their caches. Separate sessions may
share a model, with execution serialized by one model mutex. A single session
must not be called concurrently. Decoding libraries and Python are unnecessary
for C++ callers supplying pixels directly.

| Weights | CPU | Metal | `Auto` |
| --- | --- | --- | --- |
| FP32 | Implemented | Explicit selection implemented; local model acceptance pending | CPU |
| FP16 | Implemented | Implemented | Compatible Metal when available, CPU when unavailable |

Inspect `model.backend()` or `model.info()` for the resolved backend. Explicit
Metal selection fails when unavailable. Graph execution errors are surfaced;
they do not trigger a silent backend retry. Numerical acceptance status is
recorded in the [FP32 validation plan](docs/plans/20261001-121633-fp32-cpu-metal-validation.md) separately from this implementation matrix.
Official model/Meta comparisons, Metal hardware checks and new benchmarks are
reserved for local execution; no new accepted configuration or measured result
is claimed by the cloud implementation.

`ModelInfo::precision` describes checkpoint storage. CPU loading promotes FP16
values exactly to FP32 to avoid narrowing activations in GGML's half-weight dot
path; the weight buffer is therefore approximately 3.4 GB for either precision.
Metal retains packed FP16 weights and requests precise matrix/attention
arithmetic. `weight_bytes` reports the loaded representation, and runtime buffer
statistics include actual allocation padding. Incompatible Metal precision is
an error even with `Auto`; use a compatible GGML build or explicitly select CPU.
SAM constructs convolutions with FP32 `im2col` explicitly because the pinned
upstream convenience helpers otherwise narrow activations to FP16.

Plain text initially supports printable ASCII and ASCII whitespace, with the
official 32-token truncation behavior. Unicode and HTML/entity-encoded text are
rejected explicitly. For those inputs, call `segment_tokens` with the pinned
official tokenizer's 32 IDs, starting with `49406`, ending with `49407`, and
zero-padded. IDs must be in `[0, 49407]`.

## Checkpoints and command-line example

The reader accepts SAM schema-1 image and experimental schema-2 full video GGUF v3 files in FP32 or mixed FP16.
Schema-2 conversion/loading and internal tracking graphs are implemented;
`VideoSession`, video CLI and full temporal-policy integration are still pending.
`ModelInfo::task` and `profile` identify the file contract, rather than numerical acceptance. See the
[GGUF contract](docs/gguf.md) for required architecture/task metadata, named
parameters, tokenizer and tensor layout. Bounded metadata, canonical encoding,
model schema and all file ranges are checked before backend weight allocation.

Legacy custom `.ggml` files require reconversion from the original `.pt`
checkpoint using the commands below. Renaming a file does not convert it. GGUF
contains the complete canonical tokenizer; incomplete legacy merge lists are
rejected. The retained `ModelInfo::tokenizer_compatibility_repaired` field and
CLI field are false for supported GGUF files.

```sh
build/metal/examples/sam_image \
  --model models/sam3-f16.gguf --image models/fixtures/truck.jpg \
  --text truck --backend metal --threads 4 --score-threshold 0.5 \
  --output build/truck-result --repeat 5
```

The example writes `results.json` and one PNG mask per detection. It refuses to
overwrite an existing output directory. `--repeat` measures warmed full-image
inference and repeated-result-cache calls separately. Cache timing is not the
latency of a new prompt. `timing_ms.warmed_full_image_runs` retains each run's
total/image/text/inference timings; `repeated_result_cache_runs` retains each
cache sample alongside the existing medians. Compare interleaved runs with the
same input, weights, compiler, thread count, and power/thermal conditions; pause
other project computation. Historical medians alone do not establish a regression.
Reported inference stage time includes text encoding;
those two fields must not be added together. `cold_start` measures the first
pipeline call including decoding and model loading; process startup is excluded,
and operating-system file/shader caches may already be warm.

Runtime statistics include cumulative executed CPU/Metal node counts, graph
partitions, explicit session host upload/download bytes, backend buffer sizes, and process peak RSS.
Host byte counts exclude initial weight loading and internal scheduler transfers. Backend buffers and RSS
are different measurements and should not be added together as total memory.
`compute_buffer_bytes` is the largest scheduled graph allocation observed by the
session, separate from its model weights and host-side caches.

The local patch supplies native Metal window partition/restoration. The tested
full image graph executes 3,342 Metal nodes, zero CPU graph nodes, and six graph
partitions. Host preprocessing/postprocessing still run on CPU. The previous
56 window fallbacks and 118 graph partitions are eliminated; see the
[window and performance results](docs/plans/20261001-002850-metal-window-cpu-performance-official-weights.md).

## Convert and generate references

See [MODEL_ZOO.md](MODEL_ZOO.md) for pinned HF/GitHub sources, file hashes,
download/conversion commands and the supported GGUF schema.

Obtain an authorized original `sam3.pt` checkpoint from
[Meta's SAM 3 model page](https://huggingface.co/facebook/sam3). Keep checkpoints,
credentials, and input media outside Git. Runtime inference never downloads them.

With an authenticated HF CLI and an account authorized for the gated model:

```sh
hf auth whoami
sam3_weights_dir=models/official/3c879f39826c281e95690f02c7821c4de09afae7
hf download facebook/sam3 sam3.pt \
  --revision 3c879f39826c281e95690f02c7821c4de09afae7 \
  --local-dir "$sam3_weights_dir"
curl -fL https://raw.githubusercontent.com/facebookresearch/sam3/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  -o "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz"
```

The verified original checkpoint is 3,450,062,241 bytes, SHA-256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
Authentication and model-license access are checked separately.

Use a separate Python 3.12 environment. `tools/requirements.lock` records the
resolved macOS arm64 CPU reference environment. Continue in the same shell with
`sam3_weights_dir` set, install into the environment, then convert:

```sh
python3.12 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r tools/requirements.lock
.venv-reference/bin/python tools/convert_sam3.py \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --precision f32 --output models/sam3-f32.gguf
.venv-reference/bin/python tools/convert_sam3.py \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --precision f16 --output models/sam3-f16.gguf
```

The converter uses the pinned `gguf` Python package and publishes a complete
`.gguf` file with a `.gguf.manifest.json` sidecar. It records source/output and
per-tensor hashes, actual offsets, tokenizer data, conversion options and tool
identities. Existing outputs are never replaced. Only the image detector is
included; C++ inference does not depend on Python.

Check out the official source at
[`2345a4ad109ac29c569da749c91d84f10dc08c40`](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40)
and set `SAM3_SOURCE_DIR` to its absolute path. Its BPE asset is
`sam3/assets/bpe_simple_vocab_16e6.txt.gz`. Copy `assets/images/truck.jpg` and
`assets/images/groceries.jpg` into ignored `models/fixtures/`; the fixed corpus
checks their hashes before inference.

For CPU reference export on macOS, prepare a separate runtime source copy. This
removes unused eager video imports and moves eager positional caches to CPU; the
output directory must be outside the original checkout, which remains untouched.
The exporter also records an unfused FP32
MLP adaptation because the pinned fused helper otherwise forces BF16.
The original checkpoint also contains an unused interactive SAM 2 neck. The
image exporter validates these keys, shapes and floating types against the
pinned schema, excludes them only with interactivity disabled, and records every
excluded tensor. Unknown detector keys and missing image weights remain errors.

```sh
.venv-reference/bin/python tools/prepare_reference_source.py \
  --source "$SAM3_SOURCE_DIR" --output build/reference-runtime/sam3-cpu
.venv-reference/bin/python tools/export_reference.py \
  --sam3-source "$SAM3_SOURCE_DIR" \
  --sam3-runtime-source build/reference-runtime/sam3-cpu \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --cases tests/data/sam3-image-cases.json --device cpu \
  --output models/reference/sam3-f32
.venv-reference/bin/python tools/validate_image.py \
  --build-dir build/cpu --model models/sam3-f32.gguf \
  --reference models/reference/sam3-f32 --backend cpu
```

Run the same validator for FP16/CPU and FP16/Metal using the matching model and
build directory. It checks exact tokens, intermediate tensors, scores, boxes,
masks, and provenance with the plan's fixed tolerances. Missing artifacts fail.
Supplementary references restored from community weights require the explicit
`--allow-supplementary` diagnostic option and cannot satisfy original-checkpoint
acceptance.

To include the official suite in CTest, configure `SAM_REFERENCE_DIR`,
`SAM_REFERENCE_MODEL`, and `SAM_REFERENCE_BACKEND=cpu|metal`, and set
`Python3_EXECUTABLE` to the reference environment's interpreter. Enable it only
after exporting the complete corpus.

## Development

Run the CTest commands above and `.venv-reference/bin/python tools/test_tools.py`.
The tests cover parser/input boundaries, numerical graph regressions, output
handling, backend selection, independent/repeated header inclusion, task
contracts without GGML, two-translation-unit linkage, and native Metal window
layout/padding against CPU output with direct GPU execution. An optional
checkpoint-backed session check exercises model lifetime, different prompts/images,
shared-model session isolation, cache reuse, and the compute-buffer high-water mark:

```sh
build/metal/tests/test_session models/sam3-f16.gguf \
  models/fixtures/truck.jpg models/fixtures/groceries.jpg \
  truck wheel metal 4 build/session-check
```

This is a behavior check, separate from numerical reference acceptance. Check
downstream integration with:

```sh
cmake -S tests/consumer -B build/consumer -DSAM_SOURCE_DIR=../..
cmake --build build/consumer --parallel
ctest --test-dir build/consumer --output-on-failure
```

Follow [AGENTS.md](AGENTS.md): write a timestamped plan before implementation,
maintain [changelog.md](changelog.md), and add tests for consequential behavior
instead of coverage quotas. Model-specific graphs, weight parsing, and GGML
execution have separate internal boundaries for later architectures/backends.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for pinned sources and retained
licenses. Model/media terms are separate from the C++ dependency licenses.
