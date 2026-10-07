# SAM.cpp

A C++17 inference library for the Segment Anything model families. The project
is designed for multiple SAM variants and platforms, with task-specific model
adapters and shared execution backends. The integration layer is header-only;
GGML is a compiled dependency and model weights remain external.

The current implementation provides SAM 3 text-prompted image segmentation and
experimental forward video tracking on CPU, Apple Metal and NVIDIA CUDA.
Validation covers macOS CPU/Metal and Linux x86_64 CUDA on an RTX 4090, with
F32/F16 image checks on the same Linux CPU/BLAS host. Other operating systems,
CPU/GPU architectures and model adapters need their own builds and numerical
validation. See the model catalog for task and weight-profile scope.

[Models and precision](MODEL_ZOO.md) |
[Download and conversion](docs/models/sam3-details.md) |
[Quantized models](docs/quantization.md) |
[Visual examples](docs/visual-examples.md) |
[Performance](BENCHMARK.md) |
[Architecture](docs/architecture.md)

## Current capabilities

| Task | Model / weights | Backends |
| --- | --- | --- |
| Text-prompted image segmentation | SAM 3 F32 or mixed F16/F32 | CPU, Metal, CUDA |
| Text-prompted image segmentation | SAM 3 vision-only Q8_0, Q6_K, Q5_K, Q4_K | CPU, Metal, CUDA |
| Text-prompted image segmentation | SAM 3 full-component linear Q8_0, Q6_K, Q5_K, Q4_K | CPU, Metal, CUDA |
| Forward video tracking | SAM 3 F32 or hybrid `visual-tracker-f32-v1` | CPU, Metal, CUDA |

Video conversion defaults to hybrid weights. Quantized files currently support
images only. SAM 3 also supports custom vision/text/fusion/decoder selections;
repository numerical acceptance covers the fixed vision and full presets.
Custom combinations, F16 video and legacy `image-linear-*` profiles remain
diagnostic configurations. See the model catalog for the current status of
other SAM variants and composed pipelines.

## Build

Use CMake 3.20 or newer, Git, and a C++17 compiler. The initial configuration
fetches the pinned GGML source and prepares a separate copy with this project's
required patches. Python is needed for conversion and reference tools, not for
building or running C++ inference.

```sh
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure
```

On macOS, install the Xcode Command Line Tools and use the Metal build:

```sh
cmake -S . -B build/metal -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build/metal --parallel
ctest --test-dir build/metal --output-on-failure
```

For CUDA, install the NVIDIA driver and CUDA Toolkit, then build
with an architecture matching your GPU (`89` below is an RTX 4090 example):

```sh
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_METAL=OFF -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSAM_REQUIRE_CUDA_TESTS=ON
cmake --build build/cuda --parallel
ctest --test-dir build/cuda --output-on-failure
```

`SAM_REQUIRE_CUDA_TESTS=ON` makes a missing GPU fail the CUDA tests. Otherwise,
CUDA checks skip when no device is visible. CUDA requires the project's GGML
precision/window patch and defaults to `GGML_CUDA_GRAPHS=OFF`. Quantized CUDA
uses the distinct `ggml-quantized-cuda-native-v1` arithmetic profile.

Select CUDA explicitly with `--backend cuda --cuda-device 0`, or
`sam::BackendOptions{sam::Backend::Cuda, 4, 0}`. The index is relative to
`CUDA_VISIBLE_DEVICES`. `Auto` keeps its existing CPU/Metal selection policy.
CUDA rejects unsupported operators and CPU compute fallback. Model information
and CLI JSON include `device_name` and `cuda_device`; `cuda_nodes` reports actual
scheduled compute work.

CUDA defaults to `--cuda-compute f32`: F16 weight storage alone does not enable
F16 matrix arithmetic. Select `--cuda-compute f16` for reduced dense inputs with
F32 accumulation/output and fused reduced-input attention. In C++, set
`BackendOptions::cuda_compute = sam::CudaComputeMode::F16` with an explicit CUDA
backend. Weight files and video state formats stay the same. The runtime reports
`ggml-cuda-f16-v1`, or `ggml-quantized-cuda-f16-v1` for quantized weights.
This mode is qualified by final segmentation/tracking quality; intermediate
tensors are diagnostic. See [precision details](MODEL_ZOO.md#precision-contract)
and [validation](docs/validation.md).

CUDA also selects execution strategies by storage and compute mode: quantized
image models in default compute mode combine prediction stages to reduce host
transfers, while F16 compute uses fused tracker memory attention. All modes share
batched concat launches. These strategies require no additional command-line
flags; measured latency and memory are in [the benchmarks](BENCHMARK.md).

Metal shaders are embedded and compiled at runtime. CPU builds can use a
registered GGML BLAS backend; add `-DGGML_BLAS=OFF` for native CPU execution.
Apple Accelerate manages its own SGEMM threads. For comparable CPU timing,
set `VECLIB_MAXIMUM_THREADS` before starting the process.

`SAM_BUILD_TESTS` and `SAM_BUILD_EXAMPLES` default to `ON` for standalone builds
and `OFF` when embedded. Ordinary tests do not require checkpoints.
To reuse a local GGML checkout, set
`-DFETCHCONTENT_SOURCE_DIR_GGML=/absolute/path/to/pinned/ggml`.
Use the pinned GGML 0.25.3 revision and [required patches](cmake/patches/README.md).
Other GGML revisions are not a compatibility guarantee.

## Integrate into an application

```cmake
add_subdirectory(external/SAM.cpp)
target_link_libraries(my_application PRIVATE sam::sam)
```

The `sam::sam` INTERFACE target propagates C++17, includes, and GGML linkage. A
parent-provided `ggml` target must expose the compatible API and required
precision behavior. Applications handle decoding and provide RGB pixels.

```cpp
#include <sam/sam.hpp>

sam::Result segment_rgb(sam::ImageSession& session,
                        const std::vector<std::uint8_t>& rgb,
                        int width, int height) {
    session.set_image({rgb.data(), rgb.size(), width, height,
                       static_cast<std::size_t>(width) * 3});
    return session.segment_text("truck", 0.5f);
}
```

Load a model once, then construct and reuse an `ImageSession`:

```cpp
auto model = sam::Model::load("models/sam3-f16.gguf",
                             {sam::Backend::Auto, 4});
sam::ImageSession session(model);
```

`set_image` consumes borrowed interleaved RGB8 pixels during the call. Set
`row_stride` explicitly; row padding is allowed. A new prompt reuses visual
features. Repeating the same token sequence reuses the last raw prediction,
while changing the score threshold only reruns postprocessing.

Results own their masks, boxes, and scores. Masks contain row-major bytes `0`
or `1` at source resolution. Boxes use original-image XYXY coordinates and are
not clipped to image bounds. An empty detection vector means no matches.
`query_index` identifies a detector query, not a tracking ID.

Sessions retain their model after the public model handle is released. Separate
sessions share weights and serialize execution through the model mutex. Do not
call one session concurrently. Invalid inputs throw `std::invalid_argument`;
model, backend, and execution failures throw `std::runtime_error`.

## Backend and precision selection

| Image weights | `Auto` behavior |
| --- | --- |
| F32 | CPU |
| F16 | Compatible Metal when available, otherwise CPU |
| Vision, full or custom quantized image weights | CPU |

Select Metal explicitly when needed. An unavailable or incompatible explicit
backend request fails; execution errors do not silently retry on another device.
Inspect `model.backend()` and `model.info()` for the resolved backend, task,
storage profile, and arithmetic profile.

Weight precision describes storage. CPU loading promotes F16 values to F32;
Metal and CUDA retain mixed F16/F32 weights. Quantized CPU execution retains packed
weights and uses temporary F32 matrix weights; Metal and CUDA use native quantized
kernels. See [precision details](MODEL_ZOO.md#precision-contract) before choosing
a model for memory-sensitive applications.

The current SAM 3 text adapter accepts printable ASCII and ASCII whitespace,
with a 32-token limit. Unicode and HTML/entity-encoded text require
`segment_tokens` with the official tokenizer's 32 IDs: start with `49406`, end
with `49407`, and pad with zeros. IDs must be in `[0, 49407]`.

## Command-line image inference

Download and convert a checkpoint using the [model guide](docs/models/sam3-details.md).
The runtime loads local GGUF files and does not download models. Old custom
`.ggml` files require reconversion from the original checkpoint.

```sh
build/metal/examples/sam_image \
  --model models/sam3-f16.gguf --image image.jpg \
  --text truck --backend metal --threads 4 --score-threshold 0.5 \
  --output build/truck-result
```

The CLI writes `results.json` and a PNG mask per detection. Output directories
must be new. `--repeat 5` adds five warmed full-image calls and separately
measures result-cache hits. Cache timing does not represent a new image or
prompt. JSON runtime statistics report execution, transfers, and buffer usage;
backend buffer sizes and process RSS are different measurements.

## Forward video tracking

Use a full video GGUF, one fixed prompt, a fixed resolution, and a known frame
count. Frames arrive once in order, beginning at zero.

```cpp
auto model = sam::Model::load("models/sam3-video-hybrid-v1.gguf",
                             {sam::Backend::Metal, 4});
sam::VideoSession session(model, frame_count, {8});
session.set_text("truck");
for (int frame = 0; frame < frame_count; ++frame) {
    for (auto& result : session.push_frame(frame, decoded_rgb_view(frame))) {
        consume_owned_result(result);
    }
}
```

The caller supplies `decoded_rgb_view` and `consume_owned_result`. Results carry
frame indices, persistent object IDs, source-resolution masks, boxes, and scores.
The current SAM 3 adapter buffers output during its 15-frame hotstart; pushing
the final declared frame drains remaining results. Empty frames are valid.

IDs increase without reuse. `reset(new_frame_count)` clears the prompt, IDs,
temporal state, and counters while retaining weights. Invalid arguments leave
state unchanged; reset after an execution failure. Prompt, resolution, and frame
count stay fixed until reset.

```sh
build/metal/examples/sam_video \
  --model models/sam3-video-hybrid-v1.gguf --frames frames \
  --text truck --backend metal --threads 4 --max-objects 8 \
  --output build/video-result
```

Provide contiguous PNG names `000000.png`, `000001.png`, and so on. The CLI
requires a new output directory and writes per-frame results and masks.
`--dump-tensors` exports sparse diagnostic checkpoints;
`--dump-all-tensors` exports every frame and increases time and disk usage.
Video F32 describes weight storage: the current adapter also uses F16
normalization and BF16 feature/memory boundaries, as described in the model guide.

## Verification and development

[Model verification](docs/validation.md) explains reference generation and
comparison commands. [Architecture](docs/architecture.md) covers model adapters,
backend modules, and extension points; [GGUF](docs/gguf.md) documents the current
model file contracts. Experimental results and implementation history live in
`docs/plans/`.

```sh
.venv-reference/bin/python tools/test_tools.py
python3 tools/check_docs.py
git diff --check
```

Follow [AGENTS.md](AGENTS.md) when contributing. Model-specific shapes,
tokenization, graphs, and temporal rules belong to their adapter; device
initialization, storage policy, and execution accounting belong to backends.
New model and platform combinations need numerical and behavior validation.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency licenses and
model/media terms.

Chinese guides: [模型与精度](MODEL_ZOO_zh.md), [量化模型](docs/quantization_zh.md),
[可视化样例](docs/visual-examples_zh.md), [性能测试](BENCHMARK_zh.md), [模型验证](docs/validation_zh.md).
