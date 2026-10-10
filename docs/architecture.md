# Architecture and model extensions

SAM.cpp is designed as a C++17 library for multiple segmentation and detection
models and execution backends. The current model adapter implements SAM 3 image
and forward-video tasks; the GGML runtime provides CPU, Metal and CUDA backends.
New model families and
platforms need their own implementation and numerical validation before they
are advertised as supported.

The public CMake target is `sam::sam`. Applications use `sam::Model` and the
task-specific session interfaces in `sam/sam.hpp`. Public input, result and
model-information types describe model behavior without exposing a device API.
See the [Model Zoo](../MODEL_ZOO.md) for the current task scope and the
[quantization guide](quantization.md) for storage profiles.

## Layer boundaries

```text
Application / example / composed pipeline
               |
sam/{types,model,image_session,video_session,sam}.hpp
               |
src/contracts/                     model and task contracts
               |
src/models/<family>/               model-specific schema, graphs, sessions
               |
src/runtime/<engine>/              execution integration
               |
runtime backends                   devices, buffers, scheduling, statistics

Shared leaves: src/common/input_validation.hpp, src/io/gguf_reader.hpp
```

`types.hpp` contains owned results, borrowed input views, model information,
statistics and optional diagnostic tensor snapshots. It depends only on the
standard library. The model loader selects an implemented adapter from the
file's architecture and task metadata; unsupported combinations fail clearly.
Each session delegates to that adapter and owns task state without putting
model-specific constants into the public facade.

## Repository layout

```text
include/sam/          installed public API (declarations and value types)
src/api/              compiled public wrappers
src/model_factory.cpp explicit adapter selection
src/contracts/        private task contracts
src/common/           backend-independent input validation
src/io/               bounded GGUF container reading
src/models/sam3/      SAM 3 schema, graphs, image and video sessions
src/runtime/ggml/     execution resources, graphs, workspace, backends
apps/{image,video}/   sam_image and sam_video (SAM_BUILD_EXAMPLES)
support/image_io/     decoding support for apps, probes and tests (not installed)
third_party/stb/      vendored decoding headers with their original licenses
tools/convert/        checkpoint conversion and model schema
tools/quantize/       calibration, runtime quantization and cache tools
tools/validation/     reference export, validation and acceptance tools
tools/benchmark/      timing, linear probes and graph profiling
tools/visualization/  comparison image rendering
tools/maintenance/    documentation, archive and campaign upkeep
tests/{api,models,runtime/ggml,integration,tools,data}/
cmake/                pinned GGML preparation, patches and install/export
```

Standalone builds default `SAM_BUILD_TESTS`, `SAM_BUILD_EXAMPLES` and
`SAM_BUILD_TOOLS` on and `SAM_BUILD_CUDA_PROBES` off; embedded builds default
them off. The CLI applications keep their historical `${build}/examples/`
executable paths, and the tools do too, so existing scripts keep working. The
flat `tools/<name>.py` modules from earlier revisions remain as forwarding
entry points: scripts and imports keep resolving, while the implementation
lives in exactly one grouped module. Python test suites live in `tests/tools/`
and are launched through `tools/test_tools.py`.

`SAM_ENABLE_INSTALL` (standalone default) adds standard CMake install/export
rules for the public headers, the compiled library and a relocatable
`find_package(sam CONFIG)` package. SAM-prepared GGML is installed with the
package and exported as plain imported targets; a caller-owned GGML must be
resolvable through its own CMake package or install configuration fails with an
explicit error. Private implementation, tests, tools and decoding support are
never installed.

The `internal` namespace marks implementation code that is not a compatibility
promise or public extension API. Project-owned headers use unique path-derived
include guards, remain self-contained, and define non-template free functions
as `inline`. GGML remains a compiled dependency. Image/video decoding and
checkpoint downloading stay in host applications or examples.

## Model and task adapters

Each model family owns its tensor schema, tokenizer, preprocessing, graph,
postprocessing and model/session state. Task contracts stay specific: an image
text-prompt contract does not promise point/box prompts, temporal memory or
object discovery. A future adapter may expose those through a different session
interface instead of adding empty methods to every model.

The SAM 3 adapter lives under `src/models/sam3/` and uses the private
namespace `sam::internal::sam3`. It owns SAM 3 GGUF validation, tokenizer
semantics, image transforms, prompt and image caches, postprocessing and
diagnostic tensor names. Keeping symbols within the model namespace lets other
families use names such as `ModelState` without collisions.

| SAM 3 responsibility | Implementation area |
| --- | --- |
| Parameters, weight structures and tensor bindings | `architecture.hpp`, `tensors.hpp`, `weights.hpp` |
| Shared graph operations | `ops.hpp` |
| Visual backbone and feature pyramid | `vision.hpp` |
| Text and geometry prompts | `text_encoder.hpp`, `prompt_encoder.hpp` |
| Image/prompt fusion and object queries | `fusion_encoder.hpp`, `detector.hpp` |
| Pixel features, masks and stage execution | `mask_decoder.hpp`, `execution.hpp` |
| Tokenizer, image transforms and model resources | `tokenizer.hpp`, `image_ops.hpp`, `state.hpp` |
| Loading, task dispatch and image-session cache | `model.hpp`, `image_session.hpp` |
| Temporal memory and video session | `src/models/sam3/video/` |

The image execution path separates prompt preparation, fusion, detection and
mask decoding. The detector exposes query features so a future detection-only
pipeline can stop before mask generation without duplicating the graph; this
seam does not itself implement DART.

## Runtime and backends

The runtime owns engine integration, contexts, buffers, device selection,
scheduling, transfers and execution precision. Model adapters request the
storage and arithmetic policies required by their graph; they do not initialize
hardware or branch on a platform. Runtime statistics describe where work was
actually scheduled. `Model::backend()` identifies the primary weight backend,
which may differ from a graph's assigned compute device.

The current GGML modules are split by responsibility:

| Header under `src/runtime/ggml/` | Responsibility |
| --- | --- |
| `resources.hpp` | RAII ownership of contexts, buffers, backends and schedulers |
| `backend.hpp` | Driver/device contract, identity, storage policy and node statistics |
| `backends/cpu.hpp` | CPU and optional BLAS discovery, initialization and threading |
| `backends/metal.hpp` | Metal discovery and initialization |
| `backends/cuda.hpp` | Visible CUDA device selection, initialization and required F32 arithmetic checks |
| `runtime.hpp` | Driver selection, execution order and resource ownership |
| `graph.hpp` | Shared scheduling, operation checks, transfers and execution statistics |
| `workspace.hpp` | Reusable graph allocations, safe rebinding and internal diagnostics |

Backend drivers use GGML's device registry. CPU, Metal and CUDA use the shared SAM 3
graphs; backend-specific precision and fallback rules remain in the runtime.
On CPU, eligible matrix operations may use a registered BLAS device before the
native CPU backend. Default quantized CPU execution keeps packed weights and
prepares `MUL_MAT` operands with shared F32 cast nodes. Opt-in native CPU compute
skips those casts and pins packed-weight matmuls to CPU; the backend owns RHS
Q8 packing/precision validation while dense operations retain existing placement.
Public `BackendOptions::cpu_compute`, unified configuration and arithmetic profiles
expose the choice without model-specific graph branches. Quantized Metal and CUDA use native packed-weight
kernels with separate staging and arithmetic profiles; see the
[quantization guide](quantization.md) for the profile contract.

The common GGUF reader in `src/io/gguf_reader.hpp` performs bounded
container metadata reads and file-range checks before backend weight allocation.
Model-specific metadata, tensor inventory and precision checks belong to the
selected adapter. Another family can reuse the reader while defining its own
metadata and tensor schema. The reader uses the pinned GGML GGUF APIs and
canonical metadata serialization rather than adding a second general-purpose
container parser.

The CUDA module selects an index among visible devices and checks the linked
GGML's required F32 arithmetic at initialization. Its shared scheduler pins
every compute node to that device and rejects unsupported operations or CPU
compute fallback before execution. Inputs may still use CPU buffers for copies.
Backend node counters exclude metadata-only view/reshape/permutation/transpose operations.
Quantized CUDA uses native packed-weight kernels and reports a distinct arithmetic
profile; model qualification is separate from CPU and Metal.

The driver also owns the opt-in F16 compute policy and memory-attention execution
policy. A node-configuration hook selects reduced dense operands and supported
fused attention without model-side device branches. Shared tracking graphs consume
the driver's query tile, output assembly and fused-memory-attention policy.
CUDA F16 uses fused single-head 256-dimensional memory attention with the object
axis preserved as its batch axis. The default CUDA mode uses bounded query tiles.
CPU/Metal retain their existing policies. Arithmetic profiles distinguish compute mode from storage and keep
qualification receipts from mixing different numerical contracts.

The driver separately selects graph-stage grouping. Quantized CUDA image models
in default compute mode join the existing fusion, detector and mask builders in
one prediction graph. Fusion and query features flow directly between these
stages; required result and diagnostic tensors are still downloaded into owned
host values. Dense models and F16 compute retain staged prediction. This policy
adds no cross-session mutable cache and keeps weight storage independent of
compute precision. The [dataflow plan](plans/20261007-094949-cuda-precision-and-dataflow-optimization.md)
records the measured selection and its workspace cost.

Adding another platform requires a real driver, supported-driver selection,
public configuration, device-aware statistics and matching hardware validation.
The pinned [GGML backend registry](https://github.com/ggml-org/ggml/blob/d7cb574130e6f01ad25b3289685489200febcd74/src/ggml-backend-reg.cpp)
provides the registration mechanism; a new backend still needs operator,
precision, transfer and fallback checks. A non-GGML engine would need its own
execution adapter.

## Session ownership

Model weights are immutable after loading. A session retains the model state it
uses and owns its caches; sessions from one model share weights and execution
is serialized. A session must not be called concurrently. Independent execution
contexts are an extension if an application needs concurrent sessions and can
accept their additional memory cost. Keep input validation and session state at
the public task boundary, not in backend drivers.

The SAM 3 video adapter shares the visual trunk with tracking stages and keeps
temporal memory, object pointers, association and lifecycle policy in its own
model family. Video preprocessing and feature/memory rounding are model
contracts, not backend storage policy. Other video-capable models may need
different temporal state and should not inherit SAM 3's implementation by
default. The [video session plan](plans/20261001-115321-sam3-text-video-tracking.md)
records the SAM 3 state and task boundaries.

SAM 3 tracking retains a bounded set of graph metadata and uses one shared
workspace. Current-frame features have separate ownership so several objects
can read them during propagation. Compatible objects use a batch dimension;
workspace limits select smaller batches or serial conditioning and decoding.
The backend arena and resident frame inputs are released at each frame boundary;
cached graph metadata survives. Results remain owned host values. Model adapters
define shape keys and temporal rules, while the common runtime manages allocation
and synchronization. New adapters can reuse that runtime without inheriting
SAM 3's shapes or state.

## Extension contracts

Write a plan before adding a model family, task or backend. Pin the original
source and checkpoint/container format, define coordinate systems and result
ownership, then add the implementation and reference comparisons together.

| Extension | Boundary | Contract to define |
| --- | --- | --- |
| SAM 2 / 2.1 | Separate model family with its own configuration, prompt encoder, image decoder and memory path | Point/box image tasks; video entering objects, occlusion, disappearance/reappearance and ID continuity |
| SAM 3.1 Object Multiplex | New adapter, checkpoint schema, neck/memory/decoder and bucket assignment | Object counts at bucket boundaries (1, 16, 17), removal/reassignment, accuracy, latency and memory |
| GroundingSAM / Grounded SAM 2 | Pipeline composing a grounding detector and a promptable segmenter | Shared image-space boxes, coordinate conversion, labels, empty detections and tracking behavior |
| DART | Detection-oriented SAM 3 composition reusing compatible vision/text/fusion/detection stages | Class embedding/cache, multiclass execution, class mapping and batched versus individual output |
| Another compute backend | Runtime backend module, independent of model family | Device selection, operator coverage, storage/arithmetic precision, transfers and fallback behavior on matching hardware |

The linked projects motivate these boundaries: [SAM 2](https://github.com/facebookresearch/sam2),
[Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2), and
[DART](https://github.com/mkturkcan/DART). SAM 2 must not inherit SAM 3's
32-token contract, normalization, query count or container schema. SAM 3.1
requires its actual checkpoint and model changes; renaming SAM 3 output is not
SAM 3.1 support. Share graph helpers only when dimensions, precision and
semantics match across adapters.

## Contributor integration rules

Preserve vendored licenses and vendor include guards. GGML stays compiled;
model weights stay external. Tests should exercise public contracts, numerical
behavior, boundaries and regressions while keeping header self-containment and
two-translation-unit linkage intact. Supplementary community checkpoints retain
their provenance separately from original-checkpoint comparisons.
