# Architecture and model extensions

The library currently implements SAM 3 text-prompted image segmentation.
`sam::Model`, `sam::ImageSession`, experimental `sam::VideoSession`, and the
`sam::sam` CMake target are the application entry points. Full video acceptance
has independent gates. SAM 2/2.1, GroundingSAM, DART and SAM 3.1 remain unimplemented.

## Why `internal`

`internal` names implementation code that is not part of the supported public
API. It replaces the less explicit `detail` directory and namespace. In this
header-only library these implementations must still be available to the
compiler; the name describes compatibility boundaries, not C++ access control.
Applications include `sam/sam.hpp` or its public component headers. Tests and
model implementations may include `sam/internal/...` directly.

## Layers and dependency direction

```text
Application / example / composed pipeline
               |
sam/{types,model,image_session,video_session,sam}.hpp
               |
internal/model_interface.hpp       task-specific contracts
               |
internal/models/sam3/              model implementation
               |
internal/runtime/ggml.hpp          GGML integration entry
               |
internal/runtime/ggml/             shared execution + backend modules

Shared leaves: internal/input_validation.hpp, internal/io/gguf_reader.hpp
```

`types.hpp` contains owned results, borrowed input views, model information,
statistics, and diagnostic tensor snapshots. It depends only on the standard
library. The public model loader selects the implemented model adapter; the
public session delegates execution and owns no model-specific constants.

The text-image session contract describes the existing task, rather than every
possible SAM prompt. Unsupported tasks produce a clear error. A future SAM 2
adapter can supply a point/box session without pretending to have a tokenizer
or native text encoder. Add new task contracts with their implementation and
tests, not empty methods on every model.

The SAM 3 adapter owns model-specific GGUF validation, tokenizer semantics, model
state, image/prompt caches, transforms, postprocessing, and diagnostic names.
Its private namespace is `sam::internal::sam3`. Moving a header into a model
directory is not sufficient: its symbols and callers must also use that model's
namespace so another architecture can use names such as `ModelState` safely.

The GGML layer owns contexts, buffers, backend selection, scheduling, transfers,
and precision enforcement. It does not include model headers or initialize
model-specific tensors. CPU and Metal share model graphs.

GGUF container parsing lives in `internal/io/gguf_reader.hpp` and reuses the
pinned GGML APIs with bounded metadata reads and file-range checks. SAM-specific
metadata, tensor inventory, tokenizer and precision validation stay in the
model adapter. [Schema 1](gguf.md) identifies `sam3` / `text_image`; another
family can reuse the reader while defining its own metadata and tensors.
This shared reader adds bounded metadata validation during model loading and
replaces the former custom container parser. It depends on the pinned GGML
callback and metadata serialization APIs, checked by the CMake compatibility probe.

## Backend modules

| Header under `internal/runtime/ggml/` | Responsibility |
| --- | --- |
| `resources.hpp` | RAII ownership of contexts, buffers, backends, schedulers; metadata context allocation |
| `backend.hpp` | Small driver/device contract for backend identity, storage policy and node statistics |
| `backends/cpu.hpp` | CPU discovery, initialization, thread configuration and exact FP16 promotion policy |
| `backends/metal.hpp` | Metal discovery, Metal initialization and the FP32 arithmetic probe |
| `runtime.hpp` | Supported-driver selection, ordered execution backends and ownership |
| `graph.hpp` | Shared scheduling, operation checks, transfers and execution statistics |

Backend drivers use GGML's existing device registry. They contain the hardware
and precision decisions; model loaders consume the selected storage policy.
The SAM 3 loader therefore does not branch on CPU or Metal. Shared graph code
requests precise arithmetic and asks the runtime to attribute work to the actual
scheduled backend. Unknown assignments are errors, not CPU work by default.
`Model::backend()` identifies the primary weight backend. GGML may place an
individual graph on CPU according to its input/weight buffers; the statistics
follow that actual placement. Backend tests cover both host-input CPU execution
and Metal execution with a Metal weight buffer.

SAM's local GGML patch implements native Metal `WIN_PART` and `WIN_UNPART` for
contiguous F32 tensors, including padded windows. The tested full image graph
executes all 3,332 FP32 or 3,342 FP16 nodes on Metal with six graph partitions
and no CPU graph fallback. Host preprocessing/postprocessing still run on CPU. See the
[window and performance plan](plans/20261001-002850-metal-window-cpu-performance-official-weights.md)
for direct GPU checks and full-model acceptance.

CPU remains the required fallback backend. `Auto` selects compatible Metal for
FP16 checkpoints when available and CPU for FP32 checkpoints. Explicit missing
backends and incompatible Metal arithmetic fail clearly. Explicit FP32 Metal
selection retains F32 weight storage and passed all seven original-weight
image cases plus the real-checkpoint session lifetime/cache checks locally on
2026-10-02. See the [acceptance record](plans/20261001-121633-fp32-cpu-metal-validation.md).
Auto FP32 continues selecting CPU.

Adding CUDA requires a real driver under `backends/`, supported-driver selection,
public configuration/CLI naming, device selection, and its statistics field.
Build the pinned GGML CUDA backend in a CUDA environment, then verify operator
coverage, storage/compute precision, transfers, fallback behavior and numerical
results on NVIDIA hardware. The pinned [GGML backend registry](https://github.com/ggml-org/ggml/blob/353b63b439f27ab2cc19dac97ab1681ba6d2d084/src/ggml-backend-reg.cpp)
already provides the underlying registration mechanism. Current model graphs
can be reused; add shared operator fixes only when device validation requires
them. A new backend has no supported status until these checks pass.

CUDA is not implemented or tested by this refactor. There is no empty CUDA
module or additional SDK dependency. A non-GGML engine would need its own graph
adapter and acceptance work. This separation keeps backend-specific code out
of model implementations without introducing a virtual tensor API.

## SAM 3 implementation

| Responsibility | Headers under `internal/models/sam3/` |
| --- | --- |
| Parameters, weight structures, tensor bindings | `architecture.hpp`, `tensors.hpp` |
| Shared graph math | `ops.hpp` |
| Visual backbone and feature pyramid | `vision.hpp` |
| Text and geometry prompts | `text_encoder.hpp`, `prompt_encoder.hpp` |
| Image/prompt fusion | `fusion_encoder.hpp` |
| Boxes, scores, object queries | `detector.hpp` |
| Pixel features and masks | `mask_decoder.hpp` |
| Stage execution and intermediate data | `execution.hpp` |
| GGUF model schema, tokenizer, image transforms | `weights.hpp`, `tokenizer.hpp`, `image_ops.hpp` |
| Weight/resource ownership | `state.hpp` |
| Loading, task dispatch and session cache | `model.hpp`, `image_session.hpp` |

The execution layer separates prompt preparation, fusion, detection, and mask
decoding. The existing segmentation path still executes them in that order.
The detector exposes the query features needed by the mask stage, allowing a
future detection pipeline to stop before masks without duplicating the graph.
These are private implementation seams, not a public DART API.

Model state outlives a public model handle while sessions retain it. Each
session owns its caches; separate sessions share weights and one execution
mutex. A session must not be called concurrently. Independent execution
contexts would be the next change if measured concurrency requirements justify
their memory cost. The adapter adds one small allocation per loaded model and
one per session, with virtual calls outside tensor loops. There is no dynamic
registration framework. Dependencies are one-way: `model.hpp` includes the
session implementation, which includes `state.hpp`; state does not include
the factory or session.

## Extending the library

Write a timestamped plan before adding an architecture or task. Pin its original
source and checkpoint format, define input/output coordinates and ownership,
then add the implementation and reference comparisons together.

| Extension | Where it belongs | Acceptance to add |
| --- | --- | --- |
| [SAM 2 / 2.1](https://github.com/facebookresearch/sam2) | A separate model family with its own prompt encoder, image decoder, configuration and memory path; point/box and video sessions | Official image/mask comparisons; video entering objects, occlusion and ID continuity |
| [GroundingSAM / Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) | A pipeline composing a grounding detector and a promptable segmenter, using shared image-space boxes | Detector-to-segmenter coordinate conversion, labels, empty detections and tracking behavior |
| [DART](https://github.com/mkturkcan/DART) | Reuse SAM 3 vision, text, fusion and detection stages; add class-embedding cache and multiclass execution | Batched versus individual class output, class mapping and detection-only execution |

The sources above motivate these boundaries; the proposed C++ organization is
this project's design. Do not make SAM 2 inherit SAM 3's 32-token contract,
normalization, query count, or container schema. Share a graph helper only after
its dimensions, precision, and semantics match across implementations.

## Integration and verification rules

All project-owned headers use unique `SAM_CPP_..._HPP` include guards, remain
self-contained, and define non-template free functions as `inline`. Preserve
vendored licenses and existing vendor guards. GGML remains a compiled
dependency; image/video decoding and checkpoint downloading stay outside the
core library.

Tests compile every header independently and repeatedly, link two translation
units, and exercise meaningful contracts and precision boundaries. Structural
changes also run the existing frozen numerical corpus. Same-weight community
checkpoint comparisons retain supplementary provenance; original-checkpoint
acceptance requires the authorized original file and its recorded hash.

## Forward video sessions

`tracking/` holds separate tracker weights, prompt/mask decoding, memory encoder,
256-wide memory attention, video preprocessing and forward memory selectors.
`ModelDefinition::encode_image(..., video=true)` builds the detector and tracker
necks from one shared ViT trunk, then rounds tracker feature transport through
BF16. The ordinary image call keeps its existing preprocessing and detector path.

Memory attention tiles only queries, 128 at a time, while each softmax includes
all spatial/pointer keys. Forward retention keeps complete hotstart group history;
after group membership is fixed it protects the eight-frame correction window,
four conditioning records and 15 older eligible records. The selector retains
the count of pruned conditioning records to preserve official ordering. It is
not a reverse/editable-session policy. `tracking/session.hpp` integrates logical
birth groups, quality recomputation after hotstart removal, association, periodic
reconditioning, overlap suppression and a 15-frame delayed queue. Tensor work
runs serially per object; group membership and the original batch-quality
broadcasting semantics stay intact. `tracking/execution.hpp` connects conditioned
memory, temporal pointers, SAM decoding and BF16 memory encoding to shared graphs.

SAM decoder head-16 cross attention uses F32 GGML matmul/softmax because the
pinned Metal flash implementation does not support that head size. Head-32
self-attention retains the verified precise flash path. The graph is shared by
CPU and Metal; no new device policy or dependency patch is introduced.

The public facade delegates through the text-video task contract. It retains the
model, owns prompt/temporal state and returned results, enforces fixed finite
forward input, and requires reset after execution failure. `sam_video` decodes
host PNGs, publishes exclusive output and writes an incomplete/complete receipt.
The source adapter, official exporter and differential validator record explicit
F16/BF16 boundaries and retain Meta's association/lifecycle modules. The original
CPU component fallback needs a recorded empty-batch adaptation when no object is born.
Snapshots preserve propagated conditioned features and selected masks before
periodic correction, plus the normalized mask passed to memory encoding. Thus
correction cannot overwrite the observations needed to diagnose recurrence.

Weight-free tests compare resized bytes and F16 normalization against Pillow
11.2.1, attention against an independent full-key computation, and 2,024 memory
selections against hash-verified pinned Meta functions. Official checkpoint
stage, temporal, session and backend-placement evidence is recorded separately in
the [video validation record](plans/20261002-041320-video-session-official-validation.md).
The full corpus/matrix and performance gates determine accepted video support.
