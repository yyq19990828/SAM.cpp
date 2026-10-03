# M2: SAM 3 Text-Driven Video Tracking

Created: 2026-10-01 11:53:21 Asia/Shanghai.
Status: complete on 2026-10-03 under the user-approved F32 + hybrid CPU/Metal support contract. Explicit F16 remains diagnostic; see the final acceptance record below.
Baseline: `2c23a67b58019149efe41c89aa263c1b1e770c65`.

Implementation prerequisite: complete the
[FP32 CPU/Metal image-validation plan](20261001-121633-fp32-cpu-metal-validation.md)
and its benchmark matrix first. Image acceptance does not establish video
acceptance; the video graphs retain the independent gates below.

## Outcome and scope

Deliver a header-only `VideoSession` that consumes a finite sequence of
host-supplied RGB frames and one text concept, discovers matching objects on
later frames, and returns masks, boxes, scores and persistent object IDs.
The original precision target was FP32/CPU and FP16/CPU or Metal. Final
acceptance supports F32 and hybrid on both CPU/Metal using the same model
graphs, following the user-approved precision change documented below.
Preserve the current image API, image GGUF files and all M1 numerical gates.

M2 follows the [original roadmap](20260930-192519-sam3-text-image-baseline.md#10-follow-on-milestones).
Its completion requires the complete tracking pipeline. Converter, graph and
state work below are implementation packages within this milestone, not separate
claims that video support is ready. M1 remains usable throughout.

The first video contract is forward-only, with a known frame count and a fixed
prompt and resolution per sequence. Exclude retrospective edits, reverse passes,
unknown-length live streams, user point/box correction, multiple concepts per
session, codecs in the core, GUI, quantization, new device backends and new model
families. SAM 3.1 Object Multiplex remains M3; this work does not enable it.

Planning estimate: 3-5 engineer-weeks including reference adaptation and numerical
acceptance. Tracker math and reference alignment dominate the work; this is
substantially larger than the container migration. No real-time target is assumed.

## Verified starting point and sources

- M1 has original-weight acceptance for 21 cases, CPU/Metal CTest 9/9, and a
  working downstream consumer. The commit-readiness corrections additionally
  pass 10 Python checks; both existing GGUF files pass current provenance checks.
- The original `sam3.pt` is already local, SHA-256
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
  Current conversion writes 1,133 tensors and excludes 309 tracker tensors plus
  22 tracker-neck tensors. These 331 shapes/names are already enumerated in
  `tools/sam3_tensor_schema.json`.
- Both inspected upstream checkouts are clean at the pinned revisions below.
  Official raw-source URLs and the planned Python dependency were reachable
  during planning. No video inference, download or package installation was
  performed in this planning step.

| Source | Use and limitation |
| --- | --- |
| [Meta builder](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model_builder.py) | Architecture, checkpoint and default temporal policy; use `build_sam3_video_model`, single rank, compilation disabled |
| [Meta video inference](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_video_inference.py) and [association/lifecycle](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_video_base.py) | Authoritative output buffering, association, reconditioning and removal semantics |
| [Meta tracker](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_tracker_base.py) and [predictor state](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_tracking_predictor.py) | Memory selection, object pointers, explicit BF16 state storage and conditioning-frame invalidation |
| [Meta frame input](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/io_utils.py) and [feature transport](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_image.py) | Define the selected PIL-frame preprocessing route and tracker-neck BF16 boundary |
| [PABannier C++ implementation](https://github.com/PABannier/sam3.cpp/blob/416186c501d060df7ca02989d49b38080f5f81f3/sam3.cpp) | Reuse verified tracker graph formulas and tensor mapping with retained MIT attribution; its greedy matching, memory selection and hotstart logic are not the oracle |

The simplest useful route is to share the current visual/text detector, add the
tracker graphs, and port Meta's control rules. A frame-by-frame detector alone
cannot establish persistent IDs or survive missing detections. Adopting the
community tracker wholesale would change the inspected association and memory
rules, so it is excluded from the acceptance route.

## Public behavior

| Surface | Contract |
| --- | --- |
| `VideoSession(model, frame_count, options)` | Requires video-capable weights and frame count in `[1, INT32_MAX]`; retains the model. `VideoOptions.max_objects` defaults to 8 and must be positive |
| `set_text` / `set_tokens` | Set one nonempty concept before the first frame; retain the existing ASCII/32-token contract and token-ID alternative. Encode text once per sequence |
| `push_frame(frame_index, ImageView)` | Accept exactly the next zero-based index; consume borrowed RGB pixels during the call. Resolution must match the first frame. Return zero or more owned `VideoFrameResult` values in source-frame order |
| Last declared frame | Drain all remaining delayed results and mark the sequence complete. A subsequent push fails; no redundant flush API is needed when the count is known |
| `reset(frame_count)` | Clear prompt, IDs, temporal state, pending results and counters; retain loaded weights. A new prompt and frame sequence can then start |
| `VideoFrameResult` | Original frame index and owned tracked objects with session-local ID, source-size binary mask, XYXY box and the official video-output score semantics |
| Statistics | Accepted/emitted frames, detector/tracker calls, active objects, rejected new-object admissions, retained memory/pointer records, pending outputs, transfers, backend nodes, latency and buffer high-water marks |

An empty returned batch means no frame is ready yet. A returned frame with zero
objects is a completed empty result. IDs are distinct from image query indices,
start at zero, increase within a sequence and are never recycled before reset. Short occlusions
preserve an ID only as the official lifecycle does; an object removed by that
lifecycle can receive a new ID on rediscovery. Keep-alive counters are not a
promise to delete every empty track after a fixed number of frames.
Convert official normalized XYWH boxes using the original width/height and
`x1 = x0 + width`, `y1 = y0 + height`, without adding an extra pixel. These video
boxes follow the final-mask convention; M1's regressed image boxes are unchanged.

Bad arguments fail before changing state. A backend/execution failure marks that
video session unusable until reset, rather than exposing a partially advanced
frame. Separate sessions share immutable model weights and the existing execution
mutex; one session remains non-concurrent. Model information gains a task/profile
field so callers can distinguish image and video weight files.

## Architecture and execution

```text
Host RGB frame -> video preprocessing -> shared ViT trunk
                                         |          |
                                  detector neck   tracker neck
Fixed text -> cached text features ------+          |
                              raw detector outputs  |
                                         |   memory attention + SAM mask decoder
                                         +----------+
                                            |
                               official association / lifecycle
                                            |
                              memory encoder + retained state
                                            |
                              delayed output queue -> owned results
```

Add the separate text-video task contract and public facade, following the
existing image pattern. Keep model-specific implementations under
`internal/models/sam3/tracking/`: preprocessing, prompt/mask decoder, memory
encoder/attention, state and temporal policy. Share actual matching graph math
and the ViT trunk, not global SAM-shaped configuration. The new task interface
allows a future model adapter to implement this task without inheriting SAM 3's
constants; its maintenance cost is one small facade/virtual contract outside
tensor loops. Do not create empty SAM 2, CUDA or generic plugin implementations.

Run the trunk once per frame and derive both necks from that result. Consume raw
detector logits/masks from the internal execution layer; the image API's final
postprocessing is not the video detection policy. Process tracker tensor work
serially per object to bound workspace, while retaining Meta's logical tracker
groups, group-averaged memory scores and group-level reconditioning behavior.
Changing tensor scheduling must not change logical batch membership.

For the new 256-dimensional attention head, use shared FP32 GGML matmul/softmax
graphs with query tiles of 128 and the complete key sequence in each softmax.
This bounds the score workspace without changing normalization. The existing
Metal precise-flash patch covers heads 32/64; do not assume it covers 256.
Use existing precision enforcement on both backends. Native 256-head flash
optimization can follow separately after this path establishes the oracle.

Retain include guards, inline definitions, independent/repeated header builds,
two-TU linkage and device policy inside the existing runtime/backend modules.

## Video GGUF and numerical contract

- Keep GGUF v3 and the bounded common reader. Existing image files remain SAM
  schema 1 / `text_image`. New full files use SAM schema 2 / `text_video` with
  architecture `sam3`; they also provide the image task. Old runtimes reject
  schema 2 clearly, while the new runtime continues accepting schema 1.
- Add `--task image|video` to the converter, defaulting to `image`. A full video
  file contains the 1,133 image entries plus the 331 currently excluded entries:
  exactly 1,464 tensors. Reuse the existing mapping and schema inventory; keep
  the unused pooled-text/training entry explicitly excluded and recorded.
- Require the existing identity, tokenizer and image metadata plus named tracker
  metadata: embedding width 256, memory width 64, four attention blocks, one
  256-wide head, seven memory positions, four conditioning frames, up to 16
  historical pointer candidates, and mask-memory resize 1152. Record policy ID
  `meta-sam3-temporal-v1` and explicit BF16 feature/state storage semantics.
- Preserve all image-subset tensor bytes and the current per-tensor F32/F16 policy.
  Reject missing/extra tensors, unsupported profiles and inconsistent canonical
  shapes/dtypes before backend allocation. Publish new filenames and provenance
  sidecars exclusively; never overwrite the accepted image artifacts.
- Match Meta's `list[PIL.Image]` video input route: RGB Pillow bicubic resizing
to 1008, followed by its FP16 storage/normalization rounding, then exact FP32
  promotion for computation. This differs from M1 image preprocessing and needs
  exact resized-byte and normalized-value goldens against pinned Pillow 11.2.1,
  including all 256 input byte values and stride/aspect-ratio boundaries.
- Preserve explicit BF16 rounding at tracker-neck transport and temporal-memory
  storage; promote those stored values exactly before FP32 consumers. GGML
  already provides BF16 conversion helpers. Weight precision, arithmetic
  precision and temporal storage are recorded separately.

The reference uses FP32 computation with compilation, automatic mixed precision
and TF32 disabled, retaining these explicit input/BF16 boundaries. Name it
`official-video-fp32-explicit-storage`; it is a recorded adaptation of Meta's
graph, not evidence of bitwise equality with stock GPU autocast execution.

## Temporal policy and bounded retention

Freeze the inspected temporal-disambiguation profile:

| Rule | Value |
| --- | ---: |
| Detection / new-object score thresholds | 0.5 / 0.7 |
| Detection NMS mask IoU | 0.1 |
| Detection-to-track / unmatched-track IoU thresholds | 0.1 / 0.5 |
| Output delay / hotstart unmatched / duplicate counters | 15 / 8 / 8 |
| Keep-alive initial / maximum / minimum | 30 / 30 / -1 |
| Recent-occlusion overlap threshold | 0.7 |
| Reconditioning interval | 16 frames |
| Hole/sprinkle area | 16, with the official foreground-area condition |
| Quality threshold for memory selection | 0.01 |
| Memory stride / preserve-first-conditioning-frame | 1 / false |
| Masklet confirmation | Disabled |

Port the actual ordered rules, not just their constants. Preserve mask-based
matching, rejection/suppression order, 8-connected component behavior and odd-size
padding. At the object cap, use the reference's score-based new-object admission
with the same cap and report omitted births; do not silently evict live tracks.

The output queue has at most 15 low-resolution frame results while processing,
and at most 14 after an ordinary emission. Upscale only emitted masks. Keep one
frame's visual features and shared positional encodings; never retain all RGB
frames or all historical output masks inside the library. The declared frame
count controls validation/temporal encoding, not a preallocated array of states.

Retain complete group history during its first 15 frames: removal can recompute
historical group-quality scores and downgrade conditioning frames. The inspected
default video policy groups objects by their common birth frame and only removes
them during hotstart; M2 exposes no later manual edits/removals. Once that window
closes, group membership is fixed. For the resulting forward-only state, retain
the four most recent conditioning records, the
last eight non-conditioning frame positions needed by the seven-frame invalidation
window, and the latest 15 eligible older non-conditioning records. Deduplicate
these sets: at most 27 retained records per object, plus the current in-flight
record. The initial complete hotstart history also fits this bound. Preserve
group membership and stored group-quality values. Object
pointers reuse these records. Prune retired-object metadata once no live state
or pending output can reference it; retain only the monotonic next-ID counter.
Store capped hotstart counters and only the score/suppression metadata still
needed by live tracks or delayed output; do not copy Python's entire debug and
interactive-output history into the runtime.

This bound is a design deduction from the pinned forward selectors and current-
frame correction window. Before relying on eviction, compare its selected frame
IDs and pointer order against an unpruned official selector across long synthetic
score histories, reconditioning and permitted hotstart group removals. Any mismatch blocks this
eviction policy; do not replace it with an approximate last-seven-frame cache.

## Oracle, fixtures and acceptance

Extend the isolated source-preparation workflow with `--task image|video`
(default `image`) and hashed
adaptations. Call the model builder directly on CPU with rank 0 / world size 1;
bypass the GPU-only predictor wrapper and unused multiplex imports. Replace
device-only `.cuda()` transfers, close/disable the predictor's persistent CUDA
autocast context, and reuse the recorded unfused FP32 MLP. Preserve explicit
BF16/F16 rounding followed by FP32 promotion where required for CPU computation.
Do not change association, thresholds or object lifecycle to make the port pass.

Use the official CPU connected-component fallback, adding
[`scikit-image==0.25.2`](https://pypi.org/project/scikit-image/0.25.2/) and its
resolved dependencies only to the isolated reference lock. Its CPython 3.12
macOS arm64 wheel is available; the package is currently absent. Preserve the
existing Torch/NumPy pins. Existing Pillow handles frame lists, so no video codec,
cloud service or new credential is required. Original weight authorization is
already available; runtime inference stays offline.

Store deterministic recipes in `tests/data/sam3-video-cases.json`; generated PNGs,
reference dumps and private footage remain under ignored `models/`. Start with
these defined scenarios using the pinned truck/groceries images from M1:

| Case | Frames and recipe | Required reference behavior |
| --- | --- | --- |
| Motion | 48 truck frames; horizontal offset `4 * min(f, 48-f) - 48` pixels for zero-based frame `f`, neutral padding | A persistent object with stable ID through visible motion |
| Entry | 64 frames on a 1800 x 1200 canvas; resize the truck image to 900 x 600 using pinned Pillow bicubic, paste one copy at (0,300), and a second at `(1800-min(max(f-15,0)*60,900),300)` | A new ID born after the first frame while the original ID continues |
| Occlusion | 64 truck frames; fill rectangle `[64,256,1744,920)` for frames 20-23, then restore it | ID continuity across a short occlusion when retained by the oracle |
| Hotstart removal | 24 frames; truck visible only in frames 0-1, then neutral frames | A removed hotstart object is suppressed from delayed earlier outputs |
| Negative | Repeat the groceries image for 16 frames with the `purple elephant` prompt | Valid empty frame results and complete final draining |

Use RGB (127,127,127) for neutral fills, the `truck` prompt on truck cases, and
the frozen M1 source-image hashes. Record the recipes and generated PNG hashes
in the corpus manifest. Verify that the official outputs exercise each required behavior, then
freeze the manifest before inspecting C++ results. A missing scenario is a
fixture failure, not a passing test; changes require a recorded reason.

Acceptance includes:

- Exact frame order, output-delay/drain behavior, token IDs, source shapes,
  selected memory indices, pointer order and lifecycle events. Use one ID mapping
  fixed at each track's first appearance for the whole sequence; per-frame
  rematching must not hide ID switches. Preserve discrete policy decisions on
  the designated stable cases; report threshold-adjacent cases explicitly.
- Compare detector/tracker necks, memory encoder output, conditioned features,
  decoder logits, scores and object pointers at frame 0, first propagated frame,
  frame 16 and the last frame; compare final masks/IDs on every frame. Retain M1's
  tensor limits (normalized L2 0.001 for FP32, 0.02 for FP16), score error 0.02,
  box error 1% of image dimensions, and mask IoU 0.98 / 0.95 for reference scores
  at least 0.6.
  Keep zero-norm handling and fixed thresholds. Do not relax gates after seeing
  failures; diagnose storage, layout and recurrence separately.
- Run the full video corpus on FP32/CPU, FP16/CPU and FP16/Metal. Also rerun the
  frozen 21-case image matrix and image inference using the image subset of full
  video weights. Community artifacts and synthetic checks cannot replace the
  original-weight reference.
- Fast meaningful tests cover invalid/duplicate/skipped frame indices, resolution
  changes, reset/failure recovery, empty results, capacity boundaries, delayed
  output counts around 14/15/16, association/occlusion/retirement, FP16/BF16 ties,
  256-head query-tile tails, mask component edges and bounded selection over
  1,000-frame synthetic histories. Avoid tests that merely restate implementation.
- A real multi-session/long-sequence run verifies lifetime, cache isolation,
  retained-record bounds and allocation plateau after warmup. Report
  process RSS separately from retained memory and backend allocation; do not
  claim a universal memory bound from one finite run.

## Implementation packages and file ownership

This is a change to more than eight files, with no new service or C++ dependency.

1. **Weights and oracle:** extend converter/schema inspection, video source
   preparation, `tools/export_video_reference.py` and `tools/validate_video.py`.
   Freeze the actual video metadata/tensor/reference manifests and corpus.
2. **Tracker math:** add SAM 3 tracking prompt/mask decoder, memory encoder,
   memory attention and video preprocessing headers. Share the ViT trunk and
   create both necks without encoding the same frame twice. Validate each stage
   against exported original-weight tensors before connecting lifecycle logic.
3. **State and policy:** implement logical groups, selection/eviction, official
   association, reconditioning, capacity, suppression and delayed results. Test
   these rules with independent reference fixtures and bounded counters.
4. **Integration:** add `include/sam/video_session.hpp`, public value types and
   `TextVideoSessionImplementation`; connect the SAM 3 adapter. Add `sam_video`
   using existing image decoding for directories of contiguous numeric PNG
   filenames (`000000.png` onward). Require a new output directory and record
   completion status so partial output cannot be mistaken for successful results.
5. **Acceptance and delivery:** update CTest, consumer and two-TU checks; run the
   full matrix and session checks; update README, architecture, GGUF documentation,
   MODEL_ZOO, BENCHMARK and changelog only with verified support/results.

The example adds `--frames`, `--model`, `--text`, `--backend`, `--threads`,
`--max-objects` and `--output`. The video exporter generates frame recipes and
hash manifests; the C++ example needs no JSON parser dependency. Model state and
image headers retain current behavior. A future SAM 2 or DART adapter can reuse
the public results or validated helpers without taking SAM 3 temporal constants.

## Planned verification commands and performance

The video commands below were planned interfaces in the baseline and are now
implemented. Use the existing authorized checkpoint and pinned BPE asset.

```sh
reference_python=build/reference-runtime/venv/bin/python
sam3_weights_dir=models/official/3c879f39826c281e95690f02c7821c4de09afae7
SAM3_SOURCE_DIR="$HOME/.x-repo/github.com/facebookresearch/sam3"
"$reference_python" tools/convert_sam3.py --task video \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$SAM3_SOURCE_DIR/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --precision f32 --output models/sam3-video-f32.gguf
"$reference_python" tools/convert_sam3.py --task video \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$SAM3_SOURCE_DIR/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --precision f16 --output models/sam3-video-f16.gguf

"$reference_python" tools/prepare_reference_source.py --task video \
  --source "$SAM3_SOURCE_DIR" --output build/reference-runtime/sam3-video-cpu
"$reference_python" tools/generate_video_cases.py \
  --input-root models/fixtures --output models/video-cases
"$reference_python" tools/export_video_reference.py \
  --sam3-source "$SAM3_SOURCE_DIR" \
  --sam3-runtime-source build/reference-runtime/sam3-video-cpu \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$SAM3_SOURCE_DIR/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --cases tests/data/sam3-video-cases.json --frames models/video-cases \
  --device cpu --output models/reference/sam3-video

cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure
cmake --build build/metal --parallel
ctest --test-dir build/metal --output-on-failure
"$reference_python" tools/test_tools.py
"$reference_python" tools/validate_video.py \
  --build-dir build/metal --model models/sam3-video-f16.gguf \
  --reference models/reference/sam3-video --backend metal --threads 4 \
  --output build/video-validation-metal-f16
```

Repeat the validator for FP32/CPU and FP16/CPU. The exporter accepts the same
source/runtime/checkpoint/BPE arguments as the image exporter plus the video case
manifest, and records every adaptation and package version. The paths above
reuse the verified local environment/source layout; another checkout can set
those three variables to its matching paths.

The original benchmark target was FP16; the final approved default is hybrid.
Benchmark that accepted profile on CPU and Metal sequentially, with fixed 1- and 4-object workloads
to expose object-count scaling without redundant intermediate cases. Run one
64-frame sequence per workload: record the first 16 frames as warmup and measure
the remaining 48. Record every
frame and stage time, median/p95, active-object counts, first-output delay,
retained state, peak RSS and backend buffers. Use a separate video table with
model rows and the existing two hardware/backend header rows. Keep image numbers
separate. The M1 Metal median of 6.56 seconds per image is not a video FPS claim;
CPU reference and benchmark runs may be lengthy.

## Risks, failure handling and completion

The most fragile assumption is that forward-only retention can preserve every
future selection while discarding old state. The protected recent window and
unpruned-selector comparison address this directly; an approximate cache is not
an acceptable fallback. Other main risks are BF16 boundaries, logical tracker
group semantics, and the new attention workspace.

Keep the current GGML pin and verified Metal patch. If a required operation still
cannot pass on Metal, record that gap and fix it within the existing backend
boundary before claiming Metal video support. Do not silently reroute failed
execution or broadly loosen tolerances. If the CPU oracle adaptation cannot
produce valid official artifacts, M2 numerical acceptance remains incomplete;
retain M1 support and report the exact blocker rather than substitute community
outputs.

Rollback removes the additive video API/profile and new artifacts; image GGUFs,
original checkpoints and the baseline commit remain intact. No checkpoint is
rewritten, no source checkout is patched in place, and no remote publish is part
of this plan. Completion requires all gates above plus accurate documentation
and reproducible performance receipts.

## Planning record

- Committed the complete validated image/GGUF baseline before beginning M2 planning.
- Read pinned Meta and community implementations, current contracts, schema and
  existing verification evidence; confirmed source revisions and local dependency
  availability without modifying upstream checkouts.
- Fixed the M2 scope, API behavior, format migration, temporal/storage semantics,
  retention design, implementation ownership and acceptance requirements.
- No M2 implementation, model run, package installation or remote write has occurred.

## Cloud implementation record (2026-10-01)

The user requested starting both plans in the cloud while keeping model/Meta and
Metal hardware validation local. Independent M2 foundations were implemented
alongside the FP32 backend change; the image numerical prerequisite is still
open and no completed video support is claimed.

Implemented in this batch:

- `--task image|video` conversion and schema-2 loading for all 1,464 canonical
  tensors, with named profile, tracker architecture, BF16 storage and Pillow/F16
  preprocessing metadata. Default image schema 1 and image tensor bytes are
  preserved. File/sidecar publication remains exclusive.
- Tracker architecture and weight registration, shared-trunk detector/tracker
  neck construction, SAM two-way mask decoder, memory encoder and 256-wide
  tiled memory attention. Tracker neck transport explicitly rounds through BF16.
- Pillow RGB bicubic fixed-point preprocessing and explicit F16 normalization,
  plus F16/BF16 tie checks. The tracker Gaussian matrix registers the official
  canonical `[128,2]` dimensions, correcting the inspected community convention.
- Forward memory selection and retention helpers. Retention keeps complete
  mutable hotstart history; later conditioning membership is not downgraded by
  these helpers. Four conditioning records, eight recent non-conditioning frame
  positions and 15 older eligible records suffice in this forward profile.
  The count of discarded conditioning records preserves the official selection
  branch/order. This count is caller-owned metadata and must be reset with state.
- Deterministic motion, entry, occlusion, hotstart-removal and negative recipes in
  `tests/data/sam3-video-cases.json`, plus hashed PNG generation. Generated input
  manifests explicitly remain ineligible until the official oracle verifies
  the required scenario behavior.

Weight-free selector evidence uses the original unpruned Meta functions,
extracted only after verifying source hashes:

| Source | SHA-256 |
| --- | --- |
| `sam3_tracker_base.py` | `b2b52409c002e1590262375aa794f8ab67e7476f42f8fee41a76de0c14aa62e2` |
| `sam3_tracker_utils.py` | `dc5fdeba2d4416f273394a9bd4450dff050608db6009a4546ca68adbaa24a640` |

`export_memory_selection_goldens.py` reproduces 2,024 cases spanning two
1,000-frame histories, late births, reconditioning, low-quality frames and
permitted hotstart conditioning/quality changes. The retained selector matches
spatial frame indices, temporal positions and pointer order while retaining at
most 27 records. Fixture SHA-256 is
`e4844ec8e03e4e9aab03b29e93d4c4d62d3cef08a9f0df3c22d2d0a321c02594`.
This is selector evidence, not model, association or ID-continuity acceptance.

Remaining implementation:

1. Connect tracker prompt preparation, selected memory/pointers and stage
   execution, then compare every stage with original-weight exports locally.
2. Implement logical tracker groups, official association/lifecycle,
   reconditioning and global overlap/suppression handling. Integrate retention
   with hotstart removal and group-quality recomputation.
3. Add the public video task contract, `VideoSession`, failure/reset recovery,
   source-size delayed results, statistics and `sam_video` CLI.
4. Adapt the isolated CPU video oracle and dependencies; implement full
   `export_video_reference.py` and `validate_video.py`. These commands remain
   planned interfaces and are not supplied by this batch.
5. Run original video/image matrices, multi-session/long-sequence lifetime checks,
   actual Metal placement and video performance measurements locally. Preserve
   frozen gates and mark missing scenarios as fixture failures.

Available local preparation commands from this batch:

```sh
.venv-reference/bin/python tools/convert_sam3.py --task video \
  --checkpoint models/official/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt \
  --bpe models/official/3c879f39826c281e95690f02c7821c4de09afae7/bpe_simple_vocab_16e6.txt.gz \
  --precision f32 --output models/sam3-video-f32.gguf
# Repeat with f16 and a new sam3-video-f16.gguf output.
.venv-reference/bin/python tools/generate_video_cases.py \
  --input-root models/fixtures --output models/video-cases
```

Both outputs are exclusive. Complete generated frames are not reference outputs.
Cloud check results and source/binary hashes are recorded under ignored
`build/cloud-implementation-20261001/`; no model or private media is committed.

Cloud verification completed: CPU CTest **10/10**, downstream consumer CTest
**3/3**, Python tools **13/13**, independent/repeated header compilation,
Python syntax, JSON, local documentation links and whitespace checks passed.
The selector tests cover 2,024 official weight-free cases; preprocessing covers
four Pillow bicubic layouts, padded strides and all 256 RGB byte values.
No official checkpoint was loaded, and no Metal hardware or new model benchmark
was available in this cloud run. This record covers the first cloud implementation
batch; the remaining implementation and local acceptance are listed above.

## Local foundations validation (2026-10-02)

The [FP32 CPU/Metal image prerequisite](20261001-121633-fp32-cpu-metal-validation.md)
is now complete: 28/28 original-weight image cases, actual Metal placement,
FP32/Metal session behavior and four fresh benchmark cells passed. This does not
validate any video tracker graph or temporal policy.

Both schema-2 full files were converted from the pinned original checkpoint:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `sam3-video-f32.gguf` | 3,449,345,696 | `02513232afca5ba8c174b66c7fc839c67b32590df4b53bdd6df5089a6a546844` |
| `sam3-video-f16.gguf` | 1,837,925,216 | `9c9bc86c81d11a041db10a46d3d1e8ecaa1cbcf6fad683b00901f641746bf32c` |

Complete metadata, inventory and sidecar payload checks passed. Independent
Torch dtype conversion of every original tensor reproduced all 1,464 payload
hashes in each file, including complex RoPE real pairs and the original Gaussian
matrix shape. All 1,133 image entries preserve dimensions, dtypes and payloads
exactly relative to their accepted schema-1 counterpart. Only the unused pooled
text projection is excluded and recorded.

The initial FP16 conversion failed on the original object-score head's last
weight, `[1,256]`: source-rank selection requested F16, while canonical GGML
`[256]` requires F32. A sweep of all 1,464 tensors found only this mismatch. The
existing F32 preservation list now contains this exact tensor; no image entry,
C++ precision policy or malformed-shape guard changed. The extended converter
regression failed before this fix, passed after it, and the complete Python
suite passed 13/13. Failed conversion published no FP16 output; the successful
retry used a fresh exclusive publication.

Loading each full file and running `truck-truck` passed on FP32/CPU, FP32/Metal,
FP16/CPU and FP16/Metal. All ten image tensor dumps, selected detections and
output masks match the corresponding accepted image-file run byte-for-byte.
The Metal checks retain zero CPU graph nodes. These four checks cover full-file
loading and the unchanged image subset; they are not a video-corpus run.

Re-exporting the exact hash-verified Meta selector functions reproduced all
2,024 cases and fixture hash
`e4844ec8e03e4e9aab03b29e93d4c4d62d3cef08a9f0df3c22d2d0a321c02594`.
Pinned Pillow 11.2.1 reproduced all four resize layouts. Executing the original
Meta `load_resource_as_video_frames` PIL-list branch with CPU offload reproduced
the F16 normalization goldens for all 256 byte values exactly. C++ padding,
rounding, selector retention and attention checks passed in both local CTest
builds; the maintained tiled-attention test itself executes on CPU.

The five recipes generated 216 contiguous PNG frames under ignored
`models/video-cases/`, with per-frame hashes. Generation is complete, while
`reference_behavior_verified=false` and `eligible_for_milestone=false` remain
correct. Entry, occlusion, hotstart removal and ID continuity have not yet been
established by a full official video oracle.

Receipts and reproduction scripts remain in ignored
`build/fp32-validation/20261002-032709/`: `video-conversion.json`,
`video-profile-image-smoke.json`, `meta-foundations.json`, the red/green precision
logs and `video-commands.json`. See the
[local run record](20261002-032709-local-model-meta-validation.md).
The five remaining implementation items above still block complete video
acceptance, tracker-stage comparisons and video benchmarks. No video support,
commit or push is claimed by this validation batch.

## Local integration continuation (2026-10-02)

The subsequent [video-session and official-validator run](20261002-041320-video-session-official-validation.md)
supplies the missing tracker execution, logical groups/temporal policy, public
`VideoSession`, delayed owned results, `sam_video`, isolated official CPU source,
and original video export/comparison tools. The image regression matrix remains
28/28. Four precision/backend two-frame diagnostics pass, and the full 48-frame
FP16/Metal motion comparison passes with exact temporal traces and zero CPU
fallback. Complete M2 acceptance still depends on verified scenario behavior,
the full video matrix and controlled performance receipts; see that run record
for the precise corpus failures and completed checks.


## Final M2 acceptance (2026-10-03)

The [completion run](20261002-182848-m2-complete-acceptance.md) closes the
remaining implementation, numerical, real-session and performance gates under
the user's explicit **F32 + hybrid CPU/Metal** support contract. Video conversion
defaults to hybrid when precision is omitted; image conversion still requires
explicit precision. Existing F16/F32 files and the temporal/BF16/strict-argmax
policies are unchanged.

| Gate | Completed evidence |
| --- | --- |
| Original video corpus | All five behavior-verified cases, 216 frames per cell, pass F32/CPU, F32/Metal, hybrid/CPU and hybrid/Metal. Maximum stage L2 respectively 0.000203, 0.000346, 0.002650 and 0.002594 under the original 0.001 / 0.02 gates. Candidates, IDs, lifecycle, outputs, retained-state bounds and backend placement pass. |
| Image compatibility | 56 fresh schema-1/full-schema-2 F32/F16 image cases plus 14 matching sealed hybrid cases pass. |
| Real session behavior | Six short configuration checks and two hybrid backend checks, each interleaving 64 positive and 64 negative frames after caller Model destruction. Positive public outputs match standalone exactly; negatives are empty; caches, owned-result lifetime and bounds pass. |
| Performance | Four hybrid backend/object-count cells complete 64 frames with 16 warmup and 48 measured samples. All frame/stage samples, final drain, first output, current/peak RSS, state/backend allocations and AC/no-sleep conditions are retained in [BENCHMARK.md](../../BENCHMARK.md#video-64-frame-protocol). |
| Tooling/integration | 20 isolated Python tests pass. Existing immutable CPU/Metal builds retain their 11/11 CTest results; the new long test independently compiles and runs on both. Separate CMake configuration verifies its optional registration without rebuilding those artifacts. |

Explicit F16/Metal completes the corpus but fails candidate selection at entry
frames 23/24. Exact-F16 original-module replay reproduces those choices with all
1,464 payloads verified. The old 17-frame pressure failure is also retained.
The optional full F16/CPU diagnostic is deferred, not passed; the original
incomplete legacy queue remains preserved. These facts supersede earlier
subset/pending statements for current support, without rewriting historical
receipts or relaxing a gate.

Matching artifact/input/output seals permit baseline reuse. Documentation and
verified CLI dispatch-only changes do not trigger full CPU inference; changed
weight values, preprocessing, graph/backend arithmetic, dependency/toolchain
behavior or temporal state require affected acceptance again. Finite
allocation/RSS observations are not a universal lifetime memory guarantee.
The final local receipts are indexed by
`build/video-validation/20261002-182848/completion-summary.json`.
No commit, push or model upload is part of this completion.
