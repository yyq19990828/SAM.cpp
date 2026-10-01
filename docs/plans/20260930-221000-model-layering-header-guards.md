# Model layering and header guards

Created: 2026-09-30 22:10 Asia/Shanghai.
Status: completed for structure, guards, and regression verification. Original
checkpoint acceptance was pending at this stage; it is now recorded in the
[original-weight acceptance plan](20261001-002850-metal-window-cpu-performance-official-weights.md).

## Scope

Refactor the working SAM 3 image implementation into explicit public, task,
model, and GGML runtime layers. Preserve `sam::Model`, `sam::ImageSession`,
`sam::sam`, checkpoint compatibility, CPU/F16 promotion, Metal precision,
cache behavior, and numerical outputs. Replace every project-owned header's
`#pragma once` with a unique path-derived include guard.

SAM 2/2.1, GroundingSAM, DART, SAM 3.1, and video tracking are extension
scenarios, not implementations delivered by this change. This touches more
than eight files because the current 2,464-line SAM 3 graph and the model
logic inside public headers cross the intended boundaries.

## Approach and boundaries

```text
Host / examples / future composed pipelines
                 |
Public Model + ImageSession + shared value types
                 |
Task-specific text-image session contract
                 |
SAM 3 model adapter: loading, tokenization, cache, diagnostics, transforms
                 |
SAM 3 stages: vision -> prompt/fusion -> detection -> masks
                 |
GGML runtime: resources, scheduling, precision, backend selection
```

- Keep model-specific shapes, tensor names, tokenizer rules, container schema,
  and normalization in `detail/models/sam3/`, scoped to its own namespace.
- Keep common input validation and bounded binary reading independent of model
  implementations. Common public value types must not include GGML headers.
- A small task-specific session contract permits dispatch without assuming
  that every model supports text prompts. A model may reject an unsupported
  task. Future point/box and video sessions get their own contracts when ported.
- Split SAM 3 graphs by their actual stages; preserve imported calculations
  and attribution. Expose detection and mask execution as separate internal
  stages so detection-only pipelines need not run mask computation.
- Preserve the shared-model execution lock and session-owned caches. Avoid a
  plugin registry, generic graph framework, and placeholder model classes.
- [SAM 2/2.1](https://github.com/facebookresearch/sam2) will supply its own
  prompt and memory path. [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2)
  composes a text detector with segmentation; it is not a single checkpoint
  architecture. [DART](https://github.com/mkturkcan/DART) reuses SAM 3 with cached
  class prompts and detection-only execution; multiclass batching is deferred.
- The interface costs one small adapter allocation per loaded model, one session
  allocation, and virtual dispatch per public call outside tensor loops. Plain
  shared state, session implementation, and model adapter form a one-way include
  chain; each graph stage has a direct include dependency.

## Steps

1. Extract common value types and the task contract; move loading and session
   execution into the SAM 3 adapter while keeping the public API stable.
2. Split the imported graph, isolate model-owned IO/tokenizer/transforms, and
   remove SAM 3-specific tensor initialization from the GGML runtime.
3. Add include guards and update targeted contract/graph/header integration
   checks. Document the layers and extension workflow in `docs/architecture.md`.
4. Update `AGENTS.md`, `README.md`, and `changelog.md` with the conventions and
   completed behavior. Record evidence and remaining limits here.
5. Check the user's installed Hugging Face plugin/download access. Prefer its
   authenticated path when exposed; otherwise inspect the existing supported
   HF CLI login without printing credentials. Download original weights only
   through an authenticated, authorized route. License acceptance and tool
   authentication are separate states; do not mark official acceptance complete
   without original-checkpoint provenance and the reference comparison.

## Verification

- CPU and Metal Release builds and CTest; standalone consumer and two-TU link
  checks. Compile all project headers independently and repeatedly, checking
  unique guards and dependency direction without testing trivial accessors.
- Existing meaningful contract, graph, backend, and precision checks continue
  to run. Add a focused task capability/lifetime check only where the new
  dispatch boundary introduces behavior.
- Reuse the seven-case same-weight reference corpus for numerical regression,
  including empty detections, two images, changed prompts, and mask output.
  Compare new artifacts with the saved accepted outputs; retain all thresholds.
  This refactor does not turn supplementary provenance into official acceptance.
- Run tool tests in the isolated reference environment and `git diff --check`.

## Progress and results

- Before implementation: confirmed the existing public headers contain SAM 3
  constants/session logic and the GGML runtime initializes SAM 3 tensor names.
- Hugging Face plugin discovery reports installed/enabled; this session's tool
  inventory does not currently contain HF model or download connector tools.
- The supported local HF CLI and SDK authentication checks returned HTTP 401.
  No original checkpoint was downloaded. This is independent of the user's
  accepted license. Privacy-safe evidence: `build/refactor-validation/hf-auth-status.json`.
- Five existing Python tool tests passed in the isolated reference environment.
- The nine graph modules preserve all 31 imported executable definitions.
  Twenty-nine definitions are byte-identical; two differ only in corrected
  pretrained-grid comments. See `build/refactor-baseline/final-structural-audit.json`;
  the original monoliths remain beside that report.
- Regression scope: run all seven frozen cases on Metal and the representative
  `truck-truck` case on CPU in both FP16 and FP32, then compare tensors, selected
  detections, boxes, scores, and masks bitwise with pre-refactor accepted outputs.
  This exercises both loader representations and complete graph paths without
  repeating the unchanged full three-backend benchmark/acceptance matrix.

### Delivered

- Backend-independent public value types and text-image task contracts; the
  existing public model and session calls remain unchanged. An unsupported
  task fails explicitly without requiring a tokenizer in every model.
- Model-scoped loading, tokenization, transforms, postprocessing and caches;
  plain shared state and a cycle-free adapter/session include chain.
- Independent SAM 3 prompt, fusion, detector, and mask execution stages; all
  model-specific zero-input names moved out of GGML runtime code.
- Unique include guards in all 26 owned headers (24 library, two example).
  Vendored STB headers and the pinned GGML precision patch remain unchanged.
- Updated architecture guide, README, contributor instructions, and changelog.

### Verification results

| Check | Result |
| --- | --- |
| CPU / Metal Release builds | Passed, including independent/repeated inclusion of every owned header |
| CPU / Metal CTest | 8/8 each, including two-TU linkage and backend-independent unsupported-task handling |
| Downstream consumer, CPU / Metal | Build and 2/2 CTest each |
| Python tool tests | 5/5 in `build/reference-runtime/venv` |
| Metal frozen numerical corpus | 7/7 passed with unchanged gates |
| Before/after bitwise regression | 9 runs, 90 tensors, 8 masks; identical tokens, selected queries, scores and boxes |
| Metal session behavior | Passed after destroying the last public Model handle; changed prompts/images, shared-session isolation and cache reuse remain correct |
| Whitespace | `git diff --check` passed; all 73 untracked source/document files also checked independently |

Bitwise receipts: `build/refactor-validation/metal-f16-bitwise.json`,
`cpu-f16-bitwise.json`, and `cpu-f32-bitwise.json`. The first covers all seven
cases; each CPU receipt covers `truck-truck` through the complete pipeline.
Session evidence is `build/refactor-validation/session-metal/results.json`.
The final rebuild and CTest runs followed restoration of source comments;
the structural audit confirms those final comment edits changed no executable
definitions. No benchmark or complete CPU corpus was repeated for this refactor.

The precision patch's blank unified-diff context markers were preserved rather
than trimmed; its pinned SHA-256 and patch syntax still validate. All regression
artifacts remain supplementary, with `eligible_for_milestone: false`. No original
checkpoint, new model family, or video implementation is claimed. No commit or
push was made.
