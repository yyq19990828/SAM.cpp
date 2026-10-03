# Quantization output-quality policy and visual examples

Created: 2026-10-04 01:28:10 Asia/Shanghai.

Status: complete.

## Scope

Make the user's acceptance policy explicit: quantized checkpoints are accepted
by observable segmentation quality, while intermediate tensor errors are
reported separately for diagnosis and comparison. Keep structural validity,
finite outputs, correct preprocessing and actual backend execution as required
checks. Preserve existing numerical receipts and frozen thresholds.

Provide a user-facing, bilingual Markdown gallery with actual outputs for all
currently supported image weight configurations: F32, mixed F16/F32, and the
four vision-only quantization profiles. Use identical images, prompts and
selection thresholds across configurations. Broader diagnostic profiles and
unimplemented SAM adapters must not be advertised as validated configurations.

## Approach

Reuse retained original-checkpoint comparisons and actual C++ masks rather
than generate illustrative predictions. Verify source-image rights before
embedding image pixels; keep private or non-redistributable media out of Git.
Show selected objects, boundaries and output metrics alongside each sample.
Use consistent colors and scale, include positive, multi-object and negative
prompts, and identify the model, storage profile and backend.

User documents contain the comparison images, interpretation and reproducible
usage. Input provenance, receipt hashes, generation details, diagnostic tensor
statistics and verification results belong in this plan or ignored build
artifacts. Reuse existing rendering and reference utilities where practical.

## Steps

1. Check the quantized acceptance path and existing regression coverage.
2. Locate verified output masks, metrics, source fixtures and their licenses.
3. Render a consistent set of comparison assets and bilingual gallery pages.
4. Link the gallery from README, model lists and quantization guides; state the
   output-quality policy and precision-specific object tolerances clearly.
5. Validate the rendered images, their source/output identity, documentation
   links and relevant numerical policy regressions; record the results here.

## Verification

- Quantized primary pass depends on object-quality gates, not tensor fidelity.
- Each claimed weight configuration has actual mask images on the same corpus.
- Images and reported metrics agree with retained original-checkpoint outputs.
- No source photographs with unresolved redistribution rights are committed.
- Gallery image paths and bilingual links pass the documentation checker.
- Inspect rendered assets visually and run `git diff --check`.

## User-specified threshold

The user requested a score threshold of 0.2 for these examples. Frozen 0.5
acceptance receipts remain unchanged. Reapply the current C++ postprocessor
to retained C++ predictions and the pinned official `Sam3Processor` to retained
Meta FP32 predictions. First prove that both paths exactly reproduce every
retained 0.5 selected mask, score and box. Changing the selection threshold
does not change the cached network predictions.

The 0.2 comparison checks every selected object, including lower-confidence
ones, with the existing precision-specific mask/score/box tolerances. Report
selected-query changes explicitly. This is an additional sample comparison,
not a relabeling of the frozen full-inference 0.5 acceptance corpus.

## Results

### Deliverables

- `docs/visual-examples.md` and `docs/visual-examples_zh.md` embed 27 actual
  comparison images: source/reference panels, six-configuration panels for
  each backend, and expandable differences for all positive cases.
- `docs/assets/visual-examples/sam3-{cpu,metal}/comparison.json` contains
  recomputed object metrics and separately reported tensor errors.
- 29 published asset files total 7,058,693 bytes. Unused duplicate reference
  pictures and empty difference panels moved to ignored
  `build/visual-examples/20261004/unused-assets/`.
- `tools/render_image_comparison.py` renders receipt-bound actual masks,
  verifies results/input/mask hashes, handles valid selection thresholds,
  keeps one backend per gallery, and publishes exclusively.
- README, both model lists, both quantization guides and the changelog link or
  describe the gallery and output-quality policy. The document checker also
  compares the bilingual gallery and quantization tables.

The acceptance implementation already used final output quality as its primary
result. It required no numerical-policy mutation in this task. The existing
regression proving a quantized case can pass output quality while failing raw
tensor fidelity still passes. Structural validity, finite values, preprocessing
and actual backend execution remain required.

### Provenance and threshold reproduction

Original predictions came from the 12 retained batches below, under
`build/quantization-repair/20261003/`. Their immutable original-model reference
manifest is
`models/official/3c879f39826c281e95690f02c7821c4de09afae7/reference-unfused-fp32-corpus/manifest.json`.

| Backend | F32 | F16 | Q8_0 | Q6_K | Q5_K | Q4_K |
| --- | --- | --- | --- | --- | --- | --- |
| CPU/BLAS | `legacy-f32-cpu-blas` | `legacy-f16-cpu-blas` | `fresh-q8-cpu-blas` | `quantized-q6_k-cpu-blas` | `quantized-q5_k-cpu-blas` | `quantized-q4_k-cpu-blas` |
| Metal | `legacy-f32-metal` | `legacy-f16-metal` | `quantized-q8_0-metal` | `quantized-q6_k-metal` | `quantized-q5_k-metal` | `quantized-q4_k-metal` |

The private driver hashes the source results and the four actual detector-output
tensors against the original batch receipt before reading them. Original Meta
arrays are checked against their tensor metadata. The current C++
`postprocess_detections` handles actual predictions; the pinned official
`Sam3Processor._forward_grounding` handles original FP32 predictions through
a prediction-returning model stub. Neither path substitutes rounded weights
for the official reference.

Before producing threshold-0.2 outputs, the driver exactly reproduced all
selected masks, scores and boxes in the seven official threshold-0.5 cases
and all 84 C++ threshold-0.5 cases. At 0.2, all 84 configuration/case
comparisons retain identical selected-query sets and meet the existing
precision-specific object tolerances, applied to every selected reference
candidate. This is supplemental sample evidence, not a replacement for the
frozen full-inference acceptance.

`wheel` changes from four selected queries at 0.5 to seven at 0.2. Whole truck
and cropped truck each select one query; the four remaining prompt/image
pairs select none. The gallery reports candidate counts, not physical-object
ground truth. It does not characterize empty grocery outputs as correct
recognition or as proven false negatives without annotations.

### Output measurements at threshold 0.2

Minimum IoU, maximum score error and maximum per-coordinate box error across
all selected reference candidates; box errors here are image-dimension
fractions, not percentages.

| Configuration | Backend | Minimum mask IoU | Maximum score error | Maximum box fraction |
| --- | --- | ---: | ---: | ---: |
| F32 | CPU/BLAS | 1.000000 | 0.000001639 | 0.000000240 |
| F16 | CPU/BLAS | 1.000000 | 0.000133783 | 0.000003255 |
| Q8_0 | CPU/BLAS | 0.998361 | 0.000706018 | 0.000057966 |
| Q6_K | CPU/BLAS | 0.997051 | 0.006856561 | 0.000158727 |
| Q5_K | CPU/BLAS | 0.993781 | 0.010080576 | 0.000416756 |
| Q4_K | CPU/BLAS | 0.978615 | 0.020008370 | 0.000696767 |
| F32 | Metal | 1.000000 | 0.000001684 | 0.000000134 |
| F16 | Metal | 1.000000 | 0.000133440 | 0.000003153 |
| Q8_0 | Metal | 0.998233 | 0.000867665 | 0.000059255 |
| Q6_K | Metal | 0.996725 | 0.006805540 | 0.000158370 |
| Q5_K | Metal | 0.993781 | 0.010150254 | 0.000414276 |
| Q4_K | Metal | 0.978615 | 0.020226315 | 0.000688019 |

### Retained generation evidence

All generation and failed-attempt artifacts are ignored under
`build/visual-examples/20261004/`. The initial adapter invocation failed before
publishing any outputs because the bare reference processor needed its CPU
device property. `prepared-v2` uses that corrected property. The old empty
`prepared` directory is retained, rather than silently mixed into results.

| Artifact | SHA-256 |
| --- | --- |
| `proof.json` | `71cdec41eb11745c067bff6031c97bb3a2ee857df6cc6e50f65b25217ec5ae25` |
| `prepare.py` | `5b37d1fc0b419f26bcae3d94eaafd8f39b0e5e2656522110482ad978c73a8d61` |
| `rethreshold.cpp` | `887a891835aabcda4133a7c935822819fd24c26572f3eb004db2e255a5703045` |
| `rethreshold` executable | `36a925d310fb3bec4a4a21046348d0b0692d630b4e5862f415d76bf4c8d25807` |
| `asset-index.json` | `67a72872d25003bd7526e2b2bb5ce0a67d1e669028fe526301843c06748d61a0` |
| Threshold-0.2 reference manifest | `e15207851a0a5cdd71f49ac3f4f660b597c32d408bf8b8d2e4a26e9e78ca3c94` |
| CPU gallery `comparison.json` | `1facea58701c979bb6b0cca0454ad27401a67385f3b98339a946c6f787a2ab7c` |
| Metal gallery `comparison.json` | `fc4cb5d88d33871457d94f0230e3dce36ad12788d2a8f99896cdc29872b88917` |
| `tools/render_image_comparison.py` | `002a9aa204030a90522b02ed0d07233ceee8ea423d0c9681f190730ed2cd8ee6` |
| Current C++ `image_ops.hpp` | `4d0badbe2f08c206151cc64f9b6c124671bd069720974228bdb5410b17caa16a` |
| Pinned official `sam3_image_processor.py` | `d8738a0efb6138b01c0dc5deceffd29de9e675860a9a1ed3822b08766373333b` |

Reference source-media and model license match the retained
`licenses/SAM-model-license.txt` byte-for-byte. The gallery and third-party
notices credit Meta and explicitly keep derived image/media licensing
separate from the project's C++ MIT license. Source checkpoints, complete raw
tensor dumps and original photographs remain in ignored model/build roots.

### Verification

- Existing quantization policy regressions: 15/15 pass.
- Isolated reference-environment tool suite: 47/47 pass, including the new
  actual-mask renderer check. After the final boundary and box-error assertions,
  the focused renderer test passes again.
- Actual-mask regression proves a changed mask produces IoU 0.5, a half-pixel
  box deviation over two pixels gives dimension fraction 0.25, altered mask
  bytes are rejected, output is not overwritten, and duplicate case IDs are
  rejected before publication.
- Recomputed public CPU and Metal object metrics match all 84 threshold-0.2
  comparison receipts exactly. All 29 public assets match the frozen index.
- Visually inspected the full contact sheet, wheel overlays and boundary
  differences, and the grocery source/reference panel. No visual mask edits.
- Bilingual tables and local image/document links: 45 documents pass.
- Punctuation checks pass for all nine changed user-facing documents.
- `git diff --check` passes. No runtime/model changes or new performance
  measurements in this task; original receipts and frozen gates are preserved.

## Later gallery extension

The user subsequently requested full-component quantization examples and
candidate-confidence labels. Updated published images and their new asset
index are documented in the [modular weight quantization plan](20261004-021711-sam3-modular-weight-quantization.md).
Original assets and this plan's original 29-asset index are preserved in that
work's ignored `gallery/before-confidence/` and
`gallery/original-gallery-asset-index.json`; earlier numerical receipts are
unchanged.
