# SAM 3 modular weight quantization

Created: 2026-10-04 02:17:11 Asia/Shanghai.

Status: complete.

## Scope

Implement user-selectable local weight quantization for SAM 3 image inference.
Expose model components such as vision, text, fusion and decoder, allowing
users to select one or combine them. Keep activation precision unchanged.
Retain floating-point bias, normalization, positional/token parameters and
weights without a supported quantized row layout. "Full" means all eligible
weights across model components, not every tensor or activation in low bits.

Repository original-model numerical testing will cover a bounded preset set:
vision-only and full eligible-weight quantization, at Q8_0/Q6_K/Q5_K/Q4_K on
CPU and Metal. Arbitrary component combinations are user-configurable and
structurally tested; they are not advertised as individually quality-validated.
Existing schema-3 profiles, output gates, old receipts and galleries remain
interpretable and unchanged.

## Approach

Use the existing converter, packed GGML row formats and shared CPU/Metal
execution. Add an explicit versioned module-selection contract to GGUF rather
than infer policy from a tensor's actual dtype or broaden old profile names.
The initial module selection uses one requested bit format for all selected
components, with deterministic layout fallbacks. Fine-grained scope is the
requested feature; arbitrary per-tensor bit-width rules are outside this step.

Confirm the tensor-to-component mapping against the canonical image inventory
and actual graph consumers before finalizing the new format. Expand eligible
linear weights first; retain convolution/embedding or irregular matrices when
the existing operators/layouts cannot consume their quantized representations.
Describe exact exceptions and allocation counts in conversion manifests.

## Steps

1. Audit canonical tensors and consumers; freeze the module and fallback rules.
2. Implement component selection in conversion and encode it in versioned GGUF
   metadata; reject conflicting options and invalid module names.
3. Validate canonical tensor types against module policy in the C++ loader
   before allocating backend weights. Preserve existing schemas and profiles.
4. Add behavior and malformed-metadata/layout regressions in Python and C++.
5. Convert all four full presets from the original checkpoint. Build CPU and
   Metal, run meaningful tests and original-model comparisons for both preset
   families. Report final output quality separately from tensor fidelity.
6. Extend the visual Markdown gallery with full-preset outputs at the user's
   0.2 sample threshold. Update user-facing conversion/support docs and changelog.
7. Record complete numerical, provenance and validation results in this plan;
   do not claim full presets supported if their output quality fails.

## Format and interface decisions

- New image-only SAM schema 4, retaining GGUF container v3. Existing SAM
  schemas 1/2/3 and every existing profile's allocation stay unchanged.
  Packed encoding continues to require `general.quantization_version=2`.
- `image-full-linear-{precision}-v1` is the all-component preset.
  `image-modules-linear-{precision}-v1` is a custom component allocation.
- `sam.quantization.modules` is a required STRING containing a nonempty,
  canonical-order CSV. Canonical order is `vision,text,fusion,decoder`.
  Profile names and schema encode version and full/custom semantics; no
  redundant GGUF preset/version keys are needed.
- Sidecars and trailing `ModelInfo::quantization_modules` use a canonical
  array of component names. Full requires all four; custom remains diagnostic
  even if its selection happens to contain all four components. The new runtime
  field remains empty for legacy schemas; schema-4 reports the validated list.
- CLI `--quantize-modules` accepts user module selection and derives the
  custom profile. Existing `--storage-profile` selects vision or full presets.
  Contradictory options, unknown names and duplicates are rejected.
- Module boundaries: vision covers ViT linear weights, with neck convolutions
  left floating-point; text covers transformer linears plus its resizer;
  fusion covers `fenc`; decoder covers detector, geometric-prompt, scoring
  and segmentation linears (`ddec`, `geom`, `scoring`, `seg`). Exact eligibility
  and exceptions follow consumer/layout audit, not arbitrary name substrings.
- Full eligibility is 348 matrices: vision 128, text 97, fusion 36,
  decoder 87. This includes underscore-suffixed fused projection weights and
  the text resizer, not merely tensors whose names end in `.weight`.
- The sole K-format fallback remains the 32 ViT `mlp.lin2.weight` matrices
  with `ne[0]=4736`, stored as Q8_0. All other eligible rows are 256-aligned.
  Embeddings, convolutions, normalization, bias and the small/non-aligned
  geometry/RPB matrices stay F32. Legacy rules remain exact and unchanged.
  Canonical singleton-vector parameters also stay F32, including the tiny
  scalar-output `ddec.presence_token_head.layers.2.weight` (`[256]` canonical,
  `[1,256]` original). The decoder module includes canonical point/geometry
  linears not consumed by the current text-only graph; compressing their
  storage does not implement or validate point/box prompting.
- Full presets use separately frozen raw/output gate files with the existing
  precision-specific numerical thresholds. Custom comparisons may derive
  diagnostic tolerances but must never become release-eligible merely by
  selecting all modules or passing a numerical comparison.

## Verification

- Unit and format checks cover supported component combinations, canonical
  allocation, unsupported names, invalid metadata types, profile disagreement,
  row fallback, finite packed values and legacy compatibility.
- Header-only linkage and Release CTest pass on selected CPU/Metal builds.
- Isolated conversion/reference tool checks pass.
- Original FP32 references remain the output oracle. Use precision-specific
  object-quality tolerances, exact input/token/shape checks, finite values and
  actual backend execution; raw tensor errors are diagnostic.
- Runtime changes require fresh numerical evidence. If full-profile quality
  fails, investigate the concrete output failure and retain honest support
  boundaries rather than relaxing gates after observing results.
- Gallery sample threshold is 0.2; frozen 0.5 acceptance receipts remain intact.
- User docs distinguish configurable capability from tested presets, modules
  from model families, and measured hardware from project platform goals.
- `git diff --check` and bilingual/link/image checks pass.

## Results

### Implementation and verified scope

Implemented image-only SAM schema 4 with explicit component selection, fixed
full presets and custom diagnostic profiles. `--quantize-modules` controls
vision/text/fusion/decoder scope with one requested quantization type. Sidecars
record module selection and each tensor's exact storage/retention reason;
C++ loading checks schema/profile/CSV and canonical dtype closure before
backend weight allocation. Runtime image JSON and trailing `ModelInfo` report
schema-4 modules without changing legacy aggregate initialization.

The 348 eligible matrices contain 774,312,960 parameters: vision
444,596,224, text 302,252,032, fusion 9,437,184, decoder 18,027,520.
Full Q8 has 348 Q8_0 tensors and 785 F32 tensors; full K profiles have
316 main K tensors, 32 Q8_0 fallback tensors and 785 F32 tensors. These are
all-component mixed-weight profiles, not every parameter or activation in
low bits. No activation, video state or backend arithmetic policy changes.

### Original-model numerical acceptance

A fresh sequential matrix used the original Meta checkpoint/oracle, identical
PPM inputs, original seven prompts/transforms, threshold 0.5 and four threads.
CPU used the registered BLAS backend with `VECLIB_MAXIMUM_THREADS=4`; Metal
required zero CPU/BLAS compute fallback. Independent new build directories
preserved older binaries and immutable archived evidence.

- Full presets: four precisions x CPU/Metal x seven cases = 56/56 pass.
- Existing vision presets, using unchanged old GGUFs: 56/56 pass.
- F32/F16 image regressions across CPU/Metal: 28/28 pass.
- Primary outcome remains output quality; raw tensor errors stay separate.
- Raw fidelity differences are material, e.g. full Q4 text-feature relative L2
  reaches about 0.414 while selected masks meet the 0.90 IoU requirement.
- This is limited-corpus reference agreement, not human-annotation accuracy.

Full-preset output measurements at the frozen 0.5 threshold:

| Precision | Backend | Minimum mask IoU | Maximum score absolute error | Maximum box fraction error |
| --- | --- | ---: | ---: | ---: |
| q8_0 | cpu | 0.998527 | 0.001040 | 0.000374 |
| q8_0 | metal | 0.998233 | 0.000895 | 0.000375 |
| q6_k | cpu | 0.997074 | 0.004478 | 0.001065 |
| q6_k | metal | 0.997074 | 0.004368 | 0.001066 |
| q5_k | cpu | 0.993199 | 0.010425 | 0.001635 |
| q5_k | metal | 0.993199 | 0.010253 | 0.001631 |
| q4_k | cpu | 0.969362 | 0.019633 | 0.002322 |
| q4_k | metal | 0.969965 | 0.019353 | 0.002396 |

Actual full GGUF files, converted from the original checkpoint:

| Precision | Bytes | Model SHA-256 |
| --- | ---: | --- |
| q8_0 | 1,096,595,296 | `e41f2c652ba97c1081e07317731370482e1adbc0582fb91b44852395be36ebfb` |
| q6_k | 946,651,328 | `13fbcee22d8d163345c0c2d963bf64890955e48dbadfa9bb6f99d341ca06cd02` |
| q5_k | 864,423,904 | `a4689672c8669fedc282fffdae3c28e70376af546bdf824b52aa807a8bb68e25` |
| q4_k | 787,033,440 | `a46044cb86da6ef314b42d63956a57559b6cbdd04b5ee044164c6799198590ff` |

The public machine-readable index is
[modular-quantized-image-m4pro-20261004.json](../validation-baselines/modular-quantized-image-m4pro-20261004.json).
It records the measured hardware rather than a repository platform boundary.
Full-preset latency was not formally remeasured in this task; earlier vision
performance receipts retain their original scope.

### Independent review and tool repairs

Review found three concrete issues after the initial successful matrix:
module-list duplicate detection could take quadratic time on oversized input;
retained modern dual-axis receipts were compared against raw top-level gate
identities; per-tensor module/reason annotations were not revalidated.
Two patches were prepared in ignored `review-fixes/`, then applied only after
the numerical matrix completed with every source hash unchanged. They cap
CSV/item counts before expensive parsing, bound iterables to five reads,
use a linear `seen` set, validate output and raw gate identities independently,
and recompute schema-4 tensor explanations in validator and exporter.
Historical schema-3 receipts and all three older frozen gate files remain
unchanged and compatible.

The exact initial sources are retained under `initial-source/` and indexed by
`validation-source-sha256.json`. Only Python tool/test/render files changed
after that matrix; C++ and dependency sources are unchanged. The final audit
rechecked original-model provenance, all output file hashes, modern receipt
gate bindings and every case comparison using repaired tools. All 140 case
metrics are identical. Final code review confirms the findings resolved.

### Custom component integration

A real `text,decoder` Q6_K model was converted after the parser repair. Its
sidecar/loaded model reports schema 4 and exactly those two modules, with
184 Q6_K matrices, 949 F32 tensors and no vision Q8 fallback. A seven-case CPU
comparison with `--allow-custom-quantization` passes while remaining
`profile_family=custom`, `profile_status=diagnostic`,
`eligible_for_milestone=false`, `release_support_eligible=false`.
This integration probe is separate from the official preset matrix and does
not claim exhaustive custom-combination accuracy.

### Threshold-0.2 visual examples and confidence steering

Added full-preset images to the existing bilingual gallery. All reference and
C++ outputs were re-postprocessed at 0.2, after exact reproduction of stored
0.5 selected masks/scores/boxes. The full-gallery reproduction includes
56 full cases and 28 retained F32/F16 baseline cases. Reference decoding and
original Meta threshold-0.2 masks reuse the previously verified original
visual gallery inputs.

At 0.2, wheel candidates differ: reference/baselines/vision and full Q8/Q6
select seven; full Q5 selects six, full Q4 selects four on both backends.
The exact selected query sets therefore differ for one displayed case in each
full Q5/Q4 cell. This is displayed directly, including below-threshold
confidence values; it is not hidden behind matched-object IoU. These are
candidate/query counts, not physical-wheel annotations. The frozen 0.5
acceptance and the additional 0.2 sample comparison are explicitly separated.

At the user's request, every selected candidate now has a `q` badge and a
color-keyed confidence legend, formatted to three decimal places. Source
scores remain unrounded for filtering. Confidence is distinguished from IoU
and annotated accuracy; `q` identifies an image detection query, not a video
track ID. Matching candidates retain colors/IDs across configurations.

The published gallery contains 47 images and four comparison JSON files,
51 assets totaling 13,724,188 bytes. Their hashes are in
`gallery/final-asset-index.json`. Old assets are retained in
`gallery/before-confidence/`; the previous 29-asset index is preserved in
`gallery/original-gallery-asset-index.json`. Unused duplicate reference and
empty difference images moved into ignored `gallery/unused-final-assets/`.
The Meta source-media license and retained license match byte-for-byte.

### Checks and retained evidence

- Release CPU and Metal builds pass, including independent/repeated-header
  compilation and two-TU linkage. CTest: 12/12 per build.
- Final isolated Python tool suite: 65/65 pass. Negative cases intentionally
  print failed sample-validation records; the suite exits successfully.
- Full module format checks: 10/10; quantization policy/receipt checks: 23/23.
- Candidate confidence arrays match actual selected-result scores/counts;
  all published asset hashes match the final index.
- Visually inspected CPU/Metal full wheel panels, reference panels, boundary
  differences and the overall gallery layout; no generated mask pixels edited.
- Bilingual tables and all local image/document targets: 46 documents pass.
- Punctuation checks pass across the 12 affected user-facing documents.
- `git diff --check` passes; no commit or push requested or performed.

Private evidence paths below are relative to ignored
`build/modular-quantization/20261004/`:

| Artifact | SHA-256 |
| --- | --- |
| `matrix.json` | `026b05862bc129c193725208ef9807d05e1b8e3be0026a0094f399a76cf6df97` |
| `final-audit.json` | `8f631901c726e66784a412a6570651ab88de3444985f2a3f92c1fb1246b0e9fd` |
| `conversions.json` | `f5f78e0490b8259e70da551d01360fe913db762d81d164fdbae56e9c16cd3f34` |
| `conversion-source-sha256.json` | `5855dc6a99536682b6aa04b507b068d6979d5fb16f20cc9ad12e7faa25fab1ee` |
| `validation-source-sha256.json` | `cfa42a94a3b0a0252b0bcc36aaf2abe2ad120144da3047c3803de8372be39fba` |
| `gallery/final-asset-index.json` | `0bd4f20dedb007a3060febfe0696a1f88bf01a328db468c3d943ad6569e030d8` |
| `gallery/proof.json` | `a241fe192853efcf6cb22f36c7c59e06503fa017ccdb9e72b8a8273dae59f942` |

### Commit preflight provenance label repair

The final staged-tree review found that the generic image renderer labeled
supplementary converted-weight references as official Meta FP32 references.
An integration regression reproduced the wrong panel caption on the old
renderer, then passed after captions were selected from reference kind.
Official original/rethresholded references retain their existing captions;
supplementary and unknown diagnostics no longer claim an original checkpoint.
Published gallery pixels are unchanged because all current inputs are official.
The clean staged source export is checked under
`build/commit-preflight/20261004-modular/` before the user-requested local commit.
