# Quantization reference benchmarks and precision clarity

Created: 2026-10-10 15:04:21 Asia/Shanghai.

## Scope and decision

The user defines repository COCO results as public reference measurements, not
universal deployment qualification. Application owners compare a custom recipe
with the original SAM 3 checkpoint or a converted F32 baseline on their own
images and prompts. They decide the acceptable quality/resource tradeoff.

Implement a separate advisory benchmark workflow. Preserve all v2/v3 policy
files, historical receipts and the interrupted campaign without regrading them.
Do not resume GPU inference as part of this change. The first delivery supports
existing native weight/module combinations, CUDA compute policies and image
cache options; it does not advertise new W8A8/FP8 or per-layer mixed-format
runtime support.

Hard checks concern valid evidence and implementation correctness: readable
models, valid shapes/finite values, matching input/prompt/token identities and
unaltered artifacts. Segmentation differences, missed/added reference objects,
score/box changes and resource benefits are measurements or optional user
advice. No COCO quality budget determines whether a custom model can run or
whether a benchmark can be produced.

## Approach

1. Document weight storage, activation representation, arithmetic operands and
   accumulation, and retained image features separately. A compute policy name
   is not proof that every operator uses that dtype. Current CUDA F16 hints
   apply to supported dense operations; quantized GGML dispatch has its own
   integer/dequantization paths.
2. Add an application-image benchmark entry point with explicit cases and
   identity-bound output bundles. Reuse ranked output exporters and spatial
   one-to-one matching, without invoking COCO dataset qualification or frozen
   quality profiles. Support original-checkpoint and native-F32 references.
3. Report agreement with the chosen reference, not production accuracy. Include
   union-mask IoU, matched-mask distribution, missing/added objects, confidence
   and box changes, empty-reference prompts and per-case details. A reference
   model can be wrong; human labels or review remain separate evidence.
4. Default to reporting without any quality threshold. Optional user limits
   produce advisory results and do not change the success exit code of a valid
   report. Invalid or mismatched evidence produces a command error.
5. Put the advisory workflow first in current guides; retain legacy acceptance
   commands as opt-in historical policy workflows. Keep resource measurements
   separate from segmentation agreement. Export elapsed time is not inference
   latency or a valid peak-memory comparison.

## Steps

- Add the benchmark exporter/comparator and CPU-only behavioral tests.
- Add bilingual usage documentation and a current capability matrix.
- Update tools/validation/quantization entry documentation and changelog.
- Run targeted tests, the isolated Python tool suite, documentation validation
  and `git diff --check`; verify v2/v3 policy identities are unchanged.
- Record completed scope and remaining engineering work here.

## Verification and practical limits

Tests must cover query reordering, missing/extra objects, empty references,
reference fidelity rather than GT claims, optional advisory-limit exceedance,
changed input/prompt/output bytes, incomplete exports and preservation of
historical policies. End-to-end tests use tiny synthetic images and recorded
ranked payloads; they must not initialize CUDA or run a model.

The previous GPU-loss campaign remains stopped. Local model/data artifacts
under `build/v3-first/` and `models/` are Git-ignored and are available only on
this workstation or separately retained archives.

## Subsequent work

- Add operator-level evidence for actual operand/accumulator/kernel choices;
  expose requested versus observed precision in a unified recipe report.
- Decide and implement per-module/per-layer mixed weight formats with explicit
  GGUF contracts rather than accepting unsupported configuration names.
- Promote activation experiments only after native operators, scale contracts,
  numerical checks and target-hardware measurements exist.
- Decouple reusable performance measurement from legacy quality prerequisites;
  publish COCO results as descriptive measurements with sample coverage and
  uncertainty, and let applications select optional policies and holdouts.
- Evaluate production references on a small human-labeled audit set when
  available; same-runtime F32 comparisons isolate incremental changes, while
  original-checkpoint comparisons also include conversion/runtime differences.

## Results

Implemented `tools/benchmark/quantization_benchmark.py` with two commands:

- `export` accepts application images/prompts without COCO annotations or a
  quality tier. It reuses the existing native/original ranked exporters, retains
  normalized input images and source/model identities, and publishes completion
  only after validating the output bundle.
- `compare` accepts application bundles or completed legacy ranked v2/v3
  exports. It reports reference agreement and per-prompt differences, optionally
  evaluates user advice, and succeeds for a valid report even outside those
  limits. Input/prompt/token/checkpoint mismatches and missing/changed payloads
  remain errors. Performance and production accuracy are explicitly unmeasured.

Added bilingual application guides, documented the current precision capability
boundaries, and placed the advisory workflow first in README, quantization,
validation and tools documentation. Historical v2/v3 commands remain opt-in
policy workflows with unchanged semantics; no receipt or frozen policy was
regraded.

Verification completed:

- Nine new CPU behavioral tests cover query reordering, duplicate candidates,
  missing/added objects, empty references, non-veto advice, legacy reuse,
  artifact/input/token mismatches, custom export assembly and non-publication of
  incomplete exports. Model exporters are mocked; this is not runtime model or
  GPU-kernel validation.
- `.venv-reference/bin/python -B tools/test_tools.py`: 197 tests in 52.266 s,
  `OK (skipped=1)`. The optional CUDA runtime-quantization test skips because
  CUDA is unavailable to the isolated test environment. Expected `FAIL` output
  from negative fixtures does not indicate a suite failure.
- Documentation validation: local links and bilingual tables pass for 144
  documents. `git diff --check` passes.
- Frozen v2/v3 policy SHA-256 values remain
  `4cf06bdc609e5a143b2d74b42f394480bbe39e7510726ecedfe44fde3212a802` and
  `822a617b90dcd5b45153b777d5e720272f234d38ae8908b3422d27357d842281`.

No inference or long GPU acceptance was started. No model checkpoint, usable
GGUF or previous campaign output was modified. Test intermediates use temporary
directories and were removed automatically. Remaining work is listed above;
this delivery does not yet add activation/per-layer mixed-format execution,
per-kernel precision tracing, lower-than-0.5 mask export, GT evaluation on
arbitrary application labels or an independent performance runner.
