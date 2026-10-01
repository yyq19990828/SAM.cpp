# FP16 Compute Precision Correction

Status: Complete for the supplementary validation scope. This is a necessary extension of the first image milestone;
the user has authorized implementation with GPT-6.1-sol/xhigh subagents.

## Evidence and Scope

The seven-case supplementary FP32 CPU suite passes. Existing FP16 CPU and Metal
paths pass high-confidence detection checks but fail some frozen raw-tensor
gates. Running the exact FP16 stored values in the pinned Meta FP32 runtime passes
all seven FP16 cases, with maximum normalized L2 0.0130666. Weight rounding alone
therefore does not explain the implementation failures.

A small matrix probe proves that the pinned Metal dense matrix kernel narrows
FP32 activations to half, even when `GGML_PREC_F32` is requested. Its current
precision flag does not select a different kernel. Preserve the frozen numerical
gates, weight-file format, public session ownership, and actual Metal execution.

## Approach

1. Promote validated FP16 values to FP32 when allocating CPU model weights. Keep
   storage precision distinct from actual buffer size and report the memory cost.
   Do not re-quantize or alter source files. CPU graph construction must use the
   resulting FP32 convolution and matrix paths.
2. Add the smallest explicit Metal dependency patch that honors requested FP32
   arithmetic for dense matrix multiplication, including floating-point operand
   staging and correctly sized threadgroup memory. Check attention and convolution
   paths for additional narrowing using measured stage errors.
3. Keep the pinned shared GGML checkout unchanged. Store the reproducible patch
   and its provenance in this repository; prepare and build a separate source
   copy. An embedding application's existing GGML target remains caller-owned and
   must satisfy the documented precision contract.
4. Request the required precision from shared SAM graph helpers. Avoid separate
   CPU/Metal model graphs, thousands of per-column graph nodes, or undisclosed CPU
   fallback for the large Metal operations.

The CPU weight buffer may grow from approximately 1.8 GB to 3.4 GB for an FP16
container. Precise Metal arithmetic may be slower. Measure both instead of
retaining the previous latency/memory figures as claims about the corrected path.

## Delegation and Sequence

- Core agent: CPU weight promotion, SAM precision requests, activation handling,
  and internal runtime metadata.
- Metal agent: isolated GGML kernel/dispatch patch and small arithmetic probes.
- Integration agent: deterministic patch application, dependency compatibility
  documentation, and meaningful precision regression checks.
- Coordinator: integrate only tested changes, rerun model comparisons, record
  performance and remaining limits, update README/changelog and the first plan.

Complete tiny arithmetic regressions before expensive model inference. Reuse the
existing source hashes, converted containers, reference corpus, and validator.

## Verification and Acceptance

- A precision-sensitive F32 matrix probe must retain fractional input information
  on CPU and actual Metal; default and explicit precision behavior must be clear.
- Build both backend configurations and standalone consumers; retain the
  two-translation-unit check and five existing fast behavior tests.
- Rerun the full seven-case FP32 CPU, FP16 CPU, and FP16 Metal matrix with unchanged
  tolerances. Attribute any remaining differences stage by stage.
- Verify that Metal still executes large matrix/attention work, that session
  cache/ownership checks pass, and that buffer/RSS reporting reflects promotion.
- Rerun five warmed full-image timings after source stabilizes. Preserve previous
  failed reports as diagnostic history.
- Verify that no shared upstream source or external checkpoint was modified.

Original Meta checkpoint access is still a separate missing prerequisite. Passing
supplementary references must never be relabeled as original-checkpoint acceptance.

## Progress

- [x] Record failed gates and prove the same-weight FP32 arithmetic floor.
- [x] Implement and validate the bounded precision corrections.
- [x] Integrate patched dependency support and regression checks.
- [x] Rerun the model matrix, benchmarks, and documentation checks.

## Results

All seven supplementary cases pass the unchanged gates for FP32/CPU, FP16/CPU,
and FP16/Metal. Worst normalized tensor L2 is 0.000570083, 0.0130565, and
0.0130338 respectively. Every high-confidence mask has IoU 1.0. Direct CPU/Metal
comparison with identical FP16 weights passes all 70 tensors, maximum L2
0.000163574, with no selection differences.

The three-file GGML patch is recorded in `cmake/patches/` with source and patched
hashes. It preserves full float staging for requested precise dense products and
uses the existing F32 matrix attention path for short queries. Large and small
operations still execute on Metal. Runtime initialization rejects an incompatible
Metal dependency instead of silently using its half-input arithmetic.

CMake copies supplied/fetched source, applies the patch outside parent Git
discovery, verifies all three output hashes, and records truthful dependency
provenance. A regression reproduces a source copy inside an existing repository,
unchanged reconfiguration, drift repair, and original-source preservation.

CPU loading preserves all 63,488 finite FP16 bit patterns during promotion and
roundtrip. Container precision remains visible in `ModelInfo`; actual allocation
statistics report the increased CPU weight representation. Source/container files
are unchanged.

Corrected matrix receipts are under `build/validation-precise-*`; direct backend
comparison is under `build/backend-agreement-precise/`. Earlier failed receipts
remain available. Original-checkpoint provenance is still a separate incomplete
gate in the parent milestone.

Final standalone builds pass seven CTest checks per backend; downstream consumers
pass linking and precision checks in both configurations. The corrected Metal
two-session/caching test passes. Five warmed FP16 full-image runs measure medians
of 59.288 seconds on CPU and 6.836 seconds on Metal, with peak process RSS of
4,834,148,352 and 2,627,977,216 bytes respectively. The parent plan records input,
hardware, memory interpretation, initialization costs, and artifact paths.

The explicit startup precision probe caught an initially skipped Git patch;
preparation now verifies final file hashes as well as command success. Shared
upstream checkouts are clean. Local links, Markdown punctuation, project
whitespace, and patch application checks pass. No original-checkpoint acceptance,
remote CI, commit, or publication is claimed.
