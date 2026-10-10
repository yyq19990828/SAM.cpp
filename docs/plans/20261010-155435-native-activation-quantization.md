# Native activation quantization implementation plan

## Status and scope

Planned, not implemented. The unified image configuration currently accepts
only `activation.mode=backend-selected`. Explicit INT8/FP8/F16 activation
settings are rejected. CUDA quantized kernels may already stage RHS values as
Q8 internally; that is backend behavior, not a user-controlled activation policy
or evidence of full-model W8A8 execution.

The first native activation feature will expose a narrowly scoped dynamic Q8
RHS policy for eligible vision linear operations on CUDA with Q8_0 stored
weights. Other operations remain backend-selected. Independent FP16 graph
activations, full-model INT8/FP8, other quantized weight formats and video memory
are later extensions. Do not run large GPU acceptance as part of this planning
or configuration work.

## Existing evidence and gaps

- `export_calibration.py` and `study_activation_quantization.py` provide vision
  statistics and offline numerical comparisons. `runtime_quantization.py`
  implements PyTorch studies, including reparameterization and W8A8 variants.
  Floating-point simulation does not demonstrate native INT8 execution.
- Existing CUDA Q8 RHS/MMQ RHS and raw-dot probes bind packed bytes and integer
  dots to the pinned GGML library. They are useful building blocks, not proof
  that the complete model uses that path for every linear shape.
- The backend owns policy through `src/runtime/ggml/backend.hpp` and
  `backends/cuda.hpp`; model graphs must not select device kernels directly.
  Current public `BackendOptions` exposes compute hints, not activation policy.
- Q8_0 weights, Q8_1 RHS packing, MMQ packing and PyTorch symmetric per-token
  INT8 use different layouts/scales. Do not substitute one encoder's arithmetic
  contract for another or rename all of them to a generic A8 guarantee.

## Design and first policy contract

1. Use an explicit scoped mode, provisionally `ggml-q8-rhs`, with selected vision
   linear families. Enable its configuration schema only after native controls,
   capability checks and execution reporting exist. Initially accept CUDA,
   Q8_0 target weights and known eligible vision matrices; reject other targets
   before model execution. Protected/nonlinear/attention/convolution operations
   retain their documented policies and are outside the A8 claim.
2. Resolve the pinned backend's MMVQ/MMQ packing and shape support before choosing
   a forced execution path. Make path selection explicit for the supported
   shapes. If GGML cannot honor the requested RHS policy, return unsupported;
   never silently take floating-point dequantized GEMM and call it INT8.
3. Specify the actual Q8 RHS block scale, rounding, clipping, zero-block behavior,
   sums/layout, decode and dot accumulation for each enabled kernel family.
   Reuse the pinned native encoder and independently verify bytes/results.
   `compute=f16` must not be presented as overriding integer accumulation.
   Reject a compute combination unless its precise interaction is implemented.
4. Record requested activation policy, eligible/executed operation counts,
   shape/type/path identifiers, packed input/output representation and measured
   transient bytes. Graph tensor types, packing receipts and actual kernel
   execution evidence are different fields. Missing trace evidence stays
   `NOT_COLLECTED`; a global runtime label cannot replace that evidence.
5. Keep image-feature cache independent. An A8 RHS temporary does not change the
   retained cache and does not turn `mixed-q8_0` into an activation-wide mode.
   Cache reuse must include any activation policy that changes the graph/results
   in its invalidation identity.
6. Add a public value type/options only after the private path and backend
   contract are stable. Public headers remain standard-library-only; JSON stays
   in the host tooling. The public API must reject unsupported model/backend
   combinations, and C++ examples must report resolved policies accurately.

Dynamic RHS block quantization needs no precomputed calibration for its initial
encoder. Existing calibration statistics can help identify sensitive layers but
do not define correctness thresholds. If smoothing/reparameterization is added
later, bind calibration inputs, transformation scales and altered weights to a
new artifact identity. Never apply a calibration transform to an existing model
silently or treat a study's preset quality limits as deployment requirements.

## Implementation slices and verification

| Slice | Implementation responsibility | Bounded verification |
| --- | --- | --- |
| A1 precise numerical contract | `tools/quantize/` references and native Q8 RHS/MMQ probes; document actual pinned packers | CPU reference vectors for finite/zero/tail/rounding/extreme inputs; independent packed-byte and dot comparisons |
| A2 backend capability and controls | Private runtime/backend policy and targeted GGML integration patch, pinned/license-preserving | Representative SAM vision shapes; explicit unsupported outcomes; direct execution-path evidence on later matching hardware |
| A3 scoped model execution | Vision linear graph integration through backend policy; policy-aware reuse/invalidation | Tiny graphs plus one/two image comparisons; all requested eligible nodes accounted for, strict placement |
| A4 unified configuration / public API | Versioned activation mode, `BackendOptions` value types when stable, conversion/runtime reports and docs | Four-axis conflict tests, model/backend rejection before execution, request/resolution/trace separation, two-TU/public-header checks |
| A5 additional modes | Other Q weight formats, per-layer exceptions, independent F16 graph activations, then FP8 if a useful native path exists | Separate formats/contracts, capability evidence and cost measurements per mode; no inherited W8A8/FP8 claim |

Implement A1-A4 as reviewable slices; A2/A3 hardware verification requires a
later explicitly authorized bounded run. Keep `backend-selected` as the current
default. No incomplete mode should become selectable merely because its JSON
name or floating-point simulation exists.

## Measurement and acceptance meaning

- Codec layout, finite outputs, kernel/path identity, backend placement and
  cache invalidation are implementation correctness checks. They can fail a run.
- Compare selected application images/prompts with original SAM 3 or native F32
  pseudo-reference outputs. Report missing/added objects, mask agreement and
  diagnostic score/box changes. User limits remain optional advice; COCO is
  supplementary, not a universal hard quality floor.
- Time packing, matrix execution and full inference separately; report packing
  overhead, scratch/retained memory, model size and load/prepack costs. Isolated
  GEMM speed does not prove an end-to-end gain. Valid regressions remain reports.
- Start later device work with a single operator and a small image set; stop on
  GPU loss/unsupported placement. Do not resume the interrupted all-preset
  campaign or introduce a large GPU sweep through this plan.

## Results

Plan written. No native activation mode, forced Q8 execution path, public API
extension or new hardware qualification has been added in this turn.
