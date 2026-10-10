# Mixed weight quantization implementation plan

## Status and scope

M1–M3 implemented and CPU fixture validation completed on 2026-10-10; see the
[implementation/results](20261010-161049-module-mixed-weight-implementation.md).
The [unified configuration](../quantization-config.md) now supports schema-2
module-format policies in addition to schema-1 single-format allocations.
Exact tensor overrides (the first part of M5), conversion previews and CPU cost
diagnostics continue under the [next implementation/results plan](20261010-165611-quantization-preview-tensor-policy-costs.md).
Complete-model mixed quality/performance, M4 device evidence and mixed F16 remain
pending. Reference agreement and performance remain independent advisory reports.

Initial scope: module-level F32/Q8_0/Q6_K/Q5_K/Q4_K selections. F16 inside a
quantized mixed model and exact per-tensor overrides follow separately, after
loader/backend handling is explicit. Video memory, activation quantization,
automatic quality optimization and automatic fallback across devices are out
of the first slice. No large GPU validation is authorized by this plan.

## Baseline constraints before implementation

- `tools/quantize/weight_policy.py` owns versioned single-format allocations;
  `tools/convert/sam3_gguf.py` owns tensor eligibility, dtype and row-layout rules.
- `src/models/sam3/weights.hpp` validates schema 3/4 profiles and expected types
  before allocating weights. `src/models/sam3/model.hpp` repeats the model's
  tensor contract. Both currently derive a matrix type from one global profile.
- GGML carries individual tensor encodings, but that alone does not make an
  arbitrary combination a supported SAM model. CPU casts, backend capability
  checks and producer-reported arithmetic profiles also need consistent rules.
- Benchmark recipes and probes currently report one `weight_precision`, a
  profile and schema-4 module list. The stored tensor inventory is richer than
  that label and must remain authoritative.

## Design

1. Add configuration schema 2 only when the new conversion/loader path exists.
   A mixed weight policy declares a base Q format plus module-format overrides.
   Base precision applies to eligible matrices across the four modules; F32
   overrides retain a module's matrices. Protect the existing mandatory F32
   tensors regardless of module requests. Per-tensor attempts to bypass protection
   are outside this schema and rejected.
2. Introduce SAM image GGUF schema 5 with profile `image-mixed-linear-v1`.
   Record the base precision, canonical module-format allocation and a canonical
   policy hash. `general.file_type` follows the declared base Q format; schema
   5 metadata and actual tensor types describe deviations. Existing schema 3/4
   files keep their interpretation. Old loaders reject schema 5 rather than
   treating it as a legacy single-format file.
3. Resolve module overrides deterministically, then apply the existing exact
   tensor eligibility and fixed row fallback. A requested K format on vision
   MLP `lin2` retains its explicit Q8_0 exception; ordinary ineligible matrices
   retain F32 with a recorded reason. Unknown modules, unknown formats,
   contradictory policy metadata and unused exact overrides are errors.
4. Convert every tensor directly from the original checkpoint's floating-point
   values. Do not dequantize/requantize a previously quantized GGUF as a default
   conversion source. Reuse the pinned row quantizer and preserve original
   checkpoint, quantizer binary/library and format provenance.
5. The conversion manifest records requested type, resolved type, selector,
   fallback/protection reason, shape and byte count per tensor. The native
   loader independently resolves the same schema contract, checks the entire
   inventory before backend allocation, and reports `precision=mixed` plus the
   profile. Runtime dispatch follows each tensor and backend capability.
6. Reports preserve all four axes. Determine whether a model is quantized from
   its actual inventory/policy, not membership in a single-format string tuple.
   Keep requested allocation, actual stored types, runtime policy identifiers
   and unavailable kernel arithmetic separate.

The first extension is intentionally finite and module-based. It avoids a
general wildcard rule engine and its ordering/maintenance burden. A subsequent
exact-tensor override slice uses exact name > module > base precedence; every
override must match an eligible canonical tensor exactly once. Rules cannot
bypass mandatory F32 protection or silently alter row-block constraints.

## Implementation slices and ownership

| Slice | Files / responsibilities | Completion evidence |
| --- | --- | --- |
| M1 policy and format | `tools/quantize/weight_policy.py`, `quantization_config.py`, `tools/convert/sam3_gguf.py`, `convert_sam3.py`, `docs/gguf.md` and Chinese format guide | Deterministic mixed allocation; schema-5 metadata; small GGUF round trips; old files unchanged |
| M2 native loader and CPU | `src/models/sam3/weights.hpp`, `model.hpp`, private runtime/backend policies, model/runtime tests | Reject bad types/metadata before allocation; tiny mixed CPU matrix graphs; correct model information |
| M3 probes and reports | `support/image_io/`, precision probes, `tools/benchmark/precision_reporting.py`, independent benchmark tools | Wire fields and configuration identities agree; no fake single-format label; offline receipt checks |
| M4 bounded device evidence | CUDA/Metal capability checks and per-device probe fixtures | Correct kernels and placement on matching hardware; unsupported tensor/shape combinations fail explicitly |
| M5 exact overrides / mixed F16 | Versioned policy extension, loader promotion/cast paths and tests | Exact names, zero unused overrides, protected tensors intact; F16 handling checked per stored tensor |

Land each slice with its own implementation plan/results and changelog entry.
Do not advertise mixed-model inference until M1-M3 pass together. M4 is opt-in
later and does not imply that every backend/device is qualified.

## Verification and stopping conditions

- CPU-only fixtures cover every module, several mixed bit assignments, protected
  tensors, the `4736` row fallback, metadata bounds, duplicate/unknown selectors,
  wrong dtype/byte count and schema-3/4 compatibility. Compare packed bytes and
  decoded values against independent GGML/format references where available.
- Tiny matrix graphs verify dispatch/outputs and CPU casts; loading invalid
  allocations must not reach backend weight allocation. Include mixed F16 only
  after per-tensor CPU promotion and supported device operations are exercised.
- On later explicitly authorized hardware runs, start with one matrix shape and
  one or two application images. Use original-checkpoint/native-F32 outputs as
  pseudo-reference, report per-case differences, and leave optional limits to
  the application owner. COCO remains supplementary reference evidence.
- Measure latency, temporary/retained memory, model size and initialization costs
  independently. A valid slowdown or quality tradeoff remains a valid report;
  a failed codec/shape/identity/device check remains an implementation error.
- Stop device work on GPU loss/placement errors. Preserve completed receipts and
  original checkpoints. No large corpus, all-preset sweep or model-copying task
  is part of this plan's execution.

## Results

Plan written; no mixed-format GGUF schema, loader or new quantization kernel has
been implemented or hardware-validated in this turn.
