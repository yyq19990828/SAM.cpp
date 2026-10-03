# Localize video visual precision and prepare a repair

Created: 2026-10-02 13:11:47 Asia/Shanghai.
Baseline: preserved local video implementation and
`build/video-validation/20261002-112808/astra-independent/report.md`.
Status: visual diagnosis complete; conditional repair plan ready. No operator
defect or passing FP16 repair was established. This delivery covers diagnosis
and planning; implementation remains for the execution phase. No commit or
push requested.

## Scope

Locate the frame-0 C++/Metal versus exact mixed-weight original-module visual
difference before BF16 transport. Reuse the old fixed 17-frame stress input,
original checkpoint, GGUF values and isolated original runtime. Keep this
same-weight implementation question separate from original FP32 versus mixed
weight-rounding acceptance. Preserve all existing WIP and passing revised
entry/hotstart evidence.

## Hypotheses and discriminating experiments

1. Capture unrounded patch/prefix, selected ViT blocks and tracker neck output
   with the existing graph builders in an independent ignored C++ probe. Use
   the exact saved normalized frame-0 input and unchanged GGUF. Original-module
   hooks capture corresponding NCHW/NHWC values with all 1,464 mixed payload
   identities verified. Validate the probe's final BF16 output against the
   existing frame-0 dump before interpreting internal differences.
2. Find the first error increase. Replay only that operator/block on identical
   input values in both engines, and compare accumulation, attention, activation
   and storage policies. Use a small independent diagnostic build linked to
   the pinned existing GGML libraries; do not rebuild shared verification
   executables or alter shared source.
3. Check every caller of a demonstrated problematic helper or backend path.
   A numerical difference alone is not a bug: distinguish valid reduction-order
   differences from extra narrowing, a mismatched formula or a wrong layout.
4. Prepare the smallest evidence-backed repair plan. A justified diagnostic
   counterfactual may be tested in isolated sources only. Do not relax numerical
   gates, alter argmax tie/epsilon behavior, remove Meta BF16 storage, change
   temporal policy, or introduce automatic fallback.

## Verification and final deliverable

- Run heavy inference sequentially. Record input, GGUF, source, probe and linked
  binary fingerprints. Keep all experimental models and outputs distinct from
  original-weight acceptance.
- Reuse existing exact decoder/memory/attention and preprocessing exclusions;
  do not repeat them unless a new change affects them.
- The final plan must identify the proven cause and remaining uncertainty,
  exact code/storage locations, affected callers, smallest regression, and
  why the proposed repair addresses the observed cause.
- Verification order during repair execution: exact-operator regression;
  same-weight frame-0/tracker recurrence; original-weight old 17-frame pressure
  input; revised entry 64 and hotstart 24 frames; affected image/video CPU/Metal
  checks. Keep unchanged F32 L2 <=0.001 / F16 <=0.02, candidate/state/ID/output
  gates, immutable-batch guard, and separate complete 216-frame matrix and
  performance acceptance.
- Record risks and explicit exit conditions. If no safe local repair is proved,
  retain the unsolved FP16 boundary and say what remains necessary; do not claim
  a general mixed-precision impossibility from finite rejected candidates.

## Verified visual results

The independent probe links the existing pinned GGML dylibs and reuses the
current graph builders. No production source, shared executable, library,
checkpoint or formal GGUF was changed. Raw outputs are under
`/private/tmp/sam-video-20261002-131147/`; scripts and receipts are under
`build/video-validation/20261002-131147/`.

### Observation fidelity

The initial `visual_probe.cpp` marks patch/prefix/all 32 block results as graph
outputs. That changes Metal fusion and therefore is an intervention: its final
tracker FPN2 has 3,464 BF16 values different from the old C++ snapshot. Its
intermediate comparisons identify candidate locations only; they are not
accepted as exact captures of the old production execution.

`visual_probe_faithful.cpp` instead preserves the production graph size and
detector-then-tracker output order/flags. All six final detector/tracker outputs
are **bitwise equal** to the old frame-0 C++ snapshot after the existing tracker
BF16 cast. `faithful-visual-comparison.json` records this 6/6 control. Use this
probe for final visual error claims; require this control for future dumps.

### First arithmetic difference and precision boundary

Patch convolution and positional addition in the instrumented comparison are
bitwise equal to the exact-mixed-weight Meta modules. The first nonzero
difference is the prefix LayerNorm. A separate same-input `norm_probe.cpp`
replays that operation directly, with the same F32 parameters and epsilon
1e-5. It reproduces the instrumented prefix bitwise and is checked against
FP64 centered population variance:

| Comparison | Normalized L2 | Maximum absolute error |
|---|---:|---:|
| Metal LayerNorm vs Meta CPU LayerNorm | 6.61981e-8 | 3.81470e-6 |
| Metal LayerNorm vs independent FP64 formula | 6.17595e-8 | 3.90336e-6 |
| Meta CPU LayerNorm vs independent FP64 formula | 6.80301e-8 | 3.96491e-6 |
| Block 0, Metal vs Meta on the exact Meta prefix | 7.34104e-7 | 6.86646e-5 |

The pinned Metal normalization kernel computes mean and centered variance with
F32 SIMD reductions and may fuse affine multiply/add. These are ordinary F32
rounding/reduction/FMA differences at the tested boundary; Metal is slightly
closer to the FP64 formula here. There is no evidence to justify changing the
LayerNorm formula, epsilon, or shared normalization helper to imitate a CPU
implementation bitwise. The experiment does not separate the last-ULP shares
of reduction order and compiler FMA, and need not do so to reject a large
precision/formula defect.

The actual Metal logs select `kernel_mul_mm_f16_f32_prec_f32` for ViT dense
operations and `kernel_flash_attn_ext_f32_dk64_dv64_prec_f32` for attention.
The pinned patch stages these operands in F32, not half. Faithful execution
uses zero CPU graph nodes. An extra F16 activation-staging defect was not found
on the observed visual path; existing dense/tail precision tests already cover
the small-matrix contract and were not redundantly rerun.

With the faithful full visual graph, unrounded tracker FPN2 differs from the
same-weight Meta output by **1.14645e-5** L2. The unchanged BF16 cast raises that
to **1.36528e-4**. Exactly **5,989 / 1,327,104** values differ; all 5,989 pairs of
unrounded values straddle the midpoint between their resulting BF16 values.
The rounding itself reproduces the old C++ snapshot exactly. The instrumented
block series shows gradual error growth rather than a large isolated jump,
but its fusion caveat remains applicable.

Evidence: `faithful-visual-comparison.json`, `norm-fp64-comparison.json`,
`block0-exact-comparison.json`, `visual-comparison.json` (instrumented only),
and `faithful-metal.log`, `norm-metal.log`, `block0-metal.log`.

### Causal conclusion and remaining uncertainty

The same-weight frame-0 difference is now explained at a concrete precision
boundary: small valid F32 arithmetic differences are enlarged by BF16 feature
rounding, then by the already-verified tracker recurrence and candidate
selection. Earlier exact-input replays cover memory attention, decoder, memory
encoder and mask-to-memory construction. A large local implementation defect
is not evidenced at the tested stages. This is not a proof that all inputs,
operators or backend configurations are free of bugs.

This finding **does not fix or explain away original-FP32 versus mixed-weight
acceptance**: the exact mixed weights already change frame-4 selection inside
Meta modules, independently of C++. The C++ versus same-weight first switch at
frame 12 is a distinct chain. Repairing an implementation operator, if a future
counterexample justifies it, must pass the same-weight chain first and cannot
be claimed to eliminate the original-vs-quantized frame-4 cliff automatically.

## Minimal actionable repair plan

### 1. Do not change arithmetic without a failing contract

No production operator patch is presently justified. Preserve shared
`ops.hpp::sam3_layer_norm`, the direct prefix normalization in `vision.hpp`,
the pinned Metal precision patch, BF16 transport/memory, strict argmax, and Meta
temporal policy. Do not add an epsilon tie-break, auto-fallback, or a new
normalization implementation merely to chase this trajectory.

Caller audit: shared LayerNorm is used by vision blocks, text, geometry prompt,
fusion, detector, image mask and tracker mask decoders, and memory attention.
The prefix directly calls `ggml_norm` with the same 1e-5 epsilon; LayerNorm2d
uses its separately required 1e-6 epsilon. Changing the common helper would
affect all these paths, while the FP64 check supplies no defect to repair.

### 2. Screen one joint precision candidate before changing storage support

Restore **original FP32 values** for the shared visual trunk, every tracker
neck scale and the entire tracker together. Leave detector-only and text
weights under the current mixed policy. This is the smallest next *joint
boundary* supported by the available experiments, not a proven minimum-byte
or successful solution: trunk-only and tracker-plus-neck-only failed; each
left part of the visual-to-memory recurrence quantized. Exact original FPN2
injection also failed while tracker/high-resolution paths remained mixed.
The proposed union addresses those remaining paths simultaneously.

`candidate-precision.json` records the exact 236 canonical tensor names and
source names that change from the existing F16 payload to original F32 values:

| Source group | F16 tensors restored | Extra tensor bytes |
|---|---:|---:|
| `detector.backbone.vision_backbone.trunk.` | 128 | 889,192,448 |
| `detector.backbone.vision_backbone.sam2_convs.` | 11 | 15,597,568 |
| `tracker.` | 97 | 22,297,352 |
| Total | 236 | **927,087,368** |

This is about 884 MiB of additional weight payload, before metadata/alignment.
Generate the list with `prepare_candidate_manifest.py`; it verifies the count
and byte total. Existing F32 tensors remain unchanged. Promoting already
rounded F16 values to F32 **cannot** recover the original values and is not
this candidate.

An ignored, executable diagnostic entry point is prepared but **not run**:
`screen_candidate.py`. It is derived from the existing policy screen, checks
checkpoint/manifest/input fingerprints, initializes all original 17 frames
even for the 5-frame prefix (preserving pointer temporal normalization), and
stops on the first wrong candidate or unchanged F16 L2 gate failure. Only
syntax compilation has been checked; no candidate success is claimed.

Execution-stage commands, run sequentially from the repository root:

```sh
rtk proxy build/reference-runtime/venv/bin/python build/video-validation/20261002-131147/prepare_candidate_manifest.py
rtk proxy build/reference-runtime/venv/bin/python build/video-validation/20261002-131147/screen_candidate.py --frames 5 --output build/video-validation/20261002-131147/candidate-prefix5.json
```

Continue only if that receipt has `checked_prefix_passed=true`, exact candidate
identity on every propagated frame, and all recorded original-reference gates
pass. Otherwise reject the candidate, retain the F16 limitation and stop this
storage proposal; do not convert a new model or broaden to arbitrary policies.
If the 5-frame gate passes, run:

```sh
rtk proxy build/reference-runtime/venv/bin/python build/video-validation/20261002-131147/screen_candidate.py --frames 17 --output build/video-validation/20261002-131147/candidate-prefix17.json
```

The in-memory screen is a supplementary filter, not complete acceptance: it
compares saved object stages, tracker FPN2 and candidate identities. Remaining
detector/text quantization can still perturb birth/correction inputs, especially
at frame 16. Failure stops the candidate; success permits the following work.

### 3. Only after a passing 17-frame screen, add one explicit hybrid profile

Implementation remains for the execution phase. Use a distinct temporary
GGUF filename and an explicit, versioned storage profile, described
as **original-FP32 visual/tracker plus mixed detector/text**, never as plain
FP16 or full FP32. Preserve the default F16/F32 profiles and the temporal
`meta-sam3-temporal-v1` profile; storage and temporal metadata are distinct.

Required code locations and order:

1. `tools/sam3_gguf.py` (`storage_dtype`, metadata writing/validation, tensor
   inspection) and `tools/convert_sam3.py`: implement exactly the declared
   candidate closure from original checkpoint values. Do not globally broaden
   `KEEP_F32`. Record profile, source hashes, selected tensor identities and
   actual byte accounting in provenance.
2. `include/sam/internal/models/sam3/weights.hpp`: extend the closed, validated
   type policy for that explicitly declared video storage profile. Unknown
   profiles, undeclared hybrid types, mismatched tensor types/shapes and invalid
   bounds must still fail before allocating weights.
3. `include/sam/internal/models/sam3/model.hpp`: keep its second type check
   consistent with the reader and report the real storage profile in model
   information. Reuse `tensors.hpp`'s validated source-type mapping and existing
   backend policies; no new graph, generic precision factory or blanket CPU/
   Metal cast is needed. Audit allocation/accounting for the larger payload.
4. `tools/validate_video.py` and the shared artifact validators: recognize only
   the explicit profile, preserve all original-reference/state/output gates,
   and keep same-weight diagnostics ineligible for milestone acceptance.
   Apply the unchanged F16 gate to this mixed artifact; do not label it an
   FP32 acceptance result. F32 controls retain the unchanged 0.001 gate.

### 4. Smallest regressions and validation sequence

Add one meaningful profile conversion/validation regression in the existing
tools tests: restored groups retain original non-F16-representable values;
unrestored groups retain the old exact payloads; inconsistent declarations are
rejected. Exercise the C++ reader's profile/type rejection path before weight
allocation using the existing fixture pattern. Preserve independent headers
and two-TU linkage. Reuse the existing BF16 halfway and Metal dense/tail tests;
do not duplicate them or assert internal implementation structure.

Then validate in this order, stopping on failure:

1. Original-checkpoint-to-profile payload identity, schema rejection and source
   bounds checks. Build isolated CPU/Metal binaries once and hash them.
2. Same-weight Meta replay vs C++ frame 0: exact input, faithful unrounded neck
   capture with the 6/6 output-control check, BF16 boundary counts, and actual
   backend dispatch. Run the unchanged 17-frame same-weight recurrence to
   expose any new implementation discrepancy before original acceptance.
3. Original-weight old 17-frame stress comparison on CPU and Metal, with
   every-frame pre-correction stages and exact candidate/pointer/memory/ID
   traces. Preserve normalized L2 <=0.02 for the candidate, original output
   bounds, and the full-FP32 control's <=0.001 bound. A same-weight pass cannot
   substitute for this original-weight check.
4. Re-run revised entry **64 frames** and hotstart **24 frames**, including late
   birth/ID continuity, hotstart retirement/delayed drain, stage/output/state
   bounds, zero unexpected fallback and immutable-batch validation. Use fresh
   output directories; retain previously accepted runs and rejected attempts.
5. Run affected image regressions because the shared trunk is used by image
   inference too, existing CPU/Metal CTest/header/two-TU checks, isolated
   `tools/test_tools.py`, and `git diff --check`.
6. Only then evaluate the complete **216-frame corpus**, advertised backend/
   precision cells and controlled latency/peak-memory measurements. This plan's
   diagnostics and short screens do not complete those milestones.

## Risks and acceptance exit conditions

- Ordinary F32 reduction/FMA differences remain across engines; preserved
  BF16 and hard selection can amplify them even with a less-quantized model.
  The joint candidate can therefore fail and is not promised to work.
- The larger payload increases resident/peak memory. Measure actual model load,
  image/tracker latency and peak memory; do not infer performance from storage.
- Detector/text remain mixed and can change discovery/correction masks. A
  stage-only short screen cannot establish their full lifecycle equivalence.
- Instrumentation can change fusion. Results without final-output fidelity
  checks are diagnostic interventions, not faithful production captures.

Exit as a successful repair only after the explicit artifact passes original
old-prefix and revised-case numerical/state/output checks, required CPU/Metal
regressions and immutable-batch guards, with truthful precision/profile claims.
Any failed original gate leaves the existing F16 stress limitation in place;
do not relax thresholds or mark support. The existing full-FP32 17-frame pass
remains a verified separate operating choice, not an automatic fallback or a
claim that the mixed-F16 issue was repaired.

No production implementation, candidate inference, candidate conversion,
default-weight change, commit or push was performed in this planning task.
