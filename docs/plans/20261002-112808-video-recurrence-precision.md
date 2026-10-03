# Diagnose video recurrence precision

Created: 2026-10-02 11:28:08 Asia/Shanghai.
Baseline: `2ddee1b` plus preserved local implementation and validation receipts.
Status: root cause and FP32 alternative verified; diagnostic optimization
complete. Existing mixed-F16 stress limit and full M2 acceptance remain open.
No commit or push requested.

## Scope and approach

Continue the old entry FP16 stress discrepancy without changing its inputs,
original checkpoint, numerical gates or Meta temporal policy. Keep the accepted
revised entry/hotstart cases and old receipts separate.

1. Record every propagated candidate IoU/selection before correction. Add an
   explicit CLI option for every-frame tensor dumps; defaults stay sparse.
2. Run the same 17-frame old prefix through original FP32 modules, exact GGUF
   mixed-F16 values in those modules (supplementary only), and C++/Metal. Capture
   every object stage to identify the first selection/state/numerical divergence.
3. Replay the first divergent decoder or memory stage in the original modules
   using the exact C++ inputs. Distinguish local arithmetic from accumulated
   storage/quantization error. Prove a cause before changing model execution or
   storage policy; test any proposed precision change in the original modules
   before converting new weights. Preserve existing model files.
4. Apply the smallest justified correction and a behavior/numerical regression.
   Verify the original stress prefix and affected revised cases, CPU/Metal
   headers/CTest and isolated tools. Keep executables immutable during batches.
5. Update support claims and changelog from actual results. Full corpus/matrix
   and controlled performance acceptance remain separate until completed.

## Verification

Reuse the pinned original source and isolated CPU adaptation/environment.
Run heavy inference sequentially; use fresh `/private/tmp` outputs and ignored
`build/video-validation/20261002-112808/` receipts. Keep F32 L2 <=0.001,
F16 <=0.02 and existing output/trace/state/backend gates. Mixed-weight and old
recipe stress diagnostics never count as complete M2 acceptance.

## Results

The original and exact-GGUF mixed-module 17-frame runs complete. C++/Metal
completes with every-frame snapshots and no algorithm changes. The first
original-vs-mixed candidate divergence is frame 4 (3 -> 1): conditioned-feature
L2 is 0.0012598 and all-candidate-mask L2 is 0.0010865, but selected-mask L2 jumps
to 0.1862994, pointer L2 to 0.4290453 and memory L2 to 0.0730423. Subsequent
conditioned features exceed 0.09. The C++-vs-mixed first index divergence is
frame 12 (3 vs 1), after smaller closed-loop input differences accumulate.

Exact C++ decoder/memory inputs replayed in the mixed original modules choose
the same candidates at frames 4/12. Decoder-mask L2 is <=0.00000120, pointer
L2 <=0.00000600 and memory L2 <=0.00000143. Thus the large observed discrepancy
is not a local SAM/memory encoder layout or arithmetic fault. Restoring only
the original FP32 IoU head or SAM decoder on those same inputs does not restore
the original choice. Replacing conditioned features with the original features,
while retaining mixed high-resolution neck inputs, restores choices 3/1.
The conditioned-input perturbation crossing the hard IoU argmax and switching
both mask and pointer explains the later recurrence amplification. This does
not establish a complete precision/storage-policy remedy.

The new trajectory formatter now casts Meta NumPy IDs to ordinary integers.
A real NumPy-ID/tied-candidate/batch-boundary regression fails on an isolated
copy without that cast and passes with it. Python tools pass 16/16; Metal CTest
passes 11/11. Initial incomplete probe output is preserved under
`failed-json-int64-original-f32-stress`. Receipts include
`stress-comparison.json` and `exact-input-replay.json` in this run's ignored
directory. Baseline models, inputs, gates and temporal policy remain unchanged.

Three supplementary full-prefix policy screens stop at frame 4 with a wrong
choice and pointer L2 about 0.429: memory-attention FP32, all tracker FP32, and
all tracker plus tracker-neck FP32. These tests retain the shared visual trunk
in mixed-F16 form. No candidate policy is published or installed. The existing
original FP32 GGUF/Metal model is now being checked on every old-prefix frame,
instead of introducing a new ineffective storage configuration. This cannot
be called a solved FP16 precision limit. `precision-screening.json` records all
three rejected screens.

The existing FP32/Metal GGUF passes every object/tracker-neck stage on all 17
old-prefix frames against the original FP32 modules, including exact candidate
and pointer-token choices. Maximum normalized L2 is 0.0001958593, below the
unchanged 0.001 FP32 gate. Receipt: `f32-every-frame-stress.json`. This is an
original-weight stress diagnostic, not full-corpus/matrix acceptance.

Root cause: small conditioned-feature perturbations from the mixed-F16 model
and closed-loop floating/storage differences cross the pinned hard IoU argmax
in `TrackerExecution::decode` / Meta `_forward_sam_heads`, switching both the
chosen mask and projected pointer; those discrete choices amplify later memory
and conditioned-feature differences. Frame 4 is the original-vs-mixed first
switch, frame 12 is the C++-vs-mixed first switch. Sparse corrected snapshots
previously hid the first transition. Exact-input decoder/memory replay confirms
the local operators agree; replacing the conditioned input restores original
choices. The tested local precision policies are rejected, not published.

Use the existing FP32 model when original candidate identity and strict
intermediate agreement are required for this pressure input. Do not advertise
the mixed-F16 stress case as fixed or relax its gate. New per-frame candidate
traces and opt-in full tensor dumps make future precision cliffs observable.
The NumPy-ID trace serialization fix has a red/green regression; sibling trace
ID writers already cast IDs (`export` results/groups/births) or emit native C++
integers. Final CPU and Metal Release/headers/two-TU/CTest each pass 11/11,
isolated Python tests pass 16/16. No checkpoints, tokens, private media, commits
or pushes are added to Git.

## Independent GPT-6 Astra review

At the user's request, GPT-6 Astra with max reasoning independently checked
the two divergence chains. Diagnostic scripts and receipts are preserved in
`build/video-validation/20261002-112808/astra-independent/`; `report.md`
records the exclusions, interventions, reproduction commands and fingerprints.
No production source, executable, model or acceptance threshold was changed.

Exact saved-input replay now also covers memory-attention construction,
including selected spatial memories, BF16 storage, pointer temporal encoding
and positional encoding. Original/mixed self-replays are bitwise identical;
C++ input replay at frames 1/3/4/6/12 has L2 0.00000080 to 0.00000162. Exact
frame-3 decoder/memory replay likewise agrees (maximum stage L2 0.00000784,
IoUs bitwise equal, both choose token 2). All 17 preprocessed frames are
bitwise equal to the pinned Meta loader. Source and weight hashes are verified.

Component cleanup changes zero pixels in either run on sampled frames
1/2/3/4/5/6/12. Exact Meta cleanup/resize/sigmoid replay matches the C++ memory
input with L2 <=0.0000000164. BF16 memory storage does enlarge early numerical
differences: frame-1 memory L2 rises from 0.0000632462 before rounding to
0.0003499551 after storage. At frame 12, changing only historical memory and
pointers contributes conditioned L2 0.00538056, while changing only current
FPN2 contributes 0.00007041; total observed L2 is 0.00538183. These independent
interventions are not an additive error decomposition.

Two additional supplementary screens both fail at frame 4: original FP32
shared trunk with all other weights mixed, and exact original transported
tracker FPN2 injected into the mixed model. The trunk-only policy adds
889,192,448 weight bytes without restoring the original candidate. Thus neither
the trunk alone nor the low-resolution visual input alone is a verified repair
boundary. The tested local policies do not prove all mixed-precision policies
impossible. Full FP32 remains the existing verified choice for this stress
prefix; no FP16 fix or complete M2 acceptance is claimed.

The small frame-0 C++ versus same-weight visual difference has only been
observed after BF16 transport (L2 0.000136528); unrounded visual intermediates
have not localized it to a particular operator. The independent evidence
supports BF16 state/recurrence amplification followed by the hard candidate
switch, while leaving that initial arithmetic source and other precision
combinations open. Complete corpus/matrix/performance acceptance is unchanged.

Further visual diagnosis and the conditional repair plan are recorded in
`20261002-131147-video-visual-precision-repair.md`. A faithful visual probe
matches all six production outputs bitwise and explains the BF16 boundary
differences; same-input LayerNorm/FP64 and block-0 checks do not establish an
operator defect. The joint original-FP32 visual/tracker storage candidate is
specified for a 5-frame then 17-frame screen, but has not been run or converted.
