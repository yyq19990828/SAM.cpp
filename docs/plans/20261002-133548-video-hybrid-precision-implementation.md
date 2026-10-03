# Implement the screened video precision repair

Created: 2026-10-02 13:35:48 Asia/Shanghai.
Baseline: `2ddee1b` plus preserved video implementation and diagnostic WIP.
Status: implementation and the stated hybrid validation are complete; the
default-profile M2 matrix and long performance protocol remain open. No commit
or push requested.

## Scope and approach

Execute `20261002-131147-video-visual-precision-repair.md`. First screen the
declared union of original-FP32 trunk, tracker neck and tracker values, retaining
the current mixed detector/text values. Preserve original models, reference
inputs, BF16 state policy, strict argmax and numerical/behavior gates.

1. Run the prepared in-memory candidate on the first 5 frames of the unchanged
   17-frame sequence. Check the receipt and all recorded original-reference
   tensor/selection gates, not only process exit status.
2. If it passes, run all 17 frames. A failed screen rejects this candidate and
   stops its profile implementation; record the failure without converting or
   advertising a new model. Do not broaden to speculative precision policies.
3. Only after both screens pass, implement an explicit versioned hybrid storage
   profile in the existing converter, bounded reader/type validation, model
   reporting and artifact validators. Restore original checkpoint values;
   preserve default profiles and backend graphs. Add the smallest meaningful
   conversion/type-rejection regressions and provenance checks.
4. Convert to a distinct ignored artifact, verify every payload, build CPU/Metal
   once, and keep executables immutable during numerical batches. Check the
   same-weight and original-weight stress sequence, revised entry/hotstart,
   affected image paths, headers/two-TU/CTest and isolated tools.
5. If those pass, run the remaining full-corpus/matrix and controlled performance
   acceptance before extending support claims. Update README/GGUF/model support,
   changelog and this record only from completed results.

## Verification

Reuse the pinned clean Meta source, isolated CPU adaptation and locked Python
environment. Run heavy inference sequentially. Keep original-reference F32
L2 <=0.001 and mixed L2 <=0.02; retain output, candidate, state, ID, backend and
immutable-batch gates. Supplementary in-memory screens are not milestone
acceptance. Receipts go under ignored `build/video-validation/20261002-133548/`.
Run `git diff --check`; do not commit weights or private inputs.

## Results

The 5-frame and full 17-frame supplementary screens pass with every propagated
mask/pointer choice matching the original reference. All recorded stages have
zero normalized L2 and maximum absolute error (53 and 185 comparisons,
respectively), including the
frame-16 correction. Receipts: `candidate-prefix5.json` and
`candidate-prefix17.json` in this run's ignored receipt directory.

Implemented a video-only `hybrid` precision with the explicit storage profile
`visual-tracker-f32-v1`; conversion restores original FP32 visual/tracker values
and preserves existing detector/text payloads. The new conversion/declaration
regression fails on the old converter and passes on the implementation.
Original profiles, arithmetic, BF16 state and temporal policy remain unchanged.
The supplementary screens establish this candidate's recorded-stage equality;
complete original-reference acceptance is recorded separately below.

All 1,464 converted payloads pass independent comparison with the accepted
F32/F16 manifests: 236 match original F32 values and 1,228 retain their baseline
payloads. Artifact: `models/sam3-video-hybrid-v1.gguf`, 2,765,012,640 bytes,
SHA-256 `3975b4b1a10b962c6fad5022e2798baa6266aad7dbcc39210b459537208e1a05`.
Fresh CPU/Metal Release builds preserve the old builds and each pass 11/11
CTest checks, independent/repeated headers and two-TU linkage. The real hybrid
checkpoint's metadata/type-rejection checks pass before backend weight loading.

Every original stress frame passes on CPU/Metal with exact candidate identity,
selected memory/pointer order, state/output bounds and zero unexpected Metal
fallback. Maximum stage L2 is 0.0002494640 / 0.0001958593 respectively. These are
17-frame diagnostics, not complete corpus acceptance. The initial Metal
analysis incorrectly compared an extra C++ group birth descriptor absent from
the Meta probe; its rejected receipt is preserved, and the corrected analysis
compares the shared ID/spatial/pointer fields under identical binary/model/lib
hashes. Initial stress outputs were not sealed by a file snapshot before that
reanalysis; the later formal validators now capture and verify output snapshots.

Independent review confirms the exact Python/C++ dtype closure and exposes a
pre-existing validation provenance gap. Image/video validation now freezes and
rechecks model, sidecar, executable and shared libraries in the selected build,
and snapshots completed output files before numerical comparison. A real
replacement regression fails on both old entry points and passes with these
guards; altered output snapshots are rejected. Isolated tools pass 18/18.
The complete revised original reference contains all 216 frames, with all five
required behaviors verified and milestone eligibility recorded. Standard
hybrid validation passes every case on **both CPU and Metal**, using the
unchanged mixed gate (normalized L2 <=0.02):

| Case | Frames | CPU maximum stage L2 | Metal maximum stage L2 |
| --- | ---: | ---: | ---: |
| motion | 48 | 0.0013240080 | 0.0013239837 |
| entry | 64 | 0.0014297947 | 0.0014297849 |
| occlusion | 64 | 0.0013240080 | 0.0013239837 |
| hotstart-removal | 24 | 0.0021808850 | 0.0021060102 |
| negative | 16 | 0.0026496627 | 0.0025937675 |

Each backend passes 378 recorded-stage comparisons and 220 output-object
comparisons, together with every-frame candidate/pointer identity, fixed birth
ID mapping, memory order, lifecycle, output delay/drain and state bounds. Entry
retains both IDs after the frame-16 birth; hotstart removes ID 0 at frame 9 and
suppresses all object outputs. Minimum compared mask IoU is 0.9999764153 on CPU
and 0.9999685510 on Metal. Metal has zero CPU graph fallback; CPU has zero Metal
nodes. The final audit rechecks all 1,265 frozen original-reference files,
1,270 output files per backend, and 15 CPU / 18 Metal model, sidecar, binary and
library paths. Receipt: `full-video-acceptance-audit.json`; standard metrics are
under `/private/tmp/sam-video-20261002-133548/validation-{cpu,metal}-hybrid-full/`.

### Fresh same-weight visual and recurrence evidence

After the full video batch, a separate probe binary was compiled against the
unchanged Metal libraries. `run_hybrid_visual_v2.py` uses the existing faithful
graph builders and output flags; it does not mark intermediate blocks as graph
outputs. All **six final detector/tracker neck outputs are bitwise equal** to
the new formal hybrid motion/frame-0 dump (uint32 comparison, after the existing
tracker BF16 rounding). The exact input and all six formal outputs are bound to
the standard validator's output hashes. New C++ and Meta outputs are frozen
before comparison and rechecked afterward, alongside source, model, sidecar,
probe and linked-library hashes.

Meta's final model state verifies all 1,464 canonical payloads, including exact
F16-to-F32 promotion and the converter's existing complex-RoPE real-pair mapping.
The first probe attempt stopped before numerical comparison because an added
diagnostic assertion incorrectly required the legitimate complex64 RoPE cache
to be float32; that attempt is retained. The corrected v2 probe uses the same
`tensor_array` / `converted_array` conversion contract and distinct outputs.

| Tracker neck | Unrounded same-weight L2 | After BF16 L2 | Different BF16 values / total |
| --- | ---: | ---: | ---: |
| FPN0 | 6.0809430e-6 | 1.1461004e-4 | 77,752 / 21,233,664 |
| FPN1 | 1.1346079e-5 | 1.4769523e-4 | 33,881 / 5,308,416 |
| FPN2 | 1.0009515e-5 | 1.4182929e-4 | 6,722 / 1,327,104 |

Every differing pair straddles its BF16 midpoint. The probe executes 1,367
Metal nodes and zero CPU nodes. Its actual pipeline-selection log records
F32/F32 precise ViT matmul and F32 dk64 attention, with the remaining mixed
detector neck using precise F16/F32 matmul. This freshly reproduces the small
arithmetic-difference/BF16-boundary mechanism for the hybrid artifact; it does
not claim bitwise agreement with Meta or justify changing an operator.
Receipts: `hybrid-faithful-visual-comparison.json`,
`hybrid-visual-provenance.json` and `hybrid-visual-output-snapshots.json`.
Raw v2 captures are in `/private/tmp/sam-video-20261002-133548/same-weight-visual-v2/`.

The 17-frame same-weight recurrence check is **transitive reuse**, not a new raw
reference export. `audit_hybrid_same_weight.py` binds the candidate's source/type
closure to all actual hybrid payloads and the freshly verified Meta state.
Its 185 saved stages have zero numerical error against the same original oracle;
the existing CPU/Metal original-reference errors and exact candidate choices
therefore also apply at those stages against this exact-weight candidate.
The audit verifies shared oracle stage hashes and records reference-norm
roundoff only as a diagnostic. It does not assert signed-zero bit equality or
retroactively seal the initial stress outputs. Receipt:
`hybrid-same-weight-recurrence.json`.

### Image regression and controlled performance

The standard original seven-case image regression passes on **both backends**:
maximum stage L2 is 0.0042057835 CPU / 0.0041465730 Metal; minimum matched mask IoU
is 0.9999984283 for both. All 14 actual `results.json` files explicitly report
`precision=hybrid` and `storage_profile=visual-tracker-f32-v1`. The extra label
audit, existing original gates, backend-node checks and artifact/output hashes
all pass. Receipt: `image-acceptance-audit.json`; standard outputs are under
`/private/tmp/sam-video-20261002-133548/image-{cpu,metal}-hybrid-full/`.

One no-dump three-frame motion process per backend ran sequentially after all
acceptance inference, with threads=4 and the same hybrid artifact:

| Backend | Per-frame samples (s) | Frames 1–2 median (s) | Model load (s) | Peak RSS, time -l (bytes) |
| --- | --- | ---: | ---: | ---: |
| Metal | 6.173846, 6.529094, 6.704886 | 6.616990 | 0.920708 | 3,807,756,288 |
| CPU | 41.438141, 42.924738, 43.808755 | 43.366747 | 0.957970 | 5,170,331,648 |

Frame 0 initializes the sequence. Frames 1–2 have growing memory, and the final
sample includes all three delayed outputs' resize/drain. Declared frame_count=3
also retains its corresponding pointer temporal normalization. The runs have
one object, three retained records (1,993,728 bytes), zero pending output at end,
and no unexpected backend fallback. Metal weight/maximum compute allocation
is 2,763,224,992 / 1,245,268,288 bytes; CPU is 3,447,558,112 / 1,020,660,736 bytes.
Runtime peak RSS agrees with `/usr/bin/time -l`; accounting categories must not
be added together. Full wall times are 20.43 / 129.24 s, including load, decode
and file writing outside the per-frame samples.

Runs were Metal then CPU from 18:12:56–18:16:01 Asia/Shanghai, on AC at 100%
battery, without concurrent SAM/validation jobs. No thermal/performance warning
was recorded; ordinary desktop/macOS background services remained active and
clocks/cooling were not controlled. This is a bounded short-sequence sample,
not steady-state throughput, four-object load or complete M2 performance
acceptance. Full conditions and identities are in `benchmark-summary.json` and
`benchmark-{cpu,metal}-hybrid-3frames.json`; [BENCHMARK.md](../../BENCHMARK.md)
contains samples, accounting and standard CLI reproduction commands.

### Execution entry points and remaining boundaries

The ignored audit/probe entry points are `audit_full_video.py`,
`run_hybrid_visual_v2.py`, `audit_hybrid_same_weight.py`, `audit_images.py` and
`benchmark_hybrid_three_frames.py`, all in this run's receipt directory. The
standard acceptance commands used the immutable builds at
`/private/tmp/sam-video-20261002-133548/{metal,cpu}`:

```sh
rtk proxy build/reference-runtime/venv/bin/python tools/validate_video.py \
  --build-dir /private/tmp/sam-video-20261002-133548/metal \
  --model models/sam3-video-hybrid-v1.gguf \
  --reference /private/tmp/sam-video-20261002-133548/reference-full \
  --backend metal --threads 4 --output /private/tmp/sam-video-20261002-133548/validation-metal-hybrid-full
rtk proxy build/reference-runtime/venv/bin/python tools/validate_image.py \
  --build-dir /private/tmp/sam-video-20261002-133548/metal \
  --model models/sam3-video-hybrid-v1.gguf \
  --reference models/official/3c879f39826c281e95690f02c7821c4de09afae7/reference-unfused-fp32-corpus \
  --backend metal --threads 4 --output /private/tmp/sam-video-20261002-133548/image-metal-hybrid-full
```

The CPU commands substitute the CPU build/backend and new CPU output paths.
Existing paths are preserved and must not be reused for a fresh run. Final
README, model catalog, benchmark and changelog statements are limited to these
completed results. The default F16 stress failure remains explicit; full
F32/F16 M2 backend/corpus cells and the 64-frame one/four-object performance
protocol are **not completed by the hybrid result**. No threshold, strict argmax,
BF16 boundary, temporal policy or automatic fallback was changed. No additional
production edits or shared rebuilds were needed during the final acceptance
phase. Final whitespace verification is recorded in the run's `diff-check.json`.
