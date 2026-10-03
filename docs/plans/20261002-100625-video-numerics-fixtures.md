# Resolve video numerical divergence and scenario fixtures

Created: 2026-10-02 10:06:25 Asia/Shanghai.
Baseline: `2ddee1b` plus the preserved local video/image implementation and receipts.
Status: observability, buffer ownership and scenario optimization verified;
legacy numerical stress case and complete matrix/performance acceptance pending.

## Scope

Continue the user's video validation work: diagnose the first entry FP16 memory
error, fix its proven cause, and make entry/hotstart fixtures exercise their
required original-model behaviors. Preserve existing weights, references and
unrelated changes. Numerical gates and the pinned Meta temporal policy stay fixed.
No commit or push is requested.

## Approach

1. Capture propagated decoder/mask and actual memory inputs before periodic
   correction, so the first divergent stage cannot be overwritten by seed dumps.
   Compare original FP32, exact GGUF mixed-F16 values executed in original modules
   (diagnostic only), and C++ on a short reproducible prefix. Determine whether
   the mismatch is arithmetic/layout/control flow or quantization/recurrence.
2. Apply only an evidence-backed correction in the shared execution path. Keep
   original-reference acceptance separate from the mixed-weight diagnostic.
   Add a runnable regression that distinguishes the wrong behavior.
3. Verify candidate entry and hotstart recipes with the original modules first.
   Record why the old recipes failed, freeze successful recipes and hashes, and
   only then compare C++. Preserve the old corpus and receipts.
4. Run targeted original stage/sequence checks, CPU/Metal behavior/header checks
   and Python tools. Broaden to the complete validated corpus when the short
   checks and required official behaviors pass.
5. Update README/model status/changelog and this record using demonstrated
   results. Do not infer full acceptance from mask IoU or diagnostic parity.

## Verification

Use isolated existing runtime `build/reference-runtime/venv/bin/python`, the
pinned original checkpoint/source and fresh `/private/tmp` outputs. Run heavy
inference sequentially to avoid the preceding batch's resource contention.
Retain provenance, selected memory/pointer/ID traces, exact input normalization,
F32 <=0.001 / F16 <=0.02 tensor gates and all existing mask/score/box bounds.

## Results

The first fresh 17-frame original export and FP16/Metal comparison completed.
New snapshots retain the selected propagated mask before correction and the
actual normalized 1152-square mask input. Frame 16 propagating-mask normalized
L2 is 0.0347775185; memory-mask L2 is 0.0033635996; memory-features L2 remains
0.0490653679. The corrected decoder snapshots still pass. Thus the existing
memory failure follows a propagation divergence already hidden by correction.
This observation does not yet identify its cause.

The mixed-weight original-module probe verifies all 1,464 GGUF tensor payload
hashes before execution and is explicitly supplementary/ineligible. Its first
two frames agree with C++ within 0.000064 memory-feature L2. At frame 16 its
propagated mask L2 against original weights is 0.0476128137 and memory-feature
L2 is 0.0466228016, so mixed-F16 weights in the original modules also reproduce
the failure. C++ versus mixed-module propagated-mask L2 remains 0.0220570434
(memory L2 0.0122572021); this additional recurrence discrepancy is not yet
fully explained. Do not report a pure weight-rounding cause or a fixed old
entry numerical stress case from this evidence.
Receipts: `build/video-validation/20261002-100625/stage-comparison.json` and
`/private/tmp/sam-video-20261002-100625/` (private raw outputs).

Metal Release build, 11/11 CTest, repeated/independent header and two-TU linkage
checks, and 15/15 isolated Python tests pass. The explicit head-16 F32 matmul now
uses the pinned `ggml_prec_set_acc` API; its three targeted math/header/behavior
checks pass after rebuilding.

The original body-only hotstart probe preserves the wheels and yields nonempty
unmatched ID 0 on frames 2-9, removing it at frame 9. Its proof is saved in
`body-occlusion-probe.json`. The hotstart recipe now uses the exact half-open
rectangle [64,256,1744,760) rather than an entirely blank frame. The previous
four-frame original separated/mirrored 800x533 entry probe demonstrated a
second ID with first-ID continuity; the revised frozen recipe inserts that
second tile on frame 16, whose full 64-frame behavior is still being verified.
Entry eligibility now requires an actual birth at/after frame 16, late-ID
visibility on the final frame and an ID visible throughout the sequence.
Both recipes preserve the original checkpoint, thresholds and 216-frame total.
The old generated corpus and its failed receipts remain intact. New generated
PNG/hash manifest: `/private/tmp/sam-video-20261002-100625/frames-v2/`.

The revised full 24-frame hotstart original export verifies its required
behavior: ID 0 retires at frame 9 and is removed from every delayed output.
FP16/Metal comparison passes all 24 frame/trace/output checks and selected stage
gates (maximum L2 0.0034760326, CPU graph nodes 0). This is a verified case,
not full-corpus or complete precision/backend-matrix acceptance. Receipts:
`hotstart-reference-proof.json`, `hotstart-reference-manifest.json` and
`hotstart-f16-metal-metrics.json` in this run's ignored receipt directory.
The revised entry full 64-frame original export completes and verifies its
required behavior: ID 1 is born exactly at frame 16, ID 0 is visible on every
frame, both IDs remain visible on frame 63 and all 64 outputs drain. Export
inference wall time is 423.149 seconds (diagnostic, not a controlled benchmark).
`entry-reference-manifest.json` and `entry-reference-proof.json` retain the
original evidence. FP16/Metal comparison is running. Both revised scenarios
now have full-length original-model behavior proof; subset manifests remain
ineligible for full-corpus acceptance.

The diagnostic memory mask now transfers its already-consumed input vector to
the debug snapshot after synchronous encoding, avoiding a 5,308,416-byte copy
per object/frame on both birth and propagation paths. Model math and state are
unchanged. The complete revised-case runs use the archived
`sam_video-before-memory-mask-move` executable; final-source short original
checks are required for the ownership change. Its final CPU/Metal builds pass.

The revised entry's 64 outputs complete, but rebuilding the live executable
before `validate_video.py` finished correctly triggered its immutable-batch
guard. The original rejection stays in `entry-f16-metal-metrics.json`.
Offline analysis verifies the archived executable against that initial binary
hash, rechecks all original provenance and every output hash before/after
comparison, and passes unchanged numerical/trace/output gates: maximum L2
0.0034531852, minimum mask IoU 0.9961796730, zero CPU graph fallback.
`entry-f16-metal-reanalysis.json` explicitly records `batch_guard_passed=false`
and `eligible_for_milestone=false`. It does not turn the rejected batch into an
accepted run. A new stable final-binary 64-frame validation is required and will
follow the final-source ownership smoke checks.

Final-source original two-frame motion checks pass FP32/CPU (maximum L2
0.0001704631) and FP16/Metal (zero CPU graph fallback). Fresh final CPU/Metal
CTest each passes 11/11; isolated Python tools pass 15/15. The final-binary
64-frame entry rerun is underway with no further executable rebuilds.

The stable final-binary 64-frame entry rerun passes the normal validator,
including its executable guard. Maximum tensor L2 is 0.0034531852; minimum
high-confidence output mask IoU is 0.9961796730. Every ID, birth/removal,
memory/pointer-order, score/box and emission check passes. CPU graph nodes are
0 on every frame, visual encodes are 64 and text encodes are 1. Retained records
peak at 53 for two objects (bound 54); queued frames peak at 14 between calls
and finish at 0. Final executable hash matches the report.
`final-entry-f16-metal-metrics.json` and `final-summary.json` retain this clean
run, while the earlier rejection/offline analysis remain separate.

Both revised full-length scenarios now pass their original behavior checks and
FP16/Metal numerical/state comparisons: entry 64 frames and hotstart 24 frames.
The memory-buffer optimization additionally passes final-source original
FP32/CPU and FP16/Metal short checks (latter maximum L2 0.0020983330).
CPU/Metal CTest each passes 11/11, independent/repeated headers and two-TU
linkage pass, isolated tools pass 15/15, and `git diff --check` is clean.
The live Metal executable was unchanged throughout the final batch.

These are complete case results, with subset references explicitly marked
ineligible for M2. The complete revised 216-frame export/matrix, controlled
performance gates and the preserved old FP16 entry stress discrepancy remain
separate follow-up work. No weights, tolerance or temporal-policy thresholds
were altered. Prior uncommitted work is retained; no commit or push was made.
