# Visual encoding profile and optimization

Created: 2026-10-03T13:45:34.515638+08:00.
Baseline: main56a4cde plus current P1/P2 WIP. No commit/push requested.
Status: complete. All changes remain uncommitted; no push performed.

## Scope and approach

Measure graph construction/allocation/upload/compute/download/BF16 rounding and
geometry in the existing fused visual encoder. Separately measure original ViT
and neck graphs in an isolated diagnostic to locate compute cost; splitting is
not a production change. Use unchanged weights, input preprocessing and F32
requests. Compare diagnostic outputs against the unchanged full encoder.
Choose a concrete optimization only after these measurements establish its
benefit; preserve candidate, precision, temporal and numerical gates.

## Steps and verification

1. Compile a standalone private probe from current headers, linked to immutable
   Metal libraries. Keep production graphs unchanged while measuring phases.
2. Run matched input/weights serially under recorded power conditions. Verify
   canonical output values and stage placement. Do not compare this battery/
   short diagnostic to old AC steady-state numbers as a speedup.
3. Implement one measured simplification/optimization if justified, leaving a
   regression that checks the changed behavior. Revalidate affected outputs
   using matching baselines; do not repeat the entire CPU suite unnecessarily.
4. Update plan, changelog and synchronized bilingual summaries as appropriate.

## Results

Current CPU/Metal Release builds, independent headers and CTest pass (11/11
each). The current no-BLAS/no-Metal/non-native quick build also compiles all
targets and independent headers; CTest11/11 passes in12.01s. Its receipt is
`build/p1-p2/20261003/reaccept/no-blas-checks.json`. Documentation links/table
parity and whitespace checks pass. Remote CI has not run.

The new supported numerical matrix and subsequent image, short/long-session
and 64-frame performance runs are serially queued. The complete F32/hybrid
CPU/Metal video matrix passes: four five-case/216-frame cells,864 frames, with
original gates and artifact/output guards. Metal has zero CPU/BLAS graph nodes
in every case. The matrix controller exits0. Subsequent image, session and
performance checks remain pending; video acceptance is not performance evidence.

CPU/Metal image regressions pass all70 cases (schema1 F32/F16 and schema2
F32/F16/hybrid, seven each/backend). Real short sessions pass all six profiles. The
isolated current CPU long-session driver compiles and passes128 interleaved
pushes. Its positive session matches exact outputs/history with the new accepted
CPU entry64; negative outputs/state stay empty, lifecycle/owned-output checks
pass, and weight/compute high-water remains constant after warmup. Metal's
isolated long-session driver also passes the same128-push comparison and
lifecycle/bounds checks with zero CPU graph nodes. All correctness/session
checks are complete. Four fresh64/16/48 video performance cells and four
one-warmup/five-repeat image cells also pass, with AC endpoints and no system
sleep inside measured intervals. CPU video medians are9.155/13.887s for1/4
objects (historical51.102/81.644s); Metal7.550/13.166s. These are same-protocol
historical comparisons, not interleaved A/B runs. Metal four-object tracking
(7.277s) is the remaining largest stage in that workload.

The final closure verifies373 source/model/binary/library/input identities and
104 output trees. New durable private archive:14664 files/43333663701 logical
bytes, all APFS clones; creation and independent inventory/hash verification
pass. The original archive remains the shared model/reference parent. Current
bilingual summaries/details, baseline index, source-backed precision boundaries
and preserved failure scopes are updated. Documentation links/table parity and
whitespace pass. Current CPU/Metal Release CTest11/11 each and no-BLAS quick
CTest11/11 are retained; production Python tools are unchanged since tools24/24.
Remote quick CI remains unrun.

Four image performance cells run after the video queue succeeds, using the
original one-warmup/five-full-image protocol on truck.jpg. Their warmed times
exclude loading, decoding, file writes and repeated-result-cache calls. Each
cell binds the accepted seven-case image receipt, current CLI/source/library/
model/input hashes, exact accepted truck detections/masks and power/sleep logs.
This also refreshes the image table for the changed shared ViT implementation.
Preflight source readback corrected the private image-repeat counter from one
text encode (video semantics) to six (each replaced image encodes text). The
original waiter was stopped before any inference and remains an incomplete
record; `image_performance_v2.py` replaces it with no runtime/source change.

The v2 image benchmark stopped after its first Metal/F16 measurement because
its extra exact-output assertion compared stb-decoded JPEG with the official
Pillow-exported PPM validation input. Native decode readback proves163934 of
6480000 channel values differ (max3); same-JPEG single versus repeated inference
has exactly equal detections and mask bytes. The v2 failed receipt/raw output
is retained. V3 keeps exact comparison but first runs a guarded, identical-JPEG
single-inference bridge for each profile, then uses fresh measurement paths.
The bridge is not a new original-Meta oracle. Official numerical gates and
runtime sources are unchanged; no threshold is relaxed.

### Measured candidate and implementation scope

Q/K RoPE batching preserves seven feature-file bits but does not improve matched
A/B/A encoder latency (+0.35%); Metal concurrency-disable also has no benefit
(+0.20%). Both remain private rejected experiments.

ViT channel projections currently treat spatial height/window batches as many
small GEMMs. Folding shared-weight positions into one column dimension preserves
all seven Metal feature files bitwise and reduces the isolated encoder sample to
about5.15s. The selected CPU scheduler currently omits the registered BLAS ACCEL
device. With column folding and explicitly scheduled CPU BLAS, the same input
encoder falls from37.95s to6.56s, executing153 BLAS nodes. Post-BF16 feature L2
differences are at most0.000135; this is not full numerical acceptance.

Implement backend-neutral shared-channel projection and optional registry-based
BLAS in CPU driver/scheduler. Weights remain allocated on the selected CPU device;
BLAS is a priority host accelerator, counted as a subset of CPU work. Metal keeps
its existing selected/fallback list. Environments without BLAS retain native CPU.
GGML/BLAS thread requests are forwarded; Accelerate manages its own SGEMM threads,
so controlled experiments request VECLIB_MAXIMUM_THREADS=4 before process startup.
No global environment mutation or new public precision mode is introduced.

The graph/scheduling change requires new F32/hybrid CPU and Metal original-reference
checks on new binaries, targeted math/backend regressions and matched timing.
Old evidence remains archived; no failure is relabeled.
