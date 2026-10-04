# SAM 3 pipeline execution and unified benchmarks

Created: 2026-10-04 04:02:40 Asia/Shanghai.

Status: complete for the three common execution optimizations and the selected
unified measurement scope. Frozen v8 passed fresh builds/tests, session and
tracker correctness, all 210 original-reference image cases, all 864 original-
reference video frames, 24 targeted tracker timing cells, focused RSS repeats,
the independent MPS comparison and all 30 unified image timing cells. Previous
candidate generations and intermediate pending states below are historical
evidence. D256 attention remains deferred; MPS was measured and not adopted.

## Authorization and baseline

The user requested committing the completed quantization work first, then
implementing the previously unimplemented pipeline optimization plan and
measuring timings only after implementation. Local commit
`ae1b7afb0a09b2d9a20fb1ecfb98b497ee573c1f` is the clean starting point.
No push was requested. The selected design is the
[tracker acceleration plan](20261003-191814-backend-portable-tracker-acceleration.md).
Its earlier planning-only authorization is historical and superseded by this
explicit implementation request.

## Delivery scope

Implement the plan's three common execution improvements: bounded graph and
workspace reuse, frame-feature residency/condition-to-decoder chaining, and
compatible-object propagation batches. Preserve public image/video task APIs,
owned results, existing precision/state rounding, tracker association/ID policy,
memory selection and 15-frame output timing. Seed/birth/reconditioning paths
remain serial as prescribed. No model/GGUF changes are needed.

Memory-attention fusion and MPS are independent backend candidates in the
original plan. Investigate their actual precision/layout applicability, then
perform any isolated comparisons during the final unified measurement phase.
They do not require speculative production routing to deliver the three common
optimizations. Keep unsupported or unmeasured candidates clearly separate.

## Implementation approach

- One session-owned shared GGML scheduler/workspace, with bounded cached
  graph/context metadata per tracker stage/branch. Before switching arenas,
  clear only graph-owned tensor allocation pointers, preserving model weights
  and externally owned frame tensors. Synchronize before rebinding or release.
- Shape keys include all dimensions, memory/pointer counts, seed/presence branch
  and diagnostic outputs. Reused graphs always receive fresh object/frame input.
  Shape switches and exceptions must not leave stale allocation pointers.
- Keep one resident tracker frame on the actual backend. Upload shared features
  and constants once at the appropriate frame/session boundary. Preserve
  BF16-rounded host values and required diagnostic downloads.
- Join conditioning and decoder through a graph tensor instead of downloading
  and re-uploading conditioned features. Required diagnostics remain owned host
  vectors after compute.
- Batch compatible propagation objects on a batch dimension, never across the
  attention token axis. Stable object order, separate memory/softmax and no
  extra-frame waiting. Probe workspace requirements and deterministically split
  oversized batches. Runtime/backend policy owns execution choices.
- Add internal graph/allocation/transfer diagnostics; public RuntimeStats
  positional layout is preserved. Keep the library header-only.

## Steps

1. Preserve the committed baseline executable/source and original-model receipts.
2. Implement reusable workspace/graph ownership with lifecycle tests.
3. Integrate tracker caches and resident/chained frame execution.
4. Extend supported graph operations to true object batches and wire compatible
   propagation grouping without changing temporal policy.
5. Run CPU/Metal CTest, isolated tool checks, ownership/reset/shape/exception
   tests, original image quantization regressions and affected original video
   F32/hybrid comparisons. Do not relax gates or state boundaries.
6. Once implementation and correctness are stable, run unified image timing:
   F32/F16 + four vision + four full presets, CPU/BLAS, native CPU and Metal.
   Use the same input/prompt/threshold, 1 warmup + 5 full calls, separate cache
   hits, stage timing and peak RSS/workspace.
7. Run targeted tracker one/four-object matched A/B/A timing on CPU/BLAS,
   native CPU and Metal with F32/hybrid weights, separating cold/setup from
   five warm calls. Compare the committed baseline and candidate serially
   under AC without overlapping model work or builds. The optional full-video
   timing protocol remains 64 frames, 16 warmup + 48 measured and final drain;
   it is outside this batch's default scope as clarified below. Complete
   original-reference video correctness remains required for all four supported
   F32/hybrid CPU/Metal cells. Separately record attention/MPS applicability
   and any measured adoption decision. Correctness/support/performance are
   independent claims.
8. Keep experiments, failures and artifact hashes in this plan; update only
   concise user-facing behavior/performance and Unreleased changelog entries.
   The refreshed benchmark guide will show current image and targeted tracker
   measurements. Retain the earlier 64-frame full-video numbers in their original
   profiling plan, rather than presenting them as current post-pipeline results.

## Verification

Check fresh inputs on graph reuse; same-shape/different-data calls; shape swaps
and bounded workspace; frame/session reset and model lifetime; copied outputs
after arena reuse; pointer/seed/present branches; empty pointers and objects;
batch 1/2/4/8 isolation, tails and split budgets; ID continuity/occlusion/hotstart
and final drain; no Metal CPU fallback introduced; original gates unchanged.

The final measurement requires fresh identities for source/build/model/library
and inputs. Current power observation is AC; confirm it again at measurement
start and record sampling limitations. A smaller model file does not establish
lower latency or RSS. Old performance records remain historical baselines.

## Results

Implementation checkpoint: shared `GraphWorkspace` and repeatable `GraphExecution`
binding are in place. A portable A→B→A regression verifies fresh input data,
external-weight ownership, retained output vectors, active/inactive graph
destruction, invalid-input recovery and reserve/rebind behavior. CPU and Metal passed. The test also exercises partition copies, rejects foreign runtimes/inactive compute and catches insufficient workspace capacity before a GGML assertion.

The first sizing test exposed a pinned GGML allocator detail: `reserve_size`
mutates the allocator plan and may free existing buffers even though it does not
allocate replacements. Using it on the live reusable scheduler caused a null
buffer crash. Sizing now uses a temporary metadata-only scheduler and restores
sources afterward; the session's live arena is preserved. Cached graph rebinding
also restores sources replaced by scheduler partition copies before scratch
contexts are reused. The initial standalone CTest run failed with a segmentation fault; subsequent
CPU/Metal lifecycle checks passed after this repair. That failed run does not
qualify as performance or quality evidence.

Remaining implementation and final correctness/timing receipts are pending.

## Sampling scope clarification

An optional scope question was sent before formal measurement: complete image
quantization timing with targeted video comparisons (recommended), complete
CPU/Metal video timing, or additional native-CPU video timing. The original
64-frame CPU protocol takes hours per matrix; no formal model timing has
started. In the absence of a new scope preference, proceed with the complete
30-cell image matrix and targeted tracker/video comparisons. Preserve the
original video correctness gates; targeted tracker fixtures are diagnostic and
cannot be reported as complete-video speed or original-model acceptance.

Implementation checks so far: CPU and Metal Release full builds succeeded,
including independent/repeated headers and two-TU linkage; each passed 13 CTest
cases. The isolated reference tools suite passed 65 tests after allowing
validated original F32 video benchmark cells. Final source/build identities will
be frozen after the final resident positional input change, then affected
original-model regressions and timing will run in new directories.

## Frozen implementation checkpoint

All three candidate paths (CPU/BLAS, CPU without BLAS/Accelerate and Metal)
built successfully and passed all 13 CTest cases after the final position-bank
change. Fresh logs are `build/pipeline-optimization/20261004/ctest-final-*.log`;
source, executable, libraries, models and original inputs are captured by the
new `candidate-inventory.json` in that directory. Production source is frozen.

The original F32/Metal two-session check passed model lifetime, interleaving,
replay/reset, progress isolation and saved-output ownership. The complete Metal
image matrix passed all 70 original-checkpoint cases across F32/F16 and four
vision/four full quantization profiles. CPU/BLAS image validation is running.
These are correctness runs; their wall times are not formal performance samples.

New passed tensor outputs may share hardlinks only after matching size/mode and
verifying both SHA-256 values. Their paths, bytes and receipts are unchanged.
This bounds repeated validation storage without modifying older evidence.

## Four-object decoder integration failure

Both image matrices completed: 140/140 original-checkpoint cases passed. The
first real-weight synthetic four-object tracker probe then aborted in
`sam3_build_sam_dec_graph`: its shared single-batch positional embedding was
reshaped as if it already contained four batches. The failure was observed
before formal timing. The diagnostic uses six spatial memories and sixteen
pointers; it is not a complete-video quality oracle. Its stdout/stderr and
partial setup are preserved in `tracker-correctness/candidate-f32-metal-four.*`.

The repair scope is the omitted decoder broadcast and a complete small decoder
graph regression. Earlier attention/deconvolution checks did not cover that
combination. Original gates remain frozen. Candidate identities and affected
checks must be refreshed after the repair; earlier passing image receipts stay
intact as the previous candidate generation.

The decoder repair passed the complete small-H B4-vs-B1 regression and the
real-weight Metal four-object probe. Four objects split deterministically into
two compatible batches under a 600,113,152-byte old serial workspace budget;
462,520,320 arena bytes plus 136,693,760 resident bytes fit that cap. All five
exported tensor groups for all four objects were bit-identical both to single
object execution and to the committed baseline's tracker implementation. This
is a synthetic same-shape diagnostic, not full-video acceptance or timing.

Fresh v2 full builds and all 13 CTest cases passed on CPU/BLAS, native CPU and
Metal. Prior source/binaries were preserved with their original hashes in
`candidate-v1-source` and `candidate-v1-binaries`. V2 identities and outputs use
`build/pipeline-optimization/20261004/v2/`; old receipts are not overwritten.
The original five-case Metal video matrix is now running.

## Resident memory and frame-boundary review

A read-only lifecycle review confirmed that scheduler reset cleared tensor
pointers but retained the 462,520,320-byte tracker allocation. It would overlap
with the next frame's vision graph. The numerical v2 queue was stopped, with
its original receipts and partial outputs preserved. Motion alone passed all
48 frames; the interrupted cell is not accepted as a complete matrix.

`release_storage()` now discards the active binding and releases the scheduler's
compute buffers at frame end. A later bind recreates the scheduler lazily while
cached graph contexts and cumulative diagnostics survive. A standalone CPU/Metal
lifecycle regression passed allocation release, A→B→A metadata reuse, reallocated
external-frame inputs and owned output preservation.

Metadata-only real-weight Metal probes clarified the budget: at S6/P16 the
old condition graph sets the 600,113,152-byte cap. Memory encoding uses
47,775,744 bytes. At S1/P1 the seed decoder instead sets a 371,888,128-byte cap;
the nonseed decoder plus low-resident core would total 393,152,512 bytes and
exceed it by 21,264,384 bytes. These are allocation probes, not timing or
model-output acceptance. Full→low FPN residency demotion alone cannot guarantee
all legal branches remain below the cap. The final fallback therefore returns
to local graph inputs while retaining graph metadata when residency still
exceeds budget. All dynamic inputs remain fresh; numerical and temporal rules
are unchanged. No formal performance samples have been collected.

## Actual allocation budget correction

The final no-compute allocation report found a scheduler sizing distinction:
the old S1 seed decoder requires 371,888,128 bytes in the size probe but actually
allocates 377,196,544 bytes. Candidate limits now compare actual allocated bytes
with the same old serial graphs' actual allocations; required size remains a
separate diagnostic. This preserves the old legal seed workload rather than
rejecting it using a smaller estimate. Model/output gates are unchanged.

Old baseline allocation probes run after candidate storage and resident inputs
are released, and release each probe arena before frame inputs are rebound.
Allocation guards inspect both the new graph requirement and the retained
arena's actual bytes, releasing/rebinding if a previous larger graph would
exceed the allowance. A frame-scoped none-resident fallback uses fresh local
inputs and guarded serial conditioning/decoding when full/low residency cannot
fit. All graph/context metadata and returned outputs retain correct ownership.

Reproducible sizing sources, actual allocation logs and build/test logs are in
`build/pipeline-optimization/20261004/memory-budget-probes/README.md`. Production
source is frozen for v3 acceptance. Fresh full builds and 13 CTest cases on each
CPU/BLAS, native CPU and Metal path passed, as did the isolated 65-test tool
suite. All four fresh original F32/hybrid CPU/Metal session checks passed model
lifetime, independent-session progress, reset/replay and owned output snapshots.
Remaining source-bound model acceptance and formal performance are in progress.

## Backend-specific budget validation

The first v3 private tracker runner incorrectly compared a CPU cap with the
Metal-only allocation report. That checker stopped before formal timing. The
production path computes its cap using the current runtime's old serial graphs;
no production change was needed. Independent no-compute probes found that CPU
and native CPU allocate 207,030,272 bytes for S1/P0 and S1/P1, and 207,257,600
bytes for S6/P16. Metal retains the previously recorded 377,196,544-byte and
600,113,152-byte caps. Backend allocation layout is not interchangeable.

The original stop receipt remains untouched. A separate correction receipt in
`v3/tracker-correctness/cpu-s1-p0-o1/corrected-by-backend-cap-evidence.json`
records matching CPU evidence. Its five raw tensor arrays are bit-identical to
the committed baseline; candidate arena plus resident bytes stay within the
matching CPU cap, and both live allocations are zero after `end_frame`.

`graph_reuses` counts reuse of an already active arena binding. Frame-end
storage release requires a new binding even when cached graph metadata survives;
the performance report uses graph-build deltas to evaluate metadata reuse.

## Mutable batch-limit failure

The next v3 CPU S1/P0 four-object check exposed a production bug: the committed
baseline returned four predictions, but the candidate returned one after two
batch splits and a serial fallback. No formal timing or complete v3 video
acceptance had started. Full raw outputs are preserved in
`v3/tracker-correctness-after-cap/cpu-s1-p0-o4`.

Both propagation and pointer projection advanced their chunk loops using the
mutable batch limit. A recursive residency fallback reset that limit; the
outer loop then skipped unprocessed objects. The repair advances by the actual
captured chunk count, checks returned coverage, and adds a regression that
changes the limit during a callback. V3 remains a failed historical candidate;
the repaired source will use a new v4 inventory and receipts.

Both affected loops now share a small SAM 3 chunk iterator that advances by
the captured count. Per-chunk and final count checks reject missing results.
The regression changes the next limit from two to eight during the first
callback and verifies complete ordered coverage of nine objects. One fused
graph is bounded at eight objects, the existing verified batch range; arbitrary
larger object sets remain ordered chunks instead of constructing an unbounded
graph before memory sizing. This does not reduce `VideoOptions::max_objects`.
Targeted Metal math/workspace checks passed 2/2. V4 full builds are running;
current-source model comparisons and final measurements will follow serially.

V4 fresh full builds subsequently passed on CPU/BLAS, native CPU and Metal;
each CTest log records 13 passed tests and no failures. The isolated tool suite
passed 65 tests. `v4/candidate-inventory-v4.json` and
`v4/candidate-v4-snapshot-manifest.json` preserve 97 production/test source files
and 63 executable/library artifacts, verified against their captured hashes.
The separate backend allocation probes reproduce the CPU/native and Metal caps
above. Current model comparisons follow this v4 inventory.

## V4 tracker correctness

The legal CPU S1/P0 four-object failure is repaired: the candidate returns all
four objects and its 20 raw tensor groups match the committed baseline exactly.
The complete private shape matrix then passed all 18 unique cells: CPU/BLAS,
native CPU and Metal × S1/P0, S1/P1, S6/P16 × one/four objects. All 225 array
comparisons are bit-identical, including batch-vs-single object isolation;
the matching actual cap and zero live arena/resident storage after frame end
were checked. These synthetic tracker checks do not replace original-video
reference acceptance.

The ignored comparison CLI initially rejected P1 before model loading. Its
parameter validator now accepts 0/1/16 pointers, with all planned combinations
preflighted using a missing checkpoint. Production source and its v4 inventory
are unchanged. Earlier CLI-stop outputs stay in `v4/tracker-correctness/`;
remaining comparisons use `v4/tracker-correctness-p1/`, and the isolated bug
regression is in `v4/tracker-correctness-s1p0-o4/`. Each receipt binds its actual
helper executable and raw arrays; the 18-cell count excludes duplicate reruns.

Fresh v4 original-model session checks passed 4/4 (F32/hybrid × CPU/Metal).
Model lifetime, owned outputs, interleaving/progress isolation and stable reset
replay snapshots are all true. The second session uses the distinct crop/resize
PPM with a negative prompt; the earlier two labeled corpus cases intentionally
shared the same PPM bytes. Source/model/binary/input hashes stayed stable before
and after the checks. Results are in `v4/session-checks/session-checks-v4.json`.
The short matched tracker timing phase follows these passing checks, before
complete original-reference image/video acceptance. Its results remain candidate
performance evidence until the full affected quality matrix passes.

## First matched timing and policy repair

The v4 CPU/BLAS F32 S6/P16 one-object cell ran an A1/B/A2 process sequence,
each with one cold call and five matching warm fixture seeds. Source/model/
executable/inventory, AC, exclusivity and thermal guards passed. Its warm median
A-bracket/B ratio was 1.0061 (about 0.6% faster), below the planned adoption
threshold. Peak process RSS was approximately 4.17/4.24/4.17 decimal GB.
The candidate rebuilt two graphs and ran two size probes on every warm frame;
serial and none-resident fallback counters advanced each frame. This is a
performance-policy problem despite passing output and arena-cap checks.

The complete receipt remains in `v4/tracker-performance-smoke-v4/`; it is not
reported as an accepted optimization. Workspace and resident independent maxima
cannot be added to infer a simultaneous peak, and the process RSS log has no
stage timestamps to isolate the exact cold peak. A narrow repair retains one
matching none-resident serial decision across frames and removes memory position
values from none-mode graph keys, since that mode uploads them as fresh dynamic
data. Runtime/profile/shape/budget changes still require a new decision; resident
bank views retain position-dependent keys. Final acceptance will use v5 receipts.

Read-only disk estimation from actual v2 motion outputs found 48 frame folders
and four selected tensor snapshots totaling about 1.40 GB, with ordinary frame
folders averaging 2.16 MB/frame. A conservative estimate is 35 GB for four
complete video cells and 34.5 GB for 30 image qualification cells, below the
observed 283 GiB free space even with additional margin. Old evidence is retained.

V5 retains one none-resident serial decision keyed by runtime/backend,
storage/arithmetic profile, S/P and actual budget, recorded only after a real
none-fused budget rejection. A matching propagation call activates it before
resident selection/upload and preserves it through batch-policy configuration.
Seed and memory branches retain independent guards. None-mode position values
are fresh uploaded data; resident view layouts still carry position keys.
The policy-key regression checks invalidation across runtime/profile/shape/
budget changes. Targeted math/workspace checks passed before the final small
state-assignment simplification; fresh CPU/BLAS, native CPU and Metal full builds
then succeeded. Full current-source checks and the failed-policy timing cell
are being repeated. Allocation/upload staging was left unchanged because the
previous RSS log cannot isolate a peak stage.

## V5 measured cache repair and cold-upload follow-up

Fresh v5 CPU/BLAS, native CPU and Metal builds each passed 13 CTest checks;
the isolated tools suite passed 65. In the repeated CPU/BLAS F32 S6/P16 O1
smoke, warm graph builds/probes were `[1,0,0,0,0]` (one first-warm boundary
rebuild), and the serial fallback counter stayed at one. The median A-bracket/B
ratio was 1.0242, with process RSS about 4.183/4.242/4.170 GB. Subsequent CPU
cells were within a few percent of their baselines; these are not adopted CPU
speedup claims.

Eight CPU/BLAS timing cells completed before the queue was stopped for the
follow-up. A native-CPU A1 process was interrupted with owned-PID SIGTERM;
its return-143 partial output remains preserved. One CPU F32 S1/P0 O4 cell's
B process (07:36:53–07:37:05 CST) may overlap a small Metal CTest ending at
07:37:01. The entire A/B/A cell is marked unqualified and will be remeasured;
its raw records are not overwritten. Later hybrid cells started after that
test, but all v5 measurements remain historical candidate evidence.

Upload counters confirm a concrete cold waste: the v4 cold call uploaded
106,168,320 bytes more than each warm call, exactly the high-FPN buffer later
demoted by the budget guard. The v6 repair allocates/binds external buffers for
sizing but defers writing resident data until an accepted graph actually reads
it. None-resident serial execution should not upload rejected frame buffers.
Numerical/shape/precision policies and actual-arena caps stay unchanged. Before
the edit, the timing worker confirmed all owned runner/model/caffeinate processes
had ended. During formal samples no agent builds or runs CTest; only the timing
worker runs inference.

V6 separates external buffer allocation from writing resident values. Core and
high-FPN uploads occur once per frame immediately before an accepted graph
reads them. Pointer projection only receives its own input. Frame/probe/none-mode
release resets both upload flags; high-FPN demotion keeps an already uploaded
core valid. An upload callback failure leaves the completion flag false. The
small regression covers repeated calls, a new frame, high demotion and exception
retry. Targeted Metal math checks passed; fresh final builds/model comparisons
and timing are delegated to the single execution worker.

A read-only memory-selection audit also identified legal maxima S10/P18 (four
conditioning plus six recent spatial records, and up to eighteen pointers).
The representative S6/P16 cell is not that upper bound. A separate synthetic
S10/P18 eight-object check will use valid spatial positions
`[0,0,0,0,1,2,3,4,5,6]` and compare all objects with serial execution; standard
original-model gates and the eighteen representative tracker cells stay intact.

## V6 complete matched timing and metadata capacity

All 24 synthetic tracker A/B/A cells completed with output/arena/source/power/
exclusivity checks passing. The report and per-sample direction counts are in
`v6/tracker-performance-v6/tracker-performance-summary-v6.md` and
`tracker-performance-direction-counts-v6.md` with JSON counterparts. CPU/BLAS
and native CPU remained within a few percent of baseline. These measurements
do not establish a CPU acceleration benefit.

For Metal S6/P16 four objects, F32 and hybrid median A-bracket/B ratios were
1.237 and 1.243 (about 19.2%/19.5% lower candidate latency); all five matched
warm comparisons had the same direction. A1/A2 median drift was 0.18%/0.22%.
Process RSS A1/B/A2 was 4.572/4.345/4.582 GB for F32 and
3.890/3.654/3.883 GB for hybrid. This is synthetic tracker timing, not complete
video frame latency. CPU S6/P16 O4 had higher candidate process RSS and less
than 5% latency benefit, so its memory cost still needs correction.

The no-runtime/no-weight metadata probe verified the full 1,464-tensor model,
four memory-attention layers and two decoder layers at legal S10/P18/B8.
Worst fused graphs had 3,165 nodes, 3,173 context tensors and 254 reachable
leaves within a 32,768-capacity graph; condition had 2,401 nodes, seed decoder
431, memory encoder 148 and pointer projection two. Fixed retained-slot context
reservation was about 35.4 MB; the pinned scheduler context formula for graph
size 65,536 reserved about 1.324 GB, excluding its other allocations. These are
metadata capacity/reservation estimates, not touched pages or RSS. Source,
binary, inventory and raw output are bound in
`v6/memory-budget-probes/metadata-footprint-summary-v6.json`.

The compact candidate uses 4,096 for fused/conditioning, 1,024 for decoder,
512 for memory and 128 for pointer graphs, preserving measured margin at the
legal maximum. Old serial actual-budget probes retain their original graph
requests and private workspaces; their frozen allocation reference remains
unchanged. Their cold diagnostics and costs remain accounted for. V6 raw
correctness stopped after 11 complete cells (130 bit-identical arrays) before
this source change; a partial native O4 baseline remains preserved. Final
acceptance will repeat against the compact source.

## V7 compact-source checkpoint

The compact metadata probe passes all 18 maximum-shape node/leaf/context-slot
checks. F32 retained graph-context reservation is 4,476,240 bytes; the scheduler
context formula is 165,480,912 bytes. Old serial probes keep their original
graph capacities with private workspaces, and their actual peaks and cold
diagnostics remain accounted for. All three fresh full builds passed 13 CTest
checks each, and the isolated tool suite passed 65. V7 inventory/snapshots bind
97 source files and 63 executable/library artifacts.

CPU/BLAS F32 S6/P16 O1 A/B/A then passed output/source/power/cap guards. Cold
upload is 180,537,344 bytes, retaining the 136,693,760-byte reduction against
v5. Warm builds/probes are `[1,0,0,0,0]`; the fallback counter remains one and
frame-end arena/resident bytes are zero. Actual serial peak/cap is the unchanged
207,257,600 bytes. Process RSS A1/B/A2 is 4.1736/4.1751/4.1701 GB, a roughly
3.1 MB (0.08%) candidate delta relative to the bracket, instead of v6's
approximately 70 MB excess. Median A-bracket/B is 1.0122; this remains a
performance-neutral CPU observation, not an adopted CPU speedup claim.
The receipt is `v7/tracker-performance-smoke-v7/`; final matrices follow it.

V7's standard tracker matrix passed all 18 cells on CPU/BLAS, native CPU and
Metal: 225 exported tensor groups were bit-identical to the committed serial
baseline, and every four-object batch matched its single-object execution.
The legal S10/P18/B8 boundary passed on all three paths, adding 120 bit-identical
groups across three cells. Every object was returned in order, and the actual
serial budget, frame-end arena release and resident-buffer release checks passed.
These 345 comparisons exercise real weights with synthetic tracker inputs;
they do not replace full original-reference video validation.

Four actual-session checks (F32/hybrid × CPU/Metal) passed model lifetime,
owned-result preservation, reset/replay, interleaving and progress isolation.
The two sessions use distinct decoded image pixels. Current receipts are
`v7/tracker-correctness/tracker-matrix-summary.json`,
`v7/tracker-boundary-v7/` and `v7/session-checks/session-checks-v7.json`.

## Independent backend candidates

The pinned precise Metal flash-attention specialization supports equal head
sizes 32 and 64, not the tracker's 256-wide F32 head. The existing tiled graph
therefore remains the production path. This batch does not add a fused D256
kernel or change accumulation/state precision. A new kernel requires its own
backend patch, complete-key softmax/tail tests, full video qualification and
end-to-end benefit before adoption.

The standalone MPS probe passed F32 projection, transposed/batched/tail and
attention-layout fixtures. It reuses persistent graph/kernel objects and uploads
weights once for the prepared matched benchmark. Measured calls will include
activation copies, synchronization and output readback. Its host-softmax
attention fixture is correctness-only. The public GGML Metal API does not
provide native buffer/queue handles needed for an ordered zero-copy production
bridge, so MPS is an independent matrix comparison, not a new SAM backend.
The artifact source/contracts and correctness records are in
`build/pipeline-optimization/20261004/backend-candidates/README.md`; formal MPS
timing and the final adoption decision are still pending.

## V7 complete timing and host payload follow-up

All 24 targeted tracker A/B/A cells passed output, actual arena-cap, frame-end
release, source, power and exclusive-execution checks. CPU/BLAS latency changed
by -1.97% to +0.26%; native CPU by -0.50% to +0.33%. These small changes do not
establish a CPU acceleration benefit. Metal S6/P16 O4 warm median A-bracket/B
ratios were 1.2602 (F32) and 1.2507 (hybrid), about 20.6%/20.0% lower latency;
all five warm pairs were faster. A1/A2 median drift was -1.07%/+0.23%, and
process RSS A1/B/A2 was 4.584/4.258/4.585 GB and 3.869/3.570/3.868 GB.
These are synthetic tracker results, not complete video-frame timings.
Receipts are in `v7/tracker-performance-v7/`.

The full matrix reveals a remaining CPU/BLAS S6/P16 O4 RSS increase of
34.5 MiB (F32, +0.85%) and 27.9 MiB (hybrid, +0.69%). The hybrid baseline
A1/A2 RSS span was only 0.11%, so the positive delta is not established as
sampling noise. A read-only review found that `propagate_batch` still expands
all objects' BF16 memories to one F32 payload before budget splitting or serial
fallback, then copies slices. Four objects use 31,916,032 bytes for the combined
memory payload alone. This avoidable staging also affects the cold sample and
larger groups, despite the backend arena staying within its limit.

The narrow follow-up moves payload construction after actual chunk acceptance;
serial fallback constructs one object's payload at a time. Small projected
pointer-position values can pass through recursive splitting with explicit
object offsets. Math, precision, temporal rules, output ordering and budgets
stay fixed. Recheck all source-bound tests, raw batch/serial outputs and timing
after this change. Before editing, the execution worker confirmed no owned
tracker/model/MPS/caffeinate process remained. Full original image/video
qualification and MPS timing had not started; v7 evidence remains intact.

V8 changes only payload staging and its internal diagnostic: recursive splits
carry small projected pointer positions; accepted chunks construct their own
memory data after allocation, and serial execution releases each object's
payload before decoding. Spatial-record lookup and pointer-width checks still
run before propagation. `memory_payload_peak_bytes` records the capacity of
the owned MemoryPayload vectors, not total host heap or process RSS. The
targeted Metal build and two math/workspace checks passed. No model or benchmark
was run during implementation. The source is now handed back to the single
execution worker for fresh full builds, O4 memory validation and final matrices.

Fresh v8 CPU/BLAS, native CPU and Metal full builds each passed 13 CTest checks
(39 total); the isolated tools suite passed 65. The new inventory and snapshots
bind 97 source files and 63 executable/library artifacts. All four actual-model
session checks passed with current source/model/binary/input guards. Source
remains frozen while fresh actual-cap/raw-output checks and the focused O4
RSS validation proceed. These checks do not yet qualify the full original
image/video matrices or final performance tables.

The focused CPU/BLAS S6/P16 O4 raw check passed all 20 baseline comparisons
and batch-versus-single equality. Both F32 and hybrid A/B/A smoke cells then
passed output, source, power, cap and frame-end release guards. The observed
MemoryPayload capacity highwater is 7,995,392 bytes, one object's data including
pointer positions; rejected cold batches do not prepack the four-object array.

F32 process RSS A1/B/A2 is 4.2556/4.2503/4.2375 GB; hybrid is
4.2715/4.2528/4.2422 GB. Both candidate peaks lie within their observed baseline
brackets. Median matched A-bracket/B ratios are 1.0163 and 1.0159, remaining
below the 5% latency-reduction adoption threshold. Cold candidate samples and
their old serial-budget probes remain included. Build/probe deltas are
`[8,1,0,0,0,0]` for cold plus five warm calls. Receipts are
`v8/tracker-performance-smoke-cpu-f32-o4/` and
`v8/tracker-performance-smoke-cpu-hybrid-o4/`; these focused checks precede the
complete final matrices and do not replace them.

V8's complete standard 18-cell tracker matrix passed all 225 raw tensor-group
comparisons bitwise; its S10/P18/B8 boundary passed all three backends with
120 additional bitwise groups and batch-versus-single equality. Actual serial
caps are unchanged, all frame-end arena/resident values are zero, and no budget
failure occurred. CPU/native S6/P16 O4 MemoryPayload highwater is the same
7,995,392 bytes as O1; Metal O4 records 15,990,784 bytes for its accepted
two-object chunk. Current receipts are
`v8/tracker-correctness-standard18/tracker-matrix-summary.json` and
`v8/tracker-boundary-v8/tracker-matrix-summary.json`. The source is unchanged
for the complete matched timing and original-reference validation that follow.

## V8 complete targeted timing and RSS repeats

All 24 targeted A/B/A cells completed with output/source/power/thermal/sleep/
exclusive-execution and arena/release checks passing. The report is
`v8/tracker-performance-v8/tracker-performance-v8.json`, with the derived
`tracker-performance-summary-v8.md` and
`tracker-performance-direction-counts-v8-corrected.md` views. One A1/B/A2
process sequence supplies five matched warm seeds per cell; these are not
five independent restarts. Cold calls, old private budget probes and process
setup/RSS remain separately visible.

Metal S6/P16 O4 warm median ratios are 1.2338 (F32) and 1.2355 (hybrid),
18.95%/19.06% lower candidate latency. All five warm pairs are faster;
A1/A2 median drift is -0.22%/+0.13%. Candidate median/p95 is
3.1186/3.1252 seconds (F32) and 3.1162/3.1259 seconds (hybrid).
Process RSS A1/B/A2 is 4.5712/4.2329/4.5757 GB and
3.8807/3.5408/3.8763 GB. These establish a targeted tracker benefit on this
device; full-video quality and end-to-end frame performance remain separate.
CPU changes stay below the 5% latency adoption threshold, with both faster
and slower cells; no general CPU acceleration or memory-saving claim follows.

CPU/BLAS hybrid S6/P16 O4 was repeated in three independent sequences: the
focused smoke, full matrix and an additional new-directory repeat. Baseline
RSS values span 4.2401–4.2715 GB; candidate peaks are
4.2528/4.2578/4.2409 GB. Their respective bracket deltas are
-3.82/+15.88/-13.65 MiB. Candidate values lie within the pooled observed
baseline range, but the full matrix's positive delta does not lie inside its
own A1/A2 bracket. The samples do not establish a fixed RSS increase or saving.
Owned MemoryPayload highwater stays 7,995,392 bytes in all three, confirming
the all-object staging removal. The additional receipt is
`v8/tracker-performance-repeat-cpu-hybrid-s6p16-o4/tracker-performance-v8.json`.

## Independent MPS matched result

The persistent-object matrix probe completed four shapes × five matched A/B/A
cycles (20 measured cycles), with output/AC/thermal/exclusive/sleep checks
passing. Total host wall time includes activation copies, command submission,
synchronization and output readback; setup/one-time weight upload is separate.
Process peak RSS was 131,432,448 bytes. The receipt is
`v8/mps-performance-v8/mps-benchmark-receipt-v8.json`.

| Matrix fixture | GGML median total ms | MPS median total ms |
| --- | ---: | ---: |
| Tracker projection, M256/N5184/K256 | 1.0503 | 1.0246 |
| Decoder projection, M256/N4/K256 | 0.1405 | 0.1395 |
| Transposed tail, M13/N17/K67 | 0.1474 | 0.1486 |
| Batched transposed tail, M19/N11/K72/B2 | 0.1377 | 0.1298 |

Representative tracker and decoder projection improvements are only about
2.4%/0.7%; the small batched tail's 5.8% result does not establish a production
tracker benefit. This is synthetic matrix timing, not a complete attention or
video pipeline. Retain the GGML Metal route: these results do not justify a new
production MPS bridge, public backend or default dispatch. The D256 fused
attention candidate remains outside this common execution change.

## V8 original-reference regression

CPU/BLAS and Metal image matrices each completed all ten profiles × seven
original-reference cases (70/70 per backend). F32, mixed F16/F32, four vision
and four full-linear presets passed their unchanged profile gates. Native CPU
image validation and the four full video cells are still running; these passes
alone do not qualify the complete 210-case/864-frame matrix.

Independent numerical workers use separate output directories and may run
concurrently with available memory. Their wall times are not performance
samples. All numerical processes must finish before the exclusive 30-cell image
timing begins. The current image/video receipts are under `v8/runs/`.

An initial disk estimate mistakenly counted 216 frames per scenario. The
correct video scope is five scenarios totaling 216 frames per cell, four cells
and 864 frames overall. No expanded video workload was started. The append-only
`v8/disk-budget-v8-correction.json` keeps the earlier conservative estimate and
records about 69.13 GiB including contingency, with about 208.62 GiB remaining
at that snapshot. Immutable machine-readable evidence is preserved.

The image regression is now complete: each of CPU/BLAS, native CPU and Metal
passed all ten profiles × seven cases, 210/210 overall. Root independently
checked each matrix's complete/pass state, eligible metric and seven passing
case entries per profile. `v8/image-qualification-index.json` binds all 30
backend/profile cells to the current source/build/model inventory. The initial
copied indexer omitted native CPU; the ignored helper was corrected before
index generation, without changing production source, gates or receipts.

Original F32 video also passed all five scenarios/216 frames on both CPU and
Metal. Root verified the Metal cell's eligible metric and all five case pass
entries. Hybrid video is still running. No formal image timing begins until
the entire numerical queue has completed and exited.

Final numerical qualification is complete. Both video matrix indexes are
complete and passed, with F32 and hybrid passing all five scenarios on each of
CPU and Metal: 216 frames per cell, 864 overall. The scenario lengths are
motion 48, entry 64, occlusion 64, hotstart-removal 24 and negative 16. All
image/video metrics remain eligible under the unchanged original-reference
gates. No timing is inferred from these correctness runs.

The compact receipt is `v8/original-validation-summary-v8.json`. It binds the
three image and two video matrix indexes, each metric, the 30 image
qualification entries and the same 97-file production source snapshot. Its
`source_artifact_maps_identical_across_matrices` and `all_passed` values are
true. Root independently checked the final index states, metric pass/eligibility
and scenario entries. All numerical processes exited before formal image
measurement.

## V8 unified image measurement

The first measurement attempt in `v8/image-performance-v8/` stopped at the
exclusive-execution preflight before launching a model or collecting samples.
Its guard mistakenly matched its own ancestor `rtk` launcher as competing
work. The ignored runner now excludes only its own process ancestry, while
retaining detection of other model/build processes. Production source and
qualification gates did not change; failed evidence is preserved. Formal
measurement restarted in the new `v8/image-performance-v8-guardfix/` directory.
Its `report.json` records the complete 30-cell protocol and live progress.

The final report is complete and passed, with all 30 unique backend/profile
cells, zero failures, one warmup and five uncached full-image samples per cell.
All cells retain current output-quality qualification, input/model/source/build
identity and successful power/thermal/exclusive/sleep checks. Root independently
checked each cell's completion, return code, five-sample median, output quality,
before/after artifact-map equality and the common candidate inventory binding.
No production source changed during measurement.

Current image median ranges are 7.321–7.435 seconds on CPU/BLAS,
39.500–39.754 seconds on native CPU and 5.360–5.420 seconds on Metal. Whole-
process peak RSS ranges are 2.698–5.060, 2.672–5.028 and 1.770–4.363 decimal GB,
respectively. Full Q4_K records 7.431 seconds / 2.698 GB on CPU/BLAS,
39.754 / 2.672 on native CPU and 5.386 / 1.770 on Metal. Compared with this
run's CPU/BLAS F32 RSS of 5.049 GB, full Q4_K is about 46.6% lower. Similar
image latency across storage formats does not establish a quantization speedup.

Final image report SHA-256:
`bc9c6d009da9e6c175e26fe87b5aa01eb87f4b31b2098eab32a3ea9fa8d0ba4b`.
The source snapshot is
`7822d01ad3750f245830a70377ecec53ee5ef483ff32785ac496e9e353c321ab`;
candidate inventory SHA-256 is
`ccbe9813ddb09cb5d272c08c396e054233c4dc9c0869aea0529328a6df9dfbf4`.
All paths in the table below are relative to
`build/pipeline-optimization/20261004/`.

## Final verification and delivery

| Check | Final result | Current evidence |
| --- | --- | --- |
| Release builds / CTest | CPU/BLAS, native CPU and Metal: 13/13 each, 39 total | `v8/verification/build-test-receipt-v8.json` |
| Isolated reference tools | 65/65 | `v8/verification/tools-test-v8.log` |
| Actual-model session lifecycle | F32/hybrid × CPU/Metal, 4/4 | `v8/session-checks/session-checks-v8.json` |
| Tracker serial/batch tensors and limits | 21 cells, 345 unique raw groups bitwise, budget/release checks pass | `v8/tracker-correctness-standard18/tracker-matrix-summary.json`, `v8/tracker-boundary-v8/tracker-matrix-summary.json` |
| Original image quality | 3 backends × 10 profiles × 7 cases, 210/210 | `v8/original-validation-summary-v8.json`, `v8/image-qualification-index.json` |
| Original video quality | CPU/Metal × F32/hybrid × 216 frames, 864/864 | `v8/original-validation-summary-v8.json` |
| Targeted tracker performance | 24/24 A/B/A cells plus focused RSS repeats | `v8/tracker-performance-v8/tracker-performance-v8.json` |
| Independent MPS comparison | 4 shapes × 5 matched cycles, 20/20; retain GGML | `v8/mps-performance-v8/mps-benchmark-receipt-v8.json` |
| Unified image performance | 30/30, all output/source/environment guards pass | `v8/image-performance-v8-guardfix/report.json` |

The tracker table in the bilingual benchmark guide summarizes the twelve
primary S6/P16 cells. Its timing excludes outer fixture-data generation, while
per-call memory/pointer payload preparation remains inside the measured tracker
call. A read-only composition review confirmed the statistics, units, precision
scope and stage-only claim; the fixture wording was narrowed accordingly.
The image table was generated from the completed report and contains all ten
profiles on all three backends. Historical full-video timings stay in their
original profiling plan. The image and video manifests share the official
`sam3.pt` checkpoint SHA-256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.

Delivery implements bounded graph metadata/workspaces, frame-feature ownership
and chaining, and compatible-object batches with actual-budget splits and
serial fallback. Metal four-object tracker propagation is about 19% faster in
the measured fixture, with lower RSS and identical outputs. CPU measurements
do not establish an acceleration benefit. No complete-video latency or FPS
claim follows from these targeted measurements. The independent D256 kernel
requires separate precision-preserving work; measured MPS projections do not
justify a production bridge. Public task APIs and model temporal rules are
preserved. The initial quantization commit remains local; this pipeline patch
is left in the working tree for review, without a second commit or push.

Final documentation verification passed bilingual table parity and local links
in all 47 Markdown documents; `git diff --check` passed. The execution worker
confirmed that all owned model/timing/build/test processes exited and the final
97-file production source map still equals the frozen inventory. Archived v8
build/test logs, numerical receipts and formal reports remain unchanged.

The subsequent user request authorizes committing this completed patch before
operator-level profiling. The pre-commit refresh found all 97 source hashes
identical to v8 and reused the source-bound validation above. Read-only review
also noted a bounded host-memory opportunity: serial `decode` holds its caller's
conditioned vector while `finish_prediction` copies it into the owned result,
where the earlier serial implementation moved it. One tracker feature contains
1,327,104 F32 values, about 5.06 MiB. This is a transient copy, not an unbounded
cache, arena-budget violation or observed output defect. Preserve the validated
source for this commit; assess its cost in the next profiling plan.

Pre-commit architecture and four adversarial passes completed without a verified
new correctness blocker. A cascade reviewer raised allocator-failure recovery;
an independent skeptic and root compared the actual callers with `ae1b7af`.
Budget rejection returns false and selects demotion/splitting/serial paths.
Underlying allocation failure still throws and requires session reset, as the
baseline did, with added frame/workspace cleanup. The plan does not promise
recovery from system OOM. This static observation is a possible separate
hardening topic, not a demonstrated regression; no failure injection was run.
