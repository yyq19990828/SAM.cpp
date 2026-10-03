# Complete remaining M2 acceptance

Created: 2026-10-02 18:28:48 Asia/Shanghai.
Baseline: preserved `2ddee1b` plus current video/hybrid WIP.
Status: complete on 2026-10-03 under the user-approved F32 + hybrid CPU/Metal contract. No commit or push requested.

Current scope: required video support is F32 and hybrid on CPU/Metal. The
unstarted, unsupported F16/CPU 216-frame diagnostic is deferred; the completed
F16/Metal failure remains preserved. This supersedes the original four-cell
legacy queue below without changing any required support gate.

## Original scope

At plan creation, the user requests all remaining M2 work after the completed
explicit hybrid repair: the F32/F16 numerical matrix and the original 64-frame
one/four-object performance protocol, plus outstanding real session/lifetime,
image-subset and allocation-plateau checks. Reuse verified hybrid acceptance;
do not overwrite original weights, references, receipts or immutable builds.

## Original steps

The precision targets in steps 2 and 5 are superseded by the explicit support
decision and deferred diagnostic below; the other gates are retained.

1. Audit prior receipts against current model/executable/library/source hashes.
   Enumerate every remaining cell and behavior check. Never count a failed or
   incomplete receipt as a passing matrix cell.
2. Write one serial execution queue using the current immutable CPU/Metal
   builds and the behavior-verified original 216-frame reference. Run F32/CPU,
   F16/CPU and F16/Metal; also run F32/Metal because it is now an advertised
   backend. Use the existing standard validator with unchanged numerical,
   candidate/pointer, ID, lifecycle, state, output and backend gates. Preserve
   failure receipts; investigate the earliest failed contract before any fix.
3. Reuse only evidence bound to matching artifacts for the original image
   matrix, full-video image subsets and real multi-session/reset/lifetime tests.
   Run missing or insufficiently bound checks with fresh output directories.
   Check hybrid sessions too. Rebuild only if a demonstrated production defect
   requires it, then invalidate affected receipts and rerun required gates.
4. Create separate deterministic 64-frame performance fixtures with one and
   four objects; keep the frozen correctness corpus unchanged. First prove the
   expected objects/continuity with original Meta modules on a short prefix,
   then verify all actual measured frame counts and IDs. Record source/recipe/
   PNG hashes. A failed fixture is not a passing four-object benchmark.
5. Implement the minimum reproducible benchmark runner around `sam_video` and
   existing artifact helpers. Run processes sequentially with threads=4 and no
   tensor dumps, after correctness work. Required default F16 cells are CPU and
   Metal, each with one/four objects; measure the validated hybrid pair too.
   Preserve all 64 frame/stage samples, first-output delay, object counts,
   retained records/state, model/compute allocation, process RSS and conditions.
   Warm up frames 0-15; report median/p95 over frames 16-63 and explicitly retain
   the final-drain cost. Check allocation behavior after warmup without claiming
   a universal RSS bound from a finite run.
6. Update README, MODEL_ZOO, BENCHMARK, changelog and original plan status only
   from completed evidence. Run appropriate tools/CTest/headers/consumer checks
   if implementation changes, plus `git diff --check` for final delivery.

## Constraints and failure handling

Retain F32 L2 <=0.001, mixed L2 <=0.02, original output/score/box/IoU and exact
selection gates. Never change strict argmax, BF16 temporal storage, Meta policy,
reference recipes or default payload identities to hide an error. The old F16
17-frame pressure limitation is separately recorded; finite passing cases do
not erase it. An irreducible original-vs-rounded-weight failure must remain an
explicit unsupported boundary, with its exact evidence and any support-contract
change presented as such, rather than silently substituted by hybrid acceptance.

Heavy inference and performance measurements run serially. Use fresh ignored
paths under `build/video-validation/20261002-182848/` and
`/private/tmp/sam-video-20261002-182848/`. The existing builds are
`/private/tmp/sam-video-20261002-133548/{cpu,metal}` and reference is
`/private/tmp/sam-video-20261002-133548/reference-full`.

## Verification and results

Required matrix receipts use `tools/validate_video.py`; artifacts/output files
are frozen and rechecked. Benchmark reports must reject wrong frame count,
precision/profile, backend placement, object-count/continuity, pending-output,
state bounds or changed model/binary/library/input/output fingerprints.
One focused regression must demonstrate that malformed/incomplete benchmark
receipts fail. No unrequested dependency or generic benchmark framework.

The records below preserve the original sequence of decisions; the final results section is authoritative for current scope.

### Initial qualification and default F16/Metal result

Independent one/four-object performance fixtures each contain 64 hashed PNGs.
Original Meta modules qualify 17 frames with declared length 64: exactly one /
four fixed visible IDs, no further birth/removal, nonempty masks, and first
output frame 0 emitted after processing frame 14. The fixture uses max_objects=8,
not an artificial cap of four. Both source processes and guards finish cleanly.
This prefix is fixture qualification, not a full numerical oracle replacement.

The standard default F16/Metal 216-frame cell completes with four cases passing
and entry failing only exact mask/pointer candidate identity at frames 23 and 24.
All its other recorded tensor/output/state/ID checks pass; that does not override
the failed discrete gate. At frame 23, original object 0 has iou3-iou1 =
+1.25169754e-5 (chooses 3), versus -2.83120e-5 in C++ (chooses 1). Frame 24 is
already conditioned by that different pointer/history. These margins identify
a selection boundary, not an operator defect.

The matrix supervisor is held after this cell while an exact-F16 Meta replay
runs frames 0-25 with full 64-frame initialization and all 1,464 model payloads
verified against the actual GGUF. This separates original-value quantization
from C++ versus same-weight arithmetic. No graph, model, storage policy or gate
is altered. The original supervisor resumes after this diagnostic window.

Old default image/lifetime receipts lack sufficient pre-comparison output and
current executable/library binding. Fresh image/schema-subset and real-session
checks are therefore queued. A permanent optional long-session test and separate
64/16/48 benchmark runner are implemented without modifying runtime behavior;
independent CPU/Metal long-test compilation succeeds against immutable libraries.
Tools now pass 19/19, including a malformed benchmark/ID-replacement regression.

### Proven precision boundary and user-approved default

The exact-F16 Meta replay completes all 26 frames. All 1,464 actual GGUF payloads
match the model state. Meta and C++ choose the same candidates on every frame;
Meta also diverges from original FP32 at precisely frames 23/24. Frame-23
Meta iou3-iou1 is -2.79545784e-5 versus C++ -2.83120e-5; frame-24 values are
+5.45978546e-5 / +5.47170e-5. Small same-weight arithmetic differences remain,
but a C++ operator defect is unnecessary to explain this failed gate: weight
rounding reproduces it within unchanged original modules. Evidence:
`entry-exact-f16-replay.json` and `entry-exact-f16-causal-summary.json`.

The user explicitly chooses hybrid as the video default, preserving explicit
F16 and all failed receipts. This deliberately revises the supported M2 precision
contract; it does not relabel F16 as passing. Conversion with omitted precision
now selects hybrid only for video. The image CLI previously requires explicit
precision and retains that requirement; explicit F16/F32 and existing artifacts
remain unchanged. The supported full matrix is F32 and hybrid on CPU/Metal.
The legacy F32/F16 results remain separate; the subsequently deferred F16/CPU diagnostic is explicitly recorded below.

The 64/16/48 performance protocol now measures the validated hybrid default on
CPU/Metal, one/four objects (four complete cells). F16 lacks numerical eligibility
and is retained as a diagnostic profile rather than measured/advertised as an
accepted performance cell. There is no automatic inference fallback, changed
argmax, changed BF16 policy or relaxed numerical gate. The former benchmark
step's F16 cells are superseded by this explicit user decision.

### Avoid optional CPU reruns (2026-10-03)

Under the user-approved F32+hybrid support contract, the unstarted F16/CPU
216-frame run is diagnostic rather than a support prerequisite. It is deferred
to avoid roughly three hours of optional CPU work. It is recorded as
deferred/not-run, never PASS; F16/Metal's entry frames 23/24 failure and its exact-payload
Meta reproduction remain unchanged. F32/Metal completed all 216 frames, and the
F32/CPU validator continued without restart or altered inputs/gates.

Only scheduler PID 9099 was stopped to prevent automatic launch of the optional
cell. Validator 13266 and its SAM child continued normally; task-scoped caffeinate
also covered 13266 without changing system power settings. All five F32/CPU
cases and its actual exit were verified before closing the scheduler.
Its original incomplete default-video-matrix.json stays unchanged. The separate
supported-video-matrix-audit.json now binds the two actual F32 runs and the two
already sealed hybrid runs; optional-f16-cpu-deferred.json records the rescope.
The closure and actual validator exit evidence are described below.

Reuse accepted numerical baselines when model, binary, backend libraries,
inputs, gates and sealed outputs still match. Documentation or CLI dispatch-only
changes do not require repeating model inference when their limited scope is
verified. Changes to model/storage values, preprocessing, graphs, backend math,
dependency/toolchain behavior or temporal state require checks for the affected
profiles. Long interleaved-session behavior and controlled performance remain
separate first-time checks; short runs or dump-heavy validation timings cannot
replace them. No cache framework or numerical tolerance change is introduced.


## Final results and evidence (2026-10-03)

All required execution stages finish successfully, ending at 08:21:51
Asia/Shanghai. `remaining-queue-v2-progress.json` records each phase's actual
exit 0 and hashes its completed receipt. No model, official executable, backend
library, numerical gate or temporal policy changed during these batches.

### Original numerical and regression acceptance

| Precision / backend | Frames / cases | Maximum stage L2 | Result |
| --- | ---: | ---: | --- |
| F32 / CPU | 216 / 5 | 0.000203442011 | PASS |
| F32 / Metal | 216 / 5 | 0.000346343011 | PASS |
| Hybrid / CPU | 216 / 5 | 0.002649662678 | PASS; matching sealed run reused |
| Hybrid / Metal | 216 / 5 | 0.002593767526 | PASS; matching sealed run reused |
| Explicit F16 / Metal | 216 / 5 | 0.004667089644 | FAIL: entry 23/24 candidate identity |
| Explicit F16 / CPU | — | — | Optional diagnostic deferred, not run |

The full original reference proves all five behaviors and is eligible. Required
gates remain F32 L2 <=0.001, hybrid mixed L2 <=0.02, and unchanged mask/score/box,
candidate, fixed-ID, group/memory/pointer order, lifecycle, state and placement
checks. `supported-video-matrix-audit.json` independently verifies all required
864 frames and preserves the failed diagnostic cell. F32 CPU's motion run
crosses host sleep, so its wall timings are excluded from performance; its
numerical output and actual exit 0 remain valid.

`image-session-regressions.json` and `image-short-session-audit.json` prove
56 fresh original image cases (schema 1 and full schema 2, F32/F16, CPU/Metal),
14 reused sealed hybrid image cases, and six real short session checks.
All result precision/profile labels are checked where the format exposes them;
short-session JSON has no storage-profile field, so its profile is bound through
the validated GGUF and run artifacts rather than an invented result field.
Earlier image receipts lacking current executable/output binding are not
promoted. Schema-1 files use their actual official-revision directory.

`hybrid-long-sessions.json` records two successful backend runs, each with
64 positive entry pushes interleaved with 64 negative grocery/`purple elephant`
pushes. The caller Model is destroyed before processing; actual moved output
survives subsequent pushes and both session destructors. Positive masks, IDs,
scores, boxes, choices, history and state match the same-backend accepted
standalone entry sequence exactly; all 64 negative outputs are empty.
Each session encodes text once and vision 64 times; the other session's counters
remain unchanged at each push. Positive state peaks at 53 records / 35,222,528
bytes; negative state stays empty.

Observed graph high-water is constant after frame 16: 1,020,660,736 bytes CPU
and 1,245,268,288 Metal. Long-run peak RSS is 5,476,302,848 / 4,584,030,208
bytes respectively. Current-RSS median windows (completed pairs 16–31,
32–47, 48–62) are CPU 4,577,886,208 / 4,642,783,232 / 4,644,372,480 and
Metal 3,834,380,288 / 3,893,624,832 / 3,901,702,144 bytes. These finite
observations show later stabilization, not a universal process-RSS bound.

### Controlled default-profile performance

The original-Meta-qualified one/four-object fixtures are independent of the
correctness corpus. All four hybrid cells verify the expected nonempty masks,
counts and stable IDs on every actual frame. Each completes 64 frames,
warms up 0–15 and measures 16–63, including frame-63 drain of 15 results.

| Backend / objects | Frame median / p95 (s) | Peak RSS (bytes) | Model load (ms) |
| --- | ---: | ---: | ---: |
| Metal / 1 | 7.833198 / 7.860934 | 4,165,976,064 | 942.182417 |
| Metal / 4 | 13.356173 / 13.430080 | 4,294,803,456 | 902.359667 |
| CPU / 1 | 51.101675 / 52.173591 | 5,182,652,416 | 991.386208 |
| CPU / 4 | 81.644055 / 85.854698 | 5,290,917,888 | 981.972375 |

All runs use four threads, max_objects=8 and no tensor dumps. Text is encoded
once, vision 64 times; first output follows frame 14 and final pending count is
zero. Available timing stages are frame/tracker/memory; separate visual and
detector stage timings are not exposed. `hybrid-performance-matrix.json` binds
each raw `benchmark.json`, all 64 samples, 48-sample median/p95, current-RSS
stream, time-l peak, actual backend placement, state and buffer accounting.
`performance-values.json` and [BENCHMARK.md](../../BENCHMARK.md#video-64-frame-protocol)
record details. One/four-object retained-state maxima are 27 / 108 records,
17,943,552 / 71,774,208 bytes. Weight and compute high-water remain constant
after warmup; process RSS is reported separately.

Runs are serial on the current M4 Pro Mac from 05:43–08:21 on 2026-10-03.
Every start/end is AC power with battery 100%; no sleep events or recorded
thermal/performance warnings occur. Task-scoped caffeinate covers the queue;
no system power settings change. No other SAM/Meta inference or compilation
overlaps performance. Ordinary desktop services remain; clocks are not forced.
The cancelled cloud handoff performs no upload.

### Tool changes, scheduler closure and reproducibility

The CLI default regression first fails against the old required-precision
entry point, then passes video omitted→hybrid, explicit F16/F32 unchanged and
image omitted→argument error using mocked conversion. All 20 isolated Python
tests pass (`tools-video-default-green.log`). The Python `convert(...)` API
and payload conversion do not change. `converter-cli-change-audit.json` binds
the historical Meta-replay converter hash to the current source by reversing
only its CLI edits; historical source receipts are not rewritten.

`tests/test_video_long_session.cpp` is permanent, with optional CTest arguments
in `tests/CMakeLists.txt`. Independent CPU/Metal probes compile against the
immutable libraries and execute the real long checks. Separate configure-only
output registers 11 ordinary tests plus the optional long target; this does
not claim a new 12/12 CTest run. Existing immutable builds retain 11/11 CTest
evidence. `long-target-configure.json` and `long-target-ctest-registration.json`
bind that verification without rebuilding existing executables or libraries.

The first continuation controller fails on an exit-observation assumption:
the completed validator PID disappears instead of remaining a visible zombie.
Its `remaining-queue-progress.json` is retained, and no later phase starts.
After all F32 CPU outputs pass, a fresh, owned output guard prevents optional
F16 CPU launch while the original supervisor is resumed solely to collect
the real F32 subprocess exit 0. Its expected refusal is preserved as the
legacy controller's exit 1. `cpu-f32-validator-exit.json` and
`legacy-native-close-report.json` bind the native exit evidence;
`legacy-queue-closed.json` proves no F16 CPU inference launched and the
original incomplete `default-video-matrix.json` was restored byte-for-byte.
This scheduling correction is not a model/threshold failure. The fresh v2
controller then completes each required phase sequentially.

All local receipts/scripts below are under
`build/video-validation/20261002-182848/`, with raw outputs under
`/private/tmp/sam-video-20261002-182848/`:

- `audit_supported_matrix.py` → `supported-video-matrix-audit.json`.
- `run_regressions.py` then `audit_regressions.py` → image/session receipts.
- `long_session.py compile|run` and `run_long_sessions.py` → standalone probe
  build receipts and both `long-session-{metal,cpu}/long-session.json` files.
- `prepare_performance_fixtures.py` and `qualify_performance_prefix.py` →
  64-PNG manifest and original `{one,four}-object-oracle-prefix.json`.
- `run_benchmarks.py` → four standard `tools/benchmark_video.py` invocations
  and `hybrid-performance-matrix.json`.
- `run_remaining_v2.py` → `remaining-queue-v2-progress.json` and
  `phase-{support-matrix,image-and-short-session,image-and-short-audit,long-sessions,hybrid-performance}.log`.
- `completion-summary.json` → final receipt, relevant source and documentation
  fingerprints, result-label audit and whitespace result.

Use each recorded command with fresh output paths for reproduction.
[README](../../README.md#official-video-references) supplies standard numerical
and long-test entry points; [BENCHMARK](../../BENCHMARK.md#video-64-frame-protocol)
supplies the qualified workload command and recipe. Reuse the pinned original
checkpoint/reference and existing model hashes from [MODEL_ZOO](../../MODEL_ZOO.md).
Models, media, raw tensors and local receipts remain ignored.

M2 is complete for the stated supported contract. Explicit F16 failures and
its unrun CPU diagnostic remain limitations; original-vs-rounded quantization
is not erased by same-weight arithmetic agreement or hybrid success. No
argmax epsilon, BF16 removal, implicit fallback, gate relaxation, model
replacement, commit or push is performed.

### Pre-commit reproduction command check

The final review found doubled shell line-continuation backslashes in the
64-frame benchmark documentation. Correct only that snippet and check its
actual shell argument expansion using an isolated recording executable; no
model or numerical acceptance is repeated. The sealed completion snapshot
remains the historical pre-review record; commit readiness records this delta.

Result: the old command fails shell expansion; the corrected command passes
and supplies all 11 expected option/value pairs to the recording executable.
All model, graph, library and numerical acceptance inputs remain unchanged.
