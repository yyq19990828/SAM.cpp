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
`performance-values.json` and [BENCHMARK.md](../../BENCHMARK.md#sam-3-video-tracking)
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
[README](../validation.md#compare-video-inference) supplies standard numerical
and long-test entry points; [BENCHMARK](../../BENCHMARK.md#sam-3-video-tracking)
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


<a id="historical-performance-record"></a>

## 历史性能与测量过程记录

迁自 `docs/benchmarks/m4-pro-20261003.md`，原始文件 SHA-256：`5ce7ab6ce17feb5f496ea72d2b1d9b2b1550bc14c127f6193ee94d6909f1d78c`。测量值与历史结论保持原样。

> Historical/detail record retained from the completed M2 work. Current summaries are in the root documentation.

### Benchmarks

The first table records GGUF image measurements on 2026-10-02.
Model files and conversion instructions are in
[MODEL_ZOO.md](../../MODEL_ZOO.md). Each image cell reports **warmed full-image
median latency / peak process RSS**. Lower latency is better; GB means
1,000,000,000 bytes. Unmeasured configurations have no inferred numbers.

<table>
  <thead>
    <tr>
      <th rowspan="2" scope="col">Model / GGUF weight storage</th>
      <th colspan="2" scope="colgroup">Apple M4 Pro: 14 CPU cores (10P + 4E), 20 GPU cores, 48 GiB unified memory<br>macOS 27.0 (26A428)</th>
    </tr>
    <tr>
      <th scope="col">CPU: 4 threads<br>GGML 0.25.3, AppleClang 21.0.0</th>
      <th scope="col">Metal: runtime-compiled shaders<br>GGML 0.25.3, macOS SDK 27.0, AppleClang 21.0.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">SAM 3 image / mixed FP16+FP32 GGUF</th>
      <td><strong>38.994 s</strong> / 4.927 GB</td>
      <td><strong>5.576 s</strong> / 2.660 GB</td>
    </tr>
    <tr>
      <th scope="row">SAM 3 image / FP32 weights</th>
      <td><strong>38.986 s</strong> / 4.908 GB</td>
      <td><strong>5.651 s</strong> / 4.249 GB</td>
    </tr>
  </tbody>
</table>

Only CPU and Metal have measured results here. CUDA and other model families
have no benchmark entries. The [64-frame video protocol](#video-64-frame-protocol)
below measures the accepted hybrid default separately. The historical
[short video sample](#short-video-sample) and dump-heavy numerical diagnostics
are not substitutes for that protocol.

#### Precision labels

All table labels describe **weight storage**. The image FP16 file is mixed
FP16/FP32; CPU loading promotes its half values to FP32, while Metal retains
mixed resident weights. Graph activations and the specified dense/attention
operations follow the FP32 [precision contract](../../cmake/patches/README.md), with
FP16 attention masks where the GGML API requires them. An FP16 filename is
not evidence of native FP16 computation or a particular speedup.

The measured video cells all use hybrid weights and the same Meta-preserving
FP16/BF16 boundaries:

| Measured video backend | Resident weights / arithmetic contract | Input and tracking state |
| --- | --- | --- |
| CPU | FP32 resident weights, including promoted stored half values; FP32 graph arithmetic | FP16-rounded normalization; BF16-rounded tracker features in FP32 vectors; BF16 mask-memory cache consumed as FP32 |
| Metal | Mixed FP32/FP16 resident weights; specified matrix/attention arithmetic in FP32 | The same FP16 input and BF16 feature/memory boundaries; no native BF16-kernel throughput claim |

Object pointers/host mask logits remain FP32; emitted binary masks are uint8.
The image benchmark has no temporal BF16 state. Even the separately accepted
F32 **video** file keeps these input/storage boundaries, so its label means
F32 weights, not an end-to-end FP32 video pipeline. Precision/storage changes
require new matching validation and timing evidence; the numbers below are
unchanged measurements of the stated policies. The authoritative tensor/storage
map is in [MODEL_ZOO.md](../../MODEL_ZOO.md#precision-contract).

#### Measurement conditions

- Official SAM 3 checkpoint revision `3c879f39826c281e95690f02c7821c4de09afae7`;
  original SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
  GGUF v3 / SAM schema 1 conversion hashes are
  `cb13ecd5012a2fe19b06d840049be6daa6b177b35256af8c4afeb12125352486` (FP32) and
  `66731fa5def347677f78d7422b81979be0f8e2f7ead941db9a466d1cfa715120` (FP16).
- One RGB image, `truck.jpg`, 1800 x 1200, resized to the model's 1008 x 1008
  input; prompt `truck`, score threshold `0.5`, batch size one.
- One initial pipeline call followed by five measured complete-image calls.
  Each measured call replaces the image and runs text segmentation, including
  preprocessing and postprocessing. Model loading, image decoding, output-file
  writes and repeated-result-cache calls are excluded from the warm latency.
- Release build, native CPU instructions, AppleClang
  `21.0.0` (`clang-2100.3.34.2`), GGML commit
  `353b63b439f27ab2cc19dac97ab1681ba6d2d084`,
  [precision/window patch](../../cmake/patches/README.md)
  `0a0b80dd15c2a8b5a05d148a31e9e53f8df4d0555852ef301da336a9d97b7c48`.
  Apple Accelerate/BLAS is enabled; KleidiAI is disabled.
- CPU uses four threads. Metal also receives a four-thread CPU fallback setting;
  the tested full image graph runs 3,332 Metal nodes for FP32 or 3,342 for FP16,
  zero CPU nodes and six graph partitions. Host preprocessing/postprocessing still run on CPU.
- Fresh processes ran sequentially: FP32/Metal, FP16/Metal, FP32/CPU, FP16/CPU,
  from 03:47 to 03:56 Asia/Shanghai. AC power was attached and the battery was
  not charging (19% to 13%). Other project computation stopped during timing;
  ordinary desktop activity remained. No thermal or performance warning was reported;
  constant CPU/GPU clocks were not verified.

FP16 names the checkpoint storage policy, which preserves selected tensors in
FP32. CPU loading promotes stored FP16 values exactly to FP32; Metal retains
packed FP16 weights and requests precise arithmetic. See the
[precision contract](../../cmake/patches/README.md). Peak RSS is the process high-water
mark across loading and inference. It is not dedicated GPU VRAM, and must not
be added to backend buffer sizes.

These single-image timings do not establish video frame rate. The independent
seven-case numerical suite passed FP32/CPU, FP32/Metal, FP16/CPU and FP16/Metal
acceptance (28/28 cases). CPU promotes FP16 weights to the same F32 execution
representation; its similar FP32/FP16 latency is expected from that policy.
Raw samples, memory figures and provenance are preserved in the
[local acceptance record](20261002-032709-local-model-meta-validation.md).
The local ignored receipts are
`build/fp32-validation/20261002-032709/benchmark-{cpu,metal}-{f32,f16}/results.json`,
`benchmark-summary.json`, `benchmark-conditions.json`, `binary-identities.json`
and `artifact-identities.json` in that run directory. The old FP16 measurements
(60.588 s CPU, 6.556 s Metal) and their receipts remain in the
[GGUF acceptance record](20261001-020507-gguf-conversion-loading.md).

Earlier custom-container measurements and controlled GGML/window comparisons
remain in the [previous performance record](20261001-002850-metal-window-cpu-performance-official-weights.md).
The current table is a new measurement batch, not a controlled before/after
comparison. Power state and measurement conditions differ from the old batch;
these numbers do not establish a software or container speedup.

#### Video: 64-frame protocol

The accepted default `visual-tracker-f32-v1` hybrid artifact was measured on
2026-10-03. Each cell reports **frame median / p95 / peak process RSS**.
All 64 frames run in one fresh process; frames 0–15 warm up the session and
frames 16–63 supply 48 measured samples. CPU and Metal run sequentially with
four threads, `max_objects=8` and no tensor dumps.

<table>
  <thead>
    <tr>
      <th rowspan="2" scope="col">Model / workload</th>
      <th colspan="2" scope="colgroup">Apple M4 Pro: 14 CPU cores (10P + 4E), 20 GPU cores, 48 GiB unified memory<br>macOS 27.0 (26A428)</th>
    </tr>
    <tr>
      <th scope="col">CPU: 4 threads<br>GGML 0.25.3, AppleClang 21.0.0</th>
      <th scope="col">Metal: runtime-compiled shaders<br>GGML 0.25.3, macOS SDK 27.0, AppleClang 21.0.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">SAM 3 hybrid weights / 1 object</th>
      <td><strong>51.102 s</strong> / 52.174 s / 5.183 GB</td>
      <td><strong>7.833 s</strong> / 7.861 s / 4.166 GB</td>
    </tr>
    <tr>
      <th scope="row">SAM 3 hybrid weights / 4 objects</th>
      <td><strong>81.644 s</strong> / 85.855 s / 5.291 GB</td>
      <td><strong>13.356 s</strong> / 13.430 s / 4.295 GB</td>
    </tr>
  </tbody>
</table>

The original checkpoint, pinned GGML revision/patch, Release compiler and
Accelerate settings are the same as the image conditions above. The schema-2
hybrid file is 2,765,012,640 bytes, SHA-256
`3975b4b1a10b962c6fad5022e2798baa6266aad7dbcc39210b459537208e1a05`.
It preserves original FP32 visual/tracker values and mixed detector/text
weights. Each backend first passes all five original-reference video cases
(216 frames), under the unchanged mixed gate. Explicit F16 fails that
eligibility check and has no accepted video performance cell.

The separate deterministic fixtures use 1800 x 1200 RGB canvases, gray value
127, and the pinned `truck.jpg` resized with Pillow bicubic to 800 x 533.
The one-object base is (500,330); the four bases are (40,30), (960,30),
(40,630), (960,630). The second and third copies are mirrored. Motion phases
are 0 for one object and 0/8/16/24 for four; horizontal/vertical offsets are
`2*(16-abs(((frame+phase)%32)-16)-8)` and
`8-abs(((frame+phase)%16)-8)` pixels. Prompt is `truck`. The original Meta
modules qualify a 17-frame prefix with declared length 64: exactly one/four
nonempty masks, stable IDs and no later birth/removal. The benchmark then
checks those counts and IDs on all 64 actual C++ frames. No object cap is used
to conceal additional detections. The frozen five-case correctness corpus
is unchanged.

Every frame/stage sample is retained. `frame_ms` includes preprocessing,
inference and output emission inside `push_frame`; PNG decoding and file
writes are outside it. The available stage fields are tracker and memory;
there are no separately exposed visual-trunk/detector timings. Median and
linearly interpolated p95 use the 48 samples (percentile position
`(n-1)*0.95`). Frame 63 drains 15 delayed results and remains in the sample:
7.956/13.754 seconds on Metal and 52.096/86.152 seconds on CPU for one/four
objects. These are finite-sequence measurements, not an indefinite stream.

| Backend / objects | Tracker median / p95 (s) | Memory median / p95 (s) | Model load (s) | First output observed (s) |
| --- | ---: | ---: | ---: | ---: |
| Metal / 1 | 1.796 / 1.819 | 0.039 / 0.040 | 0.942 | 108.750 |
| Metal / 4 | 7.181 / 7.247 | 0.160 / 0.161 | 0.902 | 158.408 |
| CPU / 1 | 10.042 / 11.098 | 0.160 / 0.161 | 0.991 | 709.934 |
| CPU / 4 | 40.092 / 44.275 | 0.645 / 0.650 | 0.982 | 993.105 |

First output is frame 0 after processing frame 14 in every cell. Observation
is sampled about once per second from process launch and includes model load,
decode and file work; it is separate from summed frame time. Each process
encodes text once, runs vision 64 times and emits all 64 frames. CPU cells
execute zero Metal nodes and Metal cells execute zero CPU graph nodes.

| Backend / objects | Current RSS after warmup, min / median / max (GB) | Weight buffers (bytes) | Graph allocation high-water (bytes) | Max retained records / bytes |
| --- | ---: | ---: | ---: | ---: |
| Metal / 1 | 3.270 / 3.478 / 4.166 | 2,763,224,992 | 1,245,268,288 | 27 / 17,943,552 |
| Metal / 4 | 3.424 / 3.681 / 4.295 | 2,763,224,992 | 1,245,268,288 | 108 / 71,774,208 |
| CPU / 1 | 3.924 / 4.268 / 5.118 | 3,447,558,112 | 1,020,660,736 | 27 / 17,943,552 |
| CPU / 4 | 4.091 / 4.426 / 5.246 | 3,447,558,112 | 1,020,660,736 | 108 / 71,774,208 |

Current RSS comes from asynchronous `ps` samples tagged with the latest
completed frame, using tags 16–62. Peak RSS in the main table is the independent
`/usr/bin/time -l` process high-water mark, including load and final drain;
it need not equal the maximum sampled current RSS. Weight buffers and graph
high-water remain constant after warmup. Retained state respects the 27-record
per-object bound. These observations do not establish a universal process-RSS
plateau; allocation counters must not be added to RSS.

Runs occurred sequentially from 05:43 to 08:21 Asia/Shanghai, with AC power,
100% battery at every start/end, no reported thermal/performance warning and
no system sleep event during any cell. Task-scoped `caffeinate` prevented idle
sleep without changing system settings. No other SAM/Meta inference or
compilation overlapped; ordinary desktop services remained active. Clocks
were not forced. The earlier sleeping, dump-heavy CPU correctness run is
excluded from all performance numbers.

`tools/benchmark_video.py` requires a passing full numerical receipt and the
original-module fixture qualification, verifies source/model/binary/library/
input/output hashes, and refuses malformed frame counts, IDs, fallback or
allocation evidence. The local receipt bundle is
`build/video-validation/20261002-182848/`: `hybrid-performance-matrix.json`,
`performance-values.json`, `prepare_performance_fixtures.py`,
`qualify_performance_prefix.py`, `{one,four}-object-oracle-prefix.json` and
`run_benchmarks.py`. Raw 64-frame samples and RSS streams are in
`/private/tmp/sam-video-20261002-182848/benchmark-hybrid-{metal,cpu}-{one-object,four-object}/`.
The fixture manifest SHA-256 is
`68d7b5574a69d015256c53265e7be1e9e258020e3694f3a0a38e7a823003c37f`.
Models, media and generated receipts stay outside Git.

For a local rerun with that verified input/qualification bundle, choose a fresh
output and run this command for each backend/workload sequentially (change all
matching backend/workload arguments):

```sh
build/reference-runtime/venv/bin/python tools/benchmark_video.py \
  --build-dir /private/tmp/sam-video-20261002-133548/metal \
  --model models/sam3-video-hybrid-v1.gguf \
  --reference /private/tmp/sam-video-20261002-133548/reference-full \
  --validation /private/tmp/sam-video-20261002-133548/validation-metal-hybrid-full/metrics.json \
  --fixture-manifest /private/tmp/sam-video-20261002-182848/performance-fixtures/fixture-manifest.json \
  --qualification build/video-validation/20261002-182848/one-object-oracle-prefix.json \
  --recipe-script build/video-validation/20261002-182848/prepare_performance_fixtures.py \
  --workload one-object --backend metal --threads 4 \
  --output /private/tmp/sam-video-benchmark-metal-one-64-rerun
```

New fixtures require their own original-module prefix qualification before
measurement. Preserve AC/sleep conditions and keep other inference stopped;
the local serial wrapper records these conditions and rejects contaminated
cells. See the [completion record](20261002-182848-m2-complete-acceptance.md)
for numerical and lifetime evidence.

#### Short video sample

The explicit `visual-tracker-f32-v1` hybrid video artifact was measured on the
same M4 Pro, macOS and toolchain described above. It passes the original
five-case/216-frame video suite and seven image-regression cases on both CPU
and Metal. The checkpoint is the same original Meta file; GGUF SHA-256 is
`3975b4b1a10b962c6fad5022e2798baa6266aad7dbcc39210b459537208e1a05`
(2,765,012,640 bytes, schema 2, original-FP32 visual/tracker plus mixed detector/text).

Each backend ran once in a fresh process with threads=4, max_objects=8 and no
tensor dumps, using the first three frozen `motion` PNGs (1800 x 1200, prompt
`truck`). The declared sequence length is **3**, including its normal pointer
temporal normalization. Frame 0 initializes the sequence; frames 1 and 2 are
propagated-frame samples with growing memory. Frame 2 also resizes/drains all
three delayed outputs. Its cost is included below. These two propagated samples
are not steady-state throughput or the full M2 performance protocol.

| Backend | Frame 0 / initialization (s) | Frame 1 (s) | Frame 2 + drain (s) | Frames 1–2 median (s) | Model load (s) | Peak process RSS (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Metal | 6.174 | 6.529 | 6.705 | 6.617 | 0.921 | 3.808 |
| CPU | 41.438 | 42.925 | 43.809 | 43.367 | 0.958 | 5.170 |

Per-frame times are `VideoStats::frame_ms`, including preprocessing, inference
and any output emission inside `push_frame`. PNG decoding and file writes are
outside these samples. `/usr/bin/time -l` measured complete process wall times
of 20.43 s (Metal) and 129.24 s (CPU), and peak RSS of 3,807,756,288 and
5,170,331,648 bytes. The runtime's RSS counters agree with those process peaks.

| Backend | Weight buffers (bytes) | Maximum graph compute buffers (bytes) | Retained memory after frame 2 (bytes) | CPU / Metal graph nodes |
| --- | ---: | ---: | ---: | ---: |
| Metal | 2,763,224,992 | 1,245,268,288 | 1,993,728 | 0 / 14,111 |
| CPU | 3,447,558,112 | 1,020,660,736 | 1,993,728 | 14,081 / 0 |

Both runs retain 1, 2 and 3 memory records across the three frames: 664,576,
1,329,152 and 1,993,728 bytes. Pending outputs are 1, 2 and 0 after each call
(high-water 3); all three frames are emitted at the end. Weight/compute/state
counters describe different allocations and must not be added to process RSS.
CPU promotes stored F16 values to F32; the hybrid storage label does not imply
F16 CPU execution. Neither run has unexpected backend fallback.

Metal ran from 18:12:56 to 18:13:17 and CPU from 18:13:52 to 18:16:01
Asia/Shanghai on 2026-10-02, after the validation batches. No validation or other
SAM inference overlapped either measurement. AC power was attached, battery
100%, and no thermal/performance warning was recorded. Ordinary desktop and
macOS background services remained active; there was no forced cool-down or
clock control, and OS/shader caches could already be warm. This sample supplies
measured latency and memory for the stated short workload. The completed
64-frame protocol above supplies the M2 performance evidence; these earlier
three-frame samples remain separately scoped.

Reproduce the workload after generating the frozen motion recipe and converting
the hybrid model as described in the README. Use fresh directories and run the
two commands sequentially:

```sh
mkdir models/video-cases/motion-first3
cp models/video-cases/motion/000000.png models/video-cases/motion-first3/
cp models/video-cases/motion/000001.png models/video-cases/motion-first3/
cp models/video-cases/motion/000002.png models/video-cases/motion-first3/
/usr/bin/time -l build/metal/examples/sam_video \
  --model models/sam3-video-hybrid-v1.gguf --frames models/video-cases/motion-first3 \
  --text truck --backend metal --threads 4 --max-objects 8 \
  --output /private/tmp/sam-video-benchmark-metal-3frames
/usr/bin/time -l build/cpu/examples/sam_video \
  --model models/sam3-video-hybrid-v1.gguf --frames models/video-cases/motion-first3 \
  --text truck --backend cpu --threads 4 --max-objects 8 \
  --output /private/tmp/sam-video-benchmark-cpu-3frames
```

The local ignored receipts are in `build/video-validation/20261002-133548/`:
`benchmark-{metal,cpu}-hybrid-3frames.json`, their raw `/usr/bin/time -l` logs,
`benchmark-summary.json` and `benchmark_hybrid_three_frames.py`. They record all
samples, conditions, fixed input hashes and model/binary/library/output hashes.
The independent fixture is
`/private/tmp/sam-video-20261002-133548/benchmark-motion-3frames/fixture-manifest.json`.

#### Reproduce

Run from the repository root after following [model conversion](../models/sam3-details.md#convert-the-supported-runtime-files)
and the [CPU/Metal build instructions](../../README.md#build). Prepare the pinned input:

```sh
mkdir -p models/fixtures
curl -fL https://raw.githubusercontent.com/facebookresearch/sam3/2345a4ad109ac29c569da749c91d84f10dc08c40/assets/images/truck.jpg \
  -o models/fixtures/truck.jpg
shasum -a 256 models/fixtures/truck.jpg
```

Expected image SHA-256:
`941715e721c8864324a1425b445ea4dde0498b995c45ddce0141a58971c6ff99`.
Choose new output directories on a local, unsynchronized disk; the CLI refuses
to overwrite existing directories. Run the following commands sequentially:

```sh
sam_benchmark_run="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
for sam_benchmark_backend in metal cpu; do
  for sam_benchmark_precision in f32 f16; do
    "build/$sam_benchmark_backend/examples/sam_image" \
      --model "models/sam3-$sam_benchmark_precision.gguf" \
      --image models/fixtures/truck.jpg --text truck \
      --backend "$sam_benchmark_backend" --threads 4 --score-threshold 0.5 \
      --repeat 5 \
      --output "/private/tmp/sam-benchmark-$sam_benchmark_run-$sam_benchmark_backend-$sam_benchmark_precision" || exit
  done
done
```

Read `timing_ms.warmed_full_image_median` (milliseconds) and
`runtime.process_peak_rss_bytes` from each `results.json` for the table.
Retain `warmed_full_image_runs` and `repeated_result_cache_runs` for inspection.
The inference stage includes text encoding; do not add those stage times twice.
`cold_start` includes initial loading/decoding but excludes process startup and
may reuse OS/shader caches. Cache timing measures reuse of the previous result.

#### Add a result

Add a model/checkpoint/precision row and a hardware column group using the same
two header rows. Record exact CPU/GPU variant, RAM/VRAM, OS, backend, compiler,
GGML revision/patch and relevant backend versions. A CUDA column must name the
actual GPU, CUDA Toolkit and NVIDIA driver versions; add cuDNN only if used.
Record the checkpoint hash, input/workload, thread count, warmup/repeats,
individual samples, peak-memory definition, power state and numerical acceptance.
Keep different workloads in separate tables and mark unsupported or unmeasured
combinations explicitly.

#### Encoding diagnosis after the baseline

A separate three-frame early sequence used the new runtime JSON timing output
on Metal with the same hybrid weights and 1/4 objects. Both endpoints were
Battery Power; declared length3 changes the early memory/pointer horizon.
The two propagated samples include final draining. These are diagnostic values,
not a new 64-frame benchmark or a before/after speedup comparison.

| Objects | Frame median (s) | Image encoding (s) | Detector pipeline (s) | Tracker (s) | Memory (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.894 | 9.198 | 0.905 | 0.671 | 0.048 |
| 4 | 14.108 | 9.673 | 0.918 | 3.160 | 0.248 |

Image encoding includes ViT, both necks, geometry, allocation and host transfers;
the detector pipeline includes prompt preparation, fusion, detection and masks.
The next profiling target is inside image encoding, which dominates these
early samples. No kernel/storage/hotstart change was made. The guarded receipt
is `build/p1-p2/20261003/encoding-profile.json`.
