# SAM 3 RoPE vector layout optimization

Created: 2026-10-04 18:27:16 Asia/Shanghai.
Status: fresh builds, 43 CTests, 65 tool checks, six complete-graph bitwise
parity probes, all 140 original-reference image cases and all 864 video frames
passed. Four default full-image A/B/A cells passed and demonstrate benefit.
The layout change is adopted and final documentation/whitespace checks passed.
Completed locally; no commit or push was requested for this optimization.

## Scope and baseline

The user requested continuing after the completed operator assessment.
Implement the highest-priority concrete target from the
[profiling plan](20261004-164643-sam3-operator-profiling-and-bottleneck-assessment.md):
SAM 3 vision RoPE's first-dimension-1 real/imag slices. Baseline is local commit
`434cdb52b8bfcee1a8c483366234e1cef461c7c7`. Preserve the existing uncommitted
profiling plan, unrelated cleanup plans and archive-retirement metadata.
No commit or push is implied for the new optimization.

Preserve C++17/header-only integration, checkpoint frequencies, adjacent
real/imag pairing, arithmetic/state precision, public APIs, model temporal
rules and all existing acceptance gates. Keep the optimization in the SAM 3
graph adapter and use shared GGML operations across CPU/Metal. This experiment
does not add a model/backend or change the repository's platform goals.

## Approach

Try the smallest graph-only route before adding a compiled operator: permute
and pack the complex pair axis so each real/imag channel vector is contiguous
and has width `head_dim/2` (32 for this model). Keep the same four multiplies,
subtract and add using checkpoint cos/sin tables, then restore adjacent pairs
and the original `[head_dim,tokens,heads*batch]` output layout. Account for the
input/frequency packing and output copies and their live workspace. Do not
replace the frequency table with regenerated phases or NeoX split-half pairs.

The prior serialized Metal trace shows about 55% of GPU intervals in these
RoPE arithmetic nodes, but it does not predict default full-graph speedup.
Default fusion/concurrency/graph optimization and no observer must be used for
candidate performance. If this graph route does not deliver a useful benefit
within memory/numerical constraints, reject it and reassess the fused-kernel
candidate separately; do not ship a regression for an attractive microbenchmark.

## Steps

1. Freeze an uninstrumented baseline source/binary/library identity and create
   new ignored artifacts under `build/rope-optimization/20261004/`. Existing
   profiling/validation artifacts are immutable. Verify retained model/reference
   paths after concurrent cleanup rather than trusting older receipts alone.
2. Add focused Release-active tests against an independent scalar complex
   multiplication formula: different frequencies per token/pair, several heads
   and batches, identity/quarter-turn, non-FP16-representable data, tiny/tail and
   real window/global shapes, repeated fresh inputs and frequency broadcasts.
3. Implement only the layout change in `vision.hpp`; keep operations/rounding
   and ownership fixed. Run CPU/Metal checks, independent/repeated headers and
   two-TU linkage. Check native CPU arithmetic on the focused regression too.
4. Compare complete RoPE graphs on representative window/global shapes with
   the baseline, including every packing copy and actual workspace/RSS.
   Compare outputs first; formal timing is sequential under AC without other
   builds/tests/model runs.
5. Qualify the frozen implementation on original-reference image presets and
   affected F32/hybrid CPU/Metal video cases with unchanged gates. Reuse the
   original oracle, not changed-code generated outputs. Source changes invalidate
   executable-dependent acceptance and require fresh receipts.
6. Once the candidate is stable and numerically qualified, run uninstrumented
   full-image baseline/candidate/baseline timing with matched inputs, thread
   request, backend and power conditions. Pilot F32 and full Q4 on CPU/BLAS and
   Metal, extend supported storage results as justified. Separate cold/setup
   from five uncached warm calls and exclude result-cache hits. Quantization
   precision/storage is fixed for each comparison.
7. Adopt only with demonstrated end-to-end benefit and bounded memory/no
   material CPU regression. Record implementation, checks, failed attempts,
   raw evidence identities and limits here; update Unreleased impact and concise
   user performance results only after actual completion.

## Verification and resource boundary

The original `sam3.pt` exports, image oracle, isolated Python, video F32/hybrid
models and official video oracle are currently present. The cleanup retirement
metadata also names a retained official video reference under
`models/reference/official-sam3-video-20261004-174839`; verify its manifest/hash
before using it. Retired historical byte bundles are not a fresh acceptance
source. Preserve raw traces and receipts and bound storage with the existing
validated artifact freezing/deduplication protocol where appropriate.

Validate pair ordering, token/head/batch isolation, actual shader/layout
selection, output ownership, additional copies, peak scheduler arena and process
RSS separately. GPU node counts/fallback must remain consistent with supported
profiles. A smaller per-op time or changed diagnostic setting is insufficient
for a production speed claim. No numerical gate is relaxed for this patch.

## Results

The implementation packs `[2,half,tokens,heads*batch]` into contiguous
`[half,2,tokens,heads*batch]` before slicing and arithmetic, and packs the
checkpoint frequency table in the same way. Real/imag inputs are
`[half,1,tokens,heads*batch]`, frequencies `[half,1,tokens,1]`. Concatenation on
the pair axis, inverse permutation and a final contiguous copy restore the
original adjacent-pair layout. No new operator, arithmetic type, regenerated
frequency or public backend policy was introduced.

Independent source review checked the axis mapping, offsets (`4*half` bytes
for the second F32 plane), frequency head/batch broadcast and restored layout.
An inaccurate frequency-view comment was corrected before source freeze.
The SAM 3 tracking memory RoPE is a separate function and was not changed;
video shares the optimized vision encoder.

Fresh Release CPU/BLAS and native CPU each passed 14/14 CTests; Metal passed
15/15, including explicit `sam_rope_metal` with actual Metal compute and zero
CPU fallback. Full independent/repeated-header and two-TU builds succeeded;
the isolated reference environment passed 65 tool checks. The new regression
uses an independent scalar formula with non-FP16-representable data, varying
per-token/pair frequency values, several heads/batches, identity/quarter-turn,
single token, tail, window/global counts and fresh repeated graph inputs.

The isolated comparison helper includes the unchanged `434cdb5` legacy graph
and calls the production candidate. Its first CPU window invocation stopped in
fixture setup before graph computation: it incorrectly interpreted
`ggml_backend_sched_get_n_copies()==1` as a performed cross-backend copy. The
pinned implementation returns pipeline copy-slot capacity (serial is always
one). Preserve the initial helper/result; v2 names this field
`scheduler_copy_slots`, requires one serial slot and one scheduler backend,
and retains the actual node/buffer-placement checks. Production source and
numerical gates did not change. The six CPU/Metal × window/global/tail parity
probes run in a new `helper-parity-v2/` directory before timing.

The Metal v2 helper exposed a second setup issue: GGML requires CPU last in the
scheduler list even if no compute node falls back. Its one-backend Metal list
asserted before graph comparison. Preserve that failed setup; v3 supplies the
required CPU terminal backend and still rejects any actual CPU/other compute
placement. It also excludes logical views from physical-compute accounting.
All six shapes were rechecked under the same v3 helper in `helper-parity-v3/`.

CPU and Metal window/global/tail all passed exact old/new equality (zero max
absolute/relative error, no nonfinite or over-tolerance values). Window/global
each compare 5,308,416 F32 outputs. Each graph has 8 baseline versus 10 candidate
physical operations, one serial copy slot and one split; all three Metal
shapes run on `MTL0` with zero CPU fallback. Actual arena sizes in bytes:

| Shape | CPU baseline / candidate | Metal baseline / candidate |
| --- | ---: | ---: |
| Window | 42,467,328 / 53,231,616 | 53,084,160 / 63,848,448 |
| Global | 42,467,328 / 54,411,264 | 53,084,160 / 65,028,096 |
| Tail | 516,096 / 646,912 | 645,120 / 775,936 |

The two representative shapes add about 10.3–11.4 MiB to the isolated arena.
Whole-process RSS includes both graphs and their shared fixture, so these
probe snapshots cannot establish a candidate-only process RSS difference.
Default fusion/concurrency/graph optimization remain enabled, without an
operator observer. CPU v2 data was only a preliminary helper version; the
consistent v3 rows above supersede it for the comparison.

The retained official video oracle passed all 1,265 manifest-listed file
hashes and matches the old reference manifest identity; its logical bytes
are 6,877,797,588. All seven original image input/tensor/result metadata hashes
were checked too. `oracle-integrity-v1.json` records this read-only preflight;
it does not constitute execution of the new model binaries. Production source
still matches the 98-file build snapshot while the v3 diagnostic micrograph
timing queue runs exclusively.

The v3 helper completed six guarded micrograph timing cells, each with a cold
A1/B/A2 triplet, one unmeasured warm triplet and five matched warm triplets.
Compute wall includes every packing/arithmetic/concat/cont operation plus final
synchronization; input upload, fixture and graph setup are separate. No observer
or disabled Metal execution controls are used. Current evidence is
`helper-timing-v3/helper-timing-v3-report.json` under the ignored rope directory.
Source/build/helper/library/checkpoint identity and AC/thermal/exclusive/sleep
checks passed. Process RSS is jointly resident A+B graph arenas and fixture,
so no candidate-only RSS claim is made.

| Backend / shape | Baseline warm median ms | Candidate warm median ms | Candidate / baseline |
| --- | ---: | ---: | ---: |
| CPU window | 16.973 | 10.658 | 0.628 |
| CPU global | 17.237 | 11.049 | 0.641 |
| CPU tail | 0.2559 | 0.1692 | 0.661 |
| Metal window | 47.274 | 4.444 | 0.0940 |
| Metal global | 47.263 | 4.494 | 0.0951 |
| Metal tail | 0.8224 | 0.2405 | 0.2925 |

Window/global brackets were stable within about 2.8%; those representative
graphs improve about 36–37% on CPU and 90.5–90.6% on Metal. Tiny-tail sampling
was noisy: the first measured Metal bracket drift was -60.14%, and CPU +16.97%;
later triplets were more stable. Tail ratios remain diagnostic, not adoption
evidence. These results justify retaining the layout candidate for full
qualification, not a whole-model acceleration claim.

The next queue is original-reference image CPU/Metal × ten storage profiles ×
seven cases (140 cases), and video CPU/Metal × F32/hybrid × five scenarios
totaling 216 frames per cell (864 frames). Independent numerical processes may
run with bounded memory; their wall times are not performance data. Gates,
temporal/state policies, oracle identities and source remain fixed. Only after
all numerical processes exit will default, uninstrumented full-image A/B/A
measurement begin in new directories.

### Frozen preliminary evidence

Paths below are relative to the ignored `build/rope-optimization/20261004/`
directory. Each record remains immutable; later numerical/performance phases
write separate records rather than upgrading these earlier-phase receipts.

| Record | SHA-256 |
| --- | --- |
| `candidate-inventory-rope-v1.json` | `11d57a3ea7269d1752b4800f9acaa816edae1c084d39004493bed46e00e14e25` |
| `build-test-receipt-rope-v1.json` | `d406551a6a77d046402bb8be9fb736a3988cb11ff866c39e61394cf998a31d24` |
| `oracle-integrity-v1.json` | `cd0aaec30ee9ecb0c42ae6e649e6b508a7ba6cb47f2f304c5d00407580727498` |
| `helper-parity-v3-report.json` | `2f98a4e36b9d18a9720529f5822ea52ec0f99ce57dfb8de5a7cbd57746064c0a` |
| `helper-timing-v3/helper-timing-v3-report.json` | `a59551d4926347efc438aff966abfb4409fa4d2d4f19202eea0fc7f9e6ff32f2` |

The proposed full-image comparison covers four matched cells: F32 and
full-component Q4_K, each on CPU/BLAS and Metal. It does not refresh the prior
30-cell table, especially its native CPU measurements. If the four cells qualify,
either publish this explicitly measured subset or extend the candidate matrix
before replacing that table. Do not combine new and old executable results into
one current-performance snapshot. Tracker propagation timing excludes vision
encoding, so this change does not require rerunning that separate stage benchmark.

### Original-reference image qualification

The two new image indices completed at 19:16:20 (Metal) and 19:21:30 (CPU)
Asia/Shanghai. Each covers seven original cases for F32, mixed F16/F32, all
four vision presets and all four full-component presets. All 20 cells passed
their unchanged release/output gates, totaling 140/140 cases. Every Metal case
has zero CPU compute nodes. Quantized tensor-fidelity gates remain a separate
diagnostic and do not determine output-quality acceptance; their failed
diagnostic comparisons are retained in the per-cell metrics.

| Record under `original-num-runs-v1/` | SHA-256 |
| --- | --- |
| `image-cpu-matrix.json` | `c04788271f3748317d992c4c37eaa2b22a18944ea3794fe119a089985088e641` |
| `image-metal-matrix.json` | `ba022d58e568f78d1db31fdb53ea9880de32b94962777d5bdc7d3b6b62d67a50` |

The indices bind all 98 frozen production/build/test paths plus the numerical
runner, driver plan and inventory (101 paths). A read-only reconciliation found
no missing or changed production entries and no live source drift. Candidate
binary/library/model identities are bound separately in every cell. Numerical
queues ran concurrently by backend with bounded memory; all wall times and RSS
in this phase remain correctness diagnostics rather than formal performance.

### Video environment interruption and retry

The first video queue began with one F32 process per backend. With 69% memory
free and observed peaks of about 5.55 GB (CPU) and 4.93 GB (Metal), it was
expanded to at most four independent correctness processes, adding hybrid
through existing one-cell label filters. Output/index paths were distinct;
the frozen runner and driver were not modified. The four-process snapshot
still had 50% memory free and about 14.4 GiB total observed process RSS.

During the separate user-authorized iCloud Desktop/Documents migration,
macOS briefly removed the canonical Documents path before local contents were
restored. All four numerical runners exited on missing source paths during
artifact verification. Their incomplete indices and partial outputs in
`original-num-runs-v1/` are preserved; none is a complete video pass or a
numerical failure. No performance samples were collected in that phase.

After the directory returned, read-only reconciliation verified unchanged
production sources, inventory, runner, driver, completed image indices and
video reference manifest/payloads. There were no remaining model/build/test
processes. The separate migration record then reported completed local
restoration, and a process check found no active relocation/copy work. A new
video-only queue is authorized in `original-num-runs-v2/`, with at most four
independent correctness processes and fresh runner/driver identities where
needed to select the new output root. Existing gates and original oracle are
unchanged. Full-image timing still requires all 864 video frames to qualify
and all correctness processes to exit.

### Completed original-reference video qualification

All four v2 selected indices completed and passed. F32 and hybrid each pass
the original CPU/Metal reference across motion (48 frames), entry (64),
occlusion (64), hotstart/removal (24) and negative prompt (16): 216 frames per
cell, 864/864 overall. Each metrics file reports five passing eligible cases.
Root checked index completion, metrics hashes, eligibility, case counts and
trace frame totals separately from the worker's artifact/source reconciliation.
The user also confirmed the directory migration was completed and stable.
No acceptance gate, checkpoint frequency or precision policy was changed.

| Record under `original-num-runs-v2/` | SHA-256 |
| --- | --- |
| `video-cpu-selected-video-f32-cpu.json` | `93262c6f04c4230d4a862ce667c9b2b0063f57cbd7a36b57e7f3043c14df3481` |
| `video-cpu-selected-video-hybrid-cpu.json` | `02c0d89ce41bc7c4af63647f6ce9d34c83f3b8f157f53070d35bb99f9bf60014` |
| `video-metal-selected-video-f32-metal.json` | `c0b8aa4dc32dd14632cf21579eae5bb9ff43b667241b9d4ec98f1bf7519f4538` |
| `video-metal-selected-video-hybrid-metal.json` | `0160c376995f46716cf7613a7db30b66983c0c31b8e1097cc0aaf82d38a3afe2` |

The performance driver is being reviewed before its first execution. Static
review identified interface mismatches: the original image validator runs
`test_image`, not `sam_image`; F32 storage profile may be null; an error message
used an undefined backend name; inherited Metal diagnostic controls need to
be cleared and recorded. These are fixture/preflight issues, not production
source changes or relaxed validation. Final timing must bind the CLI through
the same frozen build inventory and freshly check its output against the oracle.

The final candidate numerical summary is
`original-num-runs-v2/original-num-summary-v2.json`, SHA-256
`0962647620cd93ac41841666b97f66a356bdafd2f2d3dbd45ef0580865a77472`.
It reconciles the 140 image cases and 864 video frames, all 101 live source
guard entries and 33 unique video execution-artifact paths. All four numerical
processes have exited. Immutable earlier records were not upgraded or overwritten.

### Default full-image performance preparation

Further static review requires clearing and recording inherited Metal diagnostic
controls, verifying embedded Metal shader libraries, and binding the baseline
executable/libraries to a source/config build receipt. A1/A2 file equality alone
does not establish the binary's build provenance. The chosen baseline strategy
is an isolated fresh build from the exact `434cdb5` archive with pinned original
GGML and the existing precision patch. Do not reuse the profiling shadow tree
or its diagnostic libraries. New baseline F32/full-Q4_K CPU/Metal qualification
will cover all seven original image cases (28 additional baseline cases) before
timing. These are separate from the candidate's 140 cases.

Each pilot cell runs A1/B/A2 in three independent processes, each with one cold
call and five uncached warm calls. Pair A1/A2 by warm-sample index, average each
pair, then take the median of the five averages; compare it with the median of
B's five samples. Record cold/loading, excluded cache replays and each arm's
whole-process RSS separately. Require default fusion/graph optimization,
matching source/model/input/build identities, original output-quality checks,
AC/thermal/exclusive/sleep guards and embedded shader-resource identity. No
formal whole-image sample has been collected yet.

Fresh isolated baseline CPU/BLAS and embedded-Metal Release builds completed.
`baseline-434/build-receipt-v2.json` binds the normalized 225-file archive tree,
toolchain/configuration, 19 executables per build, 12/15 libraries, prepared GGML
provenance/fingerprints and logs. Its SHA-256 is
`20221d5bbf27976625b76154ea40d0365350b058ff52db027387f2285eeaad5b`.
The initial configure pointed at the archive parent instead of its
`source-434/` root and failed before compilation; its log is preserved. Archive
comparison now normalizes that recorded prefix before exact-file verification.

The baseline numerical wrapper also stopped twice in preflight before any model
started: importing the baseline helper wrote one Python bytecode cache file into
the otherwise exact archive tree; a later attempt referenced outdated build
receipt field names. Only the wrapper-created cache was removed, bytecode
writing was disabled, and schema references were corrected. Separate failure
records remain preserved, including `preflight-failure-v2.json` with SHA-256
`76ec8159b2328872b340fd507718a5909f620e384db88d61051a484ce3b3951a`.
These setup errors did not change production source, reference gates, model
outputs or any measured performance result. Full interface/identity preflight
must pass before starting baseline qualification or timing.

### Final baseline qualification and pilot freeze

Additional wrapper field mismatches occurred after complete F32/CPU numerical
output, including passing a build record to the archive checker and expecting
an executable `name` instead of its `path`. Failed qualification v1/v2/v3
indices and outputs are retained. Rather than repeatedly recomputing unchanged
inference solely for receipt formatting, a separate read-only reconciliation
validated the complete v2 F32/CPU metrics, all seven output maps, 15 runtime
artifacts, original gates/reference and the exact build/source identity. The
original `run.json` did not persist return code; successful validator completion
is supported by the complete passing metrics and wrapper control flow reaching
postflight after its return-code check. This limitation is explicit in the audit.
The other three baseline cells were then freshly validated in qualification v4.
No old failure record or output was altered, and no quality gate was regraded.

Baseline qualification is now four cells / 28 passing original cases, with no
active numerical processes. Final baseline index
`baseline-434/baseline-image-qualification-final-v1.json` SHA-256 is
`54d5863616515e5d9d7737dd2d40eaaab52d69bdf42e92f0525d28b40ca8ac2c`;
the F32/CPU reconciliation is `baseline-434/qualification-reconciliation-v1.json`,
SHA-256 `3a0eb1449d20673731e832fc0e1417193336204d1e573f733305a03cfb2d6bd7`.

Final static review caught a stale `baseline` variable in the execution branch
and a stdout-only 210 versus 140 case-count typo before sampling. v1 script and
preflight remain preserved. v3 corrected those and recognized its own filename
in the competing-process guard. A final v4 closes the full-corpus binding by
requiring the accepted NUM-summary hash, each of its two image and four video
index hashes, exact labels/metrics and per-scenario frame counts. This prevents
counting repeated or incorrectly selected cells as a complete matrix. Standard
library symbol-table scanning reports no unresolved global references.

The authorized pilot uses `run_image_rope_aba_v4.py`, SHA-256
`14b499d77cabe4e40a1ccb0ebcd65827af06aebf756fbd05ea49cdb7e70d706e`.
Its no-model preflight is `image-full-aba-preflight-v4.json`, SHA-256
`520a940df3e7b33acbd1c56b514575c651159b76e869875c6d6bf5ebfbc27ba3`.
Both bind candidate 140 image / 864 video, baseline 28 image, input/oracle/gates,
exact build/source/library identity and default execution controls. All earlier
preflights remain unchanged. Output is the new `image-full-aba-v4/` directory;
only its twelve serial processes are authorized now. No other builds, tests or
model workloads may overlap its formal timing. Root edits only documentation
and reads lightweight metadata until the pilot completes.

### Completed default full-image comparison and adoption

All twelve serial arms completed and passed their output-quality, source,
build/library/model/input, AC/thermal/exclusive/sleep and output-file checks.
Root independently recomputed paired baseline medians and candidate medians,
confirmed five uncached warm samples per arm, before/after artifact-map equality,
all quality flags and empty competing-process/sleep records. No other model
workers remain. The immutable `image-full-aba-v4/report.json` SHA-256 is
`7970296b876990b41545568c0655517624248e5d1a08ac120dab5c1e8ff0147f`.

| Backend / weights | Paired baseline median s | Candidate median s | Latency reduction | RSS A1 / B / A2, decimal GB |
| --- | ---: | ---: | ---: | ---: |
| CPU/BLAS F32 | 7.490894 | 7.059693 | 5.7563% | 5.065114 / 5.023023 / 5.064180 |
| CPU/BLAS full Q4_K | 7.543906 | 7.085760 | 6.0731% | 2.709766 / 2.714649 / 2.695299 |
| Metal F32 | 5.472881 | 2.706973 | 50.5384% | 4.349673 / 4.363878 / 4.362977 |
| Metal full Q4_K | 5.497197 | 2.671081 | 51.4101% | 1.773732 / 1.774518 / 1.772028 |

Each cell has five of five matched warm comparisons favoring the candidate.
A2/A1 warm-median drift was +1.568%, +0.763%, -0.017% and -1.346%, respectively.
The full-image compute arena is identical in all A1/B/A2 arms: 961,062,048 bytes
on CPU/BLAS and 1,242,646,848 on Metal. That whole-graph result supersedes any
attempt to extrapolate the isolated RoPE arena increase to whole-process memory.
F32/CPU RSS falls; the largest increase versus either bracket arm is about
19.35 MB for full-Q4 CPU, below 0.8% of that process's peak. The implementation
delivers end-to-end benefit with bounded memory and no CPU regression.

The receipt field `cold_full_image_ms` preserves the CLI's `cold_start` value:
decode + model load + session setup + first segmentation. It is not pure cold
inference. In particular, Metal F32 A1 load was 17.369 seconds versus B's 1.117;
the cause is not established and the difference is not a RoPE speedup claim.
Only the five warmed full-image calls support the percentages above; loading,
decode, final file output and cache hits are excluded from those latency samples.

Adopt the shared-GGML layout change. This round's four measured configurations
are sufficient for its scope; no additional sixteen-profile or native-CPU full
image timing was run. All ten CPU/Metal image storage profiles retain current
numerical qualification, but their unmeasured latencies are not inferred from
these four cells. The bilingual performance guides show only this current
four-cell comparison. The complete prior thirty-cell table is retained in the
pipeline plan, without mixing its old executable's results into the new table.
Tracker propagation excludes vision encoding and remains a separate prior
stage measurement. Weight/arithmetic policies, public API and supported-model
boundaries are unchanged. Changelog records the completed user-visible impact.

### Final verification and delivery

| Check | Result |
| --- | --- |
| Fresh Release builds, independent/repeated headers and two-TU linkage | CPU/BLAS, native CPU, Metal passed |
| Release CTest | 14 + 14 + 15 = 43/43 |
| Isolated reference tool checks | 65/65 |
| Complete RoPE legacy/candidate parity | 6/6, bitwise exact |
| Candidate original image output quality | 20 cells, 140/140 cases |
| Candidate original video | Four cells, five scenarios each, 864/864 frames |
| Isolated baseline original image qualification | Four cells, 28/28 cases; F32 CPU uses disclosed read-only reconciliation |
| Default full-image A/B/A | Four cells, twelve arms, all guards/quality passed |
| Documentation links and bilingual tables | 52 documents passed |
| Final live production/build/test fingerprint | All 98 paths match frozen inventory |
| Whitespace | `git diff --check` passed |

Only documentation changed after the numerical/performance source freeze; the
compiled implementation and gates did not change. No further model runs or
repeat CTests are needed for these prose/table edits. User-facing results are in
the bilingual benchmark guides; source-level diagnostics, failures, provenance
and historical measurements remain in the plans. Preserve unrelated cleanup
plans and retirement metadata. No files were staged, committed or pushed by
this optimization turn.

### Subsequent complete performance refresh

The user subsequently requested complete latest records in both performance
guides. The [complete performance measurement plan](20261004-214547-latest-complete-model-performance-records.md)
owns the thirty-cell current image matrix and current full-video refresh.
It verifies the four candidate B arms above for reuse and measures the remaining
configurations using the same frozen implementation. This plan retains the
historical A/B/A comparison and its experimental evidence; those comparisons
are removed from the user performance pages.
