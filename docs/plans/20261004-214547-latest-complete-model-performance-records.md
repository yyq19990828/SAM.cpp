# Complete latest model performance records

Created: 2026-10-04 21:45:47 Asia/Shanghai.
Status: complete.

## Scope

The user requires BENCHMARK.md and BENCHMARK_zh.md to contain the complete
latest performance records for available model configurations. Old records,
optimization comparisons, experiments and diagnostic history belong in their
corresponding plans. Preserve the uncommitted RoPE implementation and unrelated
cleanup records. This task does not authorize a commit or push.

The currently implemented adapter is SAM 3. Refresh its ten image storage
profiles across CPU/BLAS, native CPU and Metal (30 cells), and its supported F32
and hybrid video profiles across CPU/BLAS and Metal. User tables will contain
current values, not before/after columns or improvement percentages. Other SAM
adapters/platforms remain roadmap items, not measured implementations.

## Approach and steps

1. Verify the frozen current implementation/build/model identities and retained
   resources. Write new ignored evidence; preserve all existing receipts.
2. Complete fresh original-reference qualification for the native CPU image
   matrix (ten profiles, seven cases each). Existing current CPU/Metal image
   140-case and F32/hybrid video 864-frame qualification remains reusable only
   while its exact executable/library/source/model identities match.
3. Measure current image configurations serially on AC, default execution
   controls, four requested threads and VECLIB_MAXIMUM_THREADS=4. Each process
   has one cold call, five uncached warm full-image calls and separately excluded
   cache replays. The four current candidate B arms from the completed RoPE
   comparison may be reused after exact protocol/identity/output verification;
   the other 26 cells need fresh measurements. No old-source values may fill
   gaps in this matrix.
4. Verify the retained original-qualified video benchmark fixture/qualification.
   Run the complete video pipeline, including frame encoding, detection and
   tracking, with F32/hybrid on CPU/BLAS and Metal. Use the established 64-frame,
   16-warmup/48-measured protocol and one/four-object fixtures when qualified.
   Reuse original-module fixture qualification only for identical pixels and
   task settings. Do not publish synthetic tracker-stage timing as whole-video
   performance. Do not add unqualified native-video claims.
5. Preserve the current four-cell image A/B/A and tracker comparison tables in
   their corresponding RoPE/pipeline plans. Replace both user pages with the
   complete current image/video tables, concise reproducible conditions, units
   and current measurement limits. All historical comparison columns leave the
   user pages.
6. Validate source/qualification/measurement identity, sample counts, arithmetic
   profiles, actual backend placement, power/thermal/exclusive/sleep conditions,
   output quality and RSS. Check bilingual numeric parity, local links and
   git diff --check. Update Unreleased documentation impact and this plan.

## Verification boundaries

Numerical processes may run with bounded memory; their elapsed times are not
performance data. All numerical/build/model work must exit before formal timing.
Do not change C++ code, precision policies, acceptance gates, thresholds, models
or original references to complete these documentation/measurement matrices.
Keep immutable raw evidence unchanged; disclose any reused current cells and
their exact source/protocol identity here. Dates/times use Asia/Shanghai and
must reflect actual captures if the run crosses midnight.

## Results

The preceding RoPE source has
43 passing CTests, 65 tool checks, 140 original image cases and 864 video frames.
Its four default full-image A/B/A cells are complete and source-bound; previous
thirty-cell image values came from the older executable and remain historical.

### Current image qualification and measurement

Fresh native CPU qualification passed all ten profiles and seven cases per
profile (70/70). Every case used native CPU nodes, with zero BLAS/Metal nodes.
Together with the retained current CPU/BLAS and Metal qualification, the image
matrix has 210 passing original-reference cases. Source, arithmetic profiles,
models, references, executable and library identities remained unchanged.

- [Native qualification summary](../../build/rope-optimization/20261004/native-image-qualification-summary-v1.json), SHA256 `0526653cccf836a7e1eba657bda175075d7c146cee11fab08a50f7046c070b0e`.
- [Merged image qualification index](../../build/rope-optimization/20261004/candidate-image-qualification-index-v1.json), SHA256 `ee234f2df95be7f9f60fc916395f7814706e679ccced6f24997c7f60010159d9`.
- [Measurement preflight](../../build/rope-optimization/20261004/candidate-image-performance-preflight-v1.json), SHA256 `7bb1db3251950e0384e42f364b2d1a5f54f13ba26060c71206ed5024b213170f`.
- Frozen ignored runner `build/rope-optimization/20261004/run_latest_image_matrix_candidate_v1.py`, SHA256 `35a65a2607c8419a8facad2ebf7a5e818ba7116a375d946fb6f3e2ef56d48671`.
- [Complete image summary](../../build/rope-optimization/20261004/candidate-image-performance-summary-v1.json), SHA256 `807db07c23b8c2cef714fb8d4b262e2a5d04e795148f246777806b23672ea42b`.
- [Immutable raw image report](../../build/rope-optimization/20261004/current-image-performance-v2/report.json), SHA256 `0f03148a7465d870a224e50a347c13f6e8cca2fbf22db9885fe1c1349875e1f0`.

The complete matrix passed 30/30 cells: 26 fresh cells and four verified current
candidate B arms. Reuse covers F32 and full Q4_K on CPU/BLAS and Metal, captured
on 2026-10-04 at 21:07–21:15 Asia/Shanghai. The fresh matrix ran serially at
22:57–23:49 on the same date. Reuse verification bound source/model/input,
build/qualification, runner/report/output hashes, backend placement, unchanged
artifact maps, quality gates, default execution environment and identical
one-cold/five-uncached/five-cache protocol. Historical A arms were not reused.

Parent review independently verified thirty unique profile/backend pairs,
five uncached warm samples and five excluded cache samples per pair, recomputed
all warm medians, and checked the sealed report/receipt/log hashes. All fresh
cells passed before/during/after power, thermal, exclusive-process, sleep,
source/artifact and output-quality checks. The runner exited before video work.
The `cold_full_image_ms` field measures CLI startup (decode, loading, session
setup and the first segmentation), so it is not published as cold inference.
Published warm latency excludes those startup/file stages; peak process RSS
includes the complete process.

| Image weights | GGUF GB | CPU/BLAS seconds / RSS GB | Native CPU seconds / RSS GB | Metal seconds / RSS GB |
| --- | ---: | ---: | ---: | ---: |
| F32 | 3.371 | 7.060 / 5.023 | 39.128 / 5.045 | 2.707 / 4.364 |
| Mixed F16/F32 | 1.798 | 6.911 / 5.070 | 39.121 / 5.027 | 2.669 / 2.787 |
| Vision Q8_0 | 2.065 | 6.980 / 3.967 | 39.349 / 3.927 | 2.615 / 3.054 |
| Full-component Q8_0 | 1.097 | 6.991 / 3.003 | 39.357 / 2.985 | 2.611 / 2.079 |
| Vision Q6_K | 1.995 | 6.956 / 3.895 | 39.444 / 3.857 | 2.640 / 2.985 |
| Full-component Q6_K | 0.947 | 6.995 / 2.873 | 39.434 / 2.830 | 2.626 / 1.929 |
| Vision Q5_K | 1.957 | 6.954 / 3.875 | 39.378 / 3.815 | 2.640 / 2.946 |
| Full-component Q5_K | 0.864 | 7.028 / 2.772 | 39.358 / 2.751 | 2.644 / 1.851 |
| Vision Q4_K | 1.920 | 6.966 / 3.864 | 39.322 / 3.785 | 2.613 / 2.909 |
| Full-component Q4_K | 0.787 | 7.086 / 2.715 | 39.356 / 2.674 | 2.671 / 1.775 |

### Retained original video fixture qualification

The retained one/four-object fixtures each contain 64 identical 1800 × 1200
PNG frames, with 16 warmup and 48 measured frames. Their original-module
qualification provides 17 stable ID/object prefix records and three delayed
outputs. It is a performance prerequisite, not a full milestone qualification.
The current four original video receipts separately provide the complete
864-frame numerical acceptance required by the production benchmark tool.

The older fixture receipts bound a previous `benchmark_video.py` helper.
Controlled new derived receipts preserve the original records and bind the
current helper after checking the original checkpoint, all 128 PNG hashes,
all other qualification source identities, the old helper recovered from Git,
and identical ASTs for `validate_qualification_records` and `encoding_timings`.
Original inference was not rerun and `eligible_for_milestone` remains false.
The helper changes concern performance profile acceptance/documentation, not
the original oracle or fixture qualification logic. No production gate changed.

[Rebinding report](../../build/rope-optimization/20261004/video-fixture-rebinding-v1/qualification-rebinding-report-v1.json),
SHA256 `d894489d8cce47134f92f0a9c2c3a356f8badb502575f9343aa0c20ad7b6bf99`,
records both parent/derived hashes and the verified scope. An initial ignored
derivation preflight incorrectly required milestone eligibility for the fixture
prefix and stopped before inference; it was corrected to the actual performance
prerequisite contract. No failed attempt contributes performance samples.

### Historical record locations

The user pages no longer include old-source values, baseline/candidate arrows,
percentage improvements, or isolated tracker-stage comparison tables.

- [Pipeline plan](20261004-040240-sam3-pipeline-execution-and-unified-benchmarks.md#v8-unified-image-measurement): preceding complete thirty-cell image matrix and published tracker comparison snapshot.
- [RoPE plan](20261004-182716-sam3-rope-vector-layout-optimization.md#completed-default-full-image-comparison-and-adoption): four-cell A/B/A comparison, numerical evidence and experiment history.
- [Previous visual encoding profiling plan](20261003-134534-visual-encoding-profile-and-optimization.md): earlier full-video results.

All raw receipts are preserved unchanged. Only current candidate B arms with
verified identical execution identity/protocol populate the current image page.

### Video driver preflight review

The first ignored video driver passed its no-model preflight but was held before
timing. Parent and independent peer review found that the production benchmark
freshly verifies its own source, per-cell validation, fixture, model/build and
outputs, while the wrapper only compared its merged index/preflight identities
once before the matrix. That omitted fresh before/after verification of the
complete NUM source and wrapper qualification closure during the eight cells.

Preserve v1 unchanged. A separate v2 must require the reviewed preflight/index
hashes at execution and freshly verify the frozen source and qualification
closure before and after every cell. Hash caching is allowed only within the
no-model preflight; execution checks re-read bytes. No v1 timing was started,
and no production source, precision policy or acceptance gate was changed.

The v2 no-model preflight passed on 2026-10-05 at 00:05 Asia/Shanghai.

- Frozen ignored runner `build/rope-optimization/20261004/run_video_latest_performance_v2.py`, SHA256 `9a67e4be12db07805965088b2d4c3b3344c23289d85b4d7aeec5429ba8011d36`.
- [Video qualification index v2](../../build/rope-optimization/20261004/candidate-video-qualification-index-v2.json), SHA256 `a48810dd2f20b3e18c8b14fdd6229ca0f77b16772fa4d947e82554df0c8e3466`.
- [Video performance preflight v2](../../build/rope-optimization/20261004/candidate-video-performance-preflight-v2.json), SHA256 `2c9c55b7da0d35df05bbabdb5d121f25b73b821cc5bda29643717a753e729ab2`.

The frozen index contains 131 source/qualification/build paths; execution adds
the driver, preflight and merged index, for 134 fresh paths per before/after
check. The existing production benchmark independently checks per-cell model,
binary/library, decoded input source files and outputs. Parent inspection
verified exact eight unique cells, the unchanged 64/16/48 protocol, prefix
milestone eligibility false, all 98 production paths inside the frozen map,
map digest/count consistency, a fresh output directory, runner/preflight identity,
and no unresolved global references in the driver. No preflight model ran.

Parent and independent peer review approved v2 execution. The eight cells run
serially with exact reviewed preflight/index SHA arguments, four threads,
default unprofiled Metal execution, and the unchanged production benchmark.
The first cell started on 2026-10-05 at 00:08 Asia/Shanghai. Numerical/build/model
work outside this single timed executable is excluded throughout the matrix.

### Completed current full-video measurements

All eight cells passed: F32/hybrid × CPU/BLAS/Metal × one/four objects,
512 processed frames and 384 measured frames. Actual capture ran on
2026-10-05 from 00:08:04 to 01:21:01 Asia/Shanghai; report sealing ended at
01:21:06. Each cell's original-reference qualification remained current, and
the production benchmark checked all 64 output frames, object counts/IDs,
nonempty masks, placement, encode counts, output delay/drain and allocation
plateau. These performance workload checks are separate from the full original
864-frame numerical acceptance and do not replace its oracle.

- [Complete video summary](../../build/rope-optimization/20261004/candidate-video-performance-summary-v2.json), SHA256 `321611afba18291b4bb03d84bb15d80a1496e692225f3b15c7996995654afa3b`.
- [Immutable raw video report](../../build/rope-optimization/20261004/current-video-performance-v2/report.json), SHA256 `0f2ad8b7bc2a73ace67a2ea04f5a4781cf94ce13512d6d8fc2c24685542a8796`.

| Video weights | GGUF GB | Backend | Objects | Median seconds/frame | P95 seconds/frame | Peak RSS GB |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| F32 | 3.449 | CPU/BLAS | 1 | 8.852 | 9.023 | 5.397 |
| F32 | 3.449 | CPU/BLAS | 4 | 13.503 | 14.573 | 5.480 |
| F32 | 3.449 | Metal | 1 | 4.224 | 4.298 | 4.682 |
| F32 | 3.449 | Metal | 4 | 7.914 | 8.160 | 4.795 |
| Hybrid | 2.765 | CPU/BLAS | 1 | 8.838 | 9.063 | 5.415 |
| Hybrid | 2.765 | CPU/BLAS | 4 | 13.463 | 14.249 | 5.390 |
| Hybrid | 2.765 | Metal | 1 | 4.246 | 4.368 | 3.989 |
| Hybrid | 2.765 | Metal | 4 | 7.967 | 8.179 | 4.087 |

Per-frame processing covers VideoSession preprocessing, frame encoding,
detection, tracking, mask/result preparation and internal transfers/allocations.
Loading, external PNG decoding and file output are excluded from frame timers
but included in process peak RSS. The first 16 frames are excluded from latency
statistics; the remaining 48 include the final drain. P95 uses linear
interpolation at `(n-1)*0.95`. The fixed fourteen-frame output delay is reported
separately from processing time. No stage-only timing fills these tables.

Every cell passed 134-path fresh frozen checks before/after and its independent
production benchmark model/binary/library/input/output checks. AC, thermal,
exclusive-process and sleep checks passed. Execution ended naturally with no
remaining model/benchmark processes. Parent review independently confirmed the
eight unique combinations, all raw 48-frame median/P95 calculations, exact
benchmark/log hashes, dynamic frozen-map digest/count, actual CPU/BLAS or Metal
placement, encode counts and every sampled power/thermal condition.

### Delivery checks

The bilingual user pages now contain ten image rows (thirty backend cells) and
eight full-video rows, with current absolute measurements, GGUF sizes, dates,
hardware, units and timing scope. Older results and optimization comparisons
remain solely in their corresponding plans. Their raw machine evidence is
unchanged. Experimental/native-video, F16-video and custom quantization cells
are not presented as newly validated combinations.

The 98 frozen production/build/test source paths still match the inventory.
No C++/production tool, precision, threshold, gate, checkpoint or reference
changed during this refresh. The preceding 43 CTests and 65 tooling checks
remain applicable; documentation-only edits do not require repeated model or
CTest runs. Bilingual table/local-link verification passed across 53 documents,
and `git diff --check` passed. Unreleased records the user-visible documentation
change. Preserve unrelated cleanup plans/retirement metadata. Nothing was
staged, committed or pushed by this task.

Final parent validation matched all thirty image cells and eight video rows
against sealed receipts at the published three-decimal precision. Independent
read-only peer review confirmed both languages, dates, timing scope, units,
output delay, complete current scope and the absence of historical comparison
columns. No remaining documentation or measurement blocker was found.
