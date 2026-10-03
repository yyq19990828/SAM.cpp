# P1/P2 baseline, profiling and documentation work

Created: 2026-10-03T13:08:48.050568+08:00.
Baseline: main 56a4cde plus the uncommitted precision-documentation update.
Status: complete. The subsequent measured optimization is complete in the [visual encoding plan](20261003-134534-visual-encoding-profile-and-optimization.md). No commit/push requested for this batch.

## Scope

1. Shorten BENCHMARK and MODEL_ZOO while preserving precision, measurements,
   identities and support limits. Maintain matching BENCHMARK_zh.md and
   MODEL_ZOO_zh.md, and retain historical detail in normal docs files.
2. Preserve the completed local acceptance inputs/builds/outputs/receipts in
   a durable ignored private bundle. Add a small versioned discovery index
   and a stdlib archive verifier; relocation does not relabel old receipts.
3. Promote existing deterministic one/four-object fixture and original Meta
   17-frame qualification recipes into portable tools. Permit explicit
   qualification reuse only when every 64-frame PNG and recipe property
   matches the sealed parent; retain the parent identity and scope.
4. Expose existing RuntimeStats image_ms/inference_ms in JSON and report
   optional stage distributions. Use a short real Metal run to identify the
   next bottleneck; do not alter kernels, weights, candidate policy, BF16/F16
   boundaries or the 15-frame hotstart delay in this profiling phase.
5. Add pinned read-only quick CI for CPU/header/tool/docs checks without
   external model downloads. Preserve explicit F16 failure/CPU diagnostic
   deferral; do not spend another full CPU run on this known rounding limit.

## Verification

- Focused failure/acceptance regressions for archive corruption/overlap,
  qualification mismatch, optional timing metadata and bilingual table drift.
- Locked isolated tools suite, current weight-free CPU/Metal builds and header
  checks as appropriate; no repeated full CPU numerical/performance matrix.
- Fresh fixture PNG hashes must match the old 128 PNGs before reusing oracle
  qualification. A diagnostic profile is not a new 64-frame benchmark PASS.
- Verify all copied archive bytes, preserve historical manifests and bind new
  evidence to actual source/model/binary/libs. Check docs links/parity and
  git diff --check. Remote CI is pending until an authorized push/run occurs.

## Results

- Root summaries reduced to 63-line BENCHMARK and 75-line MODEL_ZOO, with
  corresponding _zh documents and a Chinese validation workflow. Historical
  precision, identities and measurements are retained in detail records.
  Numeric table parity and local links pass automated checks.
- A durable private bundle under models/validation-baselines/m2-m4pro-20261003
  preserves 21,239 files / 85,994,641,754 logical bytes using APFS clones. Each
  copy hash was verified. All151 sealed receipt identities also match archived
  counterparts. The small versioned baseline index binds the archive manifest.
- Portable generator produces all128 PNG hashes identical to the previous
  fixtures. Original Meta qualification is explicitly inherited through parent
  identity and identical protocol/input checks; final v3 receipts bind current
  tools. Fresh Meta execution uses the promoted original recipe but is not
  repeated in this batch. Corrupted inputs/invalid prefix/reuse drift are tested.
- Existing image_ms/text_ms/inference_ms now serialize into runtime JSON.
  Benchmark analysis handles new stages and explicitly missing historical data.
  No graph, weight, precision, candidate or temporal policy was changed.
- Independent current CPU/Metal CLI builds and serialization checks pass.
  Fresh CPU quick build (Metal/BLAS/native disabled, pinned local GGML) compiles
  all targets and independent headers; CTest11/11 passes. Isolated tools24/24,
  documentation and whitespace checks pass. Full CPU model acceptance is reused.
- Two short Metal diagnostic clips complete with stable 1/4 objects and zero
  CPU graph nodes. They were Battery Power, declared length3, two propagated
  samples including final drain: image encoding medians9.198/9.673s, detector
  pipeline0.905/0.918s, tracker0.671/3.160s. They identify the image path as
  dominant, not a new steady-state benchmark or proof of speedup.
- Pinned, read-only quick CI is configured without model downloads. Action
  tag identities and the official CPU Torch wheel were checked; YAML and local
  commands pass. Remote CI has not run for this uncommitted workflow.

The subsequent visual encoding plan profiles ViT/necks/geometry/transfers,
implements shared projection folding and optional CPU BLAS, and completes fresh
numerical/session/performance acceptance. Current CPU video medians are
9.155/13.887s (1/4 objects). Four-object Metal tracking remains a future profiling
target; the official hotstart delay stays unchanged.
F16 remains an explicit diagnostic limit, and full CPU216 stays deferred.

Receipts: build/p1-p2/20261003/{archive-result.json,encoding-profile.json,
current-cli-builds.json,*-qualification-v3.json,tools-tests-checked.log} and
the private archive manifest. The earlier malformed profile-launch script
failed before inference; profile.log is retained separately from profile-v2.
