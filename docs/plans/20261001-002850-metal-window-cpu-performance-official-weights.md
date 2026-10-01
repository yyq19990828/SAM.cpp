# Restore Metal window operators and diagnose CPU performance

Created: 2026-10-01 00:28:50 Asia/Shanghai.
Status: implementation and validation complete. Controlled CPU investigation did not reproduce the historical 13.6% difference; its specific historical cause remains unresolved, so no CPU kernel fix is claimed.

## Scope and acceptance

Keep official GGML 0.25.3 at `353b63b439f27ab2cc19dac97ab1681ba6d2d084`, the C++17 header-only public API, model/backend layering, checkpoint schema, and frozen numerical gates. Restore actual Metal execution of `WIN_PART` and `WIN_UNPART`; identify the cause of the reported CPU timing difference through controlled evidence and fix the proven cause. Download the authorized original SAM 3 checkpoint and use it for subsequent acceptance.

The previous migration passed all 21 supplementary cases but executes 28 window partitions and 28 restorations on CPU, creating 118 graph partitions. Historical five-run medians were 59.288/6.836 seconds for CPU/Metal and 67.323/7.420 seconds after migration. These historical deltas alone do not establish a CPU kernel regression.

## Approach and ownership

1. Preserve the accepted upstream source/patch, binaries, configuration and reports under ignored `build/`. Shared official and PABannier checkouts remain read-only. Record exact compiler, flags, model hashes, thread count and source identities.
2. A Metal worker ports the older fork's native F32 window kernels, dispatch, support checks and argument layout into an isolated official source copy. Preserve padding, layout and precision behavior. Add one maintained regression covering representative rectangular/padded inputs, multiple channels, and actual direct GPU execution against CPU results. Produce an incremental patch for coordinator integration; do not overwrite another worker's patch changes.
3. A CPU worker compares old/new source and build configuration, constructs isolated comparable binaries, then obtains interleaved same-machine timings and a profile with other project computation paused. Narrow any reproducible difference with representative operator probes. Apply the smallest fix only after the causal evidence identifies it. If the historical delta does not reproduce, investigate and correct the comparison methodology rather than inventing a kernel fix. Record remaining uncertainty honestly.
4. A weights worker uses the connected Hugging Face tools and installed `hf` CLI, checking authentication separately from license acceptance. Prefer the already inspected `facebook/sam3` revision `3c879f39826c281e95690f02c7821c4de09afae7` and verify repository metadata, size and SHA-256 of `sam3.pt`. Store weights and provenance only in ignored `models/`/`build/`; never print tokens or request them in chat. Download may overlap code inspection, but conversion/reference generation must wait for the coordinator's CPU measurement window to finish.
5. The coordinator merges frozen GGML changes, regenerates the complete patch and exact source-tree hashes, and runs fresh CPU/Metal builds and downstream integration checks. Retain licenses, source-copy isolation and cache-preparation protection. Keep all numerical thresholds unchanged.
6. Convert original weights to FP32/FP16 and export the pinned Meta unfused-FP32 reference using the existing isolated Python environment and CPU source adaptation. Validate the complete seven-case corpus on FP32/CPU, FP16/CPU and FP16/Metal. Keep original-checkpoint acceptance distinct from the earlier supplementary results. If authentication is blocked, continue the operator/performance work and report the exact download status.
7. Verify session ownership, image/prompt reuse and actual backend accounting. Benchmark the final implementation under the same fixed four-thread protocol with other builds/inference/downloads stopped. Refresh canonical builds after acceptance; update README, patch documentation, changelog, this plan and machine-readable receipts.

## Measurement and verification rules

- CPU timing work requires a coordinated exclusive compute window: no concurrent reference exports, GPU inference, builds or other project benchmarks. Inspect power/thermal information where available; do not change system settings. Do not compare one current run to an old median as proof of causality.
- Measure with the same weights, input pixels, prompt, compiler, optimization/native flags, thread count and precision policy. Separate first-call, warmed full-image, and cached-result timing; retain per-run/stage data where needed. Compare numerical outputs when substituting kernels or graphs.
- Window tests must check observable layout, padding and bounds, not mirror kernel code. The Metal regression must execute on the real GPU without scheduler fallback. The full SAM pipeline should eliminate the 56 window CPU assignments; investigate any remaining fallback explicitly.
- Run Release CTest, guarded/self-contained headers, two-TU linkage, caller-owned GGML consumers and isolated Python tooling checks. Source/patch provenance and whitespace checks remain required. No coverage quota or redundant per-helper suite.
- Final reports include checkpoint identity, source/patch hashes, hardware, latency, memory and confirmed versus unresolved causes. No original-weight claim until its file and full reference suite are verified. No commit or push is requested.

## Progress and results

### Native Metal windows and integration

The combined patch adds native `WIN_PART`/`WIN_UNPART` to official GGML while
preserving the accepted precision changes. It touches nine GGML files. The
six direct CPU/MTL0 layout comparisons include padded/rectangular inputs,
multiple channels, partial threadgroups, window size one, and SAM's
C1024/W72/H72/window24 shape. Partition and restoration match exactly.

- Patch SHA-256: `0a0b80dd15c2a8b5a05d148a31e9e53f8df4d0555852ef301da336a9d97b7c48`.
- Original tree: `5fc1277d1894e92a0b1a812c3ac88bb46564a2cccd9873e02ad7f8533346b48c`.
- Patched tree: `5816fc926f890714e8a8fa853c428550387f0653794a19537a0ba0b420987c8a`.
- Fresh CPU/Metal CTest: 9/9 each; caller-owned GGML consumers: 3/3 each.
- Source-copy, complete-tree provenance, cache drift and overlap protections pass.
- All seven original-weight Metal cases execute 3,342 Metal nodes, zero CPU
  nodes and six graph partitions. The former 56 CPU window assignments and
  118 partitions are eliminated. Host preprocessing/postprocessing remain CPU work.
- The original-weight Metal session regression passes retained model lifetime,
  image/prompt changes, repeated prediction reuse and shared-model isolation.
  The compute-buffer high-water mark is 1,245,268,288 bytes.

Receipts: `build/perf-repair/integration-validation.json`,
`metal-window-metadata.json`, `official-validation-metal-f16/metrics.json`, and
`session-metal-official/results.json`.

### CPU diagnosis and measurement correction

Comparison executables used the same frozen SAM headers, decoded-image object,
weights, native compiler flags, four threads, and exact FP16-to-FP32 storage
policy. Only the old dependency needed a precision-setter API shim. Outputs
were measured outside the synchronized Documents tree. Other project builds,
inference, reference export and downloads were paused during the comparison.
One profiled warmup per variant was excluded, followed by three interleaved
pairs (`old/new`, `new/old`, `old/new`).

| Measurement | Old PABannier GGML | Official GGML |
| --- | --- | --- |
| Warm full-image runs | 63.186, 60.476, 61.362 s | 64.538, 62.206, 61.722 s |
| Median | 61.362 s | 62.206 s |
| Within-variant range / median | 4.42% | 4.53% |
| CPU nodes / graph partitions | 3,332 / 6 | 3,332 / 6 |
| Largest compute allocation | 961,062,048 bytes | 961,062,048 bytes |

The median difference is +1.37%, with exact agreement of scores, boxes and mask
bytes. Both profiles are dominated by FP32 dot products; all 123 normalized ARM
instructions in that hot function match. A representative F32 matrix probe
(K/M/N=1024/3072/576) run in ABBA order differs by +0.67%, with identical
reference error. These observations do not establish a CPU kernel regression,
and three samples cannot establish exact performance equivalence either.

The historical 59.288/67.323-second difference did not reproduce. Its precise
cause remains unknown because historical individual samples, profiles, power,
thermal and background records were not retained. Current `bird` activity
near one CPU core is recorded as a confounder, not evidence of historical
causality. No CPU kernel change is justified. The maintained fix retains
individual full-image, stage and cache samples in CLI JSON so subsequent
comparisons can be audited; existing medians and their semantics are unchanged.
Text time is included within inference time and must not be counted twice.

Receipts: `build/perf-repair/cpu-evidence/cpu-controlled-comparison.json` and its
raw process samples, profiles, disassembly, operator probe and compiler identities.
The frozen reproduction harness is in `build/perf-repair/cpu-harness/`.

### Original checkpoint and reference

HF CLI 2.0.0 and the connected HF tool both authenticate; gated model access is
also verified. Downloaded `facebook/sam3` revision
`3c879f39826c281e95690f02c7821c4de09afae7`:

- Original `sam3.pt`: 3,450,062,241 bytes, SHA-256
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`, matching HF LFS metadata.
- FP32 container: `2e8d55036d71f2c46fb16bd6538086522ae84e70710208ef9ad3b7441e673c67`.
- FP16 container: `ad3619fa725efce76bec6214a0aaa2d66f901859fbab00fdf2b711b0d4dcce7f`.

Both converted containers are byte-identical to the previous supplementary
conversions. New sidecars preserve the independently verified original source;
old supplementary receipts are not relabeled. The original checkpoint contains
22 interactive SAM 2 neck tensors that the image-only reference does not use.
The exporter now validates these exclusions against the existing frozen schema
and records them; missing image tensors, unknown names, wrong shapes or
non-floating types still fail. Six isolated Python tooling tests pass.

The seven-case reference uses pinned Meta source
`2345a4ad109ac29c569da749c91d84f10dc08c40`, its recorded CPU adaptations and unfused
FP32 graph, four threads, disabled autocast/TF32/compilation, and the exact BPE.
The reference manifest SHA-256 is
`64d471b644615f26350e37a235b06e330e3e16f1d3aa2cdc6cc553b6fe0f814b`;
it declares `official-checkpoint` and `eligible_for_milestone=true`.

Receipt: `build/perf-repair/weights/provenance.json`. Models/reference artifacts
remain ignored under `models/official/3c879f39826c281e95690f02c7821c4de09afae7/`.
Original-weight numerical acceptance passed all 21 cases without
`--allow-supplementary`:

| Configuration | Cases | Maximum normalized tensor L2 | Minimum high-confidence mask IoU |
| --- | --- | --- | --- |
| FP32 / CPU | 7 / 7 | 0.000570083 | 1.0 |
| FP16 / CPU | 7 / 7 | 0.0130565 | 1.0 |
| FP16 / Metal | 7 / 7 | 0.0129848 | 1.0 |

All token, tensor, score, box, mask and low-confidence detection gates pass.
The 70 same-weight CPU/Metal tensor comparisons also pass: maximum normalized
L2 0.000151892, minimum mask IoU 1.0, maximum score difference 3.58e-7 and
zero query-selection differences. All 70 Metal tensors are byte-identical to
the pre-window-patch outputs generated from the identical converted container;
that comparison remains separately labeled because the old run was supplementary.

Receipts: `build/perf-repair/official-acceptance.json`,
`official-validation-{cpu-f32,cpu-f16,metal-f16}/metrics.json`,
`backend-agreement.json`, and `metal-before-after-tensors.json`.
Isolated final timing and canonical build refresh are complete.


### Controlled Metal timing

Three interleaved before/after window-patch pairs used the same official FP16
container, truck image, prompt, compiler/Release/native options and four CPU
threads. Each process performed a first inference before the measured warm run.
No other project inference or builds ran; unmanaged desktop services remained
active and are recorded. The pre-window canonical binary was retained until
comparison completed.

| Pair | CPU-window fallback | Native Metal windows | After / before |
| --- | --- | --- | --- |
| 1 (before/after) | 6.898 s | 6.812 s | 0.9875 |
| 2 (after/before) | 7.744 s | 7.663 s | 0.9896 |
| 3 (before/after) | 6.816 s | 6.673 s | 0.9791 |
| Median across three warm runs | 6.898 s | 6.812 s | 0.9875 |

The measured median improvement is 1.25%; all three adjacent pairs improve.
The historical 8.5% Metal difference was a comparison between different periods
and cannot be attributed entirely to window fallback. GPU coverage and the
partition reduction are confirmed independently of these timings. This small
sample on an active desktop is not a universal throughput guarantee.


### Final standalone benchmark

The final CLI ran on Apple M4 Pro / 48 GiB, macOS 27.0 (26A428), AppleClang 21,
Release/native settings, four CPU threads, the official FP16 conversion above,
`truck.jpg` (1800 x 1200), prompt `truck`, and score threshold 0.5. CPU and Metal
ran sequentially with other project computation stopped. Output was written
under `/private/tmp/sam-final-benchmarks-20261001` and copied to ignored
`build/perf-repair/benchmarks/` after timing. This battery-powered desktop retained
unmanaged background activity; no thermal/performance warning was reported, which
is not proof of constant CPU/GPU clocks. Power and process samples are retained.

| Metric | CPU | Metal |
| --- | --- | --- |
| Five warm full-image runs | 64.253, 62.489, 61.553, 61.784, 64.505 s | 6.948, 7.204, 7.052, 6.875, 7.210 s |
| Warm full-image median | 62.489 s | 7.052 s |
| First pipeline call including load/decode | 63.655 s | 7.310 s |
| Repeated-result-cache median | 10.700 ms | 10.634 ms |
| Peak process RSS | 4,836,556,800 bytes | 2,649,751,552 bytes |
| Weight buffer | 3,369,375,008 bytes | 1,795,849,440 bytes |
| Largest compute allocation | 961,062,048 bytes | 1,245,268,288 bytes |

The five-run standalone benchmark is separate from the three paired comparisons.
First-call timing excludes process startup and may reuse OS/shader caches.
Cache latency is not a new-prompt or video frame rate; buffer sizes and RSS must
not be summed. All new JSON arrays have the expected sample count, finite
nonnegative values, and medians equal to the retained median fields. The six
paired Metal runs also have exactly equal detections and PNG mask bytes.

Receipt: `build/perf-repair/benchmarks/receipt.json`; raw result directories are
`06-metal_after` and `07-cpu_after`. No CPU kernel change or historical-cause
claim is inferred from these final standalone timings.

### Final integration

Canonical `build/cpu` and `build/metal` were refreshed after all timing stopped;
both pass 9/9 CTest checks. Calling `ggml_version()` and `ggml_commit()` from
each actual canonical library returns `0.25.3` and
`353b63b4-sam-0a0b80dd15c2`. Canonical Metal directly executes all six window
shapes on `MTL0` with exact CPU agreement. The full numerical matrix used the
fresh matching builds whose binary/source hashes are retained separately.

Receipt: `build/perf-repair/canonical-validation.json`; combined results:
`build/perf-repair/summary.json`. Project-owned text and updated local links
pass whitespace/link checks; unified-diff context markers remain intact.
Shared dependency/reference checkouts remain unchanged. No commit or push was
performed. Video, SAM 3.1, other model families, CUDA and the unexecuted Metal
tensor-API branch remain outside this acceptance scope.
