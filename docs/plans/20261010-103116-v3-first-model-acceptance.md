# First SAM 3 v3 model acceptance campaign

Created: 2026-10-10 10:31:16, Asia/Shanghai.

## Scope

Copy the user's COCO 2017 dataset from SSH host `工作站2` into the local,
Git-ignored `models/` tree, inspect its instance segmentation annotations and
begin real model acceptance with the newly implemented v3 policy. Preserve all
v2 results, existing source edits and original checkpoints. Do not reinterpret
old final results as fresh v3 evidence.

The user selected the expanded first round: explicit `balanced` quality tier
with an F32 baseline, F16, all eight vision-only/full-linear Q8_0/Q6_K/Q5_K/Q4_K
presets, the previously scoped custom text/fusion/decoder Q8 recipe and mixed-Q8
feature-cache variants. Start with F32 compute and F32 cache for weight presets;
mixed-cache variants are separately declared configurations. Expand final evaluation only to
development-eligible configurations, after freezing the exact recipes, source,
hardware, input split and optional performance comparisons.

## Approach and steps

1. Inventory local storage, CUDA hardware/build tools, reference dependencies,
   original weight/source/tokenizer identities and recoverable historical data
   manifests. Inspect remote dataset size and annotation types before copying.
2. Copy annotations and images with resumable transfer into `models/coco2017/`.
   Verify file inventory/content and keep a local transfer receipt. Both bounding
   boxes and segmentation annotations should be examined; the v3 task metrics
   consume segmentation masks.
3. Recover every consumed v2/follow-up dataset manifest, retain the unopened
   reserve and exclude prior image IDs and content hashes. Prepare development
   and independently unused final inputs using the existing selectors. Do not
   claim history completeness if the manifests cannot be recovered.
4. Build the matching backend and prepare a pinned isolated CUDA reference
   environment. Convert/check the required GGUF models and manifests from the
   verified original SAM 3 checkpoint. Reuse compatible verified artifacts.
5. Run correctness/fixed-regression and development task-quality checks under
   v3. Record score/box/tight-mask diagnostics independently. Stop failed
   configurations before consuming fresh final data.
6. Freeze eligible recipes and their history, run original/native final outputs
   and evaluate absolute plus applicable cache-incremental quality. Measure
   optional paired performance only with eligible frozen baselines. Keep
   arithmetic, task quality, diagnostic observations and benefit labels separate.
7. Record actual evidence, remaining gaps and resource use. Update concise user
   documentation only for proved results. Clean regenerable intermediates after
   validation, preserving useful models, conversions, current build and evidence.

## Verification

Use `rtk proxy` for agent shell commands. Verify source/weight/BPE/gate hashes,
dataset disjointness, complete image/prompt inventories, strict backend
execution and original-model reference settings. Run matching CTest checks and
the isolated tool suite where implementation/build changes require them.
Performance evidence needs an uncontaminated device and independent processes;
small improvements or no benefit label do not invalidate passing task quality.
No unit test, transfer success or single component check alone qualifies a model.

## Results

Preparation started. All `models/`, `build/` and `.venv*/` artifacts referenced
here are Git-ignored local data, unavailable in a clean checkout.

### Initial preflight and adjustment

The local RTX 4060 Laptop has 8 GiB VRAM. The original CUDA FP32 reference
completed all three truck prompts, then failed while switching to groceries
(`build/v3-first/reference-fixed.log`). Inspection found that the fixed tensor
exporter retains the previous state's feature tensors and raw mask logits while
allocating the next image. Before repeating the preflight, release per-prompt
tensor references after writing their evidence and release the old state before
encoding a different image. Keep same-image feature reuse and numerical settings
unchanged. Validate this lifecycle fix with the full seven-case real CUDA export
and the isolated Python tool suite. A failed export is not model qualification.

Replaying the three recorded train2017 selections reproduced all published
SHA-256 values exactly (seeds 20261009, 20261010 and 20261011; 3,072 image IDs).
The first selector predates the optional exclusion field; including a null field
produced a different serialization on the initial reconstruction attempt. Both
attempt logs are retained. The verified reconstruction audit is
`build/v3-first/recovered-history-v2/audit.json`. It is not a substitute for the
unavailable complete historical dataset manifests. Treat every val2017 image as
excluded from new inference to protect the unidentified old reserve; use the
recovered train selections only as conservative exclusions for new development.

The complete seven-case original CUDA FP32 export subsequently passed with
`PYTORCH_ALLOC_CONF=expandable_segments:True`; the retained reference is
`build/v3-first/reference-fixed-expandable/`. The first lifecycle retry overlapped
the CUDA Python test process and is invalid for capacity conclusions. An isolated
retry still encountered allocator fragmentation; its allocated memory fell from
7.03 GiB to 6.74 GiB, but 517 MiB reserved/unallocated space could not satisfy the
412 MiB request. The successful retry changes allocator behavior only. Future
model/GPU tests run sequentially. The pinned CUDA Python environment passed all
188 tool tests.

The host CUDA toolkit is outside the system loader path. Reconfiguration with
`-DCMAKE_BUILD_RPATH=/usr/local/cuda/lib64` resolved the native link failure; the
CUDA build completed without changing system linker configuration.

Initial K-format conversion exposed a provenance mismatch: CMake emits the
three-patch GGML identity including `ggml-short-dot-cuda.patch`, while the Python
converter allowlist only included the earlier one/two-patch identities. Add the
exact SHA-derived three-patch identity, retain the historical identities, and
continue rejecting unknown suffixes. Verify with the existing identity tests,
the actual built quantizer and complete Q6_K/Q5_K/Q4_K conversions. Preserve the
initial failed conversion receipt; retry into new logs.

Before any new COCO inference, select 64 calibration and 512 development images
from train2017 using seed 20261012 and annotation-only category/area coverage.
Allocate disjoint 1,024-image evaluation and reserve roles for later use, but do
not infer on either without the required final campaign history. Exclude all
5,000 val2017 IDs and the exactly reconstructed 3,072 earlier train2017 IDs; verify
content hashes before materializing inputs. This is a v3 development/retest
campaign while full historical manifests are unavailable. A passing fixed case
or development result cannot be advertised as complete independent acceptance.

### Completed local preflight

Hardware: Linux x86_64, RTX 4060 Laptop 8 GiB, driver 580.178.04. Native tools use
CUDA toolkit 12.1 and architecture 89; the pinned original-model environment uses
Python 3.12.13 and Torch 2.10.0+cu128. These results apply to this local setup.

The original checkpoint exported all seven fixed cases. Every requested native
configuration passed the v3 `balanced` fixed regression with F32 compute:

| Weights | Feature cache | Fixed regression | Minimum matched mask IoU (diagnostic) |
| --- | --- | --- | ---: |
| F32 | F32 | PASS | 1.000000 |
| F16 | F32 | PASS | 0.999925 |
| Custom text/fusion/decoder Q8_0 | F32 | PASS | 0.999674 |
| F32 | mixed-Q8_0 | PASS, including cache increment | 0.999674 |
| Custom text/fusion/decoder Q8_0 | mixed-Q8_0 | PASS, including cache increment | 0.999349 |
| Vision Q8_0 | F32 | PASS | 0.998379 |
| Full Q8_0 | F32 | PASS | 0.998698 |
| Vision Q6_K | F32 | PASS | 0.996747 |
| Full Q6_K | F32 | PASS | 0.996749 |
| Vision Q5_K | F32 | PASS | 0.988987 |
| Full Q5_K | F32 | PASS | 0.989032 |
| Vision Q4_K | F32 | PASS | 0.977033 |
| Full Q4_K | F32 | PASS | 0.967813 |

This fixture comparison has only six matched high-confidence objects and does
not establish COCO task-quality floors, whole-graph arithmetic or performance
benefits. All score/box/tight-mask fidelity diagnostics remain non-vetoing.
Receipts are `build/v3-first/fixed-matrix-results.json` and each configuration's
`build/v3-first/fixed/<name>/manifest.json`; all eleven usable models and their
conversion manifests are retained under `models/v3/`.

Validation completed: CTest **31/31**, including required CUDA tests; the pinned
CUDA Python environment **188/188** before the quantizer identity fix; after the
fix, the isolated CPU environment **187 passed, 1 CUDA-only skipped**, plus both
targeted quantizer identity checks passed. The actual compiled quantizer identity
was verified and all six K-format models converted successfully. `git diff
--check` passed. `build/v3-first/preflight-summary.json` binds the evidence hashes.

### COCO development run in progress

The annotations and all 5,000 val2017 images are local. A priority copy completed
for the 2,624 selected images and the 3,072 historical train2017 images needed for
content exclusions. Full train2017 copying and remote/local SHA-256 inventory
verification continue independently. COCO instance annotations contain both
`bbox` and `segmentation`; val2017 has 36,781 instances, with 36,335 polygon masks
and 446 RLE crowd masks. This campaign evaluates segmentation masks.

All 8,072 conservative historical exclusions have distinct verified local
content hashes. The new 2,624-image selection has zero ID/content overlap with
them and zero cross-role content duplicates. Dataset SHA-256:
`ff658d9f7860eb70d920af58678fabd95a6095dda8530885070f465db7e11aad`.
Only the 576 development/calibration images (3,565 prompts) may be inferred on.
The 1,024 evaluation and 1,024 reserve images remain closed.

The serial development runner is `build/v3-first/run_development_matrix.py`, with
PID/log metadata in `build/v3-first/development-job.json` and live progress in
`build/v3-first/development-matrix-progress.json`. It normalizes the shared RGB
inputs, exports the original FP32 oracle, exports every fixed-regression-eligible
native recipe and evaluates absolute plus applicable cache-incremental quality.
Its 252 acceptance source files are frozen under
`build/v3-first/frozen-source/`, so later workspace edits cannot change this
running cohort. Original FP32 export uses the successful expandable allocator
setting and GPU stages run serially. Completed per-configuration receipts remain
immutable; errors are recorded separately from measured quality failures.

Independent final acceptance, complete recipe arithmetic and optional paired
performance benefits are **NOT_RUN**. The running development cohort is not
advertised as deployment qualification. All earlier v2 evidence and gate hashes
retain their historical meaning.

### Continuation and result reporting

Continue the existing serial runner rather than starting overlapping GPU jobs.
The original FP32 export reached 109/576 images at 11:03; no export failure has
occurred. Full train2017 copying reached 66%. The remote SHA inventory is still
reading source files, and local verification waits for that complete inventory.

Eight v3 performance workloads were selected using annotations only and frozen
in `build/v3-first/performance-cases-v3.json`. This is preparation, not a latency
measurement or benefit label. Final-quality prerequisites remain in force.

Add a local, read-only reporting companion under `build/v3-first/` that follows
the existing runner and copy receipts. Its live summary must show all 13
configurations, quality checks that fail or lack evidence, diagnostic statistics,
cache-increment results, development eligibility and separate NOT_RUN final,
arithmetic and performance fields. Verify completed metric receipt hashes before
summarizing them. Produce an immutable summary when the running cohort finishes;
never convert a 576-image development result into final PASS. This companion may
read completed evidence but must not infer on evaluation/reserve images or
modify the frozen sources, models, gates or v2 receipts.

The full expanded matrix is a sustained run: fixed-case F32 runtime telemetry
shows roughly 1.08 s/image encode and 0.27 s/prompt inference before ranked-mask
export, so 576 images and 3,565 prompts across 13 configurations can take hours.
Overlap one CPU quality-evaluation process with the next serial GPU export to
avoid leaving the GPU idle during 2,000-repetition bootstrap calculations.
Pause only the old Python coordinator while its current original-reference child
continues unchanged. A replacement coordinator must adopt the original only
after its command exits successfully and its complete 576-image/3,565-prompt
manifest passes the normal identity checks. Keep normalization, original output,
source freeze, dataset, policy, native binaries and all recipes unchanged. Record
the handoff, terminate the paused obsolete coordinator, and use exclusive new
native/metric output directories. Only one GPU model and one CPU metric process
may run at a time. Run durations remain operational diagnostics, not performance
certification.

Independent F32 GGUF/checkpoint payload verification passed for all **1,133
tensors / 842,343,734 values**, with zero mismatches, using the frozen verifier.
Receipt: `build/v3-first/f32-weight-parity.json`. This proves the declared F32
weight conversion; it does not replace graph arithmetic or task-quality checks.

Usable GGUF file sizes (including uncompressed islands/metadata) are 3,215.0 MiB
for F32, 1,714.3 MiB for F16 and 2,291.3 MiB for the custom Q8 recipe. Full-linear
Q8/Q6/Q5/Q4 files are 1,045.8 / 902.8 / 824.4 / 750.6 MiB, reductions of 67.5% /
71.9% / 74.4% / 76.7% relative to F32. Vision-only files are 1,969.5 / 1,902.6 /
1,866.0 / 1,831.5 MiB. These are storage observations, not GPU peak-memory or
latency benefit labels.

The full local COCO copy completed at approximately 11:20 and all **123,293 files
/ 20,963,587,850 bytes** passed remote/local SHA-256 verification by 11:21: six
annotation files, 5,000 val2017 images and 118,287 train2017 images. Transfer,
remote inventory and local verification commands all exited successfully.
Receipts are `build/v3-first/coco-copy-receipt.json` and
`build/v3-first/coco-remote-inventory.json` (inventory SHA-256
`61affe2936a9141f6eb09ca4cdb5c29453618ff3da5f9c0409c44ddbef93ed08`).
The original remote dataset remains unchanged. Local data is in
`models/coco2017/`; copying/hashing evaluation and reserve files does not authorize
inference on those roles.

The original FP32 development export completed successfully at approximately
11:29: **576 images / 3,565 prompted outputs**, command wait status 0. The complete
manifest SHA-256 is
`a3cc777f3ef397febe375324702d6d229ba9ddc950cab576655c431894078cc4`.
The continuation coordinator validated that manifest with the standard loader,
recorded `build/v3-first/development-pipeline-adoption.json`, terminated only the
paused obsolete coordinator and started the F32 native export. Native output
progress was observed at 19/3,565 prompts, confirming the handoff executed.

The active coordinator is `build/v3-first/run_development_pipeline.py`; process
metadata and logs are `build/v3-first/development-pipeline-job.json` and
`build/v3-first/development-pipeline.log`. It continues all 13 frozen recipes
without further input, with one GPU export and one CPU evaluation at a time.
The read-only companion follows it in `build/v3-first/acceptance-status.json`
and `build/v3-first/acceptance-status.md`, then writes an immutable
`build/v3-first/development-summary.json` / `.md` when the cohort finishes (or a
separate error summary if the coordinator fails). Its metadata/logs are
`build/v3-first/report-development-pipeline-job.json` / `.log`.

At this checkpoint, candidate COCO task-quality results are still pending.
Independent final acceptance, whole-graph arithmetic and formal performance
measurement/benefit labels remain **NOT_RUN**. Documentation checks passed for
140 files and `git diff --check` passed after these updates. Existing v2 gate
SHA-256 remains unchanged.

### Storage budget and low-space guard

At 11:31 the filesystem had 143.5 GiB available. Full COCO (20 GiB), original
checkpoint (3.3 GiB), all eleven GGUF models (18 GiB) and the CUDA environment
(7 GiB) are already local. The complete original development output occupies
458 MiB; the first approximately 300 native prompts occupy 30 MiB. The remaining
13-recipe outputs and small metric/source archives are estimated at 6–10 GiB;
allow 20 GiB conservatively. COCO export writes ranked masks/JSON and does not
produce full intermediate tensor dumps for every image.

Before adding protection, the disk/size inventory above was read without
changing the campaign. Add a local guardian outside the frozen acceptance
sources that checks available space every five seconds. Below 50 GiB, stop only
the recorded coordinator and its owned descendant processes with SIGSTOP,
preserve their partial outputs, write a separate guard receipt and mark the
mutable progress report PAUSED_LOW_DISK. Do not remove checkpoints, COCO files,
usable GGUF models, environments or retained evidence. Bind process start times
to avoid acting on reused PIDs. Validate the pause path against an isolated dummy
process tree, then launch the guardian for the existing pipeline. This changes
scheduling only; quality gates, recipes, frozen sources and v2 evidence remain
unchanged.

The guard is implemented in the local ignored orchestration helper
`build/v3-first/guard_development_storage.py`. The forced low-space test at 11:36
paused all three processes in an isolated dummy tree and changed only its dummy
progress to PAUSED_LOW_DISK. The actual campaign remained RUNNING and advanced
from native prompt 945 to 947 during the test. All dummy processes were then
resumed and terminated. Test evidence is retained in
`build/v3-first/storage-guard-test-20261010-113605/`.

The real guard started successfully at 11:36 with a 50 GiB threshold and five
second interval. Its separate identity/status receipts are in
`build/v3-first/storage-guard/`, with launch metadata in
`build/v3-first/storage-guard-job.json`. The initial guard status is WATCHING;
filesystem availability was 183,057,149,952 bytes (170.5 GiB) at that check.
Available space may also change due to other workloads. No campaign dataset,
checkpoint, model or evidence was deleted to obtain this headroom. These local
orchestration helpers/receipts are Git-ignored and are not distributed in a clean
checkout. A guard-triggered pause preserves completed numerical evidence and
requires explicitly restoring the saved progress status and resuming the
recorded process identities after space is available; it does not resume itself.
