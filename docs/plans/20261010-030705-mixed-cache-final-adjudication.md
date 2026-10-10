# Mixed-cache final evidence adjudication

Created: 2026-10-10 03:07:05, Asia/Shanghai.

## Scope

Finish the requirement audit for the two recipes frozen in the
[final acceptance campaign](20261009-223907-shortdot-mixed-cache-final-acceptance.md):
F32 weights/F32 compute/mixed-Q8_0 cache and custom nonvision Q8_0
weights/F32 compute/mixed-Q8_0 cache, on the frozen RTX 4090 CUDA build.
The [v2 policy](20261007-200954-precision-acceptance-gates-v2.md) and final
evaluation inputs remain frozen. The reserve remains unopened.

## Approach and steps

1. Let the existing serial arithmetic, regression and performance queues finish.
   Preserve their immutable reports and terminal failures. Do not restart final
   inference or run competing GPU probes.
2. Prepare a separate ignored adjudicator that checks complete evidence before
   writing a verdict. Rehash retained report, identity and cleanup inventories;
   confirm the complete recipe, graph coverage and current-library boundary
   applicability. Missing prerequisites must not become a passing verdict.
3. Verify the five final exports, four sealed quality reports, all four current
   seven-case spatial regressions, legacy F32 regression, CUDA CTest and the
   isolated Python tool suite. Recompute stored fixed-case assessments.
4. Reassess each fresh 96-process paired performance report against the exact
   same-weight F32-cache parent. Keep each workload's PASS, FAIL and INCONCLUSIVE
   labels. The separately bridged Q8 parent measurements are reference evidence;
   do not transfer or multiply their benefits into either mixed-cache result.
5. Write separate arithmetic, quality, regression and deployment conclusions.
   Quality-passing recipes without a qualifying performance label remain
   experimental. Update concise bilingual user documentation and changelog only
   after these conclusions exist, then check documentation and whitespace.
6. Confirm verified raw cleanup completed and retain usable models, original
   checkpoints, conversion manifests, current build and compact evidence.

## Verification

Use `rtk proxy` and `.venv-reference`. Test incomplete-evidence rejection and
verdict classification without rerunning model inference. All evidence outputs
use fresh, exclusive paths. Do not edit any file already bound by a live queue.
Run the adjudicator only after the remaining-verification receipt is complete
and paired performance is no longer active.

## Results

At preparation, all five final exports and four final quality reports were
complete; all four quality reports passed. The isolated Python suite passed
171 tests. Native F32 matrix/bias arithmetic and Q8 mixed-cache matrix,
attention and norm/unary components passed. Q8 elementwise capture was live;
F32 nonmatrix arithmetic and remaining regression/performance queues were
waiting in order. No whole-recipe or deployment verdict has been issued here.

Ignored local evidence is under `build/mixed-cache-final-20261009/` and
`/mnt/SSD1/samcpp-validation/mixed-cache-final-20261009/arithmetic/`; these paths
are unavailable in a clean checkout. The detailed observer recovery and raw
cleanup history remains in the
[session arithmetic plan](20261010-005446-mixed-cache-session-arithmetic.md).

The separate ignored `adjudicate_final_evidence.py` is prepared with SHA-256
`d538962b724e37a4e0596917c9294241f735eb9fb964ac6df60cea0130947b05`.
Seven guards cover partial/incomplete prerequisites, no output before queue
completion, exact workload label membership, failed prerequisite suppression
and experimental classification when no benefit label qualifies. Syntax and
these guards passed without inference. The retained Q8 parent 96-process
assessment recomputed exactly under current gates, and its labels remained
limited to full-image/changed-prompt GPU-memory; repeated-result labels stayed
empty. This checks the adjudicator, not either pending mixed-cache candidate.

Expected graph totals were checked against each fresh metadata report. The Q8
mixed-cache graph has 4,910 leaves, rather than the preceding F32-cache parent's
4,922; its 24,004 nodes partition into 14,538 compute and 9,466 metadata nodes.
The F32 mixed-cache graph has 23,992 nodes, 14,532 compute, 9,460 metadata and
4,958 leaves. No numerical tolerance changed. The ignored preparation receipt
records these checks while explicitly leaving whole-recipe/deployment status
`NOT_RUN`. Documentation checks passed for 133 documents and whitespace was
clean. The final adjudicator has not been launched against live queues.

Preparation v2 preserves the original adjudicator under a separate ignored
filename and adds a complete legacy F32 provenance/output hash closure and
read-only per-case reassessment with the existing unchanged comparator. Ten
guard fixtures and syntax passed, including detection of changed/extra legacy
output files. The current adjudicator SHA-256 is
`09f36cdc570ccf94342945a78ddbd470f7108cc706638a7c9058750b21403882`;
the earlier preparation receipt remains unchanged.

A separate waiting-only wrapper now pins this adjudicator and waits for the
complete remaining-verification and performance-stage receipts. It starts no
inference and will launch the read-only audit only after the existing serial
queues finish. Missing/partial receipts from a terminal producer are errors;
there is no automatic restart. Its ignored queue-binding receipt records the
live PID/session and source identities. Documentation remains a subsequent
step after an actual adjudication result, not an assumed passing conclusion.

During preparation, the Q8 mixed-cache elementwise group passed 5,930 nodes
and 10,510,916,614 output values; spatial passed 204 nodes and 3,355,872,768
values. Remaining Q8 groups and the queued F32 groups still require their own
complete reports and raw reconciliation/cleanup before arithmetic acceptance.

## F16 raw cleanup recovery

The Q8 aggregate completed PASS for all 24,004 nodes, 14,538 compute nodes,
9,466 metadata nodes and 4,910 leaves. Its immutable SHA-256 is
`e97e58e146c2027f763e75139e24322948eb6c7e27a669df8f8069153dd4fa56`.
The subsequent cleanup exited before deleting any file: the suffix allowlist
in both mixed-cache reconcilers omitted `.f16`, while the unchanged attention
observer explicitly writes `.mask.f16`. The complete inventory contains 222
such files, and all 34,310 inventory paths still exist. This explains the
cleanup assertion and the consequent protected exits of all three downstream
waiting queues; it does not invalidate the numerical or aggregate PASS.

Recovery approach, before implementation: preserve both bound reconcilers and
the complete aggregate unchanged. Add one shared ignored cleanup helper that
accepts the actual F16 raw format alongside the existing types, verifies the
entire type/path/report/hash closure before deleting anything, retains JSON
descriptors, and writes the same cleanup receipt contract with its own source
identity. Test F16 cleanup, unchanged retained evidence, unknown types,
tampering and path escapes. A separate F32 queue derivative will keep the
original numerical captures/auditors/reconciler and use this helper for its
eventual cleanup. Restart only terminal waiting queues with new PID bindings;
do not repeat completed inference, arithmetic or quality. A source sweep found
the same missing type in the two active ignored reconcilers and no corresponding
cleanup predicate in production maintenance/validation tools.

The shared helper passed ten retained boundary tests and selected exactly
32,472 raw files, including all 222 F16 files, while retaining 1,838 JSON
descriptors. The old real cleanup supplies the failing reproduction; the new
selector accepts the same unchanged inventory. Shared helper SHA-256 is
`854fd4a3927c7ebeae8ad49e92e48c968f1103b4d45d7132c32e3de8ef69245e`.
It is now rehashing the real Q8 aggregate before deletion.

The fresh F32 queue derivative SHA-256 is
`7b3ec7685c0189ee397e0bf43d46a9cd8874a1279e80b9421973b0af5cd0f5be`.
Its entire numerical capture/auditor loop equals the original source exactly;
only cleanup routing, the shared-helper identity check and live prerequisite
binding differ. The original reconcilers remain unchanged. New live queue
bindings are recorded in ignored `f16-cleanup-*-20261010.json` receipts. The
first preparation-receipt writer lacked the sidecar import path after creating
and checking this derivative; its metadata write was completed using direct
hashing, without restarting the live queue. All numerical and frozen producer
identities remain unchanged. Real cleanup completion is still required before
calling this recovery resolved.

The real shared cleanup completed with exit 0: **32,472 files /
203,178,934,272 allocated bytes (189.2251 GiB)** removed. The cleanup receipt
SHA-256 is `32d561a1608ec48fc7fad98bd45a62872ebd399f4d8bb4387e9bd1f99e9d3c5d`.
A separate read-only confirmation matched its complete deletion inventory to
the unchanged aggregate, confirmed every selected file absent, rehashed all
1,838 retained JSON descriptors and 14 reports, and checked both original
reconcilers unchanged. The F32 queue then started its fresh attention capture.
System/SSD1 availability at this checkpoint was 229/563 GiB.

Root cause: two bound ignored reconcilers omitted the observer's F16 masks
from their cleanup allowlist. Fix: route both recipes through one shared,
fully hash-checked cleanup helper; numerical producers/reports/gates stay
unchanged. Sibling sweep: both affected callers now use that helper, no matching
production cleanup predicate was found. Confirmed: real cleanup plus the
post-deletion closure check passed. Tests/guard: ten retained deletion-boundary
tests in ignored `build/mixed-cache-final-20261009/test_aggregate_cleanup.py`;
the original real invocation failed before deletion and the fixed invocation
passed on those same bytes. Recovery status: resolved; changes are uncommitted.

The resumed F32 native attention audit completed PASS for 370 nodes and
451,679,232 output values, with worst relative L2 `7.309457601233054e-7` and
exact nonruntime output parity (maximum score delta zero). Its immutable report
SHA-256 is `08c60845cf4ffec734724ec6a7f914c3eefd1e4f961d504f774a3a8180ae29c5`.
Norm/unary is now live. Remaining arithmetic, fixed regressions, CUDA CTest,
fresh paired performance and final adjudication are still required.

All eight F32 nonmatrix groups subsequently completed PASS with exact native
nonruntime output parity and zero score delta. The complete native F32
reconciliation passed 28 graphs / 23,992 nodes, partitioned into 14,532 compute
and 9,460 metadata nodes, with 4,958 leaves. Its SHA-256 is
`a1fe08d5ad30a0ee540b0d8e027549a11f631b4cba2c95b44499f4e850eaadcd`.
It closes over 30,570 raw files and 383 report-identity checks. Its compute
partition includes 2,438 F32 matrices, 50 separately covered fused ADD
endpoints, 5,880 other elementwise nodes, and every remaining family. The
normalization/unary and elementwise reports have SHA-256 values
`fef0b363d46b651f1aee38e577e77b01ca4b2fb493eee9551e5a4e4285140fc7`
and `1e20ef1a06609ff7885a2b6170d8bd4603aab7fc470a70f7889d59a9ce8a56ad`.
No numerical tolerance changed. The shared cleanup is performing its complete
second raw-hash verification before deletion; fixed regression, current CUDA
CTest and fresh paired performance remain subsequent requirements.

F32 raw cleanup completed with exit 0, removing **30,570 files /
206,868,934,656 allocated bytes (192.6617 GiB)**. Cleanup and native recovery
receipt SHA-256 values are
`089da3e8d670314f745f9ab53d08310622ce8f9b6424619e6ab88a2712d30234`
and `fd54193048bb931be41556e962fc2f7d7a5e86660c609bf848febefee493b8bf`.
A read-only confirmation matched the whole removed inventory, confirmed every
selected raw file absent and rehashed all 13 retained reports. The two recipe
cleanups together released 381.8868 GiB; retained evidence and current models
remain available locally.

All four current native seven-case spatial regressions passed absolute v2
zero-tail gates, and both mixed-cache variants passed their same-weight-parent
incremental gates. The legacy F32 seven-case comparison also passed. Current
CUDA CTest passed **31/31 in 13.32 seconds**, including the required GPU boundary
tests. Together with the retained isolated Python **171/171** result, the test
prerequisites are complete. The serial queue has begun the fresh F32/mixed-cache
96-process paired benchmark, to be followed by the Q8/mixed-cache benchmark.
No additional GPU probes or CPU-heavy validation will run during measurement.
Final benefit labels and complete-policy adjudication remain pending.

## Completion audit preparation

Before final documentation, prepare a separate ignored completion helper. It
will require the completed adjudication and its waiting-stage receipt, preserve
their identities, and check every applicable campaign requirement against the
actual final result. It must not turn a failed or inconclusive performance
label into a passing label. A quality-passing recipe with no qualifying benefit
remains experimental; completing its evaluation does not require a deployment
PASS. The two fresh mixed-cache comparisons remain independent of the Q8
parent's archived comparison.

The completion receipt will bind the final English/Chinese guides, model
catalog, README, changelog and these plans after documentation and whitespace
checks pass. It will also compare the current producer-source inventory with
the frozen campaign, retain the unopened-reserve restriction, and record the
already verified raw cleanup. This adds no inference, gate changes or new
measurements. Run it only after the serial benchmark and final adjudicator
have terminated successfully.

The ignored completion helper is prepared at
`build/mixed-cache-final-20261009/complete_final_acceptance.py`, SHA-256
`9ca8dde7dffdec38e15749c9f97c6d9527dcfb0d39db7bf8ac246d27f8b306d5`.
Its syntax check passed. It checks 13 requirement groups, binds both fresh
96-process reports and their separate parents, preserves all four performance
label statuses per workload, and requires the current 250-file producer
inventory to match the campaign. It runs final documentation and whitespace
checks before writing an exclusive completion receipt. This is preparation;
the helper has not run, no completion receipt exists, and all performance and
deployment conclusions still await the actual measurements and adjudication.

The requirement review also checked the current Release CUDA build cache:
`GGML_CUDA=ON` and `SAM_REQUIRE_CUDA_TESTS=ON`. The retained CTest log lists
all 31 tests as passed with no skips. `Testing/Temporary/LastTest.log` records
actual short-dot execution at K=1/32/256/257/1024/1025, zero input and an offset
view, plus the Q8 MMQ finite-boundary PASS. Extend the unbound completion
helper to require and retain this execution/configuration closure; preserve
its first prepared version. This checks existing evidence and starts no GPU
work or new regression run.

The extended completion helper's syntax check passed; its SHA-256 is
`1ea200a3a54346fd336f1d0129a0a1a396acda9d7f46f6cd196da1450f851eba`.
The first version remains at the ignored
`complete_final_acceptance.pre-boundary-closure.py` with its original
`9ca8dde7dffdec38e15749c9f97c6d9527dcfb0d39db7bf8ac246d27f8b306d5`
identity. Neither version has been run against pending final evidence.

## First complete performance comparison

The fresh F32-weight mixed-cache comparison completed all **96 processes**
(48 latency, 48 independent memory) on the eight predeclared cases. Its ignored
`final/performance/f32-cache-mixed-q8/performance.json` SHA-256 is
`1e832e1e9a083d4907f994822a4285a803616a9c249f28737b6e095e9e21fbe0`.
It binds the frozen campaign/gates and the exact same-weight F32-cache parent,
reports `contaminated=false`, and passes both quality prerequisites.

| Workload | p50 ratio | p95 ratio | GPU process-peak ratio | RSS peak ratio | Passing performance labels |
| --- | ---: | ---: | ---: | ---: | --- |
| Full image | 0.8892286196 | 0.8892931278 | 1.0013169447 | 0.8584604470 | latency |
| Changed prompt | 0.8033599804 | 0.8111018131 | 1.0013169447 | 0.8584604470 | latency |
| Repeated result | 1.0007448670 | 1.0082236956 | 1.0013169447 | 0.8584604470 | none |

Full-image and changed-prompt latency labels are PASS; their GPU-memory,
combined latency/GPU-memory and host-memory labels are FAIL. All four repeated
result labels are FAIL. The 14.15% lower RSS peak remains below the frozen 15%
host-memory benefit requirement; the GPU process peak increases by 0.13%.
Do not relax either gate. The performance report deliberately leaves whole
arithmetic/deployment `NOT_RUN`; the independent final adjudicator remains
responsible for combining the existing arithmetic/regression evidence.

The serial queue then launched the fresh Q8-weight mixed-cache comparison
against its Q8/F32-cache parent. Final documentation and completion still await
that second complete report and the final adjudicator.

The final quality coverage review read both candidates' actual absolute and
incremental check lists. Each comparison has 67 PASS and 25 NOT_APPLICABLE
checks. All 25 omitted checks are per-category AP checks whose frozen coverage
is below 20 positive prompts or 50 instances; the other 55 category checks and
all applicable aggregate/size/negative/object checks pass. All 80 categories
are reported, but this does not qualify the 25 insufficiently covered classes.
Each report still binds the same 1,024 images and 4,729 prompts. Preserve this
scope in the final bilingual result text.

## Second complete performance comparison

The fresh Q8-weight mixed-cache comparison completed all **96 processes**
(48 latency, 48 independent memory). Its ignored
`final/performance/q8-nonvision-cache-mixed-q8/performance.json` SHA-256 is
`12edfcf4448d11f1230126c9fb6f2dfb1549a291645090f847e375d98d38e701`.
The exact baseline is the custom text/fusion/decoder Q8-weight recipe with F32
cache, not the F32-weight parent used for the other comparison.

| Workload | p50 ratio | p95 ratio | GPU process-peak ratio | RSS peak ratio | Passing performance labels |
| --- | ---: | ---: | ---: | ---: | --- |
| Full image | 0.8761789775 | 0.8802597462 | 1.0016313214 | 0.8565384302 | latency |
| Changed prompt | 0.7877388027 | 0.7920127208 | 1.0016313214 | 0.8565384302 | latency |
| Repeated result | 1.0004646787 | 1.0033174987 | 1.0016313214 | 0.8565384302 | none |

Full-image and changed-prompt latency labels are PASS. Their GPU-memory,
combined latency/GPU-memory and host-memory labels are FAIL; all four repeated
result labels are FAIL. RSS improves by 14.35%, below the unchanged 15% gate;
GPU process peak increases by 0.16%. No parent benefit is transferred or
multiplied into these ratios.

The performance stage and remaining-verification queue completed with exit 0.
The performance-stage SHA-256 is
`3d61bb5d7c4d5133a5ff45d4e07f7014239b7ffefaac4b4e34a58316b86b3a59`.
The pinned final adjudicator has now started against those completed receipts;
its whole-policy verdict and the subsequent documentation/completion checks
remain pending.

## Complete evidence adjudication

The pinned adjudicator completed with exit 0. The final adjudication SHA-256 is
`d98cf137508be441e3c059c721a365d610c927159f6f43edf62fd379eddcb240`;
its waiting-stage receipt SHA-256 is
`23a9a4d76242e8458d2b82b814f72f97dad2874af963f307e32d7e4c99b78bb5`.
Both candidates have whole-recipe arithmetic, absolute quality, incremental
quality, regression prerequisites and scoped deployment status PASS. Their
qualified labels are exactly `latency` for `full_image` and `changed_prompt`,
with no repeated-result or memory label. All reported failures remain intact.

The audit reverified five 1,024-image / 4,729-prompt final exports, all four
quality seals, current-library codec/staging/epilogue fixtures, both complete
compute partitions, all four current spatial regressions and both incremental
comparisons, the complete legacy F32 output/provenance closure, required-GPU
CUDA CTest 31/31, isolated Python tools 171/171, both fresh paired reports and
the separately scoped Q8-parent reference report. Its retained evidence map
contains 2,580 identities; verified raw cleanup remains complete. The reserve
contains 1,024 images with inference disallowed. No frozen producer or gate
changed, and no further model inference was needed.

Reviewed English/Chinese quantization guides now record the two exact
same-weight-parent latency comparisons and their failed memory gates, limited
category coverage and experimental cache-tool/public-loading boundary. README,
both model catalogs and `changelog.md` carry the same scope. The initial v2
campaign's 3-PASS/5-FAIL history remains unchanged and is explicitly separated
from this follow-up. The primary campaign and session audit tables now record
the actual completed evidence. Final documentation/whitespace checks and the
exclusive completion receipt are the only remaining steps.

## Documentation and final closeout

The reviewed result text matches each candidate's actual baseline, labels and
rounded measurements. The documentation checker passed bilingual tables and
local links in all **133 documents**; `git diff --check` passed. A small Chinese
wording adjustment clarifies that the 96 processes were completed measurements,
not a claim that every benefit label passed.

The separate completion helper performs the closing checks again after these
documentation edits and writes
`build/mixed-cache-final-20261009/final-completion-20261010.json` exclusively.
That receipt binds the final documentation hashes, immutable adjudication/stage
identities, all 13 requirement groups, unchanged 250-file producer inventory,
actual CUDA boundary execution/configuration, the two separate 96-process
comparisons, the unopened reserve and the 381.8868-GiB verified raw cleanup.
It preserves each failed performance label. No further inference, gate tuning,
producer edits or release action is part of this closeout.
