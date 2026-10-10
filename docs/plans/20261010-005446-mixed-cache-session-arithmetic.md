# Mixed-cache session arithmetic

Created: 2026-10-10 00:54:46, Asia/Shanghai.

## Scope and approach

Continue the [frozen mixed-cache campaign](20261009-223907-shortdot-mixed-cache-final-acceptance.md)
after NVIDIA access was restored. Verify the two complete mixed-Q8-cache
recipes using the exact frozen image probe, CUDA library and model identities.
Keep production sources and frozen gates unchanged. Neither a cache cast
component nor an earlier F32-cache graph establishes mixed-cache correctness.

## Steps

1. Finish the serial five-recipe final exports before launching another GPU
   process. Keep final quality scoring and all reserve data outside arithmetic
   fixture selection.
2. Replay the previously declared seven development calls: small-image first,
   changed, repeated, whitespace and long prompts; large-image first and changed
   prompts. Use the exact existing development PNGs and prompt TSV. Bind fresh
   unobserved output, graph inventory and allocation metadata per recipe.
3. Reuse the existing independent same-operand numerical oracles, retaining
   their unchanged tolerances. Capture every active F32/Q8 matrix and each
   nonmatrix operation family in fresh directories. Split F32/F16 copies from
   F32/Q8 cache copies explicitly; verify every cache encode and every decoded
   consumer and reject unmatched packed bytes. Check observed output parity,
   complete graph headers, one-to-one node coverage, strict CUDA placement,
   aliases, bounds and raw hashes. Derive expected node counts from the bound
   full inventory rather than using the F32-cache counts.
4. Map the complete recipe's input, model, numerical boundary, active graph,
   final quality and paired performance requirements to immutable evidence.
   Reuse exact-library kernel boundary fixtures only with explicit hash and
   finite-domain binding. Preserve component verdicts and make any broader
   adjudication a separate receipt.
5. Run the separate seven fixed spatial regressions for both candidates and
   their same-weight F32-cache parents against the retained ranked original
   reference. Require zero-tail absolute and cache-incremental checks; neither
   the seven-call arithmetic session nor CTest substitutes for this corpus.
   Keep legacy fixed checks separately scoped and verify their applicability
   before final adjudication.
6. Reuse the already complete Q8/F32-cache parent benchmark only if the exact
   two recipes, benchmark binary, source closure, gates and all eight case
   contracts/input bytes match this campaign. Rehash every old artifact and
   recompute the unchanged performance assessment. Keep the original report
   immutable and record an explicit provenance bridge, without presenting old
   measurements as new processes. The acceptance runner must separately
   require this campaign's passing quality reports before using the reused
   measurement. Both mixed-cache candidates still need fresh paired runs.
7. Overlap CPU-only quality scoring with the remaining serial GPU exports for
   the first three native recipes. Run the unchanged frozen scorer on complete
   exports in fresh staging directories. Check the complete report, object
   hashes and gate consistency, then publish a complete directory atomically
   with no replacement. The main runner keeps ownership if it has already
   started that canonical quality directory. Publish every verdict unchanged;
   this scheduling optimization must never choose reports by PASS/FAIL. Seal
   generated metrics/object evidence separately without editing the scorer's
   immutable report. Leave the last candidate's scoring to the main runner.

## Verification

Use ignored observer/verifier sidecars and `rtk proxy`. Keep GPU capture and
staging serial; paired performance requires exclusive GPU and an accepted final
quality result. Existing final outputs and attempts are never overwritten.
Record failures without changing final gates or using the unopened reserve.
Rehash required evidence before pruning regenerable tensor payloads and retain
all compact reports, original checkpoints, usable models and verified build.
Run the relevant existing regression and documentation checks at completion.

## Results

The restored local environment exposes the frozen RTX 4090 and driver
610.57.04. The final-stage runner is live and is producing the official
checkpoint reference export. No additional GPU workload has been launched.

Ignored sidecars now serialize fresh session captures, reuse the unchanged
independent operator oracles and reconcile complete graph coverage. The copy
observer was derived to capture only F32/F16 copies; the separate cache
observer covers all F32/Q8 copies, preventing either omission or overlapping
claims. All three Python sidecars passed syntax checks and the filtered copy
observer compiled successfully. The queue waits for all five complete final
export manifests and validates their archived identities before GPU work.
No arithmetic capture or whole-recipe PASS is claimed by this preparation.

The retained seven-case ranked original reference passed its full archive and
payload verification, with manifest SHA-256
`6eef5240803f98185db925ae443d44ab905d8252d87ab31f76e4a8e45268dc5b`.
Its checkpoint, CUDA environment and reference packages match the current
freeze. A second queue now waits for final quality and both complete graph
reconciliations before running four same-recipe fixed regressions, the legacy
F32 fixed check, CUDA CTest and paired performance. An actual dead-prerequisite
subprocess check passed: it exits without a verification output directory or
GPU work. The queues remain preparation until their execution receipts exist.
Image-probe positional arguments were checked against its C++ CLI contract
before any arithmetic capture; the corrected sidecar identities are retained
in ignored `session-arithmetic-preparation-v2-20261010.json`.

The Q8/F32-cache parent's existing 96-process benchmark passed the exact-reuse
audit. Both recipes, benchmark binary, every current producer source, gates,
eight complete case contracts and input payload hashes match; all 719 source
artifacts rehashed and the current frozen assessor reproduced every aggregate
field exactly. The original report remains unchanged at SHA-256
`551a9be2d1ce179592a2432a48c3f339c58f5c853209196bdb2feefe784180dc`.
An explicitly provenance-marked bridge under ignored
`final/performance/q8-nonvision/performance.json` has SHA-256
`3584a4e6e6727c30857482b6972fb66c2ad7c11a99167d71b43fcd7fae63d368`.
It records zero new GPU processes and retains the source measurement's quality
prerequisite scope. This campaign's final quality is still required separately
by the performance runner; the bridge makes no new arithmetic or deployment
claim. Fresh paired measurements remain required for both mixed-cache recipes.

The official, F32 and F32/mixed-Q8-cache final exports have each completed
all 1,024 images and 4,729 prompted pairs. Their manifest SHA-256 values are
`f4f6feae1c54bfd5eb8f40f4e71bb84f431d847b2fbe7d9de825c5a9298beb81`,
`ae3c5d5c75ffca66a46e789a7cfa6e0d684c047b8d1d846a85d5e47649911f71`
and `510b5bb0690ec640bc6ac51dbcf0a41174ac75e7f26cfcaa9784a5357584eabd`
respectively. The CPU queue published the complete F32 final quality report
without replacement: SHA-256
`d210a9df1e0e81232e69803ac880980676259bd4773ec224bda069d4c338601b`.
It reports absolute quality PASS with 67 passing checks and 25 coverage-based
NOT_APPLICABLE checks, 2,000 bootstrap repetitions, 8,768 high-reference
objects and 2,316 protected objects; bad, missing, extra and protected-miss
counts are all zero. Its separate evidence seal rehashes generated reports
and confirms the unchanged gates exactly, SHA-256
`7d6639ef993fdbe24f3d657eb160185cf7f14a2d0c4c51a2ec0eaeef3e59e905`.
This is F32 quality evidence only: arithmetic, performance and deployment
remain separate requirements. F32/mixed-cache CPU scoring is now running;
the two Q8-weight exports and their final verdicts remain pending.

## Completion audit requirements

The current freeze requires the following separate evidence before a new
complete mixed-cache deployment conclusion. The checkpoints below are an
audit checklist, not a replacement for the underlying reports.

| Requirement | Authoritative evidence | State at this audit |
| --- | --- | --- |
| Exact recipe, source, gates and untouched final inputs | Frozen campaign identity closure; normalized input and annotation hashes; one-attempt ledger; reserve policy | Verified before final inference; sources and gates unchanged |
| Original checkpoint reference | Complete final reference manifest and all 4,729 hash-bound payloads | Complete, 1,024 images |
| Candidate absolute and cache-incremental quality | Four complete final `metrics.json` reports, including 2,000 paired-image bootstrap repetitions and coverage denominators | All four native final quality reports PASS; both mixed caches pass absolute and incremental comparisons |
| Input/token/cache paths and full active graph | Per-recipe fresh boundary, topology, metadata, all compute-family reports and independent one-to-one reconciliation | Both recipes' boundary/topology/metadata, all arithmetic groups and complete compute reconciliation PASS; F32 native observed outputs match exactly |
| Encoding and numerical boundary behavior | Exact-library cache codec fixtures; Q8 staging/dot/epilogue fixtures for the Q8-weight recipe; direct captured operands | Current-library codec and Q8 staging/dot/epilogue fixtures PASS; both recipe cache operands and complete compute reconciliations PASS; current required-GPU CUDA boundary CTest PASS |
| Fixed regressions | Four current-recipe seven-case spatial reports, cache-parent comparisons and supported F32 legacy report | All four current native absolute regressions PASS; both cache-incremental regressions PASS; legacy F32 seven cases PASS |
| Regression suite | Current CUDA CTest plus isolated tool suite | Python 171/171 PASS; current CUDA CTest 31/31 PASS |
| Workload-specific benefits | Frozen eight-case paired measurements with exclusive-GPU evidence and unchanged label assessor | Both fresh 96-process reports complete and reassessed exactly; both full-image/changed-prompt latency labels PASS, all other candidate labels FAIL; Q8 parent reference remains separately scoped |
| Final scope, documentation and storage | Separate requirement-by-requirement adjudication; concise support/results documentation; immutable retained evidence and rehash-verified raw cleanup | Final adjudication complete: both full recipes PASS only for the two measured latency workloads; both raw cleanups complete; scoped bilingual documentation reviewed and documentation/whitespace prechecks PASS; final completion receipt binds the closing inventory |

Any FAIL or INCONCLUSIVE remains visible. A completed experiment does not
turn an unsupported precision, other GPU/backend, video path or uncovered
category into a qualified configuration. Neither parent performance reuse nor
passing component arithmetic can fill a missing candidate requirement.

## Current-library cache boundary closure

A pre-execution audit found that the retained cache codec boundary receipt
binds CUDA library SHA-256
`1e4f954340ea2c30cb60deb49f4ccda6780993b2e09cb81bc8105aa053c925ef`,
not this campaign's short-dot library
`dc0ebfc0cbf0055b78a8111b02e2ab7ba245002f54602347fe8514baa339f6de`.
The new real-input cache captures do bind the latter library, but cannot
transfer that old synthetic boundary result by status alone.

Before either recipe's arithmetic evidence is finalized, replay the retained
17-row, 96-channel `zero-ties-and-tail-rows` fixture (51 Q8_0 blocks) with the
current build's existing `sam_cache_codec_probe`. Its input SHA-256 is
`723ac4b397a7b63575dca9bf0c9a6d5e0191aa29e8a628261698c273caa63af6`.
Require the exact linked CUDA library, independent scale/integer/decode checks,
strict CUDA execution, and immutable probe/input/output/verifier identities.
The shared codec fixture applies to both exact recipes because the native
cache operation and library are identical; each recipe still needs its own
fresh captured cache operands and complete graph reconciliation.

Prepare this as an ignored helper consumed by the queued baseline stage only
after all five final exports. Archive the earlier arithmetic helper version,
record the new preparation identities before capture, and bind the new codec
receipt into each boundary audit. No production code, final gate, final image
or reserve is changed. The arithmetic output root does not yet exist, so no
capture or bound helper evidence is being rewritten. This preparation alone
makes no new boundary or whole-recipe PASS claim.

The replay helper and baseline integration passed Python compilation and an
actual read-only preflight. It verifies the current native codec binary,
its exact linked library, the retained 51-block fixture, both complete recipe
identities and all 250 current frozen producer sources. Its preflight receipt
SHA-256 is
`840417db6546a30b59b142698d5a71259561e8d30d0ac0216c78c328861008a1`.
Neither a native codec output nor the arithmetic output root exists yet.
The original session helper is archived at SHA-256
`6fa56c246f21d6e425afa187d64ea3ab2f67a46e571d3a840c5325790dc4177c`;
new ignored `session-arithmetic-preparation-v3-20261010.json` binds the updated
helpers before capture. The existing live queue invokes a fresh helper process
for each stage, so its next baseline will consume this prepared version without
restarting model exports or repeating any final inference. The shared native
codec receipt will be rehashed and bound into each recipe's boundary audit.
Numerical limits and the independent oracle are unchanged.

## Same-library Q8 boundary and epilogue reuse audit

Rehash the retained short-dot-library MMVQ/MMQ staging boundary and packed-dot
bias/GELU receipts, including every bound raw fixture and verifier. Recompute
their independent CPU staging, packed-dot and epilogue checks under the same
frozen gates before binding them to this campaign's unchanged custom Q8 model.
Keep the original receipts immutable and record zero new GPU processes in a
separate provenance bridge. Preserve the declared finite-domain limit and the
expected rejection of the deliberately overflowing F16-sum fixture. This can
satisfy only shared Q8 kernel boundary requirements; new mixed-cache operands,
whole-graph coverage, final quality, fixed cases and performance remain separate.

The F32/mixed-Q8-cache final report is now complete for 1,024 images and
4,729 prompts: absolute F32 quality PASS and cache-incremental quality PASS.
Each comparison has 67 passing checks and 25 coverage-based NOT_APPLICABLE
checks, with 2,000 bootstrap repetitions. Both comparisons report zero bad,
missing, extra or protected-miss objects among 8,768 high-reference and
2,316 protected objects. The final metrics SHA-256 is
`cb12d88504a1bb851eba8d510e424e6365570bb318511fb9143b2f214902afcd`;
its separate exact gate-reassessment/hash seal has SHA-256
`046a5fa5ee96a175214e851bf89500951ad12f8ca243c5886158c2866e1be74b`.
All generated report hashes in the seal reverified successfully. This remains
a quality verdict, without a whole-recipe arithmetic or deployment claim.

The Q8/F32-cache parent has also written a complete 1,024-image/4,729-prompt
final export, with manifest SHA-256
`acfeec0a61a2856907287edfb1b9015b000751113d5c173b47dc9e7e5deaa23d`.
The queue is verifying its output closure before the fifth native export.
The early CPU scoring queue has advanced to the Q8 parent; no final Q8
quality result is claimed yet.

The same-library Q8 reuse audit completed without new GPU processes. It
rehashes all retained component identities and raw operands, verifies the
current probe-library links and unchanged model/recipe/source identities,
and independently recomputes all five staging boundary cases plus 317,312
packed dots and F32 bias/GELU outputs. Four boundary cases PASS; the declared
overflowing F16-sum case remains EXPECTED_REJECTION. All packed-dot and
postprocessing checks pass their unchanged numerical rules. The separate
provenance bridge `q8-component-reuse-20261010.json` has SHA-256
`89890c327319b545d28f4b4ddf1f9d69ebb948e8490ef111c1e7123674fb5e40`.
It applies only to shared Q8 kernel components of the two exact Q8-weight
recipes, retains their finite-domain limit and leaves whole-recipe arithmetic
and deployment NOT_RUN. All original receipts remain unchanged.

The main runner has finished validating the Q8/F32-cache export closure and
started the fifth, Q8/mixed-cache export. Early CPU scoring is running for
the Q8 parent. Arithmetic remains queued until all five exports are complete.

The early CPU queue completed successfully and published the Q8/F32-cache
parent's final quality PASS: 1,024 images, 4,729 prompts, 2,000 bootstrap
repetitions, 67 passing checks and 25 coverage-based NOT_APPLICABLE checks.
It reports 20 bad objects among 8,768 high-reference objects, including one
missing and one extra high-confidence object; all 2,316 protected objects
are retained. These measured tails pass the unchanged Q8 profile. Its metrics
SHA-256 is
`17e6c304b37c8cc5ad87296a9622982ec43ba067668242fb4fdb1077c744b482`;
the separate exact-reassessment/hash seal has SHA-256
`df1b723f930c6fc1ce49c95e2028a3dbf0f8b54f342167ab2f07cde5d9e2d8a0`.
The benchmark tool's actual accepted-quality validator reverified both this
report and the F32 parent's report against the current campaign and payload
identities, satisfying the current-quality prerequisite for the exact reused
Q8-parent measurements. The original performance bridge remains unchanged;
fixed regressions and final policy adjudication are still required.


## Complete-JSON queue handoff guard

The final export writer creates manifest.json with a normal, non-atomic write.
A waiting arithmetic queue that tests existence alone can read a partial JSON
manifest. Before any arithmetic or remaining-verification outputs exist,
archive and replace only the two waiting queue processes. Require parseable,
complete JSON and a live named producer for missing/partial receipts, followed
by the existing full identity/hash loader before any capture. Apply the same
readiness guard to remaining-verification receipt handoffs. Do not repeat or
stop final inference, modify production writers, change numerical gates or
rewrite any existing evidence. Verify ready, partial, and dead-producer cases
with temporary CPU-only fixtures, record old terminal statuses and new queue
identities, then resume serialized work with the new arithmetic PID.

The final Q8/mixed-cache export is now complete: 1,024 images and 4,729
prompts, manifest SHA-256
`2f1aef6f34be3b7e75e19f3c64627fdcb42c8c6520a42efe75c514e5c20f1da3`.
Only the two still-waiting queues were replaced; both old handles confirmed
terminal exit 143, without any arithmetic or remaining-verification outputs.
The final inference process was neither stopped nor repeated. The new
complete-JSON guard passed five CPU-only handoff cases: already-complete,
partial-write with live producer, missing/partial with dead producer, and
explicitly incomplete receipt rejection. Ignored preparation/restart receipts
bind the archived versions, helper hashes and new queue PIDs. All five
exports now enter the unchanged full identity/hash loader before captures;
the main runner retains ownership of the last quality score.


## Recipe-specific topology diagnosis

The first fresh F32/mixed-cache unobserved seven-call baseline completed, but
the ignored helper rejected its CUDA-node deltas: 3,464 for first calls and
1,901 for changed prompts, versus its predeclared 3,465/1,902 expectations.
No numerical oracle has failed, and no component or whole-graph verdict was
written. Preserve these outputs and helper versions. Inspect the native
counting implementation and capture the complete development graph inventory
with the existing immutable observer before changing the expectation.
Hypothesis: the shared tuple incorrectly assumes that F32 and Q8-weight text
graphs have identical nodes. Require ordered graph/source-shape evidence to
confirm or reject this, and derive any correction by recipe and full inventory
rather than weakening numerical gates. Final evaluation inputs and outputs
remain unchanged. Resume from verified completed baseline/inventory without
overwriting or repeating final inference.

The text-graph hypothesis was rejected: both recipes have 847-node text
graphs. The root cause is now confirmed by source and two fresh full native
inventories: `make_cuda_backend` combines prediction stages only for quantized
weights with F32 compute. The ignored helper used that 16-graph Q8 tuple for
the F32-weight recipe, which executes 28 split graphs. F32 has 23,992 nodes
and CUDA deltas `(3464,1901,0,1901,1901,3464,1901)`; Q8 has 24,004 nodes and
the originally expected `(3465,1902,0,1902,1902,3465,1902)`. Preserve all
completed baseline/inventory bytes and the failed helper versions in the
separate topology-diagnosis receipt. Correct recipe-specific structural
expectations in parity, topology, auditor derivation and final reconciliation,
including the generated metadata/CONT graph-range checks. Verify both retained
native baselines and inventories before resuming. No numerical tolerance,
model, final sample, production source or final inference is changed.

The corrected ignored helper passed both retained native/inventory checks
with exact output parity and independent per-call compute counts. The archived
helper still reproduces the F32 counter assertion on the same unchanged bytes.
All 250 producer sources rehash unchanged. Derived nonmetadata auditors compile
for both recipes; F32 CONT expects the actual 2,082 nodes across 28 graphs,
while Q8 retains 2,088 across 16. Preparation v4 binds all helper hashes and
the diagnosis receipt before resumed captures. The final reconciler also binds
every ordered matrix node's M/N/K and F32 batch dimensions, and every cache
copy's shape/direction, to the complete inventory before raw cleanup.
The new queue reuses only the separately hash-bound completed baseline and
inventory captures; all numerical captures remain fresh.

The resumed F32 recipe now completes the native input/cache boundary audit,
28-graph topology audit, direct metadata alias/bounds audit and all 16 cache
copy checks. The exact current-library 51-block synthetic cache codec replay
is PASS. Metadata covers 9,460 aliases, 14,532 compute nodes and the freshly
measured leaves; complete F32 matrix arithmetic is now executing. These remain
component results while the remaining compute families and final whole-recipe
adjudication are pending.


## F32 matrix observer parity diagnosis

The first F32 full-matrix capture completed 2,438 native matrix nodes, but
the inherited auditor rejects final observer parity before running numerical
oracles: query boxes/scores differ and three cases also differ in mask bits.
The largest reported box-coordinate difference is 0.007553 pixels. Preserve
all captured operands, outputs, logs and the unmodified auditor. Do not relax
the frozen observer-parity or same-operand rules. Trace the GGML evaluation
callback and CUDA fusion paths, then test whether instrumentation splits a
fused matrix/bias computation. Complete final quality scoring stays unchanged.
If an unchanged-output capture is necessary, use fresh ignored observer
sidecars and fresh output directories, binding any verified replacement
explicitly rather than overwriting this attempt.

The fusion hypothesis is confirmed: the scheduler evaluation callback cuts
the graph immediately after each requested matrix, while CUDA normally fuses
single-column F32 matrix/bias pairs. A separate seven-call development run
with fusion explicitly disabled matches the matrix-observed output exactly
for every nonruntime field in all seven cases, including all differing masks.
This diagnostic environment is not the frozen recipe and cannot qualify it.
Preserve the failed capture and proceed with the independent Q8-weight recipe
using unchanged sidecars, while preparing fusion-preserving F32 capture in
fresh directories. The current final evaluation remains untouched. A future
F32 closure must verify the actual fused result from its exact operands and
bias and retain native output parity; the unfused capture cannot substitute.

A fresh ignored observer now requests singleton matrix/bias endpoints after
the pair, preserving native CUDA fusion. Its real seven-call preflight has
exact nonruntime output parity and unchanged runtime counters: all 2,438
matrices are enumerated, including 50 matrix/bias endpoints, without full raw
dumps. The independent pair oracle derives the existing F64 matrix checker
with only exact captured bias addition; unchanged numerical limits reject an
incorrect result and nonfinite bias in CPU fixtures. Preparation records
these checks and hashes before full capture. Other operations still need
fresh one-to-one coverage, including explicit exclusion of the 50 fused ADD
intermediates from standalone elementwise checking and their coverage by the
native pair oracle. This preflight is not a matrix-arithmetic PASS.

The sibling observer sweep found the same fusion boundary risk in all eight
nonmatrix callback sidecars: callbacks may stop at otherwise unselected
matrices to save later operands. Prepare fresh F32-only derivatives that defer
the 50 paired matrices to their bias endpoint. Elementwise checking excludes
those ADD intermediates and binds their actual combined endpoints to the
independent native matrix/bias report. All other semantic nodes retain the
existing independent oracles, counts, shape/stride checks and numerical limits.
Keep the original Q8 sidecars and every already bound helper unchanged.
After Q8 finishes its serial GPU queue and the native F32 matrix report is
complete, capture the remaining F32 families in fresh directories and reconcile
a disjoint union including all 50 paired ADD nodes before raw cleanup.

All four current final quality reports now PASS for 1,024 images and
4,729 prompts. Q8/mixed-cache absolute and cache-incremental checks each have
67 PASS and 25 coverage-based NOT_APPLICABLE results, with 2,000 bootstrap
repetitions. Absolute matching has 20 bad objects (one missing and one extra)
among 8,768 high-reference objects, with no misses among 2,316 protected
objects. Its cache-incremental comparison has one bad object, zero missing or
extra high-confidence objects and zero protected misses among 2,312 protected
objects. Metrics SHA-256 is
`8c176bf4216a5e8ef431dba9f4fa76fb20cdbc53d263989f6ba957bdf5a3e309`;
the separate exact gate-reassessment/hash seal is
`bb5e395807efd235980b2b25a543e0ea6484431e4915e0468d5fcca3df4c5225`.
The actual benchmark accepted-quality validator reverified this report too.
The main quality runner is terminal exit 0; no final inference is repeated.

The fresh fusion-preserving F32 capture has completed all 2,438 matrices and
50 native matrix/bias endpoints with exact final output parity and passing
independent F64 same-operand checks. Its component report SHA-256 is
`36a82fa1644979078c9f912c2119f80a423a432f7e1f25c59daca0550570d1f7`.
Fresh derivatives of all eight nonmatrix observers compile and are identity
bound; the next queue waits for the serial Q8 recipe to finish before replay.
The F32 reconciliation preparation verifies the disjoint 14,532-compute-node
partition including all 50 paired ADD nodes. All 250 producer sources remain
unchanged. Whole-recipe arithmetic, fixed regression and performance stay
open. The obsolete failed unfused matrix capture was hash-inventoried, rehashed
and pruned: 7,314 raw files, 58,969,866,240 allocated bytes. Its failed parity
receipt, native/observed/no-fusion JSONs, ledgers, sources and logs are retained
immutably; no numerical PASS is inferred from the pruned failed attempt.

All eight new F32 nonmatrix observers now pass actual seven-call preflight
with exact nonruntime output and runtime-counter parity, using zero raw
capture limits. The immutable preflight SHA-256 is
`e2ac3c1d6e872892d865537dcde7e0447ecb29eb71e128af929ff2966e2cc992`. Full numerical captures remain
queued after the serial Q8 recipe; these preflights do not establish arithmetic
PASS. Q8 mixed-cache has now independently passed all 600 F32 and 1,838 Q8
matrices (4,432,408,758 and 936,772,736 checked dots, respectively), including
15,173,800 Q8 staging blocks, plus attention and norm/unary groups. Remaining
Q8 families are live, followed by the queued F32 families, disjoint full-graph
reconciliation and rehash-verified raw cleanup. A fresh remaining-verification
queue waits on the named F32 recovery producer for both complete cleanup
receipts, then owns fixed regressions, CUDA CTest and paired performance.
The documentation checker passes all 132 documents and whitespace is clean.
