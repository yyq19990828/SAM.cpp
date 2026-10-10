# Complete frozen active-graph arithmetic across remaining performance inputs

Created: 2026-10-09 08:51:52, Asia/Shanghai.

## Scope

Extend the frozen custom-Q8 CUDA active-graph arithmetic audit from the
two independently checked development inputs to the six remaining
predeclared calibration/development performance inputs. All eight already
pass exact-operand Q8 and F32 matrix checks. Keep the campaign, diagnostic
model, image binary, CUDA library, graph identities, numerical/bitwise
thresholds and final quality result unchanged. Use no final evaluation or
unopened reserve input for tuning. Exclusive-GPU performance is a separate
gate and remains unmeasured while the remote-desktop GPU process is active.

## Approach and steps

1. Generate case-bound wrappers for the already reviewed operator verifiers
   and graph reconcilers. Bind each to its predeclared image/prompt, native
   output, independent graph inventory and eight-case matrix reconciliation.
   Preserve each wrapper's source identity before generating a receipt.
2. Sequentially capture attention, normalization/activation, elementwise,
   spatial/reduction, pointwise/repeat, concat, gather/pool/F16 copy and CONT
   on the exact frozen binary. Use fresh per-case directories on the
   separate mounted evidence volume. Independently recompute each group
   under its frozen gate, rehash retained payloads and reconcile all node
   order, shapes, types and graph identities.
3. Audit metadata alias/buffer relationships and leaf inventory for each
   independent graph. Reconcile 3,461 compute nodes plus 2,615 metadata
   nodes and 1,155 leaves per input. Promote the bounded eight-input
   active-graph arithmetic result only if every case passes and a final
   aggregate rechecks all eight receipts; leave whole-recipe scope,
   performance and deployment separate.

## Verification

Run the frozen binaries and isolated Python verifiers, then rehash all
evidence in a read-only aggregate. Run the isolated Python tool suite,
documentation checker and `git diff --check` after completed cycles.
Clean disposable intermediate files, retaining the current verified build,
model, graph inventories, raw captures and immutable receipts. Monitor
space on the separate evidence volume before each large capture.

## Results

All six additional predeclared inputs passed their own case-bound full-graph
audit under the unchanged arithmetic rules. The `perf-source-middle`,
`perf-source-large`, `perf-instances-few` calibration input,
`perf-instances-many`, `perf-objects-small` and `perf-objects-large`
development inputs each reconciled all 3,461 compute nodes, 2,615 metadata
nodes and 1,155 leaves to an independent graph inventory. For every input,
all 87 attention, 317 normalization/activation, 1,509 elementwise, 76
spatial/reduction, 242 pointwise/repeat, 132 concat, 15 gather/pool/F16-copy
and 564 CONT nodes passed their existing numerical or exact-bit rule. The
previous 519 matrix nodes per input remained bound to the eight-input
Q8/F32 matrix reconciliation. Final observer outputs preserved each input's
native masks, boxes, ranking and scores, excluding runtime measurements.

The six per-input full-graph receipts under
`build/q8-integrated-arithmetic-20261009/` have these SHA-256 digests:

| Performance input | Full-graph receipt SHA-256 |
| --- | --- |
| `perf-source-middle` | `884377d711f56a71c723e6e35716da8d9df9ae2fe63278ff617035f882ff82ff` |
| `perf-source-large` | `9495b54b05949db834caddcd7f77343cd31cdabee5fa8ba149cfe37398b817ea` |
| `perf-instances-few` | `abb8d200f410d501e0c6a6681894023db1f284d158d59c45bda3dd36ff1b7ab5` |
| `perf-instances-many` | `9ad2b9b293edd3156700a28c13804565feed25d9e8fc0d47c517dc35dec87163` |
| `perf-objects-small` | `b339e07fea896a6c7c0f17e609820d0461007dc1901af314dce361d074fcd40d` |
| `perf-objects-large` | `5ba04e5a7a6b14140101e7cc5efb8247ef3aa546eac8d2f39173fee63701b415` |

The final read-only aggregate also checked the earlier `perf-source-small`
and `perf-negative` full-graph receipts. All eight together cover 27,688
compute-node instances, 20,920 metadata-node instances and 9,240 leaves;
4,152 of the compute instances are matrix nodes with 16,473,472,072
previously checked same-operand Q8/F32 dots. It rehashed 34,392 distinct
raw payload files and 515 identity files, finding no altered or conflicting
evidence. `build/q8-integrated-arithmetic-20261009/eight-full-graph-reconciled.json`
has SHA-256 `b13de109bd53413f7814410192c4ff29fbdfe5ee7c494206f982531d6bc1ee51`.
Its precise status is `performance_input_arithmetic_status: PASS`, scoped
to these eight predeclared calibration/development inputs. It keeps
`whole_recipe_arithmetic_status`, exclusive-GPU performance and deployment
at `NOT_RUN`; this audit did not traverse the full final-quality population
or provide a clean-GPU benchmark.

The Git-ignored `build/q8-integrated-arithmetic-20261009/many-external/`
link points to `/mnt/SSD1/samcpp-validation` and holds about 480 GiB of
retained raw evidence and staging. Those files are available only in this
validation workspace. About 161 GiB remains free on that volume and 1.2 GiB
on the repository volume. The accidental no-capture trial output from the
first additional input was removed before its real capture; no required
raw evidence or verified build output was deleted. No model, campaign,
threshold, final evaluation input or unopened reserve input changed.

The completed validation cycle passed the isolated Python tool suite
(170/170), the bilingual/local-link documentation checker (110 documents)
and `git diff --check`. All eight final active-graph receipts and the
cross-input aggregate remain immutable under the Git-ignored `build/` path.

Later workspace cleanup (2026-10-09): after confirming 133 associated
component/graph receipt statuses were `PASS`, the 69 historical top-level
`/mnt/SSD1/samcpp-validation/*-capture/` directories were reduced to their
`nodes.tsv` inventories. Their 48,075 raw `.bin`, `.f16`, `.f32` and `.q8_0`
payload files (about 477 GiB apparent size) are no longer locally available
for rehashing. The original JSON receipts and reconciliation records remain
unchanged in `build/q8-integrated-arithmetic-20261009/`. The corrected-candidate
archive under `q8-mmq-rounded-candidate-20261009/` was not touched.

The same cleanup removed approximately 70.7 GiB of earlier one-input
`capture-full`, `negative-capture`, F32 and non-matrix `*-full` raw payloads
under `build/q8-integrated-arithmetic-20261009/`, after checking each
corresponding verification or graph reconciliation status as `PASS`.
Their `nodes.tsv` inventories, diagnostic code and immutable receipts remain;
these removed local raw payloads are likewise no longer rehashable.
