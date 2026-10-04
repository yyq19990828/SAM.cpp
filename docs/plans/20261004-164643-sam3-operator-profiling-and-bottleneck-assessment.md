# SAM 3 operator profiling and bottleneck assessment

Created: 2026-10-04 16:46:43 Asia/Shanghai.
Status: complete. Eight representative cells passed strict A/B/A output parity,
individual output quality, trace coverage and sampling guards. The final ranked
assessment is recorded below. No production source or acceptance gates changed.

## Authorization and fixed baseline

The user requested committing the completed pipeline patch first, then doing
operator-level profiling to assess performance bottlenecks. The first action
completed as local commit `434cdb52b8bfcee1a8c483366234e1cef461c7c7`
(`perf: reuse SAM 3 tracker graphs and batch propagation`); the worktree was
clean immediately afterward. No push or production optimization is requested.
The previous source-bound v8 acceptance and complete timing remain in the
[pipeline implementation plan](20261004-040240-sam3-pipeline-execution-and-unified-benchmarks.md).

This task diagnoses why image weight quantization reduces RSS with nearly
unchanged full-image latency on the measured CPU and Metal paths. It does not
change precision, quantization policy, public APIs, supported models/backends,
output gates or production dispatch. Measurements on the current M4 Pro are
hardware observations, not repository platform boundaries.

## Scope and questions

- Identify CPU native/BLAS and Metal costs by operation, source weight versus
  activation, tensor shape, dtype and model stage. Separate weight projections
  from activation attention products, flash attention, normalization, layout
  copies, convolutions, softmax and host graph/transfer/synchronization time.
- Measure quantized-to-F32 CPU casts and compare with actual GEMM costs.
  Check whether text/decoder quantization changes only small fractions of time.
- On Metal, verify actual quantized kernels and dense precision routes, and
  measure GPU execution separately from host submission/wait. Do not infer GPU
  duration from an asynchronous host call or claim a bandwidth/compute bound
  without sufficient evidence.
- Assess profiler perturbation and output parity. Preserve normal fusion and
  scheduling when the available instrumentation permits it. Serialized-node
  measurements must be labelled and cannot be presented as an unmodified
  critical-path percentage.
- Rank concrete optimization candidates by measured time share and likely
  benefit, with evidence limits. Implementation of faster arithmetic or a new
  kernel is a subsequent decision, not part of this diagnosis.

Primary workload: the same official `sam3.pt` image exports, 1800 × 1200 truck
pixels, prompt `truck`, threshold 0.5, 1008 × 1008 model input, four requested
CPU threads and `VECLIB_MAXIMUM_THREADS=4`. Pilot F32 versus full Q4_K on
CPU/BLAS and Metal; extend to mixed F16/F32 and full Q8_0 after the profiler
works, for eight representative cells. Add native-CPU or isolated shape probes
only if necessary to resolve a remaining hypothesis. Complete video timing and
re-running all thirty storage/backend cells are outside this diagnostic scope.

## Approach

1. Keep the committed production files and previously accepted binaries and
   evidence unchanged. Put source snapshots, instrumented dependency copies,
   helper programs, builds and JSON traces under ignored
   `build/operator-profiling/20261004/`. Record source, compiler flags, binaries,
   libraries, input and model identities for every run.
2. Inspect the pinned GGML and installed SDK before choosing hooks. The installed
   command-line tools cannot find `xctrace`; do not install software. GGML's
   scheduler evaluation callback exists but divides graphs and synchronizes
   after observed nodes. It is suitable for diagnostic isolation with explicit
   perturbation controls, not an automatic low-overhead whole-graph profiler.
3. Prefer native CPU timers around existing compute/fusion groups and existing
   barriers, without adding a barrier to every operator. Record fused groups
   once rather than assigning their whole duration to each constituent node.
   Retain CPU/BLAS backend identity and per-stage graph counters.
4. Probe Metal timestamp-counter support. If dispatch sampling is available,
   use it while retaining native groups; otherwise use existing command-buffer
   GPU timestamps with isolated-node/group measurements and clearly document
   scheduling/fusion changes. Report host wall time, GPU time and observation
   overhead separately, with no double-counting of overlapping intervals.
5. Compile only the isolated diagnostic surface. Verify a small graph with
   known operation shapes and positive times, and verify observer-disabled
   output/behavior. Compare each profiled model cell's final scores, boxes and
   masks with its unobserved control; profile metadata alone is not correctness.
6. Run measurements exclusively on AC with power/thermal/sleep and competing
   model/build guards. Keep warmup separate; collect three warm observations
   with matched unobserved controls bracketing each diagnostic process where
   practical. Capture cold costs and total wall/RSS independently. No build or
   numerical job overlaps model sampling.
7. Aggregate operation/shape/dtype/stage totals and top contributors. Check
   coverage against expected graph operations and make unexplained time visible.
   Compare rankings and profiler wall time with unobserved controls. Add narrow
   representative kernel probes only when an attribution remains ambiguous.
8. Record all procedures, failed probes, hashes, raw trace paths and findings in
   this plan. User benchmark guides retain the accepted concise measurements.

## Verification and limits

The starting 97-file production source map matches v8 acceptance: 39 CTests,
65 tool checks, 210 image cases and 864 video frames, with thirty image and
twenty-four targeted tracker timing cells. These are reused receipts, not new
profiling runs. Documentation links/table checks and whitespace passed during
the current commit preflight. Profiling-only changes do not invalidate that
production commit; any future production arithmetic change requires new
behavior/numerical acceptance before performance claims.

Assess timer units, non-empty coverage, final output parity, profiler enabled
versus disabled overhead, fusion membership, CPU/BLAS node accounting (BLAS is
a CPU subset), GPU completion before timestamp reads and peak RSS. Small
operator times near timer/submission overhead must be labelled uncertain.
Operation totals are diagnostic observations; do not add stage medians to
reconstruct the end-to-end median. Sampling and changed scheduling cannot
establish an unmodified hardware utilization or bottleneck percentage alone.

A prior read-only review noted a separate bounded serial-tracker copy of about
5.06 MiB of conditioned features. Preserve it as a possible host-copy
observation, but prioritize the image quantization question. Existing allocator
failure/reset behavior is unchanged; optional OOM recovery is outside scope.

## Results

The isolated project uses the committed SAM archive and a separate copy of the
same pinned, precision-patched GGML. CPU/BLAS node fields are explicitly zeroed
in the private tensor constructor and accumulate group time only when
`SAM_PROFILE_ENABLE=1`. Native CPU groups use existing barriers; BLAS wraps
synchronous calls. All diagnostic libraries/helpers share the changed private
tensor layout. The production dependency and binaries remain unchanged.

The M4 Pro counter probe reported `AtDispatchBoundary=false`,
`AtStageBoundary=true` and the single `timestamp/GPUTimestamp` counter set.
Completed-command-buffer GPU start/end timestamps were valid for a known
compute kernel. The Metal diagnostic backend exports completed GPU interval
sum/union and actual pipeline names; disabled observers skip collection and
related hot-path locks. This task therefore uses serialized scheduler-node
observations on Metal, with separate host and GPU time. It does not claim
native concurrency/fusion percentages.

Release diagnostic CPU and Metal builds succeeded. A known 128 × 128 matrix
product/add and RMS-normalization/scale graph produced correct outputs with
observation enabled and disabled. CPU RMS norm/MUL was recorded once as a
two-operation fusion group. The final model-like external-weight Metal smoke
ran all four operators on actual `MTL0`, returned positive GPU timestamps and
four distinct pipeline names. Current small-graph traces are
`smoke-cpu-weights-v3.jsonl` and `smoke-metal-weights-v3.jsonl` under
`build/operator-profiling/20261004/`.

An early smoke exposed stale GPU attribution when a small node was scheduled
on CPU: the first observer queried the selected Metal backend's prior timestamp
without checking the node's actual backend. The corrected observer queries GPU
time only for actual Metal nodes. Earlier smoke traces remain preserved and are
not performance evidence. The subsequent attribution check passed, and the
external-weight smoke verified non-empty actual GPU coverage. A validation
script initially expected the string `METAL`; this pinned backend uses `MTL0`,
so that script's assertion was corrected without changing arithmetic.

The fixed comparison set is F32, mixed F16/F32, full Q8_0 and full Q4_K on
CPU/BLAS and Metal (eight cells). Using full Q8 and full Q4 keeps the eligible
matrix scope matched. `operator-spec-v1.json` binds 97 production sources,
807 diagnostic/helper sources, each diagnostic library and binary, input and
model manifests. The driver additionally records its own identity. Pilot F32
and full Q4 cells precede the remaining F16/full Q8 cells. Each A1/B/A2 process
uses one cold call plus three uncached warm calls. Initial parity is exact
binary masks, score error ≤1e-5 and box error ≤0.01 pixels; sampling will stop
for diagnosis instead of silently relaxing these limits.

The graph JSONL has six labelled graphs per full-image call (vision, geometry,
text, fusion, detection, segmentation), with run 0 for cold and 1–3 for warm.
It contains operation, shape/strides/dtype, original source metadata, actual
backend, CPU fusion time/member count, and Metal host/GPU intervals plus
pipeline names. GPU interval validity is checked only for actual `MTL*` nodes;
CPU fallback must not inherit prior GPU times. CPU and GPU measurements remain
separate. The first F32 CPU/BLAS and Metal cells passed all A1/B/A2 output
parity, official output-quality and 24-line stage/run trace checks; full Q4_K
sampling is in progress. No conclusions use the raw process-wall polling
duration: observer overhead uses the CLI's matched warm full-image times.

A read-only aggregation review confirmed one accounting qualification: CPU
fusion time belongs to the group head, not a separately timed primitive. The
derived summarizer now marks such categories as root-operation fusion groups
and includes actual backend and fusion membership in shape signatures. It
separates cold records from the three warm observations. The runner's all-run
CPU-microsecond record sum is only a coverage statistic, not per-call latency.
Large trace parsing occurs between model cells or after sampling to preserve
exclusive model execution. Ranked bottleneck findings remain pending.

## Strict parity stop and controlled fusion follow-up

The v1 pilot's F32 CPU/BLAS, F32 Metal and full Q4 CPU/BLAS cells passed all
three arms, original output-quality checks and trace coverage. Full Q4 Metal's
individual arms also passed original output quality and valid trace coverage,
but observer B differed from A1 by 7 of 2,160,000 mask pixels (IoU about
0.99998899), score 2.5212e-5 and at most 0.004723 pixels in box coordinates.
Its A2 control exactly matched A1. Thus its exact-mask and score parity failed;
stop and retain `runs-pilot-v1/operator-matrix-report.json` with SHA-256
`0192d24b95946f7afed421d3dd79c6bf3df50fc217846f662e520fc648ac43b5`.
No F16 or full Q8 cell had run. No tolerance was relaxed.

The next diagnostic protocol controls the arithmetic/fusion path explicitly:
all Metal A1/B/A2 arms jointly set existing
`GGML_METAL_FUSION_DISABLE=1` and `GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1`.
CPU keeps its natural groups. This isolates observation overhead from full-
graph versus split-graph fusion/rewrites; it does not change production defaults
or establish normal fused/concurrent GPU cost percentages. Check Q4 Metal
first under the same strict parity gates, then complete the remaining cells
only if the controlled experiment passes. Compare the unfused controls with
the saved normal controls to quantify the scope difference. Preserve both
protocols and their separate identities.

Successful v1 CPU cells can be reused in the final eight-cell composition:
their binaries, source, model, input and execution environment are unchanged.
The final composition must retain each cell's own receipt/protocol, not edit
the failed pilot into a success. The new driver can select exact cells from
the full fixed specification so passed CPU cells need not be repeated.

Initial v1 warm CPU observations (three warm calls, natural groups): F32
recorded about 7.381 s of operator groups versus 7.407 s compute wall; full Q4
about 7.442 versus 7.469 s. Weight GEMMs cost about 1.581/1.557 s. Full Q4's
weight casts cost about 0.0567 s, while flash attention across vision and other
stages costs about 3.117/3.113 s. This already suggests that avoiding casts
alone has limited whole-image potential. These are measured observations,
not proof that a different quantized arithmetic kernel will preserve quality.

The passed F32 Metal v1 observation highlights a different candidate: vision
elementwise tensors shaped `[1,32,576,144]`. Their MUL/ADD/SUB groups cost about
1.753/0.441/0.441 s in serialized GPU observations, versus about 0.786 s for
all weight matmuls. Source inspection will establish their role and the v2
controlled experiment will check whether this ranking survives. The failed
Q4 Metal trace is exploratory evidence only and is not a qualifying final cell.

The corresponding source path is `sam3_apply_rope` in
`include/sam/internal/models/sam3/vision.hpp`: each complex real/imag slice keeps
first dimension 1, and performs four multiplies, one subtract and one add before
concatenating the pair. For window attention, its Q/K slices are
`[1,32,576,144]`, with frequency broadcasts `[1,32,576,1]`. The actual v1
kernel names lack the `_4` vector variant. Pinned GGML's
`ggml_metal_library_get_pipeline_bin` requires both first dimensions divisible
by four for that variant; `ggml_metal_op_bin` then chooses `nth=1` when output
width is 1 and dispatches `ne01 × ne02 × ne03` groups. This shape therefore
launches 2,654,208 one-thread groups per such operation. This is a concrete
layout/dispatch inefficiency in the observed path; hardware bandwidth or total
natural critical-path utilization is not measured by these timestamps.

The v2 Q4 Metal pilot passed exact binary-mask parity, the original score/box
limits, individual output quality, trace coverage and environment/artifact
guards. Warm medians A1/B/A2 were 5429.80/5970.76/5501.90 ms; observer overhead
relative to the mean control medians was about 9.24%. All arms explicitly used
the same unfused, graph-optimization-disabled configuration.

Its A1 output exactly matched the prior v1 observer B. Relative to the saved
normally fused v1 controls, it reproduced the same 7 changed mask pixels and
score delta 2.5212e-5. This isolates the prior strict-parity stop to the changed
fusion/graph-rewrite path; it does not make the failed v1 cell a success or
change production math. The completed v2 pilot is in `runs-v2-q4metal-pilot/`.

Each v2 warm vision trace confirms 224 MUL, 56 SUB and 56 ADD nodes with the
window-RoPE slice shape `[1,32,576,144]`. Their per-call serialized GPU interval
total was about 2571.9–2622.4 ms. The five remaining cells run in a separate
`runs-v2-rest/` directory; the first two natural CPU cells retain their original
passing v1 evidence. The final composition will distinguish these protocols.

An independent read-only source review confirmed the scalar dispatch derivation
and its scope: the current complex RoPE DAG is not covered by the pinned Metal
norm/add-chain/snake/MoE fusion patterns, and graph reordering does not itself
vectorize the first-dimension-1 views. Default full-graph concurrency may still
change end-to-end latency. A future candidate must therefore be tested without
the observer under default fusion/graph optimization before claiming speedup.

The most direct future RoPE candidate reads the existing checkpoint frequency
table and adjacent real/imag pairs together, producing both outputs without
four separate scalar multiply paths and their intermediates. Merely switching
to `GGML_ROPE_TYPE_VISION` is not justified: that API uses NeoX split-half
pairing and position-derived phases, while this SAM 3 graph uses adjacent
interleaved pairs and preserved per-block `freqs_cis`. Ordering, phases and
floating-point behavior need independent equivalence checks. Layout repacking
is another candidate whose extra copies must be measured. No candidate kernel
or production route was implemented in this profiling task.

## Final eight-cell assessment

All eight unique final cells passed: reuse only the two natural CPU cells from
v1, plus the Q4 Metal v2 pilot and the other five v2 cells. This is 24 A/B/A
processes, 96 full-image calls including cold calls, and 192 non-empty graph
rows in the eight observer traces. Each cell has three warm observations,
exact binary-mask parity and the unchanged score/box limits. Every arm passed
the individual original-reference output-quality check and source/model/
library/binary/input/AC/thermal/exclusive/sleep guards. The old failed v1
report remains complete-but-failed, unchanged and excluded from final Q4 Metal
qualification. All model processes exited and the execution lock was removed.

Canonical machine-readable composition:
`build/operator-profiling/20261004/operator-assessment-final-v2.json`, SHA-256
`5f688108ef844ddcf8b8c383f51a466b7da5fda131f4b7332b9431529a994652`.
It binds the three source reports and eight observer traces by path/hash and
records each cell's protocol, warm controls, overhead, category totals and
coverage. Each observer directory contains `operator-summary-final-v2.json`
with the reviewed warm-only operation/shape/backend/fusion/kernel aggregation.
The derived scripts are `summarize_operators.py` and `compose_assessment_v2.py`;
raw machine-readable run evidence was not edited.

The canonical composition combines attention categories within each warm call
before taking the median; it does not add independent category medians. The
first derived `operator-assessment-final.json` and its views remain preserved
as an earlier statistical view. Corrected per-cell views use
`operator-summary-final-v2.json`; their trace and source-report hashes are
unchanged. This correction is post-processing only, with no repeated sampling.

### CPU/BLAS natural execution

Times below are medians of three warm observations, in seconds. The observed
sum is CPU/BLAS elapsed group time, not summed CPU-core utilization. Fusion
groups belong to their root and are charged once. RoPE binary time covers its
MUL/ADD/SUB shapes, excluding the separate packing/concatenation copies.

| Weights | Observed group sum | Flash attention | Weight matmuls | Separate weight casts | Vision RoPE binary ops |
| --- | ---: | ---: | ---: | ---: | ---: |
| F32 | 7.381 | 3.117 | 1.581 | 0.000 | 0.991 |
| Mixed F16/F32 | 7.282 | 3.106 | 1.528 | 0.000 | 0.985 |
| Full Q8_0 | 7.304 | 3.092 | 1.532 | 0.033 | 0.988 |
| Full Q4_K | 7.442 | 3.116 | 1.557 | 0.057 | 0.994 |

Flash attention is 42.1–42.7% of observed group time; eligible weight matmuls
are 20.9–21.4%. Vision RoPE's arithmetic is 13.4–13.5%. F32 GELU_ERF is about
0.611 s (8.3%) and layout/copy about 0.394 s (5.3%). These distinguish actual
cost from parameter count. Full Q8/Q4 casts are only about 0.45%/0.76%, while
their matrix multiplication still uses F32 operands and the same BLAS route.
Avoiding casts alone therefore cannot explain or deliver a large full-image
speedup. A different quantized matmul kernel could target the larger GEMM share,
but is not evaluated or adopted here.

The recorded CPU group sums account for about 99.6% of measured compute wall
in these cells; about 25–27 ms per call remains scheduler/transfer/timer and
other surrounding work. Whole-image times additionally include preprocessing,
graph setup, downloads and diagnostic JSON writing. Observer warm-median
differences from mean bracketing controls span approximately -1.01% to +0.00%,
within the small run's observed drift; do not interpret negative overhead as
an optimization. No unexplained compute nodes remained in the warm coverage
check; logical views and recorded fusion members were handled separately.

### Metal serialized, unfused diagnostic execution

Every arm jointly disabled fusion and graph optimization; B additionally
isolated compute nodes through the scheduler callback. Times below are medians
of summed, completed GPU interval unions for three warm observations. They
describe this diagnostic path, not default full-graph concurrency/critical-path
shares. Dequantization inside native quantized kernels is included in the
weight-matmul time, so there is no separate comparable CPU-cast column.

| Weights | Observed GPU interval sum | Flash attention | Weight matmuls | Vision RoPE binary ops |
| --- | ---: | ---: | ---: | ---: |
| F32 | 5.342 | 0.299 | 0.776 | 2.946 |
| Mixed F16/F32 | 5.352 | 0.307 | 0.755 | 2.957 |
| Full Q8_0 | 5.322 | 0.307 | 0.727 | 2.957 |
| Full Q4_K | 5.380 | 0.309 | 0.753 | 2.953 |

Vision RoPE's MUL/ADD/SUB cost remains 54.9–55.6% of these GPU observations
across storage formats. Weight matmuls are about 13.7–14.5%; flash attention
about 5.6–5.8%. Q8/Q4 reduce the observed GEMM sum relative to F32 by only
roughly 6.4%/3.0% in this small comparison, touching a minority of total cost.
That measured behavior explains why a large weight/RSS reduction need not
produce a noticeable whole-image speedup. The arithmetic/layout operators
dominating these observations do not become smaller with weight quantization.

The GPU interval sum is about 5.322–5.380 s; observed compute wall is
5.773–5.826 s. The difference includes host encoding/submission/synchronization
and diagnostic work; it is not separately measured idle-GPU or bandwidth time.
B's warm full-image median is 7.88–9.24% above the mean A1/A2 control medians.
This quantifies observer perturbation. No missing physical compute observations
remained; non-compute views had no fabricated GPU timestamps. Standard fused
v1 controls and accepted production timings remain separate evidence.

## Ranked next actions and acceptance boundary

1. **Metal: prioritize SAM 3 RoPE layout or a frequency-table-based fused
   kernel.** The measured hotspot and one-thread-workgroup source path are
   concrete. Compare a minimal layout-packed/native-op candidate against the
   current default graph without the observer, preserving adjacent-pair order,
   checkpoint frequencies and intended floating-point behavior. Check extra
   copy/workspace cost and image/video numerical effects before claiming any
   production speedup. This is the strongest structural candidate found here.
2. **CPU: prioritize flash-attention implementation/tiling and RoPE/GELU
   execution.** Attention contributes about 42% of natural group time, unlike
the much smaller explicit weight casts. Keep FP32/required rounding unless
   a separately qualified arithmetic profile is deliberately introduced.
3. **CPU quantized GEMMs: evaluate independently, below the preceding
   bottlenecks.** Native GGML Q4_K dot products pair weights with Q8_K operands,
   and Q8_0 uses Q8_0 operands; bypassing today's F32 staging can therefore
   introduce activation quantization and different error behavior. It is not
   merely a cache optimization. Require output-quality validation and explicit
   arithmetic-profile reporting for any proposed route. Persistent F32 caches
   can also sacrifice the RSS benefit for at most the small measured cast cost.
4. **Defer additional weight-bit reduction as a latency optimization.** Its
   storage/RSS value is established, but the current dominant operators retain
   floating-point activations and their existing dimensions/layout. Finer kernel
   and layout work has clearer measured targets than lower-bit storage alone.

This task completes diagnosis, not production acceleration. Eight cell outputs
and sources are validated for the stated profiling protocols; only one input,
one host and three warm observations per cell were measured. No universal
hardware bottleneck, throughput guarantee or speedup percentage is asserted.
The existing user performance guides remain unchanged. This task added this
plan to repository documentation; other concurrent working-tree files are
outside its scope and were left untouched. Profiling source/build/trace
artifacts stay in the ignored directory. The local commit `434cdb5` has not
been pushed.

Final local checks passed: 50 Markdown documents' bilingual table/local-link
checks and `git diff --check`. The production source/user performance-guide
diff against `434cdb5` is empty. Root also verified the final composition's
source-report, trace and derived-view hashes, all eight selected cell pass
states, zero unexplained compute-node gaps and the cleared execution lock.
Two concurrently added cleanup plans were observed in Git status and left
untouched; they are not part of this task's profiling delivery.
