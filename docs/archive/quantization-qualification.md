# Historical SAM 3 quantization qualification

Archived v2/v3 and fixed-corpus policy history. These thresholds and commands are not the current quantization workflow. Historical measurements retain their original scope.

## Validation scope

The four exact `image-vision-linear-*` profiles have passed output-behavior
validation on the repository's fixed image set with CPU (with and without BLAS)
and Metal. All four `image-full-linear-*` presets have also passed output
quality on the same corpus with CPU (BLAS enabled) and Metal. Both preset
families also passed the seven-case original-model output-quality checks on
Linux x86_64 CUDA with an RTX 4090. These are bounded
image-set results, not dataset-wide accuracy guarantees. Output checks cover the resulting detections, masks, scores and
boxes. Passing them does not mean every intermediate tensor is identical to the
original checkpoint; quantization changes intermediate values. Validation
criteria for these historical runs are precision-specific rather than one shared
tolerance for every format.

The historical fixed-corpus output acceptance evaluates masks, scores and boxes.
Its v2 deployment extension also requires measured workload benefits. New v3
acceptance separates task quality from optional benefit labels, as described
below. Relative L2 and maximum
absolute errors of intermediate tensors are reported separately for diagnosis
and model selection. Exceeding a tensor-fidelity tolerance does not directly
fail a quantized model. Tokenization, input transforms, shapes, finite values
and backend execution must still be correct. The fixed-corpus output limits are:

| Precision | Minimum mask IoU | Maximum absolute score error | Maximum box-coordinate error (fraction of the corresponding image dimension) |
| --- | ---: | ---: | ---: |
| Q8_0 | 0.96 | 0.02 | 0.010 |
| Q6_K | 0.94 | 0.03 | 0.015 |
| Q5_K | 0.92 | 0.04 | 0.020 |
| Q4_K | 0.90 | 0.05 | 0.030 |

The frozen acceptance corpus uses a detection threshold of 0.5. Checks reject
missing high-confidence objects and newly selected low-confidence queries;
changes near the threshold are reported separately. The [visual examples](../visual-examples.md)
show actual outputs and mask differences at a threshold of 0.2 for every
configuration. They offer a direct comparison alongside full-corpus and
application-dataset validation.

The [v2 acceptance policy](../plans/20261007-200954-precision-acceptance-gates-v2.md)
adds precision-specific quality budgets, ranked COCO mask AP, image-level
confidence bounds and one-to-one object matching. A complete recipe must pass
arithmetic, task quality and workload-specific performance separately. These
results have their own version and preserve the fixed-corpus results above.

Absolute quality compares each recipe with the original F32 checkpoint.
Compressed caches also need an incremental comparison with the same weights,
compute mode and backend using F32 cache; the complete recipe must still fit
its absolute budget. Final evaluation requires at least 1,024 unused images,
2,000 paired-image bootstrap repetitions, negative-prompt checks and zero
protected-object misses. The seven fixed regression cases have no dataset-tail
allowance. A component arithmetic check or a cache's smaller payload cannot
substitute for full-recipe validation or measured peak memory.

One separately frozen [short-F32-dot CUDA recipe](../plans/20261009-172831-cuda-f32-short-dot-candidate.md)
keeps vision weights F32 and quantizes the 220 text/fusion/decoder linears to
Q8_0, with F32 compute and F32 feature cache. Its recorded RTX 4090 result
passes the seven-call active-graph arithmetic audit and independent final
quality on 1,024 images. Paired comparison with the same-checkpoint F32 parent
measures a **19.27% lower GPU process peak** for full-image and changed-prompt
inference. Their p50 latency improves by 3.32% and 7.43%, respectively, below
the 10% latency-benefit threshold. Repeated-result inference has no benefit
label. These results apply to that exact model, CUDA build and workload;
other allocations, cache recipes, GPUs, backends and video require separate
evidence.

The [mixed-cache campaign](../plans/20261009-223907-shortdot-mixed-cache-final-acceptance.md)
separately qualifies two experimental cache-tool recipes on that RTX 4090 CUDA
build. Each uses F32 compute, compresses FPN 0/1 to Q8_0 and retains the F32
detection feature. Both pass complete active-graph arithmetic, fixed regressions,
and absolute and cache-incremental quality on **1,024 unused images / 4,729
prompts**. The applicable aggregate and 55 category checks pass; 25 categories
have insufficient coverage for a per-category claim.

Each fresh 96-process comparison uses the **same weights with F32 cache** as its
own baseline across eight predeclared cases:

| Mixed-cache weights | Full-image p50 latency reduction | Changed-prompt p50 latency reduction |
| --- | ---: | ---: |
| F32 | 11.08% | 19.66% |
| The custom text/fusion/decoder Q8_0 recipe above | 12.38% | 21.23% |

Both earn latency labels for these two workloads only. GPU process peaks rise
by 0.13% and 0.16%, respectively; RSS peaks fall by 14.15% and 14.35%, below the
15% host-memory gate. Neither earns a memory or combined latency/GPU-memory
label, and repeated-result inference has no benefit label. The Q8 parent's
separate 19.27% GPU-memory benefit is not inherited by these comparisons.
These remain experimental cache-tool results; public model loading retains
its existing cache precision.

The Linux v2 workflow uses `tools/archive/precision_v2_v3/freeze_precision_campaign.py`
to bind recipes and inputs before evaluation, `tools/archive/precision_v2_v3/export_precision_outputs.py`
and `tools/archive/precision_v2_v3/evaluate_precision.py` for ranked quality, and
`tools/archive/precision_v2_v3/benchmark_precision.py` for paired latency and memory.
Codec and same-operand operator checks provide separate arithmetic evidence;
see [model verification](../validation.md) and the policy for the requirements.
These native runners depend on Linux library/process inspection and do not
implement Metal host integration.

BF16, W8A8 and FP8 entries in the policy are research targets rather than
additional supported image-inference modes. The legacy schema-3
`image-linear-*` family also quantizes text-encoder linears and remains
diagnostic; it differs from the schema-4 full preset and does not cover fusion
or decoder linears. These image profiles do not support quantized video.

## v3 quality tiers and optional benefits

This section describes the opt-in historical qualification policy. Application
benchmarks use the advisory workflow above; a COCO threshold is not a universal
deployment requirement. Historical campaigns can explicitly select the
[v3 policy](../../tools/archive/precision_v2_v3/sam3-precision-gates-v3.json).
Choose `high-fidelity`, `balanced` or `compact` before inference according to the
application's permitted task loss. The tier is independent of weight, compute
and cache precision; a Q4 recipe does not automatically receive a larger budget.
These are initial engineering budgets, not a guarantee for another task or dataset.

| Tier | Maximum mask AP drop | Maximum union-mask mIoU drop | Task mask IoU floor | Maximum bad-object rate |
| --- | ---: | ---: | ---: | ---: |
| `high-fidelity` | 0.25 pp | 0.20 pp | 0.90 | 0.5% |
| `balanced` | 1.00 pp | 0.50 pp | 0.85 | 1.0% |
| `compact` | 2.00 pp | 1.00 pp | 0.80 | 2.0% |

Arithmetic, task quality and fixed regressions remain required. Task quality
retains confidence bounds, missing/extra objects, size/category coverage and
negative prompts. Protected objects must have reference score ≥ 0.9, area ≥
1,024 pixels and reference-to-GT IoU ≥ 0.75; the candidate must retain a unique
match at GT IoU ≥ 0.5. Score/box errors, tighter reference-mask fidelity and GT
threshold flips are diagnostics. A score change that loses a deployed object
still affects task quality. Compressed caches retain a separate incremental
budget against their exact F32-cache parent.

Performance reports separate measurement validity, non-regression and optional
benefit labels. A 10% latency reduction or 15% memory reduction remains a strong
benefit label; a valid smaller gain is recorded without making quality fail.
Full-image, changed-prompt and repeated-result results remain separate. A
non-regression failure affects deployment choice, not the task-quality verdict.
No performance measurement is required for a quality-only campaign.

The tools remain v2 by default for compatibility. Use `--policy-version 3
--quality-tier balanced` for v3 exports and fixed regressions. The [validation
guide](precision-validation.md#v3-image-acceptance) and [implementation plan](../plans/20261010-101413-precision-acceptance-v3.md)
describe the workflow. All results above remain historical v2/fixed-corpus
evidence; this policy/tool update does not qualify any model under v3.
