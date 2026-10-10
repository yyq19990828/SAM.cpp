# SAM 3 image quantization

This guide covers quantized image checkpoints implemented by the current SAM 3
adapter. The versioned profiles below are SAM 3-specific; they do not prescribe
quantization policies for other model adapters.

## Select components

SAM 3 image weights support the following components individually or in
combination. One conversion uses a single target quantization format;
component selection determines weight storage, while matrix staging follows
the backend arithmetic profile.

| Module name | Scope | Eligible linear matrices |
| --- | --- | ---: |
| `vision` | ViT visual encoder | 128 |
| `text` | Text Transformer and feature resizer | 97 |
| `fusion` | Image/text fusion encoder | 36 |
| `decoder` | Detection, geometry prompt encoding, scoring and mask heads | 87 |

`--quantize-modules text,decoder` quantizes target matrices only in those
components; other components stay F32. Embeddings, positional parameters,
biases, normalization, convolutions, certain small matrices and scalar-output
projections remain F32. Selecting geometry-weight storage does not add point
or box prompting APIs; the current task remains text-prompted image segmentation.

Two fixed preset families are available: `image-vision-linear-{precision}-v1`
quantizes vision linears, while `image-full-linear-{precision}-v1` covers 348
target matrices across all four components. Full-component quantization retains
the floating-point exceptions above. GGUF quantizes stored weights; shared
graphs primarily retain F32 activations. With explicit CUDA F16 compute, image
convolutions directly produce F16 expanded columns for the existing F16-input,
F32-output matrix kernel, reducing temporary memory. Other native GPU kernels
may stage matrix operands in lower precision. Video tracking state is outside
these image profiles.

The two preset families have separate validation results; see
[model support](../MODEL_ZOO.md). Custom combinations selected with
`--quantize-modules` use `image-modules-linear-{precision}-v1` and require their
own recipe validation, including checks on application data. Preset results do
not qualify arbitrary custom selections, even when all four components are
selected. Do not combine this option with `--storage-profile`.

## Profiles and storage trade-offs

GGUF schema 3 provides four vision-only profiles. The names and allocations are
exact: pass the matching profile explicitly during conversion.
Full and custom module profiles use SAM schema 4 within the same GGUF v3
container, recording component selection in the file. Changing components or
precision requires a new conversion from the original checkpoint.

| Precision | Vision preset storage profile | Quantized ViT weights in the vision preset | Vision preset GGUF size | Smaller than example F32 | Full preset GGUF size |
| --- | --- | --- | ---: | ---: | ---: |
| Q8_0 | `image-vision-linear-q8_0-v1` | 128 matrices in Q8_0 | 2.07 GB | about 38.7% | 1.10 GB |
| Q6_K | `image-vision-linear-q6_k-v1` | 96 matrices in Q6_K; 32 MLP `lin2` matrices in Q8_0 | 2.00 GB | about 40.8% | 0.95 GB |
| Q5_K | `image-vision-linear-q5_k-v1` | 96 matrices in Q5_K; 32 MLP `lin2` matrices in Q8_0 | 1.96 GB | about 42.0% | 0.86 GB |
| Q4_K | `image-vision-linear-q4_k-v1` | 96 matrices in Q4_K; 32 MLP `lin2` matrices in Q8_0 | 1.92 GB | about 43.0% | 0.79 GB |

Q8_0 uses an 8-bit block representation. Q6_K, Q5_K and Q4_K use nominally 6,
5 and 4 bits per value, plus block scales and metadata. The example sizes compare
files made from the same SAM 3 image checkpoint: the F32 GGUF is about 3.37 GB.
Actual sizes depend on the checkpoint and container metadata. These are mixed storage models. Vision presets keep every tensor outside
the listed ViT linears in F32, including the complete text encoder; full presets
cover additional component linears while retaining the floating-point exceptions.
K blocks require 256-element rows. When `vision` is selected, the 32 MLP
`lin2` matrices have canonical `ne[0]=4736` and use the fixed Q8_0 fallback.
This allocation is not configurable; custom selections without `vision`
contain none of those fallback matrices.

Choose Q8_0 when keeping more weight precision is the priority. Q6_K and Q5_K
reduce file size further. Q4_K is the smallest of these examples. Lower-bit
storage does not guarantee faster inference; compare output quality and memory
on the images, prompts and backend you plan to use.

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
changes near the threshold are reported separately. The [visual examples](visual-examples.md)
show actual outputs and mask differences at a threshold of 0.2 for every
configuration. They offer a direct comparison alongside full-corpus and
application-dataset validation.

The [v2 acceptance policy](plans/20261007-200954-precision-acceptance-gates-v2.md)
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

One separately frozen [short-F32-dot CUDA recipe](plans/20261009-172831-cuda-f32-short-dot-candidate.md)
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

The [mixed-cache campaign](plans/20261009-223907-shortdot-mixed-cache-final-acceptance.md)
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

The Linux v2 workflow uses `tools/maintenance/freeze_precision_campaign.py`
to bind recipes and inputs before evaluation, `tools/validation/export_precision_outputs.py`
and `tools/validation/evaluate_precision.py` for ranked quality, and
`tools/benchmark/benchmark_precision.py` for paired latency and memory.
Codec and same-operand operator checks provide separate arithmetic evidence;
see [model verification](validation.md) and the policy for the requirements.
These native runners depend on Linux library/process inspection and do not
implement Metal host integration.

BF16, W8A8 and FP8 entries in the policy are research targets rather than
additional supported image-inference modes. The legacy schema-3
`image-linear-*` family also quantizes text-encoder linears and remains
diagnostic; it differs from the schema-4 full preset and does not cover fusion
or decoder linears. These image profiles do not support quantized video.

## v3 quality tiers and optional benefits

New campaigns can explicitly select the [v3 policy](../tests/data/sam3-precision-gates-v3.json).
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
guide](validation.md#v3-image-acceptance) and [implementation plan](plans/20261010-101413-precision-acceptance-v3.md)
describe the workflow. All results above remain historical v2/fixed-corpus
evidence; this policy/tool update does not qualify any model under v3.

## Convert a checkpoint

Use Python 3.12 and the pinned reference requirements. Set `sam3_weights_dir`
to the directory containing the original `sam3.pt` checkpoint and BPE file.
The destination must be a new file; the converter publishes its GGUF manifest
beside it and will not overwrite an existing output.

Linux x86_64 CUDA reference runs use
[requirements-linux-cuda.lock](../tools/requirements-linux-cuda.lock) in place
of `tools/requirements.lock`.

```sh
python3.12 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r tools/requirements.lock
sam3_weights_dir=/absolute/path/to/sam3

.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task image --precision q8_0 \
  --storage-profile image-vision-linear-q8_0-v1 \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-image-vision-q8_0.gguf
```

Q8_0 uses the locked `gguf==0.19.0` writer and does not need the native helper.
For a K format, build the repository's row quantizer against its pinned GGML
source, then provide the helper's absolute path. For example, to make Q6_K:

```sh
cmake -S . -B build/quant-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DSAM_BUILD_EXAMPLES=ON
cmake --build build/quant-cpu --target sam_quantize_rows --parallel

.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task image --precision q6_k \
  --storage-profile image-vision-linear-q6_k-v1 \
  --quantizer "$PWD/build/quant-cpu/examples/sam_quantize_rows" \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-image-vision-q6_k.gguf
```

Use the matching `q5_k` / `image-vision-linear-q5_k-v1` or `q4_k` /
`image-vision-linear-q4_k-v1` pair to create those formats. The helper must be
built from the pinned GGML revision and supplied as an absolute executable
path; the converter checks its identity. The converter rejects a profile that
does not match the selected precision and image task. If `--storage-profile` is
omitted, the legacy broad diagnostic profile is selected, so specify the
vision-only profile when that is the intended allocation.

For the full preset, use `--storage-profile image-full-linear-q6_k-v1` instead.
To quantize text and decoder components only:

```sh
.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task image --precision q6_k --quantize-modules text,decoder \
  --quantizer "$PWD/build/quant-cpu/examples/sam_quantize_rows" \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-image-text-decoder-q6_k.gguf
```

Module names are case-sensitive; duplicates and unknown names are rejected.
CLI ordering is normalized before writing the file. The sidecar manifest lists
each tensor's module, actual storage type and quantization or floating-point
retention reason. Loaded schema-4 models report the selected components through
`ModelInfo::quantization_modules`.

Use `tools/validation/validate_image.py --allow-custom-quantization` for a separate
reference comparison of custom combinations; its report remains diagnostic.
See [model verification](validation.md) for the comparison workflow.

## Run inference

Use the `sam_image` executable from a CPU, Metal or CUDA build. The output directory
must be new. For example:

```sh
build/cpu/examples/sam_image \
  --model models/sam3-image-vision-q8_0.gguf \
  --image image.jpg --text truck --score-threshold 0.2 \
  --output outputs/truck-q8 --backend cpu
```

Use the matching build and pass `--backend metal` or `--backend cuda` to select
a GPU backend. CUDA accepts `--cuda-device N` as an index among visible devices.
Quantized `auto` selection currently chooses CPU.

## Backend behavior

GGUF precision describes stored weights, not end-to-end arithmetic. On CPU,
packed quantized weights stay resident, while the shared graph creates
transient F32 casts for `MUL_MAT`; this extra workspace can affect peak memory.
`ModelInfo` reports this as `ggml-quantized-weights-f32-v1`. Metal uses native
quantized kernels with half staging and reports `ggml-quantized-native-v1`, so
its arithmetic path and memory use differ from CPU. CUDA also keeps packed
weights resident, uses native MMVQ/MMQ with RHS Q8_1 staging on the qualified
device, and reports `ggml-quantized-cuda-native-v1`. Explicit CUDA rejects
CPU/Metal/BLAS compute fallback. Profile qualification is backend-specific;
see the model support table before deployment.
See the [GGUF schema reference](gguf.md), [model support](../MODEL_ZOO.md) and
[measured image performance](../BENCHMARK.md) for the file contract, current
support boundary and hardware-specific measurements.
