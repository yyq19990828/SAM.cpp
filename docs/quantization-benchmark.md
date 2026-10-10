# Quantization benchmarks on application images

[中文](quantization-benchmark_zh.md) · [Quantization options](quantization.md)

Repository COCO measurements are reference results for a particular dataset,
recipe and machine, not universal deployment requirements. Compare your complete
recipe with the original SAM 3 checkpoint or native F32 GGUF on representative
application images and prompts. The application owner decides the acceptable
quality/resource tradeoff.

`tools/benchmark/quantization_benchmark.py` provides ranked-output comparison,
independent performance measurement, precision inspection and opt-in CPU execution
cost diagnostics without v2/v3 quality budgets, minimum sample
counts or performance prerequisites. Export currently targets Linux CPU/CUDA;
original checkpoint export requires the prepared CUDA reference environment.
This tool adds no inference kernel or public cache-selection API.

Prefer the [unified configuration](quantization-config.md): `export` and
`inspect-precision` and `execution-cost` accept `--quantization-config`; `performance` accepts both
`--baseline-config` and `--candidate-config`. Each is checked against the
model's actual weight allocation. Separate CLI settings remain available but
cannot be mixed with configuration files.

## Reference and cases

An original `sam3.pt` reference measures combined conversion/runtime/compression
differences. Native F32 GGUF with F32 compute policy/cache measures incremental
differences from that runtime, and cannot expose errors shared by both runs.
Both are model-generated references, not human ground truth. Agreement cannot
prove production accuracy. Use human labels or review a representative subset
when available, especially small objects and high-cost errors. Public COCO
reports should keep GT AP/mIoU separate from reference agreement and disclose
sample coverage and tuning exposure. Reused development data can support a
comparison but cannot establish an independent holdout claim.

Create `models/application-cases.json`, for example:

```json
{
  "schema_version": 1,
  "samples": [
    {"id": "camera-one", "image": "camera/frame001.jpg", "prompts": ["person", "forklift"]},
    {"id": "camera-two", "image": "camera/frame002.jpg", "prompts": ["person"]}
  ]
}
```

Images are relative to `--input-root`. IDs use unique lowercase letters/digits
separated by hyphens. Prompts are nonempty, unique within each image, and contain
no tabs/newlines. Each run saves normalized RGB PNG inputs; matching hashes bind
the runs to identical decoded pixels. All output directories must be new.

## Export and compare

Build the probe with `SAM_BUILD_TOOLS=ON`. The following models must already be
converted and retain their conversion manifests. Replace the illustrative paths
and options with your own:

```sh
.venv-reference/bin/python tools/benchmark/quantization_benchmark.py export \
  --cases models/application-cases.json --input-root models/application-images \
  --engine native --binary build/cuda/examples/sam_precision_image_probe \
  --model models/sam3-f32.gguf --quantization-config docs/configs/quantization/image-f32-cuda.json \
  --output build/application-f32

.venv-reference/bin/python tools/benchmark/quantization_benchmark.py export \
  --cases models/application-cases.json --input-root models/application-images \
  --engine native --binary build/cuda/examples/sam_precision_image_probe \
  --model models/sam3-custom-q4_k.gguf \
  --quantization-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --output build/application-custom

.venv-reference/bin/python tools/benchmark/quantization_benchmark.py compare \
  --reference build/application-f32 --candidate build/application-custom \
  --output build/application-comparison
```

CUDA export also accepts experimental `--cache f16` and `--cache mixed-q8_0`.
This does not add public model-loading options. `--compute f16` selects the
existing policy, without guaranteeing FP16 for every operator/quantized kernel.

For an original-checkpoint reference, prepare the CUDA environment and separate
runtime source as described in [model verification](validation.md). Set
`SAM3_SOURCE_DIR` and `sam3_weights_dir` there, then use:

```sh
.venv-reference-cuda/bin/python tools/benchmark/quantization_benchmark.py export \
  --cases models/application-cases.json --input-root models/application-images \
  --engine original --checkpoint "$sam3_weights_dir/sam3.pt" \
  --sam3-source "$SAM3_SOURCE_DIR" \
  --sam3-runtime-source build/reference-runtime/sam3-cuda \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output build/application-original
```

## Read the measurements

`report.json`, `report.md` and `cases.json` report union-mask IoU, matched-mask
distribution, missing reference objects, added candidate objects, score/box
changes and per-prompt details. Empty reference prompts are separate from
reference-positive union IoU. Missing/added objects are relative to the reference,
not GT false negatives/positives. Matched statistics exclude unmatched objects:
read them with missing/added rates. No objects gives `NO_DATA`, not perfect
object accuracy. These are descriptive statistics without bootstrap confidence
bounds, a minimum sample-size rule or a holdout qualification.

Detection uses `score > 0.5` by default. `--score-threshold` currently accepts
`[0.5, 1)` because ranked exports retain top-ranked and above-0.5 masks; lower
thresholds need an export-format extension. `--matching-iou` defines spatial
one-to-one association (default `0.5`), not a quality floor. Keep measurement
definitions identical across recipe comparisons.

Optional user advice can be supplied with `--advice models/application-limits.json`:

```json
{
  "missing_reference_rate_max": 0.02,
  "added_candidate_rate_max": 0.03,
  "positive_union_iou_mean_min": 0.90,
  "matched_mask_iou_p05_min": 0.75,
  "score_error_mean_max": 0.05
}
```

These values are examples, not recommended repository limits. Advice is
`WITHIN_USER_LIMITS`, `OUTSIDE_USER_LIMITS`, `INSUFFICIENT_DATA` or `NOT_REQUESTED`.
A valid report succeeds even outside the limits. Changed/missing payloads,
different images/prompts/tokens or source checkpoints produce an error.

## Independent performance benchmarks

`performance` accepts baseline/candidate GGUF models and compute/cache choices
directly. It requires no quality report, COCO annotations, tier or campaign.
Each selected image needs two distinct prompts; the first two define the main
and changed-prompt workloads. Select a small set with `--limit N` or repeated
`--case ID`, never both. This runner supports Linux CPU/CUDA, one backend per
comparison and four threads. CUDA uses visible device 0; select the physical
device through `CUDA_VISIBLE_DEVICES`.

```sh
.venv-reference/bin/python tools/benchmark/quantization_benchmark.py performance \
  --cases models/application-cases.json --input-root models/application-images \
  --binary build/cuda/examples/sam_precision_benchmark_probe \
  --baseline-model models/sam3-f32.gguf --candidate-model models/sam3-custom-q4_k.gguf \
  --baseline-config docs/configs/quantization/image-f32-cuda.json \
  --candidate-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --limit 1 --pairs 1 --warmups 0 --iterations 2 --memory-iterations 1 \
  --output build/application-performance

.venv-reference/bin/python tools/benchmark/quantization_benchmark.py summarize-performance \
  --run build/application-performance/run.json --benefit-min-percent 2 \
  --output build/application-performance-summary
```

Defaults are three AB/BA/AB pairs per case. Each latency process warms up five
times and measures twenty iterations of each workload; separate memory
processes use no warmups and one iteration per workload. Model load and first
inference are recorded separately from steady-state latency. Process memory
peaks include loading and first inference. Adjust `--pairs`, `--warmups`,
`--iterations` and `--memory-iterations` explicitly. For example, `--pairs 1
--warmups 0 --iterations 1` is a workflow smoke check, not stable benefit evidence.
Each child has a default 1,800-second `--timeout`; failures retain logs without
publishing completion. A default single case starts twelve model processes;
weights are read, never copied into the result directory.

Workloads stay separate: `full_image` re-encodes and segments the image;
`changed_prompt` reuses image features and alternates prompts;
`repeated_result` hits the same-prompt result cache and is not model inference.
Reports include per-case and aggregate P50/P95 and independent process pairs.
RSS/GPU memory ratios use maximum process peaks across cases/pairs, not averaged
per-case ratios. PID-bound GPU sampling targets 50 ms plus query time and can
miss short peaks. Latency runs use neither memory nor graph observers. Existing
GPU compute processes prevent launch; detected competitors or observation
errors retain raw data with an `INCONCLUSIVE` measurement status. Latency checks
competitors only before and after each process and can miss transient activity.

`report_validity=VALID` describes usable structure and identities, independently
of `quality=NOT_MEASURED`. Slowdowns, more memory or absent benefits still produce
successful reports. Tags are descriptive: default `--benefit-min-percent 5`
requires that reduction in aggregate P50 with no P95 increase for a latency tag,
including every pair; memory tags use the respective peak ratio. Tags do not
replace per-case tails or guarantee non-regression in other dimensions.
Statuses are `CONSISTENT_REDUCTION`, single-pair `OBSERVED_REDUCTION`, `UNSTABLE`,
`NOT_DEMONSTRATED`, or contaminated `INCONCLUSIVE`; none affects exit status.
Statistics make no significance or cross-hardware claim. Offline summaries
recheck retained images and raw records without loading models. Export/RLE wall
time is not inference latency; feature-cache payload is not process peak memory.

## Precision configuration and execution evidence

Weight, activation, compute and cache choices describe different boundaries.
Quality/performance reports share `requested`, tensor inventories, compute/cache
policies and `execution_evidence`; a model name is not its whole-graph arithmetic.

| Dimension | Setting | Meaning and evidence |
| --- | --- | --- |
| Weights | `weights.precision` with `modules` / `storage_profile`, schema-2 `base_precision` / `module_precisions`, or schema-3 exact `tensor_precisions` | GGUF may mix protected F32, selected Q formats and Q8 fallbacks; inspect policy hash, per-type counts/bytes and assignment reasons |
| Activations | `activation.mode=backend-selected` | No independent native INT8/FP8/F16 activation switch; kernels may transform/quantize RHS; numerical probes are not deployment modes |
| Compute | `compute.mode` | CUDA F16 hints apply to eligible floating-point matmuls/attention; quantized matmuls retain their dispatch; F32 does not imply all private kernel representations are F32 |
| Cache | `cache.mode` | Image levels 0/1/2 use F32/F32/F32, F16/F16/F16 or mixed Q8_0/Q8_0/F32; this is not an LLM KV cache |

Weight storage is fixed at conversion; compute and experimental cache policies
are selected at load. CPU/Metal expose only F32 compute/cache here; reduced
settings require explicit CUDA. Cache settings belong to probes, not a new public
model-loading API. CPU loading promotes F16 weights to F32 and casts quantized
matrices to temporary F32 operands before matmul; smaller files need not reduce
runtime memory proportionally. Exact eligible tensor overrides are supported;
regex/wildcards, mixed F16, independent activation quantization and universal
kernel precision guarantees are not implemented.

Inspect stored weights and resolved policies without inference:

```sh
.venv-reference/bin/python tools/benchmark/quantization_benchmark.py inspect-precision \
  --model models/sam3-custom-q4_k.gguf \
  --quantization-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --output build/application-precision.json
```

The command verifies model/manifest hashes and GGUF tensor headers against the
stored inventory, with `inference_executed=false`. It reads model bytes;
`--backend cuda` does not start CUDA inference. A runtime `arithmetic_profile`
is a producer-reported policy identifier, not a trace of kernel-internal math.
For graph diagnostics, use a separate new directory with
`sam_profile_graph --model ... --image ... --text ... --backend cuda
--cuda-compute f16 --feature-cache mixed-q8_0 --output ...`.
`graphs.json` contains operand/output types, source slots and accumulation/RHS
hints observed at graph allocation boundaries. They cannot prove internal
kernel multiplication/accumulation types. The observer perturbs execution and
must not provide performance timing. Untraced kernel evidence remains
`NOT_COLLECTED`.

## CPU execution cost diagnostics

Use `execution-cost` to inspect the cost of one image and one prompt on CPU,
independently of quality reports and paired performance measurements. The model
must retain its conversion manifest. For example, after building tools with
CUDA/Metal disabled:

```sh
cmake --build build/quant-cpu --target sam_execution_cost_probe -j 2
.venv-reference/bin/python tools/benchmark/quantization_benchmark.py execution-cost \
  --binary build/quant-cpu/examples/sam_execution_cost_probe \
  --model models/sam3-tensor-mixed.gguf --image /absolute/path/to/application.png \
  --text person --quantization-config docs/configs/quantization/image-tensor-mixed-cpu.json \
  --output build/application-cpu-cost
```

This explicitly runs a full image inference once, with CPU/F32 compute/F32
cache and four threads; it is not a schema-only preview. No model weights are
copied into the result directory. The default timeout is 1,800 seconds per child;
set `--timeout` as needed. Validation in this delivery used bounded matrix
fixtures, not a complete-model run or a GPU campaign.

`report.json` and `report.md` bind the model, manifest, binary/libraries, image
and optional configuration hashes. `probe/execution-cost.json` retains raw
graph/node/transfer observations. The JSON separates graph bind and synchronized
compute wall time, Q→F32 casts, matmuls, other operations, metadata, and tensor
upload/download API wall time. Inputs/output types and backend are observed;
accumulation/RHS hints remain requests. Nodes with zero calls were in the graph
snapshot but were not observed executing. Arena peak is the maximum scheduler
tensor arena, excluding weights and backend-private scratch; process RSS is a
separate whole-process peak. Neither is a sum of node output sizes.

Every node is synchronized separately to prevent neighboring work being charged
to one operation. This changes scheduling/fusion and includes dispatch overhead:
`diagnostic_only=true`, `performance_comparable=false`. Node times are already
inside graph time and must not be added again. Transfers exclude host allocation
and weight loading. Graph construction/reserve and preprocessing are not
individually timed. Internal RHS packing may occur inside matmul but is not
separately measured; it and kernel arithmetic remain `NOT_COLLECTED`. These
costs explain a path, not a speedup. Use unobserved `performance` runs for gains;
their default behavior is unchanged.

## Historical results

v2/v3 policies and quality-gated performance qualification are kept only in the
[archive](../tools/archive/precision_v2_v3/README.md); former active entry points
were removed. Historical receipts are unchanged. Completed ranked v2/v3 bundles
still work with `compare`, which checks recorded input/prompt/output identities
without regrading policy or awarding an independent final holdout claim.
Application bundles retain and recheck normalized images; legacy image identity
uses receipt hashes. Comparison does not recertify producer model/binary/kernel
execution.

The [implementation plan](plans/20261010-151710-performance-and-precision-reporting.md)
records tool, archive and precision-evidence verification; no large GPU
acceptance or new hardware speedup claim is included.
