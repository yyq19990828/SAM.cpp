# FP32 Image Validation and CPU/Metal Benchmarks

Created: 2026-10-01 12:16:33 Asia/Shanghai.
Status: cloud code implemented; official model/Meta, Metal hardware acceptance and new measurements deferred to local execution.
Baseline: `2c23a67b58019149efe41c89aa263c1b1e770c65`.
Priority: complete this image-validation matrix before implementing
[M2 video tracking](20261001-115321-sam3-text-video-tracking.md).

## Outcome and scope

Validate official SAM 3 FP32 GGUF image inference on both CPU and Metal, and
publish measured latency and peak memory for both. Refresh FP16 CPU/Metal in the
same acceptance and benchmark batch so all four cells have comparable evidence.
Separate numerical acceptance, actual graph placement and performance results.

The minimal approach is to enable explicit FP32 Metal selection and reuse the
existing GGUF loader, shared model graphs, precision patch, differential validator
and benchmark CLI. Preserve public APIs, GGUF schema, conversion, tokenizer and
frozen M1 tolerances. No new dependency, backend abstraction or benchmark framework
is needed. Video, quantization, other models, new devices and throughput tuning
are outside this plan.

## Verified starting point

| Configuration | Numerical acceptance | Benchmark | Baseline restriction |
| --- | --- | --- | --- |
| FP32 / CPU | Official corpus 7/7 passed; worst normalized L2 0.0005700826798642337 | Missing | None |
| FP32 / Metal | Full model unvalidated | Missing | Runtime and Python validator reject it |
| FP16 / CPU | Official corpus 7/7 passed | 60.588 s / 4.936 GB | Existing supported path |
| FP16 / Metal | Official corpus 7/7 passed | 6.556 s / 2.674 GB | Existing supported path |

These are historical results from the [GGUF acceptance record](20261001-020507-gguf-conversion-loading.md),
not results of this plan. The three existing `metrics.json` receipts were read
while planning; weights, sidecars, reference bundle and reference interpreter
are present locally. Full artifact hashes will be rechecked before execution.

Before implementation, the restriction had three relevant locations:

- `include/sam/internal/runtime/ggml/backends/metal.hpp` rejects all-F32
  checkpoints in `validate_metal_checkpoint`; its separate F32 arithmetic probe
  already tests real Metal execution and must remain enabled.
- `include/sam/internal/runtime/ggml/runtime.hpp` attempts Metal only when
  `fp32` is false. Removing the explicit rejection alone would silently leave
  an explicit FP32 Metal request on CPU.
- `tools/validate_image.py` rejects `precision == "f32"` with backend `metal`
  before running the differential executable.

The loader already allocates F32 GGUF tensors on the selected weight backend.
The shared graph requests `GGML_PREC_F32` for matrix multiplication and attention.
The pinned patch covers precise F32 matrix multiplication and the image model's
32/64-dimensional attention, plus F32 window operators. This supports trying the
existing path; it does not establish full-model FP32 numerical acceptance.

## Selection and precision contract

| Request | Behavior after successful acceptance |
| --- | --- |
| Explicit CPU, either checkpoint precision | CPU, with existing exact F16-to-F32 promotion for FP16 checkpoints |
| Explicit Metal, either checkpoint precision | Metal when available and its precision probe passes; otherwise a clear error |
| Auto, FP32 checkpoint | Keep CPU, preserving the current default |
| Auto, FP16 checkpoint | Keep current compatible Metal selection, otherwise CPU when no Metal device initializes |

Keep Auto's FP32 choice unchanged because this task adds and validates an explicit
backend option; changing applications' default device, memory use and latency is
unnecessary. Users opt into FP32 Metal through the existing backend option. A
registered device that fails the arithmetic probe remains an error, including
under Auto; do not hide numerical incompatibility through a new silent fallback.

FP32 names checkpoint storage, not a claim that every graph tensor has F32 dtype.
Preserve existing masks and operator-specific representations, and never downcast
FP32 weights to pass acceptance. Metal device creation, precision enforcement and
accounting stay in backend/runtime modules; model graphs remain shared.

## Implementation steps

Ship the backend change, validation and documentation together as one completed
image milestone. The following are execution steps, not separate support claims.

1. **Preserve the baseline.** Record source commit/diff, binary and artifact
   hashes, build options and hardware under a fresh ignored
   `build/fp32-validation/<run-id>/` directory. Keep the existing weights,
   references and historical receipts intact. Reuse the current dependency pins.
2. **Enable the requested path.** Remove the blanket checkpoint rejection in
   `backends/metal.hpp`. In `runtime.hpp`, attempt Metal for an explicit Metal
   request regardless of checkpoint precision, while retaining the Auto policy
   above, CPU scheduler fallback and the arithmetic probe. Remove the matching
   blanket rejection in `validate_image.py`; retain provenance and dtype-based
   gates without special Metal tolerances.
3. **Check backend behavior.** Update `tests/test_backend.cpp` to exercise explicit
   FP32 Metal with a Metal-backed F32 weight buffer and verify the selected
   backend, actual scheduled GPU work and numerical output. Preserve Auto FP32
   CPU behavior, FP16 storage policy, unknown-backend accounting and clear failure
   for explicit Metal in a CPU-only build. Reuse `test_precision.cpp` for direct
   GPU arithmetic; do not duplicate it with implementation-mirroring assertions.
4. **Run official acceptance.** Use the original unfused FP32 reference for all
   seven frozen cases on each of FP32/CPU, FP32/Metal, FP16/CPU and FP16/Metal.
   Record 28 case results and all intermediate comparisons. Run the existing
   checkpoint-backed session/lifetime/cache check once on the newly enabled
   FP32 Metal path using the truck/groceries reference images.
5. **Measure the complete matrix.** After numerical checks and other project
   computation finish, run the four configurations sequentially in fresh
   processes using the benchmark protocol below. Preserve raw samples and
   execution counters before writing rounded table values.
6. **Publish verified status.** Update `BENCHMARK.md`, `MODEL_ZOO.md`, `README.md`,
   `docs/architecture.md`, this plan's results and `changelog.md`. Explain the
   explicit FP32 Metal option and unchanged Auto policy. Update M2's image
   prerequisite without claiming that image acceptance validates video graphs.

The normal implementation touches four source/test files and five documentation
files, plus this plan and the M2 prerequisite. Only a reproduced arithmetic or
operator failure justifies expanding the patch/graph scope and its regression
checks. Do not upgrade GGML merely to enable a previously blocked configuration.

## Numerical acceptance and failure diagnosis

Reuse `tests/data/sam3-image-cases.json`: truck/truck, truck/wheel, truck/purple
elephant, groceries/fruit, groceries/bottle, groceries/purple elephant and the
truck crop/resize case. Do not substitute a single visual example or supplementary
community-weight reference for this corpus.

| Check | FP32 | FP16 |
| --- | ---: | ---: |
| Normalized tensor L2 | <= 0.001 | <= 0.02 |
| High-confidence mask IoU | >= 0.98 | >= 0.95 |
| Detection score absolute error | <= 0.02 | <= 0.02 |
| Box error as fraction of image dimension | <= 0.01 | <= 0.01 |

Keep the remaining existing gates: exact token IDs; finite/equal-shaped tensors;
preprocessing maximum absolute error `2/255`; zero-norm tensor absolute error
`1e-5` at reference norm <= `1e-12`; high-confidence threshold `0.6`, low-confidence
threshold `0.4`, output selection threshold `0.5`. Threshold-adjacent queries remain
reported separately. The validator chooses thresholds from checkpoint precision;
FP32 Metal must pass FP32 gates.

Record selected backend, CPU/Metal node counts and graph partitions for every
case. Completion of the intended Metal path requires GPU nodes and zero CPU
compute-graph nodes across the corpus, excluding normal host preprocessing and
postprocessing. Existing FP16 counts are evidence, not hardcoded expected counts
for FP32. If any operators fall back, identify them and their shapes/types; do not
publish a hybrid run as a fully GPU-executed Metal result.

The fragile assumption is that existing precise Metal kernels remain accurate
through the complete FP32 model. If a case fails, locate the first divergent
recorded stage, then the responsible operation, and add one focused regression
for the demonstrated failure. Correct precision, dispatch or layout in the
existing shared layer. If the dependency patch changes, refresh its checksum,
prepared-tree identities and patch documentation; preserve source provenance.
Never relax tolerances, silently narrow weights, or relabel CPU work to pass.

If FP32 Metal remains unsuccessful, retain the explicit restriction for the
published library and report the actual failing stage/operator with its measured
error. CPU benchmark completion can still be recorded, but this plan stays
incomplete and Metal remains unvalidated with a specific reason. Rollback removes
only owned source changes/new artifacts; existing model data need no migration.

## Reproducible inputs and commands

Use the existing reference environment and local official assets. No new package,
service, account or credential is required. Do not re-download or reconvert unchanged
weights. Verify these identities before using the acceptance results:

| Artifact | SHA-256 |
| --- | --- |
| Original `sam3.pt` | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| `sam3-f32.gguf` | `cb13ecd5012a2fe19b06d840049be6daa6b177b35256af8c4afeb12125352486` |
| `sam3-f16.gguf` | `66731fa5def347677f78d7422b81979be0f8e2f7ead941db9a466d1cfa715120` |
| Reference `manifest.json` | `64d471b644615f26350e37a235b06e330e3e16f1d3aa2cdc6cc553b6fe0f814b` |
| Benchmark `truck.jpg` | `941715e721c8864324a1425b445ea4dde0498b995c45ddce0141a58971c6ff99` |

From the repository root, after the implementation changes:

```sh
rtk proxy cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
rtk proxy cmake --build build/cpu --parallel
rtk proxy ctest --test-dir build/cpu --output-on-failure
rtk proxy cmake -S . -B build/metal -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
rtk proxy cmake --build build/metal --parallel
rtk proxy ctest --test-dir build/metal --output-on-failure
rtk proxy build/reference-runtime/venv/bin/python tools/test_tools.py
```

The builds include independent/repeated header compilation and two-translation-unit
linkage. Run the existing downstream consumer in CPU and Metal configurations,
including its precision and window checks. Preserve include guards and the
`sam::sam` INTERFACE target.

Run the official matrix sequentially with fresh output directories; retain one
run ID across the commands and receipts:

```sh
sam_fp32_assets='models/official/3c879f39826c281e95690f02c7821c4de09afae7'
sam_fp32_reference="$sam_fp32_assets/reference-unfused-fp32-corpus"
sam_fp32_run="$(rtk proxy env TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
for sam_fp32_backend in cpu metal; do
  for sam_fp32_precision in f32 f16; do
    rtk proxy build/reference-runtime/venv/bin/python tools/validate_image.py \
      --build-dir "build/$sam_fp32_backend" \
      --model "$sam_fp32_assets/sam3-$sam_fp32_precision.gguf" \
      --reference "$sam_fp32_reference" --backend "$sam_fp32_backend" --threads 4 \
      --output "build/fp32-validation/$sam_fp32_run/validation-$sam_fp32_backend-$sam_fp32_precision"
  done
done
```

Check each command's exit status and the aggregate reports; the loop's last exit
status alone is insufficient. Keep `--allow-supplementary` absent. Missing or
changed provenance blocks original-weight acceptance.

## Benchmark protocol and reporting

Use the same M4 Pro host, Release build, pinned GGML revision/patch and four CPU
threads as the existing protocol. Record actual OS/compiler/SDK, backend build
options, power source/battery, thermal state and concurrent activity for this
batch. Stop project builds/reference computation during timing; do not combine
old FP16 numbers with new FP32 numbers as a controlled precision comparison.

For each configuration, start a fresh CLI process with `truck.jpg` (1800 x 1200),
prompt `truck`, score threshold `0.5`, one initial full-image call and five warmed
full-image calls. Use fresh directories under `/private/tmp` and copy receipts
back only after timing. For example, using the shell variables above:

```sh
rtk proxy build/cpu/examples/sam_image \
  --model "$sam_fp32_assets/sam3-f32.gguf" --image models/fixtures/truck.jpg \
  --text truck --backend cpu --threads 4 --score-threshold 0.5 --repeat 5 \
  --output "/private/tmp/sam-fp32-$sam_fp32_run-cpu-f32"
rtk proxy build/metal/examples/sam_image \
  --model "$sam_fp32_assets/sam3-f32.gguf" --image models/fixtures/truck.jpg \
  --text truck --backend metal --threads 4 --score-threshold 0.5 --repeat 5 \
  --output "/private/tmp/sam-fp32-$sam_fp32_run-metal-f32"
```

Run the two FP16 counterparts with `sam3-f16.gguf` and distinct `cpu-f16` /
`metal-f16` output suffixes. Numerical validation must have passed before timing.
Only repeat benchmarks beyond this batch when a failure, source change or
suspected performance regression justifies a matched, interleaved comparison.

Each table cell reports `timing_ms.warmed_full_image_median` and
`runtime.process_peak_rss_bytes`, retaining all five raw samples. Also record
model-load/initial-call timing, cache-only samples, weight/compute buffers,
backend counters, binary/model/input/patch hashes and measurement order.
Separate cache reuse from new-image inference. Peak RSS spans loading/inference;
it is neither dedicated GPU VRAM nor additive with backend buffer sizes. Keep
the model rows and two hardware/backend header rows requested in `BENCHMARK.md`.

## Completion checklist and results

- [ ] CPU and Metal builds, focused backend checks, CTest, Python tooling and
  downstream integration pass; FP32 Auto remains CPU.
- [ ] All four configurations pass the frozen official corpus, 28/28 cases;
  FP32 Metal shows real GPU execution without CPU graph fallback.
- [ ] FP32 Metal session lifetime, prompt/image invalidation and caching pass.
- [ ] Four benchmark cells have fresh raw samples, hashes and defined memory
  measurements; historical receipts remain available.
- [ ] Support matrix, explicit/Auto behavior, reproduction commands and changelog
  match the evidence; documentation links and `git diff --check` pass.

Planning result: verified the existing receipts and three blocking code paths;
selected the explicit-Metal opt-in approach. No runtime changes, new acceptance
runs, benchmarks, commit or push were performed while writing this plan.

## Cloud implementation record (2026-10-01)

The user requested starting both plans in the cloud and leaving model-related
and Meta/Metal validation locally. Cloud source baseline is `a3d03ec`; no model
checkpoint, reference corpus or new performance result was used in this batch.

- Removed the blanket all-F32 rejection. Explicit Metal now attempts device
  initialization for either checkpoint precision; Auto FP32 still selects CPU.
- Retained the arithmetic probe, scheduler CPU fallback and owned-node accounting.
  Backend tests now check a real Metal-backed F32 weight tensor on matching
  hardware and clear unavailable-device failures for both precisions on CPU-only
  builds. The Metal branch is compiled but not executed in Linux.
- Removed the validator's FP32/Metal gate without changing M1 tolerances or
  placement/provenance checks. Full schema-2 files may also run the frozen image
  suite after full inventory/sidecar validation.
- Updated implementation/support documentation without adding accepted backend
  configurations or altering historical benchmark values.

Local acceptance still requires all four original-weight configurations, actual
GPU placement evidence, and the sequential same-batch measurements specified
above. Cloud compilation and synthetic tests do not complete this milestone.
See the M2 cloud record for shared verification results and remaining video work.

Cloud checks passed: CPU CTest **10/10**, consumer CTest **3/3**, Python tools
**13/13**, independent/repeated headers and document/source checks. FP32 Metal
selection fails clearly on this CPU-only host; GPU-backed numerical checks
remain unexecuted here. This record covers the first cloud implementation batch;
local model and Metal acceptance remain pending.
