# Quantization configuration

[中文](quantization-config_zh.md) · [Quantization](quantization.md) · [Application benchmark](quantization-benchmark.md)

Use one JSON configuration for SAM 3 image weight conversion, reference exports,
precision inspection and performance benchmarks. It describes four separate
choices: stored weights, activation policy, compute policy and retained
image-feature cache. A valid configuration identifies an implemented path;
it does not establish hardware compatibility, output quality or a speedup.

## Configuration format

The [custom Q4 example](configs/quantization/image-q4-vision-text-cuda.json) is:

```json
{
  "schema_version": 1,
  "kind": "sam-quantization-config",
  "task": "image",
  "backend": "cuda",
  "weights": {"precision": "q4_k", "modules": ["vision", "text"]},
  "activation": {"mode": "backend-selected"},
  "compute": {"mode": "f16"},
  "cache": {"mode": "mixed-q8_0"}
}
```

Every axis and the explicit backend are required. Unknown fields, duplicate JSON
keys and unsupported modes fail. For quantized weights, specify `modules` or
`storage_profile` explicitly. `modules` selects any nonempty subset of
`vision`, `text`, `fusion`, `decoder`, in canonical order after resolution.
All eligible matrices in that selection use one target Q format. Other tensors
stay F32; the fixed K-block Q8 fallback remains part of the weight policy.

To use an existing preset, replace `weights` with, for example:

```json
{"precision": "q4_k", "storage_profile": "image-full-linear-q4_k-v1"}
```

`image-vision-linear-*`, `image-full-linear-*` and custom
`image-modules-linear-*` are distinct allocations. Selecting all four modules
does not rename a custom allocation to the full preset. A profile and an
explicit module list may appear together only when they agree. Resolved
configurations include the selected modules; schema-3 GGUF files still have no
schema-4 `quantization_modules` wire metadata. The profile defines exact tensor
eligibility, including differences between legacy and modular allocations.

F32/F16 weight storage uses `{"precision":"f32"}` or
`{"precision":"f16"}`; quantized module selectors are unavailable for these.
The old CLI keeps its defaults. JSON quantized configurations require an
explicit allocation to avoid silently selecting the legacy vision/text policy.

## Supported combinations

| Choice | CPU | Metal | CUDA |
| --- | --- | --- | --- |
| Image weights | F32/F16/Q8_0/Q6_K/Q5_K/Q4_K | Same stored formats | Same stored formats |
| Independent activation mode | `backend-selected` only | `backend-selected` only | `backend-selected` only |
| Compute policy | `f32` | `f32` | `f32` / `f16` |
| Image-feature cache | `f32` | `f32` | `f32` / `f16` / `mixed-q8_0` in private image probes |
| Current export/performance runner | Supported path | Unavailable; inspection only | Supported path |

This table describes code paths, not new hardware qualification. CPU loading
promotes F16 weights to F32; quantized CPU matrices use temporary F32 weights.
CUDA F16 selects eligible reduced-operand/attention policies, while quantized
kernels keep their backend dispatch. It cannot promise W4A16 or FP16
accumulation throughout the model. `backend-selected` does not assert that all
activations have one dtype. Kernel staging and accumulation need separate
execution evidence.

The cache stores host image-encoder features reused by changed text prompts;
it is not an LLM KV cache. `mixed-q8_0` stores feature levels 0/1 as Q8_0 and
level 2 as F32. Changing compute/cache settings can reuse the same GGUF when
the weight allocation matches. Changing weight precision or module allocation
requires conversion from the original checkpoint.

The following requests fail instead of falling back silently:

| Request | Current result |
| --- | --- |
| Different Q formats per module/layer/tensor | Not implemented; one format plus module selection is available |
| Explicit INT8/FP8/F16 activation mode | Not implemented; studies/probes do not enable a native mode |
| CPU/Metal F16 compute or reduced image cache | Unsupported by this configuration |
| Reduced cache through the public C++ API | Unavailable; private native image probes only |
| Metal image export/performance through this runner | Unavailable; inspect the requested policy without inference |
| Video tracking/memory policy in this file | Outside the image configuration; existing video workflow is separate |
| Existing GGUF with a different weight allocation | Error before inference; reconvert or select its matching configuration |
| Backend `auto` or unknown axes/options | Error; explicit backend and known fields are required |

The public C++ `BackendOptions` still selects backend/threads/device and CUDA
compute policy; it has no JSON loader, independent activation option or reduced
cache selector. `validate --context public-api` checks whether these settings
can be represented there; it does not load a model or create that API.

## Validate and use the configuration

These read-only commands need only Python's standard library and never
initialize a GPU:

```sh
python3 tools/quantize/quantization_config.py capabilities
python3 tools/quantize/quantization_config.py validate \
  --config docs/configs/quantization/image-q4-vision-text-cuda.json
```

Use `--context conversion`, `inspect`, `benchmark` (default), `public-api` or
`original-reference` to check the intended entry point. An original-checkpoint
reference requires CUDA F32 weights, compute and cache. `--output NEW.json`
saves a resolved configuration and its canonical/source hashes without
overwriting a file. No model, dataset or measurement is required for validation.

Conversion consumes the weight axis and records the other three axes as
intended runtime requests. It performs no model inference and does not change
GGUF weight bytes according to compute/cache settings:

```sh
.venv-reference/bin/python tools/convert/convert_sam3.py \
  --quantization-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --checkpoint /absolute/path/to/sam3.pt --bpe /absolute/path/to/bpe_simple_vocab_16e6.txt.gz \
  --quantizer "$PWD/build/quant-cpu/examples/sam_quantize_rows" \
  --output models/sam3-custom-q4.gguf

.venv-reference/bin/python tools/benchmark/quantization_benchmark.py inspect-precision \
  --quantization-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --model models/sam3-custom-q4.gguf --output build/custom-precision.json
```

Q6_K/Q5_K/Q4_K still require the pinned native row quantizer. Q8_0 does not.
With a configuration file, do not also pass converter `--task`, `--precision`,
`--storage-profile`, `--quantize-modules`, or runtime `--backend`, `--compute`,
`--cache`, `--activation`. Conflicting sources are errors even when values match.

Quality exports use the same `--quantization-config` option. Paired performance
runs take two configurations; both must use the same backend:

```sh
.venv-reference/bin/python tools/benchmark/quantization_benchmark.py performance \
  --cases models/application-cases.json --input-root models/application-images \
  --binary build/cuda/examples/sam_precision_benchmark_probe \
  --baseline-model models/sam3-f32.gguf --candidate-model models/sam3-custom-q4.gguf \
  --baseline-config docs/configs/quantization/image-f32-cuda.json \
  --candidate-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --limit 1 --pairs 1 --warmups 0 --iterations 2 --memory-iterations 1 \
  --output build/application-config-performance
```

Provide both files or use the existing separate CLI settings; do not mix them.
The example is an opt-in small measurement, not a required acceptance run.
Reports bind the resolved configuration to the existing model's conversion
manifest and record source/configuration hashes. Conversion-time runtime
requests are provenance, not a permanent restriction on how that GGUF can run.
New runs may choose a different supported compute/cache policy while retaining
identical weights. Offline reports read their saved configuration and do not
require the original JSON file. Requests, producer-reported runtime identifiers,
graph types and uncollected kernel arithmetic remain separately identified.

Reference agreement and optional user advice stay separate from performance
measurements and benefit tags. Neither COCO scores nor a speedup requirement is
introduced by this configuration.

Future work is scoped in the [mixed weight plan](plans/20261010-155435-mixed-weight-quantization.md)
and [native activation plan](plans/20261010-155435-native-activation-quantization.md).
Their proposed modes are not selectable in the current configuration.
