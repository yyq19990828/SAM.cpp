# SAM 3 image quantization

This guide covers quantized image checkpoints implemented by the current SAM 3
adapter. The versioned profiles below are SAM 3-specific; they do not prescribe
quantization policies for other model adapters.

Use the [application benchmark](quantization-benchmark.md) to compare custom
recipes on your images and prompts. COCO results are descriptive references;
quality limits are optional application advice. Historical v2/v3 qualification
rules are archived and are no longer current benchmark prerequisites.

## Precision choices and current limits

| Dimension | Current implementation | Configuration boundary |
| --- | --- | --- |
| Weight storage | F32/F16 and Q8_0/Q6_K/Q5_K/Q4_K image linears | One target quantized format per conversion; module selection and fixed floating/Q8 exceptions; no arbitrary per-layer mixed-format recipe |
| Activations | Mostly F32 graph buffers; backend kernels may stage/quantize operands internally | W8A8/FP8 tools are studies/probes, not complete deployable model profiles |
| Arithmetic | CUDA F32/F16 policies; quantized kernel dispatch depends on type/shape/backend | A policy name does not guarantee one operand/accumulator dtype for the whole graph; F16 hints target supported dense operations |
| Feature cache | F32 plus experimental F16/mixed-Q8_0 image caches | Probe options on CUDA; mixed Q8 keeps the low-resolution detection feature F32; public loading retains existing cache precision |

Weight, activation and cache formats describe stored/represented values.
Arithmetic separately describes operands, multiplication, accumulation and
output types; it is not another independent tensor to quantize. Cached features
are retained activations with a distinct lifetime and storage policy. These
choices are related, and the backend must support the complete combination.

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
`--quantize-modules` use `image-modules-linear-{precision}-v1` and need their
own recipe measurements, including checks on application data. Preset results do
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

## Evaluation and historical evidence

Use the [application benchmark](quantization-benchmark.md) to compare reference
agreement and measure performance independently on representative images.
Quality advice and performance tags never veto a valid report. Token identity,
finite outputs, input provenance and correct backend execution remain required.

The [historical qualification archive](archive/quantization-qualification.md)
retains earlier fixed-corpus/v2/v3 criteria and scoped measurements. Those
policies are retired from current entry points; earlier receipts are unchanged.
Published COCO and hardware results are reference evidence, not a production
accuracy guarantee or a prerequisite for running a performance benchmark.

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
