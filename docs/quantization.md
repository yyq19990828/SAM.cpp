# SAM 3 image quantization

This guide covers quantized image checkpoints implemented by the current SAM 3
adapter. The versioned profiles below are SAM 3-specific; they do not prescribe
quantization policies for other model adapters.

## Select components

SAM 3 image weights support the following components individually or in
combination. One conversion uses a single target quantization format;
component selection does not change activation precision.

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
the floating-point exceptions above and does not quantize activations or video
tracking state.

Repository numerical acceptance focuses on those two preset families; see
[model support](../MODEL_ZOO.md) for their status. Custom combinations selected
with `--quantize-modules` use `image-modules-linear-{precision}-v1` and need
validation on application data. They remain diagnostic even when all four
components are selected. Do not combine this option with `--storage-profile`.

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
quality on the same corpus with CPU (BLAS enabled) and Metal. These are bounded
image-set results, not dataset-wide accuracy guarantees. Output checks cover the resulting detections, masks, scores and
boxes. Passing them does not mean every intermediate tensor is identical to the
original checkpoint; quantization changes intermediate values. Validation
criteria are precision-specific rather than one shared tolerance for every
format.

Final output quality determines quantized acceptance. Relative L2 and maximum
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

The broader `image-linear-*` family also quantizes text-encoder linear weights
and remains diagnostic. These schema-3 profiles are image-only; quantized video
is not supported by them.
The legacy `image-linear-*` family is distinct from the schema-4 full preset;
it does not cover fusion or decoder linears.

## Convert a checkpoint

Use Python 3.12 and the pinned reference requirements. Set `sam3_weights_dir`
to the directory containing the original `sam3.pt` checkpoint and BPE file.
The destination must be a new file; the converter publishes its GGUF manifest
beside it and will not overwrite an existing output.

```sh
python3.12 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r tools/requirements.lock
sam3_weights_dir=/absolute/path/to/sam3

.venv-reference/bin/python tools/convert_sam3.py \
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

.venv-reference/bin/python tools/convert_sam3.py \
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
.venv-reference/bin/python tools/convert_sam3.py \
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

Use `tools/validate_image.py --allow-custom-quantization` for a separate
reference comparison of custom combinations; its report remains diagnostic.
See [model verification](validation.md) for the comparison workflow.

## Run inference

Use the `sam_image` executable from a CPU or Metal build. The output directory
must be new. For example:

```sh
build/cpu/examples/sam_image \
  --model models/sam3-image-vision-q8_0.gguf \
  --image image.jpg --text truck --score-threshold 0.2 \
  --output outputs/truck-q8 --backend cpu
```

Change the executable to the Metal build and pass `--backend metal` to request
Metal explicitly. Quantized `auto` selection currently chooses CPU.

## Backend behavior

GGUF precision describes stored weights, not end-to-end arithmetic. On CPU,
packed quantized weights stay resident, while the shared graph creates
transient F32 casts for `MUL_MAT`; this extra workspace can affect peak memory.
`ModelInfo` reports this as `ggml-quantized-weights-f32-v1`. Metal uses native
quantized kernels with half staging and reports `ggml-quantized-native-v1`, so
its arithmetic path and memory use differ from CPU. Validate the backend you
intend to deploy.
See the [GGUF schema reference](gguf.md), [model support](../MODEL_ZOO.md) and
[measured image performance](../BENCHMARK.md) for the file contract, current
support boundary and hardware-specific measurements.
