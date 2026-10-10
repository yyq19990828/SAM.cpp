# SAM 3 model download and conversion

This guide downloads the pinned official checkpoint and tokenizer and converts
them to runtime GGUF files. It does not download weights during inference. See
the [Model Zoo](../../MODEL_ZOO.md) for task availability, the
[GGUF contract](../gguf.md) for file schemas, and the
[quantization guide](../quantization.md) for quantized profile details.
Run the shell commands from the repository root and keep them in one shell so
the relative paths and revision variables remain available.

## Download

SAM 3 weights are gated on Hugging Face. First accept Meta's model terms on the
model page; accepting terms and authenticating the CLI are separate steps. Sign
in when needed with `hf auth login`, then verify the active account with
`hf auth whoami`.

```sh
hf auth whoami
# If this reports no active account:
hf auth login
```

The commands below pin the checkpoint revision and the matching Meta source
revision used for the BPE tokenizer. Keep downloaded files in ignored `models/`
directories.

```sh
sam3_hf_revision=3c879f39826c281e95690f02c7821c4de09afae7
sam3_source_revision=2345a4ad109ac29c569da749c91d84f10dc08c40
sam3_weights_dir="models/official/$sam3_hf_revision"
mkdir -p "$sam3_weights_dir"

hf download facebook/sam3 sam3.pt --revision "$sam3_hf_revision" \
  --local-dir "$sam3_weights_dir"
curl -fL "https://raw.githubusercontent.com/facebookresearch/sam3/$sam3_source_revision/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  -o "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz"
```

Verify the tokenizer asset before converting. The converter also validates it
against the pinned tokenizer identity.

```sh
SAM3_BPE="$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" python3.12 - <<'PY'
import hashlib
import os
from pathlib import Path

path = Path(os.environ["SAM3_BPE"])
actual = hashlib.sha256(path.read_bytes()).hexdigest()
expected = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
if actual != expected:
    raise SystemExit(f"BPE checksum mismatch: {actual}")
print(f"BPE checksum OK: {actual}")
PY
```

## Install conversion dependencies

Use Python 3.12 and the pinned [reference dependency lock](../../tools/requirements.lock).
Conversion runs in an isolated environment; the C++ runtime does not require
Python.

For Linux x86_64 CUDA reference runs, install
[requirements-linux-cuda.lock](../../tools/requirements-linux-cuda.lock) in place
of `tools/requirements.lock` to pin the CUDA package dependencies as well.

```sh
python3.12 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r tools/requirements.lock
```

Continue in the same shell so `sam3_weights_dir` remains available.

## Convert the supported runtime files

Image conversion requires an explicit precision. F32 stores all image weights
in FP32; F16 is a mixed-F16 format with selected weights retained in FP32.

```sh
.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task image --precision f32 \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-f32.gguf

.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task image --precision f16 \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-f16.gguf
```

Video conversion uses schema 2 and includes the tracker tensors. The default
video precision is `hybrid`; spell out the precision to make each artifact's
storage contract explicit. Use F32 or hybrid for application integration;
F16 video is available for diagnostics only.

```sh
.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task video --precision f32 \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-video-f32.gguf

.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task video --precision f16 \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-video-f16.gguf

.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task video --precision hybrid \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-video-hybrid-v1.gguf
```

Hybrid restores selected shared visual and tracker weights from original FP32
checkpoint values while keeping the remaining video weights in the mixed-F16
policy. It is not a full-F32 file. Video preprocessing and BF16 state/feature
rounding are separate model boundaries and apply independently of weight
storage; see the [precision contract](../../MODEL_ZOO.md#precision-contract).

## Convert quantized image profiles

Schema 3 quantization is image-only. The dependency lock pins `gguf==0.19.0`
for Q8_0; the K profiles use the project-built helper linked to pinned GGML.
Select the exact versioned
`image-vision-linear-*` profile explicitly; omitting `--storage-profile` chooses
the legacy broad allocation. Q8_0 uses the pinned Python GGUF quantizer. Q6_K,
Q5_K and Q4_K also need the project-built `sam_quantize_rows` helper for their
K-block tensors.

Build that helper once with the pinned GGML dependency:

```sh
cmake -S . -B build/quant-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF \
  -DSAM_BUILD_EXAMPLES=ON \
  -DSAM_BUILD_TESTS=OFF
cmake --build build/quant-cpu --target sam_quantize_rows
sam3_quantizer="$PWD/build/quant-cpu/examples/sam_quantize_rows"
```

Convert Q8_0 without the helper:

```sh
.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task image --precision q8_0 \
  --storage-profile image-vision-linear-q8_0-v1 \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-vision-q8_0.gguf
```

Convert the K profiles with the absolute helper path:

```sh
for precision in q6_k q5_k q4_k; do
  .venv-reference/bin/python tools/convert/convert_sam3.py \
    --task image --precision "$precision" \
    --storage-profile "image-vision-linear-${precision}-v1" \
    --quantizer "$sam3_quantizer" \
    --checkpoint "$sam3_weights_dir/sam3.pt" \
    --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
    --output "models/sam3-vision-${precision}.gguf"
done
```

The vision-only profiles quantize ViT linears and keep the text encoder and all
other unselected tensors in F32. Quantized profiles do not apply to video.
To select the full-component preset, use `image-full-linear-${precision}-v1`
instead. `--quantize-modules text,decoder` selects a custom partial configuration
and cannot be combined with `--storage-profile`. These schema-4 choices record
the selected components in the file; see the [quantization guide](../quantization.md)
for floating-point exceptions and the distinction between presets and custom
diagnostic combinations.

Different formats per module use the [mixed configuration](../quantization-config.md#module-mixed-weights)
with `--quantization-config`, producing SAM GGUF schema 5. This configuration
selects F32/Q8_0/Q6_K/Q5_K/Q4_K per module directly from original F32 weights.
Small CPU conversion/arithmetic fixtures validate the path; full-model
quality/performance and mixed CUDA/Metal execution remain unmeasured.

## Output files and precision labels

Each conversion writes a `.gguf` file and a matching `.gguf.manifest.json`
sidecar. Keep them together. The sidecar records the input checkpoint identity,
conversion profile and tensor inventory used by validation tools. Conversion
refuses to overwrite an existing output or sidecar; choose a new path for each
conversion.

`--precision` describes GGUF weight storage, not every arithmetic operation or
video state boundary:

| Task | Precision choices | Notes |
| --- | --- | --- |
| Image | `f32`, `f16`, `q8_0`, `q6_k`, `q5_k`, `q4_k`; `mixed` through JSON configuration | Quantized formats use schema-3/4 fixed allocations, or schema-5 per-module formats through configuration schema 2. |
| Video | `f32`, `f16`, `hybrid` | Quantized video files are not defined. |

The GGUF schema and model adapter enforce the declared tensor inventory,
dimensions, tokenizer identity and storage profile before allocating backend
weights. The container format alone does not add a model family or backend.
