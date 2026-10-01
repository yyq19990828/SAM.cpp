# Model Zoo

The runtime reads **GGUF v3 with SAM schema 1**, containing named model
parameters, the complete tokenizer and source hashes. See
[GGUF schema and migration](#gguf-schema-and-migration). Performance is recorded in
[BENCHMARK.md](BENCHMARK.md).

## Supported model and sources

| Model | Task supported here | Official weights | Official implementation | Accepted configurations |
| --- | --- | --- | --- | --- |
| SAM 3 | Text-prompted image segmentation | [facebook/sam3 on HF](https://huggingface.co/facebook/sam3) | [facebookresearch/sam3 on GitHub](https://github.com/facebookresearch/sam3) | FP32/CPU, FP16/CPU, FP16/Metal; 7/7 reference cases each |

SAM 3 video, SAM 3.1, SAM 2/2.1, GroundingSAM and DART are future integrations;
the current converter does not produce runnable files for them. Their extension
boundaries are described in [architecture.md](docs/architecture.md).

The C++ port and tensor mapping derive from
[PABannier/sam3.cpp](https://github.com/PABannier/sam3.cpp/tree/416186c501d060df7ca02989d49b38080f5f81f3).
Earlier supplementary checks used
[PABannier/sam3.cpp on HF](https://huggingface.co/PABannier/sam3.cpp), revision
`a3892b63b918e872671322e116982a8910f0ffb7`. Current acceptance uses the original
Meta checkpoint independently; a community container is not evidence of original
checkpoint provenance. Model and C++ dependency licenses are recorded separately
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Pinned inputs

| Input | Revision / identity |
| --- | --- |
| HF weights | [facebook/sam3](https://huggingface.co/facebook/sam3/tree/3c879f39826c281e95690f02c7821c4de09afae7), revision `3c879f39826c281e95690f02c7821c4de09afae7` |
| Original checkpoint | `sam3.pt`, 3,450,062,241 bytes; SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| Meta source and reference graph | [GitHub revision](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40) `2345a4ad109ac29c569da749c91d84f10dc08c40` |
| Tokenizer asset | [bpe_simple_vocab_16e6.txt.gz](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/assets/bpe_simple_vocab_16e6.txt.gz); SHA-256 `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |

## Download

Run from the repository root with an installed HF CLI. The HF account needs
access to the gated SAM 3 model; authentication and license acceptance are
separate requirements. `hf auth login` can establish credentials when needed.

```sh
hf auth whoami
sam3_hf_revision=3c879f39826c281e95690f02c7821c4de09afae7
sam3_source_revision=2345a4ad109ac29c569da749c91d84f10dc08c40
sam3_weights_dir="models/official/$sam3_hf_revision"
mkdir -p "$sam3_weights_dir"

hf download facebook/sam3 sam3.pt --revision "$sam3_hf_revision" \
  --local-dir "$sam3_weights_dir"
curl -fL "https://raw.githubusercontent.com/facebookresearch/sam3/$sam3_source_revision/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  -o "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz"
shasum -a 256 "$sam3_weights_dir/sam3.pt" \
  "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz"
```

Compare the two hashes with the pinned inputs above. Keep the files in ignored
`models/`; inference itself performs no download.

## Convert the supported runtime files

Use Python 3.12 and the [reference dependency lock](tools/requirements.lock),
validated on macOS arm64. Continue in the same shell so `sam3_weights_dir` is
available. Other platforms need their Python dependency environment validated.

```sh
python3.12 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r tools/requirements.lock

.venv-reference/bin/python tools/convert_sam3.py \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --precision f32 --output models/sam3-f32.gguf

.venv-reference/bin/python tools/convert_sam3.py \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --precision f16 --output models/sam3-f16.gguf
```

The converter uses `gguf==0.19.0` to stream 1,133 image-model tensors and the
complete tokenizer into a GGUF file. Its `.gguf.manifest.json` sidecar records
source/output and per-tensor hashes, canonical dimensions, payload offsets,
package version and converter/helper identities.
Known unused tracker tensors are validated and excluded. Existing output files
or sidecars are refused; reuse a verified file or choose a new output path.

| Output | Size (bytes) | SHA-256 | Runtime |
| --- | ---: | --- | --- |
| `sam3-f32.gguf` | 3,371,139,456 | `cb13ecd5012a2fe19b06d840049be6daa6b177b35256af8c4afeb12125352486` | CPU |
| `sam3-f16.gguf` | 1,797,613,888 | `66731fa5def347677f78d7422b81979be0f8e2f7ead941db9a466d1cfa715120` | CPU or Metal |

FP16 uses the converter's mixed storage policy: selected tensors stay FP32.
CPU promotes half weights exactly to FP32 in memory; Metal keeps packed half
weights with the required arithmetic corrections. Only F32 and mixed F16 storage
are supported; quantized Q4/Q8 files are rejected. The converter requires a
`.gguf` output suffix and writes genuine GGUF bytes.

Follow [reference export and validation](README.md#convert-and-generate-references)
to compare new conversions with the pinned Meta graph. The complete original
checkpoint acceptance record is in the
[GGUF migration plan](docs/plans/20261001-020507-gguf-conversion-loading.md).

## GGUF schema and migration

GGUF provides standard typed metadata and a tensor directory. This project's
[SAM schema 1](docs/gguf.md) defines `general.architecture=sam3`,
`sam.task=text_image`, exact image parameters and CLIP tokenizer metadata.
Only supported model/task combinations are loaded; a GGUF extension alone does
not add SAM 2, SAM 3.1, CUDA or quantization support.

The C++ reader uses the pinned GGML GGUF APIs with bounded metadata access,
type checks, canonical metadata validation and complete tensor-range checks
before backend weight allocation. The public `sam::Model::load` API is unchanged.
Python is required for conversion and validation, not runtime inference.

Old custom `.ggml` files are no longer accepted. Keep the authorized original
`sam3.pt` and rerun the conversion commands above with new `.gguf` output paths.
The converter preserves existing files and produces new sidecars; renaming an
old file is insufficient. The old tokenizer alias/missing-merge repair path is
removed, and the retained compatibility-report field is false for GGUF models.

This migration preserves tensor values and the existing compute policies.
Changing the container does not itself quantize weights or guarantee faster
inference. Earlier custom-container provenance and timing remain available in
the [previous acceptance record](docs/plans/20261001-002850-metal-window-cpu-performance-official-weights.md).
