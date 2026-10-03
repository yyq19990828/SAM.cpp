# Model Zoo

[中文](MODEL_ZOO_zh.md) · [Benchmarks](BENCHMARK.md)

## Support

SAM 3 text image segmentation: F32/F16 on CPU and Metal, seven reference
cases each. Forward video: F32/hybrid on CPU and Metal, five cases/216 frames
each, plus image/lifetime/long-session checks. Video conversion defaults to
hybrid `visual-tracker-f32-v1`; image conversion requires explicit precision.

F16 video is diagnostic: complete Metal comparison fails exact candidate
selection at entry23/24; identical rounded weights reproduce it inside Meta.
The full CPU diagnostic is deferred, not passed. CUDA, Q4/Q8, reverse/interactive
video and SAM3.1/other model families are not validated integrations here.

## Sources and artifacts

Original weights: [facebook/sam3](https://huggingface.co/facebook/sam3/tree/3c879f39826c281e95690f02c7821c4de09afae7),
revision `3c879f39826c281e95690f02c7821c4de09afae7`.
Meta code: [2345a4ad](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40).
`sam3.pt` SHA-256: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
GGUF v3 uses schema1 for image, schema2 for full video. Sidecars bind metadata,
source and tensor hashes. Model access and license acceptance are separate;
weights/media stay outside Git. See [licenses](THIRD_PARTY_NOTICES.md).

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `sam3-f32.gguf` | 3371139456 | `cb13ecd5012a2fe19b06d840049be6daa6b177b35256af8c4afeb12125352486` |
| `sam3-f16.gguf` | 1797613888 | `66731fa5def347677f78d7422b81979be0f8e2f7ead941db9a466d1cfa715120` |
| `sam3-video-f32.gguf` | 3449345696 | `02513232afca5ba8c174b66c7fc839c67b32590df4b53bdd6df5089a6a546844` |
| `sam3-video-f16.gguf` | 1837925216 | `9c9bc86c81d11a041db10a46d3d1e8ecaa1cbcf6fad683b00901f641746bf32c` |
| `sam3-video-hybrid-v1.gguf` | 2765012640 | `3975b4b1a10b962c6fad5022e2798baa6266aad7dbcc39210b459537208e1a05` |

## Precision contract

| Weight label | Disk / Metal resident weights | CPU resident weights |
| --- | --- | --- |
| F32 | Original F32 | F32 |
| F16 | Mixed F16/F32 | Stored F16 promoted to F32 |
| Hybrid | Original F32 visual/tracker; mixed detector/text | Remaining F16 promoted to F32 |

Promotion preserves rounded values; it cannot restore original F32 values.
Graphs use F32 activations and the [specified F32 arithmetic](cmake/patches/README.md);
attention masks can be F16. No GGUF BF16 weights or native all-BF16 inference
are claimed. When executing VideoSession, **all three profiles** retain:

| Video boundary | Precision / representation |
| --- | --- |
| Normalization | F16 rounding at each step; F32 output buffer |
| Tracker-neck features | BF16 rounding in F32 vectors |
| Mask-memory records | BF16 storage, expanded to F32 before attention |
| Object pointers / host mask logits | F32; final binary masks are uint8 |

Thus F32 video describes weights, not an end-to-end F32 pipeline. Image
inference has no tracker BF16 state. Hybrid restores 236 payloads and retains
1228 baseline payloads, adding about 884 MiB over the F16 video artifact.

## Use

Follow [download/conversion](docs/models/sam3-details.md#download) with Python3.12
and the locked reference environment. Runtime inference needs no Python.

```sh
.venv-reference/bin/python tools/convert_sam3.py --task video \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-video-hybrid-v1.gguf
```

Use new output paths; the converter refuses existing files. Explicit F32/F16
remain available. [README](README.md#forward-video-tracking) shows public APIs;
[validation](docs/validation.md) covers baseline reuse and diagnostics. The
[detail record](docs/models/sam3-details.md) retains mapping, tokenizer, container
migration and original conversion evidence.

The current ViT projection/optional CPU BLAS implementation passes the same full
video matrix, 70 image cases, six short sessions and two 128-push interleaved checks.
Fresh four-cell video and four-cell image measurements also pass; see
[benchmarks](BENCHMARK.md) and the [optimization record](docs/plans/20261003-134534-visual-encoding-profile-and-optimization.md).
