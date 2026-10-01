# SAM GGUF format

Schema 1 uses a single little-endian [GGUF v3](https://github.com/ggml-org/ggml/blob/353b63b439f27ab2cc19dac97ab1681ba6d2d084/docs/gguf.md)
file with 32-byte alignment. GGUF identifies the container; the fields below
identify the SAM implementation contract. This schema covers SAM 3 text image
segmentation. Experimental schema 2 adds the full SAM 3 tracker inventory and
the video metadata below. Full video session integration and original-model
acceptance remain pending. Other model families require their own adapter/schema.

## Identity and provenance

All keys below are required except `general.name` and the explicit alignment key,
whose GGUF default is 32. Extra descriptive metadata is allowed within the
reader limits, but does not change model behavior.

| Key | GGUF type | Value |
| --- | --- | --- |
| `general.architecture` | STRING | `sam3` |
| `general.name` | STRING | Optional human-readable name |
| `general.file_type` | UINT32 | `0` for F32 or `1` for mixed F16 |
| `general.alignment` | UINT32 | `32` when present |
| `sam.schema_version` | UINT32 | `1` |
| `sam.task` | STRING | `text_image` |
| `sam.source.checkpoint_sha256` | STRING | 64 lowercase hexadecimal characters identifying the actual input checkpoint |
| `sam.source.code_revision` | STRING | `2345a4ad109ac29c569da749c91d84f10dc08c40` |
| `sam.tokenizer.sha256` | STRING | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |

The checkpoint hash records provenance; it is not an embedded signature or a
checksum of the GGUF payload. Conversion sidecars and numerical acceptance
also record and verify the complete output-file SHA-256.

## SAM 3 image parameters

Each scalar is UINT32 and must match the supported graph. The global attention
list is ARRAY of UINT32, in the recorded order. These keys describe the active
image graph; unused video/tracker fields from the former binary header are not
carried over.

| Key | Value |
| --- | --- |
| `sam3.vision.image_size` | 1008 |
| `sam3.vision.patch_size` | 14 |
| `sam3.vision.embedding_length` | 1024 |
| `sam3.vision.block_count` | 32 |
| `sam3.vision.attention.head_count` | 16 |
| `sam3.vision.feed_forward_length` | 4736 |
| `sam3.vision.window_size` | 24 |
| `sam3.vision.global_attention_blocks` | [7, 15, 23, 31] |
| `sam3.text.embedding_length` | 1024 |
| `sam3.text.attention.head_count` | 16 |
| `sam3.text.block_count` | 24 |
| `sam3.text.context_length` | 32 |
| `sam3.text.vocab_size` | 49408 |
| `sam3.text.output_length` | 256 |
| `sam3.neck.embedding_length` | 256 |
| `sam3.fusion.block_count` | 6 |
| `sam3.fusion.attention.head_count` | 8 |
| `sam3.fusion.feed_forward_length` | 2048 |
| `sam3.decoder.block_count` | 6 |
| `sam3.decoder.attention.head_count` | 8 |
| `sam3.decoder.feed_forward_length` | 2048 |
| `sam3.decoder.query_count` | 200 |
| `sam3.geometry.block_count` | 3 |

## Tokenizer

| Key | GGUF type | Value |
| --- | --- | --- |
| `tokenizer.ggml.model` | STRING | `clip` |
| `tokenizer.ggml.tokens` | ARRAY of STRING | 49,408 tokens, indexed by token ID |
| `tokenizer.ggml.merges` | ARRAY of STRING | 48,894 `left right` pairs, indexed by merge rank |
| `tokenizer.ggml.bos_token_id` | UINT32 | 49406 |
| `tokenizer.ggml.eos_token_id` | UINT32 | 49407 |
| `tokenizer.ggml.padding_token_id` | UINT32 | 0 |

Tokens use the complete pinned byte-to-Unicode vocabulary and canonical
`<start_of_text>` / `<end_of_text>` spellings. A merge entry has exactly one
ASCII space separating its two nonempty vocabulary symbols; their concatenation
must have token ID `512 + rank`. Require unique tokens/merges, exact byte/special
tokens and complete merge ranks. Legacy aliases/missing-merge repair is not part
of this schema. Runtime text remains printable ASCII/ASCII whitespace with the
existing 32-token behavior; the token-ID API remains available.

## Tensor data and loading

The 1,133 names and normalized GGML-order dimensions are defined by
[`sam3_tensor_schema.json`](../tools/sam3_tensor_schema.json) and validated against
the registered image graph. Missing, unknown or incompatible tensors fail before
backend weight allocation. GGUF directory offsets address one contiguous,
32-byte-padded tensor blob in directory order. Dimension counts are canonical:
omit trailing singleton GGML dimensions, retaining at least one dimension.
Metadata and tensor padding use zero bytes. Each payload and final padding
must fit within the file; incomplete data is rejected.

F32 files use F32 for every tensor. Mixed F16 files use F16 for tensors with two
or more dimensions, except names containing `embed`, `tpos`, `pe_gaussian`,
`token`, `no_obj`, `no_mem`, `gamma` or `freqs_cis`; those and all one-dimensional
tensors remain F32. Choose dtype before removing singleton dimensions, preserving
the source conversion policy. Reject source shapes whose resulting dtype conflicts
with the canonical SAM tensor's storage policy before publishing the file.
Preserve the existing name mapping, positional-embedding
layout and complex-RoPE real-pair layout. No quantized tensor types are accepted.

The common reader limits metadata reads to 16 MiB, at most 256 metadata keys and
4,096 tensor entries. Individual tokenizer symbols are limited to 2,048 bytes;
the complete tokenizer is limited to 16 MiB. Metadata is parsed without loading
the tensor blob, then SAM-specific types/values, dimensions, dtype policy and
file ranges are checked. Typed GGUF accessors are called only after type checks.
The metadata prefix must match canonical reserialization through upstream GGUF
APIs. This rejects embedded-NUL truncation in C-string accessors, noncanonical
dimension counts and ambiguous encodings without a second binary parser.

CPU exact F16 promotion and Metal precision/storage policies remain unchanged.
Using a GGUF container does not enable a new backend, quantization or mmap
loading. Original `.pt` inputs remain the supported reconversion source; old
SAM-specific `.ggml` containers are rejected with a reconversion diagnostic.

## Experimental full video profile (schema 2)

The converter selects this profile with `--task video`; the default remains
`--task image`. Schema 2 requires `sam.schema_version=2` and
`sam.task=text_video`. It retains every identity, tokenizer and image parameter
above, and includes the 1,133 image tensors plus 331 tracker/neck tensors:
exactly 1,464 canonical entries from `tools/sam3_tensor_schema.json`. The
previously excluded pooled-text/training entry remains excluded and recorded.
Image-subset tensor bytes and precision policy remain identical. Old schema-1
runtimes reject schema 2; the new loader accepts both profiles.

All additional scalar fields are UINT32:

| Key | Required value |
| --- | ---: |
| `sam3.tracker.embedding_length` | 256 |
| `sam3.tracker.memory_length` | 64 |
| `sam3.tracker.attention.block_count` | 4 |
| `sam3.tracker.attention.head_count` | 1 |
| `sam3.tracker.attention.head_length` | 256 |
| `sam3.tracker.memory_position_count` | 7 |
| `sam3.tracker.conditioning_frame_count` | 4 |
| `sam3.tracker.pointer_candidate_count` | 16 |
| `sam3.tracker.mask_memory_size` | 1152 |

Required STRING fields are `sam3.tracker.policy=meta-sam3-temporal-v1`,
`sam3.tracker.feature_storage=bf16`, `sam3.tracker.memory_storage=bf16`, and
`sam3.video.preprocessing=pillow-bicubic-f16-normalize-v1`. These storage fields
specify explicit transport/state rounding, independently of checkpoint storage
and FP32 arithmetic. Unsupported values/types, incomplete inventories, shape or
dtype mismatches fail before backend weight allocation.

Use new output filenames and sidecars, such as `sam3-video-f32.gguf`; conversion
refuses existing output files. `ModelInfo::task`/`profile` describe the file
contract. The image API may load the image subset of these full files. The video
session and oracle exporter remain unimplemented, so schema-2 files are not
accepted video-inference artifacts yet. Local original-weight conversion,
image-subset regression and tracker-stage comparisons remain required.
