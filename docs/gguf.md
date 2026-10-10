# SAM GGUF format

SAM 3 model files use little-endian [GGUF v3](https://github.com/ggml-org/ggml/blob/d7cb574130e6f01ad25b3289685489200febcd74/docs/gguf.md)
with 32-byte alignment. GGUF defines the container; the fields below define
this project's SAM 3 model contract. Schema 1 covers image F32 and mixed-F16
weights, schema 2 covers video weights and explicit state/transport metadata,
schema 3 identifies the legacy versioned quantized image profiles, schema
4 adds explicit component selection, and schema 5 assigns weight formats per module. Container support
does not itself imply that a model, task or backend is supported. See the
[Model Zoo](../MODEL_ZOO.md) and [quantization guide](quantization.md) for
current usage scope. Other model families need their own metadata and tensor
contracts.

## Identity and provenance

All keys below are required except `general.name` and the explicit alignment key,
whose GGUF default is 32. Extra descriptive metadata is allowed within the
reader limits, but does not change model behavior.

| Key | GGUF type | Value |
| --- | --- | --- |
| `general.architecture` | STRING | `sam3` |
| `general.name` | STRING | Optional human-readable name |
| `general.file_type` | UINT32 | Schema 1/2: `0` for F32 or `1` for mixed F16. Schema 3/4: profile-specific values below. Schema 5: declared base Q format. |
| `general.alignment` | UINT32 | `32` when present |
| `sam.schema_version` | UINT32 | `1` for image F32/F16, `2` for video, `3` for legacy quantized image, `4` for modular quantized image, `5` for module mixed weights |
| `sam.task` | STRING | `text_image` for schema 1/3/4/5; `text_video` for schema 2 |
| `sam.source.checkpoint_sha256` | STRING | 64 lowercase hexadecimal characters identifying the actual input checkpoint |
| `sam.source.code_revision` | STRING | `2345a4ad109ac29c569da749c91d84f10dc08c40` |
| `sam.tokenizer.sha256` | STRING | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |

The checkpoint hash records source provenance; it is not a signature or a
checksum of the GGUF payload.

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
[`sam3_tensor_schema.json`](../tools/convert/sam3_tensor_schema.json) and validated against
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
Preserve the existing name mapping, positional-embedding layout and
complex-RoPE real-pair layout. Schemas 1 and 2 accept no quantized tensor types.

Schema 2 also supports an explicit hybrid profile, selected by
`--task video --precision hybrid`. It uses `general.file_type=1` and requires
the STRING metadata `sam.storage_profile=visual-tracker-f32-v1`. Restore original
checkpoint FP32 values for `vit.`, `neck.trk.`, `mem_attn.`, `mem_enc.`, `sam_pe.`,
`sam_dec.`, `obj_ptr_proj.`, `obj_ptr_tpos_proj.` and `trk_mask_ds.` tensors;
remaining tensors follow the mixed-F16 policy. Promoting rounded F16 values does
not satisfy this profile. Unknown profiles, hybrid declarations on schema
1/F32 containers and tensor types inconsistent with the declared closure are
rejected before weight allocation. `ModelInfo::precision` reports `hybrid`, and
`storage_profile` reports the versioned policy separately from the temporal
`profile`. This is not a full-FP32 artifact.

The common reader limits metadata reads to 16 MiB, at most 256 metadata keys and
4,096 tensor entries. Individual tokenizer symbols are limited to 2,048 bytes;
the complete tokenizer is limited to 16 MiB. Metadata is parsed without loading
the tensor blob, then SAM-specific types/values, dimensions, dtype policy and
file ranges are checked. Typed GGUF accessors are called only after type checks.
The metadata prefix must match canonical reserialization through upstream GGUF
APIs. This rejects embedded-NUL truncation in C-string accessors, noncanonical
dimension counts and ambiguous encodings without a second binary parser.

For schemas 1 and 2, CPU F16 promotion and Metal/CUDA precision/storage policies are
defined by the runtime. Using a GGUF container by itself does not enable a new
backend or mmap loading. Schemas 3 and 4 share the quantized arithmetic contract
described below. Original `.pt` inputs remain the supported reconversion source;
old SAM-specific `.ggml` containers are rejected with a reconversion diagnostic.

## Quantized image profiles (schema 3)

Schema 3 requires `sam.schema_version=3` and `sam.task=text_image`. It accepts
only the exact storage profiles in the table. `general.file_type` and
`general.quantization_version` are UINT32. The per-tensor GGML type policy is
part of each profile; `general.file_type` alone does not select a tensor
allocation policy.

| Precision | `sam.storage_profile` | `general.file_type` | Main tensor type | K-profile row fallback |
| --- | --- | ---: | --- | --- |
| Q8_0 | `image-linear-q8_0-v1` or `image-vision-linear-q8_0-v1` | 7 | Q8_0 | None |
| Q6_K | `image-linear-q6_k-v1` or `image-vision-linear-q6_k-v1` | 18 | Q6_K | Q8_0 on the 32 ViT `mlp.lin2.weight` matrices with `ne[0]=4736` |
| Q5_K | `image-linear-q5_k-v1` or `image-vision-linear-q5_k-v1` | 16 | Q5_K | Q8_0 on the 32 ViT `mlp.lin2.weight` matrices with `ne[0]=4736` |
| Q4_K | `image-linear-q4_k-v1` or `image-vision-linear-q4_k-v1` | 14 | Q4_K | Q8_0 on the 32 ViT `mlp.lin2.weight` matrices with `ne[0]=4736` |

All profiles require `general.quantization_version=2`, the complete 1,133-tensor
image inventory, and exact per-tensor types and dimensions. The `image-linear-*`
family quantizes 224 ViT and text linears. The `image-vision-linear-*` family
quantizes only 128 ViT linears and preserves text and every other tensor in F32.
These profile names identify distinct versioned allocation policies; see the
[quantization guide](quantization.md) for profile selection and current usage
scope.

The loader validates `ne[0]` against the block size before creating packed
tensors, validates scales and decoded values before upload, and rejects unknown
profiles, file types, versions, tasks and tensor allocations. Schema 3 is not a
video format. `ModelInfo::precision` names the quantized storage type and
`storage_profile` identifies the allocation policy. CPU reports
`arithmetic_profile=ggml-quantized-weights-f32-v1`: packed weights remain
resident while shared graph nodes cast operands to F32 for `MUL_MAT`. Metal
reports `ggml-quantized-native-v1` and uses native quantized kernels with half
staging. CUDA reports `ggml-quantized-cuda-native-v1` and retains the same GGUF
packing for native quantized kernels; the validated MMVQ/MMQ paths use RHS Q8_1
staging. These paths differ from the strict F32 profile. Quantized `Auto`
selects CPU. Explicit Metal and CUDA retain the CPU scheduler tail required by
GGML, check every operation on the selected device and reject compute fallback.

## Modular quantized image profiles (schema 4)

Schema 4 remains image-only and retains the same 1,133-entry tensor inventory,
GGUF v3 container, file-type values and `general.quantization_version=2` as the
quantized formats above. It requires STRING `sam.storage_profile` and STRING
`sam.quantization.modules`. The latter is a nonempty canonical CSV, ordered
`vision,text,fusion,decoder`, without duplicates or extra whitespace.

`image-full-linear-{precision}-v1` requires all four components.
`image-modules-linear-{precision}-v1` accepts any nonempty component selection,
including all four, and remains a custom diagnostic allocation. The module
CSV does not convert a custom profile into the full preset. `{precision}` is
one of `q8_0`, `q6_k`, `q5_k`, `q4_k`; names, file types and tensor types must
agree. Schema 3 cannot declare these profiles or component metadata.

| Module | Tensor scope | Eligible linear matrices |
| --- | --- | ---: |
| `vision` | ViT linears; neck convolutions remain F32 | 128 |
| `text` | Text Transformer linears and resizer | 97 |
| `fusion` | `fenc` linears | 36 |
| `decoder` | `ddec`, `geom`, `scoring`, `seg` linears | 87 |

The fixed policy covers 348 canonical matrices, including fused `_in_proj_weight`
projections. Unselected components, embeddings, positional tensors, normalization,
bias, convolutions and canonical one-dimensional parameters remain F32. This
includes `ddec.presence_token_head.layers.2.weight`, whose original `[1,256]`
shape is a canonical `[256]` vector. The geometry direct/box-position projections
with row widths 2, 4 and 258, and both box-RPB input projections with row width
2, also remain F32. When `vision` is selected, K profiles retain the sole
32-matrix ViT `lin2` Q8_0 fallback at row width 4736. Selections without
`vision` contain none of those fallback matrices. Other eligible canonical
rows are 256-aligned.
The converter, tensor inventory and loader enforce this closure; the actual
dtype cannot declare or override the selected module policy.

Sidecars record canonical `quantization_modules` arrays and each tensor's
module, storage type and quantization/retention reason. Schema-4 runtime
`ModelInfo::quantization_modules` and image JSON report the validated array;
legacy schemas leave the new runtime field empty. Activation precision and
CPU/Metal/CUDA arithmetic policies remain those described above. Loading this
schema does not implement point/box prompting or quantized video tracking.

## Module mixed image weights (schema 5)

Schema 5 retains the same 1,133 canonical tensors, GGUF v3 container and
`general.quantization_version=2`. It requires these additional STRING fields:

| Key | Value |
| --- | --- |
| `sam.storage_profile` | `image-mixed-linear-v1` |
| `sam.quantization.base_precision` | `q8_0`, `q6_k`, `q5_k` or `q4_k` |
| `sam.quantization.module_precisions` | All four `module=format` entries, canonical order, no whitespace |
| `sam.quantization.policy_sha256` | Lowercase SHA-256 of the canonical encoding below |

Each module accepts `f32`, `q8_0`, `q6_k`, `q5_k`, `q4_k`.
`sam.quantization.modules` is not allowed in schema 5; its complete format
mapping replaces that selected-module wire field. JSON configuration schema 2
resolves omitted module choices from the base before writing GGUF. The GGUF CSV
must be complete, for example
`vision=q4_k,text=q8_0,fusion=q6_k,decoder=f32`.

Hash this exact ASCII encoding, including each LF and the final LF:

```text
sam3:image-mixed-linear-v1
base=q4_k
vision=q4_k
text=q8_0
fusion=q6_k
decoder=f32
```

`general.file_type` matches the declared base (7/18/16/14 for Q8_0/Q6_K/Q5_K/Q4_K),
even if overrides replace every module's base choice. It does not summarize the
actual storage distribution. The module mapping, actual tensor types and
schema-4 eligibility/protection rules determine storage. The vision K-row 4736
exception remains Q8_0. F32 module choices, protected parameters and ineligible
rows remain F32. Per-module F16 and per-tensor overrides are not supported.
All-F32 allocations are valid; runtime quantized arithmetic flags follow actual
stored types rather than the base file-type value.

The native loader recomputes the policy hash, validates file type, checks every
canonical tensor name/shape/type/range before backend weight allocation, and
retains packed types. Sidecars add `base_precision`, `module_precisions`,
`policy_sha256` and per-tensor `requested_dtype`, `resolved_dtype`, `selector`,
`quantization_reason`. Their `quantization_modules` array lists only modules
whose selected format is not F32, including an empty array for all-F32 choices.
`ModelInfo` and native JSON expose those same policy fields with `precision=mixed`.
The policy hash binds the allocation; file/checkpoint hashes separately identify
the bytes and source. Earlier schemas reject reserved mixed-policy metadata;
older schema-1–4 loaders reject schema 5.

Small CPU conversion and mixed arithmetic fixtures validate the format path.
No complete-model quality/performance or mixed CUDA/Metal qualification follows
from these checks. See the [configuration guide](quantization-config.md) for usage.

## Full video profile (schema 2)

The converter selects this profile with `--task video`; the default remains
`--task image`. Schema 2 requires `sam.schema_version=2` and
`sam.task=text_video`. It retains every identity, tokenizer and image parameter
above, and includes the 1,133 image tensors plus 331 tracker/neck tensors:
exactly 1,464 canonical entries from `tools/convert/sam3_tensor_schema.json`. The
pooled-text/training tensor is excluded. A schema-1-only reader does not
implement the schema-2 task contract.

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

`ModelInfo::task`/`profile` describe the file contract. The image API may load
the image subset of a full schema-2 file. The matching video adapter interprets
the tracker tensors and state/transport metadata; schema 2 is not an image-
quantization format.

The FP16 policy explicitly retains
`sam_dec.pred_obj_score_head.layers.2.weight` in F32: its original `[1,256]`
projection has canonical GGML dimensions `[256]`, for which the runtime requires
F32. This does not change any of the 1,133 image-subset payloads or weaken shape
validation.
