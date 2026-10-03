# Migrate conversion and loading to GGUF

Created: 2026-10-01 02:05:07 Asia/Shanghai.
Status: complete; conversion, loading, integration, numerical and performance checks passed.

## Scope and decisions

Replace the experimental SAM 3 custom `.ggml` v3 converter/reader with real
little-endian GGUF v3 conversion and loading. Keep C++17 header-only integration,
the public model/session API, GGML revision/Metal patch, model graphs, tensor
values/layouts, FP32/FP16 storage policies and numerical gates unchanged.
This coordinated format change touches more than eight files across I/O,
SAM-specific validation, conversion/validation tools, tests, CMake and docs.

Use a single GGUF path. The project is unreleased; maintaining two binary
protocols would retain a model-specific reader and its compatibility repairs.
Legacy files remain on disk but new runtime loads reject their magic with an
instruction to reconvert the original checkpoint. Require `.gguf` converter
outputs and retain exclusive publication of model plus manifest. This is an
explicit file-format breaking change, not a public API change.

The minimal implementation reuses the pinned GGML `gguf.h` APIs and official
`gguf==0.19.0` Python writer/reader. Its wheel SHA-256 is
`70bcd10edfe697fb2dad6e40af2234b9d8ece9a41a99761405121ebda1c3c1cd`;
upstream tag `gguf-v0.19.0` resolves to
`a290ce626663dae1d54f70bce3ca6d8f67aab62f`. Install only into the existing isolated
reference environment and pin it in `tools/requirements.lock`; C++ consumers
gain no Python or new compiled dependency.

## Format and architecture

The exact shared contract is [GGUF schema 1](../gguf.md). Use standard GGUF
architecture/file-type/tokenizer metadata plus named SAM 3 image parameters,
task/schema identifiers and source hashes. The vision feed-forward width is
4736; the legacy header value 4625 encoded a ratio and must not become a width.
Write the complete tokenizer; remove runtime repair of incomplete legacy BPE.
Keep the public compatibility-report boolean as false for valid GGUF models.

```text
original sam3.pt + BPE -> Python converter -> GGUF + provenance manifest
                                             |
                      bounded common reader -> SAM 3 schema -> existing graph
                                                                  |
                                                            CPU / Metal
```

Keep byte/container reading in `internal/io/gguf_reader.hpp`; keep SAM shapes,
parameter/tokenizer checks and precision policy in `internal/models/sam3/`.
The common reader is a concrete reuse point for future model adapters, not a
new graph abstraction. Only architecture `sam3`, task `text_image`, schema 1,
F32 and mixed F16 are accepted. Other architectures/tasks/quantization fail
explicitly. No video, new model family, CUDA, quantization or mmap loading is
introduced by changing the container.

Parse metadata through `gguf_init_from_callback` with `{true, nullptr}` and a
16 MiB read budget (or the smaller real file size). Upstream checks lengths
before allocation and does not read tensor data in this mode. Separately check
all tensor extents/padding against the actual file length before backend weight
allocation. Bound header counts, validate metadata types before typed getters,
and reject unsupported versions, duplicates, malformed tokenizer metadata,
wrong shapes/types, truncated data, invalid alignment and overflowing offsets.
Unknown bounded descriptive keys may be ignored for forward-compatible metadata.
Canonical metadata reserialization through GGUF APIs rejects embedded-NUL and
rank ambiguities that C-string getters cannot expose. The converter drops
trailing singleton GGML dimensions only after choosing the unchanged storage
dtype, and emits zero padding; no tensor payload bytes change.

## Implementation and ownership

1. Coordinator preserves the previous source and receipts under ignored
   `build/gguf-migration/baseline/`, freezes the shared schema and owns docs,
   provenance reconciliation and full-model acceptance.
2. Runtime worker owns `internal/io/`, SAM 3 weight inspection/loading and
   `tests/test_contracts.cpp`. Replace legacy parsing, remove unused runtime
   tracker-tensor allowances, and retain tokenizer/output/input behavior checks.
   No graph/backend arithmetic edits.
3. Tooling worker owns `tools/convert_sam3.py`, the Python GGUF helper if needed,
   `tools/validate_image.py`, `tools/test_tools.py` and dependency pin. Register
   tensor metadata first, then stream converted arrays into a temporary GGUF;
   avoid holding another full converted model in memory. Read back offsets,
   types, dimensions and hashes with the independent reader before publishing.
   Preserve source tensor/unused-tracker validation and finite/overflow checks.
4. Integration worker updates the caller-owned GGML API probe and affected
   CMake/test/example paths, then performs fresh CPU/Metal and consumer builds
   after the interfaces freeze. No changes to shared dependency checkouts.
5. Coordinator converts the already verified original checkpoint into new
   FP32/FP16 GGUF files, preserving all original `.pt`/`.ggml` files and reference
   exports. Compare every tensor byte and tokenizer entry against the accepted
   conversions, then run the complete official seven-case matrix on FP32/CPU,
   FP16/CPU and FP16/Metal without supplementary flags or relaxed gates.
6. Run original-weight session/cache/lifetime behavior and isolated final
   benchmarks. Refresh canonical builds, update MODEL_ZOO/README/architecture,
   benchmark format/provenance and changelog. Preserve historical plans/results
   with their original container identities. No commit/push is requested.

## Verification and failure handling

- Fast checks use small GGUF fixtures for successful data/metadata inspection
  and meaningful rejection boundaries; keep checks active in Release and retain
  independent/repeated header and two-translation-unit linkage tests.
- Python checks decode real writer output and exercise mixed storage, conversion
  overflow and existing-output refusal; keep official/supplementary provenance
  distinct. Validation reads GGUF metadata and manifest consistently.
- Run `cmake --build`, CTest and downstream consumers on CPU/Metal, plus
  `build/reference-runtime/venv/bin/python tools/test_tools.py`.
- Verify all 1,133 original-weight tensor payloads per precision and exact BPE
  token/merge order before model execution. Apply the unchanged official tensor,
  score, box and mask gates for all 21 cases.
- Measure performance only with other project computation stopped; record file
  hashes, format, compiler/backend/hardware, raw timings and peak memory. Earlier
  custom-container timings remain historical until new GGUF runs finish.
- Review diff/owned-text whitespace and updated documentation links. No coverage
  quotas, broad rewrites or unrelated API changes.

The fragile assumption is byte/layout parity between both writers. Any payload
or tokenizer mismatch stops numerical acceptance for diagnosis. Recovery uses
the preserved source and external artifacts; converter failures remove only
their own temporary/newly published files. No weight re-download or new HF
credentials are required because the authorized original checkpoint is local.

## Results

- Plan and shared format contract were written before implementation.

- Preserved the previous source and binary identities under
  `build/gguf-migration/baseline/`; original checkpoints, custom containers and
  reference bundles remain intact.
- Official `gguf==0.19.0` is installed only in the reference environment;
  its MIT license is retained in `licenses/gguf-py-MIT.txt`.
- A small independent Python-writer/C++-reader probe passes exact F16/F32 data,
  shape, typed scalar/array metadata and UTF-8 checks. Artifacts are under
  `build/gguf-migration/python-cpp-smoke.*`.
- Final isolated tooling checks pass 7/7 (6.698 seconds), including fractional
  values that detect accidental narrowing of tensors required to stay FP32.
- Original FP32 and FP16 GGUF files have been generated. Their exact sizes and
  SHA-256 values are in [MODEL_ZOO.md](../../MODEL_ZOO.md).
- Both actual files match the preserved custom containers for every one of
  1,133 tensor payload hashes, lengths and dtypes, all 49,408 vocabulary entries
  and all 48,894 merge ranks. Four tensor shapes per file lose only redundant
  singleton dimensions; data is unchanged. Receipt:
  `build/gguf-migration/weights/parity.json`, SHA-256
  `9689b6cd6e260ce93cf8781ed84cc5c240d70d56d66a67314c8edae2e40d3070`.
- All 27 preserved public/graph/backend/corpus hashes and the acceptance-gates
  syntax tree match the baseline. Receipt:
  `build/gguf-migration/unchanged-graph-and-gates.json`.
- Strict C++17 `-Wall -Wextra -Werror` compile/link checks pass with Homebrew
  Clang 23.1.2. Small valid/malformed GGUF contracts pass, as do the real FP32
  and FP16 files' tokenizer golden and sparse corrupt-merge rejection checks.
  Receipt: `build/gguf-migration/runtime-contracts-receipt.json`.
- Fresh and refreshed canonical AppleClang 21 CPU and Metal Release builds each
  pass CTest 9/9; caller-owned-GGML downstream consumers each pass 3/3. The API
  probe also rejects a dependency with only its bounded GGUF callback declaration
  removed. Receipt: `build/gguf-migration/integration-validation.json`.
- Real FP16/Metal GGUF session checks pass model lifetime, prompt/image changes,
  shared-model session isolation and cache reuse. The compute-buffer high-water
  mark remains 1,245,268,288 bytes; these correctness runs overlapped CPU
  validation and are not used for performance claims. Receipt:
  `build/gguf-migration/session-metal/results.json`.

### Original-checkpoint numerical acceptance

All 21 cases pass the unchanged gates against the preserved original-checkpoint
unfused FP32 reference. Each configuration compares 70 tensor snapshots plus
exact token IDs, confidence filtering, scores, boxes and masks.

| GGUF / backend | Cases | Worst normalized L2 | Limit | Minimum high-confidence mask IoU |
| --- | ---: | ---: | ---: | ---: |
| FP32 / CPU | 7/7 | 0.0005700826798642337 | 0.001 | 1.0 |
| FP16 / CPU | 7/7 | 0.013056476876752196 | 0.02 | 1.0 |
| FP16 / Metal | 7/7 | 0.012984803954191696 | 0.02 | 1.0 |

Reference manifest SHA-256:
`64d471b644615f26350e37a235b06e330e3e16f1d3aa2cdc6cc553b6fe0f814b`.
Metrics are under
`build/gguf-migration/validation-{cpu-f32,cpu-f16,metal-f16}/metrics.json`;
`build/gguf-migration/acceptance-summary.json` consolidates identities and results.
The full image graph uses 3,332 CPU nodes or 3,342 Metal nodes with no Metal-to-CPU
fallback, and six partitions on either backend. These correctness runs may
overlap; their timings are excluded from the benchmark.

All 210 tensor binary files and all 18 output mask binary files are also
byte-identical to the prior accepted custom-container runs. Token/output metadata
and runtime counters match exactly after excluding process peak RSS. Receipt:
`build/gguf-migration/execution-container-parity.json`. This separately confirms
that the container migration preserves the existing execution results.

### GGUF benchmark

CPU and Metal were measured sequentially after all builds, model validation and
payload/output comparisons finished. The input was `truck.jpg` (1800 x 1200),
prompt `truck`, threshold 0.5, four CPU threads, original-checkpoint FP16 GGUF.
After one initial pipeline call, each backend ran five complete-image warm calls;
cache-only timings were collected separately. Outputs were written outside the
synchronized workspace and copied back after measurement.

| Metric | CPU | Metal |
| --- | ---: | ---: |
| Warm full-image median (seconds) | 60.587746 | 6.555771 |
| Warm samples (seconds) | 60.587746, 60.5970169, 59.8777062, 60.3817336, 61.4762161 | 6.51789592, 6.65786438, 6.62444417, 6.54459021, 6.555771 |
| Initial pipeline including load/decode (seconds) | 61.3432365 | 28.6437559 |
| Repeated-result-cache median (milliseconds) | 10.604334 | 10.629667 |
| Peak process RSS (bytes) | 4,936,400,896 | 2,673,901,568 |
| Backend weight buffer (bytes) | 3,369,375,008 | 1,795,849,440 |
| Compute-buffer high-water mark (bytes) | 961,062,048 | 1,245,268,288 |

Hardware/software: M4 Pro (14 CPU / 20 GPU cores, 48 GiB), macOS 27.0 (26A428),
SDK 27.0, AppleClang 21.0.0, Release/native CPU, Apple Accelerate enabled,
KleidiAI disabled, unchanged GGML 0.25.3 revision and Metal patch. Metal then CPU
ran on battery power (30% to 28%); no thermal/performance warning was recorded.
Ordinary desktop activity remained and clocks were not fixed. Model loading
includes backend initialization; Metal runtime shader compilation can contribute
to initial-pipeline latency. These new samples do not establish a container
speedup relative to historical runs.

The table and reproduction steps are in [BENCHMARK.md](../../BENCHMARK.md).
The complete receipt, including binary/model/image/patch hashes, raw stage/cache
samples and power/thermal state, is `build/gguf-migration/benchmarks/receipt.json`
(SHA-256 `e282f7e4bdc88315e13c4212f0d46c4cac469f8bfd7d0a4971016bf3e243619f`).
Results are under `build/gguf-migration/benchmarks/{00-metal_after,01-cpu_after}/`.

## User-document cleanup archive: conversion identities and local results (2026-10-04)

This appendix retains the pinned artifacts, payload-size table, and local
validation statements removed from the concise model download/conversion guide
and GGUF contract page. It is historical project evidence, not a requirement
for ordinary conversion use.

### Pinned source artifacts

| Input | Revision / identity |
| --- | --- |
| HF weights | `facebook/sam3`, revision `3c879f39826c281e95690f02c7821c4de09afae7` |
| Original checkpoint | `sam3.pt`, 3,450,062,241 bytes; SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| Meta source and reference graph | revision `2345a4ad109ac29c569da749c91d84f10dc08c40` |
| Tokenizer asset | `bpe_simple_vocab_16e6.txt.gz`; SHA-256 `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |
| C++ port and tensor mapping basis | `PABannier/sam3.cpp`, revision `416186c501d060df7ca02989d49b38080f5f81f3` |
| Earlier supplementary C++ checkpoint | `PABannier/sam3.cpp` HF revision `a3892b63b918e872671322e116982a8910f0ffb7` |

### Converted GGUF artifact identities

| Output | Size (bytes) | SHA-256 | Local validation scope at the time |
| --- | ---: | --- | --- |
| `sam3-f32.gguf` | 3,371,139,456 | `cb13ecd5012a2fe19b06d840049be6daa6b177b35256af8c4afeb12125352486` | Schema-1 image file; CPU or explicitly selected Metal |
| `sam3-f16.gguf` | 1,797,613,888 | `66731fa5def347677f78d7422b81979be0f8e2f7ead941db9a466d1cfa715120` | Schema-1 image file; CPU or Metal |
| `sam3-video-f32.gguf` | 3,449,345,696 | `02513232afca5ba8c174b66c7fc839c67b32590df4b53bdd6df5089a6a546844` | All five original video cases (216 frames) pass on CPU/Metal; max stage L2 0.000203 CPU / 0.000346 Metal |
| `sam3-video-f16.gguf` | 1,837,925,216 | `9c9bc86c81d11a041db10a46d3d1e8ecaa1cbcf6fad683b00901f641746bf32c` | Diagnostic: Metal candidate selection fails at entry frames 23/24; the other four cases pass; full CPU diagnostic was deferred |
| `sam3-video-hybrid-v1.gguf` | 2,765,012,640 | `3975b4b1a10b962c6fad5022e2798baa6266aad7dbcc39210b459537208e1a05` | All 1,464 payloads checked; five-case/216-frame video corpus, image and session checks pass on CPU/Metal; max full-video L2 0.002650 CPU / 0.002594 Metal |

For the explicit F32/F16 files, every image-subset tensor matched the accepted
schema-1 file byte-for-byte. The FP16 video converter retained the final
single-output object score projection in F32 for its canonical GGML vector
shape and runtime policy. The original-checkpoint verification is also recorded
in `docs/plans/20261002-032709-local-model-meta-validation.md`; the completed
video support scope is in `docs/plans/20261002-182848-m2-complete-acceptance.md`.

### Quantized schema-3 status snapshot

At the 2026-10-03 documentation snapshot, four vision-only schema-3 profiles
passed output quality on the fixed seven-case corpus across CPU/BLAS, native CPU
and Metal (84/84). Raw tensor fidelity was a separate axis: 0/84 cells passed
the complete gate, while 500/840 tensor-by-case comparisons passed and 340
failed. These numbers describe that fixed corpus only; they are not a
dataset-wide accuracy statement. The broad ViT-plus-text profiles remained
diagnostic. Current conversion commands and support status are maintained in
the quantization guide, outside this container-migration plan.

Documentation, local links and owned-file whitespace checks pass. Historical
plans retain their original format and benchmark identities. No commit or push
was performed.
