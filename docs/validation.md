# Validation and reusable baselines

[中文](validation_zh.md)

## Quick checks

```sh
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
cmake --build build/cpu --parallel 2
ctest --test-dir build/cpu --output-on-failure
build/reference-runtime/venv/bin/python tools/test_tools.py
python3 tools/check_docs.py
```

Ordinary CTest never loads model weights. Quick CI performs CPU/header/tool/doc
checks; remote success requires a pushed workflow and an actual completed run.
Model acceptance is separate. The [current baseline index](validation-baselines/blas-vit-m4pro-20261003.json)
records the accepted CPU BLAS/ViT implementation; the
[historical parent](validation-baselines/m2-m4pro-20261003.json) preserves original
models and Meta references.

## Preserve and verify evidence

```sh
python3 tools/archive_validation.py create \
  --source /absolute/path/to/immutable-build-and-results \
  --source /absolute/path/to/model.gguf \
  --source /absolute/path/to/model.gguf.manifest.json \
  --output models/validation-baselines/new-batch
python3 tools/archive_validation.py verify models/validation-baselines/new-batch/bundle
```

The destination is exclusive. All copied bytes are hashed; APFS uses copy-on-write
clones when available. Roots and file hashes live in `bundle/archive.json`.
Failed staging remains incomplete for inspection. Archive roots/files must not
overlap the destination. Models, private media and raw data stay ignored.

The completed local M2 archive contains 21239 files and 85994641754 logical bytes
under `models/validation-baselines/m2-m4pro-20261003/bundle`. Each copy was verified,
and all 151 sealed receipt identities were checked in the archive. The original
receipts keep their original paths; the roots map identifies stored counterparts.
This is byte preservation, not a new numerical run or automatic eligibility for
a changed executable. Original paths/RPATHs may need restoration to execute
historical binaries. Keep system/backend/source/input identities with the bundle.

The current index also names a post-measurement stats-field-order/archive-verifier
repair. Its isolated builds, legacy-initializer regression, bitwise tensor/output
parity and session checks justify focused reuse for this metadata-only delta.
Recorded heavy runs retain their original source/binary identities. A newly built
executable still cannot use an old receipt as its own full-validation receipt.

The new CPU BLAS/ViT bundle contains 14664 files and 43333663701 logical bytes at
`models/validation-baselines/blas-vit-m4pro-20261003/bundle`. Creation and independent
verification check every byte and inventory. Current source/build/outputs and
retained failed diagnostics are here; unchanged original weights/Meta references
are shared through the sealed parent index. Neither archive replaces the other.

Reuse only matching model/binary/library/input evidence. Graph, storage or
preprocessing changes require affected numerical tests; docs and timing-output
changes use focused checks. Never relabel an old receipt as a new execution.
F16 video remains diagnostic; its full CPU run stays deferred.

## Portable performance fixtures

Run from repository root with the locked Python environment. The original truck
image and Meta checkpoint/source/BPE are obtained through the Model Zoo workflow.

```sh
.venv-reference/bin/python tools/generate_video_benchmark.py \
  --image models/fixtures/truck.jpg --output models/video-benchmark-fixtures
.venv-reference/bin/python tools/qualify_video_benchmark.py \
  --fixture-manifest models/video-benchmark-fixtures/fixture-manifest.json \
  --workload one-object --sam3-source "$SAM3_SOURCE_DIR" \
  --sam3-runtime-source build/reference-runtime/sam3-video-cpu \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/one-object-qualification.json
```

Repeat qualification for `four-object` with a different output. It executes the
original Meta modules on 17 frames initialized with declared length64, and checks
stable one/four IDs, nonempty masks and the 15-frame output policy. It is fixture
qualification, not a replacement for full model acceptance.

An explicit reuse path avoids repeating that oracle when all64 PNG bytes,
positions, dimensions, prompt and declared-length protocol match a trusted
sealed qualification:

```sh
.venv-reference/bin/python tools/qualify_video_benchmark.py \
  --fixture-manifest models/video-benchmark-fixtures/fixture-manifest.json \
  --workload one-object --reuse-from /path/to/sealed-prefix.json \
  --parent-fixture /path/to/original-fixture-manifest.json \
  --output models/one-object-reused-qualification.json
```

The derived receipt records the parent hash and unchanged input scope. Different
inputs, incomplete/stale parents or changed source hashes are rejected. Keep the
parent itself with the archive. Fresh qualification remains available for a new
fixture; no weight download or oracle execution occurs in reuse mode.

## Full performance protocol and profiling

```sh
.venv-reference/bin/python tools/benchmark_video.py \
  --build-dir build/metal --model models/sam3-video-hybrid-v1.gguf \
  --reference models/reference/sam3-video \
  --validation build/video-validation-metal-hybrid/metrics.json \
  --fixture-manifest models/video-benchmark-fixtures/fixture-manifest.json \
  --qualification models/one-object-qualification.json \
  --recipe-script tools/generate_video_benchmark.py \
  --workload one-object --backend metal --threads 4 \
  --output build/benchmark-metal-one
```

Run one/four workloads on CPU/Metal sequentially, on AC power without sleep,
after passing the same executable's full numerical validation. The runner checks
64 outputs,16 warmup,48 samples, fixed IDs, state, backend and artifact hashes.
Final drain remains measured. Historical results lacking encoding timers report
those fields unavailable; they are never reconstructed by subtracting medians.

Current CLI runtime JSON exposes `image_ms` for visual/necks/geometry and
`inference_ms` for prompt/fusion/detection/masks. These include allocation and
transfers. `text_ms` describes the last text encode, which occurs once per video
session. Image replacement encodes text again, included in image `inference_ms`.
A short run may identify a bottleneck, but does not become a 64-frame steady-state
benchmark. Preserve F16/BF16 boundaries, exact candidate gates and hotstart delay
while evaluating later kernel/batching/transfer improvements.

Image numerical validation consumes exported RGB PPM inputs; image timing uses
the original JPEG via stb. Different JPEG decoders can produce different RGB.
Use identical decoded pixels for exact cross-entry comparisons. Current image
measurements retain a matching-JPEG single-inference bridge; it is an output
consistency check, not an additional original-Meta oracle. Numerical gates stay
unchanged. See [measured evidence](benchmarks/m4-pro-blas-vit-20261003.md).
