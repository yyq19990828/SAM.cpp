# Model verification

[中文](validation_zh.md) · [Models](../MODEL_ZOO.md)

Use reference comparisons when converting a model, adding an adapter/backend, or
changing preprocessing and arithmetic. The current reference tools cover the
SAM 3 adapter; future model families need their own schemas and task-specific
checks. A successful build alone does not validate a new platform or model.

## Prepare the reference environment

Follow [download and conversion](models/sam3-details.md) to obtain the original
checkpoint, tokenizer, and Python environment. Set `sam3_weights_dir` to that
directory. Check out the pinned official SAM 3 source and set `SAM3_SOURCE_DIR`
to its absolute path. Copy its `assets/images/truck.jpg` and `groceries.jpg`
into ignored `models/fixtures/` for the supplied test cases.

Reference preparation uses a separate source copy so the upstream checkout
remains unchanged. Use new output directories for source copies, references,
and comparisons. The pinned dependency environment currently matches the
validated platform; another platform needs a compatible environment.

## Compare image inference

Run from the repository root:

```sh
.venv-reference/bin/python tools/prepare_reference_source.py \
  --source "$SAM3_SOURCE_DIR" --output build/reference-runtime/sam3-cpu
.venv-reference/bin/python tools/export_reference.py \
  --sam3-source "$SAM3_SOURCE_DIR" \
  --sam3-runtime-source build/reference-runtime/sam3-cpu \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --cases tests/data/sam3-image-cases.json --device cpu \
  --output models/reference/sam3-f32
.venv-reference/bin/python tools/validate_image.py \
  --build-dir build/cpu --model models/sam3-f32.gguf \
  --reference models/reference/sam3-f32 --backend cpu \
  --output build/image-validation
```

Change the model, build directory, and backend together for another configuration.
The validator checks tokenizer IDs, shapes, finite tensors, masks, scores,
coordinates, and input provenance. Quantized SAM 3 profiles use output quality
as the primary result and report full-tensor fidelity separately. Community
weights and partial references are diagnostic inputs, not replacements for the
original-model comparison.

Repository numerical acceptance focuses on the fixed vision and full-component
presets. Schema-4 custom combinations require `--allow-custom-quantization`;
their reports remain diagnostic even when every example passes. The validator
binds module selection across GGUF, sidecar and runtime output, and verifies
each tensor's allocation explanation.

Use identical decoded pixels when comparing entry points. Different JPEG
libraries can produce different RGB values; exported PPM inputs avoid that
ambiguity.

## Compare video inference

Use a full video GGUF. The supplied sequence generator and validator exercise
motion, entering objects, occlusion, hotstart, and negative prompts.

```sh
.venv-reference/bin/python tools/generate_video_cases.py \
  --input-root models/fixtures --output models/video-cases
.venv-reference/bin/python tools/prepare_reference_source.py --task video \
  --source "$SAM3_SOURCE_DIR" --output build/reference-runtime/sam3-video-cpu
.venv-reference/bin/python tools/export_video_reference.py \
  --sam3-source "$SAM3_SOURCE_DIR" \
  --sam3-runtime-source build/reference-runtime/sam3-video-cpu \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --frames models/video-cases --output models/reference/sam3-video
.venv-reference/bin/python tools/validate_video.py \
  --build-dir build/metal --model models/sam3-video-hybrid-v1.gguf \
  --reference models/reference/sam3-video --backend metal --threads 4 \
  --output build/video-validation
```

The video validator checks ordered outputs, persistent IDs, masks, model state,
and backend placement. Subset exports using `--case` or `--max-frames`, and
comparisons with `--allow-diagnostic`, do not establish complete video support.

## Run development checks

```sh
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure
.venv-reference/bin/python tools/test_tools.py
python3 tools/check_docs.py
git diff --check
```

Ordinary CTest is weight-free. Reference tests can be enabled with
`SAM_REFERENCE_DIR`, `SAM_REFERENCE_MODEL`, `SAM_REFERENCE_BACKEND`, and the
reference environment's `Python3_EXECUTABLE`, after exporting the full corpus.

For image timing, use `sam_image --repeat 5`; it measures full-image calls and
cache hits separately. `tools/generate_video_benchmark.py`,
`tools/qualify_video_benchmark.py`, and `tools/benchmark_video.py` provide video
workload generation, original-model qualification, and timing. Inspect each
script's `--help` before running it. Keep timing separate from tensor exports
and other model work. [Performance](../BENCHMARK.md) explains the reported metrics.

Keep weights, private media, and generated outputs outside Git. Experimental
procedures, specific gates, failure investigations, and preserved batch evidence
belong in the matching implementation plan under `docs/plans/`.
