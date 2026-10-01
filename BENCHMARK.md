# Benchmarks

The table below records GGUF measurements on 2026-10-01.
Model files and conversion instructions are in
[MODEL_ZOO.md](MODEL_ZOO.md). Each measured cell reports **warmed full-image
median latency / peak process RSS**. Lower latency is better; GB means
1,000,000,000 bytes. Unmeasured configurations have no inferred numbers.

<table>
  <thead>
    <tr>
      <th rowspan="2" scope="col">Model / checkpoint precision</th>
      <th colspan="2" scope="colgroup">Apple M4 Pro: 14 CPU cores (10P + 4E), 20 GPU cores, 48 GiB unified memory<br>macOS 27.0 (26A428)</th>
    </tr>
    <tr>
      <th scope="col">CPU: 4 threads<br>GGML 0.25.3, AppleClang 21.0.0</th>
      <th scope="col">Metal: runtime-compiled shaders<br>GGML 0.25.3, macOS SDK 27.0, AppleClang 21.0.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">SAM 3 image / FP16 GGUF</th>
      <td><strong>60.588 s</strong> / 4.936 GB</td>
      <td><strong>6.556 s</strong> / 2.674 GB</td>
    </tr>
    <tr>
      <th scope="row">SAM 3 image / FP32 GGUF</th>
      <td>Not benchmarked; numerical acceptance passed</td>
      <td>Not benchmarked; explicit Metal path implemented, local numerical acceptance pending</td>
    </tr>
  </tbody>
</table>

Only CPU and Metal are currently supported. CUDA, video pipelines and other
model families have no benchmark entries yet.

## Measurement conditions

- Official SAM 3 checkpoint revision `3c879f39826c281e95690f02c7821c4de09afae7`;
  converted FP16 GGUF v3 / SAM schema 1 SHA-256
  `66731fa5def347677f78d7422b81979be0f8e2f7ead941db9a466d1cfa715120`.
- One RGB image, `truck.jpg`, 1800 x 1200, resized to the model's 1008 x 1008
  input; prompt `truck`, score threshold `0.5`, batch size one.
- One initial pipeline call followed by five measured complete-image calls.
  Each measured call replaces the image and runs text segmentation, including
  preprocessing and postprocessing. Model loading, image decoding, output-file
  writes and repeated-result-cache calls are excluded from the warm latency.
- Release build, native CPU instructions, AppleClang
  `21.0.0` (`clang-2100.3.34.2`), GGML commit
  `353b63b439f27ab2cc19dac97ab1681ba6d2d084`,
  [precision/window patch](cmake/patches/README.md)
  `0a0b80dd15c2a8b5a05d148a31e9e53f8df4d0555852ef301da336a9d97b7c48`.
  Apple Accelerate/BLAS is enabled; KleidiAI is disabled.
- CPU uses four threads. Metal also receives a four-thread CPU fallback setting;
  the tested full image graph runs 3,342 Metal nodes, zero CPU nodes and six
  graph partitions. Host preprocessing/postprocessing still run on CPU.
- Backends ran sequentially, Metal then CPU, on battery power (30% to 28%), with other project computation
  stopped. Ordinary desktop activity remained. No thermal warning was reported;
  constant CPU/GPU clocks were not verified.

FP16 names the checkpoint storage policy, which preserves selected tensors in
FP32. CPU loading promotes stored FP16 values exactly to FP32; Metal retains
packed FP16 weights and requests precise arithmetic. See the
[precision contract](cmake/patches/README.md). Peak RSS is the process high-water
mark across loading and inference. It is not dedicated GPU VRAM, and must not
be added to backend buffer sizes.

These single-image timings do not establish video frame rate. The independent
seven-case numerical suite passed FP32/CPU, FP16/CPU and FP16/Metal acceptance.
Raw samples, memory figures and provenance are preserved in the
[GGUF acceptance record](docs/plans/20261001-020507-gguf-conversion-loading.md).
The local ignored receipts are
`build/gguf-migration/benchmarks/{00-metal_after,01-cpu_after}/results.json` and
`build/gguf-migration/benchmarks/receipt.json`.

Earlier custom-container measurements and controlled GGML/window comparisons
remain in the [previous performance record](docs/plans/20261001-002850-metal-window-cpu-performance-official-weights.md).
The current table is a new measurement batch, not a controlled before/after
container comparison; its difference from earlier timings does not establish
a GGUF speedup.

## Reproduce

Run from the repository root after following [model conversion](MODEL_ZOO.md#convert-the-supported-runtime-files)
and the [CPU/Metal build instructions](README.md#build). Prepare the pinned input:

```sh
mkdir -p models/fixtures
curl -fL https://raw.githubusercontent.com/facebookresearch/sam3/2345a4ad109ac29c569da749c91d84f10dc08c40/assets/images/truck.jpg \
  -o models/fixtures/truck.jpg
shasum -a 256 models/fixtures/truck.jpg
```

Expected image SHA-256:
`941715e721c8864324a1425b445ea4dde0498b995c45ddce0141a58971c6ff99`.
Choose new output directories on a local, unsynchronized disk; the CLI refuses
to overwrite existing directories. Run the following commands sequentially:

```sh
build/cpu/examples/sam_image \
  --model models/sam3-f16.gguf --image models/fixtures/truck.jpg \
  --text truck --backend cpu --threads 4 --score-threshold 0.5 \
  --repeat 5 --output /tmp/sam-benchmark-cpu

build/metal/examples/sam_image \
  --model models/sam3-f16.gguf --image models/fixtures/truck.jpg \
  --text truck --backend metal --threads 4 --score-threshold 0.5 \
  --repeat 5 --output /tmp/sam-benchmark-metal
```

Read `timing_ms.warmed_full_image_median` (milliseconds) and
`runtime.process_peak_rss_bytes` from each `results.json` for the table.
Retain `warmed_full_image_runs` and `repeated_result_cache_runs` for inspection.
The inference stage includes text encoding; do not add those stage times twice.
`cold_start` includes initial loading/decoding but excludes process startup and
may reuse OS/shader caches. Cache timing measures reuse of the previous result.

## Add a result

Add a model/checkpoint/precision row and a hardware column group using the same
two header rows. Record exact CPU/GPU variant, RAM/VRAM, OS, backend, compiler,
GGML revision/patch and relevant backend versions. A CUDA column must name the
actual GPU, CUDA Toolkit and NVIDIA driver versions; add cuDNN only if used.
Record the checkpoint hash, input/workload, thread count, warmup/repeats,
individual samples, peak-memory definition, power state and numerical acceptance.
Keep different workloads in separate tables and mark unsupported or unmeasured
combinations explicitly.
