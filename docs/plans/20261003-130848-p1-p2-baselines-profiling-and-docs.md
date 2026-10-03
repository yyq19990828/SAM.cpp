# P1/P2 baseline, profiling and documentation work

Created: 2026-10-03T13:08:48.050568+08:00.
Baseline: main 56a4cde plus the uncommitted precision-documentation update.
Status: complete. The subsequent measured optimization is complete in the [visual encoding plan](20261003-134534-visual-encoding-profile-and-optimization.md). No commit/push requested for this batch.

## Scope

1. Shorten BENCHMARK and MODEL_ZOO while preserving precision, measurements,
   identities and support limits. Maintain matching BENCHMARK_zh.md and
   MODEL_ZOO_zh.md, and retain historical detail in normal docs files.
2. Preserve the completed local acceptance inputs/builds/outputs/receipts in
   a durable ignored private bundle. Add a small versioned discovery index
   and a stdlib archive verifier; relocation does not relabel old receipts.
3. Promote existing deterministic one/four-object fixture and original Meta
   17-frame qualification recipes into portable tools. Permit explicit
   qualification reuse only when every 64-frame PNG and recipe property
   matches the sealed parent; retain the parent identity and scope.
4. Expose existing RuntimeStats image_ms/inference_ms in JSON and report
   optional stage distributions. Use a short real Metal run to identify the
   next bottleneck; do not alter kernels, weights, candidate policy, BF16/F16
   boundaries or the 15-frame hotstart delay in this profiling phase.
5. Add pinned read-only quick CI for CPU/header/tool/docs checks without
   external model downloads. Preserve explicit F16 failure/CPU diagnostic
   deferral; do not spend another full CPU run on this known rounding limit.

## Verification

- Focused failure/acceptance regressions for archive corruption/overlap,
  qualification mismatch, optional timing metadata and bilingual table drift.
- Locked isolated tools suite, current weight-free CPU/Metal builds and header
  checks as appropriate; no repeated full CPU numerical/performance matrix.
- Fresh fixture PNG hashes must match the old 128 PNGs before reusing oracle
  qualification. A diagnostic profile is not a new 64-frame benchmark PASS.
- Verify all copied archive bytes, preserve historical manifests and bind new
  evidence to actual source/model/binary/libs. Check docs links/parity and
  git diff --check. Remote CI is pending until an authorized push/run occurs.

## Results

- Root summaries reduced to 63-line BENCHMARK and 75-line MODEL_ZOO, with
  corresponding _zh documents and a Chinese validation workflow. Historical
  precision, identities and measurements are retained in detail records.
  Numeric table parity and local links pass automated checks.
- A durable private bundle under models/validation-baselines/m2-m4pro-20261003
  preserves 21,239 files / 85,994,641,754 logical bytes using APFS clones. Each
  copy hash was verified. All151 sealed receipt identities also match archived
  counterparts. The small versioned baseline index binds the archive manifest.
- Portable generator produces all128 PNG hashes identical to the previous
  fixtures. Original Meta qualification is explicitly inherited through parent
  identity and identical protocol/input checks; final v3 receipts bind current
  tools. Fresh Meta execution uses the promoted original recipe but is not
  repeated in this batch. Corrupted inputs/invalid prefix/reuse drift are tested.
- Existing image_ms/text_ms/inference_ms now serialize into runtime JSON.
  Benchmark analysis handles new stages and explicitly missing historical data.
  No graph, weight, precision, candidate or temporal policy was changed.
- Independent current CPU/Metal CLI builds and serialization checks pass.
  Fresh CPU quick build (Metal/BLAS/native disabled, pinned local GGML) compiles
  all targets and independent headers; CTest11/11 passes. Isolated tools24/24,
  documentation and whitespace checks pass. Full CPU model acceptance is reused.
- Two short Metal diagnostic clips complete with stable 1/4 objects and zero
  CPU graph nodes. They were Battery Power, declared length3, two propagated
  samples including final drain: image encoding medians9.198/9.673s, detector
  pipeline0.905/0.918s, tracker0.671/3.160s. They identify the image path as
  dominant, not a new steady-state benchmark or proof of speedup.
- Pinned, read-only quick CI is configured without model downloads. Action
  tag identities and the official CPU Torch wheel were checked; YAML and local
  commands pass. Remote CI has not run for this uncommitted workflow.

The subsequent visual encoding plan profiles ViT/necks/geometry/transfers,
implements shared projection folding and optional CPU BLAS, and completes fresh
numerical/session/performance acceptance. Current CPU video medians are
9.155/13.887s (1/4 objects). Four-object Metal tracking remains a future profiling
target; the official hotstart delay stays unchanged.
F16 remains an explicit diagnostic limit, and full CPU216 stays deferred.

Receipts: build/p1-p2/20261003/{archive-result.json,encoding-profile.json,
current-cli-builds.json,*-qualification-v3.json,tools-tests-checked.log} and
the private archive manifest. The earlier malformed profile-launch script
failed before inference; profile.log is retained separately from profile-v2.


<a id="english-validation-history"></a>

## english-validation-history

迁自 `docs/validation.md` 的本地基线、归档和复用记录；历史批次结论保持原样。

### Validation and reusable baselines

[中文](../validation_zh.md)

#### Quick checks

```sh
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
cmake --build build/cpu --parallel 2
ctest --test-dir build/cpu --output-on-failure
build/reference-runtime/venv/bin/python tools/test_tools.py
python3 tools/check_docs.py
```

Ordinary CTest never loads model weights. Quick CI performs CPU/header/tool/doc
checks; remote success requires a pushed workflow and an actual completed run.
Model acceptance is separate. The [current baseline index](../validation-baselines/blas-vit-m4pro-20261003.json)
records the accepted CPU BLAS/ViT implementation; the
[historical parent](../validation-baselines/m2-m4pro-20261003.json) preserves original
models and Meta references.

#### Preserve and verify evidence

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

#### Portable performance fixtures

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

#### Full performance protocol and profiling

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
unchanged. See [measured evidence](20261003-134534-visual-encoding-profile-and-optimization.md#english-profiling-record).



<a id="chinese-validation-history"></a>

## chinese-validation-history

迁自 `docs/validation_zh.md` 的本地基线、归档和复用记录；历史批次结论保持原样。

### 验证与基线复用

[English](../validation.md)

#### 快速检查

按英文页的命令执行 CMake 构建、CTest、`tools/test_tools.py` 和
`tools/check_docs.py`。普通 CTest 不加载模型权重；快速 CI 检查 CPU、独立头文件、
工具和文档。远端 CI 必须等工作流推送后实际运行完成才可宣称通过。
[基线索引](../validation-baselines/m2-m4pro-20261003.json)记录 M4 Pro 已验收批次。

#### 长期保存证据

`tools/archive_validation.py create` 可接收多个 `--source` 和一个新 `--output`；
`verify` 对保存的 `bundle` 校验文件数量、大小和哈希。目标路径不能覆盖已有内容，
也不能与源路径重叠。APFS 优先使用写时复制克隆，失败的暂存保留未完成记录。

本机 M2 归档位于 `models/validation-baselines/m2-m4pro-20261003/bundle`，
包含 21239 个文件、85994641754 字节逻辑数据；所有副本已校验，
151 份封存收据在归档中的身份也逐项核对。权重、私有媒体和原始数据不进入 Git。
`archive.json` 的 roots 映射说明原路径与保存位置。原收据保留历史路径；
执行旧二进制可能需要恢复其路径及 RPATH。归档保存字节，不代表重新跑过模型，
也不会自动让修改后的程序取得验收资格。

复用要求模型、二进制、库、输入等证据匹配。计算图、精度或预处理改动需要对应
数值验证；文档或计时输出改动使用针对性检查。旧收据不能改标为新执行结果。
F16 视频继续保持诊断状态，CPU 全量诊断仍延后。

#### 正式性能样本与 Meta 资格检查

在仓库根目录、锁定的 Python 环境中运行；原始卡车图像、Meta 源码、权重和 BPE
按模型文档准备。`tools/generate_video_benchmark.py` 使用 `--image`、`--output`
生成固定的 64 帧单/四对象样本，生成本身不证明对象行为已经正确。

`tools/qualify_video_benchmark.py` 接收 `--fixture-manifest`、`--workload`、
`--sam3-source`、`--sam3-runtime-source`、`--checkpoint`、`--bpe`、`--output`。
它在声明总长 64 的序列上执行原始 Meta 模块前 17 帧，检查固定对象数、ID、
非空掩码和热启动输出。分别检查 one-object、four-object，使用不同输出。
这只是样本资格检查，不能替代模型全量验收。完整命令见英文页。

已有可信封存资格记录时，可使用 `--reuse-from` 和 `--parent-fixture` 替代重新
运行 Meta；新样本必须在全部 64 张 PNG 字节、位置、尺寸、提示词与序列协议上
一致。复用收据记录父收据哈希，输入、源文件或父记录不匹配会失败。
新样本仍可做新的 Meta 检查；复用模式不下载权重，也不执行模型。

#### 完整性能协议与诊断计时

`tools/benchmark_video.py` 的 `--recipe-script` 使用
`tools/generate_video_benchmark.py`，同时提供原始数值参考、相同可执行程序的
完整通过收据、样本清单和资格收据。接电、避免睡眠，CPU/Metal 单/四对象四项
串行测量；检查 64 帧、16 帧预热、48 个测量样本、固定 ID、状态、后端和哈希。
最后一帧的排空开销包含在测量中。

当前 CLI 的 `runtime.image_ms` 计时包含视觉主干、neck 与几何编码；
`runtime.inference_ms` 包含提示准备、融合、检测和掩码，均包括传输与分配。
`text_ms` 表示最近一次文本编码，视频会话只执行一次；替换图像会再次编码，
这部分已包含在图像的 `inference_ms` 中。
历史数据没有这些字段时明确记录不可用，不用减中位数的方式补造阶段时间。
短序列可用于定位瓶颈，不计作新的 64 帧稳态性能验收。
后续优化继续保持 F16/BF16 边界、候选选择门槛与热启动延迟规则。

图像数值验收读取导出的 RGB PPM，图像计时通过 stb 读取原始 JPEG。
不同 JPEG 解码器可能产生不同 RGB；跨入口逐字比较必须使用相同像素。
本次图像测量保留同一 JPEG 的单次推理对照，检查输出一致性，不计为额外的
Meta 原始参考；数值门槛保持原样。见[当前测量](20261003-134534-visual-encoding-profile-and-optimization.md#chinese-profiling-record)。

[当前基线索引](../validation-baselines/blas-vit-m4pro-20261003.json)记录 CPU BLAS/ViT
新实现。新私有归档包含 14664 个文件、43333663701 逻辑字节，位于
`models/validation-baselines/blas-vit-m4pro-20261003/bundle`，创建和独立复核均校验
每个文件及清单。新源文件、构建、输出与保留的失败诊断在此；原始模型及 Meta
参考继续由[父索引](../validation-baselines/m2-m4pro-20261003.json)绑定的归档保存。
两个归档同时保留，不自动授予修改后二进制的通过状态。

索引另记录测量后的统计字段顺序/归档校验修正：隔离构建、旧初始化回归、
逐字节张量/输出对照及会话检查支持该元数据修正的针对性复用。
完整测量保留原始源文件/二进制身份；新构建不能把旧收据冒充为自己的完整验收。
