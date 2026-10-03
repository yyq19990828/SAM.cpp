# Visual encoding profile and optimization

Created: 2026-10-03T13:45:34.515638+08:00.
Baseline: main56a4cde plus current P1/P2 WIP. No commit/push requested.
Status: complete. All changes remain uncommitted; no push performed.

## Scope and approach

Measure graph construction/allocation/upload/compute/download/BF16 rounding and
geometry in the existing fused visual encoder. Separately measure original ViT
and neck graphs in an isolated diagnostic to locate compute cost; splitting is
not a production change. Use unchanged weights, input preprocessing and F32
requests. Compare diagnostic outputs against the unchanged full encoder.
Choose a concrete optimization only after these measurements establish its
benefit; preserve candidate, precision, temporal and numerical gates.

## Steps and verification

1. Compile a standalone private probe from current headers, linked to immutable
   Metal libraries. Keep production graphs unchanged while measuring phases.
2. Run matched input/weights serially under recorded power conditions. Verify
   canonical output values and stage placement. Do not compare this battery/
   short diagnostic to old AC steady-state numbers as a speedup.
3. Implement one measured simplification/optimization if justified, leaving a
   regression that checks the changed behavior. Revalidate affected outputs
   using matching baselines; do not repeat the entire CPU suite unnecessarily.
4. Update plan, changelog and synchronized bilingual summaries as appropriate.

## Results

Current CPU/Metal Release builds, independent headers and CTest pass (11/11
each). The current no-BLAS/no-Metal/non-native quick build also compiles all
targets and independent headers; CTest11/11 passes in12.01s. Its receipt is
`build/p1-p2/20261003/reaccept/no-blas-checks.json`. Documentation links/table
parity and whitespace checks pass. Remote CI has not run.

The new supported numerical matrix and subsequent image, short/long-session
and 64-frame performance runs are serially queued. The complete F32/hybrid
CPU/Metal video matrix passes: four five-case/216-frame cells,864 frames, with
original gates and artifact/output guards. Metal has zero CPU/BLAS graph nodes
in every case. The matrix controller exits0. Subsequent image, session and
performance checks remain pending; video acceptance is not performance evidence.

CPU/Metal image regressions pass all70 cases (schema1 F32/F16 and schema2
F32/F16/hybrid, seven each/backend). Real short sessions pass all six profiles. The
isolated current CPU long-session driver compiles and passes128 interleaved
pushes. Its positive session matches exact outputs/history with the new accepted
CPU entry64; negative outputs/state stay empty, lifecycle/owned-output checks
pass, and weight/compute high-water remains constant after warmup. Metal's
isolated long-session driver also passes the same128-push comparison and
lifecycle/bounds checks with zero CPU graph nodes. All correctness/session
checks are complete. Four fresh64/16/48 video performance cells and four
one-warmup/five-repeat image cells also pass, with AC endpoints and no system
sleep inside measured intervals. CPU video medians are9.155/13.887s for1/4
objects (historical51.102/81.644s); Metal7.550/13.166s. These are same-protocol
historical comparisons, not interleaved A/B runs. Metal four-object tracking
(7.277s) is the remaining largest stage in that workload.

The final closure verifies373 source/model/binary/library/input identities and
104 output trees. New durable private archive:14664 files/43333663701 logical
bytes, all APFS clones; creation and independent inventory/hash verification
pass. The original archive remains the shared model/reference parent. Current
bilingual summaries/details, baseline index, source-backed precision boundaries
and preserved failure scopes are updated. Documentation links/table parity and
whitespace pass. Current CPU/Metal Release CTest11/11 each and no-BLAS quick
CTest11/11 are retained; production Python tools are unchanged since tools24/24.
Remote quick CI remains unrun.

Four image performance cells run after the video queue succeeds, using the
original one-warmup/five-full-image protocol on truck.jpg. Their warmed times
exclude loading, decoding, file writes and repeated-result-cache calls. Each
cell binds the accepted seven-case image receipt, current CLI/source/library/
model/input hashes, exact accepted truck detections/masks and power/sleep logs.
This also refreshes the image table for the changed shared ViT implementation.
Preflight source readback corrected the private image-repeat counter from one
text encode (video semantics) to six (each replaced image encodes text). The
original waiter was stopped before any inference and remains an incomplete
record; `image_performance_v2.py` replaces it with no runtime/source change.

The v2 image benchmark stopped after its first Metal/F16 measurement because
its extra exact-output assertion compared stb-decoded JPEG with the official
Pillow-exported PPM validation input. Native decode readback proves163934 of
6480000 channel values differ (max3); same-JPEG single versus repeated inference
has exactly equal detections and mask bytes. The v2 failed receipt/raw output
is retained. V3 keeps exact comparison but first runs a guarded, identical-JPEG
single-inference bridge for each profile, then uses fresh measurement paths.
The bridge is not a new original-Meta oracle. Official numerical gates and
runtime sources are unchanged; no threshold is relaxed.

### Measured candidate and implementation scope

Q/K RoPE batching preserves seven feature-file bits but does not improve matched
A/B/A encoder latency (+0.35%); Metal concurrency-disable also has no benefit
(+0.20%). Both remain private rejected experiments.

ViT channel projections currently treat spatial height/window batches as many
small GEMMs. Folding shared-weight positions into one column dimension preserves
all seven Metal feature files bitwise and reduces the isolated encoder sample to
about5.15s. The selected CPU scheduler currently omits the registered BLAS ACCEL
device. With column folding and explicitly scheduled CPU BLAS, the same input
encoder falls from37.95s to6.56s, executing153 BLAS nodes. Post-BF16 feature L2
differences are at most0.000135; this is not full numerical acceptance.

Implement backend-neutral shared-channel projection and optional registry-based
BLAS in CPU driver/scheduler. Weights remain allocated on the selected CPU device;
BLAS is a priority host accelerator, counted as a subset of CPU work. Metal keeps
its existing selected/fallback list. Environments without BLAS retain native CPU.
GGML/BLAS thread requests are forwarded; Accelerate manages its own SGEMM threads,
so controlled experiments request VECLIB_MAXIMUM_THREADS=4 before process startup.
No global environment mutation or new public precision mode is introduced.

The graph/scheduling change requires new F32/hybrid CPU and Metal original-reference
checks on new binaries, targeted math/backend regressions and matched timing.
Old evidence remains archived; no failure is relabeled.


<a id="english-profiling-record"></a>

## English profiling record

迁自 `docs/benchmarks/m4-pro-blas-vit-20261003.md`，原始文件 SHA-256：`47702a579255d4ce2fb2b0cc3e85bce84e32c7d0ff44aa2c3c1962d2f217faa6`。测量值与历史结论保持原样。

### M4 Pro: ViT projection and CPU BLAS measurements

[中文](20261003-134534-visual-encoding-profile-and-optimization.md#chinese-profiling-record) · [Current summary](../../BENCHMARK.md)

Measured on 2026-10-03, Apple M4 Pro (14 CPU/20 GPU cores, 48 GiB), macOS 27.0
(26A428), Release AppleClang 21.0.0 and pinned GGML 0.25.3
`353b63b439f27ab2cc19dac97ab1681ba6d2d084` with the existing precision/window patch.
Implementation base is main 56a4cde plus uncommitted changes; exact source/binary/
library/input identities are bound by receipts, rather than inherited from HEAD.

Before commit, the BLAS counter moved to the end of the public stats aggregate
for source compatibility, and archive verification gained complete inventory/size
checks. The index records this metadata-only delta and focused CPU/Metal parity;
the measurements below remain bound to their immutable pre-fix binaries.

Original checkpoint SHA-256:
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
Video uses hybrid `visual-tracker-f32-v1`, GGUF SHA-256
`3975b4b1a10b962c6fad5022e2798baa6266aad7dbcc39210b459537208e1a05`.
Image uses the schema1 F32/F16 files identified in [Model Zoo](../../MODEL_ZOO.md).
Weight labels do not describe the whole pipeline; the existing F16 normalization
and BF16 feature/memory boundaries are unchanged.

The historical CPU build compiled BLAS but its scheduler omitted that ACCEL
device; native CPU already used Accelerate vector primitives. The new
CPU execution retains weight ownership and schedules an available registry BLAS accelerator
before native CPU for eligible matrices. `blas_nodes` is a subset of `cpu_nodes`.
ViT shared-weight channel positions are folded into GEMM columns. Metal keeps its
selected/fallback order and has zero CPU graph fallback in accepted runs.
GGML/BLAS receives four-thread requests; processes request
`VECLIB_MAXIMUM_THREADS=4` before startup. Accelerate manages SGEMM threads itself.

#### Image: one warmup and five complete calls

Original 1800×1200 `truck.jpg`, prompt `truck`, threshold 0.5, model input 1008×1008.
Loading, decoding, file writing and repeated-result-cache calls are excluded from
warmed latency. Peak RSS comes from the whole `/usr/bin/time -l` child process.
Seconds / decimal GB:

| Weight storage | CPU median / peak RSS | Metal median / peak RSS |
| --- | ---: | ---: |
| Mixed F16/F32 | 7.338 / 4.918 | 5.354 / 2.661 |
| F32 | 7.316 / 4.922 | 5.364 / 4.248 |

#### Video: 64 frames, 16 warmup, 48 measured

Independent qualified one/four-object fixtures, 1800×1200, prompt `truck`,
max_objects=8, no tensor dumps, final drain included. p95 uses linear interpolation
at `(n-1)*0.95`. Peak RSS includes loading and final drain.

| Objects | CPU median / p95 / peak RSS | Metal median / p95 / peak RSS |
| ---: | ---: | ---: |
| 1 | 9.155 / 9.352 / 5.212 | 7.550 / 7.595 / 4.193 |
| 4 | 13.887 / 14.555 / 5.356 | 13.166 / 13.370 / 4.340 |

Frame 0 appears after processing frame 14. First output-file observations, sampled
about once per second, include load/decode/file work: CPU 130.681/178.184 s;
Metal 103.536/155.065 s (1/4 objects). The temporal policy is unchanged.

##### Stage medians, seconds

| Backend | Objects | Image encoding | Detector pipeline | Tracker | Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpu | 1 | 6.448 | 1.105 | 1.480 | 0.075 |
| cpu | 4 | 6.481 | 1.109 | 5.932 | 0.309 |
| metal | 1 | 5.047 | 0.624 | 1.793 | 0.039 |
| metal | 4 | 5.043 | 0.624 | 7.277 | 0.161 |

Image encoding includes ViT, necks, geometry, transfers and allocation. Detector
pipeline includes prompt preparation, fusion, detection and masks. Video encodes
text once per session; replacing an image encodes text again, included in image
`inference_ms`. Independent stage medians cannot reconstruct the full-frame median.
Metal four-object tracking remains the largest measured stage.

All video cells retain constant weight/compute high-water after warmup: CPU
3447558112/1020660736 bytes; Metal 2763224992/1245268288 bytes. Retained memory is
bounded in this protocol by 27 records per object: 1/4 objects reach 27/108 records,
17943552/71774208 bytes. Allocation counters are not added to process RSS.
Asynchronous current-RSS samples are retained with frame tags in each receipt.
Finite observations do not prove unlimited-stream RSS bounds or real-time throughput.

#### Conditions, acceptance and preserved failures

Video cells ran serially 17:09:17–17:54:31 Asia/Shanghai; four fresh image cells
followed the same serial policy. All power endpoints were AC/100%, no thermal or
performance warning was recorded, and live power-log audit found no system sleep
inside a measured cell. Task-scoped caffeinate prevented idle sleep. No other
SAM/Meta inference or compilation overlapped measurements; desktop services stayed
active and clocks were not forced. Source/model/binary/library/input/output hashes
were verified after completion: 373 unique identities and 104 output trees.

Fresh acceptance passes 864 video frames (F32/hybrid × CPU/Metal, 216 each), 70 image
cases, six real short sessions and two 128-push interleaved long checks. Long positive
outputs/history match their accepted standalone entry64 exactly; negative state
stays empty, model lifetime/owned outputs/session isolation pass. Peak long-process
RSS is 5.555 GB CPU and 4.595 GB Metal; weight/compute high-water stays constant after
warmup. These are finite checks, not a universal memory bound.

The extra v2 image benchmark assertion mistakenly compared stb-decoded JPEG with
Pillow-exported reference PPM. 163934/6480000 RGB channel values differ (max 3).
Same-JPEG single and repeated output are exactly equal. V3 preserves the exact
comparison using an identical-JPEG single-inference bridge; the bridge is not a
new original-Meta oracle. The failed v2 receipt and pre-inference v1 counter
correction remain archived. Official numerical/candidate gates were not relaxed.
Explicit F16 video remains diagnostic, including its historical entry23/24 failure;
full CPU F16 remains deferred.

The [historical record](20261002-182848-m2-complete-acceptance.md#historical-performance-record) retains previous measurements. Relative
to its same-protocol CPU video rows, observed speed ratios are 5.58×/5.88×.
The batches were not interleaved A/B runs, so small Metal differences need caution.
Follow [validation](../validation.md) to regenerate fixtures, qualify or explicitly
reuse unchanged original-Meta prefixes, and run only matching accepted binaries.

Raw receipts: `build/p1-p2/20261003/reaccept/`; sealed summary: `final-evidence.json`.
The [new baseline index](../validation-baselines/blas-vit-m4pro-20261003.json) binds
14664 files/43333663701 logical bytes in a private APFS clone bundle,
with unchanged original model/reference resources in the parent archive.
Byte preservation does not grant automatic acceptance to changed executables;
recorded paths/RPATHs may need restoration. Remote quick CI has not run. No new
commit or push was performed.



<a id="chinese-profiling-record"></a>

## 中文性能与诊断记录

迁自 `docs/benchmarks/m4-pro-blas-vit-20261003_zh.md`，原始文件 SHA-256：`121e5e6b90b065ac61782a27a0b2bdb7ca3ff0c709687bf144e8b832f6ccaa63`。测量值与历史结论保持原样。

### M4 Pro：ViT 投影与 CPU BLAS 测量

[English](20261003-134534-visual-encoding-profile-and-optimization.md#english-profiling-record) · [当前摘要](../../BENCHMARK_zh.md)

测量日期2026-10-03，Apple M4 Pro（14 CPU/20 GPU 核，48 GiB），macOS27.0
（26A428），Release AppleClang21.0.0，固定 GGML0.25.3
`353b63b439f27ab2cc19dac97ab1681ba6d2d084` 及已有精度/窗口补丁。
基于 main56a4cde 加未提交修改；收据绑定实际源文件、二进制、库和输入哈希。

提交前将 BLAS 计数放到公共统计结构末尾以保持旧初始化兼容，并补齐归档的
清单/大小检查。索引记录该元数据修正及 CPU/Metal 针对性对照；下表测量仍
绑定修正前的不可变二进制，没有改写旧收据。

原始 checkpoint SHA-256：
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。
视频使用 hybrid `visual-tracker-f32-v1`，GGUF SHA-256：
`3975b4b1a10b962c6fad5022e2798baa6266aad7dbcc39210b459537208e1a05`。
图像使用[模型列表](../../MODEL_ZOO_zh.md)中的 schema1 F32/F16。
权重标签不代表全流程精度；F16 归一化、BF16 特征舍入与记忆存储边界保持原样。

CPU 保有权重，在原生 CPU 前调度注册的 BLAS 加速器；`blas_nodes` 属于
`cpu_nodes` 的子集。ViT 共享通道投影将空间位置合并为 GEMM 列。
Metal 的选择/回退顺序不变，通过的运行中 CPU 图节点为零。
GGML/BLAS 收到四线程请求，进程启动前请求 `VECLIB_MAXIMUM_THREADS=4`；
Accelerate 独立管理 SGEMM 线程。

#### 图像：预热一次、完整推理五次

原始1800×1200 `truck.jpg`，提示词 `truck`，阈值0.5，模型输入1008×1008。
预热后时间不包含加载、解码、写文件和结果缓存复用；峰值 RSS 来自完整的
`/usr/bin/time -l` 子进程。时间单位秒，RSS 为十进制 GB。

| 权重存储 | CPU 中位时间 / 峰值 RSS | Metal 中位时间 / 峰值 RSS |
| --- | ---: | ---: |
| Mixed F16/F32 | 7.338 / 4.918 | 5.354 / 2.661 |
| F32 | 7.316 / 4.922 | 5.364 / 4.248 |

#### 视频：64 帧、16 帧预热、48 帧测量

单/四对象独立样本已取得原始 Meta 资格，1800×1200，提示词 `truck`，
max_objects8，无张量导出，包含最后排空。p95 按 `(n-1)*0.95` 线性插值。
峰值 RSS 包含加载和最后排空。

| 对象数 | CPU 中位时间 / p95 / 峰值 RSS | Metal 中位时间 / p95 / 峰值 RSS |
| ---: | ---: | ---: |
| 1 | 9.155 / 9.352 / 5.212 | 7.550 / 7.595 / 4.193 |
| 4 | 13.887 / 14.555 / 5.356 | 13.166 / 13.370 / 4.340 |

处理第14帧后首次输出第0帧。约每秒观察一次首个输出文件，包含加载、解码和
文件工作：CPU130.681/178.184秒，Metal103.536/155.065秒（1/4对象）。
时序规则保持原样。

##### 阶段中位数，秒

| 后端 | 对象数 | 图像编码 | 检测流程 | 跟踪 | 记忆 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpu | 1 | 6.448 | 1.105 | 1.480 | 0.075 |
| cpu | 4 | 6.481 | 1.109 | 5.932 | 0.309 |
| metal | 1 | 5.047 | 0.624 | 1.793 | 0.039 |
| metal | 4 | 5.043 | 0.624 | 7.277 | 0.161 |

图像编码包含 ViT、neck、几何编码、传输和分配；检测流程包含提示准备、融合、
检测和掩码。视频会话只编码一次文本；替换图像会重新编码，已包含在图像
`inference_ms` 中。各阶段中位数不能相加还原整帧中位数。
Metal 四对象目前以跟踪为最大阶段耗时。

预热后权重/计算缓冲上界恒定：CPU3447558112/1020660736字节；
Metal2763224992/1245268288字节。本协议每对象最多27条记忆记录，单/四对象
达到27/108条，即17943552/71774208字节。不能将分配计数相加作为进程 RSS。
每份收据保留带帧标记的异步 RSS 样本。有限观察不证明无限流内存上界或实时性能。

#### 条件、验收与保留的失败

视频四项在北京时间17:09:17–17:54:31串行运行，之后四项图像测量也串行执行。
全部电源端点为接电/100%，无温度或性能警告；实查电源日志，测量区间没有
系统睡眠。任务级 caffeinate 防止空闲睡眠，无其他 SAM/Meta 推理或编译重叠，
日常桌面服务仍运行，没有固定频率。结束后核对373个唯一标识和104个输出目录。

新验收全部通过：864帧视频（F32/hybrid × CPU/Metal，各216帧）、70项图像、
六项真实短会话和两组各128次交错推帧。长会话正样本的输出/历史与同后端新
entry64 基线逐字一致，负样本状态为空，模型生命周期、输出所有权和会话隔离
通过。长会话进程峰值 RSS：CPU5.555 GB，Metal4.595 GB；预热后权重和计算
缓冲上界恒定。这些仍是有限序列检查。

v2 图像补测曾错误地逐字比较 stb 解码 JPEG 与 Pillow 导出的参考 PPM。
6480000个 RGB 通道值中163934个不同（最大差3）；同一 JPEG 的单次与重复
推理输出完全一致。v3 保留逐字检查，使用同一 JPEG 的单次推理对照；该对照
不属于新的 Meta 原始参考。v2失败及v1推理前的计数修正记录均保留归档。
官方数值和候选门槛没有放宽。F16视频仍属诊断配置，保留历史 entry23/24失败；
完整 CPU F16 仍延后。

[历史记录](20261002-182848-m2-complete-acceptance.md#historical-performance-record)保留原测量；同协议 CPU 视频观察到5.58×/5.88×
提速。两批不是交错 A/B，Metal 的小幅差异须谨慎解释。
[验证工作流](../validation_zh.md)说明生成、资格验证/显式复用和匹配二进制要求。

原始收据：`build/p1-p2/20261003/reaccept/`；封存摘要：`final-evidence.json`。
[新基线索引](../validation-baselines/blas-vit-m4pro-20261003.json)绑定私有 APFS 克隆：
14664个文件/43333663701逻辑字节；原始模型/参考仍由父归档保存。
字节保存不能自动授予修改后二进制的通过状态；记录的路径/RPATH可能需要恢复。
远程 quick CI 尚未运行，本轮未提交或推送。
