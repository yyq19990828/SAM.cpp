# Tracker 加速与跨后端适用性计划

创建时间：2026-10-03 19:18:14，Asia/Shanghai。
原始状态：规划完成；当时只新增本文件，没有实施、构建或运行性能测试。

后续状态：用户于 2026-10-04 明确要求先提交现有改动，再实施本计划并统一测量。
已固定干净提交 `ae1b7af` 为执行基线，当前批次见
[pipeline 实施与统一测量计划](20261004-040240-sam3-pipeline-execution-and-unified-benchmarks.md)。
下文“本次只授权写计划”的表述保留为原始规划记录，不限制新的实施授权。
交付状态：前三项共用执行优化已实现，最终原始模型回归与选定的统一测量已完成。
D256 attention 融合留待独立精度内核工作；MPS 完成对照后未采用。当前结果与范围以
上述实施计划的最终记录为准。
分析基线：现有 `37206da` CPU BLAS/ViT 优化及其版本化验收记录。当前工作区正在实施另一份量化计划；这里的执行基线必须在开始实施时重新固定源码、模型、构建和后端库身份。

## 目标与范围

优化 SAM 3 图像/视频共用执行开销，重点降低多对象视频 tracker 的时间和数据搬运成本。保持 header-only SAM 集成层、现有公共任务 API、共享 GGML 图及 F32/F16/hybrid 精度与时序语义。

评估五种手段：图和工作区复用、特征驻留与减少复制、多对象批处理、memory attention 融合、Apple MPS 矩阵内核对照。推荐先交付前三种共用执行优化；attention 内核和 MPS 单独裁决，任何一项未获得收益都不阻塞其他已验收项。

当前实施和验收平台为 Apple CPU/BLAS、Metal。其他 GPU/NPU 的分析用于确定可迁移边界，不增加后端支持声明。量化存储/低精度算术遵循另一份[量化计划](20261003-184759-gguf-quantized-inference.md)，不能与本计划混合变更后宣称单项收益。视频模型、输入尺寸、hotstart/reconditioning/memory-selection 策略和检测阈值不变。

本次只被授权写计划文件；此文件不启动主线程的任务，不控制其子代理，也不改动其实现、gates、Git 状态或正在运行的进程。

## 五种手段是否对所有硬件后端有效

结论是“部分策略可迁移，收益和具体实现不通用”。需要区分：数学/执行策略是否适用、后端是否有可用实现、实际设备是否加速。前四项都不能保证每个硬件受益；第五项的 MPS 实现只适用于 Apple 平台。

| 手段 | CPU / BLAS | Metal | 其他独立 GPU | NPU / 固定图加速器 | 可迁移层与主要限制 |
| --- | --- | --- | --- | --- | --- |
| 1. 图与工作区复用 | 可减少构图、分配；若计算占绝大多数，收益小 | 可减少主机调度与资源重建 | 同类策略通常适用 | 取决于编译图/shape cache API | 共用执行生命周期；动态 shape、失效规则和内存上限必须正确 |
| 2. 特征驻留、减少复制 | 可以减少 host buffer 的复制；没有独立 GPU 搬运可省 | 可减少 CPU/GPU 复制、同步和重复 upload | 主机与设备之间的搬运成本可能更大 | 需大部分算子可在设备内连续执行 | 共用张量生命周期；内存分配、同步和设备归属由 backend 处理 |
| 3. 多对象 batch | 大 GEMM 可能更适合 BLAS，也可能受 cache/线程/带宽限制 | 可增加 GPU 工作粒度并减少提交 | 常见候选，但取决于占用率和工作区 | 必须支持相应 batch 和 shape | 模型图组织可以共用；对象状态隔离、延迟和峰值内存约束统一 |
| 4. Attention 融合 | 能少写中间结果，但未必胜过既有 BLAS | 可减少 kernel dispatch 与中间结果读写 | 原理可迁移，kernel 要按设备实现 | 取决于支持的算子、head、精度和片上内存 | softmax 语义共用，tile/向量化/内核选择后端专用 |
| 5. MPS 矩阵内核 | MPS 不是 CPU BLAS；CPU 保持现有 BLAS | Apple 专用候选，需要实测 | 不能使用 MPS；只能另评估对应平台的矩阵库 | 不能把 MPS 当通用 NPU 接口 | 仅矩阵库替换这个思路可迁移，API/布局/精度/调度并不通用 |

表中“可”表示从当前代码路径推导的候选机会，尚未证明性能收益。Apple 统一内存也不意味着现有 vector/tensor upload/download 没有 memcpy、格式转换和同步成本；CPU 侧则应称为减少复制，不能宣传为 GPU 传输加速。batch 会改变执行形状，融合会改变归约顺序，不能只凭数学等价免除数值验收。

## 已核对的现状

[M4 Pro 基准](20261003-134534-visual-encoding-profile-and-optimization.md#english-profiling-record)记录 hybrid 四对象视频：CPU 13.887 s/frame、Metal 13.166 s/frame；tracker stage median 分别为 5.932/7.277 s，图像编码为 6.481/5.043 s。Metal tracker 是当前四对象 workload 的最大阶段。stage medians 不能相加重建整帧 median，也不能据此断定 tracker 内部哪个 kernel 最慢。

源码证据：

- `tracking/session.hpp` 逐 group/object 调用 `condition` 和 `decode`，然后保持原有 group quality、association 和生命周期处理。
- `tracking/execution.hpp` 对 temporal pointer、conditioning、decoder、memory encoder 分别新建 `GraphExecution` 并 allocate/compute。
- conditioning 输出下载为 host vector，decoder 又上传；相同 frame 的 tracker-neck features、position、RoPE 在不同对象中重复上传。最终 host mask/association 和公开诊断仍需要主机可读结果。
- `tracking/attention.hpp` 对 256 维单 head 按 128 个 query 分块，每个 tile 依次执行完整 key 的 matmul → scale/softmax → matmul，再 concat。当前实现保持完整 softmax 分母，不能简单拆 key 后拼接。
- `graph.hpp` 的 context/scheduler 随 `GraphExecution` 生命周期销毁。固定 GGML 已有 scheduler reserve/reset API；应先复用这些接口，不写另一套 allocator。
- 现有 Metal 精度补丁覆盖已验证的 dense 和 head-32/64 attention；256 维 tracker attention 是独立的 F32 tiled 图，不能假定切换到现有 flash 调用就有同一精度保证。

先前 [ViT 优化记录](20261003-134534-visual-encoding-profile-and-optimization.md)中，Q/K RoPE batching 和关闭 Metal concurrency 都没有改善匹配 A/B/A 编码耗时。它们不能作为已有加速成果，也不优先重复。ViT 投影折叠已经完成，不在这里重新计算一次收益。

## 选定的工程边界

最小交付：只复用重复形状的 tracker 图与 GGML 工作区，并保留完全相同的算子和 host 输出。第一阶段可独立验收、交付、撤回；即使随后优化不做，系统仍完整可用。

后续特征驻留和对象 batch 通过既有 GGML tensor/graph 表达，不增加公开 tensor API、通用 kernel plugin registry、全局 cache 或可变全局状态。SAM 模型层负责 shape、对象分组与精度边界；GGML runtime 层负责资源、调度与同步。独立 session 的缓存不得互相覆盖；保留现有模型执行锁和 session 非并发调用契约。

```text
VideoSession / birth groups / original temporal policy
                   |
TrackerExecution: shape keys, object batches, feature lifetime
                   |
Shared GGML graph: condition -> decode -> required host outputs
                   |
Session-owned workspace / GGML scheduler / buffer ownership
                   |
CPU + optional BLAS                 Metal
                                   |
                          optional measured fused kernel
```

预计整体涉及超过 8 个文件：`tracking/{execution.hpp,session.hpp,attention.hpp,memory_attention.hpp,mask_decoder.hpp}`、模型 `execution.hpp/state.hpp`、GGML `graph.hpp/resources.hpp`、必要的 backend 模块，以及测试和 benchmark/validation 文档。按阶段控制实际修改范围；不会为这份计划先创建空模块。

最脆弱的假设是“tracker 中重复构图、搬运和小工作粒度的成本足以影响端到端时间”。若实测主要受单个大 GEMM/attention 计算限制，图缓存的收益可能不足；保留较小的已验证改动，将后续投入转向对应 kernel，不能据阶段总时间承诺倍数加速。

## 独立交付步骤

### 1. 复用图与工作区

预计 2–3 个工程日加验收时间。

- 在 `TrackerExecution` 内为 temporal pointer projection、conditioning、decoder、memory encoder 分别保留最近一种 shape 的图元数据。key 包含实际 backend/驻留 dtype、完整尺寸与 strides、spatial/pointer count、seed/no-seed 分支及输出/诊断模式。不同对象的数据不是 key，不能缓存上一次的结果。
- 使用已有 GGML reserve/reset 和 scheduler 管理共用工作区。每类图最多一个活跃 shape；换 shape 时在工作完成后失效。复用主要保留 graph/context 元数据，避免四套长期大 buffer 叠加。不得用无上限的 shape map 换取“零构图”。
- 完整覆盖对象数/记忆长度改变、空 pointers、首帧 seed、warmup/drain/reset 和异常后的清理。cached tensor 不能在 context/buffer 已失效后继续使用；新输入每次都要更新。
- 增加诊断计数与计时，至少区分 graph-build、reserve/allocate、upload、backend compute、download/wait；async backend 的 GPU 时间和 CPU submit 时间分别记录，不重复相加为总延迟。计数在内部/诊断报告中提供，首版不新增公共配置开关。

完成条件：同模型输出和时序行为通过原有验收；稳定 shape 的重复调用降低构图次数；shape 切换和跨 session 正确；warmup 后峰值 workspace 不持续增长。性能候选比较必须包含实际构图时间，不能只比较不变的 GPU 算子时间。

### 2. 驻留 frame 特征并串接 condition/decoder

预计 3–5 个工程日加验收时间；阶段 1 不依赖本阶段。

- 将每个 frame 的共享 tracker-neck features、position、RoPE 绑定到有明确 ownership 的 backend buffer，同 frame 各对象复用。CPU 同样减少 vector/tensor 复制，但不改变 CPU/BLAS 调度顺序。
- 先从原始分阶段图构建出可串接的 condition → decoder 图，让 conditioned tensor 直接成为 decoder 输入。最终 mask、IoU、score、pointer 和要求保留的诊断 tensor 在执行完成后读取，取消“读取 conditioned 到 host 后立即再上传”的中间依赖。
- 保持 F16 预处理、BF16 tracker-feature 舍入、BF16 memory record 和 F32 pointer。第一版仍保留原有 host BF16 转换及 memory selection；只是转换后的 frame 数据上传一次。后续如将舍入移到 GPU，必须另外证明 bit/边界行为，不能随融合省略转换。
- group/frame 无关的 pointer projection 输入相同时允许计算一次并复用结果；对象的 memory/conditioned 数据仍独立。帧被替换、session reset、模型析构、异常处理时按现有生命周期释放。
- 公开结果和诊断继续具有原有所有权与可读取时机。需要下载的诊断可以与其他输出在末尾读取，不能借用下一次 workspace 覆盖的内存。不得为减少传输同时保留无上限的 CPU/GPU 双份历史。

完成条件：host upload/download bytes 与同步点的变化有真实计数；模型质量、diagnostic values、ID/association/temporal history 不变；同模型 CPU 和 Metal 分别验收。离散设备上的驻留方案是后续可迁移设计，当前不声称在其他 GPU 已验证。

### 3. 批处理兼容对象

预计 3–5 个工程日加验收时间；阶段 1/2 可单独使用。

- 第一版只批处理已存在对象的 propagation，按相同 spatial/pointer selection 形状、seed 模式和 tensor dtype 组织 batch；birth/reconditioning 的 seed 路径先保持当前串行实现。对象按原有稳定顺序映射输出，group-quality broadcasting 和后续 association 保持原逻辑。
- 借用现有 tensor batch 维度及 shared weights，调整 tracker 图中只接受 batch=1 的 reshape/view/attention/output 拆分。每个 batch 元素有自己的 memory 和 softmax，禁止把不同对象拼到同一 attention token 轴让它们互相注意。
- 第一版不 padding 不同记忆长度，不等待额外帧凑 batch。batch 上限为现有 `VideoOptions::max_objects`；1 对象或不兼容 shape 使用原路径。当所需工作区超过同 workload 基线预算时确定性拆分 batch，记录实际分组，不放宽对象或历史容量。
- 用 1、2、4、8 对象比较逐对象与 batch 的中间 tensor、mask/score/box、对象映射、异常恢复和 output ownership。量化模型若纳入后续对照，单独使用其冻结 gates，不能重新解释 F32/hybrid 标准。

完成条件：四对象测试实际减少重复调用/提交，首个结果不因等待 batch 延后，峰值内存可控；CPU 原生、CPU/BLAS 与 Metal 均无超出噪声范围的退化。某 backend 变慢时只在该 backend 保留原执行选择，不影响其他路径；调度选择放在 runtime/backend policy，模型不直接按设备名称分支。

### 4. 融合 256 维 memory attention

预计 5–10 个工程日加 kernel/模型验收时间，属于独立的后端优化。

- 在当前量化/共用图工作稳定后，检查实际采用的 GGML 版本是否已有支持目标 shape、完整 masking 和 F32 契约的内核；满足条件就复用。否则只为 Metal 的 tracker 256 维单 head 编写最小融合实现，保留共享 tiled 图作为受支持的参考路径。
- 使用分块 Q/K/V 和 online-softmax 的全局 max/normalizer 更新，跨 key tile 正确合并输出。保持 F32 输入、softmax 和输出，遵守本项目的精度要求；通过减小 tile 控制 threadgroup memory，不把现有 head-64 kernel 参数简单放大。
- 必须覆盖 `N_query=5184`、不同 spatial/pointer key 数、128-query 边界、尾部、重复/极端 logits 与零值，以及 mask/position/RoPE 的既有语义。与独立完整-key softmax 和原 tiled 图比较。浮点归约顺序差异由原有数值门槛裁决，不能宣传 bitwise 等价或改变 candidate 门槛。
- 修改放在 GGML 后端实现与固定 patch 中；记录 source revision、patch hash、license、生成源码 hash 和硬件限制，更新 CMake 验证。SAM core 仍 header-only。其他 backend 继续用原图，没有对应设备实测就不扩展支持。

完成条件：attention microbenchmark 和完整五 case/216 帧均通过；没有 CPU fallback；端到端 tracker/video 有收益，显存/统一内存峰值不恶化到抵消收益。论文里的其他硬件加速倍数不能套用到 M4 Pro。

### 5. Apple MPS 矩阵内核对照

预计 1–2 个工程日完成独立对照报告；本计划默认保持现有 GGML Metal 矩阵路径，MPS 生产接入不是前四项的依赖。

- 在私有诊断程序中复用同一 Metal device，比较 GGML 与 `MPSMatrixMultiplication`。使用真实 tracker shape：F32 `[256,256]` 投影 × `[256,5184]` 特征，memory attention 的实际 Q/K/V 尺寸，以及小 token 的 decoder 投影；同时覆盖 transposed/batched/tail 输入，不只测理想大方阵。
- 固定同一 payload、F32 语义及结果对照，将 command encoding、layout conversion、buffer copy、同步和工作区全部计入，并单列纯 GPU 时间。只有纯 kernel 更快而总耗时更慢，结论仍为“不接入”。
- 对照报告明确 MPS 可用 OS/device、实际输入/累加/输出行为、row stride、ownership、同步和调用限制。MPS 只能处理可支持的矩阵，不能假定替代整个 GGML graph 或直接支持压缩权重。
- 生产采用的门槛是代表 shape 的匹配 A/B/A 优势在重复批次均稳定，且推导到实际 graph 的复制/布局成本后仍成立。若达到门槛，再形成只覆盖已验证 shape 的 GGML Metal bridge 实施计划，明确 `.mm` 编译依赖与 patch provenance；不新增 SAM 公共 MPS backend 或默认路由。

这个阶段的交付物就是可复核的对照报告及采用/不采用结论。没有实际结果前不以 Apple 官方库的名称保证更快。CPU 现有 BLAS、其他设备的矩阵库和 NPU 编译图各自保持独立评估。

## 统一验收协议

每个阶段在执行前固定独立源码/build/model/library/input/gates identities，保留上一阶段可运行版本；不与主线程的量化构建、模型转换或推理同时进行基准测量。对一个阶段的候选做小范围正确性和匹配计时，通过后再跑受影响的完整数值/session matrix。失败及未采用候选保留原始记录。

主要行为检查：cache key/输入更新、shape 变化、对象进入/遮挡/消失/重现、固定 birth ID 映射、memory selection、group quality、15-frame 输出与最终 drain、session reset/lifetime、跨 session 隔离、错误后状态，以及借用/拥有 buffer 的生命周期。保留 Release 活跃 checks、独立头文件和 two-TU linkage。图像 API 若共享 runtime 被修改，必须补跑受影响的图像/缓存回归。

质量标准沿用各个已验收 profile 的原始 gates：F32、F16、hybrid 分开；不因 cache、batch 或 kernel 融合放宽。如果后续使用量化模型，则读取对应版本化量化 gates 并另记结论。BF16/F16 传输/舍入边界不变；原有 video exact candidate/时序检查继续执行。

性能保持当前协议：原始 truck 图像 1 warmup + 5 次完整图像调用；视频 one/four-object 64 帧，16 warmup + 48 measured，最终 drain 包含在内。串行 AC 测量，无重叠编译或推理，记录 power/sleep、threads、CPU BLAS 环境请求、backend placement、median/p95、首个输出时间、峰值 RSS 和 weight/compute workspace。

推荐采用标准：五次匹配的短 workload A/B/A 对照中，至少四次方向一致、median 改善至少 5%，随后完整协议确认；若标准帧间噪声大于 5%，提高可判别门槛或延长独立批次，不强行判定收益。p95 和首个输出不出现超过匹配噪声的退化，峰值 RSS/workspace 不高于同模型同 workload 基线。内存明显增加的候选先缩小缓存/batch，不能只报告吞吐。

单对象或 CPU 没有收益并不等于优化无效；对有收益的设备/对象数采用确定性执行选择，其他路径保留原实现并说明条件。硬件支持、质量通过、性能受益是三个独立结论。

实施后最低检查接口使用新的目录；本次不执行：

```sh
rtk proxy cmake -S . -B build/tracker-accel-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
rtk proxy cmake --build build/tracker-accel-cpu --parallel 2
rtk proxy ctest --test-dir build/tracker-accel-cpu --output-on-failure
rtk proxy cmake -S . -B build/tracker-accel-metal -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
rtk proxy cmake --build build/tracker-accel-metal --parallel 2
rtk proxy ctest --test-dir build/tracker-accel-metal --output-on-failure
rtk proxy build/reference-runtime/venv/bin/python tools/test_tools.py
rtk proxy python3 tools/check_docs.py
rtk git diff --check
```

真实模型、portable fixture、original Meta qualification、validator、long-session 与 benchmark 命令遵循[现有复现流程](../validation.md)，在新目录记录本阶段 receipts。新的 backend-specific 内核在对应硬件上验收；本机 CPU/Metal 结果不证明其他设备支持或受益。

无新增账户、API key、外部服务或用户安装要求。权重和私有媒体沿用现有受验证资源，不写入 Git。每一阶段通过撤回私有执行路径回到旧图；不改 checkpoint/GGUF schema/temporal data，可不迁移数据回滚。只有完成的优化写入 `changelog.md` Unreleased 和同步中英文性能/支持文档；本次不提交、推送或改远端 CI。

## 依据与本次结果

- [GGML 固定版本 scheduler 接口](https://github.com/ggml-org/ggml/blob/353b63b439f27ab2cc19dac97ab1681ba6d2d084/include/ggml-backend.h)：复用 reserve/reset/alloc，而非另建 allocator。
- [Apple Persistent Objects](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/PersistentObjects.html)：pipeline、buffer 和资源生命周期复用。
- [FlashAttention 原始论文](https://arxiv.org/abs/2205.14135)：完整 attention 的 IO-aware tiling 原理；不借用其其他硬件的速度数据。
- [MPSMatrixMultiplication 官方接口](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)：Apple 矩阵运算候选，是否适合此处仍待实际对照。

已核对当前源码路径、CPU/Metal 性能与精度记录、GGML scheduler API、资源复用及 attention/MPS 上游资料。五种手段的跨硬件边界、独立步骤、代价、风险和验收条件已写清。任何速度/内存收益尚未实测。

本次检查通过：`tools/check_docs.py` 检查了 39 份文档的本地链接与既有双语表格；新计划另行检查五种手段、local links、code fence、末尾换行和空白；`git diff --check` 通过。本次只新增此文件，没有改动主线程实现、启动模型工作或提交/推送。
