# CUDA 执行优化与 benchmark 策略调整

创建：2026-10-07 04:27:06，Asia/Shanghai。状态：已完成实施、测试与验收。
基线：`0aa4971`，开始时工作区干净；RTX 4090、CUDA 13.3、本机原始 SAM 3
checkpoint、官方参考输出和上一轮已资格化 builds/receipts 可用。

## 范围与目标

落实用户关于存储精度与速度、原生 CPU benchmark、CUDA 深度优化的三点要求。
以保持现有 F32 数值与视频状态契约的优化为主要交付；低精度计算先作独立
可行性实验，再按用户确认的最终质量标准实施、验收。不把 F16 存储描述为
FP16 计算。公共模型图与后端策略保持清晰边界。

1. 将原生 CPU 的完整精度性能矩阵移至历史计划；主表保留 CPU/BLAS 和已验证
   GPU 后端。继续 CPU-only 正确性、边界与链接测试；CPU 路径变化时做代表性
   F32 回归，不为每项 CUDA 修改重跑十种原生 CPU 性能。
2. 对量化 Q8/Q4 的实际算子结构补充证据，区分 packed 权重、activation staging、
   F32 attention、传输与启动开销。
3. 优先隔离测试：视频 head-256 memory attention 分块/输出拼接；图像 head
   batching/直接输出；CUDA Graphs 与稳定图复用。只采纳稳定提速且通过原始
   模型验收的候选。
4. 根据热点评估 cuBLASLt 算法选择/融合、CuTe/CUTLASS 定制 attention 和真正
   低精度计算的可行性，记录适用条件、收益、数值与维护成本。已有库不能
   代替具体形状的测量；高成本候选先用小范围实验决定是否实施。

## 实施与实验步骤

- 冻结当前 baseline binary/library/source/model identities，实验均使用新的
  versioned 输出路径，不覆盖已完成 evidence。编译与 GPU 计时不重叠。
- 先用算子与代表性图像/视频负载筛选候选，再进行无 profiler A/B/A；报告
  cold、warm、重复次数、真实 loader、显存/RSS，避免将 profiler 时间当发布数据。
- CUDA Graphs 检查 graph key/lifetime、指针重绑定、dynamic shape、跨 session
  ownership 与 workspace 释放；不能仅切换 CMake flag 后宣称适用。
- attention 检查 tail、mask/all-masked rows、GQA、strides、batch、完整 key softmax
  分母、视频历史 memory/state 与 workspace peak。库/调度改动不得悄悄降低
  输入精度或放宽冻结的误差门限。
- 候选成功后更新 pinned patch/tree identities 与旧 quantizer receipt 兼容列表，
  clean Release 构建；生产实现与 prototype source 对照。
- 按最后实际变更运行 CUDA/CPU CTest、isolated `tools/test_tools.py`、官方原始
  图像七 cases 与量化质量、视频五 cases/216 帧、共享 session/长序列状态检查。
  只有受影响的已有支持组合需要重新资格化；未测平台明确记录。
- 更新双语 benchmark、backend options/precision 说明、Unreleased 与本计划结果；
  全库文档检查和 `git diff --check` 必须通过。

## 采用标准

代表性负载收益明显超过 baseline drift，并记录其他工作负载的回归与额外内存。
正确性与原始模型质量 gates 不变。失败或无收益候选保留实验记录并说明未采用
原因。原始模型输出作为验收来源，原生 CPU 是辅助对照。

## 用户确认的快速模式验收

用户选择“允许独立快速模式：最终掩码、检测和视频跟踪质量达标即可”。
快速模式应显式选择并报告独立 arithmetic profile，默认保持严格 F32。低精度
算子允许改变中间张量，仍记录误差；预处理、tokenizer、最终检测/掩码、对象
生命周期、视频 ID/输出延迟和状态边界必须满足冻结验收。

实验前采用既有 F16 输出门限（含 image mask IoU、score、box、threshold-adjacent
规则及 video F16 mask/score/box/ID/state）；不得根据结果放宽。量化模型叠加
快速计算时仍使用其原有量化输出质量门限。真正支持组合以逐项验收为准。

## 结果与后续

原生 CPU 十种存储格式的原始计时完整保留在 20261004 历史计划；双语主表已
移除该列并说明后续测试策略。56 个文档的双语表/本地链接检查通过。

初版 tracker prototype helper 将 `video.cpp` 误写为 `sam_video.cpp`，在查找
编译命令时退出，未编译或运行。修正为解析 Ninja token 后生成候选命令。


### 串行候选筛选

实验记录：`build/cuda-execution-experiments-v1/screening-v1/sequence.json`。
F32 图像 baseline 前/后为 592.509/594.903 ms，direct-output 581.633 ms，
8-head batching 528.359 ms，CUDA Graphs 594.467 ms。直接输出与 head batching
通过额外 9-head/tail/strided/masked double-golden 数值检查。

视频为 32 帧前缀，16 帧预热、16 帧计时，仅用于筛选。单对象 baseline
951.182/951.537 ms，direct tile 128/512/1024 为 941.928/854.946/858.063 ms；
四对象 baseline 1743.541/1777.859 ms，对应 1767.427/1449.024/1416.442 ms。
512 在单对象与 1024 相当，并将分数 workspace 减半，因此选为 CUDA backend
policy；CPU/Metal 保留原有 128 + concat 图。正式 64/16/48 计时结果见下文。
CUDA Graphs 单/四对象为 951.174/1764.401 ms，无可靠收益，不启用；因未采用，
不将基础 numerics 成功冒充完整 graph lifetime/模型资格化。

最初 fast prototype 把 head 32 也送入上游 fused attention，因 pinned GGML
不支持该 head size 而在分配前被拒绝。修正为 head 64 使用 fused reduced-input
路径、head 32 保持 F32。`screening-fast-v2` 的第一次计时已完成，但 loader
检查误读 rtk 父进程 maps，导致控制器失败，未用作正式性能。后续
`screening-fast-v3/sequence.json` 递归检查实际子进程 maps：F32/F16 图像
393.245/407.079 ms，单/四对象视频 677.643/1256.965 ms；七个原始 F32 图像
case 均通过预先声明的 F16 最终输出质量门限。该原型尚未合入 head batching
或 tracker tile policy，且不代表正式生产构建验收。

正式 patch 为 `2e0b508b9ca279d10fcce55f52b661b62fa862937345d7082db2843eb6cd3aae`，
combined tree 为 `fb582f54af023d94e0bf6754d802b8804a495cb67738afbbfde47432932ab4b8`。
旧 `e62f040cf989` quantizer build 被列入明确兼容列表，量化编码未改动。
第一次 clean configure 因实验 helper 生成的 tree identity 不正确而被 CMake
拒绝；helper 在仓库内调用 git apply 未设置 ceiling，未完整应用 Metal patch。
改为隔离 Git root 并使用生产 CMake hash 函数后匹配两次独立 configure 的实际
hash。原始 GGML checkout 保持 clean。失败 configure 记录保留在
`build/cuda-execution-delivery-v1`，后续构建使用独立 `v2` receipts。

首次 70 项工具回归发现新增测试修改 fixture tensor 后未同步其 SHA，校验器正确
拒绝；已改为通过 dump_array 同步 metadata，保留实际验收的 hash 检查。

### 边界修正与回归

首次生产 CUDA CTest 的新增低精度 masked attention case 在 B0/H0/Q128/D0
发现 `actual=NaN, expected=0`；该 query 为完全 mask 的行。上游融合实现不满足
这个边界。backend policy 已将 reduced fused 路径限制为**无 mask 的 head 64**，
所有 masked/head-32 attention 仍走精确路径。保持原来的 5e-6 masked numerical
门限，没有放宽或删除测试。CUDA patch 本身未因此改变，只有 device policy 更新。

重新构建记录：`build/cuda-execution-delivery-v2/masked-fallback-rebuild.json`。
后续 `build/cuda-execution-checks-v2/sequence.json` 的 CUDA 23/23 与 CPU 14/14
通过；70 项隔离工具检查及其后的 benchmark compute-profile 回归通过。
正式 CUDA 源码与选中 prototype 的差异仅为注释，校验记录为
`build/cuda-execution-delivery-v2/clean-source-audit-v1.json`。

head batching 的 C 输出是列跨度中的互不相交 head 段，每个 GEMM 独立写自己的
元素。实现依据 [NVIDIA cuBLAS StridedBatchedEx 合同](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex)，
并覆盖超过 8 heads、交错 head 布局、mask/batch 以及尾 query 的独立 double goldens。

正式验收在 `build/cuda-execution-acceptance-v2`（controller v3）开始前冻结了源文件、
二进制／库、模型与参考身份。早期 acceptance v1/v2 controller 只在等待阶段退出，
没有模型输出。所有 GPU 验收、session、计时、显存与 Nsight 阶段串行运行。

### 快速视频验收分层修正

首轮正式快速视频验收在 entry 与 hotstart-removal 中发现内部 mask/pointer
候选索引不同。例如 entry 第 23 帧，原始实现的候选 1/3 IoU 预测仅差
约 1.46e-5；低精度排序翻转，后续传播的中间分支也会变化。已完成场景的
最终 mask/score/box、固定出生 ID 映射、生命周期、memory/pointer 时间顺序、
输出延迟及状态边界均通过，失败项仅为候选索引等同检查。

原先将内部候选索引也列为快速模式硬门槛，超出了用户已选择的最终质量验收。
据此明确修正快速模式的验收分层：候选编号差异与中间张量一起作为诊断；新增
实际候选分数有限值、编号范围、mask/pointer 配对、按自身 IoU 选择最高分候选
以及传播对象 ID/顺序检查。默认 F32 仍要求与参考相同的候选编号。已有最终
输出质量阈值、视频 ID/生命周期、选择的历史 memory 帧和状态边界均保持。

首轮失败 receipts 将原样保留；更新 validator 后使用新路径重新运行快速视频。
已通过且执行/验证代码未改变的图像与默认视频 receipts 可按原 SHA 引用，禁止
改写旧输出或把旧失败记录标成通过。该分层修正及其测试、重验收结果在下文记录。

### 完整模型资格化

首轮 `build/cuda-execution-acceptance-v2/sequence.json` 已完整结束，20 个图像
组合、各七 cases 均通过，共 140/140。默认 F32 计算的两种视频权重也通过各
五个 cases/216 帧；快速视频的失败只包含上述候选编号检查，原记录保持失败。

修正 validator 后，70 项工具检查通过，新增覆盖候选排序、错误 pointer 配对、
越界编号、非有限值、重复传播对象及默认模式继续拒绝候选差异；记录在
`build/cuda-execution-validation-v2/tool-tests-v1.log`。旧 validator/test 源文件
另存 `build/cuda-execution-validation-original-v1`，其 SHA 与首轮冻结值一致。

最终 `build/cuda-execution-acceptance-v3/sequence.json` 全部通过。两个快速视频
配置均以新进程重新跑完 216 帧；其余 22 个已通过 receipts 按原 SHA 引用，未
重评分或修改。C++ 二进制、GGML 库、权重和图像 validator 均未改变；默认视频
的验收逻辑和门限保持，工具回归覆盖其严格候选检查。最终覆盖 20 个图像组合
及四个视频组合，共 140 个图像 cases、864 帧视频。

F32/F16 存储的快速图像最低高置信 mask IoU 均为 0.9995579134，最大分数误差
分别为 0.0006654267 / 0.0006583927。量化组合仍采用各自既有质量门限，不能用
这组 dense 数字代表 Q4 等量化结果。每个组合的逐 case 指标保存在对应 receipt。

两种快速视频各比较 220 个对象结果，最低 mask IoU 均为 0.9962032069；最大
score 误差为 0.0010871885 / 0.0010839105，最大 box 维度比例误差为
0.0008333508。每种快速计算权重配置各有 12 帧内部候选编号差异，所有实际候选选择规则及
最终质量/ID/状态门限均通过。

`build/cuda-execution-sessions-v2/sequence.json` 的 12 项共享 session 检查全部
通过，覆盖两种计算模式的浮点图像缓存、短视频 reset/error/结果所有权，以及
F32/hybrid 视频各两条交错 64 帧序列。四个长序列测试共 512 次 push，全部
保持 session 隔离、结果所有权、GPU placement 和状态边界。

### 性能凭据更新

`build/cuda-execution-performance-v2` 的 20 项图像计时均有效；八项视频均在
启动前因旧 qualification 中 `tools/benchmark_video.py` 指纹不符而拒绝，未
产生视频计时结果。保持旧凭据不变，在 `build/cuda-video-benchmark-qualification-v4`
重新执行两个原始 Meta 模型 17 帧前缀资格化（初始化长度仍为 64），全部通过。
输入 PNG、原始权重、oracle 规则、对象/连续性/延迟验收不变。

正式性能集合改为 `build/cuda-execution-performance-v3`：按 SHA 引用 20 个
原图像计时，使用新资格化记录实际运行八个视频 cell。独立图像显存、Nsight
及最终审计也采用新的 versioned 路径，未重新标记旧失败记录。

### 最终实现与发布性能

默认模式采用 cuBLAS 分批计算最多八个 attention heads，直接写入互不重叠的
输出区域；tracker 使用 CUDA 专属的 512-query tile 与原位输出组装。分数
scratch 的 head 批次预算为 32 MiB，单 head 超过预算时仍至少处理一个 head。
默认 F32 数值合同不变，CPU/Metal 保持其原有 attention 策略。

新增显式 `--backend cuda --cuda-compute f16`；C++ 对应
`BackendOptions::cuda_compute = CudaComputeMode::F16`。dense 矩阵采用 F16
输入、F32 累加和输出；无 mask 的 head-64 attention 使用融合低精度 kernel。
masked/head-32 attention 保留精确路径，量化矩阵保留 packed kernels。
浮点与量化快速模式分别报告 `ggml-cuda-f16-v1`、`ggml-quantized-cuda-f16-v1`，
验证器和视频性能凭据绑定该计算模式，不能把存储格式用作计算精度标识。

以下均为 RTX 4090、相同 F32 权重与原有负载的无 profiler 中位数。
基线为 `0aa4971` 的上一轮正式凭据。图像取五次完整非缓存调用，视频取
64 帧中的后 48 帧；不含模型加载、外部解码和文件输出。百分比为延迟降幅。

| 负载 | 基线，ms | 本轮默认 F32，ms | 本轮快速 F16，ms | 默认降幅 | 快速降幅 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 图像 | 589.913 | 524.384 | 360.880 | 11.11% | 38.82% |
| 单对象视频，每帧 | 932.494 | 798.549 | 632.100 | 14.36% | 32.21% |
| 四对象视频，每帧 | 1751.461 | 1364.295 | 1164.533 | 22.11% | 33.51% |

全部 28 项正式性能测量完成：20 项图像、八项视频。完整双语表已更新到
[BENCHMARK](../../BENCHMARK.md)及[中文性能表](../../BENCHMARK_zh.md)，
逐项基线对比保存为 `build/cuda-execution-comparison-v1.json`。混合存储的
视频快速模式单／四对象为 633.638 / 1158.776 ms，每帧延迟分别下降
32.17% / 34.01%。全模块 Q8_0 / Q4_K 快速图像为 347.440 / 346.746 ms，
差异很小，不支持按位宽推算成倍加速。

20 项图像显存使用独立进程约 50 ms 采样，不把这些进程的耗时混入上述表格。
快速模式相比默认模式增加约 0.35–0.41 GB 采样显存峰值：F32 图像
4.777→5.184 GB，全模块 Q8_0 为 2.563→2.911 GB，Q4_K 为
2.252→2.600 GB。F32 视频为 4.914→5.320 GB，hybrid 为
4.228→4.635 GB。数值包含进程 context、代码和 pools，是采样下界；
没有把增长归因于未经单独测量的某个 workspace。

### 算子分析与精度差异解释

六组 Nsight Systems 分析各取五个 warm 图像区间，区间起点为输入 H2D、
终点为结果 D2H；冷启动排除。下表为有 profiler 时的 GPU 区间与 kernel
累计时间，**不能替代上方发布延迟，也不能把 CPU API 时间与 GPU 时间相加**。
原始 traces 及统计保存在 `build/cuda-execution-profiling-v3`，
汇总为其中的 `operator-analysis-v1.json`。

| 权重／计算 | 每请求 kernel 数 | kernel 累计中位数，ms | GPU 区间中位数，ms |
| --- | ---: | ---: | ---: |
| F32 / F32 | 16112 | 305.745 | 440.218 |
| 全模块 Q8_0 / F32 | 16718 | 199.339 | 340.173 |
| 全模块 Q4_K / F32 | 16718 | 199.925 | 337.735 |
| F32 / F16 | 14040 | 151.192 | 285.504 |
| 全模块 Q8_0 / F16 | 13924 | 133.382 | 269.230 |
| 全模块 Q4_K / F16 | 13924 | 134.070 | 269.757 |

权重存储与运算策略是两个维度。默认浮点 dense/attention 保持 F32 计算，
F16 存储主要降低常驻权重，不能据此预期 Tensor Core 的低精度吞吐。
快速模式的 trace 已出现 `ampere_s16816gemm_fp16_128x64_ldg8_tn` 等
FP16 Tensor Core kernel 及 `flash_attn_ext_f16`。F32 权重的库矩阵计算及
归约平均累计时间由 203.373 降至 42.956 ms，精确 attention softmax 由
29.315 降至 9.239 ms，另有融合 attention 7.367 ms；布局／转换开销则由
35.323 增至 51.899 ms，说明进一步收益需要控制转换和数据移动。

Q8_0 和 Q4_K 在同一计算模式下拥有相同的 kernel 数，packed matmul 平均
累计时间均约 23 ms，RHS Q8_1 staging 约 5–6 ms。默认模式还共有约
71 ms 的库矩阵计算／归约、29 ms 的精确 softmax、35 ms 的布局／转换及
35 ms 的逐元素／归一化／索引操作。快速模式对应约 12、9、40、36 ms，
另有约 7.4 ms 融合 attention。这些共同成本使 Q4 缩小文件与权重常驻显存
的收益不能直接转化为同等比例的完整请求加速。

六种配置每请求都包含 H2D 156,833,656 bytes／49 次，D2H 183,383,972
bytes／11 次，D2D 656,672,736 bytes／108 次；H2D/D2H 的 GPU copy 时间
分别约 13.7–14.2 / 17.4–18.1 ms，并有 154 次 stream synchronize。
这些数值表明仍有传输和调度成本；Nsight Compute 硬件计数器因
`ERR_NVGPUCTRPERM` 不可用，不能据此声称已证明某算子受显存带宽或占用率限制。

后续优化按此次证据排序，作为独立工作项逐项测量和验收：

1. 合并 GGML `src/ggml-cuda/concat.cu` 连续布局分支按第四维逐次启动的
   kernel。每请求 `concat_cont` 仍达 8260 次，累计 GPU 时间约 11 ms，
   但启动次数占很大比例；SAM 3 视觉 RoPE 的实／虚部拼接反复使用该路径。
   优先评估一次 launch 覆盖多个 plane，保留 tail、stride 与维度边界。
2. 减少阶段间 host/device 往返及重复转换，让可复用中间结果保留在 GPU。
   先区分公共 API 必需的结果下载与内部中间量，再验证多 session 的所有权、
   缓存及峰值内存，不能仅依据 copy 总量承诺收益。
3. 对 head-256 tracker memory attention 评估 CuTe/CUTLASS 融合 kernel，
   重点覆盖 mask、完整 key softmax、动态 memory、GQA 和长序列状态。
   cuBLAS 已用于现有矩阵运算，本轮进一步采用批量调用和低精度输入；
   cuBLASLt 算法／epilogue 调优应针对剩余具体形状开展，暂不加入没有
   实测收益的新库依赖。CuTe 的实现能力不等同于可直接替换的 SAM 算子。

CUDA Graphs 已经完成代表性筛选且没有稳定收益，保持关闭。本轮采用的改动
均已完成正式验收；上述后续项不属于尚未完成的交付承诺。

### 最终证据与适用范围

最终审计 `build/cuda-execution-delivery-audit-v3.json` 通过 2950 项一致性
检查，核对 2675 个唯一产物 hash；这些是来源与产物一致性检查，不是新增
单元测试数量。它绑定了实际加载的 CUDA 库、C++ 二进制、源文件、权重、
原始 checkpoint、参考输出及每项质量／性能记录，并验证失败记录保留和
22 项已通过验收凭据的原样复用。证据保存在以下本地忽略路径，避免为不随
Git 分发的产物创建会在全新 checkout 失效的文档链接：

| 证据 | 本地路径 | 结果 |
| --- | --- | --- |
| CUDA / CPU 构建与源码核对 | `build/cuda-execution-delivery-v2` | clean Release 与最终 policy 重建通过 |
| CUDA / 原生 CPU CTest | `build/cuda-execution-checks-v2/sequence.json` | 23/23、14/14 |
| 隔离 Python 工具回归 | `build/cuda-execution-validation-v2/tool-tests-v1.log` | 70/70 |
| 原始模型资格化 | `build/cuda-execution-acceptance-v3/sequence.json` | 140 图像 cases、864 视频帧 |
| 共享 session | `build/cuda-execution-sessions-v2/sequence.json` | 12 项；长序列共 512 次 push |
| 视频性能 fixture 资格化 | `build/cuda-video-benchmark-qualification-v4/sequence.json` | 两个原始 Meta 17 帧前缀 |
| 无 profiler 性能 | `build/cuda-execution-performance-v3/sequence.json` | 28 项 |
| 独立图像显存 | `build/cuda-execution-memory-v3/sequence.json` | 20 项 |
| Nsight Systems | `build/cuda-execution-profiling-v3/sequence.json` | 六组，各五个 warm 区间 |

本轮 CUDA 硬件资格化覆盖 RTX 4090；这是测量硬件，不是仓库支持平台的
边界。原生 CPU 完成构建和行为回归，Metal 本轮没有硬件重验，原有 Metal
补丁与默认执行策略保持。F16 快速模式按最终质量验收，中间张量与内部候选
编号差异作为诊断；默认模式继续使用原有数值和候选一致性门限。

最终 `.venv-reference/bin/python tools/check_docs.py` 通过全部 56 个文档的
双语数据表与本地链接检查；`git diff --check` 通过。既有历史产物缺失不再
阻塞文档检查，历史数据路径与身份保留在计划中。此次修改留在工作区，未提交。
