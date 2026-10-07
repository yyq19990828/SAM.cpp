# 激活、注意力与运行时状态量化规划

创建：2026-10-07 14:30:03，Asia/Shanghai。状态：首轮实现、验收与候选取舍已完成；后续研究范围见文末。

最新进展：[真实 typed cache 运行时实验](20261007-181154-typed-image-cache-runtime.md)
已完成 128 张 selection 和冻结后的 512 张独立 evaluation。F32 权重/计算的
混合 Q8_0 缓存通过严格质量门槛，完整推理中位数降低 12.0%、主机峰值 RSS
降低 15.3%，GPU 采样峰值却增加约 0.13%，因此保留内部实验、不增加公共缓存
选项。已交付的 CUDA F16 紧凑卷积存储将三种既有权重配置的 GPU 峰值降低
17.7%–26.3%；W8A8/FP8 当前候选按数值或整图门槛退出，后续研究不记作支持。

## 目标与范围

将用户所说的“减少推理市场”理解为减少推理时长，同时降低峰值显存。当前以
SAM 3 图像分割为第一阶段，视频跟踪为独立的后续验收阶段；先在已有验证环境
RTX 4090 / CUDA 上落地，再分别验证 Metal 和 CPU。该硬件只是首个测量平台，
不是库的支持边界。保持 C++17 header-only 接口、GGML 编译依赖和模型／后端分层。

建议主线是：**激活与工作区盘点 → 真正的 A16 存储 → 校准 W8A8 与 CUDA FP8
对照 → 量化融合注意力 → 视频状态压缩 → 更低位宽和重建**。每一步独立交付，
以最终分割／跟踪质量、完整推理延迟和实际峰值内存决定是否保留。A16 是低精度
基础设施，不等同于整数激活量化；把它放在前面是为了避免后续 A8 内核仍被 F32
中间张量限制。

初始规划轮只交付设计文档；用户随后要求实施，实际改动和验收见文末分阶段记录。
现有 F32/F16 与 Q8_0/Q6_K/Q5_K/Q4_K 的文件格式和默认算术保持其既定契约。

## 已核对的仓库现状

代码基线为 `d237d1ca5eb9734dbb652ac0bd7baf454f7f404a`。GGML 版本为 0.25.3，
上游固定于 `353b63b439f27ab2cc19dac97ab1681ba6d2d084`，叠加项目精度和 CUDA
优化补丁。现有 [GGUF 契约](../gguf.md) 与 [量化指南](../quantization_zh.md)
定义的是权重存储策略，不能直接作为激活计算策略。

| 部分 | 实际实现 | 本计划需要补足的能力 |
| --- | --- | --- |
| 权重 | 图像线性层支持视觉／全模块／自定义模块量化；部分矩阵保留 F32 或 Q8_0 | 按校准结果选择层、粒度和例外，绑定校准来源 |
| CPU | 压缩权重驻留，量化矩阵在图内临时转成 F32 后计算 | 单独评估原生低精度算术，不能把现有行为称作端到端 W8A8 |
| Metal | 原生压缩权重内核及 half staging | 验证真实 A16 缓冲与融合算子的支持，独立测量 |
| CUDA 量化矩阵 | 原生 MMVQ/MMQ，RHS 临时量化成 Q8_1 | 持久激活格式、量化结果复用、校准及量化／反量化融合 |
| CUDA F16 模式 | dense 矩阵 F16 输入、F32 累加／输出，适用 attention 使用融合内核 | 让生产者、消费者和分配器真的使用较小激活，而非仅更换算术 |
| 图像特征／提示缓存 | 主要使用主机 `std::vector<float>`，阶段间上传／下载 F32 | 区分主机压缩、设备驻留及传输节省，避免重复保留表示 |
| 视频历史状态 | `TrackerRecord::memory` 已用 BF16；pointer 为 F32 | A8 必须相对 BF16 比较，并验证时序误差累积 |
| 视频设备工作集 | 当前帧 resident features 和大量计算输入为 F32 | 在有界 workspace 与对象批处理下压缩活跃设备张量 |

因此，“目前只有权重量化”准确描述了公开的量化模型契约，但内核内部已有临时
低精度激活处理。下一步要扩展其生命周期和数值契约，不能把现有 Q8_1 staging
重新命名后当作新增功能。

关键实现依据：

- [图与上传下载](../../include/sam/internal/runtime/ggml/graph.hpp)：输入默认 F32，
  `download()` 要求连续 F32；CPU 量化权重有显式 F32 转换。
- [运行时策略](../../include/sam/internal/runtime/ggml/runtime.hpp) 与
  [后端策略](../../include/sam/internal/runtime/ggml/backend.hpp)：已有算术 profile、
  attention policy 和 strict compute placement，可扩展但不应把设备判断散入模型图。
- [图像执行](../../include/sam/internal/models/sam3/execution.hpp)、
  [视觉图](../../include/sam/internal/models/sam3/vision.hpp) 与
  [视频执行](../../include/sam/internal/models/sam3/tracking/execution.hpp)：缓存、
  QKV／MLP 激活、BF16 history 和 F32 resident tensors 的具体位置。
- 本地固定 GGML 的 `src/ggml.c::ggml_mul_mat()` 直接创建 F32 输出，
  `src/ggml-cuda/mmq.cu` 也要求 F32 输出。不能只修改 `CudaComputeMode` 或插入
  一个 cast 就承诺整图激活减半；需要显式输出类型能力、融合写回和消费者支持。
- 当前 `ggml_type` 没有通用 FP8 张量类型；已有 MXFP4/NVFP4 枚举也不表示
  SAM 文件契约或当前设备已经支持这些模型。

## 现有测量与优先级依据

下表摘自当前 [CUDA 性能记录](../../BENCHMARK_zh.md)，不是本轮新跑的结果。
相同 SAM 3 checkpoint，1008 × 1008 模型输入，RTX 4090，完整非缓存图像推理。
显存是约 50 ms 采样的进程峰值下界，包含 context 与 pools；GB 为十进制。

| 全模块权重 | CUDA 计算 | 延迟中位数，ms | 采样显存，GB |
| --- | --- | ---: | ---: |
| Q8_0 | 默认 | 401 | 2.569 |
| Q4_K | 默认 | 398 | 2.259 |
| Q8_0 | F16 快速模式 | 335 | 2.911 |
| Q4_K | F16 快速模式 | 328 | 2.600 |

Q8 → Q4 的快速模式延迟仅下降约 2.1%，采样显存下降约 10.7%；相同 Q8 权重，
F16 快速模式比默认模式快约 16.5%，但采样显存增加约 13.3%。这支持优先处理
激活、临时 staging 和算子数据流，而不是继续只压缩权重。以上是现象，尚不能
把新增显存全部归因于某一个 buffer；须在阶段 0 分解分配峰值。

沿用 [最近的 CUDA 优化计划](20261007-113358-cuda-layout-kernel-optimization.md)
及其不可变凭据：[性能](../../build/cuda-layout-performance-v1/sequence.json)、
[显存](../../build/cuda-layout-memory-v1/sequence.json)、
[验收](../../build/cuda-layout-acceptance-v1/sequence.json)。历史 attention、拷贝和
concat 优化已经实施，不能把重新启用 FlashAttention 当作本轮增量。

## 方法取舍

W 表示权重位数，A 表示被选中算子的激活位数；它们不涵盖默认保留浮点的
归一化、累加、softmax、残差及输出。每个 profile 都应列出这些例外。

| 方法 | 主要收益目标 | 工程代价与限制 | 优先级 |
| --- | --- | --- | --- |
| W8/W4 + A16 存储、融合写回 | 减少活跃张量、拷贝和转换缓冲 | GGML 输出类型及消费链需要配套；单纯 cast 可能增加峰值 | 第一批 |
| 校准 W8A8，静态／动态 scale 对照 | 降低激活流量并加速大线性层 | 必须胜过已有 MMQ + Q8_1，而非只胜过 F32；需要真正整数内核 | 第二批主线 |
| FP8 E4M3 权重／激活 | CUDA 上的大 GEMM 与激活带宽 | 新数据表示、scale、cuBLASLt 接入及能力探测；模型不是直接改一个开关 | 第二批对照候选 |
| QK INT8、PV F16 的融合 attention | 减少 attention 算术和中间流量 | SAM 3 的 RoPE、mask、head、stride 都需适配，不能直接接 Python 包 | 第三批 |
| 图像特征 A8、视频 memory A8 | 降低缓存容量与传输 | 历史 memory 已是 BF16，且主要在主机；时序漂移和解码开销要单独验收 | 第三批独立路线 |
| 校准混合 W4/W8 + A8，层／块重建 | 在质量预算内进一步减少权重及激活 | 转换和校准成本高；先修复敏感层，再考虑更低位宽 | 后续 |
| 全面 W4A4、非均匀低位 softmax、QAT | 更激进的容量与算力节省 | 额外内核或训练成本大；质量风险高于 A8 | 研究候选 |

### W8A8 的校准方案

从视觉编码器线性层开始，先覆盖 QKV、attention 输出投影和 MLP。SAM 3 有
32 个 ViT block，隐藏维 1024、MLP 维 4736、72 × 72 tokens、24 × 24 windows，
必须在这些实际形状上测试；特别保留 4736 与现有 K-block 256 不整除的边界。

第一组实验采用对称 INT8，权重按输出通道或与内核匹配的 group，激活对比
离线静态 scale 与运行时 per-token／per-block scale。INT32 点积累加后，按
明确的 scale 在浮点 epilogue 中处理 bias／残差；检查累加边界。分组 scale
若沿归约维变化，必须在内核中正确合并局部和，不能等价成一次最终缩放。

先测普通 min/max 和有界 clipping，再加入通道缩放。参考
[SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html) 的离线激活异常值
平滑，以及 [RepQ-ViT](https://github.com/zkkli/RepQ-ViT) 对 LayerNorm 后通道
差异的重参数化。前者证据来自 LLM，后者来自 ViT；两者都是移植依据，不是
SAM 3 的效果证明。

对 `Y = XW`，使用 `X' = X / s`、`W' = sW` 保持浮点代数等价。实施时按
GGML 的真实矩阵布局映射缩放轴；只有证明所有消费者、bias、残差路径均正确时
才折叠进前序 LayerNorm。QKV 的共享输入必须共同处理，不能给共享支路互相
冲突的缩放。重参数化自身先在未量化浮点图验证，然后再叠加量化误差。

首批保留 LayerNorm／softmax 归约、位置／RoPE 运算、残差合并以及最终 mask、
score、box 投影的既有浮点精度。text／fusion／decoder 在视觉路线通过后逐项
消融加入，避免同时改变多个敏感模块。对小矩阵优先保留现有路径，选择依据是
完整“量化 + GEMM + 写回”的时间，而不是 Tensor Core 峰值吞吐。

需要比较两种实现：复用 GGML 原生块量化算术并消除重复 staging；新增校准
INT8 packed kernel。Q8_0/Q8_1 的块布局与 scale 不是任意 per-channel INT8
布局，不能直接将一份缓存交给另一种 kernel。重复量化只在同一数据版本、
dtype、stride、scale 和消费者布局下复用；当前视觉 QKV 已合并投影，不预设
它有三份重复转换。

### CUDA FP8 的边界

[Ada 官方指南](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) 与
[cuBLAS narrow precision 文档](https://docs.nvidia.com/cuda/cublas/index.html#narrow-precision-data-types-usage)
说明 Ada 支持 FP8，cuBLASLt 的 tensorwide scaling 可用于 compute capability
8.9+。这不能外推为所有分组 scale 或布局均支持；查阅页面为 13.4 文档，而
当前测量构建使用 CUDA 13.3，实施必须核对本机头文件、库和算法返回结果。

先验证 E4M3 输入、F32 累加、F16 输出，使用真实 M/N/K 和有界 workspace；
比较校准静态 scale 与动态 amax 的总开销。SAM 公共接口不暴露 CUDA 类型。
GGML 内的新类型或显式低精度 operator 必须有可验证的 backend ABI，不能把
FP8 字节伪装为普通 I8 并交给现有运算。

FP8 是 W8A8 的竞争候选，不作为必然更快的最终方案。相对现有 Q4，FP8
权重可能更大；载入后同时常驻 GGUF 权重、解码浮点副本和 FP8 副本会抵消收益，
因此转换和驻留策略必须一起核算。原生 FP4 加速另待相应硬件验证。

### 量化 attention 与 SAM 专用方法

参考 [SageAttention 官方实现](https://github.com/thu-ml/SageAttention) 的
QK INT8 / PV FP16 路线，先验证无 mask 的视觉 attention，再扩展到其他路径。
官方实现依赖 PyTorch／Triton／CUDA，其 kernel 吞吐图不包含量化和平滑开销；
本项目应在 GGML backend 中实现或适配必要内核，计入所有转换与布局成本。

已有融合 attention 路径保持融合，不生成完整 N × N attention 矩阵。
softmax 的 max／exp／sum 和在线归一化先保留 F32，QK 的整数累加及 PV 的
累加精度明确记录，不直接继承外部实现的 FP16 累加策略。验证 head 32／64／256、
窗口 576／全局 5184 tokens、不等长 Q/KV、尾块、非连续 V、causal／additive
mask 与 decoder 位置 bias；没有覆盖的组合继续用该设备的已验证浮点路径。

[PTQ4SAM](https://openaccess.thecvf.com/content/CVPR2024/html/Lv_PTQ4SAM_Post-Training_Quantization_for_Segment_Anything_CVPR_2024_paper.html)
的 Key 双峰处理和自适应 softmax 量化可作为敏感层修复候选；其研究对象是原始
SAM，不是当前 SAM 3。先检查这里是否存在同样分布，尤其不能假设通道符号／
尺度变换可以任意穿过 SAM 3 的 RoPE。任何变换先证明浮点等价和 mask 语义。

[AHCPTQ](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_AHCPTQ_Accurate_and_Hardware-Compatible_Post-Training_Quantization_for_Segment_Anything_Model_ICCV_2025_paper.pdf)
对 post-GELU 长尾及通道分组的研究可供后续 W4A4 参考；其硬件加速证据来自
FPGA，不能当作 CUDA／Metal 的提速预期。非均匀编码只有对应消费者能直接使用，
且端到端测量获益时才进入产品路线。

## 分阶段实施及退出条件

### 阶段 0：冻结基线、盘点内存、建立校准数据

1. 复用当前验收过的构建，记录 source／GGUF／patch／运行库身份。在新目录
   保存本轮测量，不改写前轮 evidence。只先跑 dense F16、full Q8、full Q4 的
   CUDA F16 模式，以及 Q8 默认模式，作为候选筛选基线。
2. 输出按阶段、张量、dtype 的活跃字节、最后消费者、alias、allocator 高水位。
   分开记录 resident weights、持久特征、graph arena、kernel staging、pool
   used／reserved 和主机缓存；同一分配只能计数一次。同步采样只能作为辅证，
   不能将独立阶段的最大值相加当作完整进程峰值。
3. 捕获实际 GEMM／attention 形状、布局、类型、次数和时间；将量化、cast、
   数据传输、CPU 等待单独归类。诊断构建不参与正式计时。
4. 建立原始 checkpoint 的校准导出器。建议起始集为 256 张图像，每张 2–4 个
   文本提示；覆盖负提示、小对象、细边界、低对比度和多实例。逐层流式收集
   min/max、分位数、饱和率和有限容量样本，不保存所有层的全量激活。
5. 校准集、参数选择集、最终评估集按原图或源视频拆分，裁剪和相邻帧不能跨集。
   既有七个图像用例只做回归，不能拿来调 scale 或充当完整精度评估。

交付：内存归因、实际形状清单、校准 manifest、当前最快／最省内存基线。
退出条件：能够指出拟压缩张量是否处在峰值上，以及对应算子占完整延迟的比例。
若目标算子只占耗时比例 p，其无限加速的理想上限也只有 `1/(1-p)`；据此决定
下一批工作，不以论文中的 kernel 倍数预测整机收益。

### 阶段 1：A16 存储与紧凑工作区

1. 先选择占用最大、消费链明确的视觉 MLP、QKV 和 neck 特征。扩展内部 tensor
   I/O 与 backend precision policy，支持明确的存储、输入、累加及输出类型。
2. 验证 F16 消费者的连续链和必要的 BF16 候选；不是所有算子都支持相同类型。
   为 GEMM epilogue／bias／GELU 等建立有界融合方案，让结果直接写入小类型。
3. 仅用于数值筛选的“F32 输出后 cast”为实验，不作为最终内存优化；必须确认
   F32 临时结果的生命周期已消除或缩短，并计入同时存活的源、目标和 staging。
4. 图像主机缓存压缩与设备驻留分别消融。设备驻留若减少传输却增加显存，作为
   单独取舍记录；诊断导出按需解码，正常路径避免持有重复 F32 诊断副本。

交付：可选的 A16 activation profile 及逐层例外清单，保留现有模式。
进入下一阶段的条件：质量通过，实际内存峰值可重复下降，且无明显延迟回退。
若 kernel 不支持较小输出，先补齐或缩小范围，不把 CPU fallback 当作支持。

### 阶段 2：校准 W8A8 与 FP8 的小范围竞争

1. 使用阶段 0 的样本做逐层和逐块 fake-quant 消融。分开测权重量化、激活量化、
   二者组合和重参数化，找出需要 F16/F32 例外的层。fake-quant 只证明数值可能性。
2. 在真实 CUDA shape 上分别实现最小 INT8 与 FP8 原型；同步测量所有 scale、
   packing、GEMM、epilogue 和 workspace，先筛掉不能胜过当前 native Q8 的候选。
3. 将获胜路线接入视觉编码器，再逐个加入其他模块。优先融合生产者量化、复用
   消费者可共享的量化表示；避免每一层独立 Q/DQ 后仍全程保存 F32 激活。
4. 固定校准产物与新 profile，完成原始模型最终输出验收后才跑正式性能矩阵。

交付：有真实 kernel 证据的校准 profile、转换工具、原始参考比较和性能／内存结果。
若均未改善质量约束下的速度／内存组合，保留 A16 成果，终止此轮 A8 集成。

### 阶段 3：attention 与状态压缩分别验收

attention 先做 QK INT8 + PV F16，再决定是否需要 PV FP8。对 causal、位置
bias、RoPE 或形状不兼容的路径维持原算术。显式记录被量化 attention 的层数、
调用数和覆盖耗时；仅“所有节点在 CUDA”不足以证明它们使用了目标低精度。

状态压缩先做图像静态特征，再做视频 memory。后者以现有 BF16 存储为基线，
研究按通道组／空间块的对称 INT8 和每条 record 的固定 scale，保留 pointer、
object score 和时序选择逻辑的浮点表示。A8 编码一次，在必要时解码／上传；
记录 host retained bytes、上传字节、device workspace 和解码耗时。SAM 视频
memory 是空间特征与对象状态，不直接套用自回归 LLM 的 KV-cache 假设。

视频校准起始规模建议为 8–16 段 32–64 帧片段，按视频隔离评估集。视频阶段
可以先只改变运行时状态，权重仍用已有 F32/hybrid；若要加入视频权重量化，另
定义视频 profile 和完整权重契约，不复用现有 image schema-4 验收结论。

交付：attention、图像缓存、视频状态各自的消融和组合结果。若压缩只降低主机
RSS，应如实归入主机内存收益；若长序列漂移失败，状态回到 BF16。

### 阶段 4：混合位宽与精度修复

在 A8 消费链和校准工具稳定后，按单位内存节省带来的质量损失选择 W4/W8/A16
例外，做层／块输出重建。转换始终从原始 checkpoint 开始，避免重复量化已经
有损的 GGUF。先增加敏感层位宽，再评估训练成本；只有 PTQ 与重建不能达到已
冻结的质量目标时，才立项小规模 QAT／蒸馏和对应数据、算力预算。

W4A4、INT4 attention、非均匀 post-softmax 编码和 FP4 保持独立研究候选。
剪枝、降低分辨率、token 丢弃会改变计算图或任务行为，另立计划，不能混入
量化对照掩盖收益来源。

## 校准、格式及工程契约

| 层次 | 计划中的修改位置 | 责任 |
| --- | --- | --- |
| 模型与校准 | `tools/convert_sam3.py`、`tools/sam3_gguf.py`、新增离线校准工具 | 原始权重、数据集身份、缩放轴、例外层、量化参数和 provenance |
| 模型图 | `internal/models/sam3/ops.hpp`、`vision.hpp`、`execution.hpp` 及 tracker | 标记可降精度边界、正确消费类型，保留 SAM 3 数值与时序语义 |
| 运行时 | `runtime/ggml/{backend,runtime,graph,workspace}.hpp` | 能力协商、类型感知 I/O、生命周期、graph cache key 与内存计数 |
| 后端 | `runtime/ggml/backends/` 与固定 GGML 补丁 | 内核、硬件与 scale 粒度能力、输出格式、workspace、严格放置 |
| 验收 | `tests/`、`tools/validate_image.py`、`tools/validate_video.py` | 原始模型质量、数值边界、长序列行为和性能凭据 |

以下为设计提案，不是已可用的 metadata 或 API：

- 将 `storage_profile`、`arithmetic_profile`、activation／state storage policy
  独立描述。runtime 报告实际 kernel／例外和 calibration identity；图缓存 key
  包含 activation/state 类型、scale 版本、shape、stride，禁止误用其他模式缓存。
- 只改变运行时 A16 存储的模式不必重写原始 GGUF；携带新 scale、重参数化权重、
  校准布局或混合位宽规则的模型采用新 SAM schema／版本化 profile。建议从
  `sam.schema_version=5` 设计独立契约，实施前先完成字段规范；GGUF 容器版本
  不因 SAM schema 升级而自动变化。
- 校准 manifest 记录 checkpoint、tokenizer、预处理、参考代码 revision、
  tensor mapping、数据／提示清单散列、拆分、seed、算法／粒度和参数。权重与
  runtime scale 的文件身份相互绑定，不能只依赖一个可被替换的 sidecar。
- 所需 scale／zero point／tensor layout 必须在分配权重前校验类型、维度、
  数量、有限性、非零正 scale、范围及溢出；损坏、不支持或缺失的契约明确拒绝。
- CPU／Metal／CUDA 分别声明能力。CUDA 上的同设备浮点例外与禁止 CPU compute
  fallback 是两件事，均需可见。新 profile 先显式选择，不能悄悄改变 Auto。
- 保持调用者提供 GGML 的集成方式；新增 operator／ABI 除编译检查外，还需要
  启动能力及数值探针。更新补丁固定 revision、hash 和必要 license。先内部接口，
  在候选验收后再决定最小公共选项，不一次性加入所有实验模式。
- CPU 优先验证较小缓存与消除全量 F32 解码的价值；INT8 路线需探测实际 SIMD
  或库能力。Metal 优先 A16／融合与已有量化内核；A8 算术须在真实设备证明收益，
  不能沿用 CUDA FP8 或 Tensor Core 的结论。

## 验证与性能验收

### 精度和行为

所有候选同时与原始 checkpoint 和当前对应的已验收模式比较；前者决定绝对质量，
后者隔离新增误差。decoded-GGUF、fake-quant 和 kernel reference 都是补充诊断。

1. 保留全部既有图像／视频回归和原始质量门限。W8A8／FP8 的首个质量目标至少
   按现有 Q8 输出要求验证：高置信对象 mask IoU ≥ 0.96、score 绝对误差 ≤ 0.02、
   box 坐标误差 ≤ 相应图像维度的 0.01；同时执行高置信漏检与低置信误选检查。
   新 arithmetic profile 应配独立、预先冻结的 gate 文件，不修改旧 gate 冒充通过。
2. 建议扩展至至少 512 张独立图像的有标注评估，报告 mIoU、mask AP（适用时）、
   边界质量、小对象、负提示及阈值邻域变化。起始发布预算建议对原始模型平均
   mIoU 下降不超过 0.5 个百分点、mask AP 下降不超过 1 个百分点；按任务冻结后
   再运行候选，不能看结果后放宽。七个固定用例不足以证明通用精度。
3. 视频覆盖 1／4／8 对象、进入／离开、遮挡再出现、空场景和长时传播，保留
   既有 216 帧验收及独立 session 检查，再增加未参与校准的至少 256 帧序列。
   报告逐帧 mask／boundary、ID 连续性、重识别、记忆漂移和内存上界；量化不得
   改变帧顺序、输出延迟与对象管理语义。
4. 新内核覆盖零向量、极端值、饱和／clipping、非法 scale、尾维、stride、mask、
   RoPE、分组 scale、混合累加与重复运行。测试真实数学／接口行为，避免只检查
   enum 或实现分支。保留独立头文件、重复包含和 two-TU linkage 检查。

### 时间与内存

先保留现有 cold + five-full + cache 协议，完整图像、更换提示命中图像缓存、
完全缓存重放分别列出；正式候选增加至少三个独立进程、每个至少 20 次 warm
完整调用的 baseline/candidate 交错测量，用于判断小差异与 P95。GPU 串行运行，
固定输入、功率状态、build、线程与模型身份，profile 与计时分开。

视频沿用 64 帧、前 16 帧 warmup、后 48 帧统计，并对额外长序列报告稳态及
峰值。冷启动、权重 repack、校准成本另列，不用缓存命中数据代替完整推理。

内存同时提供逻辑 live bytes、allocator／pool used 与 reserved 的高水位、
采样 GPU 进程峰值和 RSS。采样值注明下界；分配轨迹无法覆盖的 context／库内
分配单独说明。显存和主机 RSS 不相加，文件大小不能代替运行时内存。

建议预先冻结两类发布目标，均是**实验目标，尚未达成**：

| 类型 | 相对同质量已验收基线的目标 | 不接受的代价 |
| --- | --- | --- |
| 均衡模式 | 完整图像延迟中位数至少降低 10%，峰值显存至少降低 10% | P95 延迟上升超过 3%，或依靠宽松质量门限取得收益 |
| 节省内存模式 | 峰值显存至少降低 15% | 中位数／P95 延迟上升超过 3% |

阶段 0 需用同一新计量口径重测基线。候选须位于现有配置的速度／内存有效前沿，
不能只与默认 F32 比较而忽略已存在的 full Q8/Q4 + F16。每层位宽减少的理论
比例只适用于该层 payload，还需扣除 scale、padding、重复表示与 workspace。

### 构建、回归和交付

每个实施批次按改变的后端使用独立 Release build：CUDA 与 CPU 回归必须运行
对应 CTest；Metal 修改在 Metal 硬件上运行，缺少硬件时保持未验证状态。在隔离
参考环境运行 `tools/test_tools.py` 及受影响的量化工具测试。通过后再发布独立的
checkpoint／后端／硬件／精度／延迟／峰值内存表及 mask／视频对照。

新 profile 的使用方法进入量化指南，格式进入 GGUF 文档，已验证范围进入 Model
Zoo，实测结论进入 BENCHMARK，完成的行为进入 changelog；失败候选、诊断分布、
实验参数和 receipts 留在本计划。每次验证后清理不再需要的构建中间物和下载／
转换缓存，保留原始权重、可用模型、当前验证构建及不可变证据。

## 初始规划结果与后续记录

- 已完成代码、固定 GGML、现有性能及质量契约检查，并核对上述论文和官方硬件
  文档；路线中的方法均为待验证候选，没有宣称 SAM 3 的新增量化已实现。
- 当前最先实施的交付应是阶段 0 的活跃内存／staging 归因和校准数据契约，再据
  峰值张量清单实施阶段 1。校准工具和 fake-quant 筛选可以先于复杂内核完成。
- 本次仅新增规划文档和相应 changelog 说明；未运行推理、转换模型或改动现有
  benchmark／acceptance 凭据。`tools/check_docs.py` 已通过 59 份文档的双语表格与
  本地链接检查；`git diff --check` 和新增文件的独立 whitespace 检查通过。
  本轮没有代码变化，因此未重新构建或运行 CTest。

后续每一阶段在此追加：假设、单变量改动、原始模型质量、完整延迟、内存归因、
保留／拒绝原因及不可变 evidence 链接。未完成阶段不记为模型支持。

### 2026-10-07 实施启动

用户已要求开始实施路线。首批实现阶段 0 的可复用诊断和校准设施，随后推进
A16 与后续候选，不以新增诊断工具作为整条路线完成的依据。

首批步骤：冻结现有构建和源文件身份；为内部 runtime 添加可选图观察接口，
由独立诊断工具导出张量形状、类型、alias、生命周期和分配区间；结合独立 CUDA
pool 诊断及基线 profiling 分离 arena、staging 和库内存；建立按来源拆分的
校准 manifest 与有界统计导出器。诊断工具运行不作为正常推理计时。

验证涵盖共享 arena 的复用／view alias／外部 resident buffer，CPU 与 CUDA
图执行回归、header/linkage，以及校准拆分和统计边界。用原始 image checkpoint
和 full Q8/Q4 在实际 RTX 4090 上产生新证据，旧构建、模型与 receipts 保持不变。

首轮 Q8 + CUDA F16 图诊断显示，视觉 neck 的 F32 `im2col` 展开为
764,411,904 bytes，位于约 870.6 MB 的图活跃区间峰值。该卷积把 columns 放在
`MUL_MAT` 左操作数；F16 模式原本已在 CUDA pool 将它转换为 F16。因此阶段 1
先验证直接生成 F16 columns、让既有 GEMM 直接消费，不产生完整 F32 columns。
只在既有显式 CUDA F16 模式采用 backend 给出的存储策略；F32／CPU／Metal 保持
原存储。先比较中间张量和最终输出是否与旧 F16 路径逐位一致，再判断是否需要
独立算术 profile。初始范围不涉及视觉线性层、attention 或历史 memory 的 A8。

用户确认优先使用本地 COCO。已找到完整 val2017 图像和实例标注；固定 seed 的
内容去重拆分为 256 calibration、128 selection、512 evaluation。它们是同一
公开 split 内互斥的自定义子集，不能称为完整官方 COCO 评估。校准覆盖 79/80 类，
提示由最多三个已标注类别加一个未标注类别组成；标注缺失不等同于真负样本。

### F16 columns 改动前的历史性能表

以下为上一批 CUDA 布局优化的 cold + five-full + five-cache 结果，
用于保留本次更新前的记录，不代表直接 F16 columns 的当前测量。单位为十进制 GB。

| Image weights | F16 compute median, s | P95, s | Peak RSS, GB | Sampled GPU, GB |
| --- | ---: | ---: | ---: | ---: |
| F32 | 0.344 | 0.348 | 1.102 | 5.184 |
| Mixed F16/F32 | 0.358 | 0.363 | 1.130 | 3.613 |
| Vision Q8_0 | 0.336 | 0.346 | 1.137 | 3.878 |
| Full-component Q8_0 | 0.335 | 0.341 | 1.140 | 2.911 |
| Vision Q6_K | 0.335 | 0.346 | 1.143 | 3.808 |
| Full-component Q6_K | 0.333 | 0.341 | 1.147 | 2.760 |
| Vision Q5_K | 0.330 | 0.341 | 1.143 | 3.771 |
| Full-component Q5_K | 0.330 | 0.339 | 1.147 | 2.678 |
| Vision Q4_K | 0.332 | 0.341 | 1.142 | 3.733 |
| Full-component Q4_K | 0.328 | 0.335 | 1.147 | 2.600 |

### 2026-10-07 阶段 0／1 首批验收

已实现：内部可选 `GraphObserver`、`sam_profile_graph`、COCO manifest 构建器、
原始 F32 视觉线性层校准导出器，以及后端决定的直接 F16 convolution columns。
F16 columns 只在显式 CUDA F16 算术策略中选择，CPU／Metal／默认 F32 仍生成
F32 columns。矩阵仍 F32 累加和输出；不修改 GGML ABI、不新增模型格式。
目前没有端到端 W8A8／FP8 profile。

旧 columns 在 F16 模式中本来就会被同一矩阵 kernel 转为 half。当前实现使
`im2col` 直接写入相同的 half 操作数，去掉完整 F32 展开和重复的 pool 表示。
Dense F16、full Q8、full Q4 的七个用例分别通过原始 checkpoint 质量验收，
全部 228 个导出的 raw tensor／mask 文件与前一 F16 实现散列完全一致，因此
沿用既有 F16 arithmetic profile；这不等于与 F32 oracle 逐位一致。

| 权重 | 原中位数，ms | 现中位数，ms | 原 P95，ms | 现 P95，ms | 原采样显存，GB | 现采样显存，GB | 显存降低 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense F16/F32 | 361.527 | 356.418 | 364.946 | 359.374 | 3.613 | 2.974 | 17.7% |
| Full Q8_0 | 335.092 | 335.929 | 340.581 | 341.029 | 2.911 | 2.227 | 23.5% |
| Full Q4_K | 335.714 | 330.429 | 342.702 | 338.887 | 2.600 | 1.917 | 26.3% |

硬件为 RTX 4090，checkpoint 与之前相同。每项 baseline／candidate 均三个
独立进程，按 AB／BA／AB 顺序，每进程 20 次完整热调用；表中合并 60 个样本。
显存另跑三次 cold + five-full + five-cache，50 ms 采样绑定实际 CUDA PID；
同一配置三次峰值一致，但仍是进程峰值下界。模型和实际载入的 CUDA 库都有身份
检查。三项均满足预设的“节省内存模式”目标；均衡模式的 10% 提速目标未达成。
Q8 延迟中位数 +0.25%、P95 +0.13%，视为基本持平，不宣传算术吞吐的额外提速。

Q8 诊断分解：图 workspace 统计从 1,072,777,632 降为 690,571,680 bytes；
VMM scratch pool used 从 383,385,600 降为 42,569,728 bytes，reserved 从
383,778,816 降为 44,040,192 bytes。这些是各自的高水位，不能相加当进程峰值。
图级统计显式排除外部 resident storage、backend padding、未归本 context
所有的 scheduler copies 和 kernel scratch；view 延长源分配寿命，地址重叠
区间只计一次。生产构建无 pool 日志，日志仅存在于独立诊断动态库。

Nsight 的三个完整调用及初始化记录中，Q8 packed GEMM 总 kernel 时间约
66.5 ms，RHS Q8_1 量化约 15.8 ms，即平均约 27.4 ms／调用；F32→half 转换
从约 5.22 降为 1.79 ms／调用。这里只是 GPU kernel duration 的和，不是完整
wall time，初始化也包含在记录内。它说明单独替换线性层核心不足以承诺整机
10% 提速；后续 A8 还需评估 bias／GELU、量化写回及整个消费链的融合收益。

不可变凭据：

- [初始源文件／旧二进制身份](../../build/runtime-quantization-baseline-v1/snapshot.json)
- [图与 VMM 分解、真实算子形状](../../build/runtime-quantization-baseline-v1/graph-analysis.json)
- [独立 VMM 诊断库构建](../../build/runtime-quantization-baseline-v1/pool-probe/build.json)
- [三种权重的原始模型验收](../../build/runtime-quantization-acceptance-v1/sequence.json)
- [与旧 F16 路径的逐位比较](../../build/runtime-quantization-acceptance-v1/exact-comparison.json)
- [交错完整性能与独立显存运行](../../build/runtime-quantization-performance-v1/sequence.json)
- [汇总对照](../../build/runtime-quantization-performance-v1/comparison.json)
- [Nsight kernel 分类](../../build/runtime-quantization-kernel-profile-v1/kernels.json)
- [CUDA CTest](../../build/runtime-quantization-cuda-v1/ctest.log) 27/27；
  [CPU CTest](../../build/runtime-quantization-cpu-v1/ctest.log) 15/15，含头文件和 linkage 构建检查。

共享图像编码／检测也用于视频。F32 与 hybrid 权重的 F16 路径均已通过独立的
五场景、216 帧原始参考输出验收，覆盖移动、进入、遮挡、添加／移除和负提示。
每种权重的全部导出 tensor、mask 和逐帧 score／box／ID／输出时序也与旧 F16
路径逐位一致，见[视频验收](../../build/runtime-quantization-video-acceptance-v1/sequence.json)
和[精确比较](../../build/runtime-quantization-video-acceptance-v1/exact-comparison-v2.json)。
每种权重各比较 378 个诊断张量、220 个导出 mask 和 216 个逐帧结果 JSON。
本表只给图像收益，视频独立计时记录见文末。既有 CPU／Metal 和默认 CUDA
存储、算术策略保持不变；没有在当前机器新增 Metal 验证。

### 校准与数值筛选的使用

本地 COCO 数据不复制进仓库。manifest 的三组为 256／128／512 张互斥图像，
对应 846／413／1676 个提示；校准／评估均覆盖 79 个已标注类别，选择集 75 类。
校准含 143 张带小对象的图像；此划分并非完整 COCO 官方评估，也未声称覆盖所有
细边界或低对比场景。所有图像按内容散列验证；对既有回归源图禁止普通校准和
参数选择。`--diagnostic` 允许显式工具烟测，其结果不得当作完整校准。

```sh
rtk proxy .venv-reference/bin/python tools/prepare_coco_calibration.py \
  --annotations /path/to/coco/annotations/instances_val2017.json \
  --input-root /path/to/coco --output models/calibration/coco2017-v1/dataset.json
rtk proxy .venv-reference/bin/python tools/export_calibration.py \
  --checkpoint /path/to/original/sam3.pt \
  --dataset models/calibration/coco2017-v1/dataset.json --input-root /path/to/coco \
  --sam3-source /path/to/pinned/sam3 \
  --sam3-runtime-source build/reference-runtime/sam3-image-cuda-v2 \
  --bpe /path/to/bpe_simple_vocab_16e6.txt.gz --device cuda --sample-rows 512 \
  --output models/calibration/coco2017-vision-f32-v1
```

导出器逐层记录准确的 min／max、F64 累计的一／二阶矩及固定容量的均匀 token
reservoir；p99／p99.9 只是 reservoir 估计，不能当作全样本尾部分位数。128 个
视觉线性层每张图必须各观察一次，并单独拦截绕过 `nn.Linear` hook 的融合 MLP。
原始权重、数据、BPE、参考代码、适配及导出代码身份都写入 manifest；输出目录
必须全新，失败不发布半成品。没有使用 selection／evaluation 激活拟合参数。

`tools/study_activation_quantization.py` 是下一阶段的离线数值筛选：使用原始权重，
比较按输出通道 W8、per-token／静态 tensor A8，以及 identity 和五个 SmoothQuant
alpha；分别报告 W-only、A-only、组合和浮点等价误差。它只在校准 reservoir 上
比较不含 bias 的线性输出，明确标记 diagnostic，不能当作分割质量验收、GGUF
新模型或真实 kernel 性能。诊断校准默认被拒绝，需显式 `--allow-diagnostic`。
运行期间文件身份改变会拒绝发布；首次工具烟测恰因迭代源码触发此保护，未发布
结果，随后在固定源码上重跑。

### 2026-10-07 COCO 校准与 W8A8 层级研究结果

256 张校准图像、846 个提示已全部完成。每个视觉线性层观察 1,327,104 行 token，
保留均匀采样的 512 行；128 层的 sample payload 为 511,705,088 bytes。
独立的 128 张 selection 和 512 张 evaluation 图像仅验证来源与散列，没有进入
激活统计或通道缩放拟合。校准身份为
`289d5dbb6b8c0ed4b3d62d27435417a933fc6ca0a0876164ada154732d531b0e`。

在同一原始 checkpoint 的 128 个线性层上，对 reservoir 进行 F32 离线模拟。
所有浮点通道重参数化的最大相对 L2 误差为 `1.406e-7`。下表为各层组合 W8A8
相对 L2 误差的中位数，既不是整图误差，也不是 COCO AP 或 mask IoU：

| 通道缩放 | Per-token A8 | 静态 per-tensor A8 |
| --- | ---: | ---: |
| Identity | 0.014610 | 0.037070 |
| SmoothQuant α=0 | 0.023643 | 0.076078 |
| α=0.25 | 0.016203 | 0.044950 |
| α=0.5 | 0.012638 | 0.025512 |
| α=0.75 | 0.011136 | 0.016764 |
| α=1 | 0.012769 | 0.014948 |

这些结果支持把 per-token α=0.75、静态 α=1 和 identity 带入独立筛选，不能直接
把同一 alpha 固定到所有层。仍需检查尾部敏感层、块间误差累积、完整输出质量，
以及 INT8／FP8 原型包含 scale、packing、写回的实际成本。当前没有导出可供 SAM
运行的 W8A8 模型，也没有在该表中测量任何整数内核。

完整导出的工具源码已单独冻结。随后为导出器补上运行前后源码身份检查和 Pillow
版本记录；新版本的一张图烟测与原版本全部 128 层统计／reservoir 逐位一致。
该元数据增强没有重写已完成的完整校准。隔离 Python 环境中 81 项工具测试通过。

凭据与复现：

- [COCO 拆分](../../models/calibration/coco2017-v1/dataset.json)
- [完整校准 manifest](../../models/calibration/coco2017-vision-f32-v1/manifest.json)
- [校准摘要](../../build/runtime-quantization-baseline-v1/calibration-receipt.json)
- [原导出源码身份](../../build/runtime-quantization-baseline-v1/calibration-sources-v1/hashes.json)
- [导出器元数据增强烟测](../../build/runtime-quantization-baseline-v1/calibration-provenance-smoke.json)
- [完整逐层研究](../../models/calibration/coco2017-layer-study-v1/study.json)
- [数值摘要](../../build/runtime-quantization-baseline-v1/study-receipt.json)
- [工具测试](../../build/runtime-quantization-baseline-v1/tools-tests-provenance.log)

```sh
rtk proxy .venv-reference/bin/python tools/study_activation_quantization.py \
  --checkpoint /path/to/original/sam3.pt \
  --calibration models/calibration/coco2017-vision-f32-v1 \
  --device cpu --threads 2 --output models/calibration/coco2017-layer-study-v1
```

### 2026-10-07 视频 F16 路径的独立性能复测

F32／hybrid 权重各自通过原始参考验收后，串行测量单对象与四对象的 64 帧负载，
前 16 帧预热、后 48 帧统计，计入末尾 drain；无 profiler、校准或其他计算任务
并行运行。GPU 进程显存沿用视频工具约一秒采样的口径，是峰值下界。旧列来自
此前布局优化的历史记录，此处没有重跑交错 A/B，因此不以细小延迟差异声称提速。

| 权重 | 对象数 | 旧中位数，ms/帧 | 现中位数，ms/帧 | 旧 P95，ms/帧 | 现 P95，ms/帧 | 旧显存，GB | 现显存，GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 | 1 | 576.921 | 579.666 | 650.169 | 649.637 | 5.320 | 4.744 |
| F32 | 4 | 1028.242 | 1029.384 | 1214.742 | 1217.525 | 5.320 | 4.744 |
| Hybrid | 1 | 581.024 | 580.641 | 653.513 | 657.275 | 4.635 | 4.056 |
| Hybrid | 4 | 1042.196 | 1044.869 | 1228.153 | 1207.257 | 4.635 | 4.056 |

F32／hybrid 的采样显存分别降低约 10.8%／12.5%，未把图像的 15% 内存目标
套用于该视频结果。当前 RSS 分别为 1.356／1.507／1.356／1.514 GB；旧值为
1.351／1.480／1.350／1.493 GB，没有主机内存节省结论。视频历史 BF16 memory
与专用 tracker 卷积仍保持原有策略。

- [新视频测量序列](../../build/runtime-quantization-video-performance-v1/sequence.json)
- [新旧视频记录对照](../../build/runtime-quantization-video-performance-v1/comparison.json)
- [历史完整性能记录](../../build/cuda-layout-performance-v1/sequence.json)

阶段 0 的诊断与校准设施，以及阶段 1 的卷积紧凑存储首批改动已经验收。MLP／QKV
较小输出的连续消费链、真正的 INT8／FP8 kernel、完整 W8A8 质量评估、量化
attention 和视频 A8 状态仍未完成，不能把本批成果视为整条路线完成。

本批修改的源码副本、通过的验收凭据、运行二进制／模型身份和原基线保持不变的
检查汇总于[首批交付归档](../../build/runtime-quantization-first-delivery-v2/receipt.json)。

### 2026-10-07 第二批：真实 INT8／FP8 内核筛选

已启动[CUDA 矩阵原型子计划](20261007-155334-cuda-w8a8-fp8-kernel-prototypes.md)。
INT8 的 per-token／静态激活、F32／F16 输出已通过四类真实视觉层和边界检查；
基线使用设备驻留输入，完整计入激活打包、bias／erf GELU 和写回。FP8 当前候选
在关闭 fast accumulation 并限制 F32 累加后，仍未通过 decoded-operand 数值门限，
尚不进入整图。原始凭据、正式内核对照及后续质量筛选决策记录在子计划中。

本批仍是独立内核工具，尚未增加 SAM W8A8 模型格式／算术 profile；主计划的
连续 A16 消费链、独立图像质量评估、量化 attention 和状态压缩继续保留。

### 2026-10-07 第三批：独立 COCO 整图筛选

[128 张整图筛选与标注评估](20261007-165128-w8a8-coco-output-screening.md)已完成。
统一 α=0.75 的视觉 W8A8 虽满足 AP／mIoU 平均下降预算，仅 102 / 128 张通过
逐对象门限，未获模型资格；未量化平滑也有两张图略超特征等价门限。工具测试
94 / 94 通过，参考 INT8 算术与真实 CUDA 原型的四类矩阵误差小于 `9e-8`。
先修复对照并研究混合层候选，512 张独立 evaluation 继续保留至候选冻结后。

### 2026-10-07 第四批：浮点对照修复与混合层退出结论

[修复和混合层研究](20261007-172634-w8a8-mixed-layer-repair.md)已完成。保留原始
Linear 的 fused bias 与输入形状后，先前两张浮点失败图恢复通过；二次幂 scale
在全部 128 张图的特征及 413 个最终输出上与原始 F32 逐位一致。量化后的
全视觉、仅 MLP、仅 attention 线性层分别仅通过 98／96／111 张，均未满足
冻结门限。本轮不集成该 W8A8 profile，转入独立的图像特征缓存筛选。原型、
校准和 99 项通过的工具检查保留，512 张 evaluation 尚未用于选择候选。

### 本轮其余候选的推进边界

按照阶段 0 的耗时占比退出规则，量化 attention 不在本轮继续接入产品。
首个拟替换的视觉 head-64 F16 FlashAttention 在
[既有 kernel 记录](../../build/runtime-quantization-kernel-profile-v1/kernels.json)
中为 96 次、22.151 ms，约三个完整调用的 7.384 ms/图；相对于同配置约
335.929 ms 的整图墙钟耗时，仅约 2.2%。这只是诊断口径的优先级估算，包含
初始化的 kernel 记录不能精确归因成加速比；即使忽略该项计算，单靠此边界
也不足以支持 10% 整图提速目标。其他带 mask、RoPE 或不同 head 形状的 attention
没有因此被宣布无优化价值；它们需要独立算术、消费者和质量实验。本轮没有
实现或宣传 QK INT8/PV FP8 attention。

视频 A8 history 留作后续独立研究，未改写已有 BF16 状态。当前
[已验收 64 帧测量](../../build/runtime-quantization-video-performance-v1/comparison.json)
的单/四对象最多保留 27/108 条 record；对应 benchmark.json 记录 host history
17,943,552/71,774,208 bytes，其中 pointer 仍是 F32。即使不计 scale 开销，
BF16 memory 改为 INT8 最多只减少约 8.96/35.83 MB 的长期主机载荷。当前
`make_memory_payload` 会将选中的 BF16 memory 解码到 F32 再消费，所以仅替换
host 存储不会直接缩小该设备输入和算子工作区。它不能凭主机压缩比例满足本轮
显存目标。真正推进该路线需同时改造消费边界，建立独立视频校准/评估，覆盖
遮挡、重入和 ID 连续性；不能沿用图像 COCO 的结论。

连续 MLP/QKV A16 输出、更多 attention 形状、视频 A8、W4A4 和 FP4 仍是路线图
候选。本轮交付边界是经验证的紧凑卷积存储，以及有原生内核/整图质量证据的
线性和图像缓存筛选；没有通过退出条件的配置不增加公共精度选项。
