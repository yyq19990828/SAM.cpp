# 量化功能剩余验收

日期：2026-10-10，Asia/Shanghai。起点：`e3edc3c`。

状态：本轮有界 CPU/CUDA 验收完成；Metal 与未实现功能的缺口见末尾，不声称全部量化路线完成。

## 范围与方法

继续已实现量化功能的未完成验收。先清理已解包下载缓存及 Python 字节码，保留原始 checkpoint、可用 GGUF／转换 manifest、现有构建、历史数值与设备中断证据。历史计划记载的 `build/cpu-precision-tools/`、`build/v3-first/` 及当日 CPU fixture 证据在本机已不可用；不能把历史记录当成本轮验证。

优先补齐模块／逐张量混合权重与 CPU 原生量化模式的完整 SAM 3 验证：使用原始 checkpoint 转换一个 schema-6 混合模型，核对预览与实际类型／字节；在两张已有应用图片上与原生 F32、同权重 F32-compute 比较；独立测量同权重两种 CPU 模式的端到端性能与执行成本。质量差异与性能变化是描述性参考，不能擅自添加 COCO 硬门槛。

未实现的独立激活策略、混合 F16、kernel scratch 抓取属于工程工作，不得勾选成已验收。旧 v2/v3 大规模 campaign 已归档终止，不属于待验项目；CUDA/Metal 的设备证据独立核对，缺少硬件或实际执行时保持未验收。

## 步骤

1. 清理过时缓存；冻结当前源码、输入和历史保留证据身份。
2. 从固定 GGML revision 的现有源码建立当前 Release CPU 构建，运行完整 CTest 与隔离 Python 工具测试。
3. 用原始 checkpoint 转换 schema-6 示例，核对 1133 张量分配、类型、载荷、文件大小和 provenance；不重编码已有量化 GGUF。
4. 两张图片、每张两个提示词导出 F32、混合 F32-compute、混合 native-quantized 输出，比较完整 ranked bundle。异常输出、身份或调度错误属于实现失败，不能解释为软质量取舍。
5. 在一个案例上独立运行配对性能与图执行成本；记录硬件、线程、构建、模型、冷启动、三种工作负载与内存；不把缓存命中称为推理加速。
6. 核对仍未完成的设备／工程条目，更新相关计划、双语指南与 changelog；运行文档与空白检查，保留本轮原始证据，清理不再需要的本轮中间产物。

## 验证与证据

本轮原始日志和 JSON 使用新 Git-ignored 目录 `build/quantization-acceptance-20261010-195100/`，仅在当前工作站可用。转换后的完整混合 GGUF 与 manifest 保存在 `models/`。所有完成声明以实际命令及输出为依据；完整模型对照使用模型生成的伪参考，不是人工真值或生产准确率。

## 结果

### 清理、构建与转换

- 初始清理仅删除两份已解包下载归档和 17 个 Python 字节码目录，共 11,757,514 bytes。没有旧 `.o` 可清理；模型、原始参考、历史数值／trace 与现有构建未删除。
- 当前机器是 Linux x86_64、i7-12700KF、RTX 4090 24 GiB、driver 610.57.04，区别于前序 CPU 小矩阵计划的 i7-13650HX。不能将两台机器的绝对耗时配对。NVIDIA 技能目录已检查，没有适用于本次 GGML 验收的强匹配技能；没有安装技能或改动驱动／电源设置。
- `build/quant-cpu-current/` 为当前源码 Release CPU 构建，CUDA/Metal/BLAS/LLAMAFILE 关闭，GGML_NATIVE 打开；GGML 为固定 v0.26.0，包含仓库既有精度 patch。完整 CTest **20/20**，5.74 秒。
- 首次 Python 工具测试 **235 项、3 项跳过**，原因是新构建目录不同于测试默认路径。指定 `SAM_TEST_QUANTIZER`、`SAM_TEST_MIXED_VALIDATOR`、`SAM_TEST_NATIVE_CPU_VALIDATOR` 后，完整 isolated suite **235/235 通过、零跳过**，76.627 秒；实际 CUDA runtime study 也运行。首次输出保留，未将条件跳过伪装成执行通过。
- 从原始 `sam3.pt` 转换 `models/sam3-image-tensor-mixed-q4-q6-q8-cpu-v1.gguf`，schema 6，1133 张量，文件 **1,002,406,784 bytes**。F32/Q8_0/Q6_K/Q5_K/Q4_K 分别为 **872/128/37/1/95** 张量。全部张量的名称、shape、请求／解析格式、selector、保护／回退原因与字节数均同真实转换前预览一致；实际文件大小与 SHA-256 同转换 manifest 一致。独立 `inspect-precision` 通过。完整 checkpoint、BPE、转换器、量化器／动态库和配置身份保存在新回执中。

### CPU 完整模型参考与调度

- 使用已有 `truck.jpg` 和 `groceries.jpg`。首批四提示为 truck/person/fruit/bottle；参考仅包含一个正目标，所以额外运行 paper bag/car 两提示。合计 **2 图片、6 提示、6 参考对象、3 空参考提示**，不是 COCO／独立留出集／生产准确率测试。
- 原始 checkpoint 的明确 F32 oracle（禁用 TF32/autocast/compile）与当前原生 F32 均完成真实推理；两组比较的全部六个匹配对象 mask IoU 为 1，缺失／新增均为 0。
- 混合权重的 F32-compute 与 native-quantized 分别完成全部六提示，和原始 checkpoint 对照均无对象缺失／新增。详细 score/box/mask 差异保留在 public benchmark 报告中，没有添加硬质量门槛。
- 公共 `compare` 正确拒绝将混合 F32-compute 当作 F32 权重参考。本轮另用明确标记的本地 `compare_same_weights.py` 调用共享配对算法，检查两个 bundle 的模型／checkpoint／策略／输入／提示／token／二进制身份后，报告同权重计算敏感性，未修改公共参考合同。同权重 native 对 decode 路径的六对象 mask IoU 均值 **0.999403**、最小 **0.998709**，无对象缺失／新增。
- 两种 CPU 模式的完整图成本诊断均真实执行一张 truck/一提示。原生模式实际 **285** 次 Q `MUL_MAT`（Q6_K/Q4_K/Q8_0/Q5_K 为 61/95/128/1），均为 CPU placement、F32 图上 RHS／输出，**零 Q→F32 cast**。源码 traits 标记 RHS Q8_K/Q8_0 为 157/128，不是 kernel scratch 捕获。默认路径为 **285** 次 Q→F32 cast、零 packed matmul；相同 519 次总 matmul 保持图职责不变。诊断时序不用于速度比较，kernel 内部打包／算术仍为 NOT_COLLECTED。

### CPU 独立性能

同一混合 GGUF 的 decode/native 两配置，固定四线程，一个 groceries-positive 案例，AB 一对独立进程，warmup 1、计时 3 次、独立内存 1 次，三种既有工作负载全部测量。为减少编译竞争，本轮自己的 CUDA 编译进程树先暂停，由带 finally 恢复及 PID start-time 检查的本地调度器恢复；没有暂停用户的其他进程。CPU 测量 exit 0 后已恢复全部 21 个仍存在的自有编译进程。无节点 observer，也没有并行的本轮推理／工具测试。该小样本仅给有界描述性结果，不授予稳定加速结论。

| 工作负载 | 默认 F32-decode P50 / P95 (ms) | Native CPU P50 / P95 (ms) | P50 候选／基线 | 默认 5% 展示标签 |
| --- | ---: | ---: | ---: | --- |
| full_image | 31016.381 / 31079.079 | 20388.631 / 20423.028 | 0.657350 | OBSERVED_REDUCTION |
| changed_prompt | 3028.184 / 3036.574 | 2889.956 / 2892.002 | 0.954353 | NOT_DEMONSTRATED |
| repeated_result | 10.968 / 11.020 | 10.977 / 11.025 | 1.000806 | NOT_DEMONSTRATED |

独立内存进程峰值 RSS 为 **2,386,554,880 / 2,368,544,768 bytes**，候选／基线 **0.992454**（仅约 0.75% 下降），没有获得内存展示标签。不能根据小矩阵速度或避免 Q→F32 cast 推断完整模型的内存显著下降。重复结果是缓存命中，不能算推理加速。

`cpu-performance/report.json` 为 VALID；`summarize-performance` 对 `run.json` 及原始记录离线重验成功，保存到独立的 `cpu-performance-rechecked/`。质量报告与性能互不作为前置门槛；同权重参考一致性不等于生产准确率。

### CUDA 设备与独立性能

- `build/quant-cuda-current/` 从当前源码与同一固定 GGML revision 完整构建，Release、架构 89、CUDA 开启、Metal/BLAS/LLAMAFILE 关闭、`SAM_REQUIRE_CUDA_TESTS=ON`、CUDA probes 关闭。完整 CTest **34/34 通过**、14.21 秒、零跳过，包含实际设备的四格式量化与 Q8 MMQ scale 数值回归。没有修改 GGML kernel。
- 当前 CUDA F32 与 schema-6 混合策略均完成两图片／六提示真实 ranked 导出。producer 核对模型／配置／算术 profile，并检查 CUDA compute nodes > 0、CPU/BLAS/Metal compute nodes 均为零。混合配置覆盖四种 Q 权重和 F32 保护张量；实际 GPU kernel 内部算术抓取仍为 NOT_COLLECTED。
- 原始 checkpoint 参考的六对象匹配 IoU：原生 F32 CPU/CUDA 均值／最小均 **1.0 / 1.0**；混合 CPU decode 为 **0.998279 / 0.996401**，混合 CPU native 为 **0.998174 / 0.995988**，混合 CUDA 为 **0.997331 / 0.994643**。五种候选均无缺失／新增，三个空参考提示均保持空。各原始报告保持不变；`quality-summary.json` 仅汇总互不重复的提示行，没有伪造新的导出包或重新授予历史资格。
- 独立 CUDA 性能比较原生 dense F32 与该混合 GGUF；两边均 CUDA F32 compute/F32 cache，四线程、同一个 groceries-positive 案例、AB/BA 两配对，预热 2、计时 5、独立内存 1。质量不是性能测量的前置门槛；模型大小／计算策略变化与 CPU 同权重模式对比是不同实验。

| 工作负载 | Dense F32 P50 / P95 (ms) | 混合 CUDA P50 / P95 (ms) | P50 候选／基线 | 默认 5% 展示标签 |
| --- | ---: | ---: | ---: | --- |
| full_image | 493.320 / 500.565 | 385.229 / 391.529 | 0.780890 | CONSISTENT_REDUCTION |
| changed_prompt | 134.116 / 139.263 | 126.282 / 130.615 | 0.941589 | CONSISTENT_REDUCTION |
| repeated_result | 11.115 / 11.514 | 11.072 / 11.134 | 0.996114 | NOT_DEMONSTRATED |

GPU 进程峰值为 **4,777,312,256 / 2,476,736,512 bytes**，比值 **0.518437**。主机 RSS 为 **1,011,040,256 / 1,044,152,320 bytes**，比值 **1.032750**，即增加约 **3.3%**，没有主机内存下降标签。GPU 显存为整个独立内存进程的峰值，50 ms 采样可能漏掉短峰值；不能解读为每个工作负载或结果缓存自己的内存。报告为 VALID，无观察到的 GPU 竞争；两个配对一致仅是该小样本内的描述性展示标签。`cuda-performance-rechecked/` 完成离线重验，整条有界队列 11 步全部 exit 0。

### 完成范围与保留缺口

| 条目 | 本轮结论 |
| --- | --- |
| 真实混合转换／预览与完整模型加载 | 完成：原始 checkpoint、1133 张量、四种 Q 类型、F32 保护／精确覆盖、文件与 manifest 身份 |
| 原生 CPU 完整模型参考／packed placement／端到端性能 | 完成：两图片六提示；实际图观察与独立性能分开 |
| 混合策略 M4 CUDA 有界设备验证 | 完成：RTX 4090，required-GPU CTest、严格 GPU 放置、完整输出与独立性能；仅实测 schema-6 分配 |
| M4 Metal 与 GGML v0.26.0 Metal 硬件验收 | 未完成：本机无 Apple/Metal 硬件，CPU/CUDA 结果不能代替 |
| CPU/BLAS 与其他 ISA／更多应用输入 | 未测量：本轮 BLAS 关闭；更多组合是后续单独测量，不继承当前收益 |
| 独立激活策略 A1–A4、混合 F16、kernel scratch／内部阶段计时 | 未实现／未采集：不是可以通过本轮验收勾选完成的功能 |

归档 v2/v3 campaign 已终止，不列入保留缺口。原中断／失败与冻结策略身份保持原样。

本轮没有更改生产代码、固定质量门槛、原始 checkpoint、旧 GGUF 或历史 receipt。双语指南、README、changelog 与关联计划已按实测范围同步。完整数值、每轮样本、转换／执行／性能来源与诊断原始数据位于前述 Git-ignored 新目录，只有当前工作站可用。最终回执 `verification-receipt-final.json` 绑定 12 个完成导出包、本轮日志、模型、报告与文档；文档检查 **159 份通过**、空白检查通过、原始输入身份不变及例行清理也在其中记录。

第一次回执组装提前给仍打开的自身 stdout 日志计算哈希；闭环复核发现仅该日志摘要不一致。初稿及日志保留原样，最终回执在所有日志关闭后重新封存文件清单，并记录取代关系。数值、模型、导出、性能与测试证据均未改写。
