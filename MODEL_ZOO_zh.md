# 模型与精度

[English](MODEL_ZOO.md) · [性能测试](BENCHMARK_zh.md)

SAM.cpp 面向多种 SAM 模型适配器，以及由检测和分割模型组合的跨平台流水线。
本页分别说明当前可用功能和后续接入方向。

## 当前可用

| 模型与任务 | 权重选择 | 当前后端 | 建议起点 |
| --- | --- | --- | --- |
| SAM 3 文本图像分割 | F32、混合 F16/F32 | CPU、Metal、CUDA | GPU 使用 F16；参考配置使用 F32 |
| SAM 3 文本提示图像分割（量化权重） | 视觉 Q8_0、Q6_K、Q5_K、Q4_K | CPU、Metal、CUDA | 优先试用 Q8_0 |
| SAM 3 文本提示图像分割（全模块混合权重量化） | 全模块线性 Q8_0、Q6_K、Q5_K、Q4_K | CPU、Metal、CUDA | 压缩优先可选全模块预设，再检查应用数据 |
| SAM 3 前向视频跟踪 | F32、hybrid | CPU、Metal、CUDA | `visual-tracker-f32-v1` hybrid |

表中的视觉量化仅覆盖 ViT 的注意力投影和 MLP 线性权重；文本编码器、融合模块、检测与掩码头，以及其余权重保留 F32。推理任务仍是文本提示图像分割。
全模块预设还覆盖文本、融合、检测与掩码等解码部分的目标线性权重，共 348 个矩阵；嵌入、卷积、偏置、归一化和明确的小权重例外继续保留 F32，激活精度保持原有策略。

用户可以用 `--quantize-modules vision,text` 等选择局部量化。仓库数值验收聚焦固定视觉和全模块预设，自定义组合需要另行验证，见[量化指南](docs/quantization_zh.md)。

[可视化样例](docs/visual-examples_zh.md)在相同图片、提示词和 0.2 检出阈值下比较各图像权重版本的 CPU、Metal 输出。量化版本以最终输出质量验收，张量误差单独供参考。

当前 SAM 3 实现仍属实验阶段，验证语料规模较小，接入应用时应检查自己的输入质量。
F16 视频、旧 `image-linear-*` 和自定义 `image-modules-linear-*` profile 默认保留诊断标签，应用集成应验证自己的数据。一份明确的[文本/融合/解码 Q8_0 CUDA 图像配方](docs/quantization_zh.md)已在 RTX 4090 上取得限定工作负载的 GPU 显存优化验收。
F32 和该 Q8_0 权重各自的实验性混合 Q8_0 缓存工具配方也已通过整图／换提示的延迟门槛；公共模型加载仍使用现有缓存精度。
目前没有量化视频、反向跟踪或交互式视频提示。

CPU 是跨平台执行路径。平台验证覆盖 macOS CPU/Metal，以及 Linux x86_64 RTX 4090
CUDA 的 F32/F16 图像、八种视觉／全模块量化图像预设和 F32/hybrid 视频。同机
Linux CPU 配合 OpenBLAS 的 F32/F16 图像也已通过。其他 CPU/GPU 硬件与操作系统
需要各自构建及数值检查。Metal 属于 Apple 平台后端，不是公共模型接口的前提。

## 模型与流水线方向

| 模型家族或流水线 | 目标任务 | 当前状态 |
| --- | --- | --- |
| 其他 SAM 代际，包括 SAM 2/2.1 | 点框图像提示、基于记忆的视频跟踪 | 待接入适配器 |
| SAM 3.1 | 独立的模型与任务契约 | 待接入适配器 |
| GroundingSAM | 文本检测模型与 SAM 分割适配器组合 | 待组合流水线 |
| DART 类流水线 | 复用检测阶段，允许跳过掩码解码 | 预留扩展边界 |

这些条目说明仓库方向，不表示已经实现。新适配器负责自己的张量格式、分词器、
变换和时序规则。[架构说明](docs/architecture.md)介绍公共接口及模型、后端的扩展边界。

## 精度约定

权重标签表示存储格式，计算和状态可以采用不同格式。以下策略属于当前 SAM 3
适配器及其后端，不应直接套用于后续模型。

| 权重标签 | 磁盘及 Metal / CUDA 驻留权重 | CPU 驻留权重 |
| --- | --- | --- |
| F32 | 原始 F32 | F32 |
| F16 | 混合 F16/F32 | 保存的 F16 升为 F32 |
| Hybrid | 视觉与跟踪器 F32，检测及文本混合 | 剩余 F16 升为 F32 |
| 视觉／全模块量化 | 所选线性层保留压缩 Q8/K，其余 F32 | 压缩权重驻留，计算使用临时 F32 矩阵权重 |

升格保留已舍入的值，不能恢复原始 F32。量化 Metal 和 CUDA 使用不同算术 profile
的原生 kernel；CUDA 已验证的 MMVQ/MMQ 路径使用 RHS Q8_1 staging。CPU 保留压缩
权重，并以临时 F32 权重完成矩阵运算。`ModelInfo::precision`、`storage_profile` 和
`arithmetic_profile` 表示实际加载配置，具体名称见[量化指南](docs/quantization_zh.md)。

CUDA 计算精度通过 `--cuda-compute f32|f16` 单独选择，C++ 对应
`BackendOptions::cuda_compute`。默认保留 F32 dense 输入与 attention；显式 F16
计算将 dense 操作数舍入为 F16，累加和输出保持 F32，并启用支持的低精度融合
attention。其 profile 为 `ggml-cuda-f16-v1`；量化权重继续使用原生压缩 kernel，
对应 `ggml-quantized-cuda-f16-v1`。存储标签与 GGUF 文件不变。
F16 计算采用既有 F16 最终掩码／分数／框门限，量化权重采用其既有输出质量门限；
中间张量误差单独记录。视频 ID、生命周期、输出延迟和状态边界仍须通过验收。

SAM 3 视频在 F32、F16、hybrid 存储下均保留以下状态边界：

| 边界 | 表示 |
| --- | --- |
| 归一化 | F16 舍入，输出为 F32 缓冲 |
| 跟踪 neck 特征 | BF16 舍入，保存在 F32 向量中 |
| 掩码记忆记录 | BF16 保存，注意力计算前展开为 F32 |
| 对象指针与主机掩码 logits | F32，最终二值掩码使用 uint8 |

F32 视频文件不表示整条流程都采用 F32，图像推理没有跟踪 BF16 状态。

## 下载与使用

当前适配器使用原始 [facebook/sam3](https://huggingface.co/facebook/sam3) 权重和分词资源。
模型访问权限、模型许可与依赖许可分别处理，见[许可说明](THIRD_PARTY_NOTICES.md)。

按[SAM 3 下载与转换指南](docs/models/sam3-details.md)准备固定版本源码、认证、Python
工具环境及图像或视频文件。运行 C++ 推理不需要 Python，转换输出必须使用新路径。

[README](README.md)提供图像、视频 API 和 CLI 示例；[模型验证](docs/validation_zh.md)
介绍转换后模型的参考对照方法；[性能测试](BENCHMARK_zh.md)按模型、硬件和后端列出测量。
