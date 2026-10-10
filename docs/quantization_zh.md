# SAM 3 图像量化

本指南介绍当前 SAM 3 adapter 实现的图像权重量化方式。下列版本化 profile
仅适用于 SAM 3，不为其他模型 adapter 规定通用量化策略。

## 按模块选择

SAM 3 图像权重支持以下模块，可单独选择或组合选择。一次转换使用同一种目标量化格式；
模块选择决定权重存储，矩阵 staging 遵循后端算术 profile。

| 模块名 | 范围 | 可量化线性矩阵 |
| --- | --- | ---: |
| `vision` | ViT 视觉编码器 | 128 |
| `text` | 文本 Transformer 和特征 resizer | 97 |
| `fusion` | 图像与文本融合编码器 | 36 |
| `decoder` | 检测、几何提示编码、评分与掩码头 | 87 |

`--quantize-modules text,decoder` 只量化所选模块的目标矩阵，其余模块保留 F32。嵌入、位置参数、偏置、归一化、卷积、部分小矩阵和标量输出投影始终保留 F32。几何权重的存储选择不会新增点或框提示 API；当前任务仍是文本提示图像分割。

仓库提供两组固定预设：`image-vision-linear-{precision}-v1` 仅量化视觉线性权重；
`image-full-linear-{precision}-v1` 覆盖四模块的 348 个目标矩阵，仍保留上述浮点例外。
GGUF 量化存储权重；共享图主要保留 F32 激活缓冲。显式 CUDA F16 计算模式下，
图像卷积直接生成 F16 展开列，交给既有 F16 输入、F32 输出的矩阵 kernel，减少
临时内存。其他原生 GPU kernel 可按后端算术策略
降低矩阵操作数 staging 精度。视频跟踪状态不属于这些图像 profile。

两组预设分别记录验证结果，具体支持状态见[模型列表](../MODEL_ZOO_zh.md)。
使用 `--quantize-modules` 的自定义组合采用 `image-modules-linear-{precision}-v1`，
需要独立验收完整配置，并检查应用数据。即使选齐四个模块，也不能继承预设的
验证结论。它不能与 `--storage-profile` 同时使用。

## Profile 与存储取舍

GGUF schema 3 提供四种视觉专用 profile。名称和张量分配是精确契约；转换时应显式指定对应 profile。
全模块及自定义模块 profile 使用 SAM schema 4，仍采用 GGUF v3 容器，并在文件中保存模块选择。模块与精度不能在加载时任意改写，需要重新从原始 checkpoint 转换。

| 精度 | 视觉预设的存储 profile | 视觉预设量化的 ViT 线性权重 | 视觉预设 GGUF 大小 | 相对示例 F32 的缩小比例 | 全模块预设 GGUF 大小 |
| --- | --- | --- | ---: | ---: | ---: |
| Q8_0 | `image-vision-linear-q8_0-v1` | 128 个矩阵使用 Q8_0 | 2.07 GB | 约 38.7% | 1.10 GB |
| Q6_K | `image-vision-linear-q6_k-v1` | 96 个矩阵使用 Q6_K；32 个 MLP `lin2` 矩阵使用 Q8_0 | 2.00 GB | 约 40.8% | 0.95 GB |
| Q5_K | `image-vision-linear-q5_k-v1` | 96 个矩阵使用 Q5_K；32 个 MLP `lin2` 矩阵使用 Q8_0 | 1.96 GB | 约 42.0% | 0.86 GB |
| Q4_K | `image-vision-linear-q4_k-v1` | 96 个矩阵使用 Q4_K；32 个 MLP `lin2` 矩阵使用 Q8_0 | 1.92 GB | 约 43.0% | 0.79 GB |

Q8_0 使用 8 位 block 表示；Q6_K、Q5_K、Q4_K 每个值名义上分别使用 6、5、4 位，并包含 block scale 和元数据。上表文件大小来自同一份 SAM 3 图像 checkpoint 的转换样例；F32 GGUF 约 3.37 GB。实际大小取决于 checkpoint 和容器元数据。这些是混合存储模型，并非全部参数都量化。视觉预设中 ViT 线性权重以外的所有 tensor 保留 F32，包括完整文本编码器；全模块预设按前述组件范围覆盖更多线性权重，并保留浮点例外。
K block 要求 256 元素行宽。选择 `vision` 时，32 个 MLP `lin2` 矩阵的 canonical `ne[0]=4736`，因此固定对其使用 Q8_0 fallback；该分配不可配置。不含 `vision` 的自定义组合没有这些 fallback 矩阵。

优先保留更多权重精度时可选择 Q8_0；Q6_K、Q5_K 可进一步缩小文件；Q4_K 在这些样例中最小。低位宽不保证推理更快。部署前应使用目标图像、提示词和 backend 比较输出质量与内存表现。

## 验证范围

四个精确的 `image-vision-linear-*` profile 已在仓库固定图像集上通过输出行为验证，
覆盖 CPU（有无 BLAS）和 Metal。四个 `image-full-linear-*` 全模块预设也已通过
同一图像集的 CPU（启用 BLAS）和 Metal 输出质量检查。两组预设在 Linux x86_64
RTX 4090 CUDA 上也已通过七 case 原始模型输出质量验收。这一结论仅适用于有限
图像集，不是数据集级准确率保证。输出检查关注最终检测、掩码、分数和边框；
通过检查不表示中间 tensor 与原始 checkpoint 完全一致，量化会改变中间数值。
不同精度使用各自的验证标准，而不是对所有格式共用一个容差。

输出质量验收检查最终掩码、分数和边框。完整部署验收还要求算术正确，并在声明的工作负载上测得实际收益。中间张量的相对 L2 和最大绝对误差单独报告，供诊断和选型参考，超出张量保真容差不会直接判定量化版本不合格。分词、输入变换、形状、有限数值及后端执行仍需正确。当前固定语料验收使用以下输出容差：

| 精度 | 掩码 IoU 下限 | 分数绝对偏差上限 | 框坐标偏差上限（对应图像边长的比例） |
| --- | ---: | ---: | ---: |
| Q8_0 | 0.96 | 0.02 | 0.010 |
| Q6_K | 0.94 | 0.03 | 0.015 |
| Q5_K | 0.92 | 0.04 | 0.020 |
| Q4_K | 0.90 | 0.05 | 0.030 |

固定验收的检出阈值为 0.5，检查高置信对象是否遗漏和低置信 query 是否新增检出，阈值附近的变化单独报告。[可视化样例](visual-examples_zh.md)按 0.2 阈值展示所有版本的真实输出和掩码差异；它提供直接对照，不能代替完整语料或应用数据集验证。

[v2 验收策略](plans/20261007-200954-precision-acceptance-gates-v2.md)增加了
分精度质量预算、ranked COCO mask AP、图像级置信界和对象一对一匹配。
完整配置分别验收算术、任务质量和具体工作负载的性能。v2 单独记录结果，
保留上述固定语料的原有结论。

绝对质量以原始 F32 checkpoint 为参考。压缩缓存还需与相同权重、计算模式和
后端下的 F32 缓存比较增量质量，整个配置仍须满足绝对预算。最终评估至少使用
1,024 张未使用图像、2,000 次成对图像 bootstrap，检查负提示，并要求受保护
对象零漏检。七个固定回归用例没有数据集尾部额度。算子分量通过或缓存载荷
变小，均不能替代完整配置验收和端到端峰值内存测量。

一份独立冻结的[短 F32 点积 CUDA 配方](plans/20261009-172831-cuda-f32-short-dot-candidate.md)
保留视觉权重 F32，将文本、融合和解码的 220 个线性权重量化为 Q8_0，并使用
F32 计算与 F32 特征缓存。其已记录的 RTX 4090 结果通过七次调用的完整活跃图
算术检查和 1,024 张独立图像的最终质量门槛。相同 checkpoint 的 F32 父配置
配对测量显示，整图和换提示词推理的 **GPU 进程峰值显存降低 19.27%**。
两者 P50 延迟分别改善 3.32% 和 7.43%，未达到 10% 延迟收益门槛；重复结果
推理没有收益标签。结论只适用于该模型、CUDA 构建和工作负载，其他自定义
分配、缓存配方、GPU、后端和视频需单独举证。

[混合缓存实验](plans/20261009-223907-shortdot-mixed-cache-final-acceptance.md)
在同一 RTX 4090 CUDA 构建上分别验收了两份实验性缓存工具配方。两者都使用
F32 计算，将 FPN 0/1 压缩为 Q8_0，保留 F32 检测特征；完整活跃图算术、固定
回归，以及 **1,024 张未使用图像 / 4,729 个提示词**的绝对和缓存增量质量均通过。
适用的总体检查和 55 个类别检查通过，另 25 个类别覆盖不足，不能宣称单类已验证。

每份配方均在八个预先选定用例上完成 96 次独立进程测量，与**相同权重的
F32 缓存**比较：

| 混合缓存的权重 | 整图 P50 延迟降低 | 换提示 P50 延迟降低 |
| --- | ---: | ---: |
| F32 | 11.08% | 19.66% |
| 上述文本／融合／解码 Q8_0 自定义配方 | 12.38% | 21.23% |

两者仅在这两种工作负载获得延迟收益标签。GPU 进程峰值显存分别增加 0.13% 和
0.16%；RSS 峰值分别降低 14.15% 和 14.35%，未达到 15% 主机内存门槛。因此
均未获得内存或延迟／显存联合收益标签，重复结果推理也没有收益标签。
Q8 父配方单独测得的 19.27% 显存收益不能继承到这两次比较中。以上仍为实验性
缓存工具的验收结果，公共模型加载继续使用现有缓存精度。

Linux v2 工作流使用 `tools/maintenance/freeze_precision_campaign.py` 在评估前
绑定配置和输入，使用 `tools/validation/export_precision_outputs.py` 与
`tools/validation/evaluate_precision.py` 评估 ranked 质量，并通过
`tools/benchmark/benchmark_precision.py` 配对测量延迟和内存。编解码及同操作数
算子检查提供单独的算术证据，要求见[模型验证](validation_zh.md)和验收策略。
这些原生运行工具依赖 Linux 动态库与进程检查接口，尚未实现 Metal 主机接入。

策略中的 BF16、W8A8 和 FP8 属于研究目标，不代表新增了相应图像推理模式。
旧的 schema-3 `image-linear-*` 也量化文本编码器线性权重，目前仅用于诊断；
它不同于 schema-4 全模块预设，不覆盖融合与解码线性权重。这些图像 profile
不支持量化视频。

## 转换 checkpoint

使用 Python 3.12 和锁定的参考环境依赖。将 `sam3_weights_dir` 设置为包含原始 `sam3.pt` checkpoint 与 BPE 文件的目录。输出必须是新文件；转换器会同时写出 GGUF manifest，并拒绝覆盖已有文件。

Linux x86_64 的 CUDA 参考验证使用 [requirements-linux-cuda.lock](../tools/requirements-linux-cuda.lock) 替换安装命令中的 `tools/requirements.lock`。

```sh
python3.12 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r tools/requirements.lock
sam3_weights_dir=/absolute/path/to/sam3

.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task image --precision q8_0 \
  --storage-profile image-vision-linear-q8_0-v1 \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-image-vision-q8_0.gguf
```

Q8_0 使用锁定的 `gguf==0.19.0` writer，不需要 native helper。K 格式需要先使用仓库锁定的 GGML 源码构建行量化 helper，并将 helper 的绝对路径传给转换器。以下示例转换 Q6_K：

```sh
cmake -S . -B build/quant-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DSAM_BUILD_EXAMPLES=ON
cmake --build build/quant-cpu --target sam_quantize_rows --parallel

.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task image --precision q6_k \
  --storage-profile image-vision-linear-q6_k-v1 \
  --quantizer "$PWD/build/quant-cpu/examples/sam_quantize_rows" \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-image-vision-q6_k.gguf
```

转换 Q5_K 或 Q4_K 时，将精度分别改为 `q5_k` 或 `q4_k`，并搭配
`image-vision-linear-q5_k-v1` 或 `image-vision-linear-q4_k-v1`。helper 必须来自锁定的 GGML revision，且需使用绝对可执行文件路径；转换器会检查其身份。profile 必须与精度和 image task 匹配。省略 `--storage-profile` 时会选择旧的宽范围诊断 profile；要使用视觉专用分配，必须显式传入 profile。

转换全模块预设时，改用 `--storage-profile image-full-linear-q6_k-v1`。只量化文本与解码部分的自定义示例如下：

```sh
.venv-reference/bin/python tools/convert/convert_sam3.py \
  --task image --precision q6_k --quantize-modules text,decoder \
  --quantizer "$PWD/build/quant-cpu/examples/sam_quantize_rows" \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-image-text-decoder-q6_k.gguf
```

模块名区分大小写，重复或未知模块会报错；命令行可以调整模块顺序，文件中会保存为规范顺序。旁置 manifest 列出每个张量所属模块、实际存储类型和量化或保留浮点的原因。加载后的 schema 4 模型通过 `ModelInfo::quantization_modules` 报告选择。

自定义组合可用 `tools/validation/validate_image.py --allow-custom-quantization` 做单独的参考对照，报告会保持诊断标识。对照方法见[模型验证](validation_zh.md)。

## 运行推理

使用 CPU、Metal 或 CUDA 构建中的 `sam_image` 可执行文件；输出目录必须不存在。例如：

```sh
build/cpu/examples/sam_image \
  --model models/sam3-image-vision-q8_0.gguf \
  --image image.jpg --text truck --score-threshold 0.2 \
  --output outputs/truck-q8 --backend cpu
```

使用对应构建，传入 `--backend metal` 或 `--backend cuda` 可显式选择 GPU 后端。
CUDA 的 `--cuda-device N` 表示可见设备序号。当前量化模型的 `auto` 选择 CPU。

## Backend 行为

GGUF 精度说明权重的存储方式，不等同于端到端算术精度。CPU 会保持 packed
量化权重常驻，并在共享计算图的 `MUL_MAT` 前创建临时 F32 cast；这部分工作区
会影响峰值内存。`ModelInfo` 将该策略报告为 `ggml-quantized-weights-f32-v1`。
Metal 使用带 half staging 的原生量化 kernel，并报告 `ggml-quantized-native-v1`。
CUDA 同样保持 packed 权重常驻，在已验证设备的原生 MMVQ/MMQ 中使用 RHS Q8_1
staging，报告 `ggml-quantized-cuda-native-v1`，并拒绝 CPU/Metal/BLAS 图计算回退。
各后端分别验收 profile，部署前应核对支持表。
格式契约见[GGUF schema 说明](gguf.md)，支持范围见[模型列表](../MODEL_ZOO_zh.md)，
硬件相关的实测图像性能见[性能测试](../BENCHMARK_zh.md)。
