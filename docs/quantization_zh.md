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

数值验收聚焦这两组预设，具体支持状态见[模型列表](../MODEL_ZOO_zh.md)。使用 `--quantize-modules` 的自定义组合采用 `image-modules-linear-{precision}-v1`，需要用户在应用数据上验证；即使选齐四个模块，也保留自定义诊断标签。它不能与 `--storage-profile` 同时使用。

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

量化版本以最终输出质量作为主要验收标准。中间张量的相对 L2 和最大绝对误差单独报告，供诊断和选型参考，超出张量保真容差不会直接判定量化版本不合格。分词、输入变换、形状、有限数值及后端执行仍需正确。当前固定语料验收使用以下输出容差：

| 精度 | 掩码 IoU 下限 | 分数绝对偏差上限 | 框坐标偏差上限（对应图像边长的比例） |
| --- | ---: | ---: | ---: |
| Q8_0 | 0.96 | 0.02 | 0.010 |
| Q6_K | 0.94 | 0.03 | 0.015 |
| Q5_K | 0.92 | 0.04 | 0.020 |
| Q4_K | 0.90 | 0.05 | 0.030 |

固定验收的检出阈值为 0.5，检查高置信对象是否遗漏和低置信 query 是否新增检出，阈值附近的变化单独报告。[可视化样例](visual-examples_zh.md)按 0.2 阈值展示所有版本的真实输出和掩码差异；它提供直接对照，不能代替完整语料或应用数据集验证。

实验性的 [v2 验收方案](plans/20261007-200954-precision-acceptance-gates-v2.md)为
F32/F16/Q8/Q6/Q5/Q4 分别设置质量预算，使用 ranked COCO mask AP、图像级置信界和
对象一对一匹配。缓存压缩同时执行完整配置总预算和缓存增量预算。v2 单独记录结果，
不改写上述固定语料的原有验收结论。
`tools/export_precision_outputs.py` 和 `tools/evaluate_precision.py` 提供 ranked
评估入口；`tools/freeze_precision_campaign.py` 在独立评估前冻结候选。
`tools/benchmark_precision.py` 分开测量延迟和内存收益。
`tools/verify_precision_f16.py` 检查原生 F16 编解码的位模式；
`tools/validate_precision_regression.py` 在原有验证器之外，对七个固定用例执行
零尾部额度的对象匹配检查。这些 v2 Python 原生运行工具当前依赖 Linux 的动态库
与进程内存检查接口，尚未实现 Metal 主机接入。策略中 BF16/W8A8/FP8
的门槛属于研究目标，不代表新增了这些运行时精度支持。

范围更广的 `image-linear-*` family 也会量化文本编码器线性权重，目前仅用于诊断。这些 schema 3 profile 只用于图像；它们不支持量化视频。
旧的 `image-linear-*` 不等同于 schema 4 的全模块预设，前者不覆盖融合与解码线性权重。

## 转换 checkpoint

使用 Python 3.12 和锁定的参考环境依赖。将 `sam3_weights_dir` 设置为包含原始 `sam3.pt` checkpoint 与 BPE 文件的目录。输出必须是新文件；转换器会同时写出 GGUF manifest，并拒绝覆盖已有文件。

Linux x86_64 的 CUDA 参考验证使用 [requirements-linux-cuda.lock](../tools/requirements-linux-cuda.lock) 替换安装命令中的 `tools/requirements.lock`。

```sh
python3.12 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r tools/requirements.lock
sam3_weights_dir=/absolute/path/to/sam3

.venv-reference/bin/python tools/convert_sam3.py \
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

.venv-reference/bin/python tools/convert_sam3.py \
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
.venv-reference/bin/python tools/convert_sam3.py \
  --task image --precision q6_k --quantize-modules text,decoder \
  --quantizer "$PWD/build/quant-cpu/examples/sam_quantize_rows" \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-image-text-decoder-q6_k.gguf
```

模块名区分大小写，重复或未知模块会报错；命令行可以调整模块顺序，文件中会保存为规范顺序。旁置 manifest 列出每个张量所属模块、实际存储类型和量化或保留浮点的原因。加载后的 schema 4 模型通过 `ModelInfo::quantization_modules` 报告选择。

自定义组合可用 `tools/validate_image.py --allow-custom-quantization` 做单独的参考对照，报告会保持诊断标识。对照方法见[模型验证](validation_zh.md)。

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
