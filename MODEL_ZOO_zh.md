# 模型列表

[English](MODEL_ZOO.md) · [性能测试](BENCHMARK_zh.md)

## 支持范围

SAM 3 文本图像分割：CPU/Metal 的 F32/F16，各通过七个参考场景。
前向视频：CPU/Metal 的 F32/hybrid，各通过五场景、216 帧及图像、生命周期、
长会话检查。视频转换默认 hybrid `visual-tracker-f32-v1`；图像转换仍需显式精度。

F16 视频属于诊断配置：Metal 全量比较在 entry23/24 帧候选选择失败，
Meta 加载相同舍入权重也复现该结果。CPU 全量诊断延后，不计通过。
CUDA、Q4/Q8、反向或交互式视频、SAM3.1 及其他模型尚未在此完成支持验收。

## 来源与文件

原始权重：[facebook/sam3](https://huggingface.co/facebook/sam3/tree/3c879f39826c281e95690f02c7821c4de09afae7)，
版本 `3c879f39826c281e95690f02c7821c4de09afae7`。
Meta 代码：[2345a4ad](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40)。
`sam3.pt` SHA-256：`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。
GGUF v3 图像文件使用 schema1，完整视频使用 schema2；侧车清单记录元数据、
源文件和张量哈希。模型访问权限与许可接受分别确认；权重、媒体不进入 Git。
许可见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `sam3-f32.gguf` | 3371139456 | `cb13ecd5012a2fe19b06d840049be6daa6b177b35256af8c4afeb12125352486` |
| `sam3-f16.gguf` | 1797613888 | `66731fa5def347677f78d7422b81979be0f8e2f7ead941db9a466d1cfa715120` |
| `sam3-video-f32.gguf` | 3449345696 | `02513232afca5ba8c174b66c7fc839c67b32590df4b53bdd6df5089a6a546844` |
| `sam3-video-f16.gguf` | 1837925216 | `9c9bc86c81d11a041db10a46d3d1e8ecaa1cbcf6fad683b00901f641746bf32c` |
| `sam3-video-hybrid-v1.gguf` | 2765012640 | `3975b4b1a10b962c6fad5022e2798baa6266aad7dbcc39210b459537208e1a05` |

## 精度约定

| 权重标签 | 磁盘 / Metal 驻留权重 | CPU 驻留权重 |
| --- | --- | --- |
| F32 | 原始 F32 | F32 |
| F16 | 混合 F16/F32 | 保存的 F16 升为 F32 |
| Hybrid | 视觉与跟踪器保留原始 F32，其余混合 | 剩余 F16 升为 F32 |

升格只保留已舍入数值，不能恢复原始 F32。计算图使用 F32 激活和
[指定的 F32 算术](cmake/patches/README.md)，注意力掩码可以是 F16。
没有 BF16 GGUF 权重，也不宣称全流程使用原生 BF16 运算。
运行 VideoSession 时，**三种权重配置都保留**以下边界：

| 视频边界 | 精度 / 表示 |
| --- | --- |
| 归一化 | 每步按 F16 舍入，输出为 F32 缓冲 |
| 跟踪 neck 特征 | 按 BF16 舍入，缓冲仍为 F32 |
| 掩码记忆特征 | BF16 保存，注意力计算前展开为 F32 |
| 对象指针 / 主机掩码 logits | F32，最终二值掩码为 uint8 |

F32 视频标签表示权重，不表示端到端全 F32。图像推理没有跟踪 BF16 状态。
Hybrid 恢复 236 个权重载荷、保留 1228 个基线载荷，比 F16 视频增加约 884 MiB。

## 使用

按[下载与转换说明](docs/models/sam3-details.md#download)准备 Python3.12 和锁定的
参考环境；运行 C++ 推理不需要 Python。

```sh
.venv-reference/bin/python tools/convert_sam3.py --task video \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output models/sam3-video-hybrid-v1.gguf
```

输出必须使用新路径，已有文件会被拒绝。仍可显式选择 F32/F16。
[README](README.md#forward-video-tracking)说明公开 API，
[验证工作流](docs/validation_zh.md)说明基线复用与诊断。
[详细记录](docs/models/sam3-details.md)保留映射、分词器、容器迁移和原始转换证据。

当前 ViT 投影与可选 CPU BLAS 实现已重新通过完整视频矩阵、70 项图像、
六项短会话和两组各 128 次交错推帧检查。四项视频和四项图像性能测量也已通过，
见[性能测试](BENCHMARK_zh.md)和[优化记录](docs/plans/20261003-134534-visual-encoding-profile-and-optimization.md)。
