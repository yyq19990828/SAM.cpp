# 模型效果可视化样例

[English](visual-examples.md) · [量化指南](quantization_zh.md) · [模型与精度](../MODEL_ZOO_zh.md)

本页展示当前 SAM 3 文本提示图像分割的实际输出，比较 F32、混合 F16/F32，以及视觉和全模块 Q8_0、Q6_K、Q5_K、Q4_K。每个样例都使用同一份解码像素、提示词和 **0.2 分数阈值**，分别展示 CPU（启用 BLAS）和 Metal。视觉组仅量化 ViT 线性权重；全模块组覆盖视觉、文本、融合及解码部分的目标线性权重，均保留浮点例外。后续模型适配器和量化范围需要各自的样例与验证。

量化版本主要看最终检出、掩码、分数和边框质量，张量误差单独供参考。下面的 IoU 衡量与官方原始 checkpoint 的 FP32 输出一致程度，不是对人工标注的准确率。样例来自有限图片，不能代替应用数据集评估。

筛选阈值越低，保留的候选通常越多，也更容易出现重复或误检。项目 API 和 CLI 的默认值仍为 0.5；本页样例显式使用 0.2，原有 0.5 验收记录没有改写。

本页复用已验证的网络预测，按 0.2 重新执行各自的官方参考与 C++ 后处理；改变筛选阈值不会改变网络预测张量。

## 如何看图

每组对照图按 F32、F16、Q8_0、Q6_K、Q5_K、Q4_K 排列。相同候选在各版本中使用相同颜色，填色是掩码，矩形是预测框。图中的 `q` 编号是图像检测 query 编号，用于区分候选，下方同色图例列出其置信度（0 至 1，显示三位小数，筛选使用未四舍五入的原始分数）。置信度不是掩码 IoU，也不是有人工标注保证的准确率。图下同时列出候选数量及该样例的最小掩码 IoU；无对应对象时显示 `n/a`。差异图中红色表示比参考新增的像素，蓝色表示缺失的像素，灰色表示掩码范围。为看清细小边界，差异颜色在原始分辨率上扩张 3 像素；IoU 和像素计数均用扩张前的原始二值掩码计算。

## 输出对照

下表汇总本页已匹配候选的最差值，检出集合相同的样例数单列；无对应对象时不计入 IoU。全模块 Q5_K、Q4_K 在 0.2 下的车轮候选减少，不能只看已匹配对象的 IoU 忽略这个变化。固定 0.5 阈值验收和本页 0.2 展示是不同的检查。

| 范围 | 版本 | 后端 | 检出集合相同样例 | 最小掩码 IoU | 最大分数绝对偏差 | 最大框坐标偏差（图像边长百分比） |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 基线 | F32 | cpu | 7/7 | 1.000000 | 0.000002 | 0.000024 |
| 基线 | F16 | cpu | 7/7 | 1.000000 | 0.000134 | 0.000325 |
| 视觉 | Q8_0 | cpu | 7/7 | 0.998361 | 0.000706 | 0.005797 |
| 视觉 | Q6_K | cpu | 7/7 | 0.997051 | 0.006857 | 0.015873 |
| 视觉 | Q5_K | cpu | 7/7 | 0.993781 | 0.010081 | 0.041676 |
| 视觉 | Q4_K | cpu | 7/7 | 0.978615 | 0.020008 | 0.069677 |
| 基线 | F32 | metal | 7/7 | 1.000000 | 0.000002 | 0.000013 |
| 基线 | F16 | metal | 7/7 | 1.000000 | 0.000133 | 0.000315 |
| 视觉 | Q8_0 | metal | 7/7 | 0.998233 | 0.000868 | 0.005925 |
| 视觉 | Q6_K | metal | 7/7 | 0.996725 | 0.006806 | 0.015837 |
| 视觉 | Q5_K | metal | 7/7 | 0.993781 | 0.010150 | 0.041428 |
| 视觉 | Q4_K | metal | 7/7 | 0.978615 | 0.020226 | 0.068802 |
| 全模块 | Q8_0 | cpu | 7/7 | 0.996728 | 0.001712 | 0.037441 |
| 全模块 | Q6_K | cpu | 7/7 | 0.996395 | 0.019934 | 0.106496 |
| 全模块 | Q5_K | cpu | 6/7 | 0.993199 | 0.015972 | 0.163469 |
| 全模块 | Q4_K | cpu | 6/7 | 0.969362 | 0.019633 | 0.232232 |
| 全模块 | Q8_0 | metal | 7/7 | 0.996728 | 0.002270 | 0.037489 |
| 全模块 | Q6_K | metal | 7/7 | 0.997050 | 0.019798 | 0.106572 |
| 全模块 | Q5_K | metal | 6/7 | 0.993199 | 0.015985 | 0.163107 |
| 全模块 | Q4_K | metal | 6/7 | 0.969965 | 0.019353 | 0.239614 |

<details>
<summary>张量误差，仅供参考</summary>

下表为量化版本在这些样例上的最大中间张量相对 L2，计算式为 `||actual - reference||₂ / ||reference||₂`。统计包括全部 query，包含未选中的背景 query；误差不会直接决定量化版本是否合格，也不随分数筛选阈值变化。

| 范围 | 版本 | 后端 | 最大相对 L2 | 对应张量 |
| --- | --- | --- | ---: | --- |
| 视觉 | Q8_0 | cpu | 0.100626 | `mask_logits` |
| 视觉 | Q6_K | cpu | 0.269654 | `mask_logits` |
| 视觉 | Q5_K | cpu | 0.269392 | `mask_logits` |
| 视觉 | Q4_K | cpu | 0.304774 | `mask_logits` |
| 视觉 | Q8_0 | metal | 0.106810 | `mask_logits` |
| 视觉 | Q6_K | metal | 0.270003 | `mask_logits` |
| 视觉 | Q5_K | metal | 0.270525 | `mask_logits` |
| 视觉 | Q4_K | metal | 0.305778 | `mask_logits` |
| 全模块 | Q8_0 | cpu | 0.097067 | `mask_logits` |
| 全模块 | Q6_K | cpu | 0.275067 | `mask_logits` |
| 全模块 | Q5_K | cpu | 0.335335 | `mask_logits` |
| 全模块 | Q4_K | cpu | 0.413463 | `text_features` |
| 全模块 | Q8_0 | metal | 0.099643 | `mask_logits` |
| 全模块 | Q6_K | metal | 0.275624 | `mask_logits` |
| 全模块 | Q5_K | metal | 0.334249 | `mask_logits` |
| 全模块 | Q4_K | metal | 0.413966 | `text_features` |

</details>

## 卡车，`truck`

单个对象的整体掩码。

![输入与官方 FP32 参考，truck-truck](assets/visual-examples/sam3-cpu/truck-truck-reference.jpg)

![CPU 六版本实际输出，truck-truck](assets/visual-examples/sam3-cpu/truck-truck-comparison.jpg)

![CPU 全模块量化与基线，truck-truck](assets/visual-examples/sam3-full-cpu/truck-truck-comparison.jpg)

![Metal 六版本实际输出，truck-truck](assets/visual-examples/sam3-metal/truck-truck-comparison.jpg)

![Metal 全模块量化与基线，truck-truck](assets/visual-examples/sam3-full-metal/truck-truck-comparison.jpg)

<details>
<summary>查看 CPU / Metal 掩码差异</summary>

![CPU 掩码差异，truck-truck](assets/visual-examples/sam3-cpu/truck-truck-differences.png)

![CPU 全模块掩码差异，truck-truck](assets/visual-examples/sam3-full-cpu/truck-truck-differences.png)

![Metal 掩码差异，truck-truck](assets/visual-examples/sam3-metal/truck-truck-differences.png)

![Metal 全模块掩码差异，truck-truck](assets/visual-examples/sam3-full-metal/truck-truck-differences.png)

</details>

## 车轮，`wheel`

参考、基线和视觉预设在 0.2 下保留 7 个候选，全模块 Q8_0/Q6_K 保留 7 个，Q5_K 保留 6 个，Q4_K 保留 4 个。候选中有重叠和较低置信输出，这不是图中真实车轮数量。下表补充全模块各版本对参考候选的置信度，低于阈值者标为“未选”。

CPU 置信度：

| 候选 | 官方 FP32 参考 | 全模块 Q8_0 | 全模块 Q6_K | 全模块 Q5_K | 全模块 Q4_K |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q42` | 0.266 | 0.266 | 0.246 | 0.156（未选） | 0.152（未选） |
| `q44` | 0.952 | 0.952 | 0.952 | 0.951 | 0.947 |
| `q51` | 0.950 | 0.950 | 0.949 | 0.949 | 0.943 |
| `q92` | 0.230 | 0.228 | 0.218 | 0.214 | 0.139（未选） |
| `q116` | 0.889 | 0.888 | 0.889 | 0.881 | 0.869 |
| `q179` | 0.906 | 0.906 | 0.905 | 0.907 | 0.895 |
| `q184` | 0.241 | 0.239 | 0.228 | 0.238 | 0.154（未选） |

Metal 置信度：

| 候选 | 官方 FP32 参考 | 全模块 Q8_0 | 全模块 Q6_K | 全模块 Q5_K | 全模块 Q4_K |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q42` | 0.266 | 0.265 | 0.247 | 0.157（未选） | 0.146（未选） |
| `q44` | 0.952 | 0.952 | 0.952 | 0.951 | 0.947 |
| `q51` | 0.950 | 0.950 | 0.949 | 0.949 | 0.943 |
| `q92` | 0.230 | 0.228 | 0.218 | 0.214 | 0.138（未选） |
| `q116` | 0.889 | 0.888 | 0.889 | 0.881 | 0.869 |
| `q179` | 0.906 | 0.906 | 0.905 | 0.907 | 0.895 |
| `q184` | 0.241 | 0.239 | 0.228 | 0.238 | 0.154（未选） |

![输入与官方 FP32 参考，truck-wheel](assets/visual-examples/sam3-cpu/truck-wheel-reference.jpg)

![CPU 六版本实际输出，truck-wheel](assets/visual-examples/sam3-cpu/truck-wheel-comparison.jpg)

![CPU 全模块量化与基线，truck-wheel](assets/visual-examples/sam3-full-cpu/truck-wheel-comparison.jpg)

![Metal 六版本实际输出，truck-wheel](assets/visual-examples/sam3-metal/truck-wheel-comparison.jpg)

![Metal 全模块量化与基线，truck-wheel](assets/visual-examples/sam3-full-metal/truck-wheel-comparison.jpg)

<details>
<summary>查看 CPU / Metal 掩码差异</summary>

![CPU 掩码差异，truck-wheel](assets/visual-examples/sam3-cpu/truck-wheel-differences.png)

![CPU 全模块掩码差异，truck-wheel](assets/visual-examples/sam3-full-cpu/truck-wheel-differences.png)

![Metal 掩码差异，truck-wheel](assets/visual-examples/sam3-metal/truck-wheel-differences.png)

![Metal 全模块掩码差异，truck-wheel](assets/visual-examples/sam3-full-metal/truck-wheel-differences.png)

</details>

## 卡车图中的负提示，`purple elephant`

参考模型和各版本均无检出。

![输入与官方 FP32 参考，truck-purple-elephant](assets/visual-examples/sam3-cpu/truck-purple-elephant-reference.jpg)

![CPU 六版本实际输出，truck-purple-elephant](assets/visual-examples/sam3-cpu/truck-purple-elephant-comparison.jpg)

![CPU 全模块量化与基线，truck-purple-elephant](assets/visual-examples/sam3-full-cpu/truck-purple-elephant-comparison.jpg)

![Metal 六版本实际输出，truck-purple-elephant](assets/visual-examples/sam3-metal/truck-purple-elephant-comparison.jpg)

![Metal 全模块量化与基线，truck-purple-elephant](assets/visual-examples/sam3-full-metal/truck-purple-elephant-comparison.jpg)

## 杂货图，`fruit`

参考模型和各版本在 0.2 下均无检出。没有人工标注时，这只能说明输出与参考一致，不能据此评价水果识别准确率。

![输入与官方 FP32 参考，groceries-fruit](assets/visual-examples/sam3-cpu/groceries-fruit-reference.jpg)

![CPU 六版本实际输出，groceries-fruit](assets/visual-examples/sam3-cpu/groceries-fruit-comparison.jpg)

![CPU 全模块量化与基线，groceries-fruit](assets/visual-examples/sam3-full-cpu/groceries-fruit-comparison.jpg)

![Metal 六版本实际输出，groceries-fruit](assets/visual-examples/sam3-metal/groceries-fruit-comparison.jpg)

![Metal 全模块量化与基线，groceries-fruit](assets/visual-examples/sam3-full-metal/groceries-fruit-comparison.jpg)

## 杂货图，`bottle`

参考模型和各版本在 0.2 下均无检出；与参考一致不能证明模型已识别瓶子。

![输入与官方 FP32 参考，groceries-bottle](assets/visual-examples/sam3-cpu/groceries-bottle-reference.jpg)

![CPU 六版本实际输出，groceries-bottle](assets/visual-examples/sam3-cpu/groceries-bottle-comparison.jpg)

![CPU 全模块量化与基线，groceries-bottle](assets/visual-examples/sam3-full-cpu/groceries-bottle-comparison.jpg)

![Metal 六版本实际输出，groceries-bottle](assets/visual-examples/sam3-metal/groceries-bottle-comparison.jpg)

![Metal 全模块量化与基线，groceries-bottle](assets/visual-examples/sam3-full-metal/groceries-bottle-comparison.jpg)

## 杂货图中的负提示，`purple elephant`

参考模型和各版本均无检出。

![输入与官方 FP32 参考，groceries-purple-elephant](assets/visual-examples/sam3-cpu/groceries-purple-elephant-reference.jpg)

![CPU 六版本实际输出，groceries-purple-elephant](assets/visual-examples/sam3-cpu/groceries-purple-elephant-comparison.jpg)

![CPU 全模块量化与基线，groceries-purple-elephant](assets/visual-examples/sam3-full-cpu/groceries-purple-elephant-comparison.jpg)

![Metal 六版本实际输出，groceries-purple-elephant](assets/visual-examples/sam3-metal/groceries-purple-elephant-comparison.jpg)

![Metal 全模块量化与基线，groceries-purple-elephant](assets/visual-examples/sam3-full-metal/groceries-purple-elephant-comparison.jpg)

## 裁剪缩放后的卡车，`truck`

裁剪原图后缩放至 640 × 384，再执行分割。

![输入与官方 FP32 参考，truck-crop-resize-truck](assets/visual-examples/sam3-cpu/truck-crop-resize-truck-reference.jpg)

![CPU 六版本实际输出，truck-crop-resize-truck](assets/visual-examples/sam3-cpu/truck-crop-resize-truck-comparison.jpg)

![CPU 全模块量化与基线，truck-crop-resize-truck](assets/visual-examples/sam3-full-cpu/truck-crop-resize-truck-comparison.jpg)

![Metal 六版本实际输出，truck-crop-resize-truck](assets/visual-examples/sam3-metal/truck-crop-resize-truck-comparison.jpg)

![Metal 全模块量化与基线，truck-crop-resize-truck](assets/visual-examples/sam3-full-metal/truck-crop-resize-truck-comparison.jpg)

<details>
<summary>查看 CPU / Metal 掩码差异</summary>

![CPU 掩码差异，truck-crop-resize-truck](assets/visual-examples/sam3-cpu/truck-crop-resize-truck-differences.png)

![CPU 全模块掩码差异，truck-crop-resize-truck](assets/visual-examples/sam3-full-cpu/truck-crop-resize-truck-differences.png)

![Metal 掩码差异，truck-crop-resize-truck](assets/visual-examples/sam3-metal/truck-crop-resize-truck-differences.png)

![Metal 全模块掩码差异，truck-crop-resize-truck](assets/visual-examples/sam3-full-metal/truck-crop-resize-truck-differences.png)

</details>

## 使用自己的图片

用相同图片、提示词和阈值运行各模型，替换下列路径和模型文件即可。输出目录必须为新目录。

```sh
build/cpu/examples/sam_image \
  --model models/sam3-image-vision-q8_0.gguf \
  --image image.jpg --text truck --score-threshold 0.2 \
  --backend cpu --output outputs/truck-q8-threshold-02
```

已有参考验证输出时，可用 `tools/render_image_comparison.py` 从真实掩码生成同样的对照图。该工具要求带校验记录的参考与比较目录，检查图片、结果和掩码身份，拒绝覆盖输出。各目录必须使用相同阈值和参考；一次只比较一个后端。

```sh
.venv-reference/bin/python tools/render_image_comparison.py \
  --reference models/reference/sam3-f32 \
  --comparison F32=build/image-validation-f32 \
  --comparison Q8_0=build/image-validation-q8 \
  --output outputs/model-comparison
```

具体生成方法和数值核对记录保存在[对应计划](plans/20261004-012810-quantization-visual-examples.md)。

## 图片来源与许可

原始图片来自 Meta 的[官方 SAM 3 示例图片](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40/assets/images)。图片像素及其派生对照图遵循随仓库保留的[SAM 许可](../licenses/SAM-model-license.txt)，独立于项目代码的 MIT 许可。掩码来自真实 C++ 输出，参考结果来自官方原始模型。
