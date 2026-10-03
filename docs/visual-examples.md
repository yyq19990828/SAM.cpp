# Visual model examples

[中文](visual-examples_zh.md) · [Quantization](quantization.md) · [Models](../MODEL_ZOO.md)

These examples show actual SAM 3 text-prompted image segmentation with F32, mixed F16/F32 and vision/full-component Q8_0, Q6_K, Q5_K and Q4_K weights. Every configuration uses identical decoded pixels, prompts and a **0.2 score threshold**, shown separately for CPU with BLAS and Metal. Vision presets quantize ViT linears; full presets cover target vision, text, fusion and decoder linears, retaining floating-point exceptions. Future model adapters and quantization allocations need their own examples and validation.

Quantized quality is assessed primarily through detections, masks, scores and boxes; tensor errors are reported separately. The IoU below measures agreement with the official original-checkpoint FP32 output, not accuracy against human annotations. These few images do not replace evaluation on application data.

Lower selection thresholds generally retain more candidates and can introduce duplicates or false positives. The API and CLI default remains 0.5; these examples explicitly use 0.2. Existing 0.5 acceptance records are preserved.

These examples reuse validated network predictions and reapply the respective official-reference and C++ postprocessors at 0.2. Changing the selection threshold does not change the prediction tensors.

## Reading the images

Comparison panels are ordered F32, F16, Q8_0, Q6_K, Q5_K and Q4_K. Matching candidates have the same colors; fills show masks and rectangles show predicted boxes. Image detection-query `q` IDs correspond to the color-keyed confidence legends below each panel (0 to 1, shown to three decimal places; selection uses the original unrounded scores). Confidence is distinct from mask IoU and is not ground-truth accuracy. Captions also report candidate counts and minimum mask IoU for that example, or `n/a` when no objects match. Difference images use red for pixels added relative to the reference, blue for missing pixels and gray for mask regions. Difference colors expand by 3 pixels at the original resolution to expose thin boundaries; IoU and pixel counts use the undilated binary masks.

## Output comparison

The table reports worst values over matched candidates, with selected-query-set agreement shown separately. Unmatched or absent objects do not contribute to IoU. Full Q5_K and Q4_K retain fewer wheel candidates at 0.2, a change that matched-object IoU alone cannot reveal. Frozen acceptance at 0.5 is separate from these 0.2 examples.

| Scope | Configuration | Backend | Examples with matching query sets | Minimum mask IoU | Maximum absolute score error | Maximum box-coordinate error (% of image dimension) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Baseline | F32 | cpu | 7/7 | 1.000000 | 0.000002 | 0.000024 |
| Baseline | F16 | cpu | 7/7 | 1.000000 | 0.000134 | 0.000325 |
| Vision | Q8_0 | cpu | 7/7 | 0.998361 | 0.000706 | 0.005797 |
| Vision | Q6_K | cpu | 7/7 | 0.997051 | 0.006857 | 0.015873 |
| Vision | Q5_K | cpu | 7/7 | 0.993781 | 0.010081 | 0.041676 |
| Vision | Q4_K | cpu | 7/7 | 0.978615 | 0.020008 | 0.069677 |
| Baseline | F32 | metal | 7/7 | 1.000000 | 0.000002 | 0.000013 |
| Baseline | F16 | metal | 7/7 | 1.000000 | 0.000133 | 0.000315 |
| Vision | Q8_0 | metal | 7/7 | 0.998233 | 0.000868 | 0.005925 |
| Vision | Q6_K | metal | 7/7 | 0.996725 | 0.006806 | 0.015837 |
| Vision | Q5_K | metal | 7/7 | 0.993781 | 0.010150 | 0.041428 |
| Vision | Q4_K | metal | 7/7 | 0.978615 | 0.020226 | 0.068802 |
| Full | Q8_0 | cpu | 7/7 | 0.996728 | 0.001712 | 0.037441 |
| Full | Q6_K | cpu | 7/7 | 0.996395 | 0.019934 | 0.106496 |
| Full | Q5_K | cpu | 6/7 | 0.993199 | 0.015972 | 0.163469 |
| Full | Q4_K | cpu | 6/7 | 0.969362 | 0.019633 | 0.232232 |
| Full | Q8_0 | metal | 7/7 | 0.996728 | 0.002270 | 0.037489 |
| Full | Q6_K | metal | 7/7 | 0.997050 | 0.019798 | 0.106572 |
| Full | Q5_K | metal | 6/7 | 0.993199 | 0.015985 | 0.163107 |
| Full | Q4_K | metal | 6/7 | 0.969965 | 0.019353 | 0.239614 |

<details>
<summary>Tensor errors, for reference</summary>

Maximum relative L2 over intermediate tensors in these examples is `||actual - reference||₂ / ||reference||₂`. This includes all queries, including unselected background queries. These errors do not directly determine quantized acceptance and are independent of the selection threshold.

| Scope | Configuration | Backend | Maximum relative L2 | Tensor |
| --- | --- | --- | ---: | --- |
| Vision | Q8_0 | cpu | 0.100626 | `mask_logits` |
| Vision | Q6_K | cpu | 0.269654 | `mask_logits` |
| Vision | Q5_K | cpu | 0.269392 | `mask_logits` |
| Vision | Q4_K | cpu | 0.304774 | `mask_logits` |
| Vision | Q8_0 | metal | 0.106810 | `mask_logits` |
| Vision | Q6_K | metal | 0.270003 | `mask_logits` |
| Vision | Q5_K | metal | 0.270525 | `mask_logits` |
| Vision | Q4_K | metal | 0.305778 | `mask_logits` |
| Full | Q8_0 | cpu | 0.097067 | `mask_logits` |
| Full | Q6_K | cpu | 0.275067 | `mask_logits` |
| Full | Q5_K | cpu | 0.335335 | `mask_logits` |
| Full | Q4_K | cpu | 0.413463 | `text_features` |
| Full | Q8_0 | metal | 0.099643 | `mask_logits` |
| Full | Q6_K | metal | 0.275624 | `mask_logits` |
| Full | Q5_K | metal | 0.334249 | `mask_logits` |
| Full | Q4_K | metal | 0.413966 | `text_features` |

</details>

## Truck, `truck`

The whole-object mask for a single detection.

![Input and official FP32 reference, truck-truck](assets/visual-examples/sam3-cpu/truck-truck-reference.jpg)

![Actual CPU outputs for six configurations, truck-truck](assets/visual-examples/sam3-cpu/truck-truck-comparison.jpg)

![Full-component CPU quantization and baselines, truck-truck](assets/visual-examples/sam3-full-cpu/truck-truck-comparison.jpg)

![Actual Metal outputs for six configurations, truck-truck](assets/visual-examples/sam3-metal/truck-truck-comparison.jpg)

![Full-component Metal quantization and baselines, truck-truck](assets/visual-examples/sam3-full-metal/truck-truck-comparison.jpg)

<details>
<summary>Show CPU / Metal mask differences</summary>

![CPU mask differences, truck-truck](assets/visual-examples/sam3-cpu/truck-truck-differences.png)

![Full-component CPU mask differences, truck-truck](assets/visual-examples/sam3-full-cpu/truck-truck-differences.png)

![Metal mask differences, truck-truck](assets/visual-examples/sam3-metal/truck-truck-differences.png)

![Full-component Metal mask differences, truck-truck](assets/visual-examples/sam3-full-metal/truck-truck-differences.png)

</details>

## Wheels, `wheel`

At 0.2, the reference, baselines and vision presets retain 7 candidates. Full Q8_0/Q6_K retain 7, Q5_K retains 6 and Q4_K retains 4. These include overlapping and low-confidence outputs, rather than a count of physical wheels. The tables show each full configuration's confidence for reference candidates; candidates below the threshold are marked filtered.

CPU confidence:

| Candidate | Official FP32 reference | Full Q8_0 | Full Q6_K | Full Q5_K | Full Q4_K |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q42` | 0.266 | 0.266 | 0.246 | 0.156 (filtered) | 0.152 (filtered) |
| `q44` | 0.952 | 0.952 | 0.952 | 0.951 | 0.947 |
| `q51` | 0.950 | 0.950 | 0.949 | 0.949 | 0.943 |
| `q92` | 0.230 | 0.228 | 0.218 | 0.214 | 0.139 (filtered) |
| `q116` | 0.889 | 0.888 | 0.889 | 0.881 | 0.869 |
| `q179` | 0.906 | 0.906 | 0.905 | 0.907 | 0.895 |
| `q184` | 0.241 | 0.239 | 0.228 | 0.238 | 0.154 (filtered) |

Metal confidence:

| Candidate | Official FP32 reference | Full Q8_0 | Full Q6_K | Full Q5_K | Full Q4_K |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q42` | 0.266 | 0.265 | 0.247 | 0.157 (filtered) | 0.146 (filtered) |
| `q44` | 0.952 | 0.952 | 0.952 | 0.951 | 0.947 |
| `q51` | 0.950 | 0.950 | 0.949 | 0.949 | 0.943 |
| `q92` | 0.230 | 0.228 | 0.218 | 0.214 | 0.138 (filtered) |
| `q116` | 0.889 | 0.888 | 0.889 | 0.881 | 0.869 |
| `q179` | 0.906 | 0.906 | 0.905 | 0.907 | 0.895 |
| `q184` | 0.241 | 0.239 | 0.228 | 0.238 | 0.154 (filtered) |

![Input and official FP32 reference, truck-wheel](assets/visual-examples/sam3-cpu/truck-wheel-reference.jpg)

![Actual CPU outputs for six configurations, truck-wheel](assets/visual-examples/sam3-cpu/truck-wheel-comparison.jpg)

![Full-component CPU quantization and baselines, truck-wheel](assets/visual-examples/sam3-full-cpu/truck-wheel-comparison.jpg)

![Actual Metal outputs for six configurations, truck-wheel](assets/visual-examples/sam3-metal/truck-wheel-comparison.jpg)

![Full-component Metal quantization and baselines, truck-wheel](assets/visual-examples/sam3-full-metal/truck-wheel-comparison.jpg)

<details>
<summary>Show CPU / Metal mask differences</summary>

![CPU mask differences, truck-wheel](assets/visual-examples/sam3-cpu/truck-wheel-differences.png)

![Full-component CPU mask differences, truck-wheel](assets/visual-examples/sam3-full-cpu/truck-wheel-differences.png)

![Metal mask differences, truck-wheel](assets/visual-examples/sam3-metal/truck-wheel-differences.png)

![Full-component Metal mask differences, truck-wheel](assets/visual-examples/sam3-full-metal/truck-wheel-differences.png)

</details>

## Negative prompt on the truck image, `purple elephant`

The reference and all configurations return no detections.

![Input and official FP32 reference, truck-purple-elephant](assets/visual-examples/sam3-cpu/truck-purple-elephant-reference.jpg)

![Actual CPU outputs for six configurations, truck-purple-elephant](assets/visual-examples/sam3-cpu/truck-purple-elephant-comparison.jpg)

![Full-component CPU quantization and baselines, truck-purple-elephant](assets/visual-examples/sam3-full-cpu/truck-purple-elephant-comparison.jpg)

![Actual Metal outputs for six configurations, truck-purple-elephant](assets/visual-examples/sam3-metal/truck-purple-elephant-comparison.jpg)

![Full-component Metal quantization and baselines, truck-purple-elephant](assets/visual-examples/sam3-full-metal/truck-purple-elephant-comparison.jpg)

## Groceries, `fruit`

The reference and all configurations return no detections at 0.2. Without human annotations, this establishes output agreement rather than fruit-recognition accuracy.

![Input and official FP32 reference, groceries-fruit](assets/visual-examples/sam3-cpu/groceries-fruit-reference.jpg)

![Actual CPU outputs for six configurations, groceries-fruit](assets/visual-examples/sam3-cpu/groceries-fruit-comparison.jpg)

![Full-component CPU quantization and baselines, groceries-fruit](assets/visual-examples/sam3-full-cpu/groceries-fruit-comparison.jpg)

![Actual Metal outputs for six configurations, groceries-fruit](assets/visual-examples/sam3-metal/groceries-fruit-comparison.jpg)

![Full-component Metal quantization and baselines, groceries-fruit](assets/visual-examples/sam3-full-metal/groceries-fruit-comparison.jpg)

## Groceries, `bottle`

The reference and all configurations return no detections at 0.2. Agreement with the reference does not establish successful bottle recognition.

![Input and official FP32 reference, groceries-bottle](assets/visual-examples/sam3-cpu/groceries-bottle-reference.jpg)

![Actual CPU outputs for six configurations, groceries-bottle](assets/visual-examples/sam3-cpu/groceries-bottle-comparison.jpg)

![Full-component CPU quantization and baselines, groceries-bottle](assets/visual-examples/sam3-full-cpu/groceries-bottle-comparison.jpg)

![Actual Metal outputs for six configurations, groceries-bottle](assets/visual-examples/sam3-metal/groceries-bottle-comparison.jpg)

![Full-component Metal quantization and baselines, groceries-bottle](assets/visual-examples/sam3-full-metal/groceries-bottle-comparison.jpg)

## Negative prompt on the groceries image, `purple elephant`

The reference and all configurations return no detections.

![Input and official FP32 reference, groceries-purple-elephant](assets/visual-examples/sam3-cpu/groceries-purple-elephant-reference.jpg)

![Actual CPU outputs for six configurations, groceries-purple-elephant](assets/visual-examples/sam3-cpu/groceries-purple-elephant-comparison.jpg)

![Full-component CPU quantization and baselines, groceries-purple-elephant](assets/visual-examples/sam3-full-cpu/groceries-purple-elephant-comparison.jpg)

![Actual Metal outputs for six configurations, groceries-purple-elephant](assets/visual-examples/sam3-metal/groceries-purple-elephant-comparison.jpg)

![Full-component Metal quantization and baselines, groceries-purple-elephant](assets/visual-examples/sam3-full-metal/groceries-purple-elephant-comparison.jpg)

## Cropped and resized truck, `truck`

The input is cropped and resized to 640 × 384 before segmentation.

![Input and official FP32 reference, truck-crop-resize-truck](assets/visual-examples/sam3-cpu/truck-crop-resize-truck-reference.jpg)

![Actual CPU outputs for six configurations, truck-crop-resize-truck](assets/visual-examples/sam3-cpu/truck-crop-resize-truck-comparison.jpg)

![Full-component CPU quantization and baselines, truck-crop-resize-truck](assets/visual-examples/sam3-full-cpu/truck-crop-resize-truck-comparison.jpg)

![Actual Metal outputs for six configurations, truck-crop-resize-truck](assets/visual-examples/sam3-metal/truck-crop-resize-truck-comparison.jpg)

![Full-component Metal quantization and baselines, truck-crop-resize-truck](assets/visual-examples/sam3-full-metal/truck-crop-resize-truck-comparison.jpg)

<details>
<summary>Show CPU / Metal mask differences</summary>

![CPU mask differences, truck-crop-resize-truck](assets/visual-examples/sam3-cpu/truck-crop-resize-truck-differences.png)

![Full-component CPU mask differences, truck-crop-resize-truck](assets/visual-examples/sam3-full-cpu/truck-crop-resize-truck-differences.png)

![Metal mask differences, truck-crop-resize-truck](assets/visual-examples/sam3-metal/truck-crop-resize-truck-differences.png)

![Full-component Metal mask differences, truck-crop-resize-truck](assets/visual-examples/sam3-full-metal/truck-crop-resize-truck-differences.png)

</details>

## Try your own images

Run each model on the same image, prompt and threshold. Replace the executable, image and model paths below. The output directory must be new.

```sh
build/cpu/examples/sam_image \
  --model models/sam3-image-vision-q8_0.gguf \
  --image image.jpg --text truck --score-threshold 0.2 \
  --backend cpu --output outputs/truck-q8-threshold-02
```

For existing reference-validation outputs, `tools/render_image_comparison.py` renders comparison panels from actual masks. It requires reference and comparison bundles with validation receipts, verifies input, result and mask identity, and refuses to overwrite an output. Bundles must use the same threshold and reference; compare one backend per invocation.

```sh
.venv-reference/bin/python tools/render_image_comparison.py \
  --reference models/reference/sam3-f32 \
  --comparison F32=build/image-validation-f32 \
  --comparison Q8_0=build/image-validation-q8 \
  --output outputs/model-comparison
```

Generation details and numerical checks are retained in the [implementation plan](plans/20261004-012810-quantization-visual-examples.md).

## Image sources and license

Source images come from Meta's [official SAM 3 example media](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40/assets/images). Source-media pixels and derived comparison images follow the retained [SAM license](../licenses/SAM-model-license.txt), separately from the project code's MIT license. Masks are actual C++ outputs; reference results use the official original model.
