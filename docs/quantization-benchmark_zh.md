# 在应用图像上评测量化配置

[English](quantization-benchmark.md) · [量化选项](quantization_zh.md)

仓库 COCO 测量用于提供特定数据集、配置和硬件上的参考，不是通用部署门槛。
应用方应在自己的代表性图像和提示词上，将完整配置与原始 SAM 3 checkpoint
或原生 F32 GGUF 比较，自行决定可接受的质量、速度和内存取舍。

`tools/benchmark/quantization_benchmark.py` 分别提供 ranked 输出对比、独立性能测量和精度检查，不要求
v2/v3 质量预算、最低样本数或性能收益。导出目前面向 Linux CPU/CUDA，原始
checkpoint 需要已准备好的 CUDA 参考环境。它没有新增推理 kernel 或公共缓存选择 API。

优先使用[统一量化配置](quantization-config_zh.md)：`export` 和 `inspect-precision`
接受 `--quantization-config`，`performance` 同时接受 `--baseline-config` 和
`--candidate-config`。工具会核对配置与模型的实际权重分配。旧独立参数继续可用，
但不能与配置文件混用。

## 参考模型与样例

原始 `sam3.pt` 参考反映转换、运行时与压缩的共同影响；使用 F32 计算策略及缓存的
原生 F32 GGUF 参考观察相对本运行时的增量变化，无法发现双方共有错误。
两者都是模型生成的参考，不是人工真值。保持一致不能证明生产准确率；有条件时
使用人工标注或抽查，重点覆盖小目标和高代价错误。公开 COCO 报告应把相对 GT 的
AP/mIoU 与参考一致性分开，并说明样本覆盖和调参使用情况。复用开发数据可以支持
比较，不能据此宣称独立留出集结论。

创建 `models/application-cases.json`，例如：

```json
{
  "schema_version": 1,
  "samples": [
    {"id": "camera-one", "image": "camera/frame001.jpg", "prompts": ["person", "forklift"]},
    {"id": "camera-two", "image": "camera/frame002.jpg", "prompts": ["person"]}
  ]
}
```

图片路径相对 `--input-root`。ID 唯一，由小写字母、数字及分隔用连字符组成；同图
提示词须非空、不重复且不含制表符或换行。每次运行保存标准化 RGB PNG，用哈希绑定
相同解码像素。导出及比较均使用新目录。

## 导出和比较

使用 `SAM_BUILD_TOOLS=ON` 构建 probe。示例中的模型须已转换且保留转换 manifest，
使用时替换为实际路径和选项：

```sh
.venv-reference/bin/python tools/benchmark/quantization_benchmark.py export \
  --cases models/application-cases.json --input-root models/application-images \
  --engine native --binary build/cuda/examples/sam_precision_image_probe \
  --model models/sam3-f32.gguf --quantization-config docs/configs/quantization/image-f32-cuda.json \
  --output build/application-f32

.venv-reference/bin/python tools/benchmark/quantization_benchmark.py export \
  --cases models/application-cases.json --input-root models/application-images \
  --engine native --binary build/cuda/examples/sam_precision_image_probe \
  --model models/sam3-custom-q4_k.gguf \
  --quantization-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --output build/application-custom

.venv-reference/bin/python tools/benchmark/quantization_benchmark.py compare \
  --reference build/application-f32 --candidate build/application-custom \
  --output build/application-comparison
```

CUDA 导出另支持实验选项 `--cache f16` 和 `--cache mixed-q8_0`，公共模型加载选项
保持现状。`--compute f16` 选择已有策略，不保证每个算子或量化 kernel 都是 FP16。

原始 checkpoint 参考需按[模型验证](validation_zh.md)准备 CUDA 环境和独立 runtime
源码，并设置 `SAM3_SOURCE_DIR` 和 `sam3_weights_dir`，之后运行：

```sh
.venv-reference-cuda/bin/python tools/benchmark/quantization_benchmark.py export \
  --cases models/application-cases.json --input-root models/application-images \
  --engine original --checkpoint "$sam3_weights_dir/sam3.pt" \
  --sam3-source "$SAM3_SOURCE_DIR" \
  --sam3-runtime-source build/reference-runtime/sam3-cuda \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output build/application-original
```

## 理解测量结果

`report.json`、`report.md` 和 `cases.json` 提供 union-mask IoU、匹配掩码分布、
参考对象缺失、候选新增对象、分数／框变化及逐提示词详情。参考为空的提示与参考
有对象的 union IoU 分开统计。缺失／新增是相对参考模型而言，不能称为 GT 漏检／误检。
匹配对象统计不包含未匹配对象，须结合缺失／新增率阅读。无对象时为 `NO_DATA`，
不能宣称对象准确率完美。报告是描述性统计，没有 bootstrap 置信界、最低样本量规则
或留出集资格结论。

默认检出规则为 `score > 0.5`。`--score-threshold` 当前支持 `[0.5, 1)`，因为 ranked
输出仅保留排名靠前及分数高于 0.5 的掩码，更低阈值需要后续扩展导出格式。
`--matching-iou` 决定空间一对一关联，默认 `0.5`；它是指标定义，不是质量门槛。
比较不同配置时保持相同指标定义。

通过 `--advice models/application-limits.json` 可加载用户建议值：

```json
{
  "missing_reference_rate_max": 0.02,
  "added_candidate_rate_max": 0.03,
  "positive_union_iou_mean_min": 0.90,
  "matched_mask_iou_p05_min": 0.75,
  "score_error_mean_max": 0.05
}
```

这些值只是配置示例，不是仓库推荐门槛。建议结果为 `WITHIN_USER_LIMITS`、
`OUTSIDE_USER_LIMITS`、`INSUFFICIENT_DATA` 或 `NOT_REQUESTED`。有效报告即使超出
建议值也成功退出；缺少或篡改输出、图片／提示词／token 不一致、原始 checkpoint
不同才会报错。

## 独立性能 benchmark

`performance` 直接接收基线／候选 GGUF、计算和缓存选项，不要求质量报告、COCO 标注、
质量档位或 campaign。每张选中的图片至少提供两个不同提示词；取前两个作为主提示和
换提示词工作负载。支持 `--limit N` 或重复 `--case ID` 选择少量图片，不能同时使用。
当前性能 runner 面向 Linux CPU/CUDA，同一次比较使用相同后端和固定 4 个线程。
CUDA 使用可见设备 0，可通过 `CUDA_VISIBLE_DEVICES` 选择物理设备。

```sh
.venv-reference/bin/python tools/benchmark/quantization_benchmark.py performance \
  --cases models/application-cases.json --input-root models/application-images \
  --binary build/cuda/examples/sam_precision_benchmark_probe \
  --baseline-model models/sam3-f32.gguf --candidate-model models/sam3-custom-q4_k.gguf \
  --baseline-config docs/configs/quantization/image-f32-cuda.json \
  --candidate-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --limit 1 --pairs 1 --warmups 0 --iterations 2 --memory-iterations 1 \
  --output build/application-performance

.venv-reference/bin/python tools/benchmark/quantization_benchmark.py summarize-performance \
  --run build/application-performance/run.json --benefit-min-percent 2 \
  --output build/application-performance-summary
```

默认每例运行 3 组 AB/BA/AB 配对。延迟进程每种工作负载预热 5 次、测量 20 次；
内存另起进程，不预热，每种工作负载测量 1 次。模型加载和首次推理单独记录，
不计入稳态延迟；进程峰值内存包含加载和首次推理。`--pairs`、`--warmups`、
`--iterations`、`--memory-iterations` 可显式调整。比如 `--pairs 1 --warmups 0
--iterations 1` 只适合排查流程，不能代表稳定收益。每个子进程超时默认 1,800 秒，
可用 `--timeout` 调整；失败保留日志且不发布完成回执。默认单例共 12 个模型进程，
模型权重只读取，不复制到结果目录。

三个工作负载分别报告：`full_image` 重新编码图片并分割；`changed_prompt` 复用图像
特征、交替提示词；`repeated_result` 命中同提示词结果缓存，不能称为模型推理提速。
延迟给出逐例及汇总 P50/P95、各独立进程配对结果；RSS/GPU 内存使用所有样例／配对
的最大进程峰值之比，不平均峰值比。CUDA 显存按 PID 采样，50 ms 加查询耗时可能漏掉
短峰值；延迟进程不运行内存或图 observer。开始前 GPU 已被其他计算进程占用时拒绝
启动，运行期间发现竞争或观察错误则保留原始数据、标记 `INCONCLUSIVE`。
延迟进程只在前后检查 GPU 竞争，中途短暂出现的竞争可能漏检。

`report_validity=VALID` 表示测量结构及身份可用，`quality=NOT_MEASURED` 与它独立。
候选更慢、内存更多或没有收益，仍然成功生成报告。收益标签只供阅读：默认
`--benefit-min-percent 5`，延迟标签要求汇总 P50 至少减少该比例、P95 不增加，且各配对
均满足；内存标签按相应峰值比例判断。它不替代逐例尾延迟，也不对其他维度作无退化保证。
标签为 `CONSISTENT_REDUCTION`、单配对 `OBSERVED_REDUCTION`、`UNSTABLE`、
`NOT_DEMONSTRATED` 或受污染时 `INCONCLUSIVE`，都不改变退出状态。统计是描述性
观察，不声明显著性或跨硬件收益。离线汇总重验保存的图片与记录哈希，无需重新加载模型。
质量导出／RLE 时间不是推理延迟，特征缓存载荷也不是进程峰值内存。

## 精度配置与执行证据

权重、激活、计算、缓存分别描述不同边界，不能把模型名称当成整图运算精度。
质量与性能报告共用 `requested`、权重清单、计算／缓存策略和 `execution_evidence`。

| 维度 | 配置 | 实际含义与证据 |
| --- | --- | --- |
| 权重 | `weights.precision` 及 `modules` / `storage_profile`，或 schema 2 的 `base_precision` / `module_precisions` | GGUF 可同时有受保护的 F32、各模块 Q 格式和 Q8 回退；检查策略哈希、逐类型张量数量／字节数及保留原因 |
| 激活 | `activation.mode=backend-selected` | 原生暂不支持独立 INT8/FP8/F16 激活开关；后端可内部转换或量化 RHS，研究 probe 不等于可部署配置 |
| 计算 | `compute.mode` | CUDA F16 只给符合条件的浮点矩阵／attention 设置提示，量化矩阵保留自己的分派；F32 也不保证内核所有临时表示都是 F32 |
| 缓存 | `cache.mode` | 图像特征 0/1/2：F32 为 F32/F32/F32，F16 为 F16/F16/F16，mixed-Q8_0 为 Q8_0/Q8_0/F32；不是 LLM KV cache |

权重格式在转换时确定；计算和实验缓存策略在加载时选择。CPU/Metal 在这里仅接入
F32 计算及 F32 缓存，低精度设置需显式 CUDA；实验缓存开关属于 probe，未增加公共
模型加载 API。CPU 加载 F16 权重会提升到 F32，量化矩阵在 CPU matmul 前转换到
临时 F32；文件变小不保证运行内存同比变小。任意逐层混合量化格式、独立激活量化
及统一内核计算精度承诺尚未实现。

无需推理即可检查 GGUF 的存储和策略：

```sh
.venv-reference/bin/python tools/benchmark/quantization_benchmark.py inspect-precision \
  --model models/sam3-custom-q4_k.gguf \
  --quantization-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --output build/application-precision.json
```

该命令核对模型／manifest 哈希、GGUF 张量头与存储清单，`inference_executed=false`。
它读取权重文件，选择 `--backend cuda` 不会启动 CUDA 推理。
运行时 `arithmetic_profile` 只是 producer 报告的策略标识，不是 kernel 内部精度追踪。
需要图级诊断时，在单独的新目录使用 `sam_profile_graph --model ... --image ... --text ...
--backend cuda --cuda-compute f16 --feature-cache mixed-q8_0 --output ...`。
`graphs.json` 保存算子输入／输出类型、源槽位及累加／RHS 提示；这些是图分配边界的
观察，不能证明 kernel 内部乘法／累加类型。该 observer 会影响执行，不用于性能计时。
内核证据没有采集时始终为 `NOT_COLLECTED`。

## 历史结果

v2/v3 策略及依赖质量通过的性能资格流程只在
[归档](../tools/archive/precision_v2_v3/README.md)中保存，原活动入口已移除。
历史结果不改写；已完成的 ranked v2/v3 导出包仍可交给 `compare`，仅核对输入身份、
提示词清单和输出哈希，不重判历史策略或授予独立最终留出集结论。应用包包含并重新
校验标准化图片；历史图片身份使用原回执哈希。比较工具不会重新认证 producer 的
模型、二进制或 kernel 执行。

[本轮实施计划](plans/20261010-151710-performance-and-precision-reporting.md)记录工具、
归档和精度证据的验证范围；没有大规模 GPU 验收或新的硬件收益结论。
