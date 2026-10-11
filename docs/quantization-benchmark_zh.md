# 在应用图像上评测量化配置

[English](quantization-benchmark.md) · [量化选项](quantization_zh.md)

仓库 COCO 测量用于提供特定数据集、配置和硬件上的参考，不是通用部署门槛。
应用方应在自己的代表性图像和提示词上，将完整配置与原始 SAM 3 checkpoint
或原生 F32 GGUF 比较，自行决定可接受的质量、速度和内存取舍。

`tools/benchmark/quantization_benchmark.py` 提供 ranked 输出对比、独立性能测量、精度检查和可选 CPU 执行开销诊断，不要求
v2/v3 质量预算、最低样本数或性能收益。导出目前面向 Linux CPU/CUDA，原始
checkpoint 需要已准备好的 CUDA 参考环境。它没有新增推理 kernel 或公共缓存选择 API。

优先使用[统一量化配置](quantization-config_zh.md)：`export` 和 `inspect-precision`
及 `execution-cost` 接受 `--quantization-config`，`performance` 同时接受 `--baseline-config` 和
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
| 权重 | `weights.precision` 及 `modules` / `storage_profile`、schema 2 的 `base_precision` / `module_precisions` 或 schema 3 的精确 `tensor_precisions` | GGUF 可同时有受保护的 F32、各 Q 格式和 Q8 回退；检查策略哈希、逐类型张量数量／字节数及保留原因 |
| 激活 | `backend-selected` / 配置 schema 5 的 `ggml-q8-rhs` | 有界 CUDA Q8_0 vision linear 强制 MMQ D4；报告检查每次 vision 编码 128 个请求／实际分派及 scratch 字节；不含全图 INT8/FP8 |
| 计算 | `compute.mode` | CPU `f32` 解码 Q 权重；CPU `native-quantized` 保留 packed 权重并允许内部 Q8 RHS 打包。CUDA F16 只给符合条件的浮点矩阵／attention 设置提示，量化矩阵保留自己的分派 |
| 缓存 | `cache.mode` | 图像特征 0/1/2：F32 为 F32/F32/F32，F16 为 F16/F16/F16，mixed-Q8_0 为 Q8_0/Q8_0/F32；不是 LLM KV cache |

权重格式在转换时确定；计算和实验缓存策略在加载时选择。CPU 支持 F32 和显式原生
量化计算，缓存仍为 F32；Metal 在这里接入 F32 计算／缓存。低精度缓存需显式 CUDA，
属于 probe 选项。CPU 加载 F16 权重会提升到 F32；默认量化矩阵乘先生成临时 F32
操作数，原生模式保留 Q 操作数。文件变小不保证运行内存同比变小。
合资格张量的精确名称覆盖已支持；正则／
通配符、全图激活量化及统一内核计算精度承诺尚未实现。混合 F16 使用配置 schema 4／GGUF schema 7。

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

CPU 原生模式使用 `ggml-quantized-cpu-native-v1`。标准 dot/tiled kernel 将图上 F32
RHS 动态打包为 Q8_0（Q8_0 权重）或 Q8_K（K 权重），使用整数局部点积、浮点缩放／
归约，输出 F32。这是源码解析的行为；可选构建／ISA 分派可能不同，报告不声称抓取
了 kernel 暂存或启用了全图 INT8 激活。新 producer 回执分别记录 `compute_mode`、
`cpu_compute`、`cuda_compute`；历史 F32/F16 回执含义保留，原生声明必须有新字段
及匹配的算术 profile。公共 API 和错误组合见[配置说明](quantization-config_zh.md#cpu-原生量化矩阵乘)。
配对 `performance` 接受原生 CPU 配置，或显式 CPU 下的
`--candidate-compute native-quantized`；质量报告仍为可选。

## 有界完整模型应用测量

一个 schema-6 模块／张量混合分配在 i7-12700KF 与 RTX 4090 上完成测量，覆盖两张
已有图片、六提示及原始 checkpoint 的六个参考对象。三种混合执行在 score > 0.5 时
均无对象缺失／新增：

| 混合执行 | 匹配掩码 IoU 均值 | 匹配掩码 IoU 最小值 |
| --- | ---: | ---: |
| CPU F32 decode | 0.998279 | 0.996401 |
| CPU native quantized | 0.998174 | 0.995988 |
| CUDA F32 policy | 0.997331 | 0.994643 |

独立四线程性能使用一个 groceries 案例，提示为 paper bag/car。CPU 比较同一混合
权重，一组进程配对、一次预热、三个样本；CUDA 比较 dense F32 与混合权重，两组
AB/BA 配对、两次预热、五个样本。内存另起进程测量：

| 比较 | 完整图像 P50 基线 → 候选 (ms) | 进程峰值基线 → 候选（十进制 GB） |
| --- | ---: | ---: |
| CPU decode → native | 31016.381 → 20388.631 | RSS 2.387 → 2.369 |
| CUDA F32 → mixed | 493.320 → 385.229 | GPU 4.777 → 2.477; RSS 1.011 → 1.044 |

换提示词 P50 在 CPU 下降 4.6%、CUDA 下降 5.8%；结果缓存命中仍约 11 ms。
CPU RSS 仅下降 0.75%，CUDA 主机 RSS 增加 3.3%。这些是单一分配及少量复用应用
图片的描述性结果，不代表生产准确率、独立留出集或普遍加速。Metal 与 kernel 内部
抓取仍未测量。精确配置、协议与本地证据可用性见
[验收计划](plans/20261010-193958-quantization-remaining-acceptance.md)。

## 有界 CPU 矩阵乘试验

无需 checkpoint 或图片，输出目录必须不存在：

```sh
cmake -S . -B build/quant-cpu -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF
cmake --build build/quant-cpu --target sam_cpu_quantized_matmul_probe -j 2
build/quant-cpu/examples/sam_cpu_quantized_matmul_probe \
  --threads 4 --warmups 3 --iterations 15 --output build/cpu-native-study
```

同一份 packed 权重和图上 F32 激活分别执行当前 Q→F32 路径与原生 CPU 路径。
21 个有界矩阵覆盖四种格式、单列／多列、QKV 视图、行尾及 Q8 的 4736 行宽。
计时复用已分配的图，排除初始化／传输，不启用节点 observer，交替 AB/BA 顺序。
`report.json` 保留原始样本、P50/P95、速度比、实际 GGML 身份／构建选项／ISA、
packed 载荷与 scheduler arena 字节。arena 不等于 RSS，不含常驻权重和 backend
私有 scratch。基线在每次图计算时解码权重，没有缓存完整解码后的 F32 模型。

正确性用独立解码的权重和 CPU 打包 RHS 做 scalar 点积，同时报告相对原始 F32 RHS
的差异。性能试验只检查选定位置；小型 CPU CTest 检查其全部输出。这是算术验证，
不代表分割质量。单独的单线程 RHS 打包试验只用于诊断，不是 kernel 内部计时，
不能从矩阵乘耗时中相减。结论仅适用于实测形状／构建／线程，不能据此宣称完整 SAM
提速。实现发现和测量保存在[计划](plans/20261010-174613-cpu-native-quantized-matmul.md)。

## CPU/CUDA 执行开销诊断

`execution-cost` 独立观察 CPU 或 CUDA 上一张图、一个提示词的执行成本，不依赖质量报告或
配对性能测量。模型须保留转换 manifest。先在关闭 CUDA/Metal 的构建中编译工具，
再显式运行：

```sh
cmake --build build/quant-cpu --target sam_execution_cost_probe -j 2
.venv-reference/bin/python tools/benchmark/quantization_benchmark.py execution-cost \
  --binary build/quant-cpu/examples/sam_execution_cost_probe \
  --model models/sam3-tensor-mixed.gguf --image /absolute/path/to/application.png \
  --text person --quantization-config docs/configs/quantization/image-tensor-mixed-cpu.json \
  --output build/application-cpu-cost
```

此命令会真正运行一次完整图像推理，使用 CPU/F32 计算/F32 缓存及 4 线程，区别于
只读转换预览。不会把模型复制到结果目录。子进程默认超时 1,800 秒，可用 `--timeout`
调整。小型矩阵 fixture 与完整 schema-6 混合模型均已实际运行两种 CPU 模式。
完整图中默认路径观察到 285 次 Q→F32 cast；原生路径为 285 次 packed CPU matmul、
零此类 cast，见[范围与证据](plans/20261010-193958-quantization-remaining-acceptance.md)。

要诊断原生 CPU 计算，可对匹配的模型使用
`docs/configs/quantization/image-tensor-mixed-cpu-native.json`。实际 packed 矩阵乘
必须位于 CPU，图上 RHS／输出为 F32。报告增加 `cpu_rhs_dot_type` 及源码证据标签，
不把未采集的 kernel 暂存和内部打包耗时误报为执行追踪。

`report.json`、`report.md` 绑定模型、manifest、二进制／动态库、图片及可选配置哈希；
`probe/execution-cost.json` 保留原始图／节点／传输观察。报告分别给出 graph bind、
同步 compute wall time、Q→F32 cast、矩阵乘、其他算子、metadata 及张量上传／下载 API
耗时。输入／输出类型和 backend 是实际观察，累加／RHS 提示仍只是请求。调用次数为
零的节点只出现在图快照中，没有观察到执行。arena peak 是 scheduler 张量 arena 的
最大值，不包括权重和 backend 私有 scratch；RSS 是独立的全进程峰值，二者都不是
节点输出字节数之和。

为避免相邻算子混入计时，诊断会逐节点同步，改变调度／融合并包含分派开销，固定标为
`diagnostic_only=true`、`performance_comparable=false`。节点耗时已经包含在图耗时中，
不能重复相加。传输计时不含主机内存分配和权重加载；图构造／reserve、预处理没有
单独计时。内核 RHS 打包可能发生在矩阵乘内部，但未独立测量，和内核算术一样保留
`NOT_COLLECTED`。成本用于解释执行路径；收益仍用不带 observer 的 `performance`
测量，其默认行为不变。

CUDA 构建启用 `GGML_CUDA=ON`，使用 CUDA 配置或显式传入
`--backend cuda --compute f32`。支持 F32/F16 compute 和 F32 cache。
嵌套 CUDA 成本格式为 `sam-cuda-execution-cost-v2`；CPU v1 的含义不变。
全部 compute 节点必须实际位于 CUDA。同步 wall time 包含分派和同步成本，
不能当作 CUDA event timestamp 或内核 packing/arithmetic 分项。

`sam_cpu_kernel_capture_probe --output NEW_DIR` 是独立的小型合成 CPU 诊断，
在真实 dot/tiled kernel 边界抓取 Q8_0/Q8_K RHS scratch 及内部阶段时间。
需要固定 CPU 扩展；进程局部 observer 安装／移除时，全部 CPU 图须空闲。
`GGML_CPU_TILED_MM=0` 运行 dot；
`GGML_CPU_TILED_MM=1 GGML_CPU_TILED_MM_FORCE=1` 可执行支持的 K-format tiled。
保存 payload、原始输入／权重／输出及阶段时间；时间包含 worker barrier，算术区间
排除 scratch-copy callback。该工具不加载 checkpoint，也不提供完整模型收益结论。
图成本报告内尚未采集的 kernel 项仍保持 `NOT_COLLECTED`。

## 历史结果

v2/v3 策略及依赖质量通过的性能资格流程只在
[归档](../tools/archive/precision_v2_v3/README.md)中保存，原活动入口已移除。
历史结果不改写；已完成的 ranked v2/v3 导出包仍可交给 `compare`，仅核对输入身份、
提示词清单和输出哈希，不重判历史策略或授予独立最终留出集结论。应用包包含并重新
校验标准化图片；历史图片身份使用原回执哈希。比较工具不会重新认证 producer 的
模型、二进制或 kernel 执行。

[本轮实施计划](plans/20261010-151710-performance-and-precision-reporting.md)记录工具、
归档和精度证据的验证范围；没有大规模 GPU 验收或新的硬件收益结论。
