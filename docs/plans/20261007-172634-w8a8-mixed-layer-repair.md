# W8A8 浮点对照与混合层修复

创建：2026-10-07 17:26:34，Asia/Shanghai。状态：浮点对照已修复，三种完整 selection 均未通过量化输出门限。

## 范围与依据

接续[统一 W8A8 的 COCO 筛选](20261007-165128-w8a8-coco-output-screening.md)。
该候选未通过逐对象门限；未量化平滑也有两幅图的特征误差略超门限。因此先
解决浮点对照，再按层型缩小量化范围。沿用已冻结的质量门限、原始 checkpoint
和 calibration／selection 拆分，不以 evaluation 结果调参。

## 方法与步骤

1. 冻结第三批完整源码和证据。保持旧报告不可变，所有新实验使用新目录。
2. 未量化对照使用与原 Linear 相同的输入形状及 fused bias 算术结构；新增
   二次幂 channel scale 作为明确候选。验证其 F32 有限范围内的等价性，检查
   underflow／overflow，不能据代数等价推断整图逐位相同。
3. 新增明确且可复现的层选择配置，未选中的线性层调用原始 F32。报告被量化
   层清单、scale 算法、数据和参数身份。先比较全视觉与 MLP／attention 消融，
   必要时进一步保留敏感层。不得把浮点例外描述为全层 W8A8。
4. 在两个原重参数化失败样本及诊断样本先核验，再运行完整 selection。量化
   候选依然执行固定 query 的漏检／误选及 mask／score／box 门限，标注指标
   单独计算。优先选择仍有可测性能潜力的范围，避免无止境搜索。
5. 有合格候选才冻结真实 runtime／GGUF 接入方案；若缩小范围后收益不足或
   质量仍失败，记录本轮 A8 退出结论并继续 A16、attention、缓存独立路线。

## 验证与交付

- 测试混合层控制、错误清单、scale 格式、原权重保留，以及二次幂缩放的边界。
- 对新 scale 对照真实 CUDA INT8 原型，不复用旧 alpha 的误差结论。
- 原始 checkpoint 最终输出与 COCO 标注同时报告；512 张 evaluation 在候选
  冻结之前不运行。浮点参考的速度和内存不冒充 C++ 运行时结果。
- 更新本计划与 changelog，执行相关工具测试及文档／空白检查；完成验证后清理
  可再生中间文件，保留各轮原始证据及可用构建。

## 结果

原重参数化诊断把 Linear 的 fused bias 改成了独立 matmul／add，并先 flatten
输入，改变了 F32 运算结构。保留原形状和 `functional.linear(..., bias)` 后，
COCO 127494／344816 的最大特征 L2 分别降至 `2.7960e-6`／`3.4543e-6`，
均通过旧门限。进一步采用二次幂 channel scale：四张诊断图及完整 128 张
selection 的 FPN 特征相对误差均为零，全部 413 个 prompt 的 JSON 输出与
原始 F32 逐字节一致。对变换中的 underflow／overflow 另做显式检查；没有
据此声称任意 F32 数值或未覆盖模型都能逐位等价。

新 scale 在四种真实矩阵上与原生 CUDA INT8 原型独立对照，相对 L2 最大
`8.561e-8`，最大绝对差 `9.537e-7`。CPU decoded-operand 对照也全部通过。
层选择、尺度边界、正常值等价、原权重保留和未选层恢复加入行为测试；隔离
参考环境的完整工具测试 **99 / 99 通过**。

完整 selection 结果如下；全部使用 α=0.75 后取最近二次幂，未选层保持原始 F32。
三次原始基线的 413 个 prompt payload 均与上一批逐字节一致。

| 量化范围 | 量化层数 | 逐图通过 | Prompted mask AP | 正类 union-mask mIoU |
| --- | ---: | ---: | ---: | ---: |
| 原始 F32 | 0 | 基线 | 0.494136 | 0.698237 |
| 全视觉线性层 | 128 | 98 / 128 | 0.491581 | 0.695831 |
| 仅 MLP | 64 | 96 / 128 | 0.493272 | 0.697467 |
| 仅 attention 线性层 | 64 | 111 / 128 | 0.490880 | 0.696725 |

三种范围的负提示检出均为零，AP／mIoU 平均下降预算均通过；逐对象门限均
失败，不进入 512 张最终评估，也不发布 C++ W8A8 profile。全视觉的低置信
query 误选／高置信丢失／mask-score-box 越界计数为 28／13／23，仅 MLP 为
30／17／18，仅 attention 线性层为 18／8／2。误差并不随层数单调变化。

对上一批普通 scale 的失败另做对象空间匹配归因：W8A8 的 27 个失败提示中，
181 个高置信对象有 166 个能匹配到满足 mask／score／box 要求的不同候选对象；
仍有 15 个不能匹配。原固定 query 检查中有 14 次 score 越界、1 次 mask IoU
越界。因此失败既有 query 编号变化，也有真实输出偏移。该诊断没有替换或
放宽 frozen gate，不能作为重新验收依据。

本轮退出 per-output W8／per-token A8 的整图集成尝试：有真实整数内核收益，
但统一和两类大范围消融都不满足质量；继续缩到少量层还会降低可覆盖耗时，
且当前原型的 F32 输出没有消除主峰值。保留工具作为后续更细粒度分组或重建
的基础，不得将本结论外推为所有 A8 方法无效。下一项独立实施见
[图像特征缓存筛选](20261007-173858-image-cache-numerical-screening.md)。

凭据：

- [完整混合层序列](../../build/runtime-quantization-mixed-selection-v1/sequence.json)
- [全视觉输出与特征](../../build/runtime-quantization-mixed-selection-v1/all/screening.json)
- [全视觉 COCO 指标](../../build/runtime-quantization-mixed-selection-v1/all-coco/metrics.json)
- [MLP COCO 指标](../../build/runtime-quantization-mixed-selection-v1/mlp-coco/metrics.json)
- [Attention 线性层 COCO 指标](../../build/runtime-quantization-mixed-selection-v1/attention-coco/metrics.json)
- [浮点修复烟测](../../build/runtime-quantization-repair-smoke-v1/sequence.json)
- [二次幂 scale 的 CUDA 对照](../../build/runtime-quantization-power-two-kernel-v1/parity.json)
- [失败归因](../../build/runtime-quantization-failure-analysis-v1/analysis.json)
- [99 项工具测试](../../build/runtime-quantization-repair-tools-v1.log)
- [本轮冻结凭据](../../build/runtime-quantization-repair-delivery-v1/receipt.json)
