# W8A8 的独立 COCO 整图筛选

归档路径说明：`build/`、`models/` 等目录中的证据不随源码分发。下文以相对仓库根目录的普通路径记录这些产物，保留历史哈希和验收结论。

创建：2026-10-07 16:51:28，Asia/Shanghai。状态：128 张筛选及标注评估完成；统一候选未通过，继续精度修复。

## 范围与依据

继续[运行时量化主计划](20261007-143003-activation-and-runtime-quantization.md)，
使用[已验证的 INT8 原型](20261007-155334-cuda-w8a8-fp8-kernel-prototypes.md)
所声明的通道 scale、per-output W8、per-token A8、INT32 点积及 F32 epilogue。
FP8 的两个输出模式尚未满足完整内核门限，本轮不进入 SAM 整图。

先在独立 selection 集做输出筛选，比较原始 F32、未量化通道重参数化、W-only、
A-only 和组合 W8A8。只改变 128 个视觉线性层，其他模块、分辨率、预处理、
tokenizer、残差、LayerNorm、softmax、RoPE 和输出阈值保持原模型契约。该参考
程序为数值研究，不测量或宣称 C++ 运行时性能，不新增可用 GGUF profile。

## 方案与步骤

1. 冻结新候选的独立质量门限，复用主计划的 Q8 级 mask／score／box 要求，
   另对未量化重参数化设严格等价检查；保留原门限文件不变。
2. 从已冻结校准身份中读取每层通道 scale，先固定 α=0.75 起点。原始权重保持
   不变；每种模式单独处理输入／权重，正确覆盖官方融合 MLP 绕过 Linear.forward
   的路径，每幅图核对所有 128 层调用。
3. 在同一原始 checkpoint、F32 oracle 配置下按图成组比较。原始与候选输出
   保存所有 query 的 score／box 以及所选 mask 的有界 RLE；每个提示保存原始
   输入身份与质量结果。有限 diagnostic 子集先验证工具，不能标为完整 selection。
4. W8A8 若使用 PyTorch INT8 矩阵乘法，则显式核验 INT32 点积和 CUDA 原型一致；
   W-only／A-only 为独立归因对照。量化过程采用 F32 除法及 nearest-even，沿
   K 维按正确方向平滑，在 INT32 点积之后应用两侧 scale 与 bias。
5. 通过工具检查后运行全部 128 张 selection。根据固定质量预算选择层例外或
   拒绝候选。若新增例外，只使用校准与 selection，不读取 evaluation 结果调参。
6. 后续补齐 COCO 标注指标并完成真实 SAM 算子接入，再在 512 张 evaluation 与
   既有七个回归上最终验收。AP 使用实际请求的图像／类别对，保留负提示，明确
   标注这是 prompt 条件下的自定义 COCO 子集，不能称为完整 COCO val2017 AP。

## 验证与输出边界

- 未量化重参数化先检查视觉特征及最终输出，再研究量化误差。参数形状、来源、
  通道 scale、有限性、层调用完整性均在发布报告前验证。
- 输出筛选沿用固定 query 的高／低置信、0.5 阈值及边界区间检查；IoU、score
  与归一化 box 误差逐个报告，漏检和误选单独列出。选出的 mask 不仅比较平均值。
- COCO 标注 AP 与正类 union-mask mIoU 的下降预算分别为 1 和 0.5 个百分点；
  空／负类误选另报，不能依赖大量空样本抬高 mIoU。指标未实现或未运行时明确
  记录未完成，不能让输出筛选报告冒充发布验收。
- 新目录保留原始 RLE、数值摘要、全部通过／失败项、校准、模型、输入、源代码
  与库身份。运行前后校验数据与源码；不改写前两批构建、模型或原始 evidence。
- 对量化方向、舍入、零值、每行 scale、真实 MLP 调用和输出门限写行为测试。
  后续实现变动后只补跑相关检查，数值筛选与正式计时分离。

## 结果

本轮使用 128 张 selection、413 个图像／类别对（285 个正提示、128 个负提示）。
每幅图的五种模式分别核对全部 128 个视觉线性层恰好执行一次。原始 checkpoint、
校准、COCO 拆分、工具和参考源码身份在运行前后保持一致。512 张 evaluation
尚未参与量化参数选择或模型推理。

| 模式 | 逐图门限通过 | Prompted mask AP | 正类 union-mask mIoU | 负提示检出数 |
| --- | ---: | ---: | ---: | ---: |
| 原始 F32 | 基线 | 0.494136 | 0.698237 | 0 |
| 未量化重参数化 | 126 / 128 | 0.494136 | 0.698237 | 0 |
| W-only | 118 / 128 | 0.494068 | 0.696108 | 0 |
| A-only | 106 / 128 | 0.492558 | 0.697687 | 0 |
| W8A8 per-token | 102 / 128 | 0.492692 | 0.694463 | 0 |

AP 仅在请求的 image/category 对上计算，使用 score > 0.5 的输出、COCO segm
IoU .50:.05:.95、maxDets=100 和原 crowd 处理；不是完整 COCO val2017 AP。
mIoU 对正提示逐对合并实例，排除 crowd 像素后取宏平均。负提示单独统计，
不把其空 mask 当作满分加入平均值。所有变体均满足 AP／mIoU 平均下降预算，
但没有变体同时通过全部逐对象／特征门限，不能据平均值宣告验收通过。

未量化重参数化的最终 mask 与 AP／mIoU 无变化，两个失败均来自 FPN 的相对 L2：
COCO 127494 为 `1.06414e-5`、344816 为 `1.22860e-5`，略超预先冻结的 `1e-5`。
W-only 的失败原因计数为低置信 query 误选 10、高置信 query 丢失 5、mask／score／box
越界 3；A-only 为 19／9／13，W8A8 为 18／9／15。一张图可以有多个失败。
这些固定 query 的检查与标注 AP 是不同约束，不因平均 AP 通过而放宽前者。

另发现 PyTorch 的 `amax / 127` 在 CUDA 上使用标量倒数乘法，会在一个真实 MLP1
权重的 nearest-even 边界与 C++ F32 除法相差一个 INT8 值。改用同形状 tensor
除数后，四类真实矩阵相对 CUDA 原型的输出 L2 均小于 `9e-8`，最大绝对差不超过
`9.537e-7`。完整 selection 使用修正后的算术；早期两张图烟测和初次差异诊断
另行保留，未覆盖原证据。`torch._int_mm` 保证 INT32 点积，未用浮点矩阵乘法
冒充整数累计。隔离参考环境的完整工具测试为 **94 / 94 通过**。

保留本轮工具和失败证据，拒绝将统一 α=0.75 的 128 层 W8A8 发布为可用 profile。
下一步先保持原 Linear 的浮点算术结构并比较二次幂通道 scale，解决未量化对照；
再用 selection 做 MLP／attention 与敏感层消融。不得改变已冻结 gate 或读取
evaluation 来修复候选。尚未实现 C++ 整图接入或声称端到端延迟／显存收益。

凭据：

- 完整输出筛选（`build/runtime-quantization-output-selection-v1/screening.json`）
- COCO 标注结果（`build/runtime-quantization-coco-selection-v1/metrics.json`）
- 修正后的 CUDA 算术对照（`build/runtime-quantization-output-screen-preflight-v2/cuda-kernel-parity.json`）
- 初次舍入差异（`build/runtime-quantization-output-screen-preflight-v1/scale-rounding-diagnostic.json`）
- 两张图诊断烟测（`build/runtime-quantization-output-screen-smoke-v1/screening.json`）
- 本轮源码及证据归档（`build/runtime-quantization-screening-delivery-v1/receipt.json`）
