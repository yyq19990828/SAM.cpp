# 量化敏感性定位与修复

创建时间：2026-10-03 20:50:04，Asia/Shanghai。
基线：`37206da` 加现有未提交量化实现与两套冻结 gates。
状态：本轮实现、修复与数值验收完成。按用户明确选择的输出质量主目标验收；全量张量保真独立报告。正式性能采集仍暂缓。

用户随后明确选择“输出质量为主，张量保真单独报告”。因此本轮新增独立输出质量 gate 集，保留两个旧完整张量 gate 文件及其失败结论。新主结论检查对象存在/误检、原高置信度 query、mask/score/box、坐标、有限值、token/预处理及 session 行为；不同位宽的这些数值门槛不放宽。旧 raw tensor 的 L2/zero-norm 标准继续作为独立保真诊断，不能把其历史失败改写为成功。CPU 改为受控 F32 输入算术时使用新 `ggml-quantized-weights-f32-v1` 版本，原 CPU-native receipts 保留。

## 已知问题

旧 ViT+text Q8 的 text_features 相对误差约 0.035978，实际 GGUF 解码回 F32 后的 Meta 运行复现该误差，C++/BLAS 对同权重 Meta 的 text 差异约 0.000006765。文本保留原始 F32 后，这一阶段恢复，但视觉量化仍导致下游原始张量超限。视觉 Q8 的最大 mask_logits L2 在 CPU/BLAS 为 0.100626，Metal 为 0.106810，native CPU 为 0.256501。高置信度 mask IoU 均仍接近 1；原始张量门槛独立执行，没有因输出 mask 较好而放宽。

本轮分别定位权重误差、解码器放大与 native CPU 内部激活量化的影响。现有诊断 profiles、文件、gate hashes、原始 receipts 和私有归档保持不可改写。

## 定位方法及修复路径

1. 用已有原始 Meta 与 12 个量化单元输出做逐张量、逐 query 误差分解。统计有效检测、阈值邻域和低分 query 各自对 mask logits L2 的贡献；检查 box/query 排序、mask logit 幅值与二值 mask 的关系。所有对象和原始 tensor 比较标准保留。
2. 在被忽略的独立诊断目录构建私有 C++ probe。只读取已经通过 schema/来源检查的原始 F32 GGUF 和视觉 Q8 GGUF，按模型 canonical 名称/shape 合并张量，运行原有共享图。比较原始权重、完整视觉 Q8、仅 MLP 量化和仅 attention 量化四种权重分配。诊断 state 不通过公共加载 API，不发布伪称符合已有 profile 的 GGUF。
3. 先在 truck-truck 与最坏 groceries-bottle 的完全相同 reference PPM 上检查代表性组，记录每层组的图像特征、fusion、boxes/class/mask logits、对象输出和权重 bytes。完整量化 probe 必须匹配已有相同二进制/输入输出趋势，以证明探针没有改变算术。若分组结果提示少量敏感层，再按明确 block 范围恢复原始 F32，避免随意改变未版本化的白名单。
4. 对 native CPU 增加相同解码权重与 F32 输入的对照，确认额外误差是否来自 Q8/K 右操作数量化。只有有实测依据时修改 backend 算术策略；压缩驻留和工作区分别报告，不用全模型 F32 升格伪称低内存量化。
5. 根据用户随后选择的输出质量主目标，使用已有版本化 storage profiles，不新增不必要的敏感层例外。新增独立输出质量 gate 集并在复评前冻结；IoU、score、box 分位宽数值不变。已有完整张量 gate 的失败保留，新的 `output_quality_passed` 与 `tensor_fidelity_passed` 两轴独立出具。CPU 算术修复单独版本化并运行新二进制，不能用历史 native CPU 数据代表新算术。

## 文件范围和分工

主代理负责私有 probe、实际模型运行与集成；数值子代理只读分析已保存 outputs 与 gate 定义；runtime 子代理只读核对 GGML native activation 转换及可复用精度能力。正式改动仍位于现有 converter、SAM profile 策略、backend/graph 算术和必要行为测试中，不增加通用量化插件或模型接口。

## 验证

每个诊断记录绑定两份实际权重、输入 PPM、prompt/token IDs、probe 源码/二进制、GGML 库、backend、策略和输出 hash。CPU/Metal 的同权重差异与对原始权重误差分别报告。新正式 profile 在 CPU/BLAS、native CPU、Metal 上分别完成完整七 case 与 session/所有权验证后才能声明相应支持；直接操作测试和编译不能替代整模型验收。原 F32/F16/hybrid 图像 42 项回归维持，gates、旧失败、非本任务 tracker 计划保留。

完成后记录原因、修复、数字、通过范围及剩余限制，更新中文/英文量化说明和 changelog。无提交、推送、依赖安装或正式跑分。

## 定位及已完成修复

- 视觉 Q8 七个样例共 1,400 个 query，仅 6 个为原始高置信度检测，没有阈值邻域 query。低置信度 query 对 mask logits 平方误差的贡献超过 99.997%；原始检测集合没有变化，最小有效 mask IoU 约 0.9965。全量 raw tensor 超限主要衡量被过滤 query 的变化，不能直接当成已检出对象质量失败。
- native CPU 的量化 `MUL_MAT` 将右侧 F32 激活内部转换成 Q8_0/Q8_K，所固定 GGML revision 的 CPU kernel 不使用 source-F32 precision flag。含异常值的 Q6_K 小型独立算术探针对同解码权重 F32 参考的 L2 为 0.033854；仅设置 flag 与默认结果完全相同。
- backend 策略对 CPU 量化图的左侧权重插入共享 F32 cast，按输出根重建计算图并回收临时工作区，避免二次量化激活。驻留权重仍使用原始 packed buffer。CPU 实际算术为 `ggml-quantized-weights-f32-v1`；Metal 保持 native packed kernel 和禁止 CPU compute fallback。
- 新 gate 集 `sam3-image-output-quality-v1` SHA256 为 `c69dd6fee4abc789e421c407c7f954e3fdfd7f04a2a79288d3192816385f11e8`。旧两套 raw gates hash 未改变，旧 receipts/归档保持原结论。复评旧 Metal 输出为新增独立报告，且验证原 receipt、输入/模型、二进制/库及输出 hash。
- 三个 Release 构建（CPU/BLAS、CPU 无 BLAS、Metal）均完成独立 header/two-TU 编译与 CTest 12/12；isolated reference 环境工具测试 46/46。CPU 算术回归对含异常值的相同 decoded 权重 F32 参考匹配到约 1e-5 绝对误差。
- 首个新 Q8 CPU/BLAS 整模型七样例输出质量 7/7 通过、raw tensor fidelity 0/7；随后完成全部十二个新构建单元。

## 最终数值验收

四个 `image-vision-linear-{q8_0,q6_k,q5_k,q4_k}-v1` profile 分别在 CPU/BLAS、无 BLAS CPU、Metal 上完成固定七样例，输出质量 84/84 通过，完整张量保真 0/84；两轴使用独立冻结 gate/hashes，没有改写旧失败。每个图像样例还执行重复 prompt 的缓存计数检查。Q8/Q4 两个边界格式在三个后端完成六个独立跨 session 检查，验证换 prompt/换图、权重共享、结果所有权、cache 不增加计算/分配，以及释放公共 Model 后 session 仍可运行。中间位宽复用相同的所有权/缓存实现，同时完成各自全部数值及缓存样例。

| 精度 | 三个后端输出通过数 | 最小高置信度 mask IoU | 最大 score 绝对误差 | 最大 box/尺寸误差 | 原完整张量通过数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 21/21 | 0.998233 | 0.000868 | 0.0000593 | 0/21 |
| Q6_K + Q8_0 | 21/21 | 0.997722 | 0.006857 | 0.0001588 | 0/21 |
| Q5_K + Q8_0 | 21/21 | 0.994853 | 0.003762 | 0.0004168 | 0/21 |
| Q4_K + Q8_0 | 21/21 | 0.978615 | 0.010446 | 0.0006968 | 0/21 |

新 CPU 无 BLAS 的 Q8 最坏 `mask_logits` 相对 L2 从原 native 0.256501 降到 0.100623，接近新 CPU/BLAS 的 0.100626；剩余权重量化漂移仍按原门槛报告失败，不把它包装成张量保真通过。四组新 Metal 的 CPU/BLAS 计算节点均为零。历史 Metal 输出的新 gate 复评 28/28 为单独附加证据，不计入上述 fresh 84。

离线 peer 对比重新核验全部输入指标/输出 hash：Q8 的新无 BLAS与新 BLAS 对 groceries-bottle 的 mask logits L2 差为 `6.51862e-5`，全语料最坏 mask peer 差为 `6.53495e-5`。旧→新无 BLAS 的 63 个模型输出 tensor×case 对中 55 项 L2 改善、1 项变差、7 项持平；mask 的七个 case 全部改善。十二个单元共 840 个必需 tensor×case 对中 500 项通过旧保真阈值、340 项失败，故完整 case 保真依旧 0/84。该诊断支持额外 CPU 算术误差已降低；不能把 peer 对照当成纯权重损失的严格分解。

原 F32/F16/hybrid 图像子集在两个新构建后端上共 42/42 回归通过，包括 F32/Metal。三个 Release 构建 CTest 各 12/12、独立 header/two-TU 编译及 isolated reference 工具 46/46 保持通过。实现源快照 106 文件校验无变化；fresh 数值/输出、模型、实际二进制/库及六份 session 收据均重新核对 hashes。

支持范围仅为这些四个视觉量化 profile 的固定语料数值验收：两张原始图像及一个 crop，共六个高置信度检测，没有阈值邻域 query；这不是数据集级 mIoU/mAP。宽范围 ViT+text profiles 保持诊断用途，视频量化不在本次完成范围。CPU 压缩驻留、计算工作区分别记录；诊断导出 RSS/计时没有替代正式性能协议。

完整数字、身份和私有归档见[新输出质量索引](../validation-baselines/quantized-image-output-m4pro-20261003.json)。旧索引、旧两套 gates、失败 receipts 和旧私有 archive 保持原状态。未提交、推送、运行 remote CI 或安装新依赖；非本任务 tracker 计划保持原样。


## 后续文档整理补充：精确 gate 与输出/张量记录

以下详细工程数据从原量化指南移至本计划，供审阅和复现。前文的独立输出
质量主 gate、raw tensor fidelity 失败结论及历史收据均保持不变。

### 冻结 gate 身份与逐精度门槛

旧完整张量 gate `sam3-image-quantization-v1` 的 SHA-256 为
`819d9de1ebe638d223db9d57c8ac652b36707fae3d36124735bdc65d681ba2cc`；视觉量化
gate `sam3-vision-image-quantization-v1` 的 SHA-256 为
`6bdb91602e8a89e8b29b208282b846b541d8fbe499ae36955e4f905e9eff7a93`。后者保留前者
的父 gate hash，两个文件均未修改。

| 精度 | Per-tensor normalized L2 上限 | 高置信度 mask IoU 下限 | Score absolute error 上限 | Box coordinate fraction 上限 | Zero-norm absolute error 上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 0.020 | 0.96 | 0.020 | 0.010 | 0.00010 |
| Q6_K + Q8_0 | 0.040 | 0.94 | 0.030 | 0.015 | 0.00020 |
| Q5_K + Q8_0 | 0.060 | 0.92 | 0.040 | 0.020 | 0.00050 |
| Q4_K + Q8_0 | 0.100 | 0.90 | 0.050 | 0.030 | 0.00100 |

normalized L2 逐必需 tensor、逐 case 检查，不跨 case 平均；参考范数不大于
`1e-12` 时改用 zero-norm absolute-error 上限。预处理输入另有 `2/255` 最大绝对误差
限制。参考检测 score 至少 `0.6` 必须保留，score 不高于 `0.4` 不得新增检测；
`0.4–0.6` 保留于诊断，阈值邻域之外的 selected-query 集合不得变化。

新增独立输出 gate `sam3-image-output-quality-v1`，SHA-256 为
`c69dd6fee4abc789e421c407c7f954e3fdfd7f04a2a79288d3192816385f11e8`。它记录两个旧
张量 gate hash，并沿用逐精度 mask IoU、score 和 box 限值：

| 精度 | 高置信度 mask IoU 下限 | Score absolute error 上限 | Box coordinate fraction 上限 |
| --- | ---: | ---: | ---: |
| Q8_0 | 0.96 | 0.020 | 0.010 |
| Q6_K + Q8_0 | 0.94 | 0.030 | 0.015 |
| Q5_K + Q8_0 | 0.92 | 0.040 | 0.020 |
| Q4_K + Q8_0 | 0.90 | 0.050 | 0.030 |

输出 gate 还绑定冻结 input bytes、官方 token IDs、必需 tensor 有限值与预处理误差；
每个 score 至少 `0.6` 的 reference query 必须保留，score 不高于 `0.4` 的 query
不得成为新检测，并检查阈值邻域之外的 selected-query 集合。

另有 native-arithmetic 诊断：C++ GGML 输出与使用实际 GGUF 解码权重的 Meta
FP32 导出对照，其逐精度 normalized L2 上限为 Q8_0 `0.005`、Q6_K `0.010`、
Q5_K `0.015`、Q4_K `0.025`。这项对照不能替代原始 checkpoint 的输出质量主验收。
早期 28/28 Metal 输出复评重用了已保存的推理结果，仅作为历史补充，不计作新推理或
正式性能样本。

### Fresh output-quality 指标

下表为四种 vision-only profile 在三个后端对固定七 case 的最终输出复评。每行均为
7/7 通过；固定语料由两张原始图像及一个 crop 构成，含六个高置信度检测、零阈值邻域
query。它不是数据集级 mIoU/mAP 声明。

| Profile | Backend | Cases | 最小高置信度 mask IoU | 最大 score error | 最大 box fraction error |
| --- | --- | ---: | ---: | ---: | ---: |
| Q8_0 | CPU/BLAS | 7/7 | 0.998380 | 0.000706018 | 0.0000563046 |
| Q8_0 | Native CPU | 7/7 | 0.998380 | 0.000706316 | 0.0000563554 |
| Q8_0 | Metal | 7/7 | 0.998233 | 0.000867665 | 0.0000592546 |
| Q6_K + Q8_0 | CPU/BLAS | 7/7 | 0.997722 | 0.006856561 | 0.0001587272 |
| Q6_K + Q8_0 | Native CPU | 7/7 | 0.997722 | 0.006855667 | 0.0001586676 |
| Q6_K + Q8_0 | Metal | 7/7 | 0.997722 | 0.006805540 | 0.0001583695 |
| Q5_K + Q8_0 | CPU/BLAS | 7/7 | 0.994853 | 0.003761589 | 0.0004167561 |
| Q5_K + Q8_0 | Native CPU | 7/7 | 0.994853 | 0.003761053 | 0.0004167561 |
| Q5_K + Q8_0 | Metal | 7/7 | 0.994999 | 0.003621876 | 0.0004142764 |
| Q4_K + Q8_0 | CPU/BLAS | 7/7 | 0.978615 | 0.010445655 | 0.0006967669 |
| Q4_K + Q8_0 | Native CPU | 7/7 | 0.978615 | 0.010444165 | 0.0006967669 |
| Q4_K + Q8_0 | Metal | 7/7 | 0.978615 | 0.010253668 | 0.0006880186 |

### Raw tensor fidelity 与历史诊断补充

完整 case raw-fidelity gate 为 0/84 通过；在 840 个必需 tensor×case 对比中，
500 项符合旧逐张量阈值、340 项超过阈值。整 case 完整 gate 和逐张量比例是不同统计，
不能解读成每个 tensor 都失败。旧 Metal 张量报告中的汇总为：

| Vision profile | 旧完整张量 gate cases | 最坏 normalized L2 | 冻结 L2 限值 | 最小高置信度 mask IoU |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 0/7 | 0.106810 | 0.020 | 0.998233 |
| Q6_K + Q8_0 | 0/7 | 0.270003 | 0.040 | 0.997722 |
| Q5_K + Q8_0 | 0/7 | 0.270525 | 0.060 | 0.994999 |
| Q4_K + Q8_0 | 0/7 | 0.305778 | 0.100 | 0.978615 |

宽范围 `image-linear-q8_0-v1` 在七个 CPU/BLAS case 均失败；`truck-truck` 的
`text_features` normalized L2 为 `0.035978`，高于 Q8 限值 `0.020`。用实际 GGUF
解码权重运行 Meta FP32 复现了该误差；C++ 对解码权重 Meta 的 native arithmetic
诊断七例通过，最坏 per-tensor normalized L2 约 `9.55e-5`。解码权重压缩诊断中，
最坏 `mask_logits` normalized L2 为 `0.089143`，最小高置信度 mask IoU 为 `0.99838`；
输出 IoU 不抵消 tensor gate 失败。

历史 vision Q8 raw-tensor receipt 在 CPU/BLAS、旧 native CPU 和 Metal 均为 0/7；
`mask_logits`、`pred_boxes`、`class_logits` 曾超出 tensor 阈值。三个旧 backend 的
最小高置信度 mask IoU 为 `0.996465`，出现在无 BLAS native CPU。该 CPU receipt
早于新 F32-weight-cast 算术 profile，不代表 fresh 输出质量结果。修复后的 Q8
`mask_logits` normalized L2 从旧 native CPU 的 `0.256501` 降至 `0.100623`，CPU/BLAS
为 `0.100626`；两者仍超过 raw 限值 `0.020`。

### 驻留权重、工作区与样本产物

| Vision profile | Packed weight buffer bytes | CPU 最大 compute buffer bytes | Metal 最大 compute buffer bytes |
| --- | ---: | ---: | ---: |
| Q8_0 | 2,063,373,600 | 961,062,048 | 1,242,648,448 |
| Q6_K + Q8_0 | 1,993,282,848 | 961,062,048 | 1,242,648,448 |
| Q5_K + Q8_0 | 1,954,845,984 | 961,062,048 | 1,242,648,448 |
| Q4_K + Q8_0 | 1,918,670,112 | 961,062,048 | 1,242,648,448 |

这些是驻留权重与最大 scheduler compute-buffer 计数，不是 process peak RSS；CPU BLAS
内部 workspace 未单独测量。独立完整性能收据现归档于
[性能验收计划附录](20261003-230244-quantized-image-performance-acceptance.md#english-performance-record)。
