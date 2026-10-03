# 量化图像推理：保留文本编码器精度

创建时间：2026-10-03 19:41:33，Asia/Shanghai。
实施基线：`37206da` 加当前量化转换、runtime 与验收工具；原计划见 [GGUF 量化推理](20261003-184759-gguf-quantized-inference.md)。
历史阶段状态：实现及数值验证完成，完整张量门槛未通过。后续按用户选择完成独立输出质量主验收与 CPU 算术修复，见[修复计划](20261003-205004-quantization-sensitivity-and-repair.md)；旧失败记录不变。正式性能测量仍暂缓。

## 根因及范围

原 `image-linear-q8_0-v1` 量化 ViT 与全部 text block 线性矩阵，在原始 Meta 七个图像 case 的 CPU/BLAS 对照中全部未通过冻结门槛。对实际 GGUF 解量化后运行 Meta FP32 的独立诊断复现相同误差；例如 truck-truck 的 text_features normalized L2 为 0.035978，而 C++ 对解量化 Meta 的 text_features 差异为 0.000006765。七 case 的原生算术诊断通过，压缩权重的质量诊断失败。因此主要问题是该权重分配策略的文本精度损失，不能通过更改后端或放宽既有门槛来解决。

新增质量优先的视觉量化 profile。只量化 32 个 ViT block 的 128 个线性权重，text、neck、fusion、detector/mask head 和其他张量全部保留原始 F32。旧 ViT+text profile 及失败证据保持独立，可供明确的诊断使用。此阶段仍为 schema 3 / text_image；视频、其他架构和正式性能测量不属于本次修复。

## 实施方案

1. 新增 `image-vision-linear-q8_0-v1`、`image-vision-linear-q6_k-v1`、`image-vision-linear-q5_k-v1`、`image-vision-linear-q4_k-v1`。精度标签仍为 q8_0/q6_k/q5_k/q4_k。K 格式仅在 32 个 `vit.blocks.N.mlp.lin2.weight`、canonical ne[0]=4736 上固定使用 Q8，其余 96 个 ViT 矩阵使用对应 K 类型。
2. 转换器新增可选 `--storage-profile`，必须与 `--precision` 和 task 精确一致。原默认 profile 保持不变并标明诊断状态；新 profile 由用户显式选择。这个参数对应已经证实的质量/压缩分配差异，不新增推理参数或通用插件框架。
3. Python 和 C++ 使用统一的确定性规则；将“是否量化 text block”作为版本化 profile 的具体策略字段。保持 common reader、canonical shape 预检、source types、压缩驻留和 Meta/GGML provenance 要求。
4. 增加 `tests/data/sam3-vision-quantization-gates.json`，在任何新 profile 模型测试前冻结。质量数值完全沿用原精度标准：Q8/Q6/Q5/Q4 normalized L2 上限分别 0.020/0.040/0.060/0.100，mask IoU 下限 0.96/0.94/0.92/0.90，score 上限 0.020/0.030/0.040/0.050，box 上限 0.010/0.015/0.020/0.030，零范数和原生算术标准也不变。新 gate 集只改变适用 profile 和范围声明，记录原 gate hash；不修改 `sam3-quantization-gates.json`。
5. 验收器按精确 storage profile 选择对应冻结 gate 集，保留两个 profile 家族的转换/模型/门槛身份。实际 GGUF 解码的 Meta 诊断与原始 Meta 发布对照继续分开。
6. 先生成视觉 Q8 并完成 CPU/BLAS、native CPU、Metal 的七 case 对照及 session 行为。依次生成其余 K profile 并单独验收；一个格式失败不撤销已通过格式，也不阻断其他格式的诊断。

预计 Q8 payload 为 2,063,373,528 字节，约 1.922 GiB，仍比原始 F32 payload 小约 38.8%。这大于现有 mixed-F16 文件；不承诺相对 Metal F16 的内存收益。实际权重 buffer、peak RSS 和速度分别记录；用户已暂缓正式性能采集。

## 文件分工及验证

转换子代理修改 `tools/sam3_gguf.py`、`tools/convert_sam3.py`、`tools/test_tools.py`；runtime 子代理修改 `weights.hpp` 和相关量化/contract 测试；验证子代理修改 gate 选择、补充 Meta exporter 及工具测试。主代理集成三种构建、原始模型回归、新 profile 全图验证、session 和文档。native 行量化工具及现有推理调用接口保持复用。

先做 profile/dtype/count、text 原始 F32 payload 保留、跨家族 gate 选择、错误 profile/precision/task 拒绝的聚焦回归，再运行 CPU/无 BLAS CPU/Metal CTest，保留独立头文件及 two-TU 检查。模型对照使用原有完整七 case、固定 inputs/tokenizer/score threshold；匹配格式只使用相应精度门槛，不用平均值覆盖单个 case 失真。

原策略失败、Metal scheduler tail 的早期错误和验收 wrapper 变化的原始记录保留。受控三层诊断已经绑定实际 GGUF、二进制、backend libraries、输入和冻结 gates；原 CPU 批次的 wrapper source drift 仅发生在被忽略的验收 driver，其主数值失败不改写为成功。

## 完成条件与回退

只有通过该 profile/后端的完整七 case、原生格式/调度检查及 session 才列为数值支持。没有通过的格式明确标为诊断/未通过，报告最坏误差和范围。旧 F32/F16/hybrid 继续回归，旧 profile 的 gate 和 artifacts 不被覆盖。

若视觉 Q8 仍失败，进一步按诊断定位具体层的敏感性，另写明确的 profile 修订方案，保留此版本及失败数据；禁止不断修改同一 profile 的白名单或门槛。无需迁移已有文件，可继续使用既有原始精度模型回退。此阶段不提交或推送。

## 结果

四种视觉量化格式分别完成 native CPU、CPU/BLAS、Metal 的七 case 对照，共 84 次；所有配置都未通过完整逐张量门槛，未列入数值支持。文本保留 F32 消除了 text_features 超限，但视觉权重引起的 mask logits、boxes/class logits 漂移仍超过各精度标准。各 Metal case 均由 Metal 执行，无 CPU/BLAS 图回退。旧标准与新标准数值没有改变。

软件验证：三个 Release 构建及独立/重复头文件全部编译成功，CTest 各 12/12；隔离 Python 工具测试 39/39。原有 F32/F16 图像及 hybrid 图像子集在 CPU/BLAS 与 Metal 的当前构建回归共 42/42 通过。正式性能及量化视频保持后续范围，不宣称延迟/峰值内存收益。结果与 artifacts 见[收据索引](../validation-baselines/quantized-image-m4pro-20261003.json)和[量化说明](../quantization_zh.md)。


## 后续文档整理补充：profile 与容器工程细节

以下字段原先放在量化用户指南中；整理后作为实现/复现资料留在计划，
不改变本计划正文记录的历史阶段结论，也不扩展支持范围。

### 精确存储策略

| Profile | 量化分配 | GGUF file type | 保留为 F32 的张量数 |
| --- | --- | ---: | ---: |
| `image-linear-q8_0-v1` | 224 个 ViT 与 text linear tensors 使用 Q8_0 | 7 | 909 |
| `image-linear-q6_k-v1` | 192 个 ViT/text linear tensors 使用 Q6_K；32 个 ViT `mlp.lin2.weight` 使用 Q8_0 | 18 | 909 |
| `image-linear-q5_k-v1` | 192 个 ViT/text linear tensors 使用 Q5_K；32 个 ViT `mlp.lin2.weight` 使用 Q8_0 | 16 | 909 |
| `image-linear-q4_k-v1` | 192 个 ViT/text linear tensors 使用 Q4_K；32 个 ViT `mlp.lin2.weight` 使用 Q8_0 | 14 | 909 |
| `image-vision-linear-q8_0-v1` | 128 个 ViT linear tensors 使用 Q8_0，text encoder 全部 F32 | 7 | 1,005 |
| `image-vision-linear-q6_k-v1` | 96 个 ViT linear tensors 使用 Q6_K；32 个 ViT `mlp.lin2.weight` 使用 Q8_0；text encoder 全部 F32 | 18 | 1,005 |
| `image-vision-linear-q5_k-v1` | 96 个 ViT linear tensors 使用 Q5_K；32 个 ViT `mlp.lin2.weight` 使用 Q8_0；text encoder 全部 F32 | 16 | 1,005 |
| `image-vision-linear-q4_k-v1` | 96 个 ViT linear tensors 使用 Q4_K；32 个 ViT `mlp.lin2.weight` 使用 Q8_0；text encoder 全部 F32 | 14 | 1,005 |

以上计数基于完整的 1,133-tensor image inventory。K profiles 的 Q8_0 fallback
限定于 `vit.blocks.0.mlp.lin2.weight` 至
`vit.blocks.31.mlp.lin2.weight`，每个 tensor 的 canonical `ne[0]=4736`；该行宽
不能被 256-element K block 整除。实现拒绝不符合该固定 fallback 的 shape。
其他 vision-only tensor、neck、fusion、detector/mask head、embedding、norm
及 bias 保持 F32。

schema 3 文件要求 `sam.schema_version=3`、`sam.task=text_image`、精确的
`sam.storage_profile`、`general.quantization_version=2` 及匹配的
`general.file_type`。file type 是容器标签，逐 tensor 类型与 storage profile
共同规定实际分配；schema 3 不表示 video 格式。Python profile table 与 C++
profile whitelist 必须完全对应。宽范围 `image-linear-*` 默认仍保留原诊断
分配；vision-only candidate 需显式选择。

### 参考转换产物尺寸与身份

以下记录来自同一原始 checkpoint 的独立转换样本，只用于来源追踪，不替代
数值验收或用户机器上的输出尺寸。F32 image GGUF 样本为 3,371,139,456 bytes。

| Storage profile | 文件字节数 | SHA-256 |
| --- | ---: | --- |
| `image-vision-linear-q8_0-v1` | 2,065,138,144 | `3acb81bbaa585bb25be7f22ef11e169189ab4a939a6fae1e27c2f329a5ea06f7` |
| `image-vision-linear-q6_k-v1` | 1,995,047,392 | `2b2226dab53f9da5ba6337be71d07f57e5ebe3eff73384f0b1a058dcb66fb3a6` |
| `image-vision-linear-q5_k-v1` | 1,956,610,528 | `f969034b572867d574b371edd58739c6a0e76125ee770a710a46acba507233f3` |
| `image-vision-linear-q4_k-v1` | 1,920,434,656 | `e191011d8ff6029807e24c8232afc44f9b41e1c408ffd0b35fc4530d29d8996a` |

The matching conversion manifest retains source checkpoint/tokenizer identity,
tensor inventory and payload hashes, exact profile, package/helper provenance,
and output hash. Existing destination files are rejected rather than replaced.
Q8_0 conversion uses the pinned `gguf==0.19.0` writer. Q6_K/Q5_K/Q4_K call the
repository `sam_quantize_rows` helper and require its absolute executable path;
the helper identity is checked against pinned GGML 0.25.3 and its GGML and
quantization-library artifacts before encoding.

### Backend arithmetic profile

Runtime `ModelInfo` reports actual backend arithmetic. CPU uses
`ggml-quantized-weights-f32-v1`: packed quantized weights remain resident, while
shared graph nodes create transient F32 casts for `MUL_MAT`. Metal uses
`ggml-quantized-native-v1`, with native quantized kernels and half staging; its
scheduler must report no CPU/BLAS compute-node fallback. The manifest's original
`arithmetic_profile=ggml-quantized-native-v1` records the conversion-side
contract and is not a claim about later CPU execution. Quantized `Auto` selects
CPU. These arithmetic paths are backend-specific, not interchangeable labels.
