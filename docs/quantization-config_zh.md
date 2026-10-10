# 统一量化配置

[English](quantization-config.md) · [量化选项](quantization_zh.md) · [应用 benchmark](quantization-benchmark_zh.md)

SAM 3 图像工具可以共用一份 JSON 配置，覆盖权重转换、质量参考导出、精度检查和
性能 benchmark。配置明确区分权重存储、激活策略、计算策略和图像特征缓存。
配置有效仅表示存在对应代码路径，不代表硬件已验证、质量合格或必然提速。

## 配置格式

[自定义 Q4 示例](configs/quantization/image-q4-vision-text-cuda.json)：

```json
{
  "schema_version": 1,
  "kind": "sam-quantization-config",
  "task": "image",
  "backend": "cuda",
  "weights": {"precision": "q4_k", "modules": ["vision", "text"]},
  "activation": {"mode": "backend-selected"},
  "compute": {"mode": "f16"},
  "cache": {"mode": "mixed-q8_0"}
}
```

四个维度和明确的后端都必须填写。未知字段、重复 JSON 键和不支持的模式会报错。
量化权重必须明确填写 `modules` 或 `storage_profile`。`modules` 可选择
`vision`、`text`、`fusion`、`decoder` 的任意非空子集，解析后按规范顺序保存。
所选模块的合资格线性矩阵共用一种目标 Q 格式，其余张量保留 F32；K block 的固定
Q8 回退规则继续适用。

使用已有预设时，可以把 `weights` 改为：

```json
{"precision": "q4_k", "storage_profile": "image-full-linear-q4_k-v1"}
```

`image-vision-linear-*`、`image-full-linear-*` 和自定义 `image-modules-linear-*`
属于不同的分配策略。选齐四个模块不会自动变成 full 预设。同时提供 profile 与模块
列表时，两者必须一致。解析后的配置会列出所选模块；schema 3 GGUF 本身仍没有
schema 4 的 `quantization_modules` 元数据。具体张量范围由 profile 决定，旧策略与
模块化策略的合资格矩阵存在区别。

F32/F16 权重分别填写 `{"precision":"f32"}` / `{"precision":"f16"}`，不能附加量化
模块选择。旧命令行默认行为保持不变；新 JSON 的量化配置要求明确选择范围，避免
无意中采用旧版视觉／文本分配策略。

## 模块级混合权重

[混合 CPU 示例](configs/quantization/image-mixed-cpu.json) 使用配置 schema 2，权重维度为：

```json
{
  "precision": "mixed",
  "base_precision": "q4_k",
  "module_precisions": {"text": "q8_0", "fusion": "q6_k", "decoder": "f32"}
}
```

基础 Q 格式应用于四个模块的合资格线性矩阵，覆盖项替换指定模块的选择。示例解析为
vision Q4_K、text Q8_0、fusion Q6_K、decoder F32。每个模块可选
F32/Q8_0/Q6_K/Q5_K/Q4_K。偏置、归一化、embedding、卷积、规范向量和受保护的小矩阵
继续保留 F32。vision 选择 K 格式时，行宽 4736 的 MLP 仍固定回退为 Q8_0；模块请求
不能绕过保护规则。

解析结果补齐四个模块，并添加 `storage_profile=image-mixed-linear-v1` 和 `policy_sha256`。
GGUF schema 5 保存此策略，加载器独立检查每个张量的类型和布局。
`ModelInfo::precision` 报告 `mixed`，`base_precision`、`module_precisions`、`policy_sha256`
与实际张量清单解释分配。全部模块覆盖为 F32 也有效，此时不宣称使用量化算术。
配置 schema 1 维持原有单格式语义，不能填写模块格式覆盖。

转换必须直接读取原始 F32 checkpoint，不支持已降精度 checkpoint 或对已有 GGUF
再次量化。任何模块请求 Q6_K/Q5_K/Q4_K 时都需要原生行量化工具；只有 Q8_0/F32 的
模块策略不需要：

```sh
python3 tools/quantize/quantization_config.py validate \
  --config docs/configs/quantization/image-mixed-cpu.json --context conversion

.venv-reference/bin/python tools/convert/convert_sam3.py \
  --quantization-config docs/configs/quantization/image-mixed-cpu.json \
  --checkpoint /absolute/path/to/sam3.pt --bpe /absolute/path/to/bpe_simple_vocab_16e6.txt.gz \
  --quantizer "$PWD/build/quant-cpu/examples/sam_quantize_rows" \
  --output models/sam3-mixed.gguf

.venv-reference/bin/python tools/benchmark/quantization_benchmark.py inspect-precision \
  --quantization-config docs/configs/quantization/image-mixed-cpu.json \
  --model models/sam3-mixed.gguf --output build/mixed-precision.json
```

manifest 逐张量记录请求／实际 dtype、模块选择器、保护或回退原因和字节数。本轮验证了
CPU 转换、格式校验和小型混合矩阵运算，尚未测量完整模型质量／性能，以及混合策略的
CUDA/Metal 执行。已有固定预设的硬件结果不能替代新混合策略的测量。

## 可用组合

| 选择 | CPU | Metal | CUDA |
| --- | --- | --- | --- |
| 图像权重 | F32/F16/Q8_0/Q6_K/Q5_K/Q4_K；模块混合策略 | 相同存储格式；混合策略硬件未验证 | 相同存储格式；混合策略硬件未验证 |
| 独立激活模式 | 仅 `backend-selected` | 仅 `backend-selected` | 仅 `backend-selected` |
| 计算策略 | `f32` | `f32` | `f32` / `f16` |
| 图像特征缓存 | `f32` | `f32` | 私有图像 probe 可用 `f32` / `f16` / `mixed-q8_0` |
| 当前导出／性能 runner | 有对应路径 | 不支持，仅可检查配置 | 有对应路径 |

此表描述代码路径，不新增硬件验收结论。CPU 加载 F16 权重时提升为 F32，量化矩阵
运算使用临时 F32 权重。CUDA F16 选择符合条件的低精度操作数／attention 策略；
量化 kernel 仍按后端分派，不能据此宣称完整模型为 W4A16 或全程 FP16 累加。
`backend-selected` 也不表示所有激活具有同一种 dtype。kernel 的临时表示与累加类型
需要单独的执行证据。

缓存是保存在主机、换文本提示时复用的图像编码器特征，不是 LLM 的 KV cache。
`mixed-q8_0` 将第 0/1 层存为 Q8_0，第 2 层保留 F32。权重分配一致时，可对同一份
GGUF 改变计算／缓存策略；改变权重精度或模块范围需要从原始 checkpoint 重新转换。

以下请求会直接报错，不会静默回退：

| 请求 | 当前结果 |
| --- | --- |
| 单独指定每层／张量的 Q 格式，或在混合策略中使用 F16 | 尚未实现；schema 2 支持按模块选择 F32/Q8/Q6/Q5/Q4 |
| 指定 INT8/FP8/F16 激活模式 | 尚未实现；研究工具不等于原生运行模式 |
| CPU/Metal F16 计算或低精度图像缓存 | 此配置不支持 |
| 公共 C++ API 使用低精度缓存 | 未开放，仅限私有原生图像 probe |
| 用当前 runner 在 Metal 导出／测性能 | 不支持，可不执行推理地检查配置 |
| 在此文件配置视频跟踪／记忆策略 | 不在图像配置范围，视频流程单独保留 |
| 配置的权重分配与现有 GGUF 不同 | 推理前报错，重新转换或选择匹配配置 |
| `auto` 后端、未知维度或选项 | 报错，必须使用明确后端和已知字段 |

公共 C++ `BackendOptions` 仍配置后端、线程、设备和 CUDA 计算策略，没有 JSON 加载器、
独立激活选项或低精度缓存入口。`validate --context public-api` 只检查这些配置是否可
映射到现有 API，不会加载模型或新增 API。

## 校验与使用

以下只读命令仅依赖 Python 标准库，不初始化 GPU：

```sh
python3 tools/quantize/quantization_config.py capabilities
python3 tools/quantize/quantization_config.py validate \
  --config docs/configs/quantization/image-q4-vision-text-cuda.json
```

可用 `--context conversion`、`inspect`、`benchmark`（默认）、`public-api` 或
`original-reference` 检查目标入口。原始 checkpoint 参考导出要求 CUDA、F32 权重、
F32 计算和 F32 缓存。`--output NEW.json` 保存解析结果和配置／源文件哈希，拒绝覆盖。
此校验不需要模型、数据集或实际性能测量。

转换只应用权重维度，把其余三个维度记录为计划采用的运行设置。它不执行模型推理，
也不会根据计算／缓存设置改变 GGUF 权重内容：

```sh
.venv-reference/bin/python tools/convert/convert_sam3.py \
  --quantization-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --checkpoint /absolute/path/to/sam3.pt --bpe /absolute/path/to/bpe_simple_vocab_16e6.txt.gz \
  --quantizer "$PWD/build/quant-cpu/examples/sam_quantize_rows" \
  --output models/sam3-custom-q4.gguf

.venv-reference/bin/python tools/benchmark/quantization_benchmark.py inspect-precision \
  --quantization-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --model models/sam3-custom-q4.gguf --output build/custom-precision.json
```

Q6_K/Q5_K/Q4_K 仍需要固定版本的原生行量化工具，Q8_0 不需要。使用配置文件时，
不要同时传转换参数 `--task`、`--precision`、`--storage-profile`、`--quantize-modules`，
或运行参数 `--backend`、`--compute`、`--cache`、`--activation`。即使值相同也会报冲突，
以保证配置来源明确。

质量导出同样使用 `--quantization-config`。性能对比提供两份配置，后端必须相同：

```sh
.venv-reference/bin/python tools/benchmark/quantization_benchmark.py performance \
  --cases models/application-cases.json --input-root models/application-images \
  --binary build/cuda/examples/sam_precision_benchmark_probe \
  --baseline-model models/sam3-f32.gguf --candidate-model models/sam3-custom-q4.gguf \
  --baseline-config docs/configs/quantization/image-f32-cuda.json \
  --candidate-config docs/configs/quantization/image-q4-vision-text-cuda.json \
  --limit 1 --pairs 1 --warmups 0 --iterations 2 --memory-iterations 1 \
  --output build/application-config-performance
```

应同时提供两份文件，或者继续使用旧的独立参数，两种方式不能混用。该命令仅展示
可选的小规模测量，不是必须执行的验收。报告将解析配置与实际模型的转换 manifest
绑定，并记录配置及源文件哈希。转换时保存的运行设置只是来源记录，不永久限制
GGUF 的运行方式；新运行可以在权重不变的前提下选择其他受支持的计算／缓存策略。
离线报告使用保存的配置，不再依赖原始 JSON 文件。请求值、运行时报告的策略编号、
图张量类型、尚未采集的 kernel 算术证据继续分开描述。

参考一致性、用户可选质量建议、性能测量和收益标签相互独立。统一配置不会新增
COCO 质量硬门槛或必须提速的要求。

[混合权重量化计划](plans/20261010-155435-mixed-weight-quantization.md)跟踪模块实现与后续
硬件／逐张量扩展。[原生激活量化计划](plans/20261010-155435-native-activation-quantization.md)
中的规划模式目前尚不能在配置中选择。
