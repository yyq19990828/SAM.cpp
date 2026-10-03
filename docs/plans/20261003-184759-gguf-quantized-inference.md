# GGUF 量化模型推理计划

创建时间：2026-10-03 18:47:59，Asia/Shanghai。
基线：`main` / `56a4cdeed5e3b408b3318becf32e52500690a195`，包含当前未提交的 CPU BLAS、ViT 投影折叠及验收文档。
历史阶段状态：图像量化实现及数值验证完成，初始存储策略未通过当时的完整张量门槛，保留诊断范围。后续输出质量主验收与 CPU 修复见[修复计划](20261003-205004-quantization-sensitivity-and-repair.md)。实施基线为 `37206da`，此前 CPU BLAS/ViT 改动已由现有提交收录。

## 目标与范围

为 SAM 3 GGUF 增加有明确权重策略、计算行为和误差标准的量化推理。推荐依次交付：图像 Q8_0/CPU、图像 Q8_0/Metal、图像 Q6_K → Q5_K → Q4_K。每个格式、后端分别验收；一个阶段完成后即可独立使用。

首版量化权重，保持现有公共 `Model` / `ImageSession` 调用、预处理、tokenizer、图结构和后处理。复用固定版本 GGML 的块格式、算子和调度器；SAM 仍为 C++17 header-only 集成层。现有 F32/F16/hybrid 契约继续单独验收。

视频作为后续独立阶段，先保留已验收 hybrid 的视觉及 tracker 原始 F32 闭包，仅量化文本编码器线性层。视频不能继承图像验收结论。CUDA、其他模型、激活的 GGUF 存储、KV/时序内存量化、mmap、分片、Q2/Q3/IQ、校准训练不在这份计划的实施范围内。

## 当前事实与设计依据

- [当前模型与精度契约](../../MODEL_ZOO.md)已记录图像 F32/F16、视频 F32/hybrid 的 CPU/Metal 本地验收。本计划以当前工作区和版本化基线为准。
- [GGUF schema](../gguf.md)和 `weights.hpp` 只接受 F32/F16/hybrid。`sam3_gguf.py` 的有界读取器也拒绝量化类型。`tensors.hpp` 已能按 `source_types` 注册单个张量的类型，但这不等于量化文件可加载或图可执行。
- 当前 CPU 会将 F16 权重提升为 F32；量化路径必须保留压缩驻留权重，避免沿用整模型提升逻辑。GGML BLAS 可逐次解量化到工作区再 SGEMM，原生 CPU Q8 路径则可量化右操作数；二者需要分别记录。
- [现有 Metal 补丁](../../cmake/patches/README.md)的严格 F32 矩阵路径只覆盖 F32/F16 左操作数。上游量化 Metal 矩阵内核使用 half staging。图中的 F32 激活、F32 输出或 F32 累加请求不能证明中间计算始终为 F32。
- 当前纯 F16 视频在 entry23/24 已出现 exact candidate 不一致，并在 Meta 的相同舍入权重中复现；因此不直接量化视频视觉/tracker，也不以更小文件代表视频支持。
- [当前可复用基线](../validation-baselines/blas-vit-m4pro-20261003.json)绑定原始权重、Meta reference、源码、二进制和后端库。新量化模型必须产生新 receipts，不能改标签复用旧模型结果。

主要依赖保持 GGML `353b63b439f27ab2cc19dac97ab1681ba6d2d084`、现有精度补丁及 `tools/requirements.lock` 的 `gguf==0.19.0`。本地锁定的 gguf-py 实现 Q8_0 量化，Q6_K/Q5_K/Q4_K 只有解量化能力；后者使用同版 GGML 的 `ggml_quantize_chunk`，不自行实现块编码。

## 推荐方案与最小实现

最小可用交付是：扩展现有 `.pt → GGUF` 转换器产生图像 Q8_0，CPU 以 Q8/F32 混合权重运行原有图，完成独立误差及内存验收。首版不增加原生转换程序、新依赖或新的推理配置开关。

随后复用同一文件在 Metal 上验收上游原生量化路径。这个路径具有不同于现有严格 F32 的计算契约，必须在模型信息、CLI 和报告中显式呈现。若误差超过 Q8 门槛，该后端保持未支持，不静默将整模型升成 F32，也不默认增加自定义 Metal kernel。

全量加载后解量化为 F32 是较小的兼容方案，但只减少磁盘体积，无法达到驻留权重目标，因此仅用于诊断 oracle。逐算子 F32 解量化仍可能增加整图工作区，亦不作为首版生产策略。

最脆弱的假设是“原生量化计算能同时通过质量门槛并降低实际内存”。如果不成立，先保留已通过的格式/后端，失败配置保留诊断状态。不能用预计压缩率代替质量或性能证据。

```text
原始 checkpoint + BPE
          |
tools/convert_sam3.py ---- 固定版本 gguf-py / 后续 GGML 行量化工具
          |
GGUF v3 + SAM schema 3 + provenance sidecar
          |
有界 GGUF 读取 -> SAM 张量/存储策略验证 -> backend 权重分配
          |
共享 SAM 3 图 -> CPU / BLAS 或 Metal -> 原有结果 API
          |
原始 Meta 对照 + 解量化权重诊断 + 分精度误差报告
```

## 张量量化策略

图像 profile 名称依次为 `image-linear-q8_0-v1`、`image-linear-q6_k-v1`、`image-linear-q5_k-v1`、`image-linear-q4_k-v1`。仅量化下列 canonical 二维矩阵，检查 block index 范围；字符串前缀本身不足以放行未知张量：

| 模块 | 矩阵集合 | 数量 |
| --- | --- | ---: |
| ViT blocks 0–31 | `attn.qkv.weight`、`attn.proj.weight`、`mlp.lin1.weight`、`mlp.lin2.weight` | 128 |
| Text blocks 0–23 | `attn.in_proj.weight`、`attn.out_proj.weight`、`mlp.fc1.weight`、`mlp.fc2.weight` | 96 |

其余 909 个图像张量全部从原始 checkpoint 保存为 F32，包括 embedding、RoPE、位置张量、norm、bias、卷积/反卷积、neck、fusion、detector、mask head 和 text 输出投影。保持复杂 RoPE 的实数对布局及既有名称/shape 映射。首版不量化任意“二维 weight”，也不先转 F16 再量化。

当前 schema 中 224 个目标矩阵含 746,586,112 个元素，占 842,343,734 个图像权重元素的约 88.63%。Q8_0 的 `ne[0]` 必须能被 32 整除，全部满足。K 格式要求 256：32 个 `vit.blocks.N.mlp.lin2.weight` 的 `ne[0]=4736` 不满足，固定使用 Q8_0；其余 192 个矩阵使用相应 K 类型。这是 profile 的必要组成，禁止隐式 padding、更改图维度或静默退回其他格式。

| Profile | GGML 主张量类型 | 块元素 / 块字节 | 非整块线性权重 | 理论 payload GiB |
| --- | --- | --- | --- | ---: |
| Q8_0 | Q8_0 | 32 / 34 | 本 inventory 无 | 1.095 |
| Q6_K | Q6_K | 256 / 210 | 32 个矩阵使用 Q8_0 | 0.962 |
| Q5_K | Q5_K | 256 / 176 | 32 个矩阵使用 Q8_0 | 0.889 |
| Q4_K | Q4_K | 256 / 144 | 32 个矩阵使用 Q8_0 | 0.820 |

以上由当前 schema 和块布局计算，包含保留 F32 的张量；不含 GGUF metadata/alignment，未实测。现有 F32/F16 payload 约 3.138/1.673 GiB。Q8 相对这两个 payload 预计减少约 65.1%/34.5%；峰值内存和速度必须单独测量。

## 文件、加载及公共信息契约

GGUF 容器仍为 v3；新增 SAM schema 3 以明确区分量化图像文件，保留 schema 1/2。第一阶段 schema 3 只接受 `sam.task=text_image` 和完整的 1,133 个张量；原版 runtime 应明确拒绝，而不是误认 F16。

| 字段 | 类型及规则 |
| --- | --- |
| `sam.schema_version` | UINT32，3 |
| `sam.task` | STRING，`text_image`；视频阶段完成前拒绝 `text_video` |
| `sam.storage_profile` | STRING，精确匹配上述已实现版本化 profile |
| `general.quantization_version` | UINT32，锁定实现的 `GGML_QNT_VERSION=2` |
| `general.file_type` | UINT32，GGUF/gguf-py 枚举：Q8_0=7，Q6_K=18，Q5_K_S=16，Q4_K_S=14 |

`general.file_type` 只是通用编码标签，SAM profile 及每个张量的真实类型是完整契约；不能将标签解释成 llama.cpp 的 S/M 张量分配策略。不能将 GGML 的 `ggml_ftype` K 类型枚举数值直接写入该字段：它与 GGUF/gguf-py 的枚举不同。

维持架构、参数、checkpoint/code revision、tokenizer 元数据的既有要求。sidecar 记录 profile、每个张量的原始/输出 dtype、shape、块大小、payload hash、转换器/helper/schema/依赖身份、完整 GGUF hash，以及后续 K 格式的量化程序和链接库 hash。运行时无需 sidecar 才能加载；数值验收必须验证 sidecar 和原始权重来源。

共同读取层显式检查 `ne[0] % block_size == 0`，不能只检查总元素数可整除；保留整数溢出、metadata budgets、canonical dimension、对齐、padding、连续目录、截断/范围、字符串和 tokenizer 检查。SAM adapter 检查精确 tensor inventory、shape 及 profile 的逐张量类型后，才分配 backend 权重。

量化 payload 的有限值检查必须解码 block scale 并验证解量化结果，不能对 packed bytes 使用 F32 检查。转换器逐张量回读并校验；runtime 在上传各个张量前验证其编码有效性，失败时通过现有 RAII 释放资源，不执行模型图。已有输出/sidecar 排他发布和失败回滚继续使用。

`ModelInfo::precision` 分别报告 `q8_0` / `q6_k` / `q5_k` / `q4_k`，`storage_profile` 报告完整混合策略。末尾追加 `arithmetic_profile` 字符串，保持旧聚合初始化兼容；值采用 `ggml-quantized-native-v1`，旧 profile 默认空值。实际 CPU/BLAS/Metal 放置及 node 数仍由现有统计给出。更新“CPU 始终 F32 保存/计算”的旧注释，避免套用到量化路径。

新增 profile 初期 `Auto` 选择 CPU，保留旧模型的 Auto 行为；通过 Metal 验收后用户可显式选 Metal，本计划不改变新 profile 的 Auto 默认值。device 初始化、驻留类型和算术策略放在 backend 模块；模型 adapter 只定义 SAM 存储白名单。不会引入通用模型/量化 plugin registry。

## 分精度误差标准

不同量化精度使用不同门槛。下表是规划建议值，尚未获得量化实测支持；实施时在任何模型验收运行前，以 `sam3-image-quantization-v1` 写入版本化 gates 文件并冻结。F32/F16/hybrid 继续使用已有标准，不能被这张表放宽。

| 权重 profile | 每个诊断张量 normalized L2 上限 | 高置信度 mask IoU 下限 | score 绝对误差上限 | box 单坐标 / 对应图像维度误差上限 | 零范数张量 max abs 上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 0.020 | 0.96 | 0.020 | 0.010 | 0.00010 |
| Q6_K + Q8_0 | 0.040 | 0.94 | 0.030 | 0.015 | 0.00020 |
| Q5_K + Q8_0 | 0.060 | 0.92 | 0.040 | 0.020 | 0.00050 |
| Q4_K + Q8_0 | 0.100 | 0.90 | 0.050 | 0.030 | 0.00100 |

这些是对原始 checkpoint / Meta F32 输出的端到端门槛，不是对另一个量化模型的比较。normalized L2 使用现有 `||actual-reference||₂ / ||reference||₂`；按 vision/text/fusion/logits 等每个必需张量、每个 case 验收，不能用总体平均掩盖局部失真。reference norm ≤ `1e-12` 时使用表中 max abs，避免除零。

IoU 按每个原始 reference score ≥ 0.6 的对象执行；score 是 0–1 概率的绝对差，box 使用对应宽/高归一化。高置信度对象必须仍检出；reference score ≤ 0.4 的 query 不得新增误检。0.4–0.6 的阈值邻域保留双侧 score、是否跨越 0.5 和 mask 差异，作为单独诊断，不以更改检测阈值来过门槛。固定七个 reference case 的指标只代表该范围，不声明数据集级 mIoU/mAP。

统一硬要求包括 shape/inventory/tokenizer 正确、输出有限且结构合法、输入和坐标合法、预处理误差 ≤ `2/255`、相同预处理/token IDs、负样例行为以及 session 生命周期/输出所有权正确。降低 bit 数不能容忍崩溃、文件解析错误或状态损坏。

报告保留三个对照层：

1. 原始 Meta F32 → 将实际量化 payload 解码为 F32 的 Meta 诊断：定位权重压缩误差，记录每个张量 L2/max abs。
2. 解量化权重 Meta 诊断 → C++ 原生量化：定位后端 staging/右操作数量化等额外误差。每个非零范数张量建议上限为 Q8 0.005、Q6 0.010、Q5 0.015、Q4 0.025；零范数使用上表 max abs。必须明确该对照是补充诊断，不是原始权重 oracle。
3. 原始 Meta F32 → C++ 原生量化：作为发布质量裁决，应用主表全部门槛。前两项不能替代它，也不能简单相减推导第三项。

同一格式 CPU/BLAS/Metal 共用质量门槛，后端引入的额外误差单独呈现。报告绑定 gates 文件 hash、profile、实际张量 dtype 分布、模型/输入/二进制/库身份；缺失或未知 gates 直接报错，禁止默认映射成 F16。失败后优化实现或保留诊断状态；若产品质量要求改变，必须新建 gate 版本并说明理由，不能根据失败结果直接改本次标准。

## 可独立交付的步骤

### A. 图像 Q8_0 / CPU

预计 3–5 个工程日，另加串行模型验收时间；这是实施工作量估计。

1. 在 `sam3_gguf.py` 定义 schema 3/profile 的确定性逐张量规则；在 `convert_sam3.py` 增加 `--task image --precision q8_0`。从原始 `.pt` 直接取 F32 值，使用固定 gguf-py 的 Q8_0；其余张量保留原始 F32。保持现有 metadata、排他发布、sidecar 与回读流程。
2. 修正 packed payload 的 tensor info/data 写入方式，逻辑 dimensions 与 uint8 packed buffer 的 shape 分离。按行量化，逐张量处理，校验输出字节数、解码有限性及未量化 payload 与 F32 源一致。
3. 更新共同 reader 与 SAM loader，先通过完整格式/模型契约检查，再按验证过的 `source_types` 分配 Q8/F32 权重。CPU 不整模型解量化；F16 旧提升策略保留。
4. 复用现有共享图和 QKV views，验证量化行步幅、`nb[1]` view offset、batch/尾部以及投影折叠。量化仅触及线性矩阵，卷积图无需扩展量化支持。
5. 新增版本化 gates `tests/data/sam3-quantization-gates.json`，调整 `validate_image.py` 读取 profile 对应标准，保留原始 oracle/provenance 限制，增加解量化权重诊断导出。将 profile/算术标签写入现有 CLI JSON。
6. 完成 CPU 无 BLAS与可用 BLAS 两条实际运行路径的算术回归、七个官方 case、session/内存检查。分别报告 BLAS 解量化工作区，首版可交付 CPU 支持。

首版预计修改超过 8 个文件：`tools/{convert_sam3.py,sam3_gguf.py,validate_image.py,test_tools.py}`、`include/sam/internal/io/gguf_reader.hpp`、`include/sam/internal/models/sam3/{weights.hpp,model.hpp}`、`include/sam/internal/runtime/ggml/{backend.hpp,runtime.hpp}`、`include/sam/internal/runtime/ggml/backends/cpu.hpp`、`include/sam/types.hpp`、`examples/image_support.hpp`，以及相关测试/gates、精度和支持文档。复用现有 tests 文件，只有跨格式的量化算术/加载行为需要独立归档时才新增 `test_quantization.cpp` 并注册 CTest。

### B. 图像 Q8_0 / Metal

预计 1–2 个工程日，另加验收时间；A 独立可用。

1. 用 A 的同一 GGUF、gates 和输入在 M4 Pro 上显式 Metal 运行。保持现有 dense/attention/window 补丁，量化矩阵先使用上游原生内核。
2. 增加真实 Q8 Metal 矩阵测试：qkv/proj/MLP 代表形状、QKV views、小 batch、大 spatial columns、非整 tile 的 M/N 尾部；记录实际内核计算差异，不以 `supports_op` 返回 true 作为数值证明。
3. 完成七个原始 Meta case、解量化权重诊断及 session 回归。支持声明要求这些模型图零 CPU/BLAS fallback；缺失 Metal 或不支持节点必须清晰报错。
4. 比较同输入 F32/F16/Q8 的加载、权重 buffer、compute buffer、RSS、单图延迟。通过后只扩展显式 Metal 支持；失败时 A 保持完整交付。

### C. 图像 K 格式

预计 2–4 个工程日加验收时间；各格式可独立发布，依次 Q6_K、Q5_K、Q4_K。

1. 增加小型 `tools/quantize_rows.cpp`，链接现有 `ggml` target，在 `SAM_BUILD_EXAMPLES` 下构建 `sam_quantize_rows`，不增加新 CMake 开关。通过已检查的 F32 行文件、行长/行数和目标类型调用 `ggml_quantize_chunk`；返回字节数并写排他输出。处理有限输入、整块、乘法溢出、截断和写入失败，不实现自有量化算法。
2. `convert_sam3.py --precision q6_k|q5_k|q4_k --quantizer /absolute/path/to/sam_quantize_rows` 用 subprocess 参数数组调用 helper，逐张量使用独立临时文件。Q8 分支继续复用 gguf-py；K 格式的 32 个非整块矩阵固定回到 Q8，不能由 runtime 再猜测。
3. 各 profile 分别校验类型策略、转换程序/库身份、块字节数及回读；不重新量化 Q8 文件，始终从原始 checkpoint 生成。
4. 逐格式完成 CPU、CPU/BLAS、Metal 的算术、七个官方 case 和独立 gates 验收。某一格式不通过不会撤回已通过的更高精度格式。
5. 更新模型文件大小、dtype/参数分布、质量/内存/速度表。Q4_K 名称只描述该 SAM profile，不复用 LLM Q4_K_M 的效果或兼容性宣传。

### D. 后续视频 Q8_0

此阶段独立于图像 K 格式，预计 2–3 个工程日加更长的时序验收时间。使用 `video-text-q8_0-v1`：96 个 text block 线性矩阵 Q8，其他张量严格遵循原始 `visual-tracker-f32-v1` 的 F32/F16 策略，直接从原始 checkpoint 生成。schema 3 扩展到 `text_video` 和完整 1,464 inventory；保留 F16 预处理、BF16 features/memory 和既有 temporal profile。

视频 Q8 端到端初始标准：每个必需非零范数张量 normalized L2 ≤ 0.015，零范数 max abs ≤ 0.00010，高置信度 mask IoU ≥ 0.96，score error ≤ 0.020，box dimension fraction ≤ 0.010。与图像 Q8 的张量上限分开，给时序传播保留误差余量；gates 在首个视频量化运行前冻结到 `sam3-video-quantization-v1`。

保留五个 case / 216 帧、进入/遮挡/消失/重现、固定 birth ID 映射、持续 ID 唯一及 continuity、frame 输出/drain/hotstart、memory selection、长期状态边界检查。原有原始模型的 exact candidate 验收规则继续保留；量化 profile 将 candidate 差异单列诊断，允许候选排序变化，但每个非阈值邻域对象必须通过最终 mask/score/box 与 ID 连续性检查，禁止仅放宽内部 candidate 规则便宣布通过。视频 Q6/Q5/Q4 不在 D 中宣称支持，需要单独规划时序质量门槛。

## 验证与完成条件

测试围绕真实错误：非法 ne[0] 而总元素整块、缺失/未知 quantization_version/profile、错误 file_type、白名单外量化 tensor、quantized norm/conv、shape/type mismatch、坏 scale/nonfinite、截断/offset/padding/overflow、转换覆盖失败回滚，以及压缩 QKV view/stride 读错。保留 Release 活跃断言、独立头文件与 two-TU linkage。

量化算术测试用非恒定值、零行、符号/幅值变化、块边界、真实 SAM K/M/N 和尾部；对照固定 GGML 编码/解码及独立标量点积，验证布局和真实计算。单层通过不等于整模型通过。后端额外算术误差和原始权重质量门槛分别核验。

实施时使用新的构建/输出目录，下面为 A/B 完成后应执行的命令接口；本次没有执行它们，也不覆盖旧 receipts：

```sh
rtk proxy cmake -S . -B build/quant-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
rtk proxy cmake --build build/quant-cpu --parallel 2
rtk proxy ctest --test-dir build/quant-cpu --output-on-failure
rtk proxy cmake -S . -B build/quant-cpu-no-blas -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_BLAS=OFF
rtk proxy cmake --build build/quant-cpu-no-blas --parallel 2
rtk proxy ctest --test-dir build/quant-cpu-no-blas --output-on-failure
rtk proxy cmake -S . -B build/quant-metal -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
rtk proxy cmake --build build/quant-metal --parallel 2
rtk proxy ctest --test-dir build/quant-metal --output-on-failure
rtk proxy build/reference-runtime/venv/bin/python tools/test_tools.py
rtk proxy python3 tools/check_docs.py
rtk git diff --check
```

沿用 `convert_sam3.py` 的 checkpoint/BPE/output 和 `validate_image.py` 的 build-dir/model/reference/cases/backend/threads/output 参数；唯一首版转换新增值为 `--precision q8_0`，不新增推理 CLI 开关。K 阶段新增 `--quantizer`，其对应失败模式是锁定 Python 包没有 K 格式编码实现。验收工具依据模型 profile 选择冻结 gates，不提供任意命令行 tolerance override。

每个受支持格式至少包含七个原始 Meta 图像 case；CPU native 和可用 BLAS 都有明确运行证据，Metal 单独验证。旧 F32/F16/hybrid 文件继续加载，公共 session/cache/tokenizer/所有权行为、two-TU 和独立头文件回归通过。使用当前源码重新运行受影响的旧矩阵，不将旧 receipts 作为新二进制的证明；未影响的原始模型/reference bytes 可按现有基线规则复用。

基准沿用当前协议：图像 1 warmup + 5 次完整替换图像推理，排除结果缓存；同一 RGB input/decoder、threads、AC/sleep 条件顺序测量。CPU BLAS 对照记录 `VECLIB_MAXIMUM_THREADS=4`，无 BLAS单独报告。记录模型和库 hash、dtype 分布、加载及首图时间、稳态 median、权重/compute/backend 工作区和 process peak RSS；禁止将 weight_bytes 当峰值内存。BLAS 自有解量化工作区如未计入现有 scheduler 统计，须独立测量或明确报告不可获得。

发布条件为：格式及数值/session 检查通过，压缩驻留权重有实测，峰值 RSS 不高于同输入/后端 F32 基线；未降低整体峰值时只宣称文件与权重 buffer 压缩。速度不设未经测量的加速承诺，报告包括回退或变慢的配置；不得把完整解量化模式算作压缩驻留模型。

完成阶段后同步 `docs/gguf.md`、架构、README、MODEL_ZOO/中文、BENCHMARK/中文、validation 文档及 `changelog.md` Unreleased，只将已完成的格式/后端列为支持。生成 source-only gates/baseline 索引，用现有 archive 工具保存新的私有模型/诊断/receipts；失败记录也保留。提交、推送和 remote CI 是后续明确交付动作，本计划不执行。

实现不需要新增 API key、账户或服务。原始 checkpoint/BPE/reference 已有本地与私有归档；重新获取时按现有 Hugging Face 流程，认证与许可接受分别检查，不在 chat 请求 token。可通过撤回新增 profile 支持并继续使用原始 F32/F16/hybrid 文件回滚，任何阶段不覆盖既有模型或验收材料。

## 上游依据

- [固定 GGML 的 GGUF v3 规范](https://github.com/ggml-org/ggml/blob/353b63b439f27ab2cc19dac97ab1681ba6d2d084/docs/gguf.md)：容器、quantization_version、file_type 与张量类型信息。
- [GGML 量化接口](https://github.com/ggml-org/ggml/blob/353b63b439f27ab2cc19dac97ab1681ba6d2d084/include/ggml.h)及[行量化示例](https://github.com/ggml-org/ggml/blob/353b63b439f27ab2cc19dac97ab1681ba6d2d084/examples/common-ggml.cpp)：复用块布局和 native 编码接口；示例的旧容器读写不移植。
- [CPU 运算实现](https://github.com/ggml-org/ggml/blob/353b63b439f27ab2cc19dac97ab1681ba6d2d084/src/ggml-cpu/ggml-cpu.c)与[Metal 矩阵实现](https://github.com/ggml-org/ggml/blob/353b63b439f27ab2cc19dac97ab1681ba6d2d084/src/ggml-metal/kernels/mul_mm.metal)：量化权重之外的计算精度也需核验。

## 本次结果

已检查现有文件/加载/后端/转换器/验收脚本、当前基线与依赖能力，并按真实张量清单计算 profile 的块边界和理论 payload。分精度及图像/视频门槛已写为明确的初始建议，尚未冻结为运行过的 gates，也未测量量化质量、延迟或峰值内存。

本次仅新增计划；未改实现、转换权重、安装依赖、运行构建/模型测试或提交/推送。`tools/check_docs.py` 通过：37 份文档的本地链接及既有双语表格检查成功；`git diff --check` 通过。新增计划另行检查了 code fence、末尾换行、空白和完整的分精度门槛，全部通过。

## 实施记录

2026-10-03 用户授权实施及子代理委派。转换/块编码、C++ runtime/算术回归、分精度 gates/验收工具分别由子代理负责；另一个子代理准备独立串行验收 harness。主代理负责集成、真实模型验收、文档与结果收尾。各代理按文件范围协作，不提交或推送，不覆盖原始权重、旧构建与旧验收记录。

图像 Q8_0/Q6_K/Q5_K/Q4_K 的转换、schema 3 校验、原生压缩驻留、CPU/Metal 调度及分精度验收工具已实现。初始 ViT+text Q8 的 CPU/BLAS 七 case 全部数值失败；解量化 Meta 诊断复现权重损失，而原生算术诊断 7/7 通过。随后按[文本精度保护计划](20261003-194133-quantized-image-text-preservation.md)增加视觉量化策略，四种精度的三个后端路径共 84 次对照全部完成，完整张量门槛仍未通过，未放宽标准或声明模型数值支持。

三个 Release 构建 CTest 各 12/12，Python 工具 39/39；原有 F32/F16/hybrid 图像子集的 CPU/BLAS、Metal 回归 42/42。量化视频在图像质量问题解决前维持后续范围；用户要求正式性能暂缓。完整当前范围见[量化说明](../quantization.md)及[结果索引](../validation-baselines/quantized-image-m4pro-20261003.json)。无提交、推送或 remote CI。
