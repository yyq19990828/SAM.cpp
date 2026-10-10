# 模块级混合权重量化实现与 CPU 验证

日期：2026-10-10，Asia/Shanghai。状态：M1–M3 已实现，CPU fixture 验证完成。

## 范围

落实 [混合权重量化计划](20261010-155435-mixed-weight-quantization.md) 的 M1–M3：统一 JSON 配置、原始 F32 checkpoint 转换、SAM GGUF schema 5、原生加载校验与精度报告、CPU 小型数值验证。前序统一配置已提交为 `43c8f52`。

本轮支持 `vision`、`text`、`fusion`、`decoder` 四个模块各自选择 F32/Q8_0/Q6_K/Q5_K/Q4_K；保护张量继续使用 F32。保留 schema 1–4 与配置 schema 1 的读取语义。暂不实现逐张量覆盖、模块 F16、激活量化，不运行大规模 GPU 验收或转换完整检查点。

## 方法

1. 配置 schema 2 使用 `weights.precision=mixed`、量化 `base_precision` 和 `module_precisions` 覆盖；解析时补齐四模块并生成规范策略哈希。未知模块、无效格式及相互冲突的字段直接报错。
2. SAM GGUF schema 5 使用 `image-mixed-linear-v1`，记录基础格式、四模块格式与规范策略哈希；`general.file_type` 对应基础格式。张量真实类型及保护/回退规则由独立的 Python/C++ 校验确认。
3. 每个张量直接从原始 checkpoint 转换，manifest 记录请求格式、实际格式、模块及保护/回退原因。原子发布与输入身份检查延续现有行为。
4. 原生加载在分配权重前检查完整张量布局和策略。报告分别给出模块权重策略、实际类型统计、计算策略和缓存策略，CPU 验证不能解释为 GPU 性能或质量证明。

## 步骤

1. 扩展共享权重策略及配置 schema 2，添加 CPU 示例与配置边界测试。
2. 扩展转换、GGUF 元数据、manifest 与读取校验，保持旧转换输出不变。
3. 实现 C++ schema 5 校验和模型信息报告，接通 benchmark 配置身份检查。
4. 小型原始 F32 fixture 覆盖所有模块及格式、保护张量、非对齐回退和损坏元数据；使用 CPU 原生行量化器与混合矩阵运算验证。
5. 更新双语用户文档、GGUF 格式合同、changelog 与本计划结果。

## 验证

- 使用隔离 reference Python 环境运行工具测试；配置 CLI 在无第三方依赖时仍可校验。
- 使用 Release CPU 构建运行相关 CTest（CUDA/Metal 关闭）；使用小型 fixture 验证数值与拒绝路径。
- 运行文档本地链接/双语检查和 `git diff --check`。保持历史策略及验收证据不变；清理本轮不必要中间产物。
- 不声称 COCO 质量、GPU 内核精度或性能收益；这些仍由后续主动选择的独立 benchmark 提供参考。

## 结果

### 已实现的行为

- `tools/quantize/weight_policy.py` 解析基础 Q 格式与模块覆盖，补齐四模块并计算规范策略 SHA-256；配置 schema 2、示例 JSON 和标准库 CLI 共用同一策略。schema 1 的原有输出与选择语义保留。
- 转换从原始 F32/complex64 参数直接编码，拒绝已降精度源权重，继续使用固定 GGML 行量化器与 Q8_0 编码器。SAM schema 5 保存基础格式、完整规范 CSV 与策略哈希，逐张量 manifest 保存请求／实际类型、模块选择器和保护／回退原因。全部 F32 覆盖无需量化器，且不会错误宣称量化算术。
- `src/models/sam3/weights.hpp` 独立解析/核验策略，模型加载按规范张量名、形状、真实类型与范围校验后再分配 backend 权重。私有 `common/sha256.hpp` 使用标准库计算小型元数据身份，避免给公共 SDK 增加密码库依赖；已验证空串、短串、跨 block 向量及 Python→C++ 策略哈希一致性。
- `ModelInfo` 末尾追加 schema-5 基础格式、模块格式和策略身份，保留原聚合初始化位置；native JSON、质量导出和独立性能记录检查相同字段。计算/缓存请求与存储分配分开，内核内部算术仍为未采集证据。
- 保护与回退复用 schema-4 规则；未增加逐层/逐张量覆盖或模块 F16。模块覆盖的回退说明取自实际 vision 格式，基础 Q8 + vision K 以及基础 K + 全部 F32 两种边界都已覆盖。

### 验证结果与范围

- Release CPU 完整构建通过，包含公开/私有/支持头的独立重复包含检查和两个 TU 链接。配置为 CUDA/Metal/BLAS 关闭、tools/tests 打开；当前验证构建为 `build/cpu-precision-tools/`（本机忽略产物，干净 checkout 不附带）。
- 完整 CPU CTest **18/18 通过**，包括新混合策略、原量化/合同/graph/两 TU/视频小型行为回归；7.36 秒。未读取原始大模型做完整推理。
- reference 环境完整 Python 工具测试最终 **223 项：222 通过、1 个原有条件跳过**，62.258 秒。新增 8 项测试包括 2,500 个完整模块分配、严格配置/metadata 边界、真实 GGML Q6/Q5/Q4 编码、Q8/F32 混合、全 F32 覆盖、native CPU 互通和记录身份拒绝。第一轮完整测试 61.799 秒通过后，修正了 schema-5 manifest 的回退说明，并完成针对性及最终完整复验。
- 针对完整 SAM 3 的 **1,133 个规范张量**，Python 与 C++ 独立分配统计一致：示例 vision Q4/text Q8/fusion Q6/decoder Q5 得到 F32=785、Q4_K=96、Q8_0=129、Q6_K=36、Q5_K=87。这里只检查真实模型 schema 与类型规则，不分配完整模型权重。
- 一个由真实原始命名 F32 参数转换的 10 张量小型 fixture 同时包含五种存储类型。固定 native encoder 产生 K payload，Python 独立解码检查，C++ 独立读取策略、检查布局、在一个 CPU graph 执行混合矩阵运算；相对解码后的高精度标量乘积，最大绝对误差 **9.61252e-06**。fixture 替换了 SAM 参数形状及 tokenizer 来源，不代表完整 SAM 推理或质量证明。
- `python3 -S` 可解析示例 schema 2，不加载第三方模型依赖或 GPU。双语表格/本地链接检查与已跟踪/新文件 whitespace 检查通过。
- 两份归档策略以及此前 GPU 中断 receipt 列出的 8 份完成记录 SHA-256 不变；未写入旧模型、历史 receipt 或原始 checkpoint。测试临时目录自动清理，保留当前 CPU 构建与本轮小型证据。

### 本机不可变证据

`build/module-mixed-20261010-162508/` 为本机 Git 忽略目录，干净 checkout 不附带；当前约 1.2 MB：

- `cpu-ctest.json`：完整 18 项 CPU 结果。
- `python-tools.json`、`python-tools-final.json`：第一轮与边界修正后的最终完整工具结果。
- `resolved-configuration.json`：标准库解析的混合 CPU 示例。
- `fixture-q4-q8-q6-q5-f32/`：约 55 KB F32 fixture checkpoint、约 1.16 MB GGUF、conversion manifest、配置、只读精度报告及 `cpu-verification.json`。该目录是测试证据，不是可用于生产的完整 SAM 模型。
- `historical-policies.json`、`historical-receipts.json`：历史身份不变的检查结果。

本轮没有运行 GPU 验收、完整模型转换、COCO 质量评估或性能测量。上述 CPU 结果不能推出整模型质量、加速收益或 CUDA/Metal 硬件结论。M4 硬件测量、M5 逐张量/混合 F16 与独立激活量化继续留在后续计划。
