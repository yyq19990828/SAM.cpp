# 转换预览、逐张量混合策略与执行开销报告

日期：2026-10-10，Asia/Shanghai。状态：三项实现及转换／CPU 验证完成。

## 范围

前序模块级混合权重量化已提交为 `dccdee3`，包含对应计划与 CPU 验证记录。本轮落实转换前预览、逐张量精确覆盖与实际执行开销报告，先验证转换和 CPU 小型图；不运行完整模型转换、COCO 或大规模 GPU 验收，不恢复此前中断的 campaign。

质量仍是应用图片上的参考比较与可选建议，性能独立测量。格式、策略和算子正确性属于实现合同，不把质量差异或速度回退变成通用否决条件。

## 方法与合同

1. **逐张量策略**：增加配置 schema 3、SAM GGUF schema 6 与 `image-tensor-mixed-linear-v1`。精确规范张量名覆盖的优先级为 tensor > module > base，支持 F32/Q8_0/Q6_K/Q5_K/Q4_K。每条覆盖必须命中当前规范清单中的可量化线性矩阵；未知名、保护张量、非法格式、重复 JSON 键和空覆盖清单报错。保留保护及 vision K 行宽 4736 的 Q8 回退。暂不增加 F16、正则、激活模式或原生量化 CPU 算术模式。已有配置 schema 1/2 和模型 schema 1–5 保持语义。
2. **策略身份与加载**：用有界规范 CSV 保存精确覆盖，版本化 ASCII 编码生成策略 SHA-256。Python 与 C++ 独立校验，原生加载在权重分配前验证覆盖、完整张量名/形状/类型。manifest、ModelInfo、精度描述、导出与性能配方绑定同一策略，不能把混合策略简化成全局类型。
3. **转换预览**：共用真实转换的逐张量分配函数，从规范 schema 生成计划，支持检查原始 checkpoint 的名称/形状/dtype。无 checkpoint 时明确标为 schema 估算，不读取大权重或启动 backend。报告请求/最终类型、选择器、保护/回退原因、各模块/类型元素与字节、预计 GGUF 文件大小及磁盘预算；区分可精确计算的 payload 与 metadata/tokenizer、manifest、转换暂存和运行内存的估算/未知。可选 checkpoint/tokenizer 预检提供更精确结果。输出使用新文件，不生成模型或量化 payload，不要求 native encoder。
4. **执行开销**：增加独立、显式启用的 CPU 诊断工具/模式，共用生产 graph 和 scheduler。记录实际图节点、真实输入类型与 backend、CPU 权重解码、矩阵乘和其他算子的同步 wall time，以及上传/下载和图整体开销。节点观察会改变调度，因此必须标为有观察开销的诊断运行；不能混入无 observer 的独立性能 benchmark。内核内部 RHS 打包、整数累加和未独立计量成本保留 `NOT_COLLECTED`，不能用残差或配置值伪造。不得对未测试 GPU 宣称成本或精度。

## 步骤

1. 扩展依赖最少的策略/配置解析，添加明确的逐张量 CPU 示例和边界测试。
2. 共用分配清单实现转换预览，再接入真实转换、GGUF schema 6、manifest 与原生读取验证。
3. 更新模型信息和精度/benchmark 配方身份，避免配置、实际存储和报告不一致。
4. 实现 opt-in CPU 开销采集、独立报告入口和小型真实混合 graph；保留原有生产/benchmark 默认行为。
5. 验证预览与真实输出一致、跨语言策略一致、实际 CPU 输出正确和诊断分类可信；更新双语指南、格式合同、tools 文档及 changelog。

## 验证与产物

- reference 环境工具测试涵盖旧策略不变、覆盖优先级、零未命中、保护规则、回退、篡改/错误模型、预览未生成 payload，以及预览字节和真实 GGUF 对照。
- Release CPU 构建（CUDA/Metal 关闭）及 CTest，包括小矩阵独立解码参考、图复用、计时观察和 public/private 头独立重复包含与两 TU 链接。
- 实际转换仅使用小型原始命名 F32 fixture；开销测量仅使用小图，绝不解释为整模型质量或加速证明。
- 完整 `tools/test_tools.py`、文档检查与 `git diff --check`。验证后清理不需要的本轮临时文件，保留当前构建和新目录中的不可变结果；历史 receipt、模型、checkpoint 不变。

## 后续结果

### 实现

- `tools/quantize/tensor_policy.py` 集中精确名称、规范层索引与保护规则；配置 schema 3、GGUF schema 6 和 `image-tensor-mixed-linear-v1` 已贯通转换、原生预分配校验、`ModelInfo`、导出／精度检查和性能配方。配置 schema 1/2、GGUF schema 1–5 的解释及 schema-5 策略哈希保持不变。
- `tools/convert/conversion_plan.py` 和 `sam3_gguf.tensor_assignment` 共用分配。`convert_sam3.py --dry-run` 无 checkpoint 时只读取固定清单，提供可选 mmap 名称／shape／dtype 预检及 BPE 精确大小计算。报告区分 payload、metadata 上下界、编码 scratch、未知 manifest 开销和非峰值的 F32 cast 清单。真实转换先登记布局、后编码一次，并在发布前检查类型、字节及精确文件大小。编码器需求按最终张量类型决定，K 请求仅产生 Q8 回退时无需 native encoder。
- 新增独立 CPU `sam_execution_cost_probe` 和 `quantization_benchmark.py execution-cost`。observer 共用生产 graph/scheduler，逐节点同步采集实际类型、backend、Q→F32、matmul、其他算子、metadata 及传输，记录 bind／compute wall time。传输包括文本 token／mask、图像缓存和 detector 零输入。每次 compute 后移除 scheduler 回调，普通 benchmark 不启用观察。Python 报告绑定模型、manifest、二进制／动态库、图片及配置哈希，检查逐节点／汇总和传输计数一致性。
- 完整模型工具路径已构建；本轮没有用完整权重运行它。原生数值和计时验证使用真实小型 CPU graph；完整工具的进程／报告边界另外使用 stub producer 测试，不能把该 stub 当成整模型执行证据。
- 已更新双语配置／量化／benchmark 指南、GGUF 合同、README、tools 文档、schema-3 示例和 changelog。模块 F16、模式匹配、激活量化、原生量化 CPU 算术和 GPU 成本测量仍独立推进。

### 验证结果与保存范围

- Release CPU 构建完成：CUDA、Metal、BLAS 和 CUDA probes 关闭。公共／私有头独立重复包含及两 TU 目标编译通过；CTest **19/19**，9.67 秒，没有配置完整 SAM 参考权重测试。
- 最终 isolated `tools/test_tools.py`：**230 tests，229 passed，1 skipped**，92.306 秒。跳过项为环境无可用 CUDA 的 runtime study；没有执行 GPU 验收。覆盖逐张量优先级、保护／未知／重复／未命中规则、跨语言哈希、旧策略、回退与 encoder 选择、预览／产物、单次 K 编码，以及报告边界与篡改拒绝。
- 标准 **1133 张量**预览无需大权重／encoder/backend，约 0.3 秒生成；保存后按最终分配代码重新核对，未改写原文件。
- 原始命名 F32 小型 checkpoint 的 10 张量真实转换包含 **F32×5、Q8_0×2、Q6_K×1、Q5_K×1、Q4_K×1**。预览与 GGUF 类型／字节逐项一致，文件精确为 **1,164,224 bytes**。测试确认 3 个 K 矩阵各编码一次。
- 原生 CPU 独立解码和 scalar 乘加参考对照：最大绝对误差 **9.61252e-06**，resident packed 类型不变。实际观察到 **10 次 matmul、5 次 Q→F32 cast**；输入／输出传输 **78,384 / 216 bytes**。该小图的计时包含观察与同步开销，仅验证采集路径，不形成整模型收益结论。
- 文档检查 **PASS：157 documents**；`git diff --check` 及 10 个新增源码／文档文件的空白检查通过。

本轮本地证据保存在 Git-ignored `build/tensor-policy-20261010-171252/`，仅在当前验证机器可用：`schema-preview.json`、`fixture/preview.json`、小型 checkpoint/GGUF/manifest、`fixture/cpu-cost.json`、`fixture-verification.json`、`validate_fixture.py`、最终 `tools-tests.log`、`ctest.log` 和 `verification-receipt.json`。fixture 明确替换标准 SAM shape 清单及 tokenizer 来源，其 `scope.json` 标明并非完整 SAM adapter／官方 tokenizer／质量或加速验收。原始大权重、可用模型、历史回执未改写；本轮没有下载／复制大模型或数据集，临时转换及 native 编码文件已自动清理。
