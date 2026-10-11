# 量化剩余功能逐项实现与验收

日期：2026-10-10，Asia/Shanghai。起点：`e3edc3c` 及工作区内已完成的有界验收文档。状态：Linux/CUDA 可推进范围已完成，Metal 待实机。

## 范围与依据

按用户要求推进五项剩余工作，保留完整目标，不以首个通过的子项代替全部完成：

1. [CPU 原生量化](20261010-174613-cpu-native-quantized-matmul.md)与实际 BLAS backend 共存的调度、数值及完整模型验证。
2. [混合权重 M4](20261010-155435-mixed-weight-quantization.md)的 Metal 设备验收。
3. 混合权重 M5 的 F16 存储、逐张量提升与设备执行。
4. [独立激活量化 A1–A4](20261010-155435-native-activation-quantization.md)：明确数值合同、原生执行控制、模型接入与配置/API。
5. 内核内部精度及阶段成本：CPU scratch/打包/算术采集与独立 GPU 成本诊断。

v2/v3 campaign 已归档终止，不属于待验收项，不恢复旧 campaign。已有 checkpoint、GGUF、manifest 和机器可读证据保持不变。质量差异与性能变化独立报告，不增加通用 COCO 门槛。

## 方法与步骤

- 先检查当前源码、工作区和依赖，例行清理可确认过时的缓存；复用固定 GGML v0.26.0 源码及现有 OpenBLAS 依赖，创建独立 CPU+BLAS 构建。
- CPU+BLAS 测试必须实际发现 BLAS backend，并运行足以触发 BLAS 的浮点矩阵作为正控制；在同一图中验证 packed Q 矩阵仍由 CPU 执行，默认解码路径可使用 BLAS。比较独立 scalar 参考、图复用和完整模型输出，保存身份与放置证据。
- Metal 先核查设备入口、严格放置及数值测试。只有实际 Apple/Metal 设备执行才完成硬件验收；当前 Linux/CUDA 结果不能代替，硬件缺口不妨碍其他项目实施。
- 混合 F16 使用显式版本化格式合同，保持旧策略/哈希解释；转换、预览、加载前校验、按实际张量提升、ModelInfo、精度/质量/性能报告和双语文档一起更新。原始 checkpoint 直接编码，保护与行宽回退继续有效。
- 独立激活量化按 A1→A4 推进，初始目标为 CUDA Q8_0 vision linear 的显式动态 Q8 RHS 策略；明确 MMVQ/MMQ 数值与布局，拒绝无法保证的形状/计算组合。实际执行与政策请求分开记录，缓存身份包含会改变结果的策略。
- 内核诊断为显式 opt-in，固定 patch 与原生扩展接口；捕获实际 scratch/路径及阶段计时，诊断 observer 不进入独立性能测量。GPU 使用匹配设备与同步计时，未知阶段保持未知，禁止从残差或源码 traits 冒充抓取。

## 验证与保存

- 每项运行与行为相符的 Release CTest、独立数值/放置检查、隔离 reference 工具测试及相关完整模型有界验证；公共头独立重复包含及两 TU 链接保留。
- 格式变更验证旧 schema/策略身份、非法与损坏输入分配前拒绝、预览/实际类型及字节一致、配置/模型/报告身份一致；CUDA 使用 required-device 检查。
- 性能使用无诊断的配对独立进程，报告输入、硬件、构建、线程、原始样本和进程峰值；不把局部核函数速度当完整模型收益。
- `tools/check_docs.py` 和 `git diff --check`；每次验证后清理无用中间产物，保留当前可用构建及所需证据。新证据位于 Git-ignored 的 `build/quantization-followups-20261010-212901/`，仅当前工作站可用。
- 本计划持续记录各项实现、实际执行结果和仍缺的证据。硬件未验或功能未实现时不标记整体完成。

## 后续结果

### CPU/BLAS 调度缺陷与小型验证

- 原配置已启用系统 OpenBLAS，使用现有 0.3.32 开发头文件；没有安装或替换系统依赖。新的 `build/quant-cpu-blas-current/` 为 Release、BLAS 开启、CUDA/Metal/LLAMAFILE 关闭、GGML_NATIVE 开启。
- 新增 `sam_cpu_quantized_blas` 回归首先在未修复源码上实际失败：`SAM CPU graph attempted compute fallback for MUL_MAT`。`GGML_SCHED_DEBUG=2` 日志确认 native packed Q 输入被分配给 BLAS，浮点正控制也实际进入 BLAS，排除“BLAS 未启用”的假通过。
- 根因是固定 GGML 的 `ggml_backend_sched_split_graph` 第三轮同 buffer-type 优先级提升覆盖显式 CPU 指派。新增独立、固定哈希的 `ggml-explicit-backend.patch`：在调度展开前保存显式指派，第三轮不提升这些节点；普通浮点节点继续采用 BLAS。没有删除执行前的严格 placement 检查。
- 同类路径核查：第三轮提升只有一个循环；SAM 的实际绑定与 size-only reserve probe 都调用共享 `assign_primary_compute`，因此补丁覆盖两者。CUDA／量化 Metal 的显式放置使用相同 API，仍需本轮相应回归证据。外部提供的 GGML 也必须遵循显式放置合同。
- CMake 校验补丁与完整源码树，build identity 包含第四个补丁；原始源码、旧三补丁及更早归档均支持只读升级。集成测试新增三补丁归档升级和输入不变检查；转换器接受新的确切 encoder identity，同时保留此前已知 identity，未重编码已有 GGUF。
- 修复后的实际 CPU/BLAS **34 组**共存验证通过：四 Q 格式、F32-decode/native 两模式、1/4 线程、QKV 视图、4736 行宽和更换 RHS 后图复用。相同图内浮点正控制为 BLAS，native Q 为 CPU 且零 Q→F32 cast；默认解码矩阵为 BLAS。全部输出独立 scalar 检查，最大 Q 参考误差 **8.85484e-06**，F32 控制误差 **6.9584e-06**；常驻 packed 字节不变。
- 完整 Release CPU+BLAS CTest **21/21**，6.03 秒；隔离 reference 工具 **235/235**，55.133 秒、零跳过，使用新构建的实际量化器和 native validators。负例打印的 `FAIL` 行属于预期 fixture，suite 返回成功。
- 原始失败、调度日志及修复后 JSON/构建/CTest/工具日志保存到本计划证据目录，失败的 JSON 是未封口的部分输出，不能当 completed report。完整模型参考／诊断／独立性能仍在串行执行，未在此预先宣称通过。

### CPU/BLAS 完整模型验证

- 同一份 schema-6 混合 GGUF 在新的实际 BLAS 构建中，默认 F32 解码与 native-quantized 均已完成两张图片、六组提示的原始 F32 oracle 比较：六个参考对象全部匹配，零漏检／新增；三组空结果保持为空。该样本只作应用回归，不是独立 holdout 或生产准确率认证。
- `decode-cost/report.json` 的完整图有 519 次 MUL_MAT，其中 BLAS 435、CPU 84；285 次 Q→F32 cast。`native-cost/report.json` 同样有 519 次 MUL_MAT，其中 BLAS 150、CPU 369；285 次 packed Q MUL_MAT 全部为 CPU，零 Q→F32 cast。普通浮点矩阵仍实际使用 BLAS。
- 两份成本报告都是序列化 observer 诊断，不能用于独立性能比较。无 observer 的配对性能仍在执行；共享调度补丁的 CUDA required-device 回归尚待完成。
- 无 observer 的配对性能及离线重新汇总已完成，报告均为 VALID、无污染。同一混合 GGUF，i7-12700KF、CPU/OpenBLAS 0.3.32、4 线程，一组进程对、warmup 1、iterations 3：完整图 p50 默认解码 **17119.50 ms**、native **19317.64 ms**（native 慢 **12.84%**）；更换提示 p50 **2348.72→2390.19 ms**；重复结果 **11.082→10.993 ms**。进程峰值 RSS **2414628864→2395582464 bytes**（约少 **0.79%**）。该组合不支持“native 一定更快”的结论，CPU 不启用 BLAS 时此前的收益不能套用到此组合；默认模式保持不变。
- 六对象加权 mask IoU 平均／最小：默认解码 **0.998279/0.996401**，native **0.998125/0.995772**，均相对原始 F32 oracle。CPU 共存验收通过；独立 CUDA 构建随后回归共享补丁，不覆盖此前机器证据绑定的二进制。
- 前序 CPU producer 已封存到 `cpu-blas-verification-receipt.json`：四个导出、327 个输入身份、268 份归档源码和 1723 个本地证据文件，保留原始失败及全部日志。该回执封口时 CUDA 回归仍待完成，其历史字段不改写。
- 后续独立 `build/quant-cuda-followups-current/` 已使用同一第四调度补丁与本轮 F16 源码完成 required-device Release CTest **35/35**，零跳过；证据为 `build/mixed-f16-acceptance-20261010/cuda-final3-ctest.log` 和对应 status。包括原 34 项和新增混合 F16/Q CUDA 小图。CPU/BLAS 这项已完成，其他 ISA 或更多图片不继承本样本收益。

### 其他项目状态

- 用户确认当前无可用 Mac。Metal 保留为实际硬件待验项目，不以 Linux 检查代替。
- 混合 F16 已实现 schema 4/7、转换／逐张量分配／原生小图与报告，CPU CTest **21/21**、CUDA **35/35**、工具 **241/241** 零跳过。原始 checkpoint 完整转换、两图片四模式输出与独立性能已[完成有界验收](20261010-213528-mixed-f16-weight-implementation.md)。所有模式六对象全匹配、零漏检／新增；CPU RSS +35.85%、CUDA 设备峰值 +11.52%，没有稳定提速结论。M5 在 CPU/BLAS 和 CUDA 完成，Metal 仍待实机。
- 独立激活 [A1–A4 已实现并完成有界验收](20261010-215837-native-activation-runtime.md)：显式 CUDA Q8_0 vision MMQ D4、公开 API／CLI／配置 schema 5，默认策略及旧 schema 不变。18 份真实 payload、14,430 个输出独立检查；两图六提示六对象全匹配，零漏检／新增，三个空结果不变。每次 vision 编码 128 个请求／实际 MMQ，零回退；与同一 Q8 模型默认输出相同。独立无 observer 两组进程对显示完整图 p50 **388.540→409.060 ms**，强制模式慢 **5.28%**、GPU 峰值相同；校验同步有成本，未默认开启。
- 第五项[内核暂存与 GPU 成本](20261010-220425-kernel-cost-capture.md)已完成声明范围：CPU dot/tiled 各 96 组真实 RHS packing/arithmetic 抓取及 scalar 验证，CUDA F32/F16 完整图成本各一次、全部 CUDA0；CPU/BLAS **23/23**、required-device CUDA **38/38**、工具 **248/248**。GPU 内核分项仍未采集，不能把节点 wall time 当设备 timestamp。
- 最终激活验收 CPU/OpenBLAS **23/23**、required-device CUDA **39/39**、隔离工具 **253/253** 零跳过；证据封存为 `build/native-activation-acceptance-20261010/verification-receipt.json`，406 个输入身份、2698 个本地证据文件，仅当前工作站可用。旧回执与其对应构建保持不变。
- 当前五项中 CPU/BLAS、M5 的 CPU/CUDA、A1–A4 和第五项声明范围已完成；唯一未关闭项是 M4/M5 的 Metal 硬件验收。用户已确认暂无 Mac，先完成其余项目；不将整体硬件目标标为完成，也不恢复 v2/v3 campaign。后续 A5 等更多模式仍属另行扩展。
- 收尾补齐了[显式 Metal 量化验收入口](20261010-225249-metal-quantization-acceptance-entry.md)及 required-device 配置：新增入口的 CPU 专项 2/2、CUDA 构建 CPU/CUDA 专项 4/4 通过，缺设备时严格模式失败、普通模式跳过。此项只证明入口及既有路径，M4/M5 Metal 仍须实际 Mac 执行及完整模型证据。
- 提交前复核修复了[测试的平台能力判断](20261011-132320-quantization-test-capability-gates.md)：非 AVX CPU 完整 CTest 21 项通过、仅 tiled 项跳过，工具 253/253；AVX dot/tiled 2/2、RTX 4090 严格激活专项 1/1 实际通过。缺设备和不支持可选激活模式的失败/跳过门禁已独立验证，不修改此前数值/性能回执，也不替代 Metal 实机验收。
