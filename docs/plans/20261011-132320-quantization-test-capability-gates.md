# 量化测试的平台能力判断

日期：2026-10-11，Asia/Shanghai。状态：实现及本机验证完成。承接用户要求提交推送前的复核。

## 范围与依据

复核发现两个测试兼容性问题，量化生产实现和此前封存的数值/性能证据不改写：

- CPU tiled 实现只有 AVX/AVX2/AVX512VNNI 构建可用，`GGML_CPU_TILED_MM_FORCE` 不能让 ARM 或非 AVX 构建执行该路径；当前 CTest 无条件要求 tiled event，会导致这些平台失败。
- 显式激活模式要求 CUDA compute capability >= 8.9 且不是 FORCE_CUBLAS；当前测试只检查有无 CUDA 设备，导致不支持这个可选模式的普通 CUDA 构建失败。

## 方法与步骤

1. 新建非 native、无 AVX 的 CPU 构建，先记录旧 tiled 检查真实失败。用独立注入的 configure=false registry capability 响应记录旧激活测试失败；该注入不是旧 GPU 或 FORCE_CUBLAS 的硬件执行证据。
2. CPU probe 增加只读 tiled 能力查询，CTest 仅在明确没有编译该路径时标记跳过；支持该路径时仍须实际抓到 tiled event，dot 检查继续执行。
3. 激活测试先调用版本化扩展的真实 configure 能力检查；不支持时验证运行时明确拒绝再跳过。通用 required-CUDA 仍使缺设备失败；新增专用 required-activation 验收开关，要求该可选路径实际运行。
4. 使用新构建/新输出目录完成非 AVX CPU 完整 CTest、既有 AVX dot/tiled 及实际 CUDA 激活专项，检查支持/不支持/缺设备以及专用 required 开关语义。复用不变生产实现的旧完整验收时注明来源。
5. 更新 README/changelog 与本计划结果，核对源码和证据身份；例行清理可再生临时产物，再提交并推送用户授权范围。

## 验证边界

本地证据保存到 Git-ignored `build/quantization-push-checks-20261011/`，仅当前工作站可用。新测试不得改写之前的回执/构建或 tensor dumps。Linux AVX 能力判断不能替代 Apple 编译与 M4/M5 Metal 实机验收；注入 capability 拒绝不能替代旧 NVIDIA 硬件验收。

## 结果

- 非 AVX Release 构建已真实复现旧 tiled 检查失败；注入 configure=false 后旧激活测试也失败。修复分别增加只读 CPU 编译能力查询，以及激活扩展的能力检查和运行时拒绝验证，没有修改量化生产实现。
- 新建 `build/quantization-push-cpu-generic/`，关闭 native、AVX/AVX2/AVX512/VNNI/F16C/FMA/SSE42/BMI2、BLAS、CUDA、Metal 和 LLAMAFILE。完整构建及头文件检查通过；CTest 注册 22 项，21 项通过、仅 tiled 项明确跳过，dot 项实际执行。隔离 reference 工具测试 **253/253**、零跳过，使用本次构建的量化器和 validators。
- 新建 AVX/OpenBLAS 专项构建，只读链接之前封存的 GGML 库，重新编译当前 SAM 和 probe；dot/tiled **2/2** 实际通过、零跳过，各自仍验证 96 组实际 payload 和独立 scalar 输出。
- 新建 CUDA 专项构建，只读链接之前封存的 GGML 库，重新编译当前 SAM 和激活测试；RTX 4090 上 required-CUDA 与 `SAM_REQUIRE_NATIVE_ACTIVATION_TESTS=ON` 的实际激活检查 **1/1** 通过、零跳过。原有 18 组 payload、14,430 个输出检查保留。
- 独立注入的 configure=false 和缺 CUDA 设备响应验证了全部门禁：可选激活不支持时先验证运行时拒绝再跳过，专用 required-activation 时失败；required-CUDA 缺设备时失败，普通缺设备时跳过。这些负例只证明测试语义，不宣称已在旧 GPU 或 FORCE_CUBLAS 硬件上执行。
- README/changelog 已说明两个能力边界与专用严格开关。此前完整 CPU/OpenBLAS **23/23**、required-CUDA **39/39** 和原始 checkpoint 有界验证继续引用各原计划及不变回执；本轮差异限于测试、诊断 probe 和文档，不把旧完整测试冒充为本轮重跑。
- 命令、预期失败、成功日志、源码归档、二进制及依赖身份保存于 `build/quantization-push-checks-20261011/`，仅当前工作站可用。M4/M5 Metal 仍待 Mac 实机验收，v2/v3 不恢复。
