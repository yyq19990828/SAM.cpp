# Metal 量化硬件验收入口

日期：2026-10-10，Asia/Shanghai。状态：显式入口及 Linux 回归完成；无可用 Mac，实机验收保持待办。

## 范围与方法

收尾审计确认 [后续计划](20261010-212901-quantization-followups-implementation.md)的 Linux/CUDA 项目已有封存证据。仍缺 M4/M5 Metal 实机结果；现有量化测试在默认入口隐式探测 Metal，混合 F16 小图没有显式 Metal 入口。补齐可在未来 Mac 上运行的严格入口，不把 Linux 编译或无设备检查当作 Metal 通过。

## 步骤

1. 为原有原生量化算术及混合权重小图增加显式 `metal` 参数；复用现有独立 scalar、QKV view、4736 行宽、常驻字节及严格放置检查。
2. 混合设备测试分别执行 F32/Q 和 F16/F32/Q fixture，保留现有 CPU/CUDA 数值门槛。Metal 仍必须零 CPU/BLAS/CUDA 计算回退。
3. 在实际 Metal GGML target 存在时注册独立 CTest；新增 required-device 选项，使无设备返回码不能在正式验收中被记作跳过。
4. 新建 Linux 验证构建，检查默认 CPU、CUDA 既有路径、无 Metal 设备以及不含 Metal target 的 required-device 配置拒绝。既有封存构建、回执及机器证据保持不变。
5. 记录实际结果、未来 Mac 命令与硬件边界，更新 changelog；验证后例行清理可再生中间产物。

## 验证与边界

- Linux 只证明测试入口编译、既有路径无回归，以及缺设备不会伪装成功；不能证明 Metal 编译、运行或完整模型结果。
- 验证证据使用新的 Git-ignored `build/metal-quantization-entry-20261010/`，仅当前工作站可用；保存源码身份和实际命令/返回码。
- `tools/check_docs.py` 与 `git diff --check`。
- 未来 Mac 必须执行实际 required-device CTest，随后用已有原始 F32 参考及同一 GGUF/manifest 做有界完整模型比较和独立性能测量。小图通过不等于完整模型验收。

## 结果

- 新增 `test_quantization metal`：实际探测 MTL device，初始化精度自检，严格 scheduler 放置，以及四种 Q、QKV view、M-tail、4736 行宽、N=3/33 的既有算术检查。默认入口的历史行为保留。
- `test_mixed_quantization metal` 和既有 CUDA 入口现在分别运行 F32/Q、F16/F32/Q 两份 fixture；复用独立 scalar、两次不同 RHS、常驻类型/字节不变检查。Metal 分支要求 Metal 节点非零且 CPU/BLAS/CUDA 计算节点为零，没有放宽数值容差。
- CMake 以实际 `ggml-metal` 或 `ggml::ggml-metal` target 注册 `sam_quantization_metal`、`sam_mixed_quantization_metal`。`SAM_REQUIRE_METAL_TESTS=ON` 在没有 target 时配置失败；存在 target 时不设置 skip-return-code，所以无设备返回 77 会导致验收失败。普通模式允许跳过。
- 新建 Release CPU/OpenBLAS 构建 `build/metal-entry-cpu-blas-current/`，两个量化 CTest **2/2** 通过；全部私有头独立/重复包含编译通过。首轮编译发现一个重命名遗漏，修正后重跑通过；失败日志和当时源码保留。
- 新建 caller-owned CUDA 验证构建，引用已封存 CUDA GGML 动态库，重新编译 SAM 和两个测试，未重建旧库。CPU/CUDA 四项量化 CTest **4/4** 通过，required CUDA 零跳过；这是本次测试入口的专项回归，不替代此前完整 **39/39** 回执。
- 两个测试在 CPU、CUDA 构建上显式选择 Metal 均返回 **77**。Linux 无 Metal target 的 required 配置按预期返回 **1**。
- 单独使用**仅含接口占位 target 的合成 CMake fixture**验证注册语义：没有 Metal 库、编译器或设备，也没有执行 Metal 算子；required 模式两个 CTest 均失败（CTest 返回 **8**），普通模式两个均标记 skipped（返回 **0**）。该 fixture 只能证明无设备验收不会伪装成功。
- 文档检查与 `git diff --check` 通过。证据目录保留实际命令、返回码、源码归档、编译/CTest 日志及 `verification-receipt.json`；机器证据仅当前工作站可用。临时合成 fixture 构建及失败配置构建在检查后清理，保留日志/配置和可用 CPU/CUDA 专项构建。

未来 Mac 首先运行（本机未执行）：

```sh
cmake -S . -B build/metal-quantization-acceptance -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_CUDA=OFF \
  -DSAM_BUILD_TESTS=ON -DSAM_REQUIRE_METAL_TESTS=ON
cmake --build build/metal-quantization-acceptance --parallel
ctest --test-dir build/metal-quantization-acceptance --output-on-failure
```

只有 Mac 上实际执行及随后的有界完整模型输出、严格放置与独立性能证据可以关闭 M4/M5 Metal。用户明确暂无可用 Mac，当前不把入口准备标记为硬件验收完成。
