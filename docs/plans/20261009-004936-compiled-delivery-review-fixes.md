# 编译库交付审核问题修复

时间：2026-10-09 00:49:36（Asia/Shanghai）

## 范围与基线

- 修复基线：`8330160a0f4f42077f84aadc724eed843f3df188`。
- 审核比较基线：`origin/main` 的 `113e165e78970cab3713e6e7b3b325c8a16c7561`。
- 处理审核确认的四项问题：分组 Python 脚本启动失败；全静态 SDK 的纯 C++ 消费失败；外部 `ggml::ggml` 与工具构建不兼容；视频 benchmark 漏记 `src/` 源码身份。
- 保持公开 API、模型计算、GGML 固定版本及精度补丁不变。保留现有不可变验收记录和 Orca 状态文件。

## 方案与步骤

1. 为直接执行的分组 Python 入口建立仓库包搜索路径，保持模块导入、历史入口和子进程调用一致；在干净环境、不同工作目录下验证真实命令入口。
2. 调整安装包的 OpenMP 链接依赖，使仅启用 C++ 的消费者不需要额外启用 C 编译器；保留静态最终链接要求并用真实安装包验证。
3. 让工具和关联后端检查复用 CMake 已解析的 GGML target，验证父工程通过已安装 GGML 包构建工具。
4. 视频 benchmark 复用覆盖完整实现目录的源码身份规则，验证修改实现会改变身份并被运行期检查拒绝。
5. 先运行回归检查确认旧行为失败，再修复并执行正向检查。更新 changelog 和必要的调用说明，完成修复差异复审。

## 验证

- Python 入口：分组脚本、历史入口及模块方式的帮助命令；量化参考导出的子进程入口；不依赖外部 `PYTHONPATH`。
- 源码身份：包含 `src/` 中 API、模型、runtime 和构建文件；实现变更负例。
- CPU Release 构建、CTest、隔离参考环境 `tools/test_tools.py`、文档检查和 `git diff --check`。
- 全静态及动态安装包的纯 C++ consumer；动态安装前缀迁移；父工程 imported GGML + `SAM_BUILD_TOOLS=ON`。
- 所有新构建和复现输出使用本次独立目录；完成验证后清理本次不再需要的中间文件。
- 本轮不修改数值实现，不把 CPU/安装验证当作新的 CUDA、Metal 或原始权重数值验收。

## 结果

四项修复已完成。CMake 修复由指定的 OpenCode 2 / `deepseek/deepseek-flash#max` 子代理实施，主代理完成 Python 修复、整体验证和差异复审。

### 实现

- 31 个分组 CLI 在直接运行时将仓库根目录加入 `sys.path`；包导入和 `python -m` 不改变搜索路径。历史入口保持转发，`export_quantized_reference.py` 启动的 `export_reference.py` 子脚本也可自行解析仓库包。
- 已编译 GGML 的 OpenMP 最终链接接口按消费者实际定义的语言 target 解析；安装包仍要求 OpenMP，且拒绝没有任何 OpenMP 语言 target 的异常 finder。纯 C++ 消费者链接真实 OpenMP runtime，无须启用 C 或伪造 C target。
- 量化工具的库身份复用 `SAM_GGML_LINK_TARGET`；工具和测试识别 `ggml-cuda` 与 `ggml::ggml-cuda`。消费测试新增 imported GGML 模式，安装消费测试默认仅启用 C++。
- 视频 benchmark 复用 `source_snapshot()`，同时保留 fixture、qualification、validation 和 recipe 身份。新行为测试通过模拟推理验证源码记录及运行中变更拒绝，不作为模型数值证据。
- 已更新 `changelog.md` 的 `Unreleased/Fixed` 和工具调用说明。

### 回归与交付验证

新证据目录为 `build/fix-20261009-004936/`，审核阶段的 `build/review-20261009-003854/` 保持只读。

| 检查 | 结果 | 证据（相对新证据目录） |
| --- | --- | --- |
| Python 修复前负例 | 31 个分组入口启动失败；视频回执不含编译实现，修改实现后仍错误通过 | `python-regressions-red.log` |
| Python 定向修复后 | 31 个分组入口从仓库外运行且清除 `PYTHONPATH` 后通过；3 个代表入口的分组、历史、模块帮助输出一致；模拟推理中的源码变更被拒绝 | `python-regressions-green.log` |
| 完整 Python 测试 | 148/148 通过；包含上述行为回归 | `python-suite-final.log` |
| CPU Release | 新构建成功，包含公共/私有独立头检查；CTest 17/17 通过 | `cpu-build.log`、`cpu-ctest.log` |
| 实际源码覆盖 | `src/` 的 52/52 个 C++/CUDA 源码与头文件全部纳入；完整源码快照 225 个文件 | `source-snapshot.json` |
| 全静态修复前负例 | 真实安装包的 CXX-only consumer 配置失败：缺少 `OpenMP::OpenMP_C` | `cmake-worker/red/install-cxx-static-configure.log` |
| 全静态修复后 | `BUILD_SHARED_LIBS=OFF`、OpenMP ON 的新包：纯 C++ 与 C+C++ consumer 均配置、构建及 CTest 1/1 通过；纯 C++ 最终链接和运行时加载确认包含 `libgomp` | `cmake-worker/green/install-cxx-static-2/`、`cmake-worker/green/install-c-cxx-static-2/` |
| 默认安装组合 | 静态 SAM + 共享 GGML 的纯 C++ consumer 通过 | `default-consumer-ctest.log` |
| 共享安装组合 | 全共享 SDK 安装后移动前缀，再配置、构建并运行纯 C++ consumer，CTest 1/1 通过；未设置动态库搜索环境变量 | `shared-consumer-ctest.log` |
| Imported GGML 修复前负例 | 实际安装的 `ggml::ggml` + 工具 ON 配置失败：`TARGET_FILE:ggml` 不存在 | `cmake-worker/red/imported-with-tools-configure.log` |
| Imported GGML 修复后 | 配置和全部工具构建成功，consumer CTest 2/2 通过；量化工具 `--identity` 成功并指向 imported 静态库 | `cmake-worker/green/imported-with-tools/` |
| Namespaced CUDA target | 已安装 CUDA GGML 包配置成功；必需 CUDA 测试正确注册，仅配置验证 | `cmake-worker/green/cuda-imported-detection-reconfigure.log` |
| 文档与差异 | 71 份文档检查通过；`git diff --check` 无问题；复审确认无模型计算实现改动 | `docs-check.log`、`diff-check.log` |

主验证命令：

```sh
rtk proxy cmake -S . -B build/fix-20261009-004936/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_GGML=/home/john/.x-repo/github.com/ggml-org/ggml \
  -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF
rtk proxy cmake --build build/fix-20261009-004936/cpu -j 4
rtk proxy ctest --test-dir build/fix-20261009-004936/cpu --output-on-failure
rtk proxy .venv-reference/bin/python -B tools/test_tools.py
rtk proxy python3 tools/maintenance/check_docs.py
rtk proxy git diff --check
```

全静态和共享 SDK 分别使用 `BUILD_SHARED_LIBS=OFF/ON` 的独立构建目录；安装 consumer 使用 `tests/install_consumer`，外部 GGML consumer 使用 `tests/consumer` 的 `SAM_CONSUMER_IMPORTED_GGML=ON`。GGML 保持固定提交 `353b63b439f27ab2cc19dac97ab1681ba6d2d084` 及原精度补丁。

本轮验证平台为 Linux CPU；没有重跑 CUDA/Metal 推理、原始权重数值或性能验收，历史不可变回执未更新。
