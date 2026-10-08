# SAM.cpp 编译库迁移技术规格

状态：迁移已实施，节点 D 最终验收通过；实际命令、证据与未验证范围见
[总计划实施记录](plans/20261008-181312-compiled-library-structure.md)。
基线：`113e165e78970cab3713e6e7b3b325c8a16c7561`。
范围：[PRD](PRD.md)；修订后的总设计：[总计划](plans/20261008-181312-compiled-library-structure.md)。
执行分配：[Orca 计划](plans/20261008-211718-compiled-library-orca-dag.md)。

## 当前基线修订

CUDA 已合入，不再安排等待 GPU 分支。46 个库头文件中，公共头通过
`internal/model_interface.hpp` 和 SAM 3 adapter 引入私有实现。根 CMake 定义
INTERFACE `sam`，tests 通过该 target 间接获取内部依赖；examples 同时承担
CLI、图像 IO 和量化探针。v2 precision 导出、评估及 provenance 工具已存在。

保持 `SAM_BUILD_CUDA_PROBES`、`SAM_REQUIRE_CUDA_TESTS`、CUDA F16 计算和压缩
缓存的既有边界。不得把迁移当成改变这些实验的公开支持状态的机会。

## 公共 API 与所有权

公共 `types.hpp` 保留现有字段和枚举，不按文件数量目标拆分。公共类只保留声明、
前置声明及必要的小内联操作；与私有类型完整定义有关的构造、析构和方法放入
`src/api/`。不得因手写析构而意外引入或删除移动操作。

| 类型 | 私有成员和所有权 | 必须保留 |
| --- | --- | --- |
| `Model` | `shared_ptr<internal::ModelImplementation>` | 可复制、共享模型状态，`info()` 引用有效期和后端信息 |
| `ImageSession` | `unique_ptr<internal::TextImageSessionImplementation>` | 禁止复制及当前移动特征；实现持有模型，拥有图像/提示缓存 |
| `VideoSession` | `unique_ptr<internal::TextVideoSessionImplementation>` | 禁止复制及当前移动特征；帧序、reset、输出队列与 trace 行为 |
| `ModelImplementation` | 模型信息和任务创建接口 | 仅已有文本图像、文本视频契约；不添加未实现任务 |

接口外置的形状如下，实际签名以基线公共头为准：

```cpp
namespace sam::internal {
class ModelImplementation;
class TextImageSessionImplementation;
class TextVideoSessionImplementation;
std::shared_ptr<ModelImplementation> load_model(
    const std::string& path, BackendOptions options);
}

// 公共头声明方法；cpp 包含 contracts，调用原有实现。
Model Model::load(const std::string& path, BackendOptions options) {
    return Model(internal::load_model(path, options));
}

// 在私有类型完整定义可见的 cpp 中定义。
ImageSession::~ImageSession() = default;
VideoSession::~VideoSession() = default;
```

模型工厂采用显式分支或不可变表，不使用全局可变注册器。生成 `sam/export.hpp`
声明公共符号的可见性，并为静态构建设置正确的导出宏。共享库不得依赖导出全部
内部符号才能运行。

## 私有目录与依赖

| 现有职责 | 新归属 |
| --- | --- |
| 公共包装器实现 | `src/api/` |
| 模型工厂 | `src/model_factory.cpp` |
| 已实现任务契约 | `src/contracts/` |
| 后端无关的输入验证 | `src/common/` |
| GGUF 容器读取 | `src/io/` |
| SAM 3 图、权重、tokenizer、session 和时序政策 | `src/models/sam3/` |
| GGML 资源、执行、workspace 和 CPU/Metal/CUDA 驱动 | `src/runtime/ggml/` |
| CLI 与解码支持 | `apps/`、`support/image_io/` |
| STB 原始文件与许可 | `third_party/stb/` |

保持 `sam::internal` 命名空间以限制修改范围。先机械移动，再在独立提交中提取
视频执行职责。`TrackerExecution` 保持协调者，用明确状态引用和返回值连接
graph_cache、memory_payload、frame_storage、workspace_policy；不为每个函数
引入接口或继承层。保持图 key、缓存槽、workspace 选择、传输顺序和释放时点。

依赖方向为：公共包装器实现到任务契约/工厂，工厂到模型 adapter，模型到 GGML
运行时，运行时到后端。runtime 不包含模型头；库不依赖 apps、support、tools。
模型数值语义仍归模型；设备存储与执行兼容性仍归后端。

公共与私有头分别检查独立包含、重复包含和唯一的路径派生 include guard。第一批
收紧 GGML 编译依赖时，立即建立不安装的专用 target，供内部 tests、私有头检查
及全部现有实验探针获得私有路径、GGML 宏与链接依赖；第二批只迁移这些路径。
公共 `sam::sam` 不传播这些编译包含路径。私有函数转为库内隐藏符号后，内部测试
使用不安装的对象或静态测试支持 target 获取实现，避免靠导出全部内部符号通过
共享库测试；不得在一个进程内链接两份 GGML runtime。保留 Release 有效检查与双
TU 链接测试。

## CMake 与安装契约

根 `sam` 改为真实库，内部首先用 `target_sources` 组织，不机械拆成大量子库。
独立构建在父工程未指定时默认静态，服从父工程已有 `BUILD_SHARED_LIBS`。
保持 `SAM_BUILD_TESTS`、`SAM_BUILD_EXAMPLES` 的 standalone/embedded 默认值。
新增 `SAM_BUILD_TOOLS`，同样 standalone 开启、embedded 关闭。

`SAM_BUILD_EXAMPLES` 在兼容期继续控制原有两个 CLI。`SAM_BUILD_CUDA_PROBES`
只控制 CUDA 专项探针；`GGML_CUDA=ON` 且 probes OFF 时仍须启用后端需要的 CUDA
构建能力。仅在 `GGML_CUDA=OFF` 且 probes OFF 的 CPU-only 配置中要求不启用
CUDA language、不发现 CUDA toolkit。probes ON 而 GGML_CUDA OFF 继续明确报错。
tests 显式请求自身所需的 support 或工具 target，不依赖“开启 examples 才偶然
存在”这一关系。
保存旧的 `${build}/examples/` CLI 与探针输出位置，迁移目录不改变脚本参数。

GGML 保持已锁定 revision、补丁和源树指纹，不改 caller-owned target 复用判断。
公共 API 无 GGML 类型；编译依赖私有化，静态最终链接依赖由导出 target 正确
传递。使用标准 `install(TARGETS)`、`install(EXPORT)`、package config 和版本文件。

SAM 准备的 GGML 随安装包交付所需依赖 target 和二进制。复用外部 GGML 时使用
可解析的依赖 package；不可导出的外部 target 必须明确拒绝安装配置，不输出
不可消费的包。安装仅包含公共头、生成 export 头、库、CMake package 和许可。

## 工具迁移与冻结证据

工具按 convert、quantize、validation、benchmark、visualization、maintenance
分组。旧 `tools/*.py` CLI 入口保留薄转发，仍只维护一份实现。同步更新导入、
脚本根目录推导、相邻 schema/lockfile 访问、CLI 帮助与 docs 路径检查。

`tools/precision_artifacts.py::source_snapshot()`、`archive_sources()` 和其它
producer/provenance 路径规则随每批源码变化一起交付，不统一留到工具整理阶段。
采集与归档覆盖真实实现、构建文件和必要的编译输入，排除 build、models、缓存、
凭据及私人数据。移动模块后同步修复仓库根目录推导，目录与后缀白名单不能漏掉
新 `.cpp`、`.hpp`、`.cu`、CMake 文件或实际编译的 vendor `.h` 文件。

| 节点 | 同批身份与归档责任 |
| --- | --- |
| A | 从新增 `src/api/*.cpp` 和工厂开始，递归覆盖 src 实现及构建文件；扩展归档目录和后缀规则；保留尚未移动的旧路径 |
| B | 覆盖迁移后的 contracts、model、runtime 与视频实现，更新引用路径；机械移动与职责提取各提交都保持采集有效 |
| C | 覆盖工具子目录、apps/support、实际编译的 third_party 及所有相关 CMake 文件，修复根目录与资源定位 |
| D | 汇总各类来源的变更检测和归档只读校验结果，不修改旧冻结凭据 |

新增行为测试：在临时目录改变新实现或嵌套工具时快照身份必须变化；归档后使
原始来源不可用，`verify_export_artifacts()` 仍可凭归档验证；篡改归档须失败。
记录旧路径到新路径的映射，并对旧归档样例执行只读校验。不得通过排除新源码
使旧 campaign 继续伪装成同一实现。

旧冻结 campaign、JSON 凭据、原始输出和归档源码不改写。旧阶段的质量
FAIL/INCONCLUSIVE、未运行项和算术缺口保持原判定。结构迁移采用新版本身份、
新报告目录及已有开发/固定回归输入；不为重构重新打开已冻结的最终评估或储备集。

## 验证矩阵

从 A 开始，每个改动节点均运行 CPU/CUDA Release 构建和 CTest，CUDA 设置
`SAM_REQUIRE_CUDA_TESTS=ON`；内部依赖、源码身份及归档检查属于同批门槛。
实际检查过的同一提交、选项和输入可复用结果；新改动必须重跑受影响检查。按
项目规则清理自己产生的无用中间文件。D 汇总并补齐以下矩阵，不用最终验收替代
前序节点的门槛。实际命令和环境写入本次计划结果，不写到历史凭据中。

| 检查 | 要求 |
| --- | --- |
| CPU Release | `GGML_METAL=OFF`、`GGML_CUDA=OFF`、`GGML_BLAS=OFF`，构建和 CTest |
| CUDA Release | 匹配 GPU 上启用 CUDA 与 `SAM_REQUIRE_CUDA_TESTS=ON`，避免 skip 冒充通过 |
| 公共头与双 TU | 仅公共 include 下独立/重复包含；真实库双 TU 链接 |
| 私有实现增量编译 | 修改私有实现后应用对象不重编译，库和必要链接更新 |
| 源码与安装消费 | static/shared、parent-owned GGML、prefix relocation、无源码树访问 |
| 依赖选项 | SDK-only、examples OFF/tools ON、CPU-only 无 CUDA toolkit、CUDA probes OFF/ON |
| 来源与归档 | 新源码和构建文件可追踪；离线归档校验通过；篡改可检出；旧冻结证据不改写 |
| Python 工具 | 隔离参考环境执行 `tools/test_tools.py`，旧入口及新实现有效等价 |
| 图像与视频 | 固定原始参考，tensor/box/score/mask、对象进入/遮挡/ID、session 和长序列 |
| 性能 | 同输入、后端、选项的新旧图像/视频延迟与峰值内存；单列构建时间 |
| 文档 | `tools/check_docs.py`、链接、双语表、`git diff --check` |

CPU 构建示例，实际每批使用各自新目录：

```sh
rtk proxy cmake -S . -B build/structure-cpu-a -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF
rtk proxy cmake --build build/structure-cpu-a --parallel 2
rtk proxy ctest --test-dir build/structure-cpu-a --output-on-failure
rtk proxy .venv-reference/bin/python -B tools/test_tools.py
rtk proxy .venv-reference/bin/python -B tools/check_docs.py
rtk proxy git diff --check
```

CUDA 基本配置使用新目录与 `-DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_CUDA=ON
-DSAM_BUILD_CUDA_PROBES=OFF -DSAM_REQUIRE_CUDA_TESTS=ON`。完成构建/CTest 后，
另用新目录配置 probes ON，构建已有全部专项探针并运行不耗用冻结评估集的回归。
工具环境使用现有 `.venv-reference`；不为整理目录替换参考依赖版本。

纯迁移优先要求相同输出。若翻译单元变化造成数值差异，先定位编译/执行原因，
再记录原门槛下结果，不使用“正常波动”解释。Metal 缺少匹配环境时留明确的
未验证项；迁移完成与跨后端数值认证分别记录，不把预定命令当作执行结果。

## 节点完成约定

在 viewer 分配的当前工作区内执行，先读取当前状态及已有用户工作。节点内为
可独立保留的变化记录提交与验证；只提交自己拥有且已验收的文件，不推送远端。
本次修订后的总计划和分配计划用于追加进度，其他历史计划不修改。每个节点的结果必须交代修改、
实际检查、未验证项和交接路径；遵守 Orca 注入的 Task/Dispatch 生命周期。

这张 DAG 是依赖有序的单工作区迁移，不需要 merge-prep 节点。每个节点继承前序
结果；模型配置固定为 OpenCode 2 的 `deepseek/deepseek-flash#max`。配置或执行
环境无法满足时报告具体原因，不换模型、不创建替代工作区、不自行重绘本 Run。
