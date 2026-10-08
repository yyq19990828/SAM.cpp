# 编译库迁移的 Orca 任务分配

创建：2026-10-08 21:17:18，Asia/Shanghai。

状态：已按复审结论修订，DAG 已创建并配置；尚未启动实施。
源码基线：`113e165e78970cab3713e6e7b3b325c8a16c7561`。
总计划：[编译库架构、模型扩展与 C/Python 接口计划](20261008-181312-compiled-library-structure.md)。

## 范围与方法

用户指定分配新合入的编译库架构迁移计划，执行代理使用 OpenCode 2、DeepSeek
V4.1 Flash、max。此次将总计划中的三批结构迁移及最终验收写入一个新的 Orca Run，
并保存 [PRD](../PRD.md) 与 [技术规格](../TECH_SPEC.md)。新模型移植、C/Python
绑定和补齐精度整配方算术证据仍属于后续专项，不包含在本 Run。

总计划的历史基线是 `54b1df7`。当前已合入 CUDA、运行时量化、压缩缓存及 v2
验收工具，不能继续按“等待 GPU 分支接入”安排任务。迁移须保留 CPU、Metal、CUDA
三个已实现后端、`SAM_BUILD_CUDA_PROBES`、现有 CUDA 检查和实验工具。

当前有 46 个库头文件；公共头直接引入模型实现，`sam` 仍是 INTERFACE target，
consumer 测试仍要求 header-only，安装包导出尚未建立。工具源码身份采集包含
`tools/*.py`、`tools/*.cpp` 和 `include/**/*.hpp` 等旧路径规则；新目录迁移必须同步
覆盖 `src/` 和工具子目录，并同步调整 `archive_sources()` 的目录/后缀规则。
这些改动随每批迁移交付，保持旧归档、冻结 campaign 与已有 JSON 凭据不变。

本次修订同时更新总计划、PRD 和技术规格：私有测试/探针 target 前移到 A；
CPU/CUDA 回归从 A 开始；源码身份与归档按 A/B/C 分批同步；CUDA 后端开启而
probes 关闭必须可用，只有两者均关闭的 CPU-only 配置才不需要 CUDA toolchain。

## 任务与依赖

| 节点 | 交付 | 依赖 |
| --- | --- | --- |
| A | 记录迁移基线，建立公共 API 编译边界、私有测试/探针 target 与 src 身份归档 | 无 |
| B | 迁移私有实现及测试路径，逐步拆分视频执行职责，同步身份归档 | A |
| C | 整理 apps/support/tools，交付静态和动态 SDK 安装包，同步嵌套工具身份归档 | B |
| D | 核验数值、消费方式与性能，完成文档和交付记录 | C |

四个节点使用当前工作区，依赖保证同一时刻只有一个编辑任务，最大并发为 1。
各任务必须在 Dispatch 已指定的工作区内执行，不自行创建 worktree、切换分支或
重新克隆仓库。此 Run 不使用跨工作区合并或隐含的人工 integrated 断言。

各节点规格包含目标、修改范围、所有权、约束和可观测验收。实施者读取本计划及
技术规格后直接执行；发生缺失硬件、输入或不可恢复失败时，通过当前 Dispatch
向 coordinator 报告具体阻塞，不放宽门槛或把未运行项记为通过。

```mermaid
flowchart LR
    A["A 编译边界 · 私有依赖 · 来源归档"] --> B["B 私有实现迁移 · 视频职责"]
    B --> C["C 工具整理 · 静态/动态 SDK"]
    C --> D["D 数值 · 消费 · 性能验收"]
```

本轮修订与建图已获授权，不另加重复的方案审批 gate。实施由用户在 viewer
启动；缺少必需资源或需要扩大范围时走 Dispatch 的 ask/escalation，不自行放宽
约束或启动额外代理。四个节点的完整 spec 由下列公共约束与各自规格拼接生成。

## 所有节点的执行约束

项目是 `/home/john/桌面/SAM.cpp`。这是已经审定的编译库迁移任务，直接完成当前
节点，不重新规划。阅读 `AGENTS.md`、`docs/PRD.md`、`docs/TECH_SPEC.md`、
`docs/plans/20261008-181312-compiled-library-structure.md` 及本计划的前序结果。
目标基线为 `113e165e78970cab3713e6e7b3b325c8a16c7561`；后继节点继承本 Run
前序节点的提交。若出现无关代码变化或基线不明，先报告具体差异，不覆盖它。

执行代理固定为 OpenCode 2、`deepseek/deepseek-flash#max`。只在当前 Dispatch
指定的工作区工作，不创建 worktree、不切分支、不克隆、不另起 Run 或子代理。
先检查 Git 状态，保留已有用户工作与 `.orca-dag.config.json`。命令通过 `rtk`
执行，必要时使用 `rtk proxy`。按路径精确暂存本节点拥有的已验收变更，形成
可独立保留的本地提交；不使用 `git add .`，不提交无关文件，不推送或发布。
本次四份规划文档可追加实施结果；不要把未运行的步骤写成成功。

保持公共 API、复制/移动与会话所有权、同一模型执行锁、GGUF schema、权重、
tokenizer、精度边界、缓存与计算顺序。保留 CPU/Metal/CUDA 及固定 GGML
`353b63b439f27ab2cc19dac97ab1681ba6d2d084`、精度补丁和 caller-owned target
复用；不扩大后端/模型支持声明。不实现新模型、语言绑定或新的量化优化。

每批同步维护 `source_snapshot()`、`archive_sources()` 及迁移后的对应实现，
覆盖真实源码、归档与旧入口，不修改冻结 campaign、JSON 凭据、
旧输出或归档。新证据写入新目录；不重新打开冻结的最终评估/储备集，使用开发
与固定回归输入。基线已有 FAIL/INCONCLUSIVE 和算术缺口保留原结论；结构迁移
要求不引入回归，不承担将既有失败改判为通过的任务。

每批实际运行 CPU Release 构建/CTest，以及匹配 GPU 上的 CUDA Release
构建/CTest（`GGML_METAL=OFF`、`GGML_BLAS=OFF`、`GGML_CUDA=ON`、
`SAM_REQUIRE_CUDA_TESTS=ON`）。CPU 配置显式关闭 CUDA、Metal 和 BLAS。
CUDA probes OFF 的正常后端构建和 probes ON 的专项工具构建都要覆盖；CPU-only
不要求 CUDA toolkit。使用独立输出目录，不覆盖现有已验证构建；相同提交、选项
和输入已有本 Run 证据时可复用，记录来源。Metal 无匹配硬件时明确未验证。
在现有 `.venv-reference` 运行受影响的 Python 行为检查及 `tools/test_tools.py`，
运行 `tools/check_docs.py` 和 `git diff --check`。不新增低价值实现镜像测试。

缺少必需 CPU/CUDA 硬件、模型、参考输入，或不能满足节点验收时，通过注入的
Dispatch ask/escalation 报告阻塞，不能跳过后宣称成功。遵守注入的 Orca 生命周期，
完成前读取 coordinator 消息；只在验收成立时报告 succeeded，否则明确 failed。
结果列出提交、变更、实际命令/结果、证据路径、未验证项和后继节点交接信息。
完成验证后只清理本节点产生且不再需要的中间文件；保留原始模型、可用 GGUF、
转换清单、当前已验证构建、必要证据及所有既有用户文件。

## 节点规格

### A：编译边界、私有依赖与源码归档

目标与所有权：修改公共 `include/sam/{model,image_session,video_session}.hpp`、
新增 `src/api/` 和模型工厂、根/src CMake、现有 tests/examples target 定义、
`tools/precision_artifacts.py` 及相关测试，更新 AGENTS、README、changelog 和
本次计划。内部模型/runtime/backend 文件本批保留原路径。

实施步骤：

1. 在修改实现前记录当前源码、GGML 指纹、工具环境、硬件、模型/输入/参考的
   具体路径与身份。复用身份一致的既有证据，补齐迁移前同输入的 CPU/CUDA
   图像/视频输出、时延、内存和构建行为基线；写入新的基线目录并在本计划索引。
   选用已有开发/固定回归输入，不消耗冻结最终评估或 reserve。后继节点沿用
   同一清单、编译选项和测量方法，不重新挑选有利样本。
2. 将公共包装器实现与工厂外置，`sam` 变为真实库，公共头仅依赖标准库、公共
   值类型和前置声明；保持默认参数、异常、移动特征与共享模型生命周期。默认
   静态并尊重父工程 `BUILD_SHARED_LIBS`；生成公共导出宏，保留双 TU 链接。
3. 在 GGML 编译依赖私有化的同一变更建立不安装的私有 tests/probes target。
   更新全部内部测试、私有头检查、profile/precision/cache/linear 探针的显式
   依赖，公共头检查单独使用公共 include。共享构建的内部测试通过私有支持
   target 获得实现，不能暴露全部内部符号或重复装载 GGML。
4. 首次新增 src 实现时同步扩展 `source_snapshot()` 和 `archive_sources()`，
   递归覆盖新 cpp/hpp/构建文件且保留旧路径。编写临时目录行为测试：改变新源
   后身份变化；归档后移除临时源副本仍可验证；篡改归档失败。修复新路径所需
   生产者调用，不修改任何旧冻结证据。
5. 同步更新 header-only 规则、consumer 的 INTERFACE 假设和实际用户说明；
   暂不迁移 apps/tools 目录，不改变数值算法。

验收：执行公共约束的 CPU/CUDA 与工具检查；确认默认 CLI、内部测试、头检查
和已有专项探针均可构建。公共头独立/重复包含不需要 GGML；双 TU 与 static/shared
源码 consumer 链接运行。CUDA probes OFF/ON 的开关组合保持有效。源码变更与
归档负例测试真正执行。交付包含可复用的基线索引、私有 target 说明及本地提交。

### B：私有实现迁移与视频职责拆分

目标与所有权：将现有 `include/sam/internal/` 的契约、输入校验、GGUF reader、
SAM 3 模型和 runtime 移入技术规格中的 src 结构；修改相关 CMake、私有测试/
探针 include、来源采集/归档规则及开发说明。复用 A 的私有 target 和基线。

实施步骤：

1. 先做机械移动，保留 `sam::internal` 命名空间和现有计算行为。runtime/backend
   的 CPU/Metal/CUDA、host tensor、observer、精度和 workspace 路径迁移单独
   提交；没有等待 GPU 分支的前置条件。记录旧新路径映射，同步来源/归档。
2. 大型加载、图构建与执行逻辑编译进库；模板和值类型允许留在私有头。更新
   路径派生 guards、公共/私有独立与重复包含检查、测试支持 target。公共头
   安装候选集不含私有实现，runtime 不反向包含模型头。
3. 在机械移动验证后分别提取视频 graph_cache、memory_payload、frame_storage、
   workspace_policy。保留 TrackerExecution 协调者，用明确状态引用连接；
   每个可独立保留的提取记录提交和受影响回归。保留图 key、内存选择、布局、
   传输/释放时机、模型执行锁和多 session 生命周期，不夹带性能优化。
4. 每个新增/移动的实现和构建文件均进入当前来源快照及归档。更新全部使用者
   到真实新路径，不用旧目录里的实现副本维持测试通过。

验收：执行公共约束检查；CPU/CUDA 的图、精度、量化、跟踪、workspace、host
tensor 和后端用例全部执行；使用 A 的同模型/输入验证图像和视频输出、实际权重
session 生命周期、长序列及对象进入/遮挡/ID 连续性。源码归档覆盖新目录。
验证修改私有跟踪实现只重编译库及必要链接，不重编译未变化应用对象。交付路径
映射、机械迁移与职责提取的提交记录、结果及未验证平台。

### C：工具目录与可安装 SDK

目标与所有权：迁移 apps、support/image_io、tools 的用途子目录、third_party/stb
和 tests 分组；修改根/子目录 CMake、安装导出/package config、consumer 测试、
工具导入/资源/源码归档路径、文档和 changelog。保持 A/B 的编译与数值契约。

实施步骤：

1. CLI 与解码支持分离，SDK-only 不依赖 STB、Python、工具或 tests。保留 STB
   许可与原始 guards。`SAM_BUILD_EXAMPLES` 继续控制两个现有 CLI；新增
   `SAM_BUILD_TOOLS` 保持 standalone ON、embedded OFF。tests 显式获得自身
   support/tool target。`SAM_BUILD_CUDA_PROBES` 只控制 CUDA 专项探针。
2. tools 按 convert/quantize/validation/benchmark/visualization/maintenance
   分组；旧 Python CLI 保留薄转发，修复 imports、仓库根推导、schema/lockfile
   和子进程调用。保存原 `${build}/examples/` CLI/探针产物路径。检查旧入口
   的实质输出，不只比较 --help；可导入的辅助模块保留必要兼容转发。
3. 同步 source_snapshot/archive_sources 的递归范围、目录与后缀、模块根路径，
   覆盖嵌套工具、apps/support、实际编译的 vendor h 文件及所有相关 CMake。
   保留旧来源归档校验，并覆盖新工具变更检测与归档负例。
4. 标准 CMake install/export/package config 同时支持 static/shared 和
   add_subdirectory/find_package。私有头及测试 target 不安装；公共 include
   不泄漏 GGML 编译依赖，静态最终链接依赖仍完整传递。SAM 准备的固定 GGML
   随包交付；caller-owned GGML 可解析则复用，不可导出的 target 明确拒绝
   安装配置。保持一个进程一份 GGML runtime，正确设置 PIC、符号与依赖定位。
5. 创建与实际结构一致的开发导航，更新真实安装、选项、路径及支持边界说明。

验收：执行公共约束检查；覆盖 SDK-only、examples OFF/tools ON、CPU-only、
CUDA probes OFF/ON。static/shared 分别做源码 consumer、安装 consumer、移动
安装前缀测试；consumer 构建环境不能访问源码树（用隔离消费环境，不移动用户
工作区）。验证 parent-owned GGML 复用与不可导出时的清晰失败。安装集只含
公共头、库、package 与许可；动态依赖和公开符号正确。全部 Python 工具测试、
旧入口语义回归、嵌套来源归档与文档检查通过，交付可消费的安装产物及提交。

### D：最终回归、性能与交付记录

目标与所有权：以 C 之后的实现完成独立验收；可修复本轮迁移导致的构建、工具、
行为或数值回归并更新对应测试。维护总计划、PRD、TECH_SPEC、本计划、README、
架构导航和 changelog。不得扩大到新模型、语言绑定或既有精度专项欠项。

实施步骤：

1. 读取 A 的基线清单与 A/B/C 的提交和实际证据，按相同模型、输入、编译选项、
   后端和硬件核验。复用严格匹配当前提交/输入的证据，补齐不足项；更改代码
   后重跑受影响检查。输出实际命令、退出状态、未运行项及证据路径。
2. 完成 CPU/CUDA Release、公共头/双 TU、static/shared 源码与安装消费、父工程
   GGML、prefix relocation、SDK-only、tools 独立构建和 CUDA probes OFF/ON
   矩阵。检查私有实现增量编译不使应用对象重编译。Python 工具在既有隔离
   环境执行，来源快照/归档的正例、变更负例和篡改负例均须有效运行。
3. 比较图像 tensor/box/score/mask、视频帧与对象进入/遮挡/ID、reset、缓存、
   多 session 和长序列，优先要求迁移前后相同输出。差异先定位编译/执行
   原因，再按原门槛判定；不以正常波动或放宽门槛掩盖回归。保留基线已知
   FAIL/INCONCLUSIVE/NOT_RUN，另列本次迁移差异。
4. 用 A 的输入与方法对比端到端图像/视频时延、峰值内存和编译时间，记录
   checkpoint、精度、后端、设备及选项；推理与编译收益分开。资源缺失不能
   宣称对应验收通过，Metal 缺少硬件则明确标为未验证。
5. 完成文档与 changelog 核对，记录交付提交、安装产物、证据索引、实际支持
   和未验证范围。仅清理本 Run 新建的无用中间文件；必要证据和当前有效构建
   保持可追溯。不要修改旧验收 JSON 或重新运行冻结最终评估/储备集。

验收：本 Run 的必需构建、契约和迁移回归项均有本次有效证据，发现的迁移问题
已经修复并验证；源码身份与归档覆盖真实实现，安装 consumer 可运行。最终报告
区分结构交付、数值结果、性能结果、既有精度缺口与 Metal 未验证范围。提交完成
的变更并向 coordinator 报告，不推送、不发布、不以计划命令代替实测结果。

## 代理设置

- Orca CLI：`/home/john/.orca-relay/bin/orca`，运行时 `1.4.222`，状态 ready。
- Viewer：当前项目的 `http://localhost:8787`，`orca-orchestration-launcher 1.0.3`。
- Viewer harness：`opencode`。本机 `opencode2` 包装器转发到同目录的 `opencode`，
  两个入口均报告 OpenCode `2.0.6`。
- 每任务模型：`deepseek/deepseek-flash#max`。执行主机的 `model.list` 已返回
  DeepSeek V4.1 Flash 和 `max` 变体；不使用另一个旧模型 `deepseek-v4-flash`。
- 不传独立的 `--effort`。当前 viewer 在显式指定 OpenCode 模型时使用
  `opencode run --auto -m` 的一次性执行路径，仍由 viewer coordinator 管理 DAG；
  不把该路径描述为原生受监督 TUI，也不手工派发重复 worker。

## 验证与证据边界

规划交付检查文档链接、空白、任务依赖、Run 隔离、四个任务的模型选择及串行设置。
此时不启动 worker，也不执行编译库重构。

上一轮审核在当前源码基线验证了 Python 工具 144/144、九个 CLI 帮助入口、67 份
文档及空白检查；139 个冻结身份和 2,485 个性能绑定路径哈希一致。CPU 16/16、
CUDA 29/29 是已核对身份的历史构建证据。上述结果不替代重构之后的新构建和模型
验收，也不补足 v2 整配方算术缺口。

每个实施节点在全新输出目录保存新证据。保留模型、原始 checkpoint、转换清单、
历史 JSON、当前已验证构建及必要原始输出。完成验证后仅清理本节点新建且已经
不需要的中间文件；不移动或清理既有用户工作。

## 分配结果

2026-10-08 21:39:56 Asia/Shanghai 回读结果：

- Run：`run_ac50d79f3e82`，源码基线仍为 `113e165`。
- 四个节点同属此 Run，完整 spec 与本文件公共约束及各节点规格逐字一致。
- viewer 已保存四个节点的 `opencode` harness、`deepseek/deepseek-flash#max`
  模型、current placement，最大并发为 1；没有独立 effort 或 lane 覆盖。
- 执行主机 `opencode2 api model.list` 返回 DeepSeek V4.1 Flash 的 max 变体，
  `opencode2 run --help` 确认 `provider/model#variant` 语法；本机 opencode2
  包装器执行同目录 opencode。以上证明配置与可选模型匹配，未产生实际模型调用。

| 节点 | Task ID | 依赖 | 创建后状态 |
| --- | --- | --- | --- |
| A | `task_226b81a7e1d5` | 无 | ready |
| B | `task_32e4f58c8f2d` | A | pending |
| C | `task_c87a0cc87415` | B | pending |
| D | `task_a5fdffe0a216` | C | pending |

`task-list --run run_ac50d79f3e82 --ready` 仅返回 A；包含远端的 Run-scoped
`worker-list` 返回 0 个 worker，尚未 dispatch 或开始实施。没有新增审批 gate；
用户在已运行的 viewer 选择此 Run 后点击 **Run with Orca** 才会执行。

规划交付检查：70 份文档的双语表及本地链接检查通过；四份规划文档的空白、
代码块和任务规格完整性检查通过；Git 差异空白检查通过。已回读任务依赖、完整
spec 和 viewer 配置。未重新运行完整构建、模型验收或性能测试，不把创建任务
视为实现交付。源码、既有证据和模型文件均未修改。

## 节点实施结果

### A：编译边界、私有依赖与源码归档

状态：完成，本地提交 `refactor: build sam as a compiled library with private tests and src identity`
（哈希见 worker 完成报告）。迁移前身份/行为基线在 `build/structure-baseline-a-v1/`
（`baseline-identity.json`、`tool-environment.json`），迁移后证据在
`build/structure-a-evidence-v1/`（索引 `index.json`）。CPU Release 17/17、CUDA
probes OFF 30/30 与 probes ON 探针 45/45、共享构建 17/17、static/shared consumer
各 2/2 均在本节点执行；图像与视频固定回归输出在计时归一化后一致；`src/` 已纳入
源码身份与归档并有变更/离线/篡改负例测试。未验证：Metal（无匹配硬件）和 CPU
完整 216 帧视频套件（本机吞吐），CPU 视频改用官方 `negative` 16 帧固定回归，
详见[总计划实施记录](20261008-181312-compiled-library-structure.md)。B 继承本提交继续。
