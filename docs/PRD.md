# SAM.cpp 编译库迁移需求

状态：新基线复审后已修订，DAG 已创建；由用户从 Orca viewer 启动实施。
基线：`113e165e78970cab3713e6e7b3b325c8a16c7561`。
依据：[总计划](plans/20261008-181312-compiled-library-structure.md)和
[本次任务分配](plans/20261008-211718-compiled-library-orca-dag.md)。

## 目标

将 SAM 层从 header-only 交付迁移为默认静态、可选动态的编译库。现有 C++ 应用
保留公共头路径、类名、函数签名、默认参数和 `sam::sam` target；应用不再解析
SAM 3 模型图和 GGML 实现。内部模型、任务契约、执行资源和工具各有明确归属。

## 本轮必须完成

1. `Model`、`ImageSession`、`VideoSession` 的实现移出公共头；保持所有权、复制和
   移动特征、返回值有效期、异常行为及会话并发语义。
2. 私有实现迁入 `src/`；CPU、Metal、CUDA 继续共用模型图，设备政策仍由后端拥有。
3. 视频执行器分步提取图缓存、memory/pointer 打包、帧驻留和 workspace 政策，
   保持执行顺序、精度、布局、对象 ID 和资源释放边界。
4. apps、解码支持、工具和 tests 不再依赖 examples 间接组织；旧 CLI 路径及
   Python 命令入口可继续使用。
5. `add_subdirectory` 与可重定位的 `find_package(sam CONFIG REQUIRED)` 均能
   消费静态、动态 SDK。安装集不泄漏私有头或 GGML 编译包含路径。
6. 第一批即交付内部 tests、私有头检查与探针的私有 target，保持默认构建可用。
7. 每批同步更新真实源码身份采集与归档，保留旧冻结证据；从第一批开始运行
   CPU/CUDA 回归，区分 CUDA 后端与专项探针的构建开关。
8. 更新真实构建方式、架构导航、AGENTS 和 changelog，保存各批实际检查结果。

## 本轮不包含

SAM 3.1、GroundingDINO、框提示能力、DART、新引擎、C ABI 和 Python wheel 的
实现分别另建专项计划。本轮不创建这些能力的空目录、空类型或未经实现的公开入口。
不改变模型格式、权重、tokenizer、量化门槛、默认精度、缓存语义或执行并发模型。
不升级 GGML、不改变精度补丁、不发布远程提交或 release。

## 成功标准

| 用户场景 | 可观测结果 |
| --- | --- |
| 引入单个公共头 | 只用 SDK 公共 include 和标准库即可独立、重复包含，无 GGML 头依赖 |
| 修改私有跟踪实现 | 重新编译库并重新链接静态应用，不重编译未变化的应用源码 |
| 源码集成 | 原有 API、默认构建选项、显式后端选择及双 TU 链接保持兼容 |
| 安装后消费 | 移动安装前缀，隐藏源码目录，静态和动态 consumer 均可构建运行 |
| 仅构建 SDK | 不引入图像解码、STB、Python、工具和 tests 依赖 |
| 关闭 examples 但启用 tools | 实验工具可独立构建；CPU-only 配置无需 CUDA toolkit |
| 启用 CUDA 但关闭 probes | CUDA 后端正常构建与运行；探针开关不控制后端可用性 |
| 新增或移动实现 | 同批纳入源码快照与归档；变更可检出，归档可脱离活动源码校验 |
| 使用现有模型和输入 | 原门槛下 tensor、框、分数、mask、视频 ID 和生命周期检查通过 |
| 查阅支持情况 | 当前实现、迁移后实测通过与未测试组合分别列明 |

同一模型的 session 继续共享既有执行锁；同一 session 不允许并发调用。模型资源
不得因公共包装器析构、复制或 out-of-line 析构调整而提前释放。

## 验收边界

当前 Linux 主机承担 CPU/CUDA 验证。Metal 必须在匹配硬件上执行，缺少该环境时
记录为未验证，不以 CPU/CUDA 或历史 Metal 结果代替迁移后的实测。新性能报告
列出 checkpoint、精度、后端、硬件、编译选项、延迟和峰值内存；构建时间收益与
推理性能分别报告。除本次明确修订的总计划外，其他历史计划和不可变机器凭据
保持原样。

## 交付方式

四个串行 Orca 节点分别完成编译边界、私有结构、工具和安装包、最终核验。每批
维护可编译状态并记录自己的检查与结果。计划写入 Orca 后由用户在 viewer 启动；
每个节点使用 OpenCode 2 与 `deepseek/deepseek-flash#max`。
