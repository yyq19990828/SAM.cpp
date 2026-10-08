# 编译库架构、模型扩展与 C/Python 接口计划

创建时间：2026-10-08 18:13:12 Asia/Shanghai。
状态：已按新合入基线修订，实施尚未开始。
初稿源码基线：`54b1df74669ff3e4be5c75b9399bc588cade4ae0`（main）。
当前实施基线：`113e165e78970cab3713e6e7b3b325c8a16c7561`（main）。
执行分配与结果：[Orca 任务计划](20261008-211718-compiled-library-orca-dag.md)。

## 目标与范围

将 SAM.cpp 从 SAM 层 header-only 交付迁移为默认编译库，建立公共 API、任务契约、模型实现、组合流程和 GGML 设备后端的依赖边界。保持现有 C++ 调用方式，并为 SAM 3.1、GroundingDINO + SAM、DART 及后续 C/Python 接口提供明确的接入位置。

当前基线已合入 CUDA 后端、计算与传输优化、运行时量化、压缩缓存及 v2 精度验收工具。迁移不再等待 GPU 分支；CPU、Metal、CUDA 均作为已有实现保留。从第一批开始执行 CPU/CUDA 回归，Metal 在匹配硬件上验证，缺少环境时明确记录未验证。保留已有精度结论及实验边界，不因结构迁移扩大支持声明。

本轮结构迁移包括编译边界、私有代码组织、工具/测试归属和 SDK 安装消费。新模型算法移植、GPU 算子实现和语言绑定分别按后续能力增量交付；模型文件格式、现有数值契约及会话并发语义不在结构迁移中变更。

## 1. 推荐方向与成功标准

采用轻量公共头文件、私有编译实现和独立工具目录。对外继续提供一个 `sam::sam` target；内部按照模型语义、执行资源、任务会话和工具用途组织。

完成后应满足：

- 现有应用继续使用 `Model`、`ImageSession`、`VideoSession`；新能力通过检测、提示分割和组合流程接口接入，不要求调用方理解内部图与设备资源。
- 单独包含任一公共头文件时不需要 GGML 头文件，也不解析模型计算图。
- 修改 SAM 3 跟踪实现会重编译对应库实现，不触发未修改的应用源码重新编译；静态链接应用仍需重新链接。
- 增加模型主要修改 `src/models/<family>/` 和模型工厂；增加 GGML 设备主要修改 `src/runtime/ggml/backends/`，不复制模型图。
- CLI、转换工具和测试有各自清楚的构建入口，不由 examples 间接承担它们的依赖。
- 现有模型文件、缓存行为、会话生命周期、后端选择和数值验收门槛保持不变。

当前最小可行方案是只将模型加载入口移到一个 `.cpp`，先形成编译边界。推荐完整方案在此基础上继续划分职责，主要收益是限制修改影响范围，并降低大型跟踪文件的维护风险。

这次调整会涉及超过 8 个文件，覆盖当前 46 个库头文件中的多数，以及构建、测试和说明文档，属于中等规模结构重构。

## 2. 当前需要解决的具体问题

| 现状 | 影响 | 调整 |
| --- | --- | --- |
| 公共 `model.hpp` 包含 SAM 3 实现，该实现同时包含图像和跟踪会话 | API 与具体模型发生编译耦合 | 创建具体模型的代码移到库内部工厂 |
| `include/sam/internal/` 是全部实现的分发目录 | 公共接口和私有实现的安装、依赖边界不清楚 | 私有文件移到 `src/`，公共安装集单独列出 |
| `tracking/execution.hpp` 约 1,670 行，包含图形状、缓存、传输、预算和执行 | 修改某一种职责时需要理解整套执行状态 | 保留一个执行协调器，按真实职责提取内部实现 |
| `examples/CMakeLists.txt` 构建 CLI、图像 IO 库和量化工具；开启测试也会进入 examples | 测试和工具依赖关系绕行 | CLI 移到 apps，工具归 tools，共用 IO 放在 support |
| consumer 测试明确要求 `INTERFACE_LIBRARY` | 测试锁定旧交付形式 | 改为验证公共头隔离、真实链接和已安装包消费 |
| tools 平铺转换、验证、基准、可视化和归档脚本 | 使用入口和内部辅助模块难以区分 | 按用途分组，暂留旧脚本作为兼容入口 |

仓库现有 `AGENTS.md` 的“Header-Only Architecture”要求与新方向冲突。建议在编译库迁移的同一提交中更新该规则、README 和对应 consumer 断言；本计划采用编译库方向，现有规则在实施该批次前继续描述当前源码状态。

## 3. 建议目录

以下是目标规划结构，不表示已经创建这些源码文件。标注“规划”的模块对应已确定的产品路线；在其实现提交中创建，不用空目录假装能力已经存在。

```text
SAM.cpp/
├── include/sam/                 # 仅安装和承诺兼容的公共接口
│   ├── sam.hpp
│   ├── types.hpp
│   ├── model.hpp
│   ├── image_session.hpp
│   ├── video_session.hpp
│   ├── detection_session.hpp   # 规划：纯检测任务
│   ├── prompt_session.hpp      # 规划：点/框提示分割
│   ├── pipelines/              # 规划：组合流程的公共入口
│   ├── c_api.h                 # 规划：C ABI，纯 C 可包含
│   └── export.hpp              # 构建时生成的符号导出定义
├── src/
│   ├── CMakeLists.txt
│   ├── model_factory.cpp       # 已实现模型的显式组装入口
│   ├── api/                    # 公共类的方法定义和句柄管理
│   │   ├── model.cpp
│   │   ├── image_session.cpp
│   │   └── video_session.cpp
│   ├── contracts/              # 私有任务契约，不包含具体模型或 GGML
│   │   ├── model.hpp
│   │   ├── text_image.hpp
│   │   └── text_video.hpp
│   ├── common/
│   │   └── input_validation.hpp
│   ├── io/
│   │   ├── gguf_reader.hpp
│   │   └── gguf_reader.cpp
│   ├── models/
│   │   ├── sam3/
│   │   │   ├── model.hpp / model.cpp
│   │   │   ├── weights.hpp / weights.cpp
│   │   │   ├── state.hpp
│   │   │   ├── tokenizer.hpp / tokenizer.cpp
│   │   │   ├── image_ops.hpp / image_ops.cpp
│   │   │   ├── graphs/         # 架构、张量和各阶段 GGML 图
│   │   │   ├── image/          # 图像会话、阶段执行、结果处理
│   │   │   └── video/          # 视频会话、时序策略和跟踪执行
│   │   ├── sam3_1/             # 规划：独立 schema、图和 Object Multiplex 状态
│   │   └── grounding_dino/     # 规划：文本检测模型
│   ├── pipelines/              # 规划：模型/阶段的组合，不拥有基础算子
│   │   ├── grounded_sam/
│   │   └── dart/
│   └── runtime/
│       └── ggml/
│           ├── resources.hpp / resources.cpp
│           ├── runtime.hpp / runtime.cpp
│           ├── graph.hpp / graph.cpp
│           ├── workspace.hpp / workspace.cpp
│           └── backends/
│               ├── backend.hpp
│               ├── cpu.cpp
│               ├── metal.cpp
│               └── cuda.cpp   # 迁移已有 CUDA 后端
├── bindings/                   # 随语言接口阶段交付
│   ├── c/                      # C ABI 实现，编译进同一个 sam 库
│   └── python/                 # Python 包、pybind11 模块、NumPy 转换
├── apps/
│   ├── image/                  # 现有 sam_image 命令
│   └── video/                  # 现有 sam_video 命令
├── examples/                   # 只依赖公开 SDK 的最小使用示例
├── support/
│   └── image_io/               # CLI/测试共用的解码实现，不安装进 SDK
├── tools/
│   ├── convert/                # 模型转换和模型 schema
│   ├── quantize/               # 含 quantize_rows.cpp
│   ├── validation/             # 参考导出、图像/视频比较和验收
│   ├── benchmark/              # 测量、统计和性能资格检查
│   ├── visualization/          # 比较图生成
│   └── maintenance/            # 文档检查、归档等维护工具
├── tests/
│   ├── api/                    # 对外行为与生命周期
│   ├── models/                 # 按 sam3、sam3_1、grounding_dino 分组
│   ├── pipelines/              # 组合流程的坐标、类别、缓存和输出契约
│   ├── bindings/               # C/Python 与 C++ 的一致性和所有权
│   ├── runtime/ggml/           # 后端、精度、调度和 workspace
│   ├── integration/            # CLI、源码集成、安装包消费、双 TU
│   ├── tools/                  # Python 工具测试
│   └── data/                   # 现有固定 fixtures 和数值门槛
├── cmake/                      # GGML 固定版本、补丁、安装和 package 配置
├── third_party/stb/             # 从 examples/stb 移入，保留许可与原始 guards
├── docs/
│   ├── architecture.md
│   ├── models/
│   ├── plans/                  # 保持历史计划和历史路径叙述
│   └── validation-baselines/   # 保持既有验收记录
└── licenses/
```

目录内的 `.hpp/.cpp` 成对形式是职责归属建议，不要求机械地为每个旧头生成两个文件。模板、小值类型和适合内联的小函数可以留在私有头文件；计算图构建、加载和较大执行逻辑编译进库。

`types.hpp` 当前只有约 107 行，先保留一个文件。只有在输入、结果或配置显著扩展时，才按明确的 API 概念拆分。

## 4. 依赖规则

箭头表示实现依赖，不是数据复制方向。

```text
C++ app / CLI     C application      Python application
      |               |                    |
      |           C ABI shim          pybind11 binding
      +---------------+--------------------+
                      |
           public C++ task/pipeline API
                      |
              model_factory.cpp
                      |
       model adapters and pipeline assembly
                      |
        family-specific graph / execution
                      |
                runtime/ggml
                      |
      GGML CPU / Metal / CUDA backend

Shared leaves:
  include/sam/types.hpp   values, no runtime dependency
  src/contracts/         task interfaces, no concrete models or GGML
  src/common/            backend-independent validation

Weight path:
  model_factory -> io/gguf_reader -> GGML GGUF C API
  model adapter -> family-specific schema and tensor checks
```

具体约束：

1. 公共头只依赖标准库、公共值类型以及前置声明。构造、析构和涉及完整私有类型的定义放在 `.cpp`，尤其注意 `unique_ptr` 的不完整类型要求。保留现有复制/移动语义。
2. `contracts/` 按文本检测、文本分割、点/框分割、视频跟踪划分任务；当前先迁移已经实现的契约，其余随对应能力加入。不增加一个带许多空方法的万能模型接口。
3. `model_factory.cpp` 是显式组装点。只有它需要知道有哪些模型实现；使用明确分支或不可变表，不使用全局可变注册器和静态初始化注册，避免静态链接裁剪掉模型注册代码。
4. 模型代码拥有 tokenizer、形状、张量名称、归一化、精度语义、时序策略和后处理。运行时拥有设备发现、资源分配、图调度、传输和执行统计。
5. 模型规定的 F16/BF16 边界属于模型语义；设备上如何存储、执行及验证兼容性属于后端职责。不能为了目录整齐把这些数值规则全部下沉。
6. GGUF reader 只做通用容器读取、范围检查和有限元数据访问。它仍依赖 GGML 的 GGUF API，不宣称引擎无关；各模型继续负责自己的 schema 验证。
7. runtime 不包含模型的头文件；support、apps、tools 不成为 sam 库的依赖。
8. 公共头安装集不包含 `src/`。内部测试通过专用测试 target 获取私有包含路径，应用不能靠公共 target 顺便获得它们。

## 5. SAM 3 内部如何拆分

模型目录的重点是把“模型结构”和“执行某次任务”分开。

| 目录/文件 | 所有职责 |
| --- | --- |
| model / weights | 模型加载、schema、张量绑定、模型信息和任务创建 |
| state.hpp | 同一模型共享的权重、必要资源和现有执行锁 |
| graphs/ | vision、text_encoder、prompt_encoder、fusion_encoder、detector、mask_decoder，以及跟踪 memory/attention 图构建 |
| image/session | 单次图像任务入口、图像与提示缓存、输入校验、结果处理 |
| image/execution | 调用图像各阶段并维护当前执行统计 |
| video/session | 帧顺序、固定提示/尺寸、hotstart、结果排队、reset |
| video/policy | 对象关联、ID 生命周期、记忆选择和可见性规则 |
| video/execution | 跟踪执行的协调器，拥有下述资源并保持原有调用顺序 |
| video/graph_cache | 图形状 key、缓存槽和图 metadata 生命周期 |
| video/memory_payload | memory/pointer 输入打包及模型规定的位置编码准备 |
| video/frame_storage | 当前帧特征驻留、上传与释放 |
| video/workspace_policy | workspace 探测、批次拆分和串行路径选择 |

这几个视频实现单元用明确的状态引用和返回值协作，保持现有 `TrackerExecution` 作为协调者。不能为了拆文件把每个方法都变成一个抽象接口。

先完成位置迁移，再逐个提取这些职责；每次提取都保留原有公式、张量布局、分支顺序和精度边界，避免同时进行算法优化。

资源所有权保持：模型持有共享权重；session 持有提示、图像或时序状态；跟踪执行器持有图缓存和 workspace；帧资源按现有帧边界释放。继续保留同一模型不同 session 的执行串行化，不在此次结构重构中改变并发模型。

## 6. 模型、平台和引擎如何扩展

三者是不同维度：

| 扩展类型 | 归属 | 需要验证的内容 |
| --- | --- | --- |
| SAM 3.1 | `src/models/sam3_1/`，登记到模型工厂 | 独立 checkpoint schema、图、对象桶、共享记忆与视频行为 |
| 已有 CUDA 后端及后续设备 | 沿 GGML 路线组织在 `src/runtime/ggml/backends/` | 保留现有设备初始化、算子、精度、传输、回退策略和硬件验收边界 |
| GroundingDINO + SAM | 检测器属于 models；流程属于 `src/pipelines/grounded_sam/` | 两套权重、坐标转换、类别/短语、提示分割能力及生命周期 |
| DART | `src/pipelines/dart/`，使用明确的 SAM 3 内部阶段接口 | 检测阶段复用、类别缓存、多类别执行和检测输出；首版采用原始 SAM 3 backbone |
| ONNX Runtime / TensorRT 等其他引擎 | 单独的执行实现及对应模型适配 | 各引擎模型表示、算子和数值，不能把它们当作 GGML 的普通设备 |
| C / Python 接口 | `bindings/c/`、`bindings/python/` | 所有权、错误传递、线程和 ABI，调用 SDK 而不是复制推理代码 |

以上是已知路线的扩展位置，不是本次已交付的新能力；不创建空目录、空类或未经实现的后端枚举。SAM 2/2.1 可沿同一模型目录约定接入，但不属于本计划的优先模型范围。

已有 CUDA 后端采用 GGML 设备扩展，因此本轮不用增加通用 engine 抽象层。CPU、Metal、CUDA 共用模型图，后端负责设备和执行政策；模型图所用算子是否在各设备上可用、精度是否满足要求仍需逐项验证。

方案最容易失效的假设转为：SAM 3 与 SAM 3.1 的组件能够直接共享。对此采用独立 adapter，只有经过数值与形状验证的模块才提取复用，确保复用不成立时不推翻整体结构。如果以后真正引入 TensorRT/ONNX Runtime，再在私有模型执行边界增加对应实现，不把它们硬塞进 GGML 的设备驱动目录。

### 6.1 三条模型路线的具体边界

**SAM 3.1：模型实现独立，复用经过验证的组件。** 官方发布说明明确其 Object Multiplex 按固定容量的对象桶联合处理，具有新的 checkpoint 和共享记忆路径。因此 `sam3_1/video/` 自己拥有桶分配、对象插入/移除、共享记忆及结果映射；不通过继承 SAM 3 的现有 VideoSession 强行复用状态机。只有图形状、精度和语义一致、已有双模型测试的算子或模块，才提取到 `models/common/`；不先移动整套 SAM 3 图再假定兼容。桶边界测试使用实际配置容量 K 的 K-1、K、K+1，并验证跨桶对象 ID 稳定性。

**GroundingDINO + SAM：两个模型，一个组合流程。** `models/grounding_dino/` 输出文本相关的框、分数和短语；具备 box-prompt 能力的 SAM adapter 接收这些框并输出 masks。pipeline 只负责调用、过滤、坐标与结果关联，各模型保留自己的 resize、normalize 和 tokenizer。pipeline 边界统一使用原图像素 XYXY 坐标，输入分割模型前再做该模型要求的变换。

这里有一个需要明确补齐的功能缺口：当前 SAM.cpp 对外只有文本分割会话，尚不能仅靠目录移动得到可组合的 box-prompt segmentation 接口。首版建议先在现有 SAM 3 adapter 补齐并验证框提示能力，再接入 GroundingDINO；这不等同于已经兼容原始 SAM 1 的 Grounded-SAM 实现，参考输出应来自所选分割模型的官方实现。不同分割器以后通过同一个任务契约替换。

**DART：检测任务独立，重用 SAM 3 的真实计算阶段。** 首版在 `pipelines/dart/` 管理类别列表、文本特征缓存、多类别执行和检测后处理，复用 SAM 3 的 vision、text、fusion 和 detector 阶段，跳过 mask decoder。不要通过完整分割后丢弃 mask 来实现检测。现有权重加载器也需要按任务维护经过验证的必需张量集合，不能随意忽略缺失张量；阶段一可以仍加载完整 SAM 3 权重，但结果必须明确它只省去了 mask 计算，并未证明减少全部权重占用。

DART 上游还包含学生 backbone 和 TensorRT 等路线；本方案明确只先实现原始 SAM 3 backbone 路径。若引入学生模型，应在 `models/` 增加对应图和权重契约，不能把新 backbone 藏在普通 pipeline 配置里。

### 6.2 公共任务与数据契约

现有 `ImageSession` 和 `VideoSession` 保留现有语义，不在同一个 session 加入所有模型才能理解的方法。新增能力使用专用接口，并由已加载 adapter 报告支持的任务。

| 任务契约 | 首批使用者 | 输出 |
| --- | --- | --- |
| 文本图像分割 | 现有 SAM 3 | 现有 boxes、scores、masks |
| 文本/类别检测 | GroundingDINO、DART | boxes、scores、label/class_id；没有强制 mask |
| 点/框提示分割 | 用于 Grounded SAM 的分割 adapter | 与提示/对象对应的 masks 与质量信息 |
| 视频跟踪 | SAM 3、SAM 3.1 各自的实现 | frame index、稳定 object ID、boxes、masks |

当前 `Detection` 含有 Mask，保持该类型兼容；纯检测新增独立结果类型，而不是约定“空 mask 表示这是另一个任务”。共享 Box、ImageView 等值类型，不共享模型隐藏状态。

模型支持某任务、二进制编译了某设备、当前机器找到设备、该组合经过数值验证，是不同信息。API 的能力查询只能反映已经实现的能力和运行时可用性，发布支持矩阵另行记录硬件验收证据。

### 6.3 C/Python 绑定路线

推荐保持 C++ SDK 为核心公开实现，C ABI 和 Python 都调用这一套 SDK。Python 优先用 pybind11 绑定 C++ API，避免每个新模型的实验接口都必须先冻结为长期 C ABI；不维护 Python 版推理逻辑。C 的稳定子集按已经验收的任务增加。

- **C ABI**：公开 `include/sam/c_api.h`，实现位于 `bindings/c/` 并编译进同一个 sam 库。使用 opaque model/session/result handles、固定宽度整数与指针/长度数据，不暴露 STL、GGML 或 C++ 类。创建/释放配对，库分配的结果由库释放；结果视图有效期绑定到 result handle。使用状态码和明确的错误信息，所有 C++ 异常在 C 边界转换。结构体包含大小/版本信息用于增量扩展，提供 ABI 版本查询。
- **Python**：`bindings/python/` 放包代码、pybind11 模块、类型提示和测试，入口使用可识别的独立包名 `sam_cpp`，避免与官方研究包混淆。首版接受 RGB、uint8、HWC NumPy 数组；显式检查 dtype/shape/stride，不隐式把 float 图像按 uint8 截断。默认返回独立拥有内存的 NumPy 结果，先确保生命周期正确，再按测量添加受控零拷贝视图。
- **线程**：纯 C++ 推理期间可以释放 GIL；释放前完成 Python 对象检查并保留输入所有者，释放期间不访问 Python 对象。保留 session 不允许并发调用的契约，Python 包装器为每个 session 提供非阻塞的 in-use 防护；重入立即报错，防止释放 GIL 后两个线程同时修改一个 session。输入数组在调用期间不得由其他线程修改。
- **验证体验**：提供最小 image、video、detection Python 示例和 notebook，能够在 Python 里调用已移植的 C++ 模型并与官方结果对比。Python 绑定不意味着未经移植的任意新 PyTorch 网络能自动由 C++ 执行；新图、权重转换和数值验证仍是模型接入工作。
- **依赖边界**：Python/pybind11/NumPy 只属于绑定与参考工具环境，纯 C/C++ SDK 构建不需要它们。PyTorch 放在参考验证依赖中，不作为 Python 推理包的默认依赖。

当绑定阶段开始时增加独立的 `SAM_BUILD_PYTHON`（默认关闭）；C 接口实现作为核心 SDK 的公共符号集交付，不另造一份推理库。Python wheel 验收需要在没有源码树、没有开发编译环境的干净环境中运行，并检查随包共享库的依赖定位。

## 7. 构建与交付

- 对外只有一个稳定的 `sam::sam` target。内部目录首先用 `target_sources` 组织同一编译库；确有编译隔离需要时才增加私有 OBJECT target，不要求用户逐一链接内部库。
- 独立构建在父工程未指定时默认静态库；允许 `BUILD_SHARED_LIBS` 选择动态库，服从父工程已设定的值。动态库通过生成的 export 定义标注公共符号，避免导出所有模型内部符号。
- 保留 `add_subdirectory` 集成方式；完成安装导出后支持 `find_package(sam CONFIG REQUIRED)`，继续使用同一个 `sam::sam` target。
- 公开 API 不暴露 GGML 类型，GGML 编译包含路径和实现宏保持私有。静态库的最终链接仍需要 GGML，必须由导出的 CMake target 正确传递，不能把 PRIVATE 误解成没有最终链接依赖。
- 固定的 GGML revision、精度补丁、源树指纹和 caller-owned target 复用逻辑继续保留。一个进程不加载两份冲突的 GGML runtime。
- 由 SAM 准备 GGML 时，安装包同时正确提供其依赖 target/二进制；复用外部 GGML 时，安装包要求依赖的包配置可被解析。若外部 target 不可导出，配置 SAM 安装时明确报告，不产生看似可安装但无法消费的包。
- 共享库验收包括动态依赖可定位、导出符号和卸载生命周期；源码兼容不等于跨编译器的 C++ ABI 兼容，不承诺后者。
- 保留现有 `SAM_BUILD_TESTS`、`SAM_BUILD_EXAMPLES` 的 standalone/embedded 默认行为和 `GGML_METAL`、`GGML_BLAS` 配置。迁移期间 `SAM_BUILD_EXAMPLES` 继续构建原有两个 CLI，保持脚本行为。
- 保留固定版本 GGML 的 `GGML_CUDA` 构建能力及已有设备选择、可用性检查和执行策略，不另造 CUDA 后端开关。`SAM_BUILD_CUDA_PROBES` 只控制专项探针；`GGML_CUDA=ON` 且 probes OFF 的正常后端构建必须通过。两者均 OFF 的 CPU-only 构建不要求 CUDA toolkit，也不启用 CUDA language。
- 新增一个 `SAM_BUILD_TOOLS` 开关，将量化可执行程序与 examples 分离，默认沿用 standalone 开启、embedded 关闭。暂不增加模型、视频或任意后端插件开关。
- 工具整理期间保留旧 Python 命令入口和现有 CLI 输出路径；兼容入口只转发到新实现，不保存两份逻辑。权重、GGUF schema 和模型下载流程不因目录调整改变。

安装导出使用 CMake 现有的 `install(TARGETS)`、`install(EXPORT)` 和 package config 机制，不自定义包发现器。

## 8. 分批迁移，每一批都能独立保留

### 第一批：建立编译边界

将公共类的实现与模型工厂移入 `src/`，私有模型头暂时留在原位置。将 `sam` 从 INTERFACE 改成真实编译库，保留公共头路径和调用签名。更新 header-only 指令及测试假设。

在收紧 `sam::sam` 的 GGML 编译依赖的同一批，建立不安装的私有测试/探针 target，调整内部 tests、私有头检查和全部现有实验探针的显式依赖。公共头检查只用公共 include；内部 target 获得私有路径、GGML 宏及链接依赖，不能等第二批再修复默认构建。

本批保留 runtime 和 backend 文件的原路径、命名与行为，以单独验证编译边界；不与 CUDA 内核、精度策略或调度优化混在同一提交中。首次新增 `src/*.cpp` 时同步扩展 `source_snapshot()` 与 `archive_sources()` 的递归采集、目录及后缀规则，覆盖新实现和构建文件；新增回归检查证明源码变更会改变身份，且归档副本可脱离活动源码校验。

验收：现有 CLI、内部 tests、头检查及探针能编译链接；公共头无需 GGML 即可独立解析；双 TU 链接通过；CPU Release 和启用 `SAM_REQUIRE_CUDA_TESTS=ON` 的 CUDA Release 行为检查及选定数值基线通过。CUDA 后端分别验证 probes OFF 和 ON；CPU-only 验证无需 CUDA toolkit。Metal 有硬件时运行，缺少环境则明确保留未验证项。

独立价值：调用方获得真实编译隔离，即使后续目录整理不进行也能长期使用。

### 第二批：整理私有实现与测试归属

迁移 `internal` 到上述 `src` 目录，保留 `sam::internal` 命名空间以控制修改范围。先做机械移动，再按独立提交提取视频执行职责。更新第一批已建立的私有 target 与 tests 路径，SDK 消费者不获得这些路径。

runtime 的路径迁移放到单独提交，覆盖已经合入的 CPU、Metal、CUDA 及 host tensor、精度、workspace 等实现，不设置 GPU 分支等待条件。提供旧路径到新路径的映射；每次移动同步更新源码快照、归档和相关 consumer，保持中间提交可构建。

验收：重复第一批 CPU/CUDA 构建与 CTest 门槛；全部既有契约、图、后端、量化、跟踪、workspace 和工具检查通过；相同模型/输入/后端对比图像及视频结果；实际权重 session 生命周期与长视频检查通过；迁移后真实实现纳入源码身份与归档。

独立价值：库自身更容易修改和定位问题，不依赖工具目录迁移。

### 第三批：整理工具与交付目录

迁移 apps、support、tools、third_party 与 tests；保留命令兼容入口。补全静态/动态安装导出、干净 consumer 测试和架构说明。同步更新源码身份与归档对工具子目录、apps/support、实际编译的 vendor 文件及子目录构建文件的覆盖。新增一份与实际目录一致的开发者导航文档。

验收：重复 CPU/CUDA 回归；只构建 SDK 不带入图像解码库、STB、Python 和测试依赖；examples 关闭时工具仍可独立构建；源码集成与安装消费都能运行；移动安装前缀后 consumer 仍能找到包；旧工具入口的实质结果保持一致；嵌套工具的变更可被身份检查发现，归档可只读校验。

这三批均不改 GGUF schema，不迁移用户模型或其他数据。回退对应代码和构建提交即可，不触碰现有数据。其他历史计划和不可变验收记录保持原样；本计划追加修订与实施结果。旧冻结 campaign 不重写或重新打开最终评估/储备集；结构回归使用新身份、新目录及开发/固定回归输入。

### 结构落地后的能力增量

后续每条能力以可独立发布的增量交付，不要求三个新路线和两种绑定全部完成才使用新结构。

1. **DART 基线**：先暴露检测阶段、独立检测结果和类别缓存，在完整 SAM 3 权重上验证原始 backbone 的检测流程。先证明输出，再考虑精简权重、学生 backbone 或新的精度方案。
2. **SAM 3.1**：独立 adapter 与转换 schema，接入 Object Multiplex；保留 SAM 3 回归套件，逐项验证可复用模块。
3. **Grounded SAM**：先验收 GroundingDINO 文本检测和分割 adapter 的框提示任务，再组合并验证坐标、标签、多框、空检测和输出映射。不能把“创建 pipeline 类”视为已经接入两个模型。
4. **C/Python**：按照上述绑定契约交付已有任务的稳定接口、NumPy 结果和安装包。为了新模型验证，允许先为已验收的 SAM 3 发布最小 Python 包，不必等三条新模型路线全部完成；新增任务再扩展绑定。

此顺序是按当前实现复用成本给出的建议，不要求各模型同时发布；实际发布顺序在各能力实现计划中记录。具体模型算法移植属于后续各自的实现计划，本文件确定其结构归属、接口边界和验收责任。

## 9. 验证与交接要求

实施时以本文件作为总计划，在下方实施记录中追加实际范围、环境与结果；新模型和绑定的专项实现另建带时间戳的计划并回链本文件。计划存在不代表重构已经完成或通过验收。

基础 CPU 检查命令沿用当前入口：

```sh
cmake -S . -B build/structure-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF
cmake --build build/structure-cpu --parallel
ctest --test-dir build/structure-cpu --output-on-failure
python3 tools/check_docs.py
git diff --check
```

Python 数值工具测试继续使用仓库指定的隔离参考环境执行 `tools/test_tools.py`；不假设系统 Python 已安装参考依赖。

完成安装配置后分别构建 `BUILD_SHARED_LIBS=OFF` 和 `ON` 的安装产物，再用仅依赖 `find_package(sam CONFIG REQUIRED)` 的 consumer 测试实际加载。consumer 不可访问源码目录或内部 include 目录。

验收覆盖：

| 类别 | 必须覆盖 |
| --- | --- |
| 正常路径 | 图像加载、文本/token 提示、视频顺序输入、模型复用与 session 生命周期 |
| 错误路径 | 不存在/损坏/不兼容的模型、不可用显式后端、无效图像尺寸/stride、无效 token、视频帧乱序 |
| 边界 | 空检测、重复提示缓存、阈值变化、新图像缓存失效、reset、最后一帧结果排空、多对象与长序列状态 |
| 数值 | 固定权重和输入的 tensor、框、分数、mask、视频 ID/关联；原门槛不放宽 |
| 构建 | 公共头独立/重复包含、双 TU、静态/动态、源码/安装消费、内部依赖不泄漏 |
| 已有 CUDA 回归 | 从第一批开始运行 `SAM_REQUIRE_CUDA_TESTS=ON`；CPU-only 不依赖 CUDA toolkit；CUDA probes OFF/ON 均能构建；显式 CUDA 请求失败时不静默改后端；匹配硬件上检验算子、精度、传输和设备统计 |
| 源码与归档身份 | 每批新增/移动的实现与构建文件均被采集；变更会改变身份；归档不依赖活动源码路径；旧冻结凭据保持不变 |
| 新任务 | DART 的检测结果和类别缓存；Grounded SAM 的坐标映射/空框/多框；SAM 3.1 的对象桶边界、增删与 ID 连续性 |
| 语言绑定 | C 纯 C 编译器消费、错误/释放协议；Python dtype/stride、结果生命周期、session 并发防护、与 C++ 相同输入输出；干净环境 wheel 安装 |
| 性能 | 相同硬件/编译选项下的完整图像与视频时延、峰值内存、构建时间；不把调用方编译提速当推理提速 |

纯文件迁移和入口外置优先要求相同环境下输出完全相同。如果仅因为翻译单元变化出现数值差异，必须定位优化或执行差异，并按既有数值门槛单独记录差异与验收结论，不能直接以“重构正常波动”解释。

本次规划只核实了源码结构、既有测试入口和构建约定，未运行新的模型验收。CPU/CUDA 实机、原始 checkpoint 和已接受参考样本是本轮实施验收前置条件；Metal 需要另有匹配硬件。缺少条件时只能报告完成了对应结构或构建检查，不能宣布该平台数值通过。仓库现有 CI 配置也不等于本次已在这些平台执行。

结构重构本身不需要新的账户、服务或 API key。若实施环境尚无原始模型，下载使用既有授权与凭据流程；GGML 网络不可用时可复用已核验的本地固定版本，不能自动改用未经核验的版本。

## 10. 参考与取舍

- [SAM.cpp 当前架构](https://github.com/yyq19990828/SAM.cpp/blob/54b1df74669ff3e4be5c75b9399bc588cade4ae0/docs/architecture.md)：保留现有任务、模型、后端边界与数值契约。
- [公共模型入口](https://github.com/yyq19990828/SAM.cpp/blob/54b1df74669ff3e4be5c75b9399bc588cade4ae0/include/sam/model.hpp)：当前具体模型实现被公共头引入的位置。
- [现有 consumer 约束](https://github.com/yyq19990828/SAM.cpp/blob/54b1df74669ff3e4be5c75b9399bc588cade4ae0/tests/consumer/CMakeLists.txt)：迁移时需同步更新 header-only 假设。
- [llama.cpp 源码构建](https://github.com/ggml-org/llama.cpp/blob/master/src/CMakeLists.txt)：参考其公共 include、私有源文件和独立模型实现编译到库的组织机制，不照搬其更复杂的模型管理系统。
- [whisper.cpp 源码构建](https://github.com/ggml-org/whisper.cpp/blob/master/src/CMakeLists.txt)：参考其编译库和可选平台实现各自拥有依赖的方式，不照搬公开 include 路径范围。
- [CMake 安装与导出指南](https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html)：采用标准的可重定位 package/target 导出机制。
- [SAM 3.1 官方发布说明](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)：Object Multiplex 与新 checkpoint 是独立状态/图契约的依据。
- [Grounded SAM 官方示例](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam_demo.py)：检测框转换后传入 SAM，支持将模型与组合流程分开的设计。
- [DART](https://github.com/mkturkcan/DART) 与 [encoder-decoder 导出实现](https://github.com/mkturkcan/DART/blob/main/sam3/trt/export_enc_dec.py)：原始 SAM 3、多类别检测、文本缓存和学生 backbone 属于不同的可选范围。
- [pybind11 NumPy 接口](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) 与 [GIL 说明](https://pybind11.readthedocs.io/en/stable/advanced/misc.html)：使用现成的数组绑定与 GIL 管理机制。

不推荐在本轮同时建立动态插件系统、通用张量 IR 或两套完整的 header-only/compiled 分发模式。这些方案会引入注册、版本或配置组合成本，而当前边界问题通过私有编译实现就能解决。

## 11. 实施记录

- 2026-10-08：完成 main 基线的结构评估、目标目录、模块契约、GPU 分支协调方式、分批迁移和验收要求；以计划文档入库。
- 本次未迁移源码，未修改现有 header-only 构建、AGENTS.md、GGUF schema 或模型行为。
- 文档检查：新增计划的本地链接、代码块闭合及差异空白检查通过；现有双语测量表一致性检查通过。
- 全量 `tools/check_docs.py` 未通过：既有 `docs/plans/20261004-214547-latest-complete-model-performance-records.md` 引用了当前检出缺少的 `build/rope-optimization/20261004/native-image-qualification-summary-v1.json`。已确认该计划与 HEAD 内容一致，失败不由新增文档引入；本次不修改历史验收记录。
- 各实现批次须追加涉及提交、执行环境、检查结果和未验证的模型/后端组合，不以计划中列出的检查代替实际结果。

### 2026-10-08 新基线复审与修订

- 在 `113e165` 上复审，纳入原基线之后合入的 CUDA、量化、压缩缓存和 v2 验收工具；取消 GPU 分支等待条件，明确每批 CPU/CUDA 验收与 Metal 未验证边界。
- 私有测试/探针 target 前移到第一批；源码身份采集与 `archive_sources()` 适配随每批目录变化交付；明确 CUDA 后端开关与探针开关相互独立。
- 复审证据：只给公共 include 编译 `test_graph.cpp` 会缺少 `ggml.h`，补上 GGML include 后语法检查通过；临时目录调用真实工具函数证明旧规则漏采新 `src/`、嵌套工具并跳过 `src/` 归档。此为迁移约束验证，不表示当前默认构建失败。
- 复审时 70 份文档检查与空白检查通过；未执行完整构建或新模型验收。修订后的文档与 DAG 检查结果记录在上述 Orca 任务计划。

### 2026-10-08 节点 A 实施结果（编译边界、私有依赖与源码归档）

实施基线：`113e165e78970cab3713e6e7b3b325c8a16c7561`（main）。本节点未迁移 apps/tools，
未改动 runtime/backend 与私有模型头路径，未改变数值算法。

**迁移前身份与行为基线**（新目录 `build/structure-baseline-a-v1/`，索引 `baseline-identity.json`、
`tool-environment.json`）：

- 旧 `source_snapshot()` 137 项，汇总 sha256 `45f2739da395c453a7a862d6e662bf7fcf762a62a632b35d00d9d8679a5ea6f2`。
- GGML `353b63b439f27ab2cc19dac97ab1681ba6d2d084`；Metal 补丁
  `0a0b80dd15c2a8b5a05d148a31e9e53f8df4d0555852ef301da336a9d97b7c48`；CUDA 补丁
  `28b1260af845c338755d25214e336d697152ad01ceed5322fc6803834d9a76a9`；未升级、未改补丁。
- 硬件/工具：12th Gen Intel Core i7-12700KF（20 线程）；NVIDIA RTX 4090 24564 MiB（驱动 610.57.04，
  计算能力 8.9）；CUDA 13.3；CMake 4.2.3；GCC 15.2.0；`.venv-reference` 版本见 `tool-environment.json`。
- 模型/固定回归输入：`models/sam3-f32.gguf`（sha256 `cb13ecd5...`）、`models/sam3-video-f32.gguf`、
  `models/reference/sam3-f32-cuda`、`models/reference/sam3-video-f32-cuda` 与
  `models/fixtures/{truck,groceries}.jpg`；完整哈希见基线索引。未使用冻结最终评估/储备集。
- 迁移前 CPU Release（`GGML_METAL=OFF`、`GGML_CUDA=OFF`、`GGML_BLAS=OFF`）：17/17 通过。
  CUDA Release（`GGML_METAL=OFF`、`GGML_BLAS=OFF`、`GGML_CUDA=ON`、probes OFF、
  `SAM_REQUIRE_CUDA_TESTS=ON`）：30/30 通过。CUDA 首次构建因构建期间工作树被并发修改而失败
  （编译期读到新公共头，非源码缺陷）；恢复 HEAD 后在同一目录增量重跑，以上述结果为准，两次日志均保留。
- 迁移前参考运行（官方 checkpoint 门槛）：CPU 图像 7/7、CUDA 图像 7/7、CUDA 视频 5/5。CPU 完整视频
  套件（216 帧）在本机吞吐下需数小时，运行至 motion 第 14 帧后终止，`cpu-video-validate.log` 记录
  `VIDEO_EXIT=143`，输出不完整、不作通过证据；CPU 视频固定回归改用官方 reference 的 `negative`
  （16 帧）用例，CUDA 运行完整套件。这是本节点明确的验证范围缩小，后继节点沿用同一清单复核。

**实施**：

- 公共 `include/sam/{model,image_session,video_session}.hpp` 只保留声明、公共值类型、标准库与前置
  声明；默认参数与异常语义不变。`Model` 保持可复制/可移动共享句柄，`ImageSession`/`VideoSession`
  保持不可复制、不可移动（双 TU 测试静态断言）。
- 新增 `src/api/{model,image_session,video_session}.cpp`、`src/model_factory.{hpp,cpp}` 与
  `src/CMakeLists.txt`；工厂为显式分支，无全局可变注册表。
- `sam` 由 INTERFACE 改为真实库：默认静态并按配置入口的 `BUILD_SHARED_LIBS` 选择共享，
  可用 `SAM_BUILD_SHARED_LIBS` 在 caller-owned GGML 场景固定类型；共享构建生成 `sam/export.hpp`
  （`SAM_API`）并隐藏内部符号；GGML include/macros 保持 PRIVATE，静态最终链接依赖继续传递。
- 新增不安装的 `sam_private` INTERFACE target，向内部 tests、私有头检查和 profile/precision/cache/
  linear 探针提供私有 include、GGML 编译依赖与链接；公共 `sam::sam` 只暴露公共 include。
- 双 TU 测试更名 `test_two_tu`；consumer 断言由 INTERFACE 改为真实库，覆盖静态/动态、双 TU 与
  caller-owned GGML 复用。
- `tools/precision_artifacts.py` 抽出 `repository_root()`；`source_snapshot()` 递归纳入
  `src/**/*.{cpp,hpp,cu}` 与 `src/**/CMakeLists.txt`（旧路径保留），`archive_sources()` 目录白名单
  加入 `src`；新增临时目录行为测试覆盖“改源码→身份变化”“移除活动源码→归档仍可验证”“篡改归档→
  校验失败”。
- AGENTS.md 的 Header-Only 规则更新为 Compiled Library Architecture；README 更新构建、集成、
  `SAM_BUILD_SHARED_LIBS` 与 `sam_private` 说明；changelog 记录 Added/Changed。

**实测结果**（迁移后证据在 `build/structure-a-evidence-v1/`，索引 `index.json`）：

| 检查 | 配置/命令 | 结果 |
| --- | --- | --- |
| CPU Release | `build/structure-cpu-a`；`GGML_METAL=OFF GGML_CUDA=OFF GGML_BLAS=OFF` | 构建 0；CTest 17/17 |
| CUDA probes OFF | `build/structure-cuda-a`；`GGML_CUDA=ON SAM_REQUIRE_CUDA_TESTS=ON` | 构建 0；CTest 30/30 |
| CUDA probes ON | `build/structure-cuda-probes-a`；`SAM_BUILD_CUDA_PROBES=ON -DCMAKE_CUDA_ARCHITECTURES=89` | 5 个探针全部构建；线性探针回归 45/45 |
| 共享构建 | `build/structure-shared-a`；`BUILD_SHARED_LIBS=ON` | 构建 0；CTest 17/17；内部测试经 `libsam.so` 运行 |
| 源码 consumer | `build/consumer-{static,shared}-a`，caller-owned GGML | 各 2/2；static 链 `libsam.a`，shared 链 `libsam.so` |
| 公共头隔离 | `sam_public_header_checks` 无 defines、仅 generated+`include/` | 编译通过；CUDA 构建同样无 GGML/CUDA include |
| 双 TU | `sam_two_tu` | 通过；复制/移动静态断言成立 |
| 图像迁移前后 | CPU 97 文件、CUDA 83 文件对比 | 除计时/运行元数据外一致（CUDA 图像为纯字节一致） |
| 视频迁移前后 | CUDA 完整套件 1275 文件（计时归一化）；CPU `negative` 88 文件 | 输出与轨迹一致 |
| Python | `.venv-reference/bin/python -B tools/test_tools.py` | 145 测试通过 |
| 文档/空白 | `tools/check_docs.py`、`git diff --check` | 70 份文档通过、无空白问题 |

第一次 probes ON 配置未传 `CMAKE_CUDA_ARCHITECTURES`，在 `.cu` 探针 target 上报空架构错误；
按 README 已有示例补 `-DCMAKE_CUDA_ARCHITECTURES=89` 后配置、构建与回归均通过。既有 probes 构建
同样显式指定该选项，这不是本迁移引入的回归。

`sam_private` 说明：不安装、不导出、不属于应用接口；内部 target 经它获得私有 include 与 GGML 依赖，
共享构建下内部实现保持隐藏符号；`BUILD_SHARED_LIBS=ON` 时 GGML 同为共享库，一个进程不加载两份
GGML runtime。公共库的动态导出符号只含 `SAM_API` 标记的包装器/工厂，抽样 `nm -D` 未见
`sam::internal::sam3` 实现符号。

未验证：Metal（当前主机无匹配硬件）；CPU 完整视频套件（吞吐原因，替代固定 `negative` 用例）。

交接 B：从本节点本地提交继续（提交信息 `refactor: build sam as a compiled library with private tests and src identity`）。
`src/` 已纳入 `source_snapshot()`/`archive_sources()`；私有文件迁移后保持每批同步覆盖。基线与迁移后证据
路径、编译选项与固定输入清单见上，后继节点沿用，不重新挑选有利样本；不修改旧冻结证据。

### 2026-10-08 节点 B 实施结果（私有实现迁移与视频职责拆分）

实施基线：继承节点 A 提交 `cc83042`；本节点结束于 `f14ab3a`。公共 API、GGUF schema、权重、
tokenizer、精度边界、缓存与执行顺序未变；未迁移 apps/tools（属节点 C）。

**机械迁移与路径映射**（各为可独立保留的本地提交）：

- `5393294`：`include/sam/internal/` 的契约、校验、GGUF reader 与 SAM 3 模型移入 `src/`；
  `model_interface.hpp` 拆为 `src/contracts/{model,text_image,text_video}.hpp`，
  `input_validation.hpp` → `src/common/`，`gguf_reader.hpp` → `src/io/`，
  `models/sam3/tracking/` → `src/models/sam3/video/`；路径派生 guards 全部更新。
- `f14ab3a`：runtime 单独提交，`include/sam/internal/runtime/ggml{,.hpp}` 与 CPU/Metal/CUDA
  驱动移入 `src/runtime/ggml/`；聚合头 `ggml.hpp` 改用 src 根路径引用。runtime 不包含模型头。
- `872a829`：历史计划链接通过 `tools/check_docs.py` 的显式旧→新映射解析，历史文档不改写；
  该映射同时作为本节点的路径对照表。

**视频职责提取**（每个提交单独跑 CPU/CUDA 回归）：

- `6a71ec4` graph_cache：`TrackerGraphShape`、shape helper、阶段容量与 `TrackerGraphSlot`
  移入 `video/graph_cache.hpp`，`TrackerGraphCache` 持有四个槽与 build 时间。
- `85927b8` memory_payload：`TrackerRecord`/`TrackerPropagationInput`、pointer 时间位置、
  pointer 切片、BF16 memory 打包与 memory position 表选择移入 `video/memory_payload.hpp`。
- `ac41091` frame_storage：`TrackerResidentUploadState`、resident context/buffer、neck 张量与
  位置表移入 `video/frame_storage.hpp`（`TrackerFrameStorage`）。
- `dc9e0af` workspace_policy：serial/seed 预算与探针、批次拆分、sticky none-resident/
  none-serial 策略与探针诊断移入 `video/workspace_policy.hpp`（`TrackerWorkspacePolicy`）。
- 各提取保留图 key、内存选择、布局、传输/释放时机、执行锁与多 session 生命周期；
  `TrackerExecution` 仍为协调者（1668 行降至 1078 行），不夹带性能优化。

**实测结果**（最终提交 `f14ab3a`；证据 `build/structure-b-evidence-v1/`，索引 `index.json`）：

| 检查 | 配置/命令 | 结果 |
| --- | --- | --- |
| CPU Release | `build/structure-cpu-b`；`GGML_METAL=OFF GGML_CUDA=OFF GGML_BLAS=OFF` | 构建 0；CTest 17/17 |
| CUDA probes OFF | `build/structure-cuda-b`；`GGML_CUDA=ON SAM_REQUIRE_CUDA_TESTS=ON` | 构建 0；CTest 30/30 |
| CUDA probes ON | `build/structure-cuda-probes-b`；`SAM_BUILD_CUDA_PROBES=ON -DCMAKE_CUDA_ARCHITECTURES=89` | 全部探针构建；线性探针回归 45/45 |
| 共享构建 | `build/structure-shared-b`；`BUILD_SHARED_LIBS=ON` | 构建 0；CTest 17/17 |
| 参考图像 | CPU 与 CUDA 官方 reference 门槛 | exit 0；与 A 输出对比各 98 文件一致 |
| 参考视频 | CUDA 完整套件；CPU `negative` 16 帧固定回归 | exit 0；CUDA 1276 文件、CPU 88 文件与 A 输出在计时/元数据归一化后一致，二进制载荷零差异 |
| session 生命周期 | `test_session`（CPU 图像）与 `test_video_session`（CUDA 视频）实际权重 | exit 0；多 session 交错、reset、帧序与对象 ID 连续性、负样本行为通过 |
| 来源与归档 | `source_snapshot()` / `archive_sources()` | 149 项（src 53、include 5）全部纳入并归档；变更/离线/篡改负例测试通过 |
| 冻结归档只读 | `build/precision-final-v2/exports/f16` | 只读校验 PASS（1187 产物、137 归档源），未写入 |
| 增量编译 | touch `src/models/sam3/video/execution.hpp` 后重建 | 库对象重编、应用重链接；未变化应用对象不重编 |
| Python | `.venv-reference/bin/python -B tools/test_tools.py` | 145 通过 |
| 文档/空白 | `tools/check_docs.py`、`git diff --check` | 70 份文档通过、无空白问题 |

未验证：Metal（无匹配硬件）；CPU 完整 216 帧视频套件（吞吐，沿用 A 的 16 帧 `negative`
固定回归）。逐模块 `.hpp/.cpp` 去内联未在本节点执行：私有实现仍为 `src/` 头文件，由库实现
TU（`src/model_factory.cpp`、`src/api/*.cpp`）编译进 `sam`，公共 target 与安装集不包含这些
路径；内部测试/探针经 `sam_private` 获取 `src` 私有 include。将大实现转为库内隐藏符号前，
需先为内部测试建立静态支持 target。

交接 C：从 `f14ab3a` 继续。`src/` 已全部纳入来源身份与归档；旧→新路径映射见 changelog 与
`tools/check_docs.py`；apps/support/tools 目录尚未迁移。

### 2026-10-08 节点 C 实施结果（工具目录与可安装 SDK）

实施基线：继承节点 B 最终提交 `f14ab3a`；本节点以单个可独立保留的本地提交结束
（`refactor: group tools and deliver an installable SDK package`，哈希见 Orca 完成报告）。
公共 API、所有权与会话语义、GGUF schema、权重、tokenizer、精度边界、缓存与计算顺序未变；
未实现新模型、语言绑定或量化优化，未改动 runtime/backend 数值路径。

**目录与构建边界**：

- `examples/` 拆分为 `apps/image/`、`apps/video/`（两个 CLI）、`support/image_io/`（apps、探针、
  tests 共用的解码实现，不安装）与 `third_party/stb/`（vendor 头原样移动，保留 STB guards 与
  许可说明）；`sam_image`、`sam_video` 及全部探针的 `${build}/examples/` 产物路径保持不变。
- `tools/` 按 convert、quantize、validation、benchmark、visualization、maintenance 分组；
  原 `tools/*.py` 扁平入口保留为薄转发（脚本执行与导入语义等价，导入返回分组实现模块）。
  `sam3_tensor_schema.json` 随 convert 分组，`requirements*.lock` 留在 tools 根。
- `tests/` 按 api、models、runtime/ggml、integration、tools、data 分组；Python 测试移入
  `tests/tools/`，`tools/test_tools.py` 继续作为兼容执行入口，`-m tests.tools.test_*` 为规范入口。
- `SAM_BUILD_EXAMPLES` 只控制两个 CLI；新增 `SAM_BUILD_TOOLS`（standalone ON、embedded OFF）
  控制独立工具；`SAM_BUILD_CUDA_PROBES` 只控制 CUDA 探针；tests 显式链接 `sam_image_io`。
- 新增 `SAM_ENABLE_INSTALL`（standalone ON）：`install(TARGETS)`/`install(EXPORT)` 生成可重定位的
  `find_package(sam CONFIG)` 包（config + version）；static/shared 均验证。SAM 准备的固定 GGML
  随包安装并导出为 plain imported targets；caller-owned GGML 需由可解析的包提供，构建树 target
  在安装配置时明确报错。安装共享库设置 `$ORIGIN`/`@loader_path` 定位同目录 GGML；`sam::sam`
  的静态最终链接依赖完整传递；私有头、tests、tools、support 不进入安装集；库设置 PIC。

**来源身份与归档**：`source_snapshot()`/`archive_sources()` 递归覆盖 `tools/**`（分组实现、
转发入口、schema、lock）、`apps/`、`support/`、`third_party/stb/*.h`、`tests/**`
（py/hpp/json/txt/inc/cmake/CMakeLists）与 `cmake/**`（含新增 `.in`）；归档目录与后缀白名单同步。
真实树 223 项全部纳入并归档；嵌套工具变更可检出；归档在活动源码缺失时仍可校验，篡改可被拒绝；
冻结 `build/precision-final-v2/exports/f16` 的 137 个归档源只读校验通过且时间戳未变。

**实测结果**（证据 `build/structure-c-evidence-v1/`，索引 `index.json`）：

| 检查 | 配置/命令 | 结果 |
| --- | --- | --- |
| CPU Release | `build/structure-cpu-c`；`GGML_METAL=OFF GGML_CUDA=OFF GGML_BLAS=OFF` | 构建 0；CTest 17/17 |
| CUDA probes OFF | `build/structure-cuda-c`；`GGML_CUDA=ON SAM_REQUIRE_CUDA_TESTS=ON`、arch 89 | 构建 0；CTest 30/30 |
| CUDA probes ON | `build/structure-cuda-probes-c`；`SAM_BUILD_CUDA_PROBES=ON` | 5 个探针全部构建；CTest 30/30；线性探针回归 45/45 |
| 共享构建 | `build/structure-shared-c`；`BUILD_SHARED_LIBS=ON` | 构建 0；CTest 17/17；`nm -D` 公共符号 28、`sam::internal` 0 |
| SDK-only | `build/sdk-c`；TESTS/EXAMPLES/TOOLS=OFF、CPU-only | 构建 0；未编译 STB、image_io、Python 或 tests |
| examples OFF / tools ON | `build/tools-c` | CLI 不存在；4 个工具构建于 `${build}/examples/` |
| 源码 consumer | `build/consumer-{static,shared}-c`，caller-owned GGML | 各 2/2 |
| 安装 consumer | `tests/install_consumer` 复制到隔离目录后构建 | static、shared、CUDA 各通过 |
| 移动安装前缀 | `prefix` → `prefix-moved` 后重配 consumer | static/shared 均通过 |
| 安装拒绝 | 构建树 GGML + `SAM_ENABLE_INSTALL=ON` | 配置失败并给出明确原因 |
| caller-owned 包复用 | 已安装 ggml 包 + `SAM_ENABLE_INSTALL=ON`，安装为独立前缀后消费 | 通过；config 含 `find_dependency(ggml CONFIG)` |
| 安装集审计 | prefix 清单与 package 文件 | 仅公共头、库、CMake package、许可；无私有头或源码路径 |
| Python | `.venv-reference/bin/python -B tools/test_tools.py` | 145 通过 |
| 旧工具入口 | `tools/<name>.py` 转发脚本、导入身份与 `-m tools.<group>.<name>` | 通过；check_docs 与 25/25 模块测试等价 |
| 身份与归档 | 真实树 `source_snapshot()`/`archive_sources()` | 223/223；篡改检出；离线校验通过 |
| 冻结归档 | `build/precision-final-v2/exports/f16` | 137 归档源只读校验通过、未写入 |
| 文档/空白 | `tools/maintenance/check_docs.py`、`git diff --check` | 70 份文档通过、无空白问题 |

未验证：Metal（无匹配硬件）；CPU 完整 216 帧视频套件（沿用 A/B 的 16 帧 `negative` 固定回归
结论）；节点 C 未重跑官方参考图像/视频数值套件与性能对比（属节点 D）。

交接 D：从本节点提交继续。安装/消费命令、证据目录、固定输入清单及未运行项见上；数值门槛沿用
A/B 清单，不重新挑选样本。

### 2026-10-09 节点 D 实施结果（最终回归、性能与交付记录）

实施基线：继承节点 C 提交 `a78d5fd`；本节点以单个可独立保留的本地提交结束
（`docs: record compiled-library migration final acceptance`，哈希见 Orca 完成报告）。
公共 API、所有权与会话语义、GGUF schema、权重、tokenizer、精度边界、缓存与计算顺序未变；
未实现新模型、语言绑定或量化优化，未改动 runtime/backend 数值路径；最终验收未发现需要
修复的迁移回归（一次 `test_session` 失败由本节点脚本漏加引号导致，修正后原样通过，不是
产品缺陷）。输入身份复核：`models/sam3-f32.gguf` `cb13ecd5…`、`models/sam3-video-f32.gguf`
`02513232…`、GGML `353b63b4` 与两个补丁哈希均与 A 基线一致。

**证据组织**：新证据目录 `build/structure-d-evidence-v1/`（索引 `index.json`、日志 `logs/`、
脚本 `scripts/`、性能 `performance-d.json`）。与当前提交、选项和输入严格一致的构建复用
C 证据并记录来源（`build/structure-cuda-probes-c` probes-ON 构建、`build/consumer-*-c`
源码 consumer、`build/structure-c-install-*` 与 `/tmp/opencode/install-consumer-*` 安装/父工程
基线）；D 在最终树重新执行对应 CTest、消费运行与全部数值/性能检查。新增全新构建目录：
`build/structure-{cpu,cuda,shared}-d`、`build/{sdk,tools}-d`、`build/structure-d-*`。

**构建与工具矩阵**（D 实测；CUDA probes OFF 与 ON 均覆盖）：

| 检查 | 配置/命令 | 结果 |
| --- | --- | --- |
| CPU Release（新构建） | `build/structure-cpu-d`；`GGML_METAL=OFF GGML_CUDA=OFF GGML_BLAS=OFF` | 配置 1s、构建 47s；CTest 17/17 |
| CUDA Release probes OFF（新构建） | `build/structure-cuda-d`；`GGML_METAL=OFF GGML_BLAS=OFF GGML_CUDA=ON SAM_BUILD_CUDA_PROBES=OFF SAM_REQUIRE_CUDA_TESTS=ON arch 89` | 配置 3s、构建 252s；CTest 30/30 |
| CUDA probes ON | `build/structure-cuda-probes-c`（复用构建）→ D 线性探针回归 | 45/45 通过 |
| 共享构建（新构建） | `build/structure-shared-d`；`BUILD_SHARED_LIBS=ON` | 构建 52s；CTest 17/17；`nm -D` 无 `sam::internal` 符号 |
| SDK-only | `build/sdk-d`；TESTS/EXAMPLES/TOOLS=OFF | 构建 29s；无 STB、image_io、CLI 产物 |
| examples OFF / tools ON | `build/tools-d` | 构建 37s；4 个工具产物；无 `sam_image` |
| 静态/共享/CUDA 安装导出 | `build/structure-d-install-{static,shared,cuda}/prefix(-moved)` | 仅公共头、库、package、许可；私有路径与构建树路径泄漏 0 |
| 隔离安装 consumer + 移动前缀 | `/tmp/opencode/d-install-{static,shared,cuda}-d`（static/shared 针对 `prefix-moved`） | 各 1/1；shared `ldd` 从移动前缀解析 `libsam.so`/GGML |
| 父工程 GGML 包复用 | 已安装 ggml CONFIG 的父工程 → SAM 安装包 → 隔离 consumer | 配置/构建/CTest/安装全 0；config 含 `find_dependency(ggml CONFIG)` |
| caller-owned 构建树 GGML | `tests/consumer` + `FETCHCONTENT_SOURCE_DIR_GGML` + `SAM_ENABLE_INSTALL=ON` | 配置失败并给出明确拒绝信息 |
| 源码 consumer | `build/consumer-{static,shared}-c` CTest（复用构建，D 重跑） | 各 2/2（含双 TU） |
| 公共头/双 TU | `sam_public_header_checks`/`sam_internal_header_checks` 重跑 + `sam_two_tu` | 编译通过；公共检查 flags 无 GGML/CUDA include；双 TU 通过 |
| 私有实现增量编译 | touch `src/models/sam3/video/execution.hpp` 后重建 `build/structure-cpu-d` | 库 TU 重编 1、应用与内部测试重链接；未变化应用对象重编 0 |
| Python 工具 | `.venv-reference/bin/python -B tools/test_tools.py` | 145 通过（含来源身份正例、变更负例、离线归档与篡改负例） |
| 来源与归档 | 真实树 `source_snapshot()`/`archive_sources()` 应用 + 嵌套工具篡改负例 | 223/223 全部纳入；篡改可检出；归档可离线校验 |
| 冻结归档 | `build/precision-final-v2/exports/f16` | 137 归档源只读校验通过、无写入 |
| 文档/空白 | `tools/check_docs.py`、`git diff --check` | 70 份文档通过、无空白问题 |

**数值回归**（与迁移前 A 基线同模型、输入、编译选项、后端与硬件；计时/路径/哈希归一化后逐文件对比）：

| 检查 | 基线（A，`113e165`） | D（`a78d5fd` 后） | 对比结果 |
| --- | --- | --- | --- |
| CPU 图像 7 例（tensor/box/score/mask） | PASS | PASS | 98 文件、0 二进制差异、0 JSON 差异 |
| CUDA 图像 7 例 | PASS | PASS | 98 文件、0 差异 |
| CUDA 视频 5 例（motion/entry/occlusion/hotstart-removal/negative） | PASS | PASS | 1276 文件、0 差异；ID 映射一致（entry 0/1，occlusion 0，negative 空） |
| CPU `negative` 16 帧固定回归 | exit 0 | exit 0 | 88 文件、0 差异；16/16 帧输出一致 |
| CPU 图像 session 生命周期（缓存/多 session/图像失效/模型提前释放） | B 通过 | 通过（D 重跑） | 无变化 |
| CUDA 视频 session（reset/帧序/负样本/ID 连续） | B 通过 | 通过（D 重跑） | 无变化 |
| CUDA 长序列 64+64 交错（双 session/结果移动/状态上界） | 未运行 | 通过 | 新增覆盖，无基线冲突 |

**性能对比**（A 的输入与方法；单次完整运行的均值，A/B/D 三点见 `performance-d.json`，
说明存在主机状态波动，不作稳定基准）：

| 指标 | A 基线 | D | B（同迁移后） |
| --- | --- | --- | --- |
| CPU 图像 image_ms / inference_ms 均值 | 36275 / 3856 | 28772 / 3050 | 40306 / 4002 |
| CPU 图像峰值 RSS（metrics / `time -v`） | 4.610 / 4.502 GiB | 4.611 / 4.502 GiB | 同量级 |
| CUDA 图像 image_ms / inference_ms 均值 | 325 / 172 | 308 / 143 | 312 / 135 |
| CUDA 图像峰值 RSS | 3.551 / 3.467 GiB | 3.548 / 3.465 GiB | 同量级 |
| CUDA 视频 5 例 frame_ms 均值 / 峰值 RSS | 997 / 3.630 GiB | 913 / 3.629 GiB | 893 / 3.630 GiB |
| CPU `negative` 16 帧 wall / 峰值 RSS | 16:08.55 / 4.733 GiB | 9:03.47 / 4.733 GiB | 8:47.92 / 4.733 GiB |

编译时间：D 全新目录 CPU 47s、CUDA 252s、共享 52s、SDK-only 29s、tools-only 37s（配置与
推理分开；另有增量编译证据）。A 基线只保留 configure/build 日志、未记录构建墙钟时间，
因此不宣称迁移前后构建时间对比；迁移的编译收益以“私有实现变更不重编译应用对象”单独记录。

**未验证**：Metal（无匹配硬件）；CPU 完整 216 帧视频套件（吞吐，沿用 16 帧 `negative`
固定回归）；逐模块 `.hpp/.cpp` 去内联仍保持 B 的范围（大型实现由库 TU 从 `src/` 编译进库）。

**交付与交接**：D 为最终节点。交付提交、安装产物（`build/structure-d-install-*/prefix*`）、
证据索引与性能记录见上；静态/动态源码与安装消费、父工程 GGML、移动前缀、SDK-only、
tools 独立构建、CUDA probes OFF/ON 矩阵均有本次有效证据。本迁移不推送远端、不发布。
