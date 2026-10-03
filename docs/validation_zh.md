# 验证与基线复用

[English](validation.md)

## 快速检查

按英文页的命令执行 CMake 构建、CTest、`tools/test_tools.py` 和
`tools/check_docs.py`。普通 CTest 不加载模型权重；快速 CI 检查 CPU、独立头文件、
工具和文档。远端 CI 必须等工作流推送后实际运行完成才可宣称通过。
[基线索引](validation-baselines/m2-m4pro-20261003.json)记录 M4 Pro 已验收批次。

## 长期保存证据

`tools/archive_validation.py create` 可接收多个 `--source` 和一个新 `--output`；
`verify` 对保存的 `bundle` 校验文件数量、大小和哈希。目标路径不能覆盖已有内容，
也不能与源路径重叠。APFS 优先使用写时复制克隆，失败的暂存保留未完成记录。

本机 M2 归档位于 `models/validation-baselines/m2-m4pro-20261003/bundle`，
包含 21239 个文件、85994641754 字节逻辑数据；所有副本已校验，
151 份封存收据在归档中的身份也逐项核对。权重、私有媒体和原始数据不进入 Git。
`archive.json` 的 roots 映射说明原路径与保存位置。原收据保留历史路径；
执行旧二进制可能需要恢复其路径及 RPATH。归档保存字节，不代表重新跑过模型，
也不会自动让修改后的程序取得验收资格。

复用要求模型、二进制、库、输入等证据匹配。计算图、精度或预处理改动需要对应
数值验证；文档或计时输出改动使用针对性检查。旧收据不能改标为新执行结果。
F16 视频继续保持诊断状态，CPU 全量诊断仍延后。

## 正式性能样本与 Meta 资格检查

在仓库根目录、锁定的 Python 环境中运行；原始卡车图像、Meta 源码、权重和 BPE
按模型文档准备。`tools/generate_video_benchmark.py` 使用 `--image`、`--output`
生成固定的 64 帧单/四对象样本，生成本身不证明对象行为已经正确。

`tools/qualify_video_benchmark.py` 接收 `--fixture-manifest`、`--workload`、
`--sam3-source`、`--sam3-runtime-source`、`--checkpoint`、`--bpe`、`--output`。
它在声明总长 64 的序列上执行原始 Meta 模块前 17 帧，检查固定对象数、ID、
非空掩码和热启动输出。分别检查 one-object、four-object，使用不同输出。
这只是样本资格检查，不能替代模型全量验收。完整命令见英文页。

已有可信封存资格记录时，可使用 `--reuse-from` 和 `--parent-fixture` 替代重新
运行 Meta；新样本必须在全部 64 张 PNG 字节、位置、尺寸、提示词与序列协议上
一致。复用收据记录父收据哈希，输入、源文件或父记录不匹配会失败。
新样本仍可做新的 Meta 检查；复用模式不下载权重，也不执行模型。

## 完整性能协议与诊断计时

`tools/benchmark_video.py` 的 `--recipe-script` 使用
`tools/generate_video_benchmark.py`，同时提供原始数值参考、相同可执行程序的
完整通过收据、样本清单和资格收据。接电、避免睡眠，CPU/Metal 单/四对象四项
串行测量；检查 64 帧、16 帧预热、48 个测量样本、固定 ID、状态、后端和哈希。
最后一帧的排空开销包含在测量中。

当前 CLI 的 `runtime.image_ms` 计时包含视觉主干、neck 与几何编码；
`runtime.inference_ms` 包含提示准备、融合、检测和掩码，均包括传输与分配。
`text_ms` 表示最近一次文本编码，视频会话只执行一次；替换图像会再次编码，
这部分已包含在图像的 `inference_ms` 中。
历史数据没有这些字段时明确记录不可用，不用减中位数的方式补造阶段时间。
短序列可用于定位瓶颈，不计作新的 64 帧稳态性能验收。
后续优化继续保持 F16/BF16 边界、候选选择门槛与热启动延迟规则。

图像数值验收读取导出的 RGB PPM，图像计时通过 stb 读取原始 JPEG。
不同 JPEG 解码器可能产生不同 RGB；跨入口逐字比较必须使用相同像素。
本次图像测量保留同一 JPEG 的单次推理对照，检查输出一致性，不计为额外的
Meta 原始参考；数值门槛保持原样。见[当前测量](benchmarks/m4-pro-blas-vit-20261003_zh.md)。

[当前基线索引](validation-baselines/blas-vit-m4pro-20261003.json)记录 CPU BLAS/ViT
新实现。新私有归档包含 14664 个文件、43333663701 逻辑字节，位于
`models/validation-baselines/blas-vit-m4pro-20261003/bundle`，创建和独立复核均校验
每个文件及清单。新源文件、构建、输出与保留的失败诊断在此；原始模型及 Meta
参考继续由[父索引](validation-baselines/m2-m4pro-20261003.json)绑定的归档保存。
两个归档同时保留，不自动授予修改后二进制的通过状态。

索引另记录测量后的统计字段顺序/归档校验修正：隔离构建、旧初始化回归、
逐字节张量/输出对照及会话检查支持该元数据修正的针对性复用。
完整测量保留原始源文件/二进制身份；新构建不能把旧收据冒充为自己的完整验收。
