# 性能

[English](BENCHMARK.md) · [模型与精度](MODEL_ZOO_zh.md)

本页汇总已验证模型配置的最新完整性能记录。当前适配器为 SAM 3，其他模型和平台通过
验证后分别增加结果表。结果适用于注明的输入与测量硬件。权重名称表示存储格式，
计算和状态格式遵循各模型与后端的精度策略。

## 按操作系统与硬件查看

| 平台 | 硬件 | 测量范围 |
| --- | --- | --- |
| [macOS](benchmarks/MacOS-m4pro_zh.md) | Apple M4 Pro | CPU/BLAS、Metal 图像分割与视频跟踪 |
| [Linux](benchmarks/Linux-4090_zh.md) | NVIDIA RTX 4090、Intel Core i7-12700KF | CUDA 图像分割与视频跟踪，CPU/BLAS 图像基线 |

性能记录按 `benchmarks/操作系统-硬件.md` 命名，中文版本使用 `_zh.md` 后缀。
新增平台时单独记录测量环境、日期、负载及验收范围。

## 测量自己的输入

使用 Release 构建，固定解码像素、模型、后端、线程请求和电源条件。图像 CLI 的
`--repeat` 分别记录完整图像调用和缓存命中。视频样本生成、原始参考对照和完整
视频计时命令见[模型验证](docs/validation_zh.md)。

权重和计算缓冲表示分配计数，RSS 表示进程驻留内存，两者不相加。各阶段中位数
也不能相加来还原总耗时中位数。应用自身的输入需要独立检查质量与性能。

测量凭据、验收细节与历史记录保留在
[macOS 性能测量计划](docs/plans/20261004-214547-latest-complete-model-performance-records.md)和
[CUDA 实施计划](docs/plans/20261006-215722-cuda-backend.md)及
[CUDA 算子 profiling 计划](docs/plans/20261007-015552-cuda-operator-profiling.md)，
计算模式优化见[CUDA 执行优化计划](docs/plans/20261007-042706-cuda-execution-optimization.md)，
数据流优化见[CUDA 分精度与数据流优化计划](docs/plans/20261007-094949-cuda-precision-and-dataflow-optimization.md)，
布局优化见[CUDA 布局拷贝优化计划](docs/plans/20261007-113358-cuda-layout-kernel-optimization.md)，
激活存储优化见[激活与运行时量化计划](docs/plans/20261007-143003-activation-and-runtime-quantization.md)。
