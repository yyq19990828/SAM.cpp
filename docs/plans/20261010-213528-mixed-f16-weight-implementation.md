# 混合 F16 权重实现与设备验证

日期：2026-10-10，Asia/Shanghai。状态：CPU/BLAS 与 CUDA 实现及有界验收完成；Metal 待硬件。承接[量化剩余功能计划](20261010-212901-quantization-followups-implementation.md)的 M5。

## 范围与合同

在现有模块／精确张量混合权重中加入 F16，与 F32/Q8_0/Q6_K/Q5_K/Q4_K 共存。使用配置 schema 4、SAM GGUF schema 7 和独立 `image-mixed-f16-linear-v1` 存储策略；旧配置 schema 1–3、GGUF schema 1–6 及策略哈希解释不变。schema 7 可有模块覆盖及可选非空精确张量覆盖，必须明确请求 F16；基础格式仍为四种 Q 格式之一。

F16 只覆盖现有合资格线性矩阵；bias、归一化、embedding、卷积、规范向量和既有保护矩阵保持 F32。精确覆盖继续 tensor > module > base，全部名称必须命中且不能绕过保护。K 格式的 4736 行宽 Q8 回退不变。全 F16/F32 覆盖不宣称 Q 算术，原生量化 CPU 模式仍要求实际存在 Q 权重。

## 方法与步骤

1. 扩展依赖最少的策略／配置解析及版本化哈希，旧 schema 明确拒绝 F16 混合。Python 与 C++ 独立实现相同合同及有界 canonical CSV。
2. 转换和预览共用逐张量分配。原始 F32 checkpoint 直接生成 F16 payload，检查有限值、类型、字节数和文件大小；只有最终实际 K 张量需要 native encoder，不把 F16 送入量化器。
3. 原生加载先校验完整原始库存，再按 backend 和每个实际存储类型决定分配：CPU 将 F16 提升至 F32，packed Q 保留；CUDA/Metal 保留支持的 F16。取消按全局 `general.file_type==1` 才提升 F16 的假设。存储 inventory 与运行时分配／内存分别报告。
4. 贯通 manifest、ModelInfo、精度检查、ranked 导出、独立性能与成本诊断；新增双语用法、格式合同、配置示例及 changelog。
5. 先跑小型真实转换与矩阵图，再由原始 checkpoint 转换一份完整混合模型，核对预览并在已有两张应用图片上做 CPU decode/native 与 CUDA 输出比较。性能独立测量，不根据 F16 文件变小推断 RSS／GPU 内存或速度收益。

## 验证

- 验证旧策略哈希／字节不变、F16 新身份、优先级、保护／未知／重复／未命中／损坏策略拒绝、F16 溢出与有限输入、类型及文件字节一致。
- CPU 小图分别验证实际 F16→F32 分配与 Q 保留，独立解码后 scalar 点积；同时覆盖 BLAS 开启和关闭、原生 packed 模式、QKV 视图和图复用。
- CUDA required-device 测试覆盖 F16/F32/Q 混合图、真实设备放置及独立数值参考；Metal 需 Apple 实机，当前用户确认无可用 Mac，保持硬件未验状态。
- Release CTest、公共头重复包含／两 TU、隔离 reference 工具全测、文档检查及 `git diff --check`。证据使用新的 Git-ignored `build/` 子目录，只有本机可用；不写旧 receipt／GGUF，验证后清理临时产物。

## 后续结果

CPU+BLAS 前序 producer 已完成并封存，开始实施 schema 4/7、F16 直接编码、逐张量 backend 分配和精度报告。四组 Python 检查已通过：版本/优先级/保护、损坏元数据、实际 F16 字节/计划/manifest 一致、非有限/溢出/reduced-source 的事务回滚。尚待原生小图、完整 CTest/工具全测和原始 checkpoint 完整模型验证，当前不宣称 M5 完成。

共享调度补丁的独立 CUDA 构建随后也包含本轮 F16 源码；最终增量重建后运行全部 required-device CTest。此前验证绑定的 CPU/CUDA 构建保持原样。新 F16 证据位于 Git-ignored `build/mixed-f16-acceptance-20261010/`，仅当前工作站可用。

小图已实际验证 F16/F32/Q4/Q5/Q6/Q8 共存、CPU F16 提升／CUDA F16 保留、QKV 切片、RHS 更换后图复用及常驻字节不变。CPU native observer 确认 packed Q 全部在 CPU，零 Q→F32 cast。CUDA 首次失败是测试误将 F32 RHS oracle 套用于会暂存 Q8_1 的 Q5_K 运算，日志定位到 Q5_K，源码确认打包路径；改用 Q8_1 scale/sum 精确可表示的 Q 侧 RHS，F16/F32 侧仍使用非 half-exact 激活，保持严格 scalar 误差阈值。CPU native 参考最初误用 Q8_K 公共空回调导致测试 SIGSEGV，gdb 定位到空函数；复用已有 CPU probe 的有界、独立 Q8_K 布局解码，未更改生产 kernel 或放宽容差。修正后上述 CPU/CUDA 小图均通过。

最终 Release CPU/BLAS CTest **21/21**、required-device CUDA **35/35**、隔离工具 **241/241**，全部零跳过。工具增加实际 F16+K 转换互操作及运行时分配字节身份检查；不再只凭存储库存推断 CPU 分配。初次完整转换因为 CLI 拒绝相对量化器路径退出，未生成 GGUF；使用绝对路径继续到新日志，原始失败保留。

原始 checkpoint 完整转换的 GGUF 为 **1288463232 bytes**，1133 张量实际 F16/F32/Q4_K/Q5_K/Q6_K/Q8_0 数量为 **97/871/95/1/36/33**。CPU 逐张量 F16 提升后的有效 payload 为 **1896969432 bytes**，CUDA 保留存储后的有效 payload 为 **1286698200 bytes**，报告已绑定原生实际分配字节；这些数值不包含 padding／scratch，不等于进程或设备峰值。

两张图片、六组提示的四种模式均相对原始 F32 oracle 完成比较：六个对象全部匹配，零漏检／新增，三组空结果保持为空。样本加权 mask IoU 平均／最小：CPU decode **0.998255/0.996357**，CPU native **0.998132/0.995990**，CUDA F32 **0.997197/0.995878**，CUDA F16 **0.997818/0.996100**。这是有界应用回归，不是 holdout 准确率认证。独立性能尚在串行测量，整体 M5 暂不提前封口。

独立无 observer 配对测量随后完成，CPU/CUDA 原报告和离线重新汇总完全一致，均 VALID、无污染；groceries 两提示，warmup 1、iterations 3、memory iteration 1，CPU 一对进程，CUDA 两对。基线为此前 schema-6 混合 Q 配置，候选为本轮 schema-7 混合 F16；这是两套权重配置比较，不是相同权重的纯 kernel 比较。

| 指标 | 基线 | 混合 F16 | 变化 |
| --- | ---: | ---: | ---: |
| CPU/OpenBLAS 完整图 p50 | 17241.413 ms | 17059.862 ms | −1.05% |
| CPU 更换提示 p50 | 2365.768 ms | 2326.190 ms | −1.67% |
| CPU 峰值 RSS | 2414690304 bytes | 3280363520 bytes | +35.85% |
| CUDA F32 完整图 p50 | 385.369 ms | 383.862 ms | −0.39% |
| CUDA F32 更换提示 p50 | 130.659 ms | 131.113 ms | +0.35% |
| CUDA 设备峰值 | 2476736512 bytes | 2761949184 bytes | +11.52% |

CUDA 两对进程的完整图比值分别 0.98736／1.00521，不能宣称稳定提速；CPU 的小幅变化也没有统计显著性保证。F16 提升与更多浮点权重增加内存符合实际分配，保留默认配置。M5 的格式、转换、分配、运行及本样本回归已完成，Metal 仍无实机验收。证据及只读 producer 源码封存在 `build/mixed-f16-acceptance-20261010/verification-receipt.json`，仅当前工作站可用。
