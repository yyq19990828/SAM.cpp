# 内核暂存抓取和 GPU 执行成本

日期：2026-10-10，Asia/Shanghai。承接[总体实施计划](20261010-212901-quantization-followups-implementation.md)。

## 范围和方法

补齐第五项工程缺口：CPU 原生 Q 权重矩阵乘的实际 RHS scratch、打包和算术阶段，以及 CUDA 同步节点成本。
诊断显式启用，独立性能测量保持关闭；旧报告与绑定的二进制、模型、manifest 和回执不改写。

- CPU 在固定 GGML 的 dot 与 tiled 两个真实实现边界采集，打包完成的线程 barrier 后读取实际 Q8_0/Q8_K payload。
  阶段时间由内核边界测量，不用单独打包实验相减。仅在启用采集时增加同步，说明 barrier 和诊断复制成本。
  用实际 scratch 独立解码并验证同操作数 scalar dot，更换 RHS 后重算，检查常驻权重不变。
- GPU 复用已经同步的 scheduler evaluation callback，明确包含同步、dispatch 和禁用融合带来的成本。
  CUDA 报告采用新 kind/version，不改写 CPU v1 的含义；检查实际 CUDA 放置和传输计数。
  GPU 内核 packing/arithmetic 未接入之前仍写 NOT_COLLECTED；同步图节点成本不冒充设备 timestamp。
- 固定新补丁、源码树和 build identity，支持四补丁旧树只读升级。创建独立构建保留混合 F16 验收构建。
- Release CPU/BLAS、required-device CUDA、隔离 Python 工具测试；真实小图和一图一提示的 GPU 完整模型诊断。
  检查文档、whitespace，清理无用缓存。证据留在新的 Git-ignored `build/` 目录，仅当前工作站可用。

## 步骤

1. 先封存正在运行的 M5 producer；再修改共享源码。
2. 实现并验证 GPU 成本报告、严格放置、旧报告兼容和篡改拒绝。
3. 实现 CPU 真实 scratch／阶段接口、小算子独立校验及明确的诊断报告。
4. 归档输入／源码／二进制身份，记录实际结果和范围；独立激活 A1–A4 继续在其计划实施。

## 结果

实现前计划。用户无可用 Mac，Metal 保留待实际硬件验收。

M5 producer 已封存，开始接入。首次新配置被完整源码树校验拒绝；根因是准备脚本用 Python Path 的分段顺序排序，而 CMake 按相对路径字符串排序。对原四补丁树按字符串重算恢复已知 `981aa6...`，对加入新补丁的树重算与实际应用结果同为 `15102cfb...`，因此修正期望身份；没有放宽完整树校验。原失败保留在证据目录。

新增固定 `ggml-kernel-capture.patch` 和 `ggml_cpu_sam_set_observer_v1` registry 接口；CPU dot／tiled 均在真实 packing barrier 后借出 payload。启用诊断才增加阶段同步，callback 不进入算术计时间隔；常规路径只有 enabled 检查、没有新增 barrier。observer 只允许空闲图期间安装／移除，进程局部，明确拒绝缺失阶段而不假定其他 repack／LLAMAFILE 路径。

`build/kernel-cost-cpu-blas-current/` Release 全部 **23/23** 通过，包括 dot-only 和实际 tiled 两个新测试、实际 BLAS 正控制、旧树升级及 public/private header／两 TU。两轮独立 probe 各 32 个执行 case、**96 组**实际 packing/arithmetic 边界：dot-only 全部 dot；tiled 轮 24 个 Q8_0 dot、72 个 K-format tiled。每轮 Q8_0/Q8_K payload 数量 24/72，实际 scratch 共 **495936 bytes**；最大 decoded scalar 误差分别 **2.21357e-06／1.43229e-06**。覆盖 1/4 线程、三 QKV 视图、N=3 strided 与 N=33、RHS 更换后复用、Q8_K sums、packed 字节、常驻权重及移除 callback 后普通计算。

新证据位于 Git-ignored `build/kernel-cost-acceptance-20261010/`，仅当前工作站可用。CUDA 成本 v2 和全工具检查仍在验证，尚未标记第五项完成。

CUDA 初次 **37/38** 通过，新增 cost 小图的参考失败。单项带数值复现为 actual **127.984375**、F32-RHS scalar **127.992188**，与 MMVQ 的 `F16(1/127)*127 = 0.99993896484375` 暂存误差吻合。将测试 RHS 改为 127/128 及其两倍，使 Q8_1 scale／original sum 精确可表示，严格误差阈值保持不变；再增加移除 observer 后同图正常执行结果完全一致检查。生产 kernel 未修改。修正后 CPU 全部 **23/23** 再通过，CUDA 全测和完整成本继续串行；原失败及 producer 源码保持不变，新 producer 另存。

隔离 reference 工具 **248/248**、64.265 秒，零跳过，包含 A1 的六项独立数值参考；这是工具合同验证，不把参考模拟当作已启用激活模式。

修正后的 required-device CUDA **38/38**、零跳过，已覆盖同步 observer 的数值、graph reuse、callback lifetime 和正常无 observer 结果一致。完整 schema-6 混合模型 truck/truck 一图一提示在 CUDA F32／F16 各运行一次：分别观察 4／6 个图、均 **519 次 MUL_MAT**，其中 **285 次 packed Q**，全部 CUDA0，无 CPU／BLAS compute。上传／下载字节均与运行时计数逐项一致。嵌套报告采用 `sam-cuda-execution-cost-v2`，CPU v1 含义不变；损坏版本、后端、计数、阶段和传输报告拒绝。

本次观测的 graph compute wall time F32／F16 为 247.816／192.162 ms，MUL_MAT 合计 60.760／67.401 ms。均为逐节点同步、禁用融合的诊断观察，**不能用于模式性能比较**，也不是设备 timestamp；GPU kernel packing/arithmetic 仍 NOT_COLLECTED。

第五项在已声明范围完成：CPU 真实 RHS 暂存和两阶段、GPU 同步节点成本；更多 ISA、repack／LLAMAFILE 或 GPU 内核分项没有继承该验收。输入／源码／二进制及原始失败保存在 `verification-receipt.json`，当前可用构建 `build/kernel-cost-cpu-blas-current/` 和 `build/kernel-cost-cuda-current/` 保留。文档检查 163 篇及 whitespace 通过，后续激活实施另建构建，不覆盖这些二进制。
