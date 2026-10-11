# 独立激活量化原生实现 A1–A4

日期：2026-10-10，Asia/Shanghai。状态：A1–A4 已实现并完成声明范围的有界验收。
承接[总体计划](20261010-212901-quantization-followups-implementation.md)和[激活量化设计](20261010-155435-native-activation-quantization.md)。

## 范围

首个模式限定 CUDA、Q8_0 存储的 SAM 3 vision 合资格线性层、F32 图激活／输出和 F32 cache。
显式动态 Q8 RHS 不是整图 INT8，也不覆盖 attention、卷积、text/fusion/decoder、视频或 FP8。
保留默认 backend-selected，拒绝未实现的后端／权重／compute／cache 组合。
现有内部 RHS 打包只有在控制、执行计数及实际路径都成立后才能作为该模式的实现证据。

## 当前源码核查

- GGML v0.26.0 的 `ggml_prec_set_src(..., GGML_PREC_Q8, 1)` 只允许最低精度；不能强制 Q8。
  CUDA `ggml_cuda_mul_mat` 仍按 MMVQ/MMQ 性能启发式选择，最后可进入 cuBLAS 浮点路径。
- 默认 Q8_0 MMVQ 暂存 Q8_1：32 元素块，half scale 和 half 原始和；MMQ 使用不同布局及 scale 合同。
  两条路径必须独立说明并验证，不能用 CPU Q8_0／Q8_K 或 PyTorch per-token INT8 编码替代。
- 混合 F16 验证再次实际证明：CUDA Q 路径的 F32 RHS oracle 必须考虑内部量化；F16 存储和激活开关是不同轴。

## 步骤

1. A1：整理可复用的纯 CPU 独立 MMVQ/MMQ 编码与点积参考，明确块宽、padding、round/clamp、zero、scale/sum 精度及支持的动态范围。
   对非有限、极端范围、tail、tie 和全零建立可复现用例；原生不满足合同须明确报错，不生成“已验证”回执。
2. A2：加入有界、固定身份的 GGML CUDA 原生控制。模型只提供通用线性角色／策略意图，后端负责能力判定和路径选择。
   强制策略禁止 cuBLAS／浮点解码 fallback，实际 kernel 路径及 scratch 字节必须能独立观测。
   不把现有允许降精度提示更名成强制量化保证；不改变默认分派。
3. A3：SAM 3 vision 图标记合资格层，完整盘点实际权重类型与支持的形状，在 backend 权重分配／图执行前拒绝不兼容组合。
   图和结果复用身份包含策略；覆盖重新设置图像、更换提示和重复结果，不沿用不同策略的缓存。
4. A4：加入标准库 public value/options、CLI 与版本化四轴配置，贯通 request/resolved/实际执行证据。
   保留历史配置、报告和 schema 解释；JSON 仍由 host 工具解析。文档和 changelog 只公布已实现范围。
5. 使用同一设备 RTX 4090 做小算子后完整两张图片有界比较，检查所有合资格请求与执行计数，无 CPU/BLAS 回退。
   原始 oracle 比较对象／mask／score／box；独立性能与诊断分开，不承诺提速或通用 COCO 质量阈值。

## 内核诊断配合

A2 的实际路径观测与后续内核成本采集共用严格 opt-in 原生边界接口，避免每个 probe 各自推断路径。
CPU 后续采集真实 Q8_0/Q8_K scratch、packing 和 arithmetic 阶段；GPU 至少提供同步的实际执行成本报告。
源代码 traits、合成参考打包和 graph observer 时间都不能冒充 kernel 内部抓取。
诊断明确额外同步／复制开销，正常性能路径关闭 observer／dump；保留未知项并说明范围。

## 验证与证据

- 独立 packed-byte／同操作数 scalar 或 INT64 dot、数值边界、支持／拒绝组合、计数／严格放置和缓存复用。
- Release CPU/BLAS 与 required-device CUDA CTest，公共头独立重复包含／两 TU、隔离工具全测。
- 新证据存入新的 Git-ignored `build/` 目录，仅本工作站可用；旧 GGUF、manifest、receipt 和对应二进制保持不变。
- 验证后清理可再生缓存／临时转换目录，文档及 whitespace 检查；不恢复 v2/v3 campaign。
- Metal 无可用设备，保持硬件待验；该缺口不影响 Linux/CUDA 工作。

## 结果

初始核查时尚不可选择该模式；混合 F16 producer 封存后才修改共享源码。

M5 producer 已封存。A1 的初始强制模式收敛为 **MMQ D4**，避免将 MMVQ 的 half scale／original-sum 合同混入同一个保证。MMVQ 既有独立验证器保留，但不是本模式的启用路径。`tools/quantize/native_activation.py` 新增独立 F64 理想编码／实际布局解码与既有独立 CUDA fast-reciprocal 容差验证：32 元素 scale、128 元素 D4 tile、512 行 padding、round-away 理想目标、允许原生四 F32 ulp 的 scale／边界相邻整数差异。

显式动态范围为每个非零 32 元素块的最大绝对值在 F32 的 `[1e-30,1e30]`，全零支持；非有限或超范围必须原生执行前报错，不能继续产生“已验证”的输出。该范围避免原生 reciprocal overflow／underflow。六项纯参考测试已通过：zero、精确可表示值、tail／多行／strided source、ties／sign、有限和动态范围拒绝、损坏 payload／非法形状。**这是数值参考，不是原生 A2–A4 完成证据**；当前仍不能选择该模式。

### A2–A4 实现与首轮验证

- 第六个固定 GGML 补丁提供 v1 configure/query/capture registry ABI。CUDA 后端强制 MMQ D4，动态范围使用实际输入的 F32 bits 检查，避免 fast-math FTZ 将 subnormal 错当全零。不能满足类型／形状／硬件或实际 WEIGHTS buffer 要求时报错；不走 MMVQ／cuBLAS／浮点 fallback。
- 模型仅标记通用 vision linear 角色；runtime 后端配置和查询实际节点计数。完整 canonical inventory 检查后、backend 权重分配前要求 32×4 个目标实际存储为 Q8_0。加载选项不可变，会话／图均保留所属 runtime，无法跨策略共享缓存。每次成功编码再次核对全部目标请求与实际分派。
- 公共标准库选项 `ActivationMode::CudaVisionQ8Rhs`、图像 CLI `--activation ggml-q8-rhs` 和配置 schema 5 已贯通。schema 1–4／原 GGUF 格式不变。报告绑定 request/resolved 策略、独立 arithmetic profile、实际 MMQ 计数及 payload／allocation 字节；性能配对支持两个不同 activation 配置。
- 原生小图首轮通过 9 个 K/N 组合、两次不同 RHS，共 14,430 个输出；真实打包 D4 scale/code/padding 与独立 F64 参考一致，实际操作数 scalar dot 最大误差 `8.63113e-06`。非有限、超范围、过小及 subnormal 输入在分派前失败；重新绑定后零输入成功，常驻 packed 权重不变。
- 首次回执断言要求 allocation 严格大于 payload，实际单列 MMQ 的 padding allocation 可以为零；打印实际值证明两者均为 576，修正测试为 allocation ≥ payload，没有修改内核结果或放宽 scalar 门槛。
- 完整模型 exploratory smoke 两次 vision 编码 256 个请求／实际 MMQ，零 CPU/BLAS fallback，换提示／重复结果不新增编码。该 smoke 尚未作为完整质量或性能证据。
- 最终源码复核发现 GGML 的 compute-capability 单位为 890 而非 89，改用 `GGML_CUDA_CC_ADA_LOVELACE`。新的第六补丁 SHA-256 `3610cb67714339da2bb98aaff13b70021bc4cd320a46b82946da381b69a4e8c4`、最终源码树 `36b384356d23e24a41132b5e4566d335822fdc39fc9c72a336188c525f4c1392`；重新构建并执行最终验收，不将旧 smoke 冒充最终 producer。
- 首轮 required-device CUDA CTest 39/39。隔离工具 251 项通过但因首次未设置新 validator 路径跳过五项；最终 suite 使用新 CPU 构建的显式 helper/validator 环境重跑，只有零跳过结果才封口。

最终证据位于 `build/native-activation-acceptance-20261010/`，仅当前工作站可用。保留初始日志、最终 producer 源码、真实 capture 与对应二进制。

### 最终有界验收

- 修正后 Release CPU/OpenBLAS CTest **23/23**、required-device CUDA **39/39**。公开头独立／重复包含、两 TU、源码升级及默认精度回归随相应检查通过。隔离 reference 工具最终 **253/253**、60.444 秒、零跳过。
- 真正原生 MMQ scratch 保存 18 组 source／实际 packed payload／Q8_0 weights／F32 output。第二个独立 Python 检查器验证 D4 bytes，并直接解析 Q8_0 wire scale/code，验证 14,430 个 scalar 输出，最大误差 **8.631133937342383e-06**。未使用 GGML traits 或原生 dot 作为该第二参考。
- 五组不兼容请求均在 CUDA 初始化前失败：dense 权重、混合 F16/Q 权重、CPU backend、F16 compute 和 F16 cache。离线 bundle 重读再次核对实际计数，重新 hash 的少一次分派也失败；F32 compute 的 cost wrapper 同样验证 activation 身份。
- 使用原始 checkpoint 的 F32 oracle，当前 Q8 vision GGUF 在 default／forced 两模式各比较两张应用图片、六个提示：六个参考对象全部匹配，零漏检／新增，三个空结果不变。平均／最小 matched mask IoU **0.999731759494 / 0.999441721553**；平均 score 差 **0.000739971813**，最大归一化 box 差 **0.000066578388**。该小样本不是生产准确率或独立 holdout 认证。
- 同一 Q8 GGUF 的 default／forced 全部 token IDs、query scores、query boxes、ranked queries 和导出 masks 相同。默认大 vision 矩阵本来可选择相同 MMQ；新模式提供强制保证、输入校验和执行回执，不虚构新的数值收益。
- 完整成本诊断两模式各 519 次 MUL_MAT，严格 CUDA0、零 CPU/BLAS fallback。forced 每次 vision 编码实际 128 次 MMQ；最大有效 RHS payload **29,859,840 bytes**、pool request **29,878,272 bytes**。诊断 observer 序列化同步时间不作为性能结果；GPU packing/arithmetic 内部分项仍未采集。
- 独立无 observer 配对测量：RTX 4090，CUDA/F32 compute/cache，同一 Q8 vision GGUF，一张 truck 图片，AB/BA 两组进程对、warmup 1／iterations 3，memory 单独进程 1 次。报告 VALID、无污染、离线汇总完全一致。full-image p50 **388.539862→409.059806 ms**（强制模式慢 **5.2813%**；两对均慢），p95 ratio **1.055246**。换提示 p50 **132.731746→133.398133 ms**，重复结果 **12.738694→12.606804 ms**，后者是结果缓存命中。
- GPU 进程峰值两模式同为 **3,535,798,272 bytes**；RSS 峰值 **1,007,321,088→1,006,981,120 bytes**，不据此宣称内存收益。动态范围 GPU 检查和每层 stream 同步已计入强制模式成本，默认模式保持不变。
- 无诊断性能进程断言整个换提示／重复结果阶段不增加 MMQ 计数；公开 `sam_image --repeat 1` 再验证两次图像编码 256 次请求／实际 MMQ，重复结果不新增编码。
- `verification-receipt.json` 已封存 **406** 个输入身份和 **2698** 个本地证据文件；对应 `build/native-activation-{cpu-blas,cuda}-current/` 的二进制和动态库不得覆盖。A1–A4 至此完成，A5／其他硬件或全图 INT8/FP8 不继承本验收。
