# CUDA W8A8 与 FP8 矩阵原型

创建：2026-10-07 15:53:34，Asia/Shanghai。状态：独立内核筛选完成，整图筛选待续。

## 范围与依据

这是[运行时量化路线](20261007-143003-activation-and-runtime-quantization.md)阶段 2
的真实内核筛选。上一批构建、模型、校准和测量保持冻结。先做独立诊断程序，
只有完整消费链在质量与性能／内存约束下获益，才接入 SAM 图和新算术 profile。
当前工具不增加可用的量化模型或后端支持声明。

实际视觉线性层为 `Y = W × X`，权重按 GGML `[K,M]`、输入 `[K,N]`、输出
`[M,N]` 储存。首组覆盖 N=5184，(M,K) 为 (3072,1024)、(1024,1024)、
(4736,1024)、(1024,4736)，分别对应 QKV、输出投影和两层 MLP。另以小矩阵、
尾行和零／大幅度输入验证布局与数值，不能只验证规整的 1024 方阵。

使用本地 CUDA 13.3／SM 8.9 和固定 GGML。官方
[cuBLAS 数据类型与布局文档](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul)
给出 INT8／FP8 的约束，但具体组合以本机 heuristic、实际执行和数值比较为准。
不把较新文档中的分组 scaling 支持外推到当前 Ada 设备。

## 原型设计

1. 从原始 checkpoint 与冻结的 COCO F32 校准抽取代表层的权重、bias、输入行
   和通道缩放。输入 reservoir 按确定顺序重复到实际 N，仅用于 kernel 尺寸与
   局部数值诊断，不当作独立图像质量评估。
2. 基线为固定 GGML 的原生 Q8_0 × F32 路径，包含 RHS Q8_1 staging；另给出
   dense F16 输入的对照。将 bias 和 MLP GELU 纳入完整链计时。
3. INT8 候选为 per-output-channel W8 与 per-token A8：INT32 累加后在 CUDA
   epilogue 做双方 scale、bias 和可选 erf GELU。先显式计入 INT32 中间输出、
   最终 F32／F16 输出和 workspace，不将反量化成 F32 的模拟称为整数 kernel。
4. FP8 候选采用 E4M3、F32 累加、tensorwide scale，比较 F32／F16 输出。
   通道重参数化在量化前执行。输入打包计入计时，静态权重转换只在初始化执行。
   保留精确的 erf GELU 语义；不直接用未经等价验证的近似 GELU epilogue。
5. heuristic 搜索和权重初始化不计入稳定调用；固定有界 workspace，记录实际
   算法与需求。分别报告量化、GEMM、epilogue、完整链的 GPU 时间和同步 wall time。
   分项计时只用于归因，不用分项中位数相加冒充完整链延迟。

## 验证与决策

- 每个内核先与解码后的相同整数／FP8 操作数的 F32 参考比较，单独评估最终量化
  对原始 F32 线性输出的误差。INT8 原始点积验证为精确 INT32，包含不饱和累加
  上界；检查 F16 输出溢出、全零行、对齐／尾部和量化 scale 方向。
- 保存原始输入、输出、参数、设备／库／源码身份和运行命令。测试在新目录发布，
  不能覆盖前轮证据。每种模式分别分配和释放设备内存，避免并存多份权重／输出
  扭曲峰值；进程内存和显式 buffer 数量分别报告。
- 在真实形状上预热后多轮完整链计时，交错候选顺序；没有并发 GPU 工作。只有
  胜过已优化原生 Q8 且无不可接受工作区开销的候选进入整图原型。
- 随后在独立 selection 图像做未量化重参数化与完整 fake-quant 筛选，选定候选
  后才接入实际算子，并在 evaluation 集和既有回归上做最终分割质量验收。
- 若两种 8-bit 路线都不能改善现有速度／内存组合，按主计划退出条件保留 A16
  成果，记录否决证据，不以单个 GEMM 吞吐或校准误差代替最终收益。

## 结果

### 构建与数值验证

增加默认关闭的 `SAM_BUILD_CUDA_PROBES`。两个独立程序分别调用固定 GGML 和
本机 cuBLASLt；没有更改运行时 SAM 算子或扩充 GGUF 格式。输入来自已冻结的
256 张 COCO 校准结果和原始 checkpoint，取 block 0 的四种矩阵、α=0.75／1
共八组，每组重复 512 条 reservoir 行到 N=5184。

CUDA 13.3.73、cuBLASLt 13.6.0.2、RTX 4090 SM 8.9 上构建成功。INT8 使用
per-output W scale，以及 per-token 或静态 A scale；INT32 点积后处理双方 scale、
bias 和 erf GELU。F32 输出复用同尺寸 INT32 缓冲；F16 输出当前需额外 INT32
结果，因此不能把输出元素减半当作整体显存减半。候选启发式限定相应 F32／INT32
累加 capability，workspace 上限 16 MiB，保留算法 ID 和实际需求。

`validate_linear_probes.py` 的 93 项检查全部通过，包括八组真实输入的六种模式、
尾行、零向量／通道、nearest-even 舍入、静态饱和、最大 K=16384 的 INT32 点积、
F16 溢出和非法输入拒绝。新 INT8 模式在 512 个位置核对精确 INT32 标量点积，
并对全部输出元素运行独立 CPU 编码／解码参考。FP8 只参与变换溢出的拒绝测试，
不计作 FP8 正常矩阵数值通过。GGML Q8 是既有已验收内核，此工具未另写 block-Q8
参考解码器。全部独立参考比较的最大相对 L2 为 `1.083e-5`。

检查期间修正了静态 A8 饱和：先在浮点域截到 [-127,127] 再转换，避免超出 INT32
范围的正数转换成负饱和。通道变换溢出显式拒绝；F16 非有限输出不发布成功报告。
此处 CUDA 与独立 CPU 参考都采用 F32 除法后舍入。前批离线研究为避免溢出，
使用 F64 除法；临界 tie 值可能不同，不声称两套 fake-quant 逐位相同。

- [诊断输入 manifest](../../models/calibration/coco2017-linear-probes-v1/manifest.json)
- [完整内核检查与来源身份](../../build/cuda-linear-probes-validation-v1/validation.json)

```sh
rtk proxy cmake -S . -B build/cuda-linear-probes-v1 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF -DGGML_CUDA_GRAPHS=OFF -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSAM_BUILD_CUDA_PROBES=ON -DSAM_BUILD_TESTS=OFF -DSAM_BUILD_EXAMPLES=OFF \
  -DFETCHCONTENT_SOURCE_DIR_GGML=/path/to/pinned/ggml
rtk proxy cmake --build build/cuda-linear-probes-v1 \
  --target sam_cuda_linear_probe sam_ggml_linear_probe
rtk proxy .venv-reference/bin/python -B tools/prepare_linear_probe.py \
  --checkpoint /path/to/original/sam3.pt \
  --calibration models/calibration/coco2017-vision-f32-v1 \
  --output models/calibration/coco2017-linear-probes-v1
rtk proxy .venv-reference/bin/python -B tools/validate_linear_probes.py \
  --binaries build/cuda-linear-probes-v1/examples \
  --inputs models/calibration/coco2017-linear-probes-v1 \
  --output build/cuda-linear-probes-validation-v1
```

上面列出本轮目录供复现定位；重跑需另选新输出目录，保留原始凭据不变。

### FP8 当前候选未通过完整数值门限

E4M3 两侧采用 tensorwide scale，显式关闭 `FAST_ACCUM`，并限制／核对 F32
累加的 numerical implementation flags。block 0 QKV 的第一个 raw GEMM 输出
实际为 `0.119235411286`，同一 packed 字节解码后的标量参考为 `0.119358591735`，
超过事先设置的 `2.33986429623e-5` 绝对容限。将两侧 scale 改为 1 后，实际点积
`4231.34375`，标量参考 `4235.71520996`，所以该差异不来自缩放或 bias/GELU。
Profiler 确认调用 `sm89_xmma_gemm_e4m3f32_e4m3f32_f32` 内核。

这些证据定位到当前 cuBLASLt 点积路径的数值行为，未确定内部成因。保留失败
报告，不调整误差阈值使它通过，也不将它推广为所有 FP8 内核都不可用。

继续单独检查 F16 输出：八组真实矩阵均通过预先定义的 F16 门限，完整解码参考
相对 L2 范围为 `2.015e-4`～`4.625e-4`，低于 `8e-4` 的既定上限。但追加边界
矩阵 M=48、N=33、K=96 时仍失败：混合正负／舍入临界输入的 raw 结果为
`18.640625`，已舍入为 F16 的解码参考为 `18.578125`，差 `0.0625`，超过预先
设置的 `0.0296122` 容限。该补充验证总计 15 项，14 项通过、1 项失败，报告
保留 `passed:false`。它阻止当前 FP8 F16 候选进入整图，不能只挑实际层的通过
结果发布。失败 JSON 中 `unscaled_fp32_actual` 只在 F32 模式执行单位缩放诊断；
F16 记录中的 0 不代表实际点积为 0。

两个输出类型的当前 FP8 候选均退出本轮整图集成；若后续提供满足契约的累加方案，
再重新做独立验证。

- [F32 累加限制后的失败记录](../../build/cuda-linear-probes-smoke-v3/fp8-f32/kernel-check-failure.json)
- [单位缩放隔离记录](../../build/cuda-linear-probes-smoke-v4/fp8-f32/kernel-check-failure.json)
- [实际内核 trace](../../build/cuda-linear-probes-smoke-v4/fp8-trace.nsys-rep)
- [最终构建的 F32 失败复现](../../build/cuda-linear-probes-diagnostics-v1/fp8-f32-final/kernel-check-failure.json)
- [F16 输出的补充验证：14/15，未通过准入](../../build/cuda-linear-probes-fp8-validation-v1/validation.json)
- [F16 边界失败](../../build/cuda-linear-probes-fp8-validation-v1/nearest-even/kernel-check-failure.json)

### 计时口径修正

首轮烟测将 GGML input 注册为 scheduler 输入，导致主机输入和每次 PCIe 传输，
与设备驻留的候选不对等。该轮约 2 ms 的 QKV 基线无效，不用于收益计算。已改为
独立 CUDA input buffer，与真实内部激活相同；修正后 Q8 烟测约 0.232 ms。
`smoke-v1` 保留为失败方法记录。正式对照在设备驻留输入下重新交错测量。

### 正式完整链对照与选择

在相同 RTX 4090、500 W power limit、driver 610.57.04 上，每个形状／模式启动
三个独立进程，改变模式顺序；每进程预热至少 250 ms，再同步测量 100 次。未运行
profiler 或并行模型计算；桌面的 Chrome 与远程桌面上下文仍驻留，单独记录在
metadata。所有下表输入使用 α=0.75。计时含 F32 输入打包和最终 bias／erf GELU，
不含权重初始化、heuristic 搜索、结果下载或数值检查。

| 矩阵 | GGML Q8，ms | Per-token INT8 → F32，ms | 改善 | 静态 INT8 → F32，ms |
| --- | ---: | ---: | ---: | ---: |
| QKV | 0.233412 | 0.204323 | 12.5% | 0.203612 |
| attention 输出投影 | 0.087123 | 0.068185 | 21.7% | 0.064961 |
| MLP lin1 + erf GELU | 0.623589 | 0.369302 | 40.8% | 0.368194 |
| MLP lin2 | 0.374077 | 0.301424 | 19.4% | 0.286185 |

这里比较的是整个线性消费链，收益包含 epilogue 融合；原生 GGML Q8 已有 RHS
Q8_1 staging，因此不能把全部差异归因于“首次使用 A8”。三轮分布、P95、F16
基线与输出对照、分项诊断时间都在原始 JSON。Nsight 确认 INT8 调用
`cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn_align16`。该名称来自本机
cuBLASLt 内部实现，本项目未新增 CUTLASS 依赖。

另启动三个独立进程，以 10 ms NVML 采样分别记录完整进程峰值下界，计入初始化
和验证阶段；它不等价于稳定推理的 allocator 高水位，也不能外推至完整 SAM。

| 矩阵 | Q8 进程峰值，MB | Per-token INT8 → F32，MB | Per-token INT8 → F16，MB |
| --- | ---: | ---: | ---: |
| QKV | 564.13 | 524.29 | 541.07 |
| attention 输出投影 | 526.39 | 461.37 | 473.96 |
| MLP lin1 | 599.79 | 559.94 | 593.49 |
| MLP lin2 | 633.34 | 578.81 | 591.40 |

上述值取三个进程的峰值中位数，MB=10^6 bytes。各进程包括不同的 library／pool
开销，不能与显式 buffer 计数混用。INT8 F32 路径显式存活分配分别为 93.44、
48.86、129.66、148.89 MB，四种形状所选 cuBLASLt workspace 均为 0 bytes。
F16 输出反而分别需要 125.29、59.47、178.76、159.51 MB：例如 MLP lin1 仍保留
98.21 MB INT32 输出，再增加 49.10 MB F16 结果。单层 F16 较快并不满足连续
A16 存储目标，下一轮需让缩放／bias／GELU 直接消费累加寄存器并写小类型，或
提供其他消除全尺寸 INT32 临时缓冲的方案。

量化误差仍须单独判定：四层 per-token INT8 F32 相对原始 F32 的 L2 误差分别为
0.005096、0.011672、0.017019、0.019499；原生 Q8 分别为 0.002158、0.003953、
0.009308、0.004654。静态 A8 更快一点，但四层误差均比 per-token 更大。这里只
使用校准 reservoir，不包含最终 mask、score、box 或 COCO 指标结论。

因此将 **per-token W8A8、F32 epilogue、α=0.75 起点**带入独立 selection 集的
整图 fake-quant／重参数化筛选；不把这个起点强制用于所有层，也不提前新增公开
profile。静态模式保留作消融，F16 输出先解决临时缓冲，当前 FP8 候选暂缓。
随后仍须完成真实 SAM 算子集成、独立 evaluation 与端到端显存／延迟验收。

内存采样第一轮遇到进程退出和 NVML 枚举之间的竞争，未发布完整结果。已完成的
72 个独立计时进程保持原样；修正采样器对已退出 PID 的处理后，仅在新目录补跑
72 个内存进程，再汇总两组记录，没有为了改变计时结果重跑基线。

- [正式计时元数据](../../build/cuda-linear-probes-performance-v1/metadata.json)
- [完整 144 进程序列](../../build/cuda-linear-probes-memory-v2/sequence.json)
- [完整时间／内存对照](../../build/cuda-linear-probes-memory-v2/comparison.json)
- [INT8 实际内核 trace](../../build/cuda-linear-probes-diagnostics-v1/int8-trace.nsys-rep)
- [INT8 kernel 统计](../../build/cuda-linear-probes-diagnostics-v1/int8-kernels.csv)
- [Compute Sanitizer：0 errors、0 bytes leaked](../../build/cuda-linear-probes-diagnostics-v1/memcheck-token-f32.log)
- [F16 分离输出缓冲的 Memcheck](../../build/cuda-linear-probes-diagnostics-v1/memcheck-static-f16.log)
- [隔离环境 81 项工具测试通过](../../build/cuda-linear-probes-diagnostics-v1/tools-tests.log)

新辅助头单独、重复包含的 C++17 编译检查通过；文档和 whitespace 检查通过。
默认 CPU 配置保持 probes 关闭，没有要求 CUDA 编译器或加入 probe target。
本批没有修改已验收的 SAM 运行时路径，未重复执行第一批的完整模型质量测试。

本批源码、实际执行文件／动态库身份、通过与拒绝的数值记录、计时和内存测量
汇总于[原型筛选归档](../../build/cuda-linear-probes-delivery-v1/receipt.json)。完整
SAM 路线继续进行，不将这份归档视为 W8A8 模型发布或整条路线完成。
