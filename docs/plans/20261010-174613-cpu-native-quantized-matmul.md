# CPU 原生量化矩阵乘与内部精度

日期：2026-10-10，Asia/Shanghai。状态：实现与小规模 CPU 验证完成。

## 范围

在 `31aaf7e` 的转换预览、逐张量混合配置与 CPU 成本诊断基础上，评估并接入 pinned GGML 的 CPU 原生 Q8_0/Q6_K/Q5_K/Q4_K 矩阵乘。保留现有量化权重先解码为 F32 的默认路径，增加显式可选模式，并把内部 RHS 激活打包、点积／缩放／输出精度及证据范围写入报告和文档。

本轮仅构建 CPU、运行小型正确性与矩阵性能试验，不转换／推理完整模型，不下载模型或数据集，不运行大规模 GPU 验收。质量与性能保持独立；数值测试验证算子实现，速度结果只给实际测量形状的参考。

## 方法

1. 阅读 pinned GGML CPU 的算子支持、类型 traits、内部 RHS 转换、tiled／SIMD 分派和 BLAS 路径。复用现有 kernel，不添加自行维护的整数矩阵乘，也不凭 `GGML_PREC_F32` 声称内部只有 F32。
2. 在公共 `BackendOptions` 添加尾部 CPU 矩阵乘策略；统一配置／工具通过 CPU `compute.mode=native-quantized` 显式选择。原有 F32 路径和历史 receipt 含义不变。新模式只接受显式 CPU，量化 MUL_MAT 保留 packed 左操作数并强制实际 CPU backend，避免 BLAS 静默解码；其他算子遵循既有策略。
3. 将量化权重的 `Q8_0→Q8_0 RHS`、`Q*_K→Q8_K RHS` 类型来源、内部动态打包／整数局部点积／浮点缩放与 F32 输出分别说明。图上激活仍为 F32；该模式不是全图 INT8 激活量化，也不是全图固定整数或 F32 算术。来源静态分析、独立重建打包参考与图上实际观察分别标记，未单独采集的 kernel 时间保留未知。
4. 接通公开加载、图调度、精度说明、导出／配对性能与成本诊断入口。配置和 recipe 绑定实际 CPU 算术 profile；拒绝无效 backend／模式、非量化模型和不支持的算子／布局，不退回旧计算却沿用新标签。
5. 新增有界 CPU probe：同一批 packed 权重／F32 激活比较「解码 F32」「CPU 原生量化」；独立解码权重与 RHS 后 scalar 乘加验证实际点积，另报告相对原始 F32 激活的差异。覆盖四种格式、QKV 视图、非整 SIMD 输出行、Q8 的 4736 行宽、零值／离群值及不同列数。
6. 性能不启用逐节点 observer；warmup 后交错测量相同线程／数据／形状，报告中位数、分位数、原始样本、scheduler arena、权重／RHS 字节和速度比。激活单独打包微测量只用于诊断，不从完整 MUL_MAT 耗时相减。明确不含整模型预处理、加载及其他图的成本。

## 步骤与验证

- 先完成 GGML 评估与模式合同，再实现 runtime／调度与配置／报告贯通。
- Release CPU 构建，CUDA/Metal 关闭；小型 CTest 覆盖边界、数值、真实图类型／backend 和原有默认路径。
- isolated `tools/test_tools.py` 检查统一配置、原有报告兼容、策略身份、错误组合及命令传参；更新双语指南、README 和 changelog。
- 运行少量代表性矩阵的 CPU benchmark，记录 CPU、GGML revision／构建选项、线程、精度和 scope。不根据少量算子的收益承诺完整 SAM 提速，也不设置质量硬门槛。
- `tools/check_docs.py`、`git diff --check`。结果保存到新的 Git-ignored `build/` 目录，保持历史证据不变；验证后清理本轮无用暂存，保留当前构建和所需证据。

## 后续结果

### 接入与合同

- 公共 `BackendOptions::cpu_compute` 增加尾部 `CpuComputeMode::NativeQuantized`，CLI 对应 `--backend cpu --cpu-compute native-quantized`；统一配置仍沿用 schema 1/2/3，使用 `compute.mode=native-quantized`。新增[逐张量混合原生 CPU 示例](../configs/quantization/image-tensor-mixed-cpu-native.json)，权重分配与原 F32-compute 示例一致，GGUF 不需重转换或升级格式。
- 默认仍为 Q→F32 图路径，旧 profile 为 `ggml-quantized-weights-f32-v1`。新模式报告 `ggml-quantized-cpu-native-v1`，跳过共享权重 cast；scheduler 强制 Q `MUL_MAT` 在实际 CPU backend 执行，并在 compute 前复核分配。浮点算子仍可遵循既有 BLAS 策略。
- 显式 CPU、实际含 Q 权重、F32 cache 是前提。未知枚举、Auto/Metal/CUDA、纯 F32/F16 模型、独立激活精度请求及不支持的矩阵类型／布局会报错。配置先检查声明，recipe／runtime 再检查实际权重清单，避免全 F32 文件获得原生量化标签。
- 精度检查、质量导出、配对性能和执行成本工具贯通新模式。新 producer 将通用 `compute_mode` 与 `cpu_compute`、`cuda_compute` 分开记录；历史 F32/F16 receipt 仍可读取，缺少新字段不能声明原生 CPU 模式。图像结果和视频 manifest 同步输出这些字段；当前视频权重没有 Q 模型，因此实际加载仍拒绝该请求。

### GGML 源码评估与精度边界

固定 GGML revision 为 `d7cb574130e6f01ad25b3289685489200febcd74`（v0.26.0）。本机实际构建源码为 `build/cpu-precision-tools/_deps/sam-ggml-src/`（Git-ignored，本机可用），未修改 GGML kernel。

- `src/ggml-cpu/ggml-cpu.c` 中 CPU type traits 将 Q8_0 的 `vec_dot_type` 指向 Q8_0，将 Q6_K/Q5_K/Q4_K 指向 Q8_K。`ggml_compute_forward_mul_mat` 先尝试 tiled，再选择可选 LLAMAFILE／通用 dot；F32 RHS 在 kernel 内按对应 CPU `from_float` 动态打包。设置 `GGML_PREC_F32` 不会取消这个打包步骤。
- `src/ggml-cpu/tiled/tiled.cpp` 的 master switch 为 `GGML_CPU_TILED_MM`，默认打开，K 格式一般在 RHS 列数至少 8 时进入候选分派；另有 ISA、repack／layout 等条件。源码分派条件只能说明可能执行路径，不能替代运行时 scratch／指令抓取。
- `src/ggml-common.h` 中 Q8_0 每 32 值包含 F16 scale 和 32 个有符号 8-bit code；Q8_K 暂存每 256 值包含 F32 scale、256 个 code 和 16 个 INT16 block sum，共 292 bytes。标准点积／tiled 路径包含整数局部点积、浮点缩放和归约，输出为 F32。不能把它描述为全程 FP32、FP16、INT8 或固定一种累加类型。
- 原生模式图上 RHS／输出仍为 F32；内部暂存 Q8 是相应 matmul 的动态舍入边界，不是全图激活量化、校准格式、特征 cache 或新 GGUF 权重类型。成本报告的 `cpu_rhs_dot_type` 标注为源码 traits 解析；kernel 内部时间和算术捕获仍为 `NOT_COLLECTED`。
- 测试 helper 最初尝试通过通用 `to_float` 解码 Q8_K，但它是内部 scratch 类型，没有该回调。已改为独立读取固定 wire layout 的 F32 scale／signed code，并检查块大小；这是验证夹具修复，未更改生产 kernel。Python 测试同时修正模拟子进程误拦截系统信息查询的问题。

### 正确性与回归

Release CPU 构建使用 `GGML_CUDA=OFF`、`GGML_METAL=OFF`、`GGML_BLAS=OFF`、`GGML_LLAMAFILE=OFF`、`GGML_NATIVE=ON`、`SAM_BUILD_CUDA_PROBES=OFF`。全构建覆盖公共／私有／support 头文件独立与重复包含；CTest 保留两 TU 链接检查。

- `sam_cpu_quantized_matmul` 覆盖 50 组小型 CPU 场景：四格式、1/4 线程、1/8/17 列、QKV 分片、非连续 RHS 行、输出行尾、零值、离群值、Q8 的 32／4736 行宽、同图更换激活与 observer 解绑。每个小场景检查全部输出，独立 scalar double 点积使用解码后的权重和 CPU-packed RHS；容差为 `5e-4 + 5e-5 × |reference|`。另以原始 F32 RHS 单独计算舍入差异，不设模型质量门槛。
- 实际 observer 验证 Q 操作数、F32 RHS／输出、CPU placement、标准 RHS traits，未插入 Q→F32 权重 cast，常驻 packed 权重字节保持不变。关闭 tiled 后再运行同一 50 场景，仍全部通过。
- 完整 CTest **20/20 通过**（7.17 s）；isolated `.venv-reference/bin/python -B tools/test_tools.py` **235 项，234 通过、1 项因 CUDA device unavailable 跳过**（72.18 s）。新增 Python 测试覆盖合法／非法组合、旧 receipt、CPU/CUDA 身份分离、质量独立的配对配置及真实小型成本图；整模型 producer 在测试中仅为明确标注的 stub，不代表 SAM 推理。
- 配置示例以 `--context public-api` 校验通过。文档检查 **158 份文档通过**。`git diff --check` 通过；新文件也单独检查末尾换行、尾随空格及冲突标记。

### CPU 性能测量

CPU 为 **13th Gen Intel Core i7-13650HX**。实际链接 GGML 为 `0.26.0 / d7cb5741-sam-86e7140e59a8-fe72eb827241-45401f8e1732`，AVX2／AVX-VNNI 可用，AVX512-VNNI 不可用，BLAS 与 LLAMAFILE 关闭。未测 BLAS 启用组合、其他 CPU ISA 或平台。

`sam_cpu_quantized_matmul_probe` 对 21 个有界矩阵分别运行 1／4 线程；另做一轮 4 线程 `GGML_CPU_TILED_MM=0` 控制。每组 warmup 3 次，随后 15 个样本交替 AB/BA，按较慢变体校准 1–64 次 compute／样本。两种模式使用相同 packed 权重和图上 F32 激活，不启用 observer。计时包含每次图 compute 的权重解码（默认路径）或内部 RHS 打包（原生），不含初始化／传输；基线不是事先缓存的解码 F32 模型。

下表为 **默认路径 P50 / 原生路径 P50**；小于 1 表示原生更慢。`K` 为行宽、`M` 为输出行、`N` 为 RHS 列；QKV 的 M 合计三个分片。三轮各自进行内部配对，不将不同轮的绝对时间当作严格的分派因果试验。

| 格式 | K × M × N | QKV | 1 线程默认分派 | 4 线程默认分派 | 4 线程 tiled 关闭 |
| --- | --- | --- | ---: | ---: | ---: |
| Q8_0 | 256 × 257 × 1 | 否 | 14.224× | 4.075× | 4.241× |
| Q8_0 | 256 × 257 × 8 | 否 | 3.120× | 2.544× | 2.692× |
| Q8_0 | 1024 × 256 × 16 | 否 | 2.397× | 2.200× | 2.435× |
| Q8_0 | 1024 × 1024 × 64 | 否 | 1.622× | 1.566× | 1.597× |
| Q8_0 | 256 × 768 × 9 | 是 | 2.821× | 2.079× | 2.425× |
| Q6_K | 256 × 257 × 1 | 否 | 16.401× | 6.241× | 5.126× |
| Q6_K | 256 × 257 × 8 | 否 | 1.692× | 0.779× | 2.666× |
| Q6_K | 1024 × 256 × 16 | 否 | 3.421× | 2.340× | 2.347× |
| Q6_K | 1024 × 1024 × 64 | 否 | 3.266× | 2.889× | 1.670× |
| Q6_K | 256 × 768 × 9 | 是 | 2.130× | 1.024× | 2.696× |
| Q5_K | 256 × 257 × 1 | 否 | 5.380× | 2.235× | 2.359× |
| Q5_K | 256 × 257 × 8 | 否 | 0.911× | 0.547× | 1.524× |
| Q5_K | 1024 × 256 × 16 | 否 | 2.341× | 1.595× | 1.526× |
| Q5_K | 1024 × 1024 × 64 | 否 | 2.882× | 2.601× | 1.192× |
| Q5_K | 256 × 768 × 9 | 是 | 1.179× | 0.646× | 1.497× |
| Q4_K | 256 × 257 × 1 | 否 | 5.862× | 2.218× | 2.204× |
| Q4_K | 256 × 257 × 8 | 否 | 0.848× | 0.529× | 1.726× |
| Q4_K | 1024 × 256 × 16 | 否 | 2.289× | 1.648× | 1.924× |
| Q4_K | 1024 × 1024 × 64 | 否 | 2.830× | 1.973× | 1.621× |
| Q4_K | 256 × 768 × 9 | 是 | 1.120× | 0.625× | 1.716× |
| Q8_0 | 4736 × 65 × 3 | 否 | 6.778× | 4.022× | 5.414× |

默认分派在 1 线程有 19/21 组更快，4 线程有 16/21 组更快；较大 `1024 × 1024 × 64` 矩阵在 1 线程约 **1.62–3.27×**，4 线程约 **1.57–2.89×**。部分窄矩阵／小型 QKV 的 K 格式退化，4 线程 Q4 的 `256 × 257 × 8` 为 **0.529×**。tiled 关闭控制中小型 K 矩阵改善，但部分大矩阵收益下降；这提示形状／分派与线程开销值得进一步调查，不据此全局关闭 tiled 或改变默认计算模式。未控制 CPU 频率／绑核，没有显著性或跨硬件结论。

性能试验只检查边界／中间输出位置，三轮最大 sampled decoded-scalar 误差均为原生 **2.51e-6**、默认 **3.58e-6**。原生相对原始 F32 激活的 sampled 最大绝对差为 Q8 **0.0611**、K 格式约 **0.0226**；这是本组合成数据的激活舍入差异，不是模型质量阈值。小型 CTest 的全输出检查范围与性能 sampled 检查分别保留。

大矩阵默认 scheduler arena 为 4,718,592 bytes，原生为 524,288 bytes，差值正好为省去的 4 MiB F32 权重矩阵；不是进程峰值内存结论。原生 RHS packed payload 为 Q8_0 的 69,632 bytes 或 Q8_K 的 74,752 bytes，位于 kernel 暂存范围。单独的单线程 RHS pack 测量保存在 JSON，不能从多线程 matmul 时间中相减，不能将其解释成完整推理的 activation-packing 占比。

### 可复现命令与证据

本机保留 `build/cpu-native-20261010-174613/`，全部为 Git-ignored 本地证据，不随 checkout 分发。原始 JSON／日志保持不变，身份汇总在该目录的 `receipt.json`；历史 `build/tensor-policy-20261010-171252/` 未修改。

```sh
cmake -S . -B build/cpu-precision-tools -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF \
  -DGGML_NATIVE=ON -DSAM_BUILD_CUDA_PROBES=OFF
cmake --build build/cpu-precision-tools --parallel 2
ctest --test-dir build/cpu-precision-tools --output-on-failure
.venv-reference/bin/python -B tools/test_tools.py
.venv-reference/bin/python -B tools/check_docs.py

# Choose fresh output directories for every rerun.
build/cpu-precision-tools/examples/sam_cpu_quantized_matmul_probe \
  --threads 1 --warmups 3 --iterations 15 --output build/new-cpu-study-1thread
build/cpu-precision-tools/examples/sam_cpu_quantized_matmul_probe \
  --threads 4 --warmups 3 --iterations 15 --output build/new-cpu-study-4thread
GGML_CPU_TILED_MM=0 build/cpu-precision-tools/examples/sam_cpu_quantized_matmul_probe \
  --threads 4 --warmups 3 --iterations 15 --output build/new-cpu-study-4thread-dot
GGML_CPU_TILED_MM=0 build/cpu-precision-tools/tests/test_cpu_quantized_matmul
```

原始结果为 `study-1thread/report.json`、`study-4thread/report.json`、`study-4thread-dot/report.json`；`cpu-cost.json` 来自真实小型 Q4_K 图。对应 `.log`、构建、CTest、完整工具测试、文档和配置校验日志与 receipt 一起保存。测试临时目录已自动回收，保留当前验证构建和所需证据；没有新增 checkpoint／GGUF、数据集复制、整模型推理或大规模 GPU 验收。

### 后续优化边界

1. 优先在这些小型形状上比较 tiled/dot、线程数量及真实 SAM 形状清单，再考虑可审查的按形状策略；不引入未经验证的全局环境开关或自动精度降级。
2. 运行量化器之外的 CPU／BLAS 组合和其他 ISA 验证时单独记录编译与分派身份。内核 scratch 抓取、内部阶段计时尚未接入；当前单独 pack 试验不能补足这项证据。
3. 完整模型的应用图像软质量对照和端到端性能仍是后续独立测量，应单独限定输入与资源范围；本轮不扩展 GPU 验收。
