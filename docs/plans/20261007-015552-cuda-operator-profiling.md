# CUDA 算子 profiling 与推理优化计划

创建：2026-10-07 01:55:52，Asia/Shanghai。状态：完成。
基线：`92b56bc`；工作区开始时干净。本机 Linux x86_64、RTX 4090、driver
610.57.04、NVCC 13.3.73、Nsight Systems/Compute 已安装。

## 范围与约束

分析当前 CUDA 的真实算子、kernel、cuBLAS、传输、分配和同步耗时；分离
模型加载／首轮初始化、完整非缓存推理与结果缓存。覆盖代表性 F32/F16/
quantized image，以及一／四对象视频。以实际主要瓶颈选择优化候选。

保持 F32 dense/attention 精度、量化独立 arithmetic、严格 CUDA placement、
共享模型图、session/result 所有权与 video 状态／输出契约。不得通过 TF32、
F16/BF16 staging 或削弱验收 gates 提速。固定 GGML revision，shared upstream
checkout 不改；任何依赖变更仍采用独立 patch、SHA/tree 和 build 校验。

## 方法与步骤

1. 冻结当前 binary/library/source/model/input identities，保存基线 Nsight
   Systems traces、SQLite/CSV summaries 和未 profile 的完整调用计时。
2. 统计 kernel family、cuBLAS 形状、次数、GPU 时长、host launch/sync、数据
   传输。需要时用 build-local NVTX annotations 标识 GGML op/stage，明确其
   profiling-only 性质；必要时用 Nsight Compute 分析主要 kernel。
3. 根据占比、调用密度和 Amdahl 上限选择有依据的候选，隔离构建，比较精度、
   scratch memory 和未 profile 的同条件 A/B/A。失败候选保留记录，不发布。
4. 有稳定收益且满足数值与内存契约的候选才进入生产 patch/实现；运行实际
   影响路径的 direct probes、Release CTest、完整原始模型验收。共用路径变化
   应复核相关 image/video/quantized profiles 和 CPU-disabled-CUDA 回归。
5. 更新计划中的算子分析、尝试、原始 receipts、最终收益与边界。成功交付
   才更新 Unreleased 和双语简洁性能数据；不把 profiler 扰动后的时间作为
   正式推理性能。未找到可接受收益时明确记录结论，不修改生产代码凑结果。

## 验证协议

图像沿用 truck 1800×1200 RGB、一次冷调用、五次替换图像的完整调用和独立
缓存统计。视频正式计时沿用 64/16/48 帧和一／四持续对象；profile 可先取
相同输入的短序列，记录实际长度，不替代完整官方五 case/216 帧验收。
全部 GPU work 串行，不与编译／其他 inference 混跑。GPU counters 必须归属
所选设备且 CPU/Metal/BLAS compute 为零。RSS、GGML allocation 和采样显存
分别统计，baseline/candidate 使用同一 driver、power、inputs 和参数。

## 结果

### 图像基线与候选

Nsight Systems 2026.1.3 的 F32/F16/full-Q8_0 完整 traces、SQLite、CSV 位于
`build/cuda-operator-profiling-v1/`；只分析五次完整 warm 调用，排除首轮。
以 12,192,768-byte 图像上传及 66,355,200-byte mask-logits 下载界定 GPU
区间，不能把该区间当作端到端时间。原始统计见
[warm kernel summary](../../build/cuda-operator-profiling-v1/warm-kernel-summary-v1.json)。
F32 每次 152,772 kernels、GPU kernel 合计中位数 552.423 ms、区间
774.760 ms；F16 153,191 / 554.831 / 781.077；full-Q8_0
153,378 / 446.763 / 673.816。小 SGEMM、split-K reduction、attention
softmax/output 是主要调用来源，F32 每次 softmax/output 各 27,672 次。

原 attention 将每个 head 的 query 分成最多 128，窗口 576 queries
要分五块；每块两个 pedantic F32 cuBLAS products、softmax 和 output copy。
保持算法、head loop、类型、mask、GQA 和数值 gates，仅比较上限
256/512/1024。实验 library 只重编译 `sam-precise.cu`，其余对象复用，明确
记录为 prototype；实际 ldd 和运行中进程 maps 均确认所选 library。
正式交付另行 clean build，不修改原 baseline build。

[未 profile A/B/A](../../build/cuda-operator-profiling-v1/tile-timing-v3.json)
为同一 F32 truck，一次冷调用、五次 full、五次 cache；中位数 baseline-before
831.549 ms，256 为 688.584，512 为 620.791，1024 为 589.295，
baseline-after 为 832.396。1024 相对首 baseline 减少 29.13%，两次
baseline 相差约 0.10%。采样 GPU peak 4,758,437,888 → 4,777,312,256
bytes（增加 18,874,368 bytes）；该结果不是正式完整模型验收或发布性能。
所有 candidate 的 truck mask SHA 一致，box 一致，score 差约 2.4e-7。

选 1024：已取得足够收益，score scratch 单 head/单 tile，当前 image
最大 keys 5,184 时 21,233,664 bytes，另有 head-dimension × tile output。
不扩展为整张 attention 矩阵，也不增加全 heads 并发。原 23 项 CTest
用 prototype library 全部通过；补充 1,025 queries 的独立 scalar gold
验证跨 tile、单 query tail、全 mask、GQA、strides 和 interleaved heads。

保留失败 harness v1（ldd alias 路径判断错误）、v2（rtk wrapper 下未递归
找到实际 sam_image process）；v3 完整且独立，只有 v3 用于 A/B/A 结论。
Nsight Compute 2026.2.1 首次参数 `--repeat 0` 被 CLI 拒绝，修正为 1
后 driver 返回 `ERR_NVGPUCTRPERM`，故无 occupancy/带宽等 hardware-counter
结果，不据此推断 counters；不更改系统 driver 权限。Nsight Systems 的
kernel/API/transfer 数据有效。额外 profiling controller v1 的 baseline
library 路径错误，尚未启动 inference；v2 使用实际 `_deps/ggml-build` 路径。

### 正式候选来源与兼容性

GGML revision、Metal patch 与其 tree 不变。CUDA patch 仅 query 上限
128 → 1024，新 SHA `e62f040cf9893f1eebdd8745f4d9a513d91befcbbe7330271bb61b41eb40c8b7`，
combined tree `25e659a3dfac0b16b5d7b455ee7e469b015573e22946aaa57f8a295c875e85c2`。
CMake 仍严格验证完整 tree、patch applicability、provenance；原 pinned
shared checkout 与 baseline binaries 保留。量化器编码及 GGUF 字节未变，
显式允许先前已验证的 `353b63b4-sam-0a0b80dd15c2-9417f66f5488` converter
来源，保留已有权重 sidecars；不接受任意 suffix。回归验证旧、新来源均
允许，未知 suffix 和缺失 encoder/provider provenance 仍拒绝。

正式构建、完整模型验收与正式性能见后续完成记录。

首个 clean configure 被 full-tree 校验拒绝：prototype 哈希脚本按 Path
components 排序，而 CMake 按完整 relative-path 字符串排序。修正排序后
baseline 正确复现旧 tree hash，新 tree 与 CMake 实际计算一致。保留
`cmake-configure-v1.log` 和 source identity v1；采用 source identity v2。

### 优化后算子结构

[统一 profiling summary](../../build/cuda-operator-profiling-v1/operator-summary-v1.json)
保留每次 GPU interval、原始 kernel family、CUDA API、传输类型与字节数。

| Image profile | Warm kernels / request | Kernel sum median, ms | GPU interval median, ms |
| --- | ---: | ---: | ---: |
| Baseline F32 | 152772 | 552.423 | 774.760 |
| Tile 1024 F32 | 39916 | 359.099 | 511.223 |
| Baseline mixed F16/F32 | 153191 | 554.831 | 781.077 |
| Baseline full Q8_0 | 153378 | 446.763 | 673.816 |
| Tile 1024 full Q8_0 | 40522 | 252.601 | 405.280 |

F32 kernel 数减少 73.87%，softmax/output 各从 27,672 降到 5,568。
两个 cuBLAS products 的 query 维更大，减少小 SGEMM 与 split-K launches；
512/1024 不改变 Q/K/V staging、pedantic mode 或 softmax 公式。prototype
warm 总 API launch 时间也下降，但 profiler 的每次 call 开销变化，不把该
时间作为 host 无扰动性能。传输字节与 compute node inventory 不因 tile 改变。

17 帧 baseline 视频（包含首帧和初始化）的 one/four-object traces
分别有 2,592,836 / 2,715,935 kernels，kernel sum 11,435.982 /
17,274.489 ms。一对象 trace softmax/output 各 464,570，四对象追踪
增加 SIMT/CUTLASS SGEMM 与 tracker 运算。此短序列只诊断算子，不提供
64/16/48 的 steady-state timing 或完整 ID/occlusion 验收结论。

优化后 F32 主要 kernel 时间转向较大的 SGEMM（如 `128x64_tn` 和
`128x128_tn`）；保持 F32 精度时，后续 batching/图像特征传输等改动需要
单独设计与验收。本轮不引入它们。cuBLAS verbose trace 只有 domain
registration、没有可用调用 range，不能声称已取得完整 GEMM API shape
trace；小 GEMM 的来源以 pinned attention 代码和实际 kernel counts 验证。

### 正式验收安排

Clean build `build/cuda-attention-optimized`，Release/Ninja、arch 89，
`GGML_CUDA=ON`、Metal/BLAS/CUDA graphs off、required CUDA tests on。
[clean source audit](../../build/cuda-operator-profiling-v1/clean-source-audit-v1.json)
确认新旧 prepared tree 文件 inventory 相同，仅 `sam-precise.cu` bytes 改变，
且与 prototype 一致。原始 GGML 和 Meta checkouts 的 Git status 均干净。

69 项 isolated-reference `tools/test_tools.py` 已通过，包含旧／新 quantizer
来源和未知 suffix 拒绝回归；日志中的负向 fixture `FAIL` 是预期拒绝样例，
unittest 总结果为 `OK`。全库文档检查当前 55 documents 通过。
正式编译完成后，串行运行 CUDA 23 / CPU-disabled-CUDA 14 CTest、dense
F32/F16 七 case、八量化预设七 case（原始 output quality + decoded diagnostics）、
F32/hybrid 五 case/216 帧 video、六项共享 session（含各 64+64 交错 long
video）。独立 benchmark fixtures 用原始 Meta 模块重新做 17-frame prefix
qualification，再测十 image、四 video cells。基线、prototype 和原始验收
records 均保留，不覆盖旧完成 receipts。

### 复现 profiling

基线 binary 路径始终是 `build/cuda-probe`，其 prepared source/tree 是旧版
query 128；新 production build 独立为 `build/cuda-attention-optimized`。
分析前保留 binary/library/model/input SHA、CLI repeat inventory。示例：

```sh
rtk proxy nsys profile --sample=none --cpuctxsw=none \
  --trace=cuda,cublas-verbose --stats=false --force-overwrite=false \
  --output build/cuda-operator-profiling-v1/image-f32 \
  build/cuda-probe/examples/sam_image --model models/sam3-f32.gguf \
  --image models/reference/sam3-f32-cuda/truck-truck/input.ppm \
  --text truck --backend cuda --threads 4 --repeat 5 \
  --output build/cuda-operator-profiling-v1/image-f32-output
rtk proxy nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  --format csv --output build/cuda-operator-profiling-v1/image-f32-stats \
  build/cuda-operator-profiling-v1/image-f32.nsys-rep
```

已有 immutable 输出路径不能重用；复现应选择新的 versioned output。
`operator-summary-v1.json` 的 image 区间来自 SQLite GPU memcpy anchors，
保留每次 sample 而不是直接除整个 trace；video aggregate 明确含 startup。
分析脚本保存在 `build/cuda-operator-profiling-v1/summarize_profiles_v1.py`。

视频后续候选：four-object trace 中
`cutlass_80_simt_sgemm_256x128_8x4_tn_align1` 占 kernel 时间 31.8%，
8,960 次；one-object 同 family 2,240 次。共享模型
`include/sam/internal/models/sam3/tracking/attention.hpp` 的
`tiled_memory_attention` 是独立 head-256/F32 GEMM 路径，query 分块为 128，
不在本轮 head-32/64 backend specialization 中。扩大该分块会改变所有
backends 的共享图和 workspace 峰值，应作为独立候选测量，不能凭此 trace
把全部 CUTLASS calls 直接归因到该函数。本轮保留该共享图边界。

Clean CUDA Release build 已完成；正式 CTest 23/23 全通过、零 skips。
新增 head-32 Q1025/K131 的 double scalar golden 最大误差 1.8276e-7，
head-64 Q1025/K137 同样通过现有 5e-6 absolute bound；未改变 bounds。
原 head-64/K5184 的误差为 1.64295e-7。完整 raw 输出在
`build/cuda-attention-optimized/Testing/Temporary/LastTest.log`，结构化报告为
[CUDA CTest](../../build/cuda-attention-optimized/cuda-runtime-v1.xml)。

CPU-disabled-CUDA fresh Release build 的 14/14 CTest 全通过、零 skips；
[CPU CTest](../../build/cpu-attention-regression/cpu-runtime-v1.xml)。
正式 CUDA F32/F16 两组各七 case 的原始 checkpoint 图像验收全部通过，
[dense original acceptance](../../build/cuda-attention-images-v1/sequence.json)。
truck F32 的 mask-logits normalized L2 为 2.5289e-5，vision features
为约 4.53e-6～1.15e-5，均满足冻结 gates；这不是 prototype mask 相等
替代原始模型验收。production ldd 独立确认新 build 的 CUDA library，见
[loader audit](../../build/cuda-operator-profiling-v1/production-loader-audit-v1.json)。
head-64 Q1025/K137 double golden 最大误差实际为 1.97789e-7。

八个 CUDA Vision/Full × Q8_0/Q6_K/Q5_K/Q4_K 预设全部通过各七 case
原始 output-quality gates（56/56），runtime CUDA arithmetic profile 均有效。
新 records：
[quantized acceptance](../../build/cuda-attention-quantized-v1/sequence.json)。
`tensor_fidelity_passed=false` 仍独立保留；未把量化中间张量描述为原始 F32
等价，也未改变压缩/native arithmetic diagnostic 的 gates 或权重数据。
结合 dense image，本轮十个 CUDA image 配置原始验收共 70/70 case 通过。

正式 F32 video 五个原始 case（motion 48、entry 64、occlusion 64、
hotstart-removal 24、negative 16）共 216 帧全部通过 tensors/masks/IDs/
state gates。记录为
[F32 video acceptance](../../build/cuda-attention-videos-v1/f32/metrics.json)。
Hybrid、shared sessions 与正式 timing 继续串行执行。

Hybrid video 同样完成全部五 case/216 帧并通过；两种 video 配置合计
432 帧原始模型 tensors/masks/IDs/state gates 通过。
[完整 video acceptance](../../build/cuda-attention-videos-v1/sequence.json)。
继续执行原始模型共享 session、长序列内存检查，以及独立已资格化 fixtures
的正式 timing；无 profiler、无编译重叠。

六个 original-weight shared/session checks 全部通过，含 F32/hybrid 各
64+64 交错 long-video state/ownership/workspace-bounds：
[session sequence](../../build/cuda-attention-sessions-v1/sequence.json)。
原始 Meta 模块重新完成 one/four-object 的 17-frame prefix qualification，
仍声明完整长度 64、维持 expected delay 14 与 ID/对象记录：
[fixture qualification v3](../../build/cuda-video-benchmark-qualification-v3/sequence.json)。
它绑定当前工具 source hashes，不覆写 v1/v2，也不绕过 provenance checks。
正式未 profile 的十 image / 四 video cells 开始执行；源码、library 与模型
identities 运行前后核对，采样 GPU memory 可能漏过短暂 scratch 峰值。

## 完成交付

[串行 delivery](../../build/cuda-attention-delivery-v1/sequence.json) 与
[qualification pipeline](../../build/cuda-attention-pipeline-v1/sequence.json)
全部完成。CUDA 23、CPU-disabled-CUDA 14、isolated tool 69 检查通过；
十 image 配置 70/70 original cases、F32/hybrid video 各五 case/216 帧、
六 original shared-session runs（含各 64+64 交错 long video）通过。
[最终产物一致性 audit](../../build/cuda-attention-delivery-audit-v1.json)
190 checks 全通过，重新核对每个独立 model/binary/library/qualification/
timing source snapshot，不重写原 completed receipts。

[正式未 profile timing](../../build/cuda-attention-performance-v1/sequence.json)
共 14 cells：十 image 的 cold/five-full/five-cache、四 video 的完整
64/16/48 一／四对象。与此前 same-host 完整合格 baseline 比较，
F32/F16 image median 837.321/851.330 → 589.913/604.971 ms；
八 quantized image median 为 485.979～496.310 ms（降低约 32.7%～34.0%）。
F32 video one/four-object median 1,176.676/2,004.978 →
932.494/1,751.461 ms；hybrid 1,176.280/2,007.518 →
934.176/1,755.855 ms。历史 CUDA receipts 来源 image v1 / video v2
不变；source patch 的版本差异是预期，baseline binaries/compiled libraries
保持原样。精确值、14-cell model/receipt hashes 和比较范围见
[final comparison](../../build/cuda-operator-profiling-v1/final-comparison-v1.json)。

独立 clean production 的
[A/B/A receipt](../../build/cuda-operator-profiling-v1/production-timing-v1.json)
使用 default loader、真实 `/proc/PID/maps`、binary/library/source-provenance
identities：baseline-before 832.519 ms、production 589.618 ms、
baseline-after 836.072 ms，latency 减少 29.1767%，baseline drift 0.4267%。
全部仍为 F32，Cuda device 0 / threads 4 / full image replacement / five samples；
不使用 Nsight 时钟作为 release 性能。首次 helper 因调用系统 Python 缺失
`gguf` 在 import 阶段退出；改用 `.venv-reference` 后执行完整独立 receipt。
未启动失败版本 inference，也未覆写任何已完成实验记录。

### GPU memory 采样修正

原每秒一次的 sampler 在约 0.49 秒 full-image calls 中有 aliasing 风险：
Vision Q6_K 的正式 timing 记录只采到 2.498 GB，不能作为真实 VRAM 峰值。
十个 image profiles 另做与 timing 相同 model/binary/input/CLI workload 的
独立 memory runs，使用持久 `nvidia-smi --loop-ms=50`；实际 PID、library
maps、artifact hashes 和 CUDA-only node placement 全核对，每个 process
74～98 个有效 samples。此过程的时钟全部排除正式性能。

[image memory profiling](../../build/cuda-attention-memory-v1/sequence.json)
全部通过；F32 peak 4,777,312,256 bytes，Vision Q6_K 为 3,460,300,800。
双语表 image GPU 列采用该单独约 20 Hz 采样；video 继续采用 64-frame
正式 timing 中每秒采样，明确均为真实峰值的下界。RSS 仍取 timing process，
不混用 memory-run RSS。Score/result scratch 在最大 image K=5,184、D=64
时从 2,686,976 增到 21,495,808 bytes（约增加 18.809 MB）；仍单 head、
单 query tile、CUDA pool/stream 顺序复用，query cap 与全 query 数独立。

双语当前性能表和 Unreleased 已更新，macOS 历史数据与同日 CPU/BLAS
baseline 不变。首个 table-update helper 未识别中文“全模块”标签，中文页
写入前 assertion 拒绝；修正标签后两页完整更新并校验，不修改任何 machine
receipts。按用户要求，此次优化以独立本地 commit 记录，未 push。

最终全库文档检查：55 documents 的双语表与本地链接 PASS；
`git diff --check` PASS。包括历史 archive-link 修复后的全库检查，未改动
checker 或历史机器凭据。production A/B/A 的三次 truck binary-mask SHA
均为 `ad619f88c7ea8f138cced42c9d35d236bcf620e8242b3de3248c5ccf3dc2394e`；
同 cadence 下 baseline GPU peak 4,758,437,888 bytes，production
4,777,312,256 bytes（增加 18,874,368 bytes，约 18.9 MB）。

[发布数据校对](../../build/cuda-attention-publication-audit-v1.json)通过：
双语表共 28 个 CUDA 单元格与已资格化 timing / 独立 memory receipts 的
median、P95、RSS、GPU memory 逐项一致，十个 memory CSV 的 SHA 再次核对。
