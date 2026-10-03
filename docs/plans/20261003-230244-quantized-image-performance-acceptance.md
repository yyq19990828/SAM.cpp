# 量化图像性能验收

创建时间：2026-10-03 23:02:44，Asia/Shanghai。
状态：正式采样及数值/身份核验完成，18/18 单元通过；用户确认接电后授权执行。
基线：37206da 加已封存量化实现/数值修复；旧收据与归档不覆盖。

## 范围与协议

只测已通过输出主验收的四个 image-vision-linear Q8_0/Q6_K/Q5_K/Q4_K profile，并以原始 F32/F16 作同构建基线。CPU/BLAS、无 BLAS CPU、Metal 分别比较，共十八个计时单元，全部串行。使用已冻结 truck-truck 的 1800×1200 PPM 输入、truck、阈值0.5、模型输入1008×1008；PPM避免JPEG解码器差异，本次数字与历史JPEG数字单独呈现。

复用 sam_image --repeat 5：一次初始完整推理预热，五次 set_image 后完整推理，不把五次结果cache命中当稳态速度。保留加载、冷启动、五个全图/stage样本及 median/min/max，process peak RSS 由 /usr/bin/time -l 整个子进程采集（含加载与最终PNG写入），权重/compute buffer分别列出。BLAS内部临时内存无法单独计数，记为不可获得，由进程RSS覆盖；不能将分配计数相加称为RSS。

请求threads4、VECLIB_MAXIMUM_THREADS=4；每个单元前后记录AC/thermal、无系统睡眠证据，caffeinate仅覆盖任务。确认没有其他SAM/Meta推理或编译重叠，日常桌面服务保留，不固定频率。不把AC attached但not charging解释为电池供电。每个单元绑定实现源快照、输入/模型/侧车、二进制/库、原数值/gate收据hash，前后确认不变。

## 步骤与验证

1. 复核已封存84输出质量、42原模型和6session资格。原始无BLAS F32/F16补齐当前构建七case数值资格（此前只有BLAS/Metal原模型回归）。这些验证与正式计时串行，不能重叠。
2. 使用现有CLI和标准库写被忽略的执行/收据脚本；不改推理代码、不安装依赖。计时CPU/BLAS六项、Metal六项后完成无BLAS资格及六项计时。每组F32先、四种量化依次，F16最后；比较顺序和非交错限制明示。
3. 检查6次vision/text/inference计数、5个全图样本、后端实际节点分配、finite时间、median一致、原高置信度检测/掩码/框/分数质量。当前数值test_image与sam_image共享实现/库；相同PPM对应的原始Meta质量门槛验证最后输出，既有逐张量失败仍独立保留。
4. 只有电源/无睡眠/身份/质量/样本完整通过才纳入正式表；失败尝试保存，修正后独立目录重测。温度警告或其他模型重叠导致该单元不合格，恢复条件再重测，不改质量门槛。
5. 报告量化对同后端F32/F16稳态时间、RSS和驻留权重比例；RSS未降低则只声明文件/权重压缩，速度如变慢如实列出，不设未经测量的加速目标。冻结新私有收据/源码/二进制，验证归档清单和所有hash，更新双语bench、量化说明、changelog及独立性能索引。

## 结果

正式 v2 在 2026-10-03 23:13:48–23:59:53（Asia/Shanghai）串行完成。十八个配置全部通过 1+5 完整推理计数、最后输出的原始 Meta mask/score/box、实际后端节点、AC/thermal/sleep 检查与源码/二进制/库/模型/输入前后哈希校验。无 BLAS F32/F16 新构建额外十四个原始完整张量样例全部通过；已有 84 量化输出质量与六个 session 资格保持原结论，量化的完整张量保真独立失败不改写。

| 配置 | CPU/BLAS 秒 / RSS GB | Metal 秒 / RSS GB | 无 BLAS CPU 秒 / RSS GB |
| --- | ---: | ---: | ---: |
| F32 | 7.353 / 4.928 | 5.373 / 4.245 | 38.857 / 4.833 |
| Vision Q8_0 | 7.352 / 3.751 | 5.322 / 2.937 | 39.234 / 3.543 |
| Vision Q6_K + Q8_0 | 7.348 / 3.678 | 5.348 / 2.868 | 38.851 / 3.676 |
| Vision Q5_K + Q8_0 | 7.364 / 3.641 | 5.370 / 2.829 | 39.230 / 3.645 |
| Vision Q4_K + Q8_0 | 7.360 / 3.606 | 5.339 / 2.793 | 39.180 / 3.413 |
| Mixed F16/F32 | 7.375 / 4.954 | 5.349 / 2.661 | 39.007 / 4.901 |

十二个量化配置全部满足本协议峰值 RSS 不高于同后端新 F32 基线：CPU/BLAS 降低 23.88–26.83%，Metal 30.82–34.21%，无 BLAS CPU 23.94–29.40%。CPU/BLAS 时间比为 0.9993–1.0016，Metal 0.9904–0.9994，无 BLAS 0.9998–1.0097；有限五样本的小差异不作为加速证明。Metal F16 的实测进程峰值内存低于四种视觉量化配置，必须保留这一比较。驻留权重相对 F32 的压缩与整体 RSS 改善分别报告；不将 BLAS 自有 workspace 误称为 scheduler compute buffer。

最初 v1 漏查 Python 模型进程，且未显式限制 Release/在无 BLAS baseline 资格前验证构建。已停止并保留三个初始完成行及中断的 Q5，全部仅诊断，不进入正式表。独立 v2 修复这些监测/构建检查、完善失败收据并通过静态复核与 runnable guard check 后重新采集，没有改动推理实现或质量标准。十秒间隔监控可能漏过短暂外部进程，桌面服务仍运行且未锁频；不存在绝对排他或跨历史 JPEG 的 A/B 声明。

完整样本、加载/冷启动、阶段计时、模型/库/gates/输入身份、两份新资格及私有归档见[独立性能索引](../validation-baselines/quantized-image-performance-m4pro-20261003.json)和[双语详细性能记录](#chinese-performance-record)。旧数值/性能收据与归档不覆盖。无提交、推送或 remote CI；非本任务 tracker 计划保持原样。


## Detailed performance records / 详细性能记录

以下双语附录保留此前独立性能报告的完整协议、表格、条件与限制；详细记录现在随本计划归档。原测量数字与封存 JSON index 不变。

### English performance record

[中文性能记录](#chinese-performance-record)

Status: **complete, 18/18 cells accepted**. The serial v2 run started at
2026-10-03 23:13:48 and ended at 23:59:53 Asia/Shanghai. Values below come from
the sealed run receipts and
`docs/validation-baselines/quantized-image-performance-m4pro-20261003.json`.

The image output-quality qualification is complete for the four
`image-vision-linear-*` profiles: all 84/84 fixed-corpus cells pass across
CPU/BLAS, native CPU and Metal. Raw tensor fidelity remains separate: 0/84
cases pass its full gate, while 500/840 individual tensor-by-case comparisons
pass and 340 fail. The fixed corpus contains two original images and a crop,
six high-confidence detections and no threshold-adjacent queries; it does not
establish dataset-wide accuracy. The sealed output-quality index is
`docs/validation-baselines/quantized-image-output-m4pro-20261003.json`.

#### Scope and protocol

The serial matrix contains 18 cells: F32 and F16 baselines plus vision Q8_0,
Q6_K, Q5_K and Q4_K on CPU/BLAS, CPU without BLAS, and Metal. Every cell uses
the same original-reference `truck-truck` PPM input (1800×1200), prompt
`truck`, score threshold 0.5, and 1008×1008 model input. This PPM run is reported
separately from the historical JPEG benchmark; the two inputs are not an A/B
comparison.

Each cell uses four requested CPU threads and
`VECLIB_MAXIMUM_THREADS=4`. `sam_image --repeat 5` performs one complete warmup,
then five full-image `set_image` plus inference runs. Cache-only repeats are
recorded separately and excluded from the latency samples. The runner records
model load and cold-start times, five full-image and stage samples, and median,
minimum and maximum latency.

Peak RSS comes from `/usr/bin/time -l` around the complete child process and
therefore includes model loading, input decode, inference and final PNG output.
Resident weight-buffer and maximum compute-buffer counters are listed
separately; they are not added to RSS. The CPU BLAS workspace cannot be
measured independently, but its cost is included in process peak RSS. No
workspace estimate is inferred by subtracting buffer counters from process RSS.

The runner binds each cell to the source snapshot, input/model/sidecar hashes,
executable and backend-library hashes, and the numerical qualification
receipt. It checks backend node placement and the last computed output against
the original output-quality gate. Metal must have zero CPU/BLAS compute-node
fallback. The fresh F32/F16 CPU-without-BLAS seven-case qualifications are
separate prerequisites to the timing cells; both passed 7/7 (14/14 total).

Before, during and after each cell, the runner samples AC power and thermal
state at approximately ten-second intervals, scans for known overlapping SAM
or Meta inference processes, and checks the system sleep log. `caffeinate` is
limited to the task. Ordinary desktop services remain active, so short-lived
external processes between snapshots may be missed; monitoring does not run at
a fixed system-wide frequency. An AC-attached but non-charging battery state
is not treated as battery operation.

The first v1 monitoring attempt was stopped because its sampling coverage was
insufficient. Its three completed rows and receipts are retained for
diagnostics only and are excluded from the formal result. The v2 run is serial;
no build, conversion or other known SAM/Meta inference should overlap a timing
cell. The results below are one serial measurement batch, not interleaved A/B
timings; latency ratios close to one are not advertised as speedups.

#### Formal timing and memory results

All 18 cells passed their pre-run quality, environment, source and artifact
checks. The index retains each of the five full-image samples and per-stage
samples; the table shows load/cold-start, median/range, peak RSS and separate
resident/compute-buffer counters. Seconds use the warmed full-image timing;
RSS is decimal GB.

| Weight profile | Backend | Model load (s) | Cold start (s) | Median full image (s) | Min–max (s) | Process peak RSS (GB) | Weight buffer (GB) | Max compute buffer (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 | CPU/BLAS | 0.945 | 8.334 | 7.353 | 7.302–7.386 | 4.928 | 3.369 | 0.961 |
| F32 | CPU without BLAS | 0.986 | 40.376 | 38.857 | 38.778–39.008 | 4.833 | 3.369 | 0.961 |
| F32 | Metal | 1.099 | 6.489 | 5.373 | 5.370–5.382 | 4.245 | 3.369 | 1.243 |
| F16 | CPU/BLAS | 0.927 | 8.338 | 7.375 | 7.364–7.393 | 4.954 | 3.369 | 0.961 |
| F16 | CPU without BLAS | 0.928 | 39.878 | 39.007 | 38.795–39.100 | 4.901 | 3.369 | 0.961 |
| F16 | Metal | 0.700 | 6.063 | 5.349 | 5.345–5.361 | 2.661 | 1.796 | 1.245 |
| Vision Q8_0 | CPU/BLAS | 0.845 | 8.223 | 7.352 | 7.329–7.356 | 3.751 | 2.063 | 0.961 |
| Vision Q8_0 | CPU without BLAS | 0.860 | 39.978 | 39.234 | 38.878–39.465 | 3.543 | 2.063 | 0.961 |
| Vision Q8_0 | Metal | 0.912 | 6.245 | 5.322 | 5.316–5.333 | 2.937 | 2.063 | 1.243 |
| Vision Q6_K | CPU/BLAS | 0.865 | 8.294 | 7.348 | 7.337–7.376 | 3.678 | 1.993 | 0.961 |
| Vision Q6_K | CPU without BLAS | 0.845 | 40.268 | 38.851 | 38.799–39.079 | 3.676 | 1.993 | 0.961 |
| Vision Q6_K | Metal | 0.905 | 6.272 | 5.348 | 5.344–5.355 | 2.868 | 1.993 | 1.243 |
| Vision Q5_K | CPU/BLAS | 0.848 | 8.215 | 7.364 | 7.343–7.384 | 3.641 | 1.955 | 0.961 |
| Vision Q5_K | CPU without BLAS | 0.851 | 39.841 | 39.230 | 39.023–39.252 | 3.645 | 1.955 | 0.961 |
| Vision Q5_K | Metal | 0.905 | 6.295 | 5.370 | 5.360–5.383 | 2.829 | 1.955 | 1.243 |
| Vision Q4_K | CPU/BLAS | 0.877 | 8.260 | 7.360 | 7.348–7.376 | 3.606 | 1.919 | 0.961 |
| Vision Q4_K | CPU without BLAS | 0.900 | 40.392 | 39.180 | 38.805–39.497 | 3.413 | 1.919 | 0.961 |
| Vision Q4_K | Metal | 0.925 | 6.275 | 5.339 | 5.334–5.354 | 2.793 | 1.919 | 1.243 |

#### Ratios against same-backend baselines

Compute ratios only from completed cells, comparing each vision profile with
F32 and F16 on the same backend. A ratio of 1.000 is equal to the baseline.

| Vision profile | Backend | Latency / F32 | Latency / F16 | Peak RSS / F32 | Peak RSS / F16 |
| --- | --- | --- | --- | --- | --- |
| Q8_0 | CPU/BLAS | 1.000 | 0.997 | 0.761 | 0.757 |
| Q8_0 | CPU without BLAS | 1.010 | 1.006 | 0.733 | 0.723 |
| Q8_0 | Metal | 0.990 | 0.995 | 0.692 | 1.103 |
| Q6_K | CPU/BLAS | 0.999 | 0.996 | 0.746 | 0.742 |
| Q6_K | CPU without BLAS | 1.000 | 0.996 | 0.761 | 0.750 |
| Q6_K | Metal | 0.995 | 1.000 | 0.676 | 1.078 |
| Q5_K | CPU/BLAS | 1.002 | 0.998 | 0.739 | 0.735 |
| Q5_K | CPU without BLAS | 1.010 | 1.006 | 0.754 | 0.744 |
| Q5_K | Metal | 0.999 | 1.004 | 0.666 | 1.063 |
| Q4_K | CPU/BLAS | 1.001 | 0.998 | 0.732 | 0.728 |
| Q4_K | CPU without BLAS | 1.008 | 1.004 | 0.706 | 0.696 |
| Q4_K | Metal | 0.994 | 0.998 | 0.658 | 1.049 |

#### Interpretation and artifacts

The output-quality gate and raw tensor-fidelity report remain separate. All 12
quantized cells pass the output gate and meet the same-backend F32 process-RSS
requirement. Packed resident-weight and transient compute-buffer counters are
separate from process RSS. CPU BLAS workspace is not independently available,
but its process-memory cost is covered by RSS. Median-time ratios stay close to
one in this serial, fixed-order run; they are not a speedup claim. Native CPU
without BLAS is about 5.3× the CPU/BLAS median for the same precision, a
configuration comparison with no attribution to a particular code change.
Quantized Metal peak RSS is 2.793–2.937 GB, above the F16 Metal value of
2.661 GB; do not claim a universal memory benefit over F16. The measurements
show lower process peak RSS than same-backend F32 for these fixed PPM runs, not
a general RSS guarantee. CPU and Metal retain their distinct arithmetic
profiles described in the [quantization record](../quantization.md).

The 0/84 raw tensor-fidelity case result and 500/840 individual passing tensor
comparisons remain reported independently. The v2 matrix did not alter those
gates or the historical failure index.

The fresh F32/F16 CPU-without-BLAS prerequisites passed 14/14 cases. The
repaired F32/F16/hybrid image regression suite passed 42/42; Q8/Q4 cross-session
checks passed 6/6, with cache assertions in each seven-case `test_image` run.
The CPU/BLAS, CPU-without-BLAS and Metal Release CTest builds each passed 12/12,
and isolated tooling passed 46/46.

The runner, report, five warmed full-image samples, per-stage samples and
environment capture are under
`build/quantization-performance/20261003-v2/`. The final source/model/library
snapshot index is [quantized-image-performance-m4pro-20261003.json](../validation-baselines/quantized-image-performance-m4pro-20261003.json).
Historical JPEG measurements remain unchanged and separate; the PPM run is not
an A/B comparison with them. The v1 rows with incomplete monitoring remain
diagnostic and are excluded from these tables.

### Chinese performance record

[English performance record](#english-performance-record)

状态：**完成，18/18 单元通过资格检查**。串行 v2 测量于 2026-10-03 23:13:48 开始，23:59:53（Asia/Shanghai）结束。以下数值来自封存运行收据和 `docs/validation-baselines/quantized-image-performance-m4pro-20261003.json`。

四种 `image-vision-linear-*` profile 已通过固定语料的输出质量验收：CPU/BLAS、无 BLAS CPU、Metal 共 84/84。张量保真是独立指标：完整门槛为 0/84 case 通过，840 个逐张量×case 比较中 500 项通过、340 项失败。固定语料包含两张原图和一个 crop、六个高置信度检测、零阈值邻域 query；结果不代表数据集级准确率。输出质量索引为 `docs/validation-baselines/quantized-image-output-m4pro-20261003.json`。

#### 范围与协议

串行矩阵包含 18 个单元：F32、F16 基线，以及视觉 Q8_0、Q6_K、Q5_K、Q4_K，分别在 CPU/BLAS、无 BLAS CPU、Metal 上测量。每个单元使用相同的原始参考 `truck-truck` PPM 输入（1800×1200）、提示词 `truck`、score threshold 0.5 和 1008×1008 模型输入。此 PPM 测量与历史 JPEG 基准分开报告，两种输入不能直接作 A/B 比较。

每个单元请求四个 CPU 线程和 `VECLIB_MAXIMUM_THREADS=4`。执行 `sam_image --repeat 5`：先完成一次完整推理预热，再进行五次完整的 `set_image` 和推理。只计算完整图像运行，不将缓存命中的重复结果当成稳态样本。运行报告记录模型加载和冷启动、五个完整图像及阶段样本，以及中位数、最小值和最大值。

峰值 RSS 由 `/usr/bin/time -l` 包围整个子进程采集，因此包括模型加载、输入解码、推理和最终 PNG 输出。驻留权重缓冲和最大 compute buffer 分开列出，不能相加后称为 RSS。CPU BLAS 内部 workspace 无法单独测量，但其成本包含在进程峰值 RSS 中；不能通过缓冲区相减推算 workspace。

每个单元都绑定源码快照、输入/模型/侧车哈希、可执行文件和 backend 库哈希及数值验收收据。runner 检查 backend 节点分配，并将最后输出与原始输出质量 gate 对照。Metal 必须没有 CPU/BLAS 计算节点回退。无 BLAS 的 F32/F16 七 case 当前构建验收是正式计时的独立前置条件，14/14 均通过。

每个单元开始前、运行期间和结束后，runner 约每十秒采集一次 AC 电源与 thermal 状态，扫描已知的并行 SAM/Meta 推理进程，并检查系统睡眠日志。`caffeinate` 仅覆盖当前任务。日常桌面服务继续运行，因此两个采样时点之间的短时外部进程可能漏检；监控不按固定的系统级频率运行。即使 AC 已连接但未充电，也不解释成电池供电。

首轮 v1 监控覆盖不足，已停止。其三个已完成单元和收据仅保留为诊断，不进入正式结果表。v2 按串行顺序运行；计时单元之间不得重叠构建、转换或其他已知 SAM/Meta 推理。

#### 正式计时与内存结果

18 个单元均通过计时前的质量、电源、thermal、源码与 artifact 检查。索引保存五次完整图像及逐阶段样本；下表列出加载/冷启动、中位数与范围、峰值 RSS，以及独立的驻留/计算缓冲计数。秒数来自预热后的完整图像推理，RSS 使用十进制 GB。

| 权重 profile | Backend | 模型加载（s） | 冷启动（s） | 完整图像中位数（s） | 最小值–最大值（s） | 进程峰值 RSS（GB） | 权重缓冲（GB） | 最大 compute buffer（GB） |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 | CPU/BLAS | 0.945 | 8.334 | 7.353 | 7.302–7.386 | 4.928 | 3.369 | 0.961 |
| F32 | 无 BLAS CPU | 0.986 | 40.376 | 38.857 | 38.778–39.008 | 4.833 | 3.369 | 0.961 |
| F32 | Metal | 1.099 | 6.489 | 5.373 | 5.370–5.382 | 4.245 | 3.369 | 1.243 |
| F16 | CPU/BLAS | 0.927 | 8.338 | 7.375 | 7.364–7.393 | 4.954 | 3.369 | 0.961 |
| F16 | 无 BLAS CPU | 0.928 | 39.878 | 39.007 | 38.795–39.100 | 4.901 | 3.369 | 0.961 |
| F16 | Metal | 0.700 | 6.063 | 5.349 | 5.345–5.361 | 2.661 | 1.796 | 1.245 |
| 视觉 Q8_0 | CPU/BLAS | 0.845 | 8.223 | 7.352 | 7.329–7.356 | 3.751 | 2.063 | 0.961 |
| 视觉 Q8_0 | 无 BLAS CPU | 0.860 | 39.978 | 39.234 | 38.878–39.465 | 3.543 | 2.063 | 0.961 |
| 视觉 Q8_0 | Metal | 0.912 | 6.245 | 5.322 | 5.316–5.333 | 2.937 | 2.063 | 1.243 |
| 视觉 Q6_K | CPU/BLAS | 0.865 | 8.294 | 7.348 | 7.337–7.376 | 3.678 | 1.993 | 0.961 |
| 视觉 Q6_K | 无 BLAS CPU | 0.845 | 40.268 | 38.851 | 38.799–39.079 | 3.676 | 1.993 | 0.961 |
| 视觉 Q6_K | Metal | 0.905 | 6.272 | 5.348 | 5.344–5.355 | 2.868 | 1.993 | 1.243 |
| 视觉 Q5_K | CPU/BLAS | 0.848 | 8.215 | 7.364 | 7.343–7.384 | 3.641 | 1.955 | 0.961 |
| 视觉 Q5_K | 无 BLAS CPU | 0.851 | 39.841 | 39.230 | 39.023–39.252 | 3.645 | 1.955 | 0.961 |
| 视觉 Q5_K | Metal | 0.905 | 6.295 | 5.370 | 5.360–5.383 | 2.829 | 1.955 | 1.243 |
| 视觉 Q4_K | CPU/BLAS | 0.877 | 8.260 | 7.360 | 7.348–7.376 | 3.606 | 1.919 | 0.961 |
| 视觉 Q4_K | 无 BLAS CPU | 0.900 | 40.392 | 39.180 | 38.805–39.497 | 3.413 | 1.919 | 0.961 |
| 视觉 Q4_K | Metal | 0.925 | 6.275 | 5.339 | 5.334–5.354 | 2.793 | 1.919 | 1.243 |

#### 相对同后端基线的比例

以下比例使用同一 backend 下的 F32/F16 基线。比例 1.000 表示与基线相同。

| 视觉 profile | Backend | 延迟 / F32 | 延迟 / F16 | 峰值 RSS / F32 | 峰值 RSS / F16 |
| --- | --- | --- | --- | --- | --- |
| Q8_0 | CPU/BLAS | 1.000 | 0.997 | 0.761 | 0.757 |
| Q8_0 | 无 BLAS CPU | 1.010 | 1.006 | 0.733 | 0.723 |
| Q8_0 | Metal | 0.990 | 0.995 | 0.692 | 1.103 |
| Q6_K | CPU/BLAS | 0.999 | 0.996 | 0.746 | 0.742 |
| Q6_K | 无 BLAS CPU | 1.000 | 0.996 | 0.761 | 0.750 |
| Q6_K | Metal | 0.995 | 1.000 | 0.676 | 1.078 |
| Q5_K | CPU/BLAS | 1.002 | 0.998 | 0.739 | 0.735 |
| Q5_K | 无 BLAS CPU | 1.010 | 1.006 | 0.754 | 0.744 |
| Q5_K | Metal | 0.999 | 1.004 | 0.666 | 1.063 |
| Q4_K | CPU/BLAS | 1.001 | 0.998 | 0.732 | 0.728 |
| Q4_K | 无 BLAS CPU | 1.008 | 1.004 | 0.706 | 0.696 |
| Q4_K | Metal | 0.994 | 0.998 | 0.658 | 1.049 |

#### 解释范围与收据

输出质量 gate 和 raw tensor-fidelity 报告保持独立。四种视觉 profile 的 12 个 backend 单元均通过输出 gate；0/84 的 raw tensor-fidelity case 结果仍保留为失败报告，不改写为通过。12 个量化单元的进程峰值 RSS 均低于同 backend F32。驻留权重缓冲和临时 compute buffer 单独列出，不能与 RSS 相加；CPU BLAS workspace 无法单独测量，但总成本包含在进程 RSS 中。

中位延迟比例接近 1.0，不据此宣称加速。无 BLAS CPU 对同精度 CPU/BLAS 的中位延迟约慢 5.3 倍；这是 backend 配置比较，不归因于某项代码改动。Metal 量化进程峰值 RSS 为 2.793–2.937 GB，高于 Metal F16 的 2.661 GB，因此不宣称量化模型普遍比 F16 省内存。同 backend F32 的 RSS 下降只适用于这次固定 PPM 测量，不构成通用保证。历史 JPEG 数据和本次 PPM 测量分开；v1 监测不足的行仅作诊断，不进入正式表。

0/84 raw tensor-fidelity case 通过和 500/840 个逐张量比较通过的结果仍独立保留。v2 矩阵没有修改旧 gate 或历史失败索引。

无 BLAS CPU 的 F32/F16 计时前资格验证通过 14/14 case。修复后的 F32/F16/hybrid 图像回归通过 42/42；Q8/Q4 跨 session 检查通过 6/6，每次七 case `test_image` 都包含缓存断言。CPU/BLAS、无 BLAS CPU 和 Metal 三个 Release CTest 构建各通过 12/12，隔离工具测试通过 46/46。

runner、五次完整图像和阶段样本、报告及环境快照位于 `build/quantization-performance/20261003-v2/`。最终源码/模型/库索引为[量化图像性能索引](../validation-baselines/quantized-image-performance-m4pro-20261003.json)。历史 JPEG 测量保持不变并与本次 PPM 测量分开；二者不能作 A/B 比较。
