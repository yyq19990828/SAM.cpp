# 图像特征缓存的 F16／A8 数值与传输筛选

创建：2026-10-07 17:38:58，Asia/Shanghai。状态：数值筛选完成；运行时接入另行验收。

## 范围与依据

继续[运行时量化计划](20261007-143003-activation-and-runtime-quantization.md)的缓存
独立路线。当前图像特征以 C 连续的 F32 host vector 保存，三个 FPN 层分别为
256 × 288 × 288、256 × 144 × 144、256 × 72 × 72，原始容量共 111,476,736 bytes。
这些输出之后还会上传到 geometry、fusion 和 mask 图。压缩 host 缓存、减少
传输与减少设备峰值是三个不同目标，必须分别测量。

已有 CUDA F16 模式的视觉 attention 诊断为三个完整调用共 22.151 ms，约
7.4 ms／图；该记录包含初始化，不是端到端计时。当前 neck 的活跃峰值约
488 MB，其中 F16 im2col 占 382 MB。不能假设量化视觉 attention 或 MLP 会
压低该峰值。因此先检查复用已有 GGML copy/cast 的缓存压缩成本，attention
整数内核仍按独立路线决定是否值得集成。

## 方法与步骤

1. 首先实现数值参考，保持全部原始权重和其他激活不变，只在 FPN 缓存边界
   做 F16 或 Q8_0 编解码。Q8_0 沿每像素的 C 轴每 32 个值一组，包含 F16
   scale；严格核对 CUDA copy/cast 的舍入和布局，不复用 per-token W8A8 编码。
2. 对照 CPU／CUDA 原生 GGML 的 pack/decode，覆盖零值、舍入边界、尾维拒绝、
   非连续视图、scale 溢出与解码有限性。生产路径不依赖 Python。
3. 复用原始模型 COCO selection 的最终输出与标注指标；F16、A8、分层例外
   分别记录。保持 frozen gate 和 512 张 evaluation 的隔离。
4. 只有质量合格才接入内部 typed storage、backend 能力和显式可选 runtime
   policy。直接下载紧凑格式、上传后按需解码，避免正常路径同时保留整个 F32
   host 副本。诊断导出可以按需还原，并计入额外成本。
5. 测量整图、更换提示、完全缓存重放的延迟、上传／下载字节、host retained
   bytes、graph arena 及 GPU 峰值。若只降低主机内存或传输，如实报告；有
   明显时间／显存回退则不发布该候选。

本轮图像缓存不改变视频 BF16 历史 memory、对象 pointer 或时序选择。视频
状态需独立序列数据和漂移验收，不能由静态图像通过结果代替。

## 验证与交付

- 先完成独立 codec 的行为测试和真实 GGML 算术对照；再运行完整 selection。
- 新输出目录保存源码、数据、模型、原始／候选结果及尺度布局身份，不改写旧证据。
- 若进入 runtime，完成对应 CPU／CUDA CTest、既有七个图像回归和独立最终评估，
  再进行正式性能对照；Metal 在匹配硬件前不声称获得验证。
- 更新结果、changelog 和稳定使用说明；按仓库约定清理可再生中间文件。

## 结果

### 原生编码与构建对照

新增 `cache_quantization.py`、`cache_codec_probe.cpp` 和一个仅供诊断的 CUDA
bridge。bridge 直接调用固定 GGML 的私有 Q8_0 编码函数，不在 Python 中近似
CUDA 的 fast-math 行为，也不构成生产运行时依赖或新 profile。

首次逐字节对照暴露了真实后端差异：标量参考用 `1 / (amax / 127)` 和 ties-away；
当前 x86 AVX2 编码用 `127 / amax` 和 nearest-even；CUDA 的 fast-math 又影响
少量舍入边界。F32/F16 本来就一致，不能将 Q8_0 差异归咎于布局或放松误差门槛。
保留[首次原生输出](../../build/cache-codec-validation-v1/validation.json)和当时源码，
按实际后端修正参考后，[第二次对照](../../build/cache-codec-validation-v2/validation.json)
复用了相同原生文件，24/24 组合的编码字节和解码 F32 均完全一致。覆盖零值、
舍入边界及三个真实 FPN 形状。CPU 结论限于本机 AVX2 路径。

独立 CMake 构建复用已验证 GGML 动态库，同时检查拥有私有 CUDA headers 和仅有
公共 GGML target 的两种集成方式。[构建检查](../../build/cache-cmake-validation-v1/validation.json)
17/17 通过：六组原生字节对照、四个 bridge 形状、公共 target 路径和六种输入拒绝。
它与模型筛选并行，不将其计时作为性能证据。完整工具测试 104/104 通过。

三层逻辑 payload 从 F32 的 111,476,736 bytes 降为 F16 的 55,738,368 bytes，
或 Q8_0 的 29,611,008 bytes；不包含容器、临时 F32 解码、分配器和模型权重。
独立编码/传输的 CUDA 诊断支持继续实验，CPU 编码则可能比 F32 拷贝慢，不能
因此自动启用 CPU 压缩。正式收益仍需完整模型测量。

### 完整整图筛选结果

[两张预检图](../../build/cache-output-screen-smoke-v1/screening.json)中三种模式都
通过，原始结果与上一阶段参考的 payload 完全一致。完整 128 张筛选已完成，
仍沿用相同 frozen gates、413 个提示和独立未使用的 512 张 evaluation。

第 83 张图的全层 Q8_0 在 `car` 提示上出现高置信 query 17 消失、低置信
query 153 被选中的失败；检测数量仍为六个，不能以数量或平均 AP 掩盖固定输出
门槛失败。低分辨率 FPN 是 fusion/detection 的输入，而前两层只进入 mask
细化。因此限定增加一次消融：仅量化 FPN 0/1、保留 FPN 2 为 F32。该组合应
保持检测 score/box 路径不变，须以完整 selection 验证。其逻辑 payload 为
33,509,376 bytes。不改变门槛、不根据 evaluation 调整层选择。

| 配方 | 逐对象输出通过 | Prompted mask AP | 正样本 union mIoU | 负提示检测数 |
| --- | ---: | ---: | ---: | ---: |
| 原始 F32 | 参考 | 0.4941355793 | 0.6982369095 | 0 |
| 三层 F32 缓存往返 | 128/128 | 0.4941355793 | 0.6982369095 | 0 |
| 三层 F16 缓存 | 128/128 | 0.4937798667 | 0.6982397266 | 0 |
| 三层 Q8_0 缓存 | 127/128，拒绝 | 0.4940874694 | 0.6982840428 | 0 |
| FPN 0/1 Q8_0、FPN 2 F32 | 128/128 | 0.4941355793 | 0.6982403160 | 0 |

两次完整运行的 413 个原始提示 payload 与前一阶段完全一致。F32 cache control
的三层特征逐 bit 一致，最终 413 个 payload 也完全一致。混合 Q8_0 的全部
query scores 和 boxes 在 413 个提示上逐值相同，只有 mask 细化受到新增量化。
F16 的 AP 下降为 0.03557 个百分点；两种通过的候选均满足已冻结的平均指标预算。
这里 AP 仍是预先选择的 image/category pairs、score > 0.5 的 prompted 指标，
不能冒充完整 COCO val2017 AP。

证据：[全层输出](../../build/cache-output-selection-v1/screening.json)、
[全层标注指标](../../build/cache-coco-selection-v1/metrics.json)、
[混合输出](../../build/cache-hybrid-output-selection-v1/screening.json)、
[混合标注指标](../../build/cache-hybrid-coco-selection-v1/metrics.json)。
全层 Q8_0 不进入运行时。F16 和混合 Q8_0 进入
[类型感知缓存实验](20261007-181154-typed-image-cache-runtime.md)，尚无新公共选项、
生产加速结论或最终 512 张评估通过声明。
