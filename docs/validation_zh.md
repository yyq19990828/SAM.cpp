# 模型验证

[English](validation.md) · [模型目录](../MODEL_ZOO_zh.md)

转换模型、接入新适配器或后端，以及修改预处理和运算时，应与参考实现对照。
当前参考工具覆盖 SAM 3，后续模型需要自己的格式和任务检查；构建成功不等于新平台或模型已验证。

## 准备参考环境

按[下载与转换指南](models/sam3-details.md)取得原始权重、分词资源及 Python 环境，
将 `sam3_weights_dir` 设为权重目录。取得固定版本的官方 SAM 3 源码，将绝对路径设为
`SAM3_SOURCE_DIR`，并把 `assets/images/truck.jpg` 和 `groceries.jpg` 复制到忽略目录
`models/fixtures/`，供仓库样例使用。

参考工具在独立源码副本中运行，保留上游目录不变。源码副本、参考导出和对照结果均使用
新输出目录。固定依赖环境对应已经验证的平台，其他平台需要准备兼容环境。

## 图像对照

在仓库根目录运行：

```sh
.venv-reference/bin/python tools/prepare_reference_source.py \
  --source "$SAM3_SOURCE_DIR" --output build/reference-runtime/sam3-cpu
.venv-reference/bin/python tools/export_reference.py \
  --sam3-source "$SAM3_SOURCE_DIR" \
  --sam3-runtime-source build/reference-runtime/sam3-cpu \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --cases tests/data/sam3-image-cases.json --device cpu \
  --output models/reference/sam3-f32
.venv-reference/bin/python tools/validate_image.py \
  --build-dir build/cpu --model models/sam3-f32.gguf \
  --reference models/reference/sam3-f32 --backend cpu \
  --output build/image-validation
```

更换配置时，同时匹配模型、构建目录和后端。验证器检查分词 ID、形状、张量有限值、
掩码、分数、坐标与输入来源。SAM 3 量化 profile 以输出质量为主结果，完整张量保真
单独报告。社区权重或不完整参考可作诊断，不能替代原始模型对照。

仓库数值验收聚焦固定视觉与全模块量化预设。schema 4 自定义组合需要显式传
`--allow-custom-quantization`；即使全部样例通过，报告也保留诊断标识，不自动成为
已验证预设。验证器会核对 GGUF、manifest 和运行结果中的模块选择及每个张量的分配原因。

跨入口比较时固定解码后的像素。不同 JPEG 库可能生成不同 RGB，参考导出的 PPM 可以
避免这种差异。

## 视频对照

使用包含跟踪器的完整视频 GGUF。样例生成器和验证器覆盖运动、新对象进入、遮挡、
热启动和负提示。

```sh
.venv-reference/bin/python tools/generate_video_cases.py \
  --input-root models/fixtures --output models/video-cases
.venv-reference/bin/python tools/prepare_reference_source.py --task video \
  --source "$SAM3_SOURCE_DIR" --output build/reference-runtime/sam3-video-cpu
.venv-reference/bin/python tools/export_video_reference.py \
  --sam3-source "$SAM3_SOURCE_DIR" \
  --sam3-runtime-source build/reference-runtime/sam3-video-cpu \
  --checkpoint "$sam3_weights_dir/sam3.pt" \
  --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --frames models/video-cases --output models/reference/sam3-video
.venv-reference/bin/python tools/validate_video.py \
  --build-dir build/metal --model models/sam3-video-hybrid-v1.gguf \
  --reference models/reference/sam3-video --backend metal --threads 4 \
  --output build/video-validation
```

视频验证器检查有序输出、持续 ID、掩码、模型状态和后端执行。`--case`、`--max-frames`
子集导出，以及 `--allow-diagnostic` 对照都不构成完整视频支持。

## 开发检查

```sh
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure
.venv-reference/bin/python tools/test_tools.py
python3 tools/check_docs.py
git diff --check
```

普通 CTest 不加载权重。完整参考导出后，可配置 `SAM_REFERENCE_DIR`、
`SAM_REFERENCE_MODEL`、`SAM_REFERENCE_BACKEND` 和参考环境的 `Python3_EXECUTABLE`
来启用参考测试。

图像计时使用 `sam_image --repeat 5`，完整图像调用与缓存命中分开统计。
`tools/generate_video_benchmark.py`、`tools/qualify_video_benchmark.py` 和
`tools/benchmark_video.py` 分别负责视频样例生成、原模型资格检查和计时，执行前阅读
各脚本的 `--help`。计时与张量导出、其他模型计算分开，指标含义见[性能说明](../BENCHMARK_zh.md)。

权重、私有媒体和生成结果不进入 Git。具体实验过程、验收门槛、失败排查和批次证据
放在 `docs/plans/` 下对应的实现计划中。
