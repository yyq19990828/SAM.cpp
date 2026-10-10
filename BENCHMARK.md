# Performance

[中文](BENCHMARK_zh.md) · [Models and precision](MODEL_ZOO.md)

This index links to the latest complete measurements for the validated model
configurations. SAM 3 is the current adapter; additional models and platforms
will have their own tables when validated. Results apply to the stated workload
and hardware. Weight labels describe storage; computation and state follow the
model/backend precision policy.

## Browse by operating system and hardware

<a id="linux-cuda-measurements"></a>
<a id="sam-3-video-tracking"></a>

| Platform | Hardware | Measurement scope |
| --- | --- | --- |
| [macOS](benchmarks/MacOS-m4pro.md) | Apple M4 Pro | CPU/BLAS and Metal image segmentation and video tracking |
| [Linux](benchmarks/Linux-4090.md) | NVIDIA RTX 4090, Intel Core i7-12700KF | CUDA image segmentation and video tracking; CPU/BLAS image baselines |

Performance records use `benchmarks/OS-hardware.md`; Chinese translations add
the `_zh.md` suffix. Add a separate page for each new platform, recording its
environment, measurement date, workload and qualification scope.

## Measure your workload

Use a Release build and fix decoded pixels, model, backend, thread request and
power conditions. The image CLI's `--repeat` reports full-image calls and cache
hits separately. Video fixture generation, original-reference comparison and
complete-video timing commands are in [model verification](docs/validation.md).

Runtime weight/compute buffers describe allocations; process RSS describes
resident memory. Do not add them together. Stage medians cannot be added to
reconstruct the total median. Application inputs need their own quality and
performance checks.

Measurement receipts, qualification details and historical records are retained
in the [macOS performance measurement plan](docs/plans/20261004-214547-latest-complete-model-performance-records.md),
the [CUDA implementation plan](docs/plans/20261006-215722-cuda-backend.md),
the [CUDA operator profiling plan](docs/plans/20261007-015552-cuda-operator-profiling.md),
the [CUDA execution optimization plan](docs/plans/20261007-042706-cuda-execution-optimization.md),
the [CUDA precision/dataflow plan](docs/plans/20261007-094949-cuda-precision-and-dataflow-optimization.md),
the [CUDA layout copy plan](docs/plans/20261007-113358-cuda-layout-kernel-optimization.md),
and the [activation and runtime quantization plan](docs/plans/20261007-143003-activation-and-runtime-quantization.md).
