# Python and C++ tools

The tools are grouped by purpose. Standalone CMake builds build the executable
tools when `SAM_BUILD_TOOLS=ON` (default); CUDA-only probes follow
`SAM_BUILD_CUDA_PROBES`. The runtime output directory stays the historical
`${build}/examples/`.

| Group | Contents |
| --- | --- |
| `convert/` | Checkpoint conversion (`convert_sam3.py`), GGUF schema (`sam3_gguf.py`, `sam3_tensor_schema.json`) and shared artifact contracts (`sam3_artifacts.py`) |
| `quantize/` | Unified image configuration, shared weight policies, calibration export, runtime quantization helpers, native cache codecs and `quantize_rows.cpp` |
| `validation/` | Reference export, image/video validation, ranked output exports, COCO evaluation and probe verification |
| `benchmark/` | Image/video timing, independent quantization performance, precision inspection, linear probes, graph profiling and video fixtures |
| `visualization/` | Comparison image rendering |
| `maintenance/` | Documentation checks, archive validation, campaign freezing and shared identities |

The retained flat `tools/<name>.py` files from earlier revisions are thin compatibility
entry points. Retired v2/v3 policy entry points are excluded. Running retained scripts and importing them both keep working;
each one forwards to `tools/<group>/<name>.py`, which is the only copy of the
implementation.

Grouped commands can be executed directly, for example
`.venv-reference/bin/python tools/convert/convert_sam3.py --help`, or as modules
from the repository root with
`.venv-reference/bin/python -m tools.convert.convert_sam3 --help`.
Direct execution also works from another working directory when the Python
interpreter and script are given absolute paths; no `PYTHONPATH` setup is needed.

`tools/test_tools.py` is the compatibility entry point for the Python test
suite; the tests themselves live in `tests/tools/`. Python dependencies are
pinned in `requirements.lock` and `requirements-linux-cuda.lock` and are
installed in the isolated `.venv-reference` environment for validation runs.

Run `python3 tools/maintenance/check_docs.py` in a Git checkout to check local
links across repository Markdown files and the bilingual measurement tables.
Links to Git-ignored artifacts are rejected even when the files exist locally.
Keep their repository-relative paths as inline code in plans, with a note that
the artifacts are stored separately from the source checkout.

C++ tool sources use the private `sam_private` and `sam_image_io` targets and
are never installed. `quantize_rows.cpp` provides the pinned GGML quantization
provenance required by conversion receipts.

## Application quantization benchmarks

Use `benchmark/quantization_benchmark.py export` with a custom image/prompt cases
file, then `compare` against an original-checkpoint or native-F32 run. Reports
measure reference agreement without COCO quality prerequisites. Optional user
limits produce advice and never change a valid report's success exit code;
invalid inputs or altered/missing evidence remain errors. Completed ranked
v2/v3 exports can be compared without changing their original receipts.

See the [application benchmark guide](../docs/quantization-benchmark.md) for
commands, precision boundaries and interpretation. `performance` runs paired latency/memory processes independently;
`summarize-performance` reads existing raw records offline, and `inspect-precision`
checks stored tensors without inference. `execution-cost` runs one CPU image
with serialized node observation and writes an independent diagnostic report;
those timings are not performance benchmark samples. Native activation modes,
regex/wildcard tensor policies and mixed F16 are not implemented.

Module-level mixed F32/Q8_0/Q6_K/Q5_K/Q4_K weights use configuration schema 2 and
SAM GGUF schema 5. See the [mixed CPU example](../docs/configs/quantization/image-mixed-cpu.json)
and [configuration guide](../docs/quantization-config.md#module-mixed-weights).
Small CPU conversion/arithmetic fixtures validate this path; complete-model
quality/performance and mixed CUDA/Metal qualification remain unmeasured.

Exact eligible tensor overrides use config schema 3 and SAM GGUF schema 6;
see the [tensor CPU example](../docs/configs/quantization/image-tensor-mixed-cpu.json).
`convert/convert_sam3.py --dry-run` shares allocation with real conversion and
produces a new JSON preview without encoding payloads or initializing a backend.
Checkpoint metadata preflight and exact tokenizer sizing are optional. Native
K matrices are encoded once during real conversion. CPU diagnostic costs use
`sam_execution_cost_probe` and the independent `execution-cost` command; kernel
packing/arithmetic stay `NOT_COLLECTED`.

Opt-in CPU `compute.mode=native-quantized` keeps packed Q8/Q6/Q5/Q4 matmul
weights and records `ggml-quantized-cpu-native-v1`; the default F32 decode path
remains available. See the [native configuration](../docs/configs/quantization/image-tensor-mixed-cpu-native.json).
F32 graph activations/output differ from temporary Q8_0/Q8_K kernel RHS; static
traits evidence does not claim runtime scratch capture or independent INT8
activation control. `sam_cpu_quantized_matmul_probe` runs bounded synthetic
correctness/performance studies without a checkpoint, GPU or node observer.
Its separate single-thread packing experiment is not kernel-internal timing.
See the [study protocol](../docs/quantization-benchmark.md#bounded-cpu-matmul-study).

`quantize/quantization_config.py capabilities` lists supported four-axis choices;
`validate --config CONFIG.json` resolves them without model dependencies or GPU
initialization. Conversion, quality exports and precision inspection accept
`--quantization-config`; performance accepts `--baseline-config` and
`--candidate-config`. Conflicting flags and model/config weight disagreements
fail before inference. See the [configuration guide](../docs/quantization-config.md).

## Archived precision policies

The [v2/v3 archive](archive/precision_v2_v3/README.md) holds historical policy
JSON, evaluators and campaign-dependent performance qualification. Those
commands no longer exist under active validation/benchmark/maintenance groups
or as flat compatibility entry points. Earlier result receipts are unchanged.
The current application tools have no quality tier, quality veto or campaign
requirement; see the [benchmark guide](../docs/quantization-benchmark.md).
