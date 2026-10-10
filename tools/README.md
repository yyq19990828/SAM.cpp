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
checks stored tensors without inference. Native activation and arbitrary per-layer
mixed-format settings are not implemented.

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
