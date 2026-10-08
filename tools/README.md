# Python and C++ tools

The tools are grouped by purpose. Standalone CMake builds build the executable
tools when `SAM_BUILD_TOOLS=ON` (default); CUDA-only probes follow
`SAM_BUILD_CUDA_PROBES`. The runtime output directory stays the historical
`${build}/examples/`.

| Group | Contents |
| --- | --- |
| `convert/` | Checkpoint conversion (`convert_sam3.py`), GGUF schema (`sam3_gguf.py`, `sam3_tensor_schema.json`) and shared artifact contracts (`sam3_artifacts.py`) |
| `quantize/` | Calibration export, runtime quantization helpers, native cache codecs and `quantize_rows.cpp` |
| `validation/` | Reference export, image/video validation, precision acceptance, COCO evaluation and probe verification |
| `benchmark/` | Image/video timing, precision performance, linear probes, graph profiling and video fixtures |
| `visualization/` | Comparison image rendering |
| `maintenance/` | Documentation checks, archive validation, campaign freezing and shared identities |

The flat `tools/<name>.py` files from earlier revisions are thin compatibility
entry points. Running them as scripts and importing them both keep working;
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
