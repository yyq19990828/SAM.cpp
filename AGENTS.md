# Repository Guidelines

## Project Structure

Header-only C++17 inference for multiple SAM variants and platforms. SAM 3 is the current model adapter; CPU and Metal are the current backends. Keep repository goals distinct from implemented and hardware-validated combinations.

- `include/sam/sam.hpp`: public entry; `types.hpp`: backend-independent values.
- `include/sam/internal/models/<family>/`: model adapters/graphs; `internal/runtime/ggml/backends/`: device drivers; `internal/runtime/ggml/`: shared execution. See [architecture](docs/architecture.md).
- `internal/io/gguf_reader.hpp`: bounded common GGUF reading; model-specific metadata and tensor contracts follow [GGUF schema](docs/gguf.md).
- `tools/`: converters; `examples/`: CLI applications.
- `tests/`: checks; `tests/data/`: redistributable fixtures.
- Ignore `models/` checkpoints and `build/` output.

## Header-Only Architecture

Inline non-template functions; avoid mutable globals. GGML remains compiled; weights stay external. The `sam::sam` INTERFACE target propagates includes/linkage. Hosts/examples handle decoding.

Allow justified abstractions for future backends/models, even with one implementation; document extension scenarios and maintenance costs.

Separate model namespaces and task contracts: SAM 2/2.1 need point/box/memory paths; GroundingSAM composes models; DART reuses detection stages. Keep SAM 3 shapes/tokenization local. Validate new models before advertising support.

`internal` denotes non-public implementation. Keep device discovery, initialization, precision/storage policies and accounting in backend modules; model code consumes those policies. Reuse GGML's registry and shared graphs. Validate CUDA or other new backends on matching hardware before claiming support.

## Planning Before Implementation

First write `docs/plans/YYYYMMDD-HHMMSS-topic.md` using Asia/Shanghai time and kebab-case topics. Include scope, approach, steps, verification, and subsequent results; scale detail appropriately.

## Documentation Audience

README and user guides explain installation, APIs, model/backend support, conversion, stable format contracts, and concise performance results. Experimental procedures, failed attempts, diagnostic tensor statistics, receipt hashes, private archive inventories, and historical measurements belong in the corresponding `docs/plans/` file. Keep immutable machine-readable evidence unchanged and link it from plans. Hardware used for a measurement is not the repository's platform boundary; one implemented adapter is not the full model roadmap.

## Build and Test

Verify the selected backend (`-DGGML_METAL=OFF` for CPU):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Document backend options and integration in `README.md`. Check whitespace with `git diff --check`.

After every completed build/validation cycle, clean up unneeded intermediate files in `build/` and `models/`, including download caches, temporary test environments, obsolete compiler intermediates and duplicate generated outputs. Preserve original checkpoints, usable GGUF models and conversion manifests, the current verified build, and required validation evidence. Keep retained evidence immutable; regenerate tensor dumps in fresh output directories. Treat routine cleanup as part of verification, without a separate cleanup plan or report.

## Coding Style

Use four spaces, `snake_case` files/functions/variables, and `PascalCase` types. Prefer stdlib/GGML and share graphs across backends. Preserve imported conventions. No formatter/linter is configured.

Use unique path-derived `SAM_CPP_..._HPP` guards, never `#pragma once`. Headers must compile independently and tolerate repeated inclusion. Preserve vendor guards.

## Testing Guidelines

Register `tests/test_*.cpp` with CTest; keep checks active in Release. Test behavior, boundaries, numerics, and regressions; avoid coverage quotas, trivial/duplicate checks, and implementation-mirroring assertions. Retain two-TU linkage. Compare official tensors/masks before quantization. Video tests cover entering objects, occlusion, and ID continuity. Benchmarks report checkpoint, precision, backend, hardware, latency, and peak memory. Run `tools/test_tools.py` in the isolated reference environment.

## Commits and Pull Requests

Use imperative commits, e.g. `feat: add text segmentation`. PRs explain behavior, link issues, list checks/untested backends, and show changed masks/videos.

## Changelog Maintenance

Maintain root `changelog.md` following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Record completed changes under `Unreleased` alongside implementation. Use `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, or `Security`; omit empty groups. Releases are newest-first with `YYYY-MM-DD` dates. Explain user impact/breaking changes; avoid commit dumps.

## Agent & Dependency Guidance

Run agent commands through `rtk` (`rtk proxy <command>` when needed). Pin dependency revisions and precision patches, retain licenses, and exclude checkpoints, tokens, and private media from Git.

Convert original checkpoints to GGUF; preserve tensor/precision/tokenizer parity and provenance. Legacy custom `.ggml` files require reconversion. Validate metadata types, bounds and model schema before allocating backend weights.

Prefer authenticated Hugging Face plugin downloads; otherwise use HF CLI credentials. Check authentication separately from license acceptance. Never request tokens in chat. Distinguish original-weight acceptance from supplementary validation.
