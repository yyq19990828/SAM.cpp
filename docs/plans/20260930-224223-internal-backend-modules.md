# Internal naming and GGML backend modules

Created: 2026-09-30 22:42 Asia/Shanghai.
Status: completed. Scope follows the user's backend modularization request and
naming feedback. CUDA implementation remains a separate future task.

## Scope and decisions

Rename the private `include/sam/detail/` tree and `sam::detail` namespace to
`internal`, a clearer description of non-public implementation. Keep the public
`sam::Model`, `sam::ImageSession`, value types, and `sam::sam` target unchanged.
Historical plans and saved validation artifacts retain the paths they describe.

Separate GGML resource ownership, device drivers, backend selection/policies,
and graph execution. CPU and Metal are implemented drivers; CUDA is an extension
scenario, not a new supported backend in this change. Do not add an untested CUDA
enum, empty driver, SDK dependency, or copied model graphs.

```text
Public API
    |
internal/models/sam3/        shared model graphs and weight schema
    |
internal/runtime/ggml.hpp    small internal aggregation header
    |
internal/runtime/ggml/
    resources.hpp           GGML RAII and context allocation
    backend.hpp             backend device metadata / driver contract
    backends/cpu.hpp        CPU discovery, thread configuration, weight policy
    backends/metal.hpp      Metal discovery, supported precision, runtime probe
    runtime.hpp             supported-driver selection and ordered ownership
    graph.hpp               scheduler, transfers, execution statistics
```

Backend modules own initialization, precision compatibility, FP16 storage policy
and node-statistics attribution. The selector composes supported drivers;
GGML still provides its own device registry. Use a small value-based driver
contract, not an additional virtual graph API or mutable plugin registry.
Graph execution must identify nodes through their actual scheduled backend;
an unknown backend must not be silently counted as CPU.

SAM 3 loading queries the chosen driver's FP16 storage policy rather than
branching on `Backend::Cpu`. Preserve exact FP16 promotion on CPU, packed FP16 on
Metal, precise graph operations, Auto behavior, explicit-unavailable errors,
and the current FP32/Metal rejection. Model code must not select hardware.

This touches more than eight files because internal include paths/namespaces and
guards change throughout the existing library and tests. The driver metadata
adds only initialization-time policy data and a short lookup for statistics;
there are no extra virtual calls in tensor execution.

## Steps

1. Save the pre-change sources under ignored `build/backend-modules/`, then
   rename private paths/namespaces/guards and update current documentation.
2. Extract CPU and Metal drivers and shared GGML resources; split runtime and
   graph execution. Move the weight representation decision out of SAM 3.
3. Extend the existing backend test with meaningful selection, weight-policy,
   and actual-backend attribution cases; retain independent/repeated-header and
   two-TU checks. Avoid a test for every factory/accessor.
4. Document `internal`, backend dependencies, and the actual CUDA integration
   checklist. Update AGENTS.md and changelog.md with completed changes.

## Verification

- CPU and Metal Release builds/CTest, downstream consumers, and independently
  compiled guarded headers. No changes to dependency revisions or precision patch.
- Frozen seven-case Metal regression plus complete `truck-truck` CPU FP16/FP32
  pipelines; compare tensor, token, box, score, and mask output bitwise against
  the accepted pre-refactor artifacts. Record CPU/Metal node attribution.
- Existing session ownership/cache regression on Metal if resource lifetimes
  change. Preserve supplementary weight provenance; do not claim original-weight
  or CUDA acceptance.
- Whitespace checks for source/docs, retaining the precision patch's required
  unified-diff context markers. No commit or push unless requested.

## Progress and results

- Inspected the original runtime: CPU and Metal initialization, the Metal
  precision probe, graph scheduling, and binary CPU/Metal accounting shared
  one file. SAM 3's loader also directly chose FP16 promotion by backend enum.
- Renamed all current private includes, namespaces and guards to `internal`;
  public API calls and value types remain unchanged. Historical plans/receipts
  retain their original paths.
- Delivered six runtime modules and a thin `ggml.hpp` aggregation header. The
  CPU/Metal factories provide owned handles, representation policy and a typed
  statistics-member pointer. Runtime selection keeps the previous support matrix
  and CPU fallback; graph code has no CPU/Metal-specific branches.
- The sole model logic change replaces the CPU enum check with
  `runtime.promote_f16_weights()`. A baseline comparison checked all 16 SAM 3
  headers after normalizing names/guards and that single replacement; no other
  model changes were found.

### Verification results

| Check | Result |
| --- | --- |
| CPU / Metal Release builds | Passed; 32 owned headers compile independently and repeatedly |
| CPU / Metal CTest | 8/8 each |
| Downstream CPU / Metal consumers | Build and 2/2 CTest each |
| Frozen Metal reference corpus | 7/7 with unchanged thresholds |
| CPU FP16 / FP32 | Complete `truck-truck` pipelines passed |
| Before/after bitwise regression | 9 runs, 90 tensors and 8 masks identical; tokens, selected queries, scores and boxes unchanged |
| Runtime accounting | Node counts, graph partitions, transfers and weight/compute buffer sizes match every compared baseline case |
| Metal session behavior | Ownership after model-handle destruction, isolation, prompt/image changes and cache checks passed |
| Source / whitespace | Headers and inference binaries remained unchanged throughout regression; whitespace clean, precision patch hash unchanged |

The new backend test initially assumed a host-input matrix graph would run on
the selected Metal backend. The pinned GGML scheduler places these inputs on the
last (CPU) backend. The corrected fixture also supplies an allocated Metal weight
buffer marked `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`, matching model loading. It now
checks both actual CPU fallback and GPU execution, rather than weakening node
accounting. Null and live foreign backend handles are rejected without modifying
counters. No runtime change was needed for this test correction.

Receipts are under `build/backend-modules/`: `source-manifest.json`,
`core-structural-audit.json`, `{metal-f16,cpu-f16,cpu-f32}-bitwise.json`, matching
`*-counters.json`, `after-metal-f16/metrics.json` and
`session-metal/results.json`. Metal cases each execute 3,342 Metal nodes; the
CPU cases each execute 3,332 CPU nodes. These counts describe this pinned graph,
not a requirement for future architectures.

The regression uses the existing supplementary same-weight corpus and cannot
satisfy original-checkpoint acceptance. CUDA and new model families were neither
implemented nor tested. Dependency revisions and the precision patch are unchanged.
No commit or push was made.
