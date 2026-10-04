---
name: mac-metal-profiling
description: Profile macOS CPU and Metal workloads using Instruments/xctrace, gpucapture, gpudebug, and available Apple GPU tools. Use for CPU/GPU bottlenecks, compute-kernel cost, submission/synchronization delays, GPU trace capture, or before/after performance verification on Mac.
---

# Mac Metal Profiling

Locate CPU, Metal compute/render, and submission/synchronization bottlenecks with evidence. Keep diagnostic profiling separate from performance verification under normal execution settings. This skill supports inference, graphics, and other Metal workloads across Mac hardware and frameworks.

## Choose the tool

| Question | Preferred tool | Evidence |
| --- | --- | --- |
| CPU hotspots, thread state, CPU/GPU timelines, submission and waits | Instruments / `xctrace`: Time Profiler, System Trace, Metal System Trace | Stacks, timelines, queue gaps, synchronization relationships |
| Expensive dispatches/shaders, resource bindings, shader hotspots and counters | Xcode Metal Debugger / `gpudebug` | Selected pipelines, dispatches, resources, profiling data |
| Capture GPU work for later analysis | `gpucapture`, or the Xcode/programmatic capture path requested by the task | `.gputrace`, capture boundaries, workload coverage |
| Recorded Metal metrics and scriptable summaries | `metalperftrace`, when available locally | Metrics by process/layer within a time window; not a substitute for all compute-dispatch analysis |
| Verify latency or RSS improvement | Application timing or benchmark tools without a profiler | Complete latency, peak memory, and output correctness for fixed inputs |

Read the relevant tool section in [Apple tool operations](references/apple-tools.md). For timing shares, GPU counters, microbenchmarks, or optimization acceptance, also read [Measurement and attribution](references/measurement.md).

## Before collecting data

- Identify the user-selected application/process, input, and stage. Determine whether the question concerns end-to-end latency, throughput, one kernel, or memory. Use existing traces first; do not automatically rerun large workloads.
- Run `python3 <skill-dir>/scripts/probe_tools.py` for read-only discovery. Add `--templates` for installed templates or `--help-text` for local tool help. Local help/man pages are also suitable. Tool presence does not establish recording permissions, counter availability, or source mapping.
- Record the actual macOS/Xcode, chip/GPU, build and symbols, input/model, storage and arithmetic precision, threads, relevant environment, and time range. Recheck tool arguments and counter support on the target machine.
- For diagnosis-only requests, inspect before editing code. When instrumentation is authorized, preserve the normal build and use isolated outputs/builds or existing markers. Do not change system permissions, developer directories, signing, or installed software as incidental setup.

## Collect and analyze

Choose the smallest representative window that answers the question. Start with a broad view, then inspect the hotspot. CPU stack samples, natural GPU timelines, serialized counter profiling, and replay are different evidence; label the mode in the report.

For short-lived CLI/headless compute targets, inspect capture boundaries first. Device/Queue counts refer to command buffers, not necessarily inference iterations. Avoid capturing only model uploads. Programmatic scopes, representative repeated work, or waiting for a capture signal must remain within the authorized target scope.

Opening a trace with `gpudebug -t` creates a background replay session and prepares GPU resources. It is not merely text inspection. While another task is formally timing the GPU, do not start replay, capture, or competing benchmarks. Manage only PIDs/sessions created by this task; terminate your own sessions when finished. Avoid `stop --all` and `terminate all`.

Connect hotspots to the actual kernel/pipeline, dispatch count and shape, dtype/stride, grid/threadgroup dimensions, transfers, and dependencies. For frameworks such as GGML/MLX, map high-level operators to the actual execution backend and shader variant. Operator names alone do not establish a bottleneck or CPU fallback.

Counter support varies by hardware/API; report missing data as unavailable. Correct capture failures using the actual error and local help. If the same failure repeats, stop blind retries, preserve logs, and choose an available path rather than accumulating unknown switches.

## Report the result

Explain the leading bottleneck, supporting trace/stack/kernel evidence, measurement scope, unproven causes, and a targeted next step. After an authorized optimization, also report output correctness under the project's acceptance criteria and before/after latency and memory without a profiler.

Preserve raw traces and necessary reproduction commands, versions, and configuration in new output directories. Reuse reliable benchmark tools instead of building a large custom framework for routine analysis or recomputing complete results that can be verified. Follow the target repository's documentation convention: user guides contain clear usage and results; experimental detail belongs in the corresponding plan or analysis record.
