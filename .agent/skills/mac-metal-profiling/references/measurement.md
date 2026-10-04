# Measurement and Attribution

## Interpret timing correctly

| Evidence | What it establishes | What it does not directly establish |
| --- | --- | --- |
| CPU stack samples | Where sampled CPU time accumulates | End-to-end waiting or GPU time shares |
| CPU encode/submit wall time | CPU preparation and submission cost | Completion of all GPU work |
| GPU command-buffer timestamps | Execution intervals readable after completion | Internal hotspots of each kernel |
| GPU counters / shader profiling | Pipeline/dispatch costs and limiter clues in that diagnostic mode | Critical-path shares under normal concurrency |
| Complete wall time with a defined completion point | Actual latency within the stated scope | The location of the bottleneck |

Metal submission is asynchronous. Define completion as CPU-visible data readiness, command-buffer completion, or the application's existing completion mechanism. Read timing code: `cold_start` may include decode/load/setup, and `inference` may time submission only. Names do not establish boundaries. Added per-dispatch waits are diagnostic changes, not normal inference timing.

Overlapping interval durations can sum to more than wall time. GPU busy union measures time with at least one active interval; it is not the dependency graph's critical path. Do not union across processes, add stage medians, or present serialized/fusion-disabled percentages as natural concurrent shares.

Apple GPU counter-profiling documentation describes non-overlapping passes during measurement. Check the corresponding support and local configuration for Intel/AMD Mac GPUs. Distinguish natural timelines, counter/replay diagnostics, and modified execution modes; inspect recording settings when the mode is unknown. Source-line cost, occupancy, and bandwidth limiters are clues to validate against layout, dispatches, and timelines.

## Diagnose and optimize

Explain the dominant wait/operator rather than merely ranking names. CPU scheduling/encoding, GPU kernels, many small dispatches, layout/stride, copies/conversions, resource dependencies, and synchronization can all limit throughput. For compute/inference, inspect actual shapes, dtypes, threadgroups, and shader variants. Compressed weights do not necessarily reduce attention, activation computation, or data movement.

Before adding a custom shader, consider whether existing framework operations, layout changes, or batching can remove the cost. Do not require architectural changes based solely on an unverified microbenchmark. Implement only the currently authorized optimization. Record instrumentation-induced synchronization, disabled fusion/concurrency, debug layers, and counter sampling so diagnostic and production builds remain distinguishable.

## Verify improvement

- Fix inputs and work: decoded pixels/data, parameters, model/weights, precision, backend, thread request, batch, and output goal. Label cache hits, recomputation, batch throughput, and single-request latency separately.
- Use the project's correctness criteria. After code/shader changes, run relevant numerical, visual, or behavioral regressions. Quantization error affects acceptance according to the task's standard; do not relax gates during profiling.
- Compare optimized builds without added capture, replay, counter sampling, debug validation, or temporary execution switches. Keep normal fusion/concurrency settings consistent. Inspect the variables changed for this task rather than clearing the user's entire environment.
- Separate cold startup, warmed calls, and cache hits. Record power/thermal conditions and meaningful competing workloads. Do not overlap formal samples with builds, tests, model inference, or GPU replay. If conditions cannot be controlled, preserve diagnostic progress and disclose the effect instead of treating contaminated samples as verified improvement.
- Use repeated samples. Without an existing project protocol, serial A1/B/A2 with a warmup and at least five matched calls is a useful starting point. Average matching A1/A2 samples, then compare their median with B's median. Report dispersion, direction consistency, and baseline drift. If the effect is comparable to drift or samples are insufficient, gather targeted additional samples or retain uncertainty.
- Bind the actual executable, runtime libraries, and shader resources to their source/configuration when needed, using digests or build records rather than trusting paths alone. Include external `.metallib`/kernel resources. Reuse reliable record formats. When complete output exists and only surrounding bookkeeping failed, audit source/binary/input/output consistency before rerunning inference merely to fix fields.

## Memory and delivery

RSS is the entire process's resident peak and may include initialization, cold calls, and output handling. Metal/framework allocation or arena counts use a different scope and are not added to RSS. On unified-memory machines, still distinguish allocation, residency, and shared resources. A microbenchmark with A+B resident simultaneously cannot establish B's independent process peak.

Preserve raw diagnostic traces, parameters, and versions; use new paths for analysis/benchmark output. Summarize the actual bottleneck, evidence location, implementation result, and measurement boundaries. Shader heat maps and serialized counter costs are diagnostic evidence; published speed claims use unprofiled end-to-end measurements. One machine/input does not validate other chips, models, or platforms.

Sources: [Apple GPU counter statistics](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics), [Metal performance analysis](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app/).
