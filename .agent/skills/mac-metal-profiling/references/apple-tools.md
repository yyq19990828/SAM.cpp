# Apple Tool Operations

Command forms were checked against local Xcode 27.0/macOS help and man pages on 2026-10-05. Replace example paths, PIDs, boundaries, and arguments with the user-selected target. On other versions, inspect local help before assuming tool or option availability. Capture/replay commands below are not read-only environment probes.

## Instruments / xctrace

```sh
xcode-select -p
xcodebuild -version
xcrun --find xctrace
xcrun xctrace list templates
xcrun xctrace help record
xcrun xctrace help export
```

Use Time Profiler for CPU stack sampling, System Trace for scheduling/system calls/waits, and Metal System Trace for CPU/GPU concurrency, submission/completion, blits, and queue gaps. Use names from the installed template list. Combined templates such as Game Performance can help when available; they are not required for non-game workloads.

Launch a confirmed target with a bounded duration and a new output path:

```sh
xcrun xctrace record --template 'Metal System Trace' --time-limit 20s \
  --output /absolute/new-session/timeline.trace \
  --launch -- /absolute/target-app target-arguments
```

For an existing process, replace launch with `--attach PID`. Use `--all-processes` only when the question requires cross-process context. Preserve CLI stdout with the locally supported `--target-stdout` option when useful. Symbols must correspond to the sampled optimized build; retaining symbols does not require a Debug benchmark.

Export the table of contents before choosing an actual schema. Do not reuse a fixed XPath from a different Xcode trace:

```sh
xcrun xctrace export --input /absolute/timeline.trace --toc \
  --output /absolute/new-analysis/toc.xml
xcrun xctrace export --input /absolute/timeline.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="observed-schema"]' \
  --output /absolute/new-analysis/table.xml
```

Newer versions support `record --template ... --show-recording-options` to inspect configurable options without recording. Performance Limiters/utilization counters may not be collected by default. Configure them from local Recording Options when needed; absent counter data is not zero.

## gpucapture: select the process and boundary

```sh
xcrun gpucapture --help
xcrun gpucapture start --help
xcrun gpucapture list
xcrun gpucapture boundaries --pid 12345
```

The local man page requires `MTL_CAPTURE_ENABLED=1` in the target's launch environment; the process becomes connectable after creating a Metal device. A short-lived CLI can use the locally supported `MTLCAPTURE_WAIT_FOR_SIGNAL=1` to wait at device creation for a capture signal. This changes execution behavior: use it only for the diagnostic process, with an external timeout/exit strategy. Remove capture controls from formal timing runs.

```sh
xcrun gpucapture start --pid 12345 --boundary 3 --count 1 \
  --output /absolute/new-session/workload.gputrace
```

Choose a boundary from the actual `boundaries` output:

| Type | Begin/end semantics | Count means |
| --- | --- | --- |
| Device / Queue | Next command-buffer creation through its scheduling | Complete command buffers |
| Layer | Drawable present through the next present | Frames |
| Scope | `beginScope` through the matching `endScope` | Completed scopes |

A headless target without a layer may default to capturing just one command buffer; one inference often spans several buffers. To capture a complete work unit, use a suitable scope or `--until-exit` on a controlled CLI that exits promptly. Bound the latter by duration/size rather than capturing an entire long-running service. Waiting for a signal stops at initialization, so check whether the first captured work is merely resource upload.

Stop only your own capture: `xcrun gpucapture stop --pid 12345`. A `config` setter changes a running process and is not a routine discovery step. Traces can include resource contents; keep them within the authorized local output scope and do not automatically share or commit them.

## gpudebug: navigate, then load profiling data

```sh
xcrun gpudebug --help
man gpudebug
```

Use a one-shot session for a small batch. Opening the trace also starts GPU replay preparation:

```sh
xcrun gpudebug --oneshot -t /absolute/workload.gputrace --json \
  -c 'list' -c 'go commands'
```

For longer analysis, reuse one session and the actual session ID printed by the tool instead of reloading a large trace for every command:

```sh
xcrun gpudebug -t /absolute/workload.gputrace -c 'status'
xcrun gpudebug -s 42 -c 'go commands' -c 'list'
xcrun gpudebug --terminate 42
```

Use `list`, `go`, `info`, and node links to locate command buffers, compute encoders, dispatches, pipelines, and resources. Node IDs belong to the current trace. Inspect kernel entry points, pipeline constants, bindings, dimensions/strides, threadgroups, and surrounding blits/synchronization.

In the local man page, `profile load` loads profiling data already embedded in the trace. Then inspect `performance/commands`, `performance/shaders`, or `performance/timeline`. If data is missing, check whether GPU profiling was enabled during capture. Do not invent `gpucapture --profile` or interpret an empty ranking as absence of GPU bottlenecks. Source hotspots require available shader source/debug mappings, which ordinary capture may lack. Before `fetch`, confirm the resource, range, and output directory.

## metalperftrace: optional metrics summaries

When available, read local `collect --help` and `overview --help`. Current collect exports an existing recorded time window; it is not a per-compute-kernel profiler:

```sh
xcrun metalperftrace collect --last 30s --json /absolute/new-session/metrics
xcrun metalperftrace overview --json --predicate 'pid == 12345' \
  /absolute/collected-trace.atrc
```

Use the actual path returned by collect. Overview primarily reports process/layer metrics; missing layer data for headless compute does not imply an idle GPU. Bound `listen`, which streams continuously. `setup` sends control notifications to a running process and is not a read-only query. Report missing tools or permission limitations rather than automatically installing Xcode, switching the developer directory, changing signing, or disabling system protections.

## Authoritative sources

- [Metal developer tools](https://developer.apple.com/metal/tools/): tool roles and entry points.
- [Debugging with interactive command-line tools](https://developer.apple.com/documentation/xcode/debugging-with-interactive-command-line-tools): gpudebug sessions and navigation.
- [Analyzing the performance of your Metal app](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app/): timelines and counter recording options.
- Local `man gpucapture`, `man gpudebug`, `xcrun xctrace help`, and `metalperftrace` subcommand help: version-matched command arguments.
