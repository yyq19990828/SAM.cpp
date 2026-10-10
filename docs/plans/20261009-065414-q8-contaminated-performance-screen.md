# Nonvision Q8 contaminated-GPU performance screen

Created: 2026-10-09 06:54:14, Asia/Shanghai.

## Scope

Estimate whether the quality-passing custom CUDA Q8_0 allocation has a
plausible end-to-end latency or memory benefit over its frozen F32 baseline
while the remote-desktop process occupies the only GPU. This is a diagnostic
screen, not the frozen exclusive-GPU performance acceptance. Keep the
campaign, eight case selection, quality reports and thresholds unchanged.

## Approach and steps

1. Recheck exact campaign/binary/model identities and NVIDIA compute PIDs.
   Use two predeclared development performance cases (small input and
   negative prompt) without accessing evaluation or reserve images.
2. For F32 baseline and custom Q8, run the existing benchmark probe once per
   case in fresh processes for its three workloads and once per case for
   PID-bound load/inference memory sampling. Mark every run diagnostic and
   record all other GPU PIDs and observer errors.
3. Report per-case P50/P95 and sampled GPU/RSS ratios as directional evidence
   only. Do not generate a frozen performance label or change any gate.
   Preserve process logs and exact identities, then clean disposable files.

## Verification

Confirm both candidate and baseline final quality PASS before measurement.
Validate each probe's recipe/case identity, new output directories, finite
timings, PID-bound memory samples and model/binary hashes. Recheck local
documentation links and `git diff --check` after recording results.

## Results

The selected frozen `perf-source-small` and `perf-negative` development cases
ran through the exact campaign benchmark binary and original/F32/Q8 model
identities. Both final quality prerequisites were PASS. Each of the eight
fresh processes performed the probe's five warmups and twenty latency
iterations or an independent PID-bound memory sample. There were no observer
errors. Every process observed the additional NVIDIA compute PID `1210592`
(`gnome-remote-desktop-daemon`, 392 MiB at the preflight), so this entire
screen is **contaminated diagnostic evidence** and has no performance label.

The candidate/F32 ratios were:

| Development case | Full-image P50 / P95 | Changed-prompt P50 / P95 | Repeated-result P50 / P95 | GPU peak | RSS peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small input | 0.956 / 0.959 | 0.910 / 0.907 | 0.972 / 0.967 | 0.807 | 1.029 |
| Negative prompt | 0.960 / 0.961 | 0.915 / 0.913 | 0.995 / 0.995 | 0.807 | 1.024 |

The sampled per-process GPU peaks were 4,777,312,256 bytes for F32 and
3,856,662,528 bytes for Q8 in both cases, a 19.3% reduction. The median
full-image times were 474.2 -> 453.3 ms and 480.0 -> 461.0 ms respectively.
This makes the frozen memory-optimization class worth testing under exclusive
GPU conditions; it does **not** establish that class because only two of
eight cases and one AB pair were measured, the GPU was shared, and the full
protocol requires three AB/BA/AB process pairs with per-case constraints.
No threshold, candidate or baseline was changed.

The exact inputs, eight logs, raw timing/memory records, observer counts,
process identities and hashes are in the ignored local receipt
`build/precision-q8-followup-20261009/performance-screen/screen.json`
(SHA-256 `54ec2b0efbae6f459ebd6e272b7ea5ab4964d26c789cb936863cc5ed219b818c`).
The formal performance status remains **NOT_RUN** until an exclusive GPU
session is available.

All 39 bound input, binary, script and process artifacts rehashed correctly
after the run. Documentation/link checks passed 94 documents and
`git diff --check` found no whitespace errors. No benchmark source, frozen
gate or campaign file was changed.
