# CUDA layout and elementwise kernel optimization

Started 2026-10-07 11:33:58 Asia/Shanghai. Status: implementation, qualification and performance verification complete.

## Scope and approach

Continue from the accepted `cuda-dataflow-optimized-v2` build, preserving its
source snapshot, binaries, measurements and acceptance evidence. Investigate
the remaining broadcast and non-contiguous copy costs first: the previous
Nsight traces attribute approximately 40 ms per image to these common kernels
across weight precisions. Specialize only where tensor layouts and bounds prove
equivalence; retain the generic paths for other inputs. Arithmetic and public
precision contracts remain unchanged.

## Steps

1. Freeze the delivered source and artifact identities. Inspect actual tensor
   shapes/strides in an isolated diagnostic library, without timing that build.
2. Build isolated kernel candidates. Measure each with interleaved baseline
   runs, inspect kernel timing, and reject changes without repeatable benefit.
3. Integrate winning changes into the pinned GGML CUDA patch, update source
   identities and add behavioral regression coverage for affected layouts,
   types, boundaries, padding and fallbacks.
4. Build fresh production binaries; run CUDA and CPU checks, reference-tool
   tests and original-weight image/video acceptance. Compare outputs with the
   delivered baseline and retain immutable machine-readable receipts.
5. Measure release latency and peak memory, update concise user-facing results
   and changelog, and record outcomes and limitations here.

## Verification

GPU work runs serially on the RTX 4090. Nsight Compute counters remain
unavailable; use Nsight Systems and release wall-clock measurements. Keep
default precision gates and the already approved fast-mode final-quality gates.
Do not infer model support from microbenchmarks. Native CPU remains a regression
and numerical reference backend, outside the routine full performance matrix.
Check documentation and whitespace.

## Results

The delivered baseline is frozen in
`build/cuda-layout-baseline-v1/snapshot.json`, including source copies, the
previous qualified audit and binary/library identities. Isolated candidates and
shape diagnostics live in `build/cuda-layout-experiments-v1/`.

- The shape-only probe identified 399 generic copies and 1,564 broadcast
  operations per full image call. The probe is excluded from timing evidence.
- `screening-v1/sequence.json` compares cheap copy indexing and F32 vectorized
  broadcast separately. Broadcast showed no repeatable release benefit across
  precisions and is not selected.
- `screening-v2/sequence.json` records an early test-harness rejection: strided
  I16 CPY is not an advertised upstream operation. Restricting CPY coverage to
  supported F32/F16/BF16/I32 types corrected the test; the backend contract was
  not broadened. No numerical mismatch occurred.
- `screening-v3/sequence.json` passed 225 copy cases on baseline, fast indexing
  and tiled candidates. Cases cover all 24 permutations, row padding and
  offsets, type conversion, reshaped destinations, batched non-square tiles,
  tails and grid-limit fallback. Same-type comparisons preserve every byte.
- Nine warm calls per release screening cell, with baseline before and after,
  give the tiled candidate the following preliminary median reductions relative
  to the mean of the surrounding baseline medians: dense F32 storage 0.64%
  default / 1.47% fast; full Q8 0.64% / 2.86%; full Q4 1.83% / 1.74%.
  These are screening results, not the public production benchmark.

`profiles-v1/candidate-comparison.json` confirms the selected copy change:
scalar plus transpose kernel time falls from 17.9480 ms to 11.4265 ms (36.3%)
in the dense fast image trace. Total instrumented GPU interval falls from
273.5634 ms to 269.6706 ms. Vectorized broadcast only reduces its own combined
kernel time from 25.4405 ms to 24.2740 ms and did not produce repeatable release
benefit, so it remains an isolated rejected experiment.

The selected `copy-tiled/cpy.cu` was integrated byte-for-byte; patch and sorted
source-tree identities are in `build/cuda-layout-experiments-v1/integration.json`.
Only `src/ggml-cuda/cpy.cu` changes relative to the previous prepared GGML tree.
Source arithmetic and model graphs remain unchanged. The old quantizer build
identity remains accepted because quantization encoding did not change.

Fresh CUDA and native CPU Release builds completed. CUDA CTest passed 25
checks (including the 225-case copy regression), CPU CTest passed 14 checks,
and the isolated reference environment passed 70 tool tests.

`build/cuda-layout-acceptance-v1/sequence.json` passed all 24 combinations:
140 image cases and 864 video frames. Default precision gates and the previously
approved fast-mode gates were unchanged. `build/cuda-layout-output-comparison-v1.json`
compares the new and previous qualified runs: all 3,912 binary outputs match
byte-for-byte, and all 1,004 result JSON records match after excluding image
runtime timing/accounting. These comparisons cover scores, boxes, object IDs,
frame ordering and intermediate tensors as well as masks.

`build/cuda-layout-delivery-v1/clean-source-audit-v1.json` confirms unchanged
model/runtime headers, examples and validation policy, unchanged previous
binaries, and exactly one changed GGML operator file (`cpy.cu`).

The production copy path simplifies source indexing for dense destinations when
the element count fits `INT32_MAX`, retaining 64-bit byte strides and the upstream
scalar cast. Eligible transposes of two contiguous dimension groups reuse the
upstream tiled transpose kernel, including dense outer batches. Tile and grid
bounds are checked; unsupported layouts and large indices retain fallback paths.
The optimization adds no global workspace and applies across storage/compute
modes without changing their arithmetic policies.

The CUDA patch SHA-256 is
`28b1260af845c338755d25214e336d697152ad01ceed5322fc6803834d9a76a9`;
the prepared GGML tree SHA-256 is
`dec61501f3b23a3b68b6b59ae92b4b0ce0a5fd8b26ad3ba3004ce7e73c3ed135`.
The pinned upstream revision and Metal patch are unchanged. Production artifacts
are in `build/cuda-layout-optimized-v1/` and
`build/cpu-layout-regression-v1/`.

### Shared-session acceptance

`build/cuda-layout-sessions-v1/sequence.json` passed all 14 checks. This includes
four long-video combinations (F32/hybrid storage, default/fast compute), each
pushing two interleaved 64-frame sequences. Ownership, model lifetime,
independent session state, bounded allocation and stability checks passed.

### Production performance and memory

`build/cuda-layout-performance-v1/sequence.json` completed all 28 cells: 20 image
storage/compute combinations and eight video weight/compute/object-count
combinations. The complete current results are published in
[BENCHMARK.md](../../BENCHMARK.md) and [BENCHMARK_zh.md](../../BENCHMARK_zh.md).
Image tables retain five warm full-image calls; video tables retain 16 warmup
frames and 48 measured frames. Hardware and workload definitions match the
preceding [dataflow optimization](20261007-094949-cuda-precision-and-dataflow-optimization.md).

The regular five-call measurements include small mixed changes: dense F32 and
mixed F16/F32 storage in default compute are respectively 0.68% and 0.26% slower
than the earlier measurements. To assess small differences with closer baseline
observations, `build/cuda-layout-experiments-v1/production-recheck-v1/sequence.json`
records four production cells, each in baseline/new/baseline order with nine warm
full-image calls per run. All 12 runs passed, and the executable and loaded
library identities were recorded. The following reductions are relative to the
mean of the surrounding baseline medians; this separate check does not replace
the five-call public table or establish a gain for every workload.

| Image storage / compute | Baseline before, ms | New, ms | Baseline after, ms | Median reduction |
| --- | ---: | ---: | ---: | ---: |
| F32 / default F32 | 502.810 | 494.462 | 503.288 | 1.707% |
| Mixed F16/F32 / default F32 | 522.105 | 518.607 | 522.131 | 0.672% |
| F32 / fast F16 | 352.082 | 340.310 | 355.190 | 3.768% |
| Full-component Q8_0 / fast F16 | 336.128 | 333.238 | 336.009 | 0.842% |

Video median changes range from 0.99% faster to 0.75% slower; this round does not
show a consistent video speedup. For example, F32 fast compute measures
576.921 ms for one object and 1028.242 ms for four; hybrid fast compute measures
581.024 ms and 1042.196 ms respectively. Preserve the measured values rather
than projecting the copy-kernel gain onto end-to-end video performance.

`build/cuda-layout-memory-v1/sequence.json` passed 20 separate image memory runs.
All 28 image/video sampled GPU peaks equal those of the previous delivered
build. Image sampling is approximately 50 ms; video sampling is approximately
one second during its timing run. These are sampled process peaks, including
context and pools, and can miss short-lived peaks. RSS is recorded separately.

### Production operator profiles

`build/cuda-layout-profiling-v1/sequence.json` passed all six Nsight Systems
traces. `build/cuda-layout-profiling-v1/operator-comparison-v1.json` compares
combined scalar-copy and transpose kernel time in the five warm image
intervals, excluding the cold call. Values are medians of per-interval kernel totals.

| Image storage / compute | Previous copy kernels, ms | New copy kernels, ms | Reduction |
| --- | ---: | ---: | ---: |
| F32 / default F32 | 18.077 | 11.380 | 37.05% |
| Full-component Q8_0 / default F32 | 17.883 | 11.468 | 35.87% |
| Full-component Q4_K / default F32 | 17.898 | 11.477 | 35.88% |
| F32 / fast F16 | 17.922 | 11.403 | 36.37% |
| Full-component Q8_0 / fast F16 | 17.948 | 11.541 | 35.69% |
| Full-component Q4_K / fast F16 | 17.971 | 11.587 | 35.53% |

All six traces preserve physical transfer counts and byte totals. These
instrumented kernel measurements show the intended local improvement; release
latency above remains the end-to-end result. The rejected vectorized broadcast
candidate remains isolated and is absent from the production patch.

### Final verification

`build/cuda-layout-pipeline-v1/sequence.json` passed all seven phases: fresh
build delivery, checks, model acceptance, shared sessions, release performance,
memory and profiling. `build/cuda-layout-delivery-audit-v1.json` passed 19,993
checks, rehashing 19,702 unique artifacts, including both old and new output
files, with no failures. It preserves quality, output equality, timings,
operator comparisons and the separate paired production timing results.

`build/cuda-layout-finalization-v1/sequence.json` passed output/profile
comparisons, the delivery audit, benchmark publication, the 58-document check
and whitespace verification. Machine-readable receipts and the previous
baseline remain unchanged. The changelog and bilingual benchmark tables now
include the completed layout-copy optimization.

Subsequent delivery request (2026-10-07): commit the completed CUDA optimization
rounds locally.
