# Exclusive-GPU performance readiness for frozen custom Q8

Created: 2026-10-09 10:05:28, Asia/Shanghai.

## Scope

Prepare the frozen eight-case CUDA benchmark for the quality-passing
custom-Q8 allocation after the bounded eight-input active-graph arithmetic
audit. Preserve the campaign, models, baseline, performance gates and
AB/BA/AB process order. Treat any other NVIDIA compute PID as
contamination. Do not terminate or disconnect the remote desktop.

## Approach and steps

1. Read-only recheck the frozen candidate and F32 baseline final-quality
   reports, campaign and performance-case identities, current models,
   benchmark binary and native recipe compatibility.
2. Inspect GPU compute occupancy. If exclusive, run the existing formal
   benchmark into a fresh ignored output directory with eight cases, three
   process pairs, separate latency and memory processes, five warmups and
   twenty iterations per process. If another process remains, do not start
   the formal run or label a contaminated screen as acceptance.
3. Verify the resulting performance receipt and compare the unchanged
   memory/latency labels. Keep arithmetic, quality, performance and
   deployment decisions separate. Record any remaining external blocker.

## Verification

Use the isolated reference environment and frozen benchmark tooling.
Recheck exact identities and GPU occupancy immediately before measurement.
Run documentation/link checks and `git diff --check` after updating plans.
Retain required evidence and clean only disposable intermediates.

## Results

Read-only preflight passed for the frozen `q8-nonvision` candidate and its
`f32` baseline. Both exact final-evaluation quality reports are `PASS`:
`build/precision-q8-followup-20261009/quality-evaluation-q8/metrics.json`
(SHA-256 `2ca27350c8aa5f59e078f095b91fe776a8b6922ce179c725da8ee0d56c04cd45`)
and `build/precision-q8-followup-20261009/quality-evaluation-f32/metrics.json`
(SHA-256 `19d9ae92bc58a502fd5932eda50827b3c0a9e245894a651e46faeb51ed6f85e8`).
The campaign and native recipes matched their current binary/model
identities. The frozen benchmark binary SHA-256 is
`34389ee1c265bcd2e21b2a9e620f435350424ad884d507668b4c833b273e1f14`.
The performance-case validator accepted exactly the eight predeclared
source-size, instance-count, object-count and negative cases.

The sole RTX 4090 still reports NVIDIA compute PID `1210592`
(`gnome-remote-desktop-daemon`, 392 MiB). The frozen benchmark refuses to
start formal measurement while another compute PID exists. No formal
performance process was launched or output directory created. The prior
two-case shared-GPU screen remains diagnostic only. An exclusive window
would run the unchanged protocol with:

```sh
rtk proxy .venv-reference/bin/python -B tools/benchmark/benchmark_precision.py \
  --campaign build/precision-q8-followup-20261009/campaign.json \
  --candidate q8-nonvision \
  --candidate-quality build/precision-q8-followup-20261009/quality-evaluation-q8/metrics.json \
  --baseline-quality build/precision-q8-followup-20261009/quality-evaluation-f32/metrics.json \
  --output build/q8-integrated-arithmetic-20261009/many-external/performance-exclusive-20261009
```

That output path is fresh and backed by the separate evidence volume. The
run requires 96 separate benchmark processes in frozen AB/BA/AB order and
no other CUDA compute process before or during the measurements. The
remote-desktop process was left untouched; performance and deployment
remain `NOT_RUN` pending an exclusive GPU window.

The readiness record passed the bilingual/local-link documentation check
(111 documents) and `git diff --check`. It created no new benchmark output
or disposable GPU intermediate.

## Exclusive measurement after user authorization

At 2026-10-09 10:34 Asia/Shanghai, the user explicitly requested stopping
the remote desktop and continuing. `systemctl --user stop
gnome-remote-desktop.service` succeeded; the service became `inactive`, and
the NVIDIA compute-process list was empty before the formal run. This later
instruction superseded this plan's earlier instruction to leave that service
untouched. The service remained inactive after the run; its enabled setting
was not changed.

The exact command above then completed successfully into its fresh output
directory. `build/q8-integrated-arithmetic-20261009/many-external/performance-exclusive-20261009/performance.json`
has SHA-256 `ad3d01eb812da5c0fc16103d283dfca07d91b5c8de945119fb5c7092dfa99b8d`.
It reports `complete: true`, `contaminated: false`, passing final-quality
prerequisites, 96 records across eight frozen cases, three independent
AB/BA/AB process pairs, separate 48 latency and 48 memory processes, five
warmups and 20 iterations per process. No record reports another CUDA
compute PID or observer error. The runner independently verified the
frozen campaign, models, benchmark binary and source identities again after
measurement. The local read-only audit rehashed all 717 bound artifacts,
matched every individual process record and its order, found 96 distinct
PIDs, and confirmed 239–282 PID-bound GPU samples per memory process. Its
receipt is `build/q8-integrated-arithmetic-20261009/performance-exclusive-audit.json`
(SHA-256 `7ada912f3afd35d9ea44fb61ba40a2795fff16a93b32b6267d35b40ff009a045`).
Both receipts and the audit source are Git-ignored and available only in
this validation workspace.

| Workload | P50 candidate/F32 | P95 candidate/F32 | Frozen performance label |
| --- | ---: | ---: | --- |
| Full image | 0.96499 | 0.96433 | GPU memory optimization |
| Changed prompt | 0.90970 | 0.91187 | GPU memory optimization |
| Repeated cached result | 1.00446 | 1.01060 | None |

The maximum PID-bound GPU memory peak was 4,777,312,256 bytes for F32 and
3,856,662,528 bytes for custom Q8, a ratio of 0.80729 (19.27% lower,
about 878 MiB saved). The host RSS peak ratio was 1.03877, within the
5% allowance for the GPU-memory label. Full-image P50 improved 3.50%, and
changed-prompt P50 improved 9.03%; neither reaches the separate 10% latency
label. The repeated-result workload exceeded the frozen per-case P95 cap
on one case and has no benefit label. The formal result is bounded to
text-prompted image inference on the measured CUDA RTX 4090 configuration.
The runner correctly leaves `arithmetic_status` and `deployment_status` at
`NOT_RUN`; performance labels do not override the separate whole-recipe
arithmetic gap or establish video/other-backend support.
