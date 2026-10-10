# v3 acceptance interruption and duration investigation

## Scope and approach

Investigate the reported machine freeze during the existing v3 development
campaign. Read the previous boot's kernel journal, native/export/evaluation logs,
process identities and completed receipts. Distinguish measured campaign duration
from the immediate interruption and from the still-unknown underlying device
fault. Do not restart GPU workloads or change host driver/power settings during
this investigation.

Correct only the mutable campaign/companion status after saving its pre-change
snapshot. Preserve original checkpoint/model/data identities, frozen acceptance
sources, completed export manifests, quality metrics, partial outputs and all v2
evidence. Record a separate interruption receipt; do not classify infrastructure
failures as quantization-quality failures.

## Steps and verification

1. Correlate previous-boot PCIe/NVIDIA events with the first failed native export
   and subsequent device-discovery failures.
2. Check whether the coordinator, reporter and storage guard survived reboot;
   inspect available storage and kernel OOM events.
3. Quantify completed GPU export and CPU evaluation time from command receipts;
   verify compute mode and CUDA graph execution from native output/source.
4. Archive the observed mutable statuses and focused kernel evidence in a fresh,
   Git-ignored directory. Mark the live report as interrupted without altering
   completed numerical evidence.
5. Recheck completed metrics/manifest digests, the corrected status, documentation
   links and whitespace. Report what remains unknown and how to resume without
   rerunning completed configurations.

## Observed findings

All times below are Asia/Shanghai on 2026-10-10. The previous boot ended at
13:44:47 and the current boot began at 13:45:22. The recorded coordinator,
reporter and storage guard PIDs no longer exist; RUNNING in their files is stale.

At 13:44:03, PCIe root port `0000:00:01.0` reported a correctable physical-layer
RxErr. At 13:44:04, NVIDIA reported Xid 79 for GPU `0000:01:00`, with
`GPU has fallen off the bus`, followed by Xid 154 specifying
`Node Reboot Required`. The custom-Q8 mixed-cache native export stopped after
1,905/3,565 prompts with `CUDA error: unspecified launch failure` in
`ggml_backend_cuda_synchronize`, while encoding the next image. Subsequent eight
Q8/Q6/Q5/Q4 presets failed device discovery (`nvidia-smi` exit 6) before inference.
These are infrastructure interruptions, not observed quality-gate failures.

The inspected 11:00–13:45 kernel window contains no system OOM-killer event. The
disk guard was still WATCHING above its 50 GiB threshold at the interruption;
space at the first post-reboot check was 163.7 GiB. Neither observation proves
the absence of every possible memory/thermal problem. Driver, PCIe, power and
thermal root causes cannot be distinguished from these logs alone; no
pre-crash temperature/power time series was captured. The GPU is discoverable
after reboot, which does not establish sustained-load stability.

Measured completed GPU export times are 1,878 / 1,853 / 1,793 / 1,740 seconds
for F32, F16, custom Q8 and F32 mixed-cache respectively (29–31 minutes each).
Every recipe processes 576 images and 3,565 prompts; 13 recipes therefore require
46,345 native prompted outputs, plus the original reference. All these recipes
use F32 CUDA arithmetic even when weight storage is F16/Q8. The final F32 native
receipt records 576 vision encodes, 3,565 text encodes/inferences, 7,669,071 CUDA
nodes and zero CPU/BLAS/Metal graph nodes. Source reuses image encoding across
prompts of the same image. Mask transfer/RLE/export also contributes to elapsed
time; the export is not a latency benchmark. F32 CPU quality evaluation took
770 seconds (12:50), including 2,000 image-bootstrap repetitions, and overlaps
the next serial GPU export. Mixed-cache evaluation additionally requires its
same-weight F32-cache comparison. These explain the multi-hour campaign; no
CPU graph fallback was observed in the completed F32 receipt.

The original and first four native export manifests are complete with 3,565
outputs each. The three finished quality-metric SHA-256 values still match their
recorded digests. F32 mixed-cache has a complete export but interrupted scoring;
custom Q8 mixed-cache has only partial export. Recovery can retain the completed
exports/metrics, finish the interrupted CPU scoring and retry incomplete native
exports in new output directories. The runner should stop its queue on device
loss instead of attempting the remaining GPU jobs. Resume requires a separate
decision on local-device stability versus running the remaining cohort on a
workstation, with hardware/provenance recorded and performance comparisons kept
on the same host.

NVIDIA's [GPU triage guidance](https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html)
identifies Xid 79 as a lost GPU connection and directs incident investigation;
it does not identify this laptop's specific hardware or software cause.

## Results

Saved the prior progress/summary/guard statuses, launch receipts, full inspected
kernel window and focused PCIe/Xid messages under
`build/v3-first/interruption-20261010-142235-gpu-loss/`, with `receipt.json`
recording the observation and completed evidence hashes. Corrected the mutable
progress and companion reports to INTERRUPTED_GPU_LOST and the guard to
STOPPED_AFTER_REBOOT. No campaign processes were restarted. The completed five
export manifests (one original plus four native) and three quality metrics
retained identical SHA-256 values after the correction. The companion report
separately labels all nine interrupted/unavailable exports as infrastructure
errors; its quality-failure count remains zero for the three completed metrics.

Local evidence is Git-ignored and unavailable in a clean checkout. This
investigation changes no numerical recipe or quality threshold. No C++/Python
implementation or driver configuration was changed, so a rebuild or GPU stress
test is not part of this diagnostic/status correction.
