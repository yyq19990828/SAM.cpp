# Local original-model and Meta validation

Created: 2026-10-02 03:27:09 Asia/Shanghai.
Status: available local validation complete; full video tracker/temporal acceptance awaits the remaining M2 implementation.
Source baseline: `2ddee1b`; clean `work` branch before this record.

## Scope and approach

Execute the deferred local validation from the
[FP32 image plan](20261001-121633-fp32-cpu-metal-validation.md) and the available
foundations of the [video plan](20261001-115321-sam3-text-video-tracking.md).
Reuse the authorized local original checkpoint, accepted image GGUF files,
frozen seven-case Meta FP32 corpus, pinned upstream checkout and isolated Python
environment. Preserve existing artifacts and tolerances. No commit or push is
included.

## Steps and verification

1. Record source, artifact/binary/patch hashes, build options and hardware in
   ignored `build/fp32-validation/20261002-032709/`. Verify original checkpoint,
   image conversions, reference manifest, BPE and benchmark image identities.
2. Reconfigure/build CPU and Metal Release targets and downstream consumers;
   run CTest and isolated Python tooling, including actual Metal precision work.
3. Run all seven original-checkpoint cases sequentially for FP32/Metal,
   FP16/Metal, FP32/CPU and FP16/CPU. Check every exit status, all frozen numerical
   gates and GPU placement (positive Metal nodes, zero CPU graph nodes).
4. Run the checkpoint-backed FP32/Metal session lifetime/cache/invalidation check.
5. After other computation finishes, benchmark each accepted image configuration
   sequentially in a fresh process: one initial call and five full-image samples,
   truck input, four threads. Retain raw samples, memory accounting and conditions.
6. Reproduce pinned Meta selector goldens and validate original-weight video
   conversion inventory, precision and image-subset payload preservation using
   existing tools. Do not claim complete video acceptance: the public video
   session, lifecycle integration and official video exporter/validator are absent.
7. Publish only verified results in the two original plans, support documentation,
   benchmark table and changelog; check documentation links and whitespace.

If a model check fails, record the first divergent stage and diagnose before any
minimal repair. Missing video implementation remains an explicit acceptance gap.

## Results

Image original-weight acceptance passed **28/28** cases across FP32/FP16 and
CPU/Metal. Both Metal paths have zero CPU graph fallback; every high-confidence
mask IoU is 1.0. CPU/Metal CTest passed **10/10** each, downstream consumers
**3/3** each and isolated Python tools **13/13**, including after the video fix.
FP32/Metal also passed the real-checkpoint session lifetime/cache check.

| Precision | CPU warmed median / peak RSS | Metal warmed median / peak RSS |
| --- | --- | --- |
| FP32 | 38.986 s / 4.908 GB | 5.651 s / 4.249 GB |
| FP16 | 38.994 s / 4.927 GB | 5.576 s / 2.660 GB |

All four measurements are new, sequential fresh-process runs with five warmed
full-image samples. See the [completed image plan](20261001-121633-fp32-cpu-metal-validation.md)
for exact values, conditions and numerical metrics, and [BENCHMARK.md](../../BENCHMARK.md)
for the published table. The old generated Metal dependency tree blocked while
hashing a duplicated file; fresh unsynchronized Metal/consumer directories
resolved the environmental blocker without modifying or bypassing source checks.

Both video GGUF conversions now pass all 1,464 original tensor payload checks;
all 1,133 image-subset tensors remain byte-identical. Four full-file image smoke
checks also reproduce accepted image outputs exactly. Re-exported 2,024 Meta
selector cases, four Pillow layouts and all 256 original Meta normalization
values match the existing fixtures. Five recipes generated 216 hashed frames.
Full tracker-stage, lifecycle/ID, video-corpus and video benchmark acceptance is
still unavailable because the corresponding M2 implementation/exporter is absent.
See the [video foundations record](20261001-115321-sam3-text-video-tracking.md).

Raw receipts and run scripts are retained in the ignored directory above.
The only runtime-adjacent source change is the Python video-conversion precision
fix below, with its regression; no C++ graph/backend, original checkpoint,
accepted image weight, frozen tolerance or dependency revision changed.
No commit or push was performed.

### Observed video FP16 conversion blocker

The full original FP32 conversion succeeded. FP16 conversion reproduced a storage
mismatch for `sam_dec.pred_obj_score_head.layers.2.weight`: original `[1,256]`
selects F16 under the converter's source-rank rule, while canonical `[256]`
requires F32 in the C++ loader. A sweep of all 1,464 original tensors found only
this mismatch. Add this exact tracker tensor to the existing F32 preservation
list and extend the video conversion regression with its real original shape.
Keep malformed singleton-vector rejection, the image subset, C++ runtime and
frozen numerical gates unchanged. Run the extended test red before the fix,
then green, rerun Python tooling and the complete original video conversion.

Result: red reproduced the same original-checkpoint conversion error; green
passed, and all 13 Python checks passed. The full FP16 original conversion and
all four full-profile image checks passed after the one-entry preservation fix.
The source-rank sweep found no other mismatches. Failure and success receipts
remain separate in the run directory.
