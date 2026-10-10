# CUDA Q8 MMQ rounded-scale candidate

Created: 2026-10-09 11:27:05, Asia/Shanghai.

## Scope

Turn the isolated fast-math MMQ division repair into a separately pinned
SAM-prepared GGML CUDA candidate. Preserve all prior frozen campaign,
model, executable, library and evidence identities. This candidate is not
qualified by the earlier final-quality or performance receipts; no gate or
threshold changes are permitted.

## Approach and steps

1. Add the single `__fdiv_rn(127.0f, amax)` change to the pinned CUDA patch
   for the MMQ Q8_1 inverse scale. Keep the MMVQ path and other backends
   unchanged. Recalculate and document the patch, touched-file and
   combined-tree hashes; verify application/reversal against the official
   pinned GGML revision. Do not edit the caller checkout or prior prepared
   tree.
2. Add a CUDA regression that executes the real GGML MMQ staging function
   on zero, ordinary, signed and finite extreme values and checks the packed
   scales/integers against an independent F64 reference. Ensure it fails
   against the old frozen library and passes against the new one. The test
   must be active in Release and require matching hardware when configured
   with `SAM_REQUIRE_CUDA_TESTS=ON`.
3. Build a fresh CUDA configuration on the separate evidence volume using
   the pinned source, changed patch and the exact current model. Run the
   selected CTests, tool tests, documentation/link check and whitespace
   check. Re-run the predeclared Q8 MMQ boundary and representative model
   staging with the new integrated library, recording source/binary/model
   identities. Compare output and arithmetic behavior before considering
   a new quality/performance campaign.
4. If the integrated candidate passes, freeze a new campaign and repeat
   required whole-recipe arithmetic, final image quality and uncontaminated
   paired performance under the original v2 gates. Reuse earlier images as
   predeclared evaluation inputs only where policy allows and preserve the
   unopened reserve. The old candidate stays `NOT_RUN` for whole-recipe
   arithmetic and deployment.
5. Since GGML's Q8 MMQ staging is shared, check the existing CUDA quantized
   image presets against their original-model regression cases before
   replacing the current pinned runtime. Their earlier quality receipts
   belong to the old library identity and cannot be silently transferred.

The complete development screen showed that the next independent evaluation
cannot be drawn entirely from the remaining eligible COCO val2017 images
while leaving the existing 1,024-image reserve unopened. Before another
final inference, obtain the official COCO train2017 annotations, select a
fresh annotation-only 1,024-image holdout with a fixed seed and no previous
image/content overlap, and fetch only its selected images. Freeze a merged
annotation file and a new dataset that retains the original
calibration/development/reserve samples and binds the new train2017 images.
Re-export and re-evaluate development recipes under that exact dataset before
freezing a final campaign. Treat this as paired precision fidelity on the
selected train2017 subset, not a model-generalization claim. The original
val2017 reserve remains unopened.

## Verification

Use `rtk proxy` commands. Keep the remote-desktop service stopped while GPU
checks run, and inspect compute occupancy before acceptance measurements.
Place new large build/evidence under Git-ignored paths on the separate SSD;
preserve prior immutable receipts. Test exact patch/source hashes, CUDA
numerics, quality, performance and public integration before any deployment
claim. Clean disposable compiler intermediates after completed cycles while
retaining the current verified build and evidence.

## Results

The pinned official `d7cb574130e6f01ad25b3289685489200febcd74`
archive had the expected original tree SHA-256
`43c54450bdc1ad5d3a5dd16d69507fa2b740d2f3283cb80f9034e9d669c79fa1`.
The updated CUDA patch applies and reverses cleanly; its SHA-256 is
`fe72eb82724131a945eefca035e644dea76e9de25030baa61bd6bd9ca2221522`.
With the unchanged Metal patch, the new prepared-tree SHA-256 is
`53160f73b48567c11dd42b66b89af7776cef6b7fb42374ec00e1cec6deadbca7`.
`cmake/prepare_ggml.cmake` and the patch README pin these identities. The
source checkout and frozen older prepared tree were not edited.

The new Release regression `sam_cuda_q8_mmq_scale` calls the real GGML CUDA
MMQ staging function on 65 x 160 F32 input with zero, signed ordinary and
finite ±3e38 blocks. It checks every stored F32 scale against an independent
F64 maximum/127 reference within four F32 ulps and checks signed endpoint
integers. Manually linked to the frozen library, it failed on a nonfinite
scale as expected. Linked to the new integrated library, it passed. The
fresh CUDA build under
`build/q8-integrated-arithmetic-20261009/many-external/q8-mmq-rounded-candidate-20261009/build`
passed **31/31 CTests** with required GPU checks, including GGML source
preparation. The preparation test now resolves its build directory before
constructing overlap fixtures, so an SSD-backed `build/` symlink cannot
produce a false failure. The isolated Python tool suite passed **170/170**,
the bilingual/local-link check passed **115** documents, and
`git diff --check` passed.

The new actual GGML CUDA library SHA-256 is
`d50d7f7c44c964d775dba8e89cb12b4a9b93e2d21b3191a4e374ec594a076228`;
the new image probe SHA-256 is
`f17ad495ec0f21223751378b93dab3baf521929b8396c5fc57b0bb6e9e2454b0`.
On the exact predeclared extreme input, the new library's MMQ payload passed
the independent verifier for 260 blocks/1,040 subblocks, with zero scale or
integer errors. Its ignored receipt is
`build/q8-integrated-arithmetic-20261009/many-external/q8-mmq-rounded-candidate-20261009/boundary-extreme-verified.json`
(SHA-256 `62b1a82dc82fa5b0d32a8b02715c14ead926da8f942dad90f6308e4c1aba22a3`).
The retained `mlp-lin1` model RHS also passed 512 blocks/2,048 subblocks;
its receipt SHA-256 is
`1460089b9319d06a6d8785317387a937bc0c17aa9af34250c4af58662021bb56`.
Both new integrated payloads match their prior isolated-trial payload hashes,
but the frozen library's failed boundary remains intact.

Seven predeclared development image-session calls completed on the new
library with CUDA-only execution. Their token IDs and prompt identities
were unchanged, but output scores, boxes and some low-ranked masks changed,
so older image-quality receipts cannot be transferred. All three
high-confidence detections across these seven calls retained their query
IDs; their matched mask IoUs were at least `0.99977045`, and their largest
same-query box-coordinate change was below `0.093` pixel. This small screen
does not establish full image quality. Its ignored comparison receipt SHA-256
is `636fb5f375bfe703d9d9575ade33daa6e126d7b455034154d416901d69a7d2eb`.

A complete 896-image/3,518-prompt calibration/development export finished
under the new binary; its manifest SHA-256 is
`2b4784cb905e8e81671f0103f231fd96c58e3944e9b5ccdcf78a035a9661338c`.
The unchanged v2 development quality evaluator found 17 bad objects among
6,447 high-confidence original objects, zero missing high objects, one extra
high object, zero protected misses among 1,910 protected objects, and no new
negative-prompt detection. Ranked mask AP drop was `0.000027594` (limit
`0.005`), positive-mask mIoU drop was `-0.000068541` (limit `0.0025`);
the paired-bootstrap upper bounds also passed. All applicable numerical
checks passed, but the formal status is **INCONCLUSIVE** solely because
896 development images are below the unchanged 1,024-image minimum. Its
ignored metrics SHA-256 is
`29ce798b87eb989ee857cec600cadb382320fce834afa7e55544cfedb5af786a`.
The F32 development baseline was recomputed under this evaluator source;
its metrics SHA-256 is
`46a8048b6af97f83c735c053ac32673a58ff5cb26d61d0aa71cdfcb6d7086986`
and is likewise numerically passing but sample-count inconclusive.

All eight existing CUDA image-vision/full-linear Q4_K, Q5_K, Q6_K and Q8_0
presets passed their seven-case ranked spatial regression on the new
library. The new custom Q8 recipe also passed the legacy seven-case original
checkpoint image validator; its ignored metrics SHA-256 is
`8172533c0ea83e9f04aeaffef4301654ed68bc17bd68662cc42160b6218c22cb`.
These fixed cases do not transfer old full-population quality decisions.

The official COCO train/val2017 annotation archive was obtained on the
external evidence volume through the project's S3 bucket (archive SHA-256
`113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`).
Its archived val2017 annotation hashes exactly match the local reference,
and train2017 has disjoint image and annotation IDs with identical category
definitions. An annotation-only selection has frozen 1,024 unused train2017
images covering all 80 categories; its ignored selection SHA-256 is
`08a35913d889d17c4540ca375cd50e45a89385c9fd723dc07bd31d334df68b21`.
The existing val2017 reserve remains unopened. Final image quality,
whole-recipe arithmetic, uncontaminated performance and deployment for this
**new** identity remain `NOT_RUN`.

The fresh download receipt completed 1,024/1,024 images with distinct content
hashes (SHA-256
`59372c60b0d96cda91138d604a1107c6feada8c1df532031523082e2255b496e`).
The merged annotation file and resulting dataset SHA-256 values are
`ee574dd08a69ba544abc03dfb719715c35bbb44d9762f3fabbf574ecd06bc54b`
and `8a4834daea2aa8786f882f1d52783d21fded3603a5d0e6d9dec5c1a7fc174eb4`.
An independent read-only check found zero prior image-ID overlap, 80/80
positive categories in the new evaluation, and byte-identical retained
calibration, development and reserve records and summaries. Its ignored
`fresh-split-audit.json` receipt has SHA-256
`d3fa3a8b8873aac6b7d3ff41aa73922e5f120c71b63590cd9a5811dd5db22219`.
The new evaluation has 4,785
prompts. Fresh normalization completed all 896 development and 1,024
evaluation images; their input manifest hashes are
`e790b89ea2861a9a5d58be84e8c151e576cff5f99ae09aa4e4df3cb9d717134f`
and `e77a4127d53dd592ddc56c6162a6649d97170ba9c2ebbf6ce5f3924072707dea`.
All 896 development normalized-image records and the Pillow version match
the prior follow-up byte-for-byte. The eight predeclared performance cases
were rebound to those identical normalized files and the new dataset; their
new ignored cases manifest SHA-256 is
`07bc7029549d052a6c42f45c7a51cc3b3760ce4d8ad5c3f41cc627190fd70e61`.
Before any inference on the fresh holdout, the exact original/F32/new-Q8
selection and new benchmark executable path were written to the ignored
`fresh-selection.json` (SHA-256
`1451cb0104bbf98b9243307ca5dceef040730a5539b7f7af2b49196667dc00d2`).
The campaign itself can only be frozen after all three complete development
exports and current-source quality reports validate.
The isolated tool suite passed **171/171**, the documentation/link check
passed **115** documents, and `git diff --check` passed after adding the
fresh-holdout constructor. No inference on the new evaluation or retained
reserve has begun.

The complete 896-image/3,518-prompt official-checkpoint development export
finished under the new dataset identity (manifest SHA-256
`0f10c781787e501a6d82efd1d70acd08157d72b0d9f8b88fde90920b48436e02`).
All 3,518 output payload hashes matched the prior original-checkpoint
development export exactly on the byte-identical normalized inputs. The
ignored parity receipt SHA-256 is
`7042db94ff3b86e997d7a018e556d40c9a9a273fdad600424bb2a8a12dd0c85c`.
The current-source Q8 development quality report still precedes campaign
freeze; the evaluation and reserve remain unopened.
After verifying that the current build uses `source-original`, the disposable
source tar and manually patched duplicate checkout were removed. The pinned
original source archive directory, prepared build, GGUF and receipts remain.

The fresh F32 native development export also completed all 896 images/3,518
prompts (manifest SHA-256
`e8e729ea987530fd9dbe2dd4e44d5384a714efba127d46fb702c6856e7a1b196`).
Read-only comparison with the preceding F32 development export found zero
non-runtime payload differences across all 3,518 prompted results; its
ignored parity receipt SHA-256 is
`a13e1da9228fa7b409e88634f313a3632ba2d138a003beceffa7a8a10dbc24db`.

The new-dataset F32 development quality report completed with zero bad,
missing or extra high-confidence objects and zero misses among 1,910
protected objects. Every applicable numerical check passed; its overall
status remains **INCONCLUSIVE** only because 896 development images fall
short of the unchanged 1,024-image final minimum. The ignored metrics
SHA-256 is
`a35e305fbf83c4268955feaf46f7f22585e1326c23bc953be1203c0e11b7afda`.

The corrected-Q8 new-dataset development export completed all 896 images and
3,518 prompts (manifest SHA-256
`a72d3b764853c7734509e5c151c8d36749452f467bb8a012389097f708d9cdcf`).
Its complete non-runtime output payloads matched the preceding export from
the same new binary on byte-identical development inputs: zero differences
across 3,518 pairs. The ignored parity receipt SHA-256 is
`1a82097f7d1cd5609481946b1534ed31db9063f124d614fafc2cf988d981ca36`.
Evaluation and reserve inference remain unopened.

That report completed with the same 17/6,447 bad-object count, zero missing
high-confidence objects, one extra high-confidence object, zero protected
misses, AP drop `0.000027594`, and mIoU drop `-0.000068541` as the prior
new-library development screen. All numerical and confidence checks passed.
The overall status is **INCONCLUSIVE** solely for 896 < 1,024 images. Its
ignored metrics SHA-256 is
`7bbe9715aa68a0c926887c51367ab46605782707fd1eed7709fc2283ae220fae`.
The precommitted three-recipe campaign passed its independent freeze/identity
check before new evaluation inference; no reserve inference has occurred.

The freeze succeeded for the original checkpoint, F32 baseline and corrected
Q8 candidate. The new campaign SHA-256 is
`6f22cccb6b6d9289b2daa06e503f0d3ce7d0ec6371700e5320e6c38f8710f56f`.
Before opening its holdout, an independent preflight rehashed all 1,024
normalized evaluation images, verified all three campaign recipes and
4,785 prompts, and observed no other GPU compute process with the remote
desktop inactive. The ignored preflight receipt SHA-256 is
`9e25c2c16e094ee714ce0de8e356bed2b5b98a182d82ad1a17f5a51f3dd0f574`.
The official-checkpoint evaluation used its first and only attempt claim;
the reserve remains unopened.

The official-checkpoint final export completed all 1,024 train2017 images
and 4,785 ranked prompts, with the frozen campaign and dataset hashes in
its manifest. Its ignored manifest SHA-256 is
`66936509a3fe76864927ffca2e379f406ef2ea8bcf367e6653b50552fe9aac1b`.

The new-library F32 final export also completed 1,024 images/4,785 prompts
with the same campaign and dataset hash (ignored manifest SHA-256
`bb5d6953ace707d89b29267cd26037437a9787eb5fb0a64eab039ce2ad1454be`).
The third and final frozen recipe, corrected Q8, is running exclusively on
the GPU. The reserve is still unopened.

The new-library F32 baseline passed the final 1,024-image/4,785-prompt
quality gate on the fresh train2017 holdout: zero bad, missing or extra
high-confidence objects among 8,928 reference objects, zero misses among
2,373 protected objects, and all AP/mIoU, negative and paired-bootstrap
checks passed. Its ignored metrics SHA-256 is
`f6712e300c93200e6ad360cdeef9abc5a41ba6d706bafbe6148553e06ca632fd`.
This qualifies the F32 quality prerequisite on this frozen campaign only;
corrected Q8 still needs its final quality verdict. The reserve remains
unopened.

The corrected-Q8 final export also completed 1,024 images/4,785 prompts
with the same frozen campaign and dataset identity. Its ignored manifest
SHA-256 is
`6e5b953db418f732963c9d1e62a414859844dcdaa52a18533b8b1b7718b955e7`.
The Q8 ranked COCO quality evaluation is running. No further holdout
inference is planned; the reserve remains unopened.

The corrected-Q8 final quality report completed with **PASS** on the fresh
1,024-image/4,785-prompt holdout. Among 8,928 high-confidence reference
objects, 39 were bad (`0.00436828`, below the `0.005` limit), none were
missing, and four extra high-confidence objects were reported (`0.00044803`,
below the `0.0025` limit). There were zero misses among 2,373 protected
objects. AP drop was `-0.0000267295` and mIoU drop was `-0.000437006`; all
applicable numerical, negative-control and paired-bootstrap checks passed.
The ignored metrics SHA-256 is
`dad24aaed3bc0f6d6d46dfc08f8e02d15b8519bc7f7e3e75a4f2d9ab7ed638a6`.
This qualifies corrected Q8 and F32 for the frozen campaign's formal
performance benchmark. It does not change the whole-recipe arithmetic or
deployment statuses, which remain **NOT_RUN**. The reserve remains unopened.

The formal eight-case, 96-process AB/BA/AB CUDA benchmark completed with
exclusive-GPU observations and both exact-campaign quality prerequisites
passing. For full-image inference and changed-prompt inference, the only
earned optimization label is **GPU memory**: sampled process peak fell from
`4,777,312,256` to `3,856,662,528` bytes (candidate/baseline
`0.80728709`, a `19.27%` reduction). Neither workload earns the latency or
host-memory label; repeated-result inference earns no optimization label.
The report retains `arithmetic_status: NOT_RUN` and
`deployment_status: NOT_RUN`. Its ignored `performance.json` SHA-256 is
`df8de89a20e8ab613c0edaa38e61237e3316d07edfddc7dc3e624de3ff60f365`.
The remote-desktop service was still inactive and no CUDA compute process
remained after the benchmark. These new labels belong to the corrected
library/campaign only, not to the earlier frozen candidate.

A read-only closure audit reran the frozen campaign identity check for all
three recipes, rehashed all three complete 1,024-image/4,785-output export
manifests and their archived artifacts, checked exactly one attempt claim per
recipe, and matched both passing quality reports and the 96-record
performance report to the campaign. It passed; the ignored
`fresh-closure-audit.json` SHA-256 is
`1b8d1ec5ebdd73cab369a2205c99a3aab8d48416a2e07bc5d9bea6e276c967c5`.
The final isolated tool suite passed **171/171**, the bilingual/local-link
check passed **115** documents, and `git diff --check` passed. The previous
**31/31** CUDA CTest result remains valid because no C++ or frozen source was
changed after that build. The retained validation evidence is on the
external SSD; no reserve inference was authorized or run.

Whole-recipe arithmetic has not been promoted. The new library passed the
exact extreme MMQ and representative real-model RHS cases, and its final
output quality passed, but the old library's eight full-graph arithmetic
captures cannot establish same-operand results for the new binary. About
153 GiB remained free on the evidence SSD after this campaign, whereas the
prior eight-input raw graph corpus occupied about 480 GiB. Preserve the old
immutable captures; a new full-graph/session audit needs a separately
capacity-planned capture strategy or additional evidence storage before
deployment qualification.
