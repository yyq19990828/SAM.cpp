# CUDA short F32 dot accumulation candidate

Created: 2026-10-09 17:28:31, Asia/Shanghai.

## Scope

Create a new GGML CUDA implementation candidate for explicit-F32,
single-output, short dense dots. Preserve the prior rounded-MMQ patch and
all corrected-candidate builds/evidence unchanged. Keep the v2 arithmetic,
quality and performance gates fixed. No current result transfers to this
new candidate identity.

## Approach and steps

1. Add a separate, pinned GGML CUDA delta patch. Dispatch only F32
   single-output unbatched dots with K in `[1, 1024]` to a bounded
   F64-product/F64-reduction kernel. Leave all quantized, F16, batched,
   multi-output and longer operations on their previous paths. Reject
   unsupported layouts at the dispatch predicate rather than silently
   changing their interpretation. Preserve HIP/MUSA behavior.
2. Test the actual GGML operation through the new backend on independent
   cancellation, zero, signed and K-boundary data, including K=1025 fallback.
   Require Release-active checks and CUDA placement. Compare with an
   independent higher-precision host oracle and the unchanged frozen gate.
3. Verify patch apply/reverse, source-tree hashes and a fresh separate CUDA
   build. Recheck the previously failing captured operand using the new
   integrated library; run selected CUDA CTests, tools, documentation and
   whitespace checks. Do not mutate frozen archived receipts.
4. If focused checks pass, freeze a new recipe/campaign identity, complete
   whole-graph boundary/path arithmetic, rerun final-quality evaluation on
   a genuinely unused annotation-selected holdout, and run uncontaminated
   paired end-to-end performance before any deployment label. Preserve the
   existing unopened reserve and do not reuse a consumed final holdout as
   fresh evidence.

For this candidate's second train2017 holdout, use the original
val2017-bound `build/precision-q8-followup-20261009/development-original/dataset.json`
as the retained calibration/development/reserve source and exclude the
prior corrected-candidate train2017 final dataset by its immutable hash.
Select **1,024** annotated train2017 images using seed **20261010** before
downloading or running inference. Reject any reused image ID or content
hash, preserve the val2017 reserve unchanged, and bind both source manifests
in the new selection and dataset receipts.

## Verification

Keep the remote desktop stopped and inspect GPU occupancy for measurements.
Use `rtk proxy` for commands. Place the new build/evidence under a fresh
Git-ignored SSD directory; keep original models, conversion manifests,
current verified build and old receipts. Clean disposable compiler and
download intermediates after each finished cycle.

## Results

The new additive patch `cmake/patches/ggml-short-dot-cuda.patch` has SHA-256
`45401f8e17327557364a95d90d78fb0c2e103e4077c01e38e4ddc4fc17c80707`.
It applies after the unchanged rounded-MMQ CUDA patch, and the prepared GGML
tree has the CMake-native SHA-256
`4394cdc89f35b65c1642f45be122d6301e397480ff0d1f9a519237703dfe1d1f`.
The separate source-preparation regression passed for an original checkout,
the prior Metal-only and Metal/CUDA stages, and an already fully patched
archive. The old patch and frozen build were not modified.

The fresh RTX 4090 Release build is under
`/mnt/SSD1/samcpp-validation/q8-mmq-shortdot-candidate-20261009/build`.
Its CUDA library SHA-256 is
`dc0ebfc0cbf0055b78a8111b02e2ab7ba245002f54602347fe8514baa339f6de`;
the image-probe binary SHA-256 is
`2a0a66b1e6ae0ce04a415731bffd20a2c32e2762ed270e8887b332abdf8a4d94`.
The model GGUF stayed
`f854afae5a2eb2c388ff2dd85804a4327b4525b45a09c1fdc9b5f5619c93142e`.
All **31/31 CTests** passed after the final rebuild with required CUDA,
including the new Release-active K=1, 32, 256, 257, 1,024 and 1,025
single-dot checks, zero output, existing batched numerics, Q8 MMQ scale and
source preparation. A subsequent offset-view K=256 case passed the focused
CUDA numerics CTest after rebuilding that test target. The K=1,025 check
exercises the unchanged cuBLAS side
of the boundary. The isolated Python suite passed **171/171**.

The candidate completed the predeclared seven-call development image
session on CUDA with no CPU/Metal/BLAS compute. Token IDs, scores, boxes,
ranked queries and masks matched the preceding rounded-MMQ candidate
exactly in all seven calls. This is a small diagnostic parity screen, not
the new candidate's formal final-quality result. Its ignored comparison
receipt `q8-mmq-shortdot-candidate-20261009/session-comparison.json` has
SHA-256 `08b5f373acbe0aa6baaf9c7460b645694f0f0d8903b7393d6d1f3bd9ed889b0f`.

A fresh selective graph-7 observer recaptured the previously failing F32
ordinal 49. Both 256-element operand files are byte-identical to the old
capture. The old cuBLAS value was `-0.0043205022811889648` (relative error
`2.3473892516524385e-5`, **FAIL**); the integrated new GGML CUDA value is
`-0.004320603795349598` (relative error `2.147239569049425e-8`, **PASS**).
The new value is correctly rounded to F32 from the independent exact-product
sum, and all seven observed exports match their unobserved new-library
exports exactly. The immutable singleton receipt
`q8-mmq-shortdot-candidate-20261009/singleton-capture/singleton-audit.json`
has SHA-256 `4d538a16094317671f42b4ee3c5f20abc33de041e5707a2acd9f471754022dcd`.

This does not transfer the prior 1,024-image quality or performance result to
the new library. New whole-graph/path arithmetic, a genuinely unused final
holdout, and uncontaminated paired end-to-end performance remain **NOT_RUN**.
The old corrected candidate's graph failure and all earlier receipts stay
unchanged; neither candidate has a deployment label.

The second train2017 holdout is now frozen under the new candidate's
`fresh-coco-source/` directory. Its 1,024 annotation-selected evaluation
images cover all 80 COCO categories; their image IDs and content hashes have
zero overlap with the previous corrected-candidate final images or retained
val2017 samples. The 1,024-image val2017 reserve is byte-identical in the
new dataset. Selection SHA-256 is
`804da49611754bd17490dc3fb904c2a1432792dcc4c5947f279d397ec60d9f2f`,
download receipt SHA-256 is
`2c3b84485b7e76bdc0018aade6278b71d1d712aad6cf0c019c3dc1412030ee61`,
merged annotation SHA-256 is
`37cd79d9c6aae8c778935ff2363f9fdf6c2d51c0d6d1738cecfdaf94b1396eda`,
and dataset SHA-256 is
`4ebcf322c8f8877849ee6f27382151d0ed59b58d58e0664c22835f97dad46883`.
The prior consumed dataset is bound as an explicit exclusion input, and the
selection tool rejects mismatched provenance. The isolated Python suite
passes **171/171** after this addition. Development re-export, new final
quality, whole-graph arithmetic and paired endpoint performance have not
yet run on this dataset.

Routine post-verification cleanup removed 82.984 GiB of the superseded
candidate's **passing** raw Q8/F32/nonmatrix component dumps, after checking
their immutable verification and reconciliation receipts. Their nonraw
identities stayed unchanged; the failed graph-7 operands, current build,
GGUF models and all compact receipts remain. Earlier plans' statements that
those raw dumps were retained describe their state at the time of each
audit; reproducing them now requires a fresh output directory.

Under the new dataset identity, all **896** calibration/development images
were normalized again, and all **3,518** prompted outputs were re-exported
for the official F32 checkpoint, the new native F32 library and the new
native Q8 library. The normalized development PNG/source identities match
the preceding holdout's retained development split exactly. The official
output files are byte-identical to the prior development reference; the
F32 and Q8 native outputs are each exactly equal to their respective prior
development outputs after excluding runtime timings. Independent parity
receipts are `fresh-f32-development-parity.json` (SHA-256
`fe99e69ad1e24f77b4135595710e973516959dc207098df437f57542d1b0b612`)
and `fresh-q8-development-parity.json` (SHA-256
`a1cfd9bec0c2963e9eb2d6f6f5cd317b568b9656f05ea645f39a1b930398c245`).
Both new 896-image development quality reports are **INCONCLUSIVE**, not
FAIL or final PASS, under the unchanged v2 policy; their `metrics.json`
SHA-256 values are `af4eff709e9642ce9d0fd54e9607f6cbbe6be8f56dbdbc6f15024eb770d466f2`
for F32 and `84f3957bbe9e917ba4d8c780dc87048d122ce142698e9be7ffe54fad0cf9960b`
for Q8.

The eight development-derived performance cases were frozen before any
final inference (`fresh-performance-cases.json`, SHA-256
`1437aeb1d8aa92f8ad74bb9585ea8bd546c2deb5a8222e0bac388384a5fea8b4`).
The three-recipe final campaign `fresh-campaign.json` has SHA-256
`742d4ed8d7b086b049f0906bfe842141c137e0eb71fd5cfdfcad2b877da122b5`.
It binds the new dataset, original v2 gates, complete development outputs,
quality reports, exact models/binaries, source snapshot and performance
cases, and prohibits reserve inference. At freeze time formal evaluation had
not completed, so no final quality or deployment status was inferred from
the development results.

On the unused 1,024-image evaluation split, the official checkpoint export
completed all 4,753 prompts (`evaluation-original-fresh/manifest.json`,
SHA-256 `39c667b3150771c355ede20e46fbcc180f505e9bba75b01226cf6cfdc034309d`).
The new-library native F32 export also completed all 4,753 prompts with
campaign and dataset identities matching the freeze
(`evaluation-f32-fresh/manifest.json`, SHA-256
`407a41e810d09cab166719842f44f0ce1be59a1e54f4331fc31b96feaa388f12`).
Its independent final quality report is **PASS** for the absolute profile;
the cache increment is `NOT_APPLICABLE` for F32
(`quality-evaluation-f32-fresh/metrics.json`, SHA-256
`47bbc79a3d29149cc9770e2d090010f755808c0bf885ffe9df4fde01662d1cf7`).
The Q8 export likewise completed all 4,753 prompts on the same split and
bound the new model and probe identities (`evaluation-q8-fresh/manifest.json`,
SHA-256 `9a10da9b2bb609e8292a4940c7596ce228f517cc5bf88ecb4512fad987eadab9`).
Its independently evaluated final absolute quality is **PASS** on the same
1,024 images and 4,753 prompts; the cache increment is `NOT_APPLICABLE` for
this F32-cache recipe (`quality-evaluation-q8-fresh/metrics.json`, SHA-256
`9758ed3be0c367a01e5a25f7546d3a21b1aa2e71e33f203e8d2b35c785d766d3`).
The unchanged gate reported 92 detailed checks. Both F32 and Q8 final
quality prerequisites now pass under the new campaign. Paired performance
and whole-recipe arithmetic remain unresolved; quality alone does not
grant a deployment status.

The new CUDA library directly replayed the frozen five-case Q8 staging
boundary matrix from the previous corrected candidate. All four valid
MMVQ/MMQ fixtures passed the unchanged independent verifier, including
16,384-wide, signed endpoint, zero and tail cases. The deliberately
overflowing F16-sum case produced the expected verifier rejection. Every
new native packed payload was byte-identical to its old counterpart. The
candidate-bound ignored receipt `q8-boundary-replay/boundary-replay.json`
has SHA-256 `4abf9cfdb7b5884b9d6f6fb2d1dc9909b047189a8e4513a7b3b1df946d4bbc62`.
This is Q8 component evidence, not an active-graph or whole-recipe PASS.

The new-library GGML linear probe also replayed the two predeclared
diagnostic Q8 MLP inputs. Fresh MMVQ and MMQ RHS staging and independent
packed-operand verifiers passed **317,312** same-operand dots. An independent
F32 bias/precise-GELU oracle passed the same 317,312 output elements. The
new and prior corrected libraries produced byte-identical Q8 weights,
F32 right operands, staged RHS, raw dots and final outputs for both inputs.
The ignored `q8-epilogue-replay/epilogue-reconciled.json` receipt has SHA-256
`82bbe55bb1384bd7c61bd99b7ee64cbec9574ab1454e129a2768441558598966`.
These diagnostic linear graphs use their own compute setting; their PASS
does not transfer to the full text-image inference graph.

An instrumented replay of the predeclared seven-call development image
session captured **16** active graphs and **23,988** nodes with the new
library. The complete node inventory was byte-identical to the prior
corrected-library inventory; all seven observed exports matched the new
library's unobserved exports exactly outside variable timings/RSS. CUDA
node counts and zero CPU/Metal/BLAS compute also matched. The ignored
`session-graph-coverage/topology-audit.json` receipt has SHA-256
`ded6445f401cc4ed893c4fef1ad5e560dfddbb495e2340515839043be2a9f12a`.
This establishes graph topology and observer parity for these calls, not
independent numerical correctness of each graph node.

The formerly failing whitespace-prompt graph was recaptured under the new
library with the reviewed filtered observer. An independent same-operand
oracle checked **54/54** active F32 matrix nodes and **159,232,713** dots
against the unchanged numerical gate; all passed, including the earlier
graph-7 ordinal-49 failure. All seven observed session outputs matched the
new unobserved session exactly outside variable timings/RSS, with no
CPU/Metal/BLAS compute. The ignored
`session-whitespace-f32/verification.json` receipt has SHA-256
`d15b7e4189ae71327def4c48535b445ddd27f4699d5bfad810d1e71c13088cc6`.
This covers that path's F32 matrix component. Q8, other compute operations,
metadata and additional input paths still need candidate-bound whole-graph
reconciliation.

The frozen eight-case, three-pair AB/BA/AB CUDA performance run completed
all **96** independent processes with no competing compute PID or sampler
error. Its ignored `performance-q8-fresh/performance.json` receipt has
SHA-256 `551a9be2d1ce179592a2432a48c3f339c58f5c853209196bdb2feefe784180dc`.
For full-image inference, the candidate/baseline p50 and p95 ratios were
`0.96675` and `0.96842`; changed-prompt ratios were `0.92566` and
`0.91853`; repeated-result ratios were `1.00937` and `1.01285`.
The GPU process peak ratio was `0.80729` (19.27% lower), while process
RSS peak ratio was `1.04175` (4.18% higher). The unchanged latency label
requires p50 ratio at most `0.90`, so **all three latency labels FAIL**.
The full-image and changed-prompt paths receive only the `gpu-memory`
label; the repeated-result path has no performance label. The `gpu-memory`
label is a valid workload-scoped performance result, while no latency gain
may be advertised under the frozen 10% p50 gate. Final F32/Q8 quality passes,
but whole-recipe arithmetic remains `NOT_RUN`, so the public/deployment
decision stays pending. Completing arithmetic could support a memory-only
claim for the two labeled workloads; it would not grant a latency label.
The frozen gate is not relaxed after observing this result.

The new library's complete seven-call development session now has direct
active-Q8 arithmetic evidence. All **1,838/1,838** captured Q8 matrices
passed independent GGUF-byte, native RHS-staging and same-operand F64-dot
checks: **936,772,736** dots, **15,173,800** staging blocks and worst
relative L2 `1.334296991e-7`. All seven instrumented exports match the
uninstrumented new-library outputs outside variable timings/RSS. The full
per-node report `session-q8-arithmetic/verification.json` has SHA-256
`9ad9337f1d43a442217b2def5a5ad08fce1247d0dc8ddb57669b192b9fa297d8`.
A separate audit rehashed all **9,190** raw payloads and **33** identity
files, checked exact capture-file coverage and retained immutable evidence;
`session-q8-arithmetic/reconciled.json` has SHA-256
`315fffc7587c21a58f9f4c4806e8f244e0e39f3b3d3771ed23c7fba2f9dc7554`.
The first read-only rehash script incorrectly rejected the normal
`libggml-cuda.so.0` identity symlink; it was corrected to hash the resolved
file content, and the complete audit then passed without changing raw data.
This is a session-path Q8 component PASS, while full-graph non-Q8 operators
and whole-recipe arithmetic remain `NOT_RUN`.

The same new-library seven-call session also yielded a direct allocated
metadata audit. All **16** graphs and **23,988** active nodes matched the
candidate-bound inventory. The audit checked buffer bounds, leaf/source
identity and exact storage aliases for all **9,466** `RESHAPE`, `TRANSPOSE`,
`PERMUTE` and `VIEW` nodes, including Q8 views; **4,922** leaves and
**14,522** compute nodes were accounted for. The instrumented seven outputs
again matched the uninstrumented new-library outputs outside variable
timings/RSS, with zero CPU/Metal/BLAS compute. The ignored
`session-metadata/audit.json` receipt has SHA-256
`28c558cd64caa18bf21c45a0ab6b9cfae647d3075dd2fede2ea6ee9c5204ca26`.
This is a direct metadata/alias PASS; arithmetic of the non-Q8 compute
nodes and the whole-recipe gate remain `NOT_RUN`.

Routine cleanup then rehashed and removed 6,208 superseded raw nonmatrix
tensor files from the preceding corrected candidate's changed-prompt and
whitespace-prompt sessions (17.722 GiB of allocated blocks). The unchanged
per-group PASS receipts, observer sources, compiled sidecars and compact
outputs remain. It also rehashed the prior Q8 seven-call session's 9,190
recorded artifacts and removed only its 7,352 raw binary tensor files
(11.905 GiB); the JSON staging probes and original verification receipts
remain. The current short-dot candidate's complete raw Q8 evidence was not
pruned. These old captures require fresh output directories to reproduce.

The new library's seven-call session subsequently captured every active F32
matrix node, with **600/600 PASS** against the same-operand F64 oracle and
**4,432,408,758** checked dots. The worst relative L2 was
`3.4897028782238643e-6`, below the frozen `2e-5` limit, and no node failed.
The callback changed only query scores by at most the observer-only tolerance;
other exported fields and CUDA placement matched the unobserved new-library
session. The ignored `session-f32-arithmetic/verification.json` receipt has
SHA-256 `e95cdfd90f6357bf168c9dec4ac9abec9e676a283d735494e906a40ab309add7`.
This closes the F32 matrix component for these session calls. The other
nonmatrix compute groups and whole-recipe gate still require reconciliation.

Direct seven-call captures under the same new binary/library also passed
**370/370** attention nodes and **1,454/1,454** normalization/unary nodes
against their existing independent same-operand references. The attention
observer's seven nonruntime outputs were exact; the normalization observer
perturbed query scores by at most `1.1600000001353583e-9` and no other
exported field. The ignored `session-nonmatrix-arithmetic/attention/verification.json`
and `session-nonmatrix-arithmetic/norm-unary/verification.json` receipts have
SHA-256 `3ee541ffe3e5d70f9d8520b40ff9091c06608b3612a54a39bd8b38b1cbfa86ec`
and `672cc420ed39e21560088f620520f62ac246f856bcf2656f731b27d211606d9b`.
Remaining operation families still prevent a whole-graph PASS.

A fresh read-only session boundary audit bound the same seven calls to the
new campaign/model/binary/library, exact 320×240 and 640×640 development
PNG bytes, prior predeclared prompt text, 32-token outputs and finite
scores/boxes. It confirmed one vision encode per distinct image, no new
compute for a repeated prompt, 1,900 CUDA nodes per changed prompt, and
empty/full tokenizer contexts for the whitespace/long cases. Every new
nonruntime output matched the old predeclared session exactly. The ignored
`session-boundary-audit.json` receipt has SHA-256
`661d653f66f458bc2fd4db9f73a7f78dcc2cb74f7bdb17dbae46d0f8a0051c2b`.
The output/cache/path contract passes for these inputs; other accepted
source geometries still rely on code tests and final-quality coverage.

The new-library complete-session spatial/reduction and pointwise/repeat
observers directly passed **204/204** and **1,412/1,412** active nodes.
Their ignored verification receipts are
`session-nonmatrix-arithmetic/spatial/verification.json` (SHA-256
`b354a7064a3b372db4dc135c7006fe873b4653cebf9cade311117518b5c97996`)
and `session-nonmatrix-arithmetic/pointwise/verification.json` (SHA-256
`bb27bedcc06d00fe80f713694b5e893e4427d9f5af888f8567c43f1dbc863760`).

The remaining seven-call compute groups also passed independently:
**5,930/5,930** elementwise nodes
(`session-nonmatrix-arithmetic/elementwise/verification.json`, SHA-256
`3ad146dd18fbc4c446d816440e9b14613a70f567408390bdf4ab37a05966223c`),
**536/536** concatenations
(`session-nonmatrix-arithmetic/concat/verification.json`, SHA-256
`811ad440454f413ed587bb4d85661fea594aa0a81cd623f7ac5de2bc89e14e85`),
**90/90** row-gather/pool/F16-copy nodes
(`session-nonmatrix-arithmetic/gather-pool-copy/verification.json`, SHA-256
`ad2126dbf6377fb2192f9e6d488643013dfc5929ce62b6e814ae0eb3c96de53c`),
and **2,088/2,088** CONT nodes with zero source/output byte mismatches
(`session-nonmatrix-arithmetic/cont/verification.json`, SHA-256
`9cec58eb61cc35f3a8365c171b3bedd95e5606e2a6dc78b76ddc39795402ef24`).
The seven-call total is awaiting independent group and raw-file coverage
reconciliation; whole-recipe arithmetic remains `NOT_RUN`.

An independent read-only reconciliation now covers the complete seven-call
active graph. It matched all **16** graphs and **23,988** nodes against the
candidate-bound topology: **9,466** metadata aliases and **14,522** compute
nodes, each compute node belonging to exactly one of the ten passing
arithmetic groups above. It also accounted for **4,922** leaves and rehashed
**34,278** raw payloads plus **296** identity files before any cleanup. The
ignored `session-compute-reconciled.json` receipt has SHA-256
`c9ff134d11b43e1c6cea79c03dfd511f64243d3701e77b6503c88b9df277114f`.
This establishes `active_graph_compute_status: PASS` for the predeclared
session. The receipt intentionally retains `whole_recipe_arithmetic_status:
NOT_RUN` pending the separate requirement-by-requirement policy decision;
the immutable component reports are not edited to imply a broader verdict.

The subsequent frozen-gate adjudication maps each previously identified
whole-recipe prerequisite to evidence from this exact candidate:

| Prerequisite | Candidate-bound evidence and decision |
| --- | --- |
| Input, tokenizer, output and cache paths | `session-boundary-audit.json` passes the predeclared seven calls with 320×240 and 640×640 images, changed/repeated/whitespace/long prompts, exact 32-token and output identities, finite results and zero backend work on a repeated prompt. `sam_contracts` covers malformed inputs, resizing/rounding/stride, tokenizer goldens and nonfinite-output rejection. The independent 1,024-image final export exercises additional source geometries and 4,753 prompts. PASS for the stated text-image CUDA workflow. |
| GGUF inventory, layout, bounds and placement | The unchanged schema-4 conversion manifest and GGUF hash bind 220 text/fusion/decoder Q8_0 tensors and 913 F32 tensors. `sam_contracts` rejects malformed schema, type, shape and offset cases. The candidate-bound topology and metadata audits account for all 23,988 active nodes, 9,466 exact aliases, 4,922 leaves and GPU buffer bounds, with no CPU/Metal/BLAS compute. PASS for this model and backend. |
| Q8 storage and boundary arithmetic | Direct GGUF-byte and native RHS staging checks pass all 1,838 active Q8 matrices and 936,772,736 same-operand dots. The fresh native MMVQ/MMQ boundary replay passes zero, signed/tie, tail, maximum-K and finite extreme fixtures; the deliberately overflowing F16-sum fixture is rejected as out of domain. The F32 bias/GELU epilogue passes 317,312 output checks. PASS for the declared finite domain. |
| Full active graph arithmetic | All ten compute groups pass independent numerical or exact-bit checks, with one-to-one coverage of 14,522 compute nodes over 16 graphs. The aggregate rehashed 34,278 raw payloads and 296 identities. Observer exports match unobserved outputs except score perturbations bounded by the observer tolerance; direct session outputs match the earlier candidate exactly. PASS for the predeclared path matrix. |
| Frozen final quality and benefit | Native F32 and Q8 both pass the unchanged absolute quality profile on the genuinely unused 1,024-image/4,753-prompt holdout; F32 cache makes an incremental cache comparison inapplicable. The uncontaminated 96-process performance run passes only the `gpu-memory` label for full-image and changed-prompt workloads. The frozen latency p50 threshold fails and the repeated-result memory label is inconclusive. |

The derived, read-only `recipe-adjudication.json` has SHA-256
`f1c61923e35b0d9cdacaac673dc17dd14ae25fcd406448726dcab70c631dbcd5`.
It rehashes the candidate model, conversion manifest, CUDA library, probe,
campaign, gates and compact component/final receipts, then records
`whole_recipe_arithmetic_status: PASS`, final absolute quality `PASS` and a
scoped `deployment_status: PASS` **only with `gpu-memory` labels for full-image
and changed-prompt CUDA inference on the measured RTX 4090**. It does not
upgrade the latency or repeated-result labels and does not qualify video,
other backends, cache formats, quantized vision or other model identities.
Earlier component, quality and performance receipts retain their original
`NOT_RUN` fields; the later adjudication is a separate conclusion over their
hash-bound evidence. No final or reserve image was used to choose the patch
or numerical gate. The final Release CUDA CTest rerun passed **31/31**;
the isolated Python suite passed **171/171** before this read-only decision.

After the whole-graph rehash and adjudication, routine evidence cleanup
rehash-verified and removed only **32,440** regenerable raw session tensor
files from the current candidate (202,104,266,752 allocated bytes, about
188.22 GiB). The exact selected-file inventory matched the capture
directories; no selected raw files remain. The aggregate and all ten compact
group-report hashes were unchanged, and rerunning the derived adjudication
after deletion returned the same SHA-256. The model, conversion manifest,
current verified CUDA build, performance and quality reports, boundary
fixtures, observer sources and focused failure diagnostics remain. The raw
captures can be regenerated only in fresh output directories. After cleanup,
`df -h` shows **230G** available on `/` and **620G** on `/mnt/SSD1`.
The large `SteamLibrary` (299G) is user content and was not touched;
the repository's older `build/` evidence and original checkpoints were kept.
The documentation/link check passed across 127 documents, Python syntax
checks passed for the ignored adjudication and cleanup scripts, and
`git diff --check` passed.
