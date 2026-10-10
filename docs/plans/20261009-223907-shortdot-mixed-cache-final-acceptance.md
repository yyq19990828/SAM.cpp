# Short-dot CUDA mixed-cache final acceptance

Created: 2026-10-09 22:39:07, Asia/Shanghai.

Current result: both frozen mixed-cache recipes passed complete evidence
adjudication on 2026-10-10, with latency labels for full-image and changed-prompt
inference on RTX 4090 CUDA only. Memory labels did not pass; repeated-result
inference has no benefit label. The earlier execution failures below retain
their original context. Final documentation checks are recorded in the
[adjudication plan](20261010-030705-mixed-cache-final-adjudication.md).

## Scope

Qualify two development-selected complete SAM 3 image recipes on the current
short-F32-dot CUDA build: F32 weights/F32 compute/mixed-Q8_0 feature cache and
custom text-fusion-decoder Q8_0 weights/F32 compute/mixed-Q8_0 feature cache.
Keep the same-weight F32-cache parents as incremental-quality and performance
comparators. Use the frozen v2 gates without tuning them to final results.
This campaign concerns RTX 4090 CUDA text/image inference only; it does not
qualify other backends, GPUs, video, or Q6/Q5/Q4/BF16/W8A8/FP8 recipes.

## Approach and steps

1. Bind the v2 gates, verified short-dot binary/library, F32 and custom Q8
   GGUF, development exports and quality reports by SHA-256. Confirm no GPU
   workload is running and that sufficient SSD space remains.
2. Select 1,024 genuinely unused annotated COCO train2017 images with frozen
   seed `20261011`. Use the original follow-up dataset as the prior manifest
   and exclude the complete consumed short-dot holdout dataset. Verify both
   image IDs and content hashes against retained prior evidence. Keep the
   original 1,024-image val2017 reserve unopened.
3. Download the selected source images, form the merged annotation/dataset
   artifacts, normalize inputs, and choose eight performance cases from the
   development split. Freeze the five exact recipes (official original,
   native F32 parent, native F32 mixed-cache, native Q8 parent, native Q8
   mixed-cache) and their benchmark baseline before evaluation inference.
4. Export and evaluate all five recipes on the same untouched final images.
   Apply the F32 or Q8 absolute quality profile and each mixed-cache
   incremental profile. Keep per-check failures and coverage explicit.
5. Run uncontaminated paired multi-case latency and GPU-memory measurements.
   Independently verify candidate-bound cache encoding and the active-graph
   arithmetic/metadata/paths for both complete combinations. A component
   check alone must not be used as a whole-recipe PASS.
6. Adjudicate each complete recipe against all frozen gates, document scoped
   support only for proven results, run appropriate CTest/tool/documentation
   checks, and clean rehash-verified regenerable intermediates while retaining
   models, manifests, immutable reports and the current verified build.

## Verification

Use `rtk proxy` and the isolated `.venv-reference` environment for the
existing tools. Avoid source edits during the frozen campaign. Freeze outputs
with exclusive creation and keep prior campaign receipts unchanged. Verify
preprocessing and annotations by hashes, no reserve overlap, strict CUDA
placement, complete image/prompt counts, bootstrap sample count, and the
performance session's exclusive GPU evidence. Use `git diff --check` and the
repository documentation checker after documenting results.

## Results

The preceding 896-image development screens found both combinations promising
but `INCONCLUSIVE` solely on the final-sample-size check; the single-case
ABBA latency screen is diagnostic only. Neither combination has a deployment
label yet.

The exact original checkpoint, BPE, F32 GGUF, custom Q8 GGUF, short-dot image
probe and GGML CUDA library rehashed to their prior SHA-256 identities:
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`,
`924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`,
`cb13ecd5012a2fe19b06d840049be6daa6b177b35256af8c4afeb12125352486`,
`f854afae5a2eb2c388ff2dd85804a4327b4525b45a09c1fdc9b5f5619c93142e`,
`2a0a66b1e6ae0ce04a415731bffd20a2c32e2762ed270e8887b332abdf8a4d94`
and `dc0ebfc0cbf0055b78a8111b02e2ab7ba245002f54602347fe8514baa339f6de`.
The unchanged v2 gates SHA-256 is
`4cf06bdc609e5a143b2d74b42f394480bbe39e7510726ecedfe44fde3212a802`.

The new selection uses seed `20261011`, the retained follow-up dataset as its
prior source and the complete consumed short-dot dataset as an explicit
exclusion. Its 1,024 unique evaluation image IDs have zero overlap with both
prior datasets and cover all 80 annotated COCO categories. The independent
download/content-hash validation passed. The ignored selection, download,
merged annotation and dataset SHA-256 values are respectively
`84acc0846100c8984933ccf805c18d1ab7e0bcc8c32d316c471554671b74da80`,
`d147b8d999e479936922b01e3a843546cbc29cc656c8cd0302985fabd2fdbeb5`,
`53ef43e35c992d9eb20b61e951b6695b27af5538765f0efdc43590963a683159`
and `0e736b17ac0eb92ef1329bf557d38aad8c9622604a3877d0779ff5d46fddd11b`.
The 256 calibration, 640 development and 1,024 reserve sample records match
the previous short-dot dataset exactly. The final split has 4,729 prompts;
the untouched reserve still has 1,024 images.
The first finalization call rejected a missing retained `val2017` source
directory before writing outputs; same-device hard links to the existing
verified val2017 files supplied those immutable sources, and the second
finalization passed.

Normalization completed 896 development and 1,024 evaluation RGB PNGs;
their ignored manifest SHA-256 values are
`28556009c8c0a50b68a797867959e00bdd8fec3c9c0e40ccd7e6bc9cb7a53c3a`
and `2dde32cd864524ab7c5a7723afae7503b4e9c3d5cee3f6bb59271fd41cf14a88`.
All 896 development source and normalized PNG hashes equal the preceding
development run. The eight newly frozen performance cases select the same
sample IDs and image hashes as the prior development campaign; their new
manifest binds the new dataset and input manifests, and has SHA-256
`eb1aa19d0d56317a82c174aacc58bc560bb759d0dc69e1813139a1acca98ecda`.
Development re-exports precede the final campaign freeze.
The five-run recipe selection has already been written under exclusive
creation, before any new final inference. It names the official reference,
F32 and Q8 F32-cache parents, and both mixed-Q8-cache candidates, with each
same-weight parent as that candidate's incremental/performance baseline.
Its ignored `recipe-selection.json` SHA-256 is
`3a8708136babfef689b820f221c7e38b41ca2fd6f71b499c2af99cf94b370ca5`.

For candidate-bound cache component evidence, an ignored `LD_PRELOAD` sidecar
was compiled against the exact GGML headers. It observes only F32-to-Q8_0
and Q8_0-to-F32 graph cast tensors on the unmodified image probe. An
independent verifier is prepared to check Q8_0 block scales/integers,
decoded bit patterns, exact packed-byte reuse and observed-versus-unobserved
outputs.

The official checkpoint re-export completed **896 images / 3,518 prompts**;
all 3,518 output payload SHA-256 values match the preceding same-image
development export exactly, with an identical recipe and the new dataset
identity. Its ignored manifest SHA-256 is
`f477fdf8f24422744b1905ce703771262fa34e93f1eff49d1bda5c8ea07fb09b`.
The native F32 parent likewise completed 896/3,518 with the new dataset
identity and strict CUDA placement. All 3,518 parsed outputs match the old
same-recipe development export after excluding only the variable runtime
object. Its ignored manifest SHA-256 is
`196271ce2da16122fc0c3821b50a91c802af6e67d52c0d7eb7505743fb8eee04`.
Its development scorer completed 2,000 paired-image bootstrap repetitions.
The absolute quality status is `INCONCLUSIVE` solely because 896 development
images are below the frozen 1,024-image final minimum; every other applicable
check passes, with zero bad objects and zero protected misses among 6,447
high-reference objects. The ignored
`quality-development-f32/metrics.json` SHA-256 is
`ca0dcd4d4524f0ed841799d1b3b1fa2002de15aa1418717ee7bba7c3320bfeee`.
The F32 mixed-cache export also completed 896/3,518 with strict CUDA
placement. All 3,518 parsed outputs equal its earlier same-recipe
development screen outside variable runtime fields. Its ignored manifest
SHA-256 is
`21b76a66485bd02e98ad5f7a4a67c9a6796b04faac971b444d232a2878888c12`.
Its absolute/incremental development scorer was interrupted after writing
only intermediate bootstrap objects; a fresh scorer was started under the
repository's ignored `build/mixed-cache-final-20261009/` directory.

The execution environment then changed to a sandbox with the validation SSD
read-only and no visible NVIDIA driver. The Q8 F32-cache wrapper had stopped
after its native subprocess completed all **896 images / 3,518 prompts** and
wrote a complete `native/run.json`, but before it could create the outer
manifest. Its unmodified native directory was copied to ignored workspace
storage. `recover_development_q8.py` reconstructed the outer manifest using
the same export validators and source snapshot, checking every copied payload
SHA-256, every strict-CUDA placement, and all 3,518 prior same-recipe payloads
outside variable runtime fields. The recovered run passed `load_run`, shared
input identity and archive verification. This recovers completed inference;
it does not claim a new GPU run in the sandbox.

The earlier complete Q8/mixed-cache development export was also reusable:
all 896 development sample records and normalized input hashes match this
campaign, as do its model, binary, recipe and current source hashes. A copied,
explicitly provenance-marked manifest now binds that retained 3,518-prompt
run to the new dataset identity in ignored workspace storage; its source run
and 3,518 output hashes, strict-CUDA placement, shared input identity and
archived sources were verified. `recipe-selection-workspace.json` preserves
the original five recipe IDs and baselines while pointing to the recovered or
reused development paths and new quality reports. Its SHA-256 is
`a6b8def878a8e75a8c9b1f81ab83b551f066afc1aed07cb847e0f0b294552f4f`.
No final evaluation image was inferred during that rebinding.

The F32/mixed-cache, Q8/F32-cache and Q8/mixed-cache development quality
reports completed 2,000 paired-image bootstrap repetitions per applicable
absolute and incremental comparison. All applicable precision checks passed;
each comparison has 40 `PASS`, 51 coverage-dependent `NOT_APPLICABLE`, and
only `evaluation_images` `INCONCLUSIVE` because development has 896 images
against the frozen final requirement of 1,024. The respective ignored
`metrics.json` SHA-256 values are
`5650c2603e60f5d16afc864d31f54562a013424af6f7ee5adb45dd50dcc5a5a5`,
`cb32bb45cc61a57cd80ca56ce7cdd89782cf145eeeabec506710cb4000edaec7`
and `2db4fdf57da0346be361a34ebfc20938007463dfabb545a08b2e9d3fa402785a`.
All reports are development-only and must not be presented as final passes.

The five-recipe `campaign.json` was frozen under ignored workspace storage
before any new evaluation inference. Its SHA-256 is
`2a664d5f3ef85f06bf96c9fb9fb7858fb426141fe1e5dbe21db45ea47ced82bd`.
The freeze tool verified complete development runs, quality reports, exact
recipe and baseline identities, source/model/binary hashes, the eight fixed
performance cases, final sample minimum and unopened reserve. A subsequent
`campaign_check` rehashed its entire identity closure successfully, and
`reserve_inference_allowed` remains false. CUDA final inference, paired GPU
performance and whole-recipe arithmetic verification remain unavailable in
this sandbox because the NVIDIA driver/device is not visible. No final
quality or deployment claim is made.

The subsequent [execution recovery plan](20261010-001526-frozen-mixed-cache-execution-recovery.md)
records the ignored final-stage sidecar, complete offline input/hash preflight
and verified pre-inference failure boundaries. It preserves this campaign's
identity and outstanding GPU acceptance requirements.

Routine space cleanup removed 36 derived Nsight SQLite exports (~2.52 GiB)
from `build/` only after confirming the matching `.nsys-rep` files and
recorded summaries remained. Five completed bootstrap-progress sidecars were
also removed after their final metrics were archived. The original
checkpoints, usable GGUF, current verified build and validation receipts were
retained.

Both candidate-bound cache observers have now run on the exact short-dot
image binary and library with the `perf-source-middle` development image.
Each captured two FPN F32-to-Q8_0 casts and both subsequent Q8_0-to-F32
consumers. The independent oracle passed **829,440/829,440** Q8_0 blocks per
recipe, with zero scale/integer/decoded-bit errors; both compact payloads
were consumed unchanged, and observed versus unobserved ranked outputs were
identical outside runtime fields. The ignored F32-weight and Q8-weight
verification receipt SHA-256 values are
`512f8218d217678426ba47e9f1bc47cad6410d058ce80ae4704fb6ca667265b0`
and `8516b1c37fe78354d12ddef7d076d30c1ab8078a0f3f2621098d1a4febedcc8f`.
The first verifier attempt rejected an `ldd` symlink spelling before receipt
creation; resolving the library path fixed that binding check. These are
candidate-bound cache *component* passes for one development image. Their
whole-recipe arithmetic remains `NOT_RUN`.

## Final results, 2026-10-10

The independent final adjudicator completed with exit 0 after both serial
performance comparisons. Its ignored
`build/mixed-cache-final-20261009/final-adjudication-20261010.json` SHA-256 is
`d98cf137508be441e3c059c721a365d610c927159f6f43edf62fd379eddcb240`;
the waiting-stage receipt SHA-256 is
`23a9a4d76242e8458d2b82b814f72f97dad2874af963f307e32d7e4c99b78bb5`.
The adjudication closes over 2,580 retained evidence identities. These ignored
local receipts are unavailable in a clean checkout; their detailed audit and
observer recovery are documented in the linked adjudication and
[session arithmetic](20261010-005446-mixed-cache-session-arithmetic.md) plans.

| Original step | Actual completed evidence |
| --- | --- |
| 1, freeze identities | Campaign, gate, source, model and executable closure reverified; the frozen recipes and numerical gates remain unchanged |
| 2, independent final data | 1,024 unused train2017 images, 4,729 prompts; the 1,024-image reserve remains unopened and inference-disallowed |
| 3, bind inputs and baselines | Five complete exports use identical frozen inputs; the two mixed-cache baselines are respectively F32 weights/F32 cache and custom Q8 weights/F32 cache |
| 4, final quality | All four native absolute comparisons PASS; both cache-incremental comparisons PASS, each with 2,000 image-bootstrap repetitions and zero protected misses |
| 5, arithmetic | F32: 28 graphs / 23,992 nodes / 14,532 compute nodes; Q8: 16 graphs / 24,004 nodes / 14,538 compute nodes. Every compute family, metadata/path/cache check and applicable current-library boundary fixture PASS |
| 5, performance | Two fresh 96-process comparisons, each 48 latency plus 48 independent memory processes over the eight frozen cases; unchanged assessment and exclusivity verified |
| 6, regressions | Four current seven-case absolute spatial regressions PASS, both incremental regressions PASS, legacy F32 seven-case regression PASS, required-GPU CUDA CTest 31/31 and isolated Python tools 171/171 PASS |
| 6, storage and documentation | Both rehash-verified raw cleanups complete; scoped bilingual guides, README, model catalog and changelog reviewed; documentation and whitespace prechecks PASS, final receipt binds the last checks and file identities |

Both complete-recipe arithmetic and deployment prerequisites are PASS. The
qualified deployment labels are limited to latency for the following measured
workloads, relative to each recipe's own same-weight F32-cache parent:

| Mixed-cache recipe | Full-image p50 reduction | Changed-prompt p50 reduction | Other benefit labels |
| --- | ---: | ---: | --- |
| F32 weights / F32 compute | 11.08% | 19.66% | none |
| Text/fusion/decoder Q8_0 weights / F32 compute | 12.38% | 21.23% | none |

The two GPU process-peak ratios are 1.0013169447 and 1.0016313214; RSS peak
ratios are 0.8584604470 and 0.8565384302. All memory/combined labels are FAIL,
and all repeated-result labels are FAIL. Preserve these outcomes without
relaxing gates or multiplying the Q8 parent's separately measured memory
benefit. The passing applicable category checks cover 55 classes; 25 classes
remain insufficiently covered. No other GPU, backend, video or precision is
qualified by this follow-up, and public loading keeps its existing cache policy.

The two cleanups removed 63,042 raw files / 410,047,868,928 allocated bytes
(381.8868 GiB), retaining original checkpoints, usable GGUF models, conversion
manifests, the current build and compact immutable evidence. The final
completion receipt will bind the reviewed documentation and final checks at
`build/mixed-cache-final-20261009/final-completion-20261010.json`.
