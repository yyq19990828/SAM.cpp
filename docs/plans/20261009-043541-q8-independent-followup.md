# Independent follow-up acceptance for nonvision Q8

Created: 2026-10-09 04:35:41, Asia/Shanghai.

## Scope

Qualify the diagnostic SAM 3 CUDA Q8_0 `text,fusion,decoder` allocation beyond
its 896-image development result without reusing the v2 final-evaluation
images that have already been exposed. Preserve the original 1,024-image
reserve unopened, the frozen v2 reports and gates, and the exact candidate
model. Final quality, arithmetic and performance remain separate gates.

## Approach and steps

1. Freeze a follow-up v2-compatible dataset: retain the original 256
   calibration and 640 development images and the untouched 1,024 reserve
   images; exclude all 1,024 used v2 evaluation images; select 1,024 new
   evaluation images from the remaining eligible unique COCO images using
   annotation-only category/area coverage and a fixed seeded order. Rebuild
   prompts from annotations for new images. Bind old dataset, annotation,
   image and usage identities. Assert image/content disjointness and no
   overlap with any old evaluation or fixed regression image.
2. Rebind the complete original-checkpoint, F32 CUDA baseline and custom-Q8
   development outputs to the new dataset through fresh exports and score the
   two native recipes under the unchanged v2 gates. Keep the custom recipe
   diagnostic until it passes an independent holdout. Do not copy or relabel
   old exports or receipts.
3. Select and freeze a new campaign with the original reference, qualifying
   F32 baseline and exact custom-Q8 model, using eight existing development
   performance cases rebound to the new dataset. Verify every development
   receipt and preflight before any new evaluation inference.
4. Run the new 1,024-image evaluation reference and candidates once, score
   object/AP/mIoU/negative/protected gates, and retain failed attempts. If
   quality passes, run the frozen performance protocol only on exclusive GPU
   hardware and continue whole-recipe arithmetic validation. If it fails,
   retain the failure and diagnose without moving the threshold or selecting
   on the new evaluation set.

## Verification

Use the isolated reference environment and retained CUDA binaries. Test the
dataset constructor's exclusion and deterministic selection, validate all
manifests/identities, fixed spatial gates and complete quality reports, run
relevant tool tests and docs checks, and check `git diff --check`. Treat any
background GPU compute process as a performance contamination, not as an
excuse to weaken benchmark gates. After each cycle remove only disposable
intermediates, retaining GGUF, manifests, verified builds and bound evidence.

## Results

The new `tools/validation/prepare_coco_followup.py` selector retained the
original 896 calibration/development images and all 1,024 unopened reserve
images, excluded every prior v2 evaluation image, and selected 1,024 new
evaluation images with 3,833 prompts from 2,008 unused candidates. The new
dataset SHA-256 is
`ed2e1a3124ebb058482f9fd656f8005a755d7a78e651b784d6aca72cfcd34e60`.
An independent content/ID audit found zero overlap with any old dataset image
and byte-identical retained development/reserve records; its local receipt is
`build/precision-q8-followup-20261009/split-audit.json` (SHA-256
`41fffe3da2b493e7aa1b94082000b138c810a590f048e7d5b9d6b19ebdcdfda1`).
The new evaluation contains positive prompts for 75 of the 80 COCO categories;
parking meter, toaster, scissors, hair drier and toothbrush are absent. This
limits the final result's category scope. A subsequent annotation-only audit
found zero positive examples for these five categories among the remaining
unused non-reserve COCO val2017 images. The reserve includes 9 parking-meter,
3 scissors and 8 toothbrush images, but has had no inference; toaster and
hair drier are absent there too. A future 80-category quality gate needs a
new independent image source, not a post-evaluation threshold change. The
local audit is `build/precision-q8-followup-20261009/coverage-audit.json`
(SHA-256 `aa2467c3fbb23ac46d71822c97aa429aba268542342e32a36994a82210eb69db`).
The reserve is still unopened for inference.

Fresh 896-image/3,518-prompt development exports were completed for the
official checkpoint, F32 GGUF baseline and exact nonvision-Q8 candidate.
Their manifest SHA-256 values are respectively
`3b501615ea292aceefc54dca45fc3f0aa1959cc3231b3c0f02906b181f4eed59`,
`94d13ab4095280f9373e2c3272067dbdebf8f90fdafa8f7f99b949fa63bc7f7d`
and `6cfc7e5714a358d3f6c582563911e308e910f3614efa3b4ee6e39586c06ff0da`.
All 3,518 official outputs were byte-identical to the old development oracle.
The two native exports differed from their old counterparts only in runtime
accounting; all 3,518 inference payloads matched exactly after removing that
field. New development scoring found zero bad objects for F32 and 17/6,447
for Q8, with every applicable numerical check passing. Both overall reports
remain INCONCLUSIVE solely for the development image count of 896. Their
metrics SHA-256 values are
`eb451f192e6d4a82e5e104e813ef87d1b0eccb3801170b710192c4984fdac855`
and `a15ede773a30c285dccdfdbda923e59c19315647a0de3a2044f87c6601b04549`.

The exact original/F32/Q8 selection was frozen **before any new evaluation
inference** as `build/precision-q8-followup-20261009/campaign.json` (SHA-256
`0d7e79b83440b8b643ea89a06f65740e90aee4e904bece87c97bb152724d9a6d`).
The preflight checked all three source/recipe identities and all 1,024
evaluation inputs; its SHA-256 is
`2268ade2e84c03558964d06c965a738fb8dc0702d61bed21a7d9b51a8e6531a3`.
All three 1,024-image/3,833-prompt evaluation exports completed with the
same sample IDs, input-image hashes and campaign hash. Their manifest SHA-256
values for original, F32 and Q8 are
`1354298e29e3ce2c2032f897c0ed5ff03e97d504b95a9db42c59762f49fdcaba`,
`062631e536159b7b17d958c617333a4e26d69441cad751b19dd020b98ae96c7f`
and `75bd5326370a184221fb700d8980a270f75a6b2f32cf30882d3d2ba681094a01`.
The immutable attempt ledger contains exactly three claims, one for each
frozen recipe. No model inference has been run on the reserve.

The independent final image-quality gate is **PASS** for both native recipes.
Against 7,067 high-confidence original-reference objects, F32 had zero bad
objects, while nonvision-Q8 had 18 (0.255%, under the unchanged 0.5% Q8
limit). Q8 had no missing or extra high-confidence objects, no newly positive
negative prompts, and no misses among 2,123 protected objects. Ranked mask AP
drop was -0.0000511 (limit 0.005); positive-mask mIoU drop was 0.0000831
(limit 0.0025). Across 2,000 paired image bootstrap repetitions, the 95th
percentile upper bounds were 0.0003487 for AP drop and 0.0004640 for mIoU
drop. Every applicable quality check passed. The F32 and Q8 metrics SHA-256
values are respectively
`19d9ae92bc58a502fd5932eda50827b3c0a9e245894a651e46faeb51ed6f85e8`
and `2ca27350c8aa5f59e078f095b91fe776a8b6922ce179c725da8ee0d56c04cd45`.
The Q8 object report SHA-256 is
`30009256905eaf819779d19ccb17e31158ab9e2d5abe223ad04fe6be9e76d0e0`.

A read-only closure audit rehashed every bound export and quality artifact,
verified the frozen campaign against current sources, and checked all three
exclusive attempt claims. Its 24 selected top-level bindings and results are
in `build/precision-q8-followup-20261009/quality-closure.json` (SHA-256
`e88b791e3128412862ba63067aa03d61f05ba43f30ec659ef648277af813df36`).
The receipts still report whole-recipe arithmetic `NOT_RUN`, performance
labels empty and deployment `NOT_RUN`. This PASS concerns only prompted image
quality on the fresh subset, including its 75-category coverage boundary; it
does not qualify unsupported categories, other backends, video or deployment.

The custom model's previously untested Q8_0 K dimensions 256, 512, 2,048 and
4,096 now have passing selected synthetic shape-only MMVQ/MMQ arithmetic
checks under the same frozen GGML library; see the
[component arithmetic follow-up](20261009-063651-q8-nonvision-shape-arithmetic.md).
Whole-recipe arithmetic remains NOT_RUN. The existing remote-desktop process
is also listed as a second NVIDIA compute PID, so the formal exclusive-GPU
performance runner cannot start while this session occupies the device.
Neither arithmetic nor performance gates were weakened.

The focused dataset tests passed 5/5, the isolated tool suite passed 170/170,
documentation checks passed 91 documents before the later shape follow-up;
the subsequent complete tool suite again passed 170/170, documentation checks
passed 92 documents, and `git diff --check` passed. All model, dataset, build
and receipt paths in this section are ignored local evidence available only
in this validation workspace.
