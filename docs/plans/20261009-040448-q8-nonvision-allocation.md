# Nonvision Q8 module allocation diagnostic

Created: 2026-10-09 04:04:48, Asia/Shanghai.

## Scope

Find whether keeping the vision encoder in F32 while quantizing SAM 3 text,
fusion and decoder linear weights improves the failed CUDA Q8 model-quality
gate. The existing 896-image development receipts show 36 bad objects for
vision-Q8 and 51 for full-Q8 against the unchanged 0.5% limit (at most 32 of
6,447 high-reference objects). Neither existing candidate passes. This new
custom allocation is diagnostic, with a distinct model and recipe identity;
it does not revise frozen v2 gates, final results or published support.

## Approach and steps

1. Independently compare the retained vision/full-Q8 failed image/prompt and
   object records, checking their source hashes, overlap and failure kinds.
   Use this only to motivate the new allocation, not to prune failing cases.
2. Convert the original checkpoint to a new schema-4 GGUF with exactly
   `text,fusion,decoder` Q8_0 linears, retaining the vision weights in F32.
   Validate checkpoint, BPE, model payload and module inventory provenance.
3. Run the seven fixed reference cases and the v2 spatial regression gate.
   If the model loads and remains valid, export the full 896-image development
   split with 3,518 prompts under a new recipe and score it against the
   retained original reference with the *unchanged* Q8 v2 gate. Do not touch
   evaluation or reserve images.
4. Record object/AP/mIoU/negative-prompt results and all identities. If this
   allocation passes development, identify the remaining arithmetic,
   final-quality and performance work before any support claim. If it fails,
   keep the failure and inspect its cause. Run relevant tests and clean
   disposable intermediates.

## Verification

Use the isolated reference environment and the CUDA backend on RTX 4090.
Maintain old receipts and source images immutable. Keep new model, conversion
manifest and diagnostic outputs under ignored `models/` and `build/` paths,
with SHA-256 binding and availability noted in this plan.

## Results

The retained vision-Q8 and full-Q8 development reports were checked against
their bound source hashes. Their 36 and 51 bad objects occur in 33 and 44
image/category prompt pairs, respectively; 26 pairs overlap. The failures
are dispersed rather than attributable to one prompt or a single category.

Conversion from the original checkpoint produced
`models/sam3-image-modules-text-fusion-decoder-q8_0-diagnostic.gguf` (2,402,596,704
bytes, SHA-256
`f854afae5a2eb2c388ff2dd85804a4327b4525b45a09c1fdc9b5f5619c93142e`).
Its conversion manifest has SHA-256
`93535a151d137fe4c158f5e9fdac0915311b08e31b0086cfa553c72bb56aa7b0`.
The declared `image-modules-linear-q8_0-v1` inventory has 97 text, 36 fusion
and 87 decoder Q8_0 tensors (220 total), with all vision tensors F32. The
remaining 913 tensors are F32. These files are available only in this
validation workspace under the ignored `models/` path.

The official-reference seven-case validator passed output quality on all
seven cases; tensor-fidelity remains a separate failed diagnostic, and the
validator marks custom allocation and release support ineligible. Its
`build/q8-nonvision-20261009/fixed-v1/metrics.json` SHA-256 is
`515e9301b2a433503cf077723a5cd8e34cce2fac01a1874e3ef4e286ba5e86a1`.
The v2 seven-case spatial regression passed absolute and overall checks;
incremental cache checks were not applicable. Its manifest SHA-256 is
`2dc99b1969bb24e4f0bf881434b16be321e6e8d3f595d9aff60bbc4ac0575fdc`.

The native CUDA F32-compute/F32-cache development export completed all 896
images and 3,518 prompts. Its manifest SHA-256 is
`5e6963a1dd51ebb2c8677cd543d6e3dcd0c063d818539622dcf3036962e98767`,
binding recipe SHA-256
`48f20a71c771573436e70e6ea1c1c3698821fa0ae5a82e1a97f37112abd93d47`,
the new model, original checkpoint and frozen v2 CUDA executable. The unchanged
Q8 v2 development scorer found **17/6,447 bad high-reference objects**
(0.264%, limit 0.5%), no missing/extra high objects, no newly positive
negative prompts and no protected misses among 1,910 protected objects.
Ranked mask AP drop was 0.0003873 (limit 0.005), positive-mask mIoU drop
0.0001984 (limit 0.0025); 2,000 image-paired bootstrap repetitions gave
upper bounds 0.0008681 and 0.0009459, respectively. Every applicable numeric
quality check passed. The single `evaluation_images` check is INCONCLUSIVE
because development contains 896 images, below the formal 1,024-image
minimum. Thus overall quality is **INCONCLUSIVE**, not PASS. The metrics and
object report SHA-256 values are
`9ce18628eeb4e063d1e61c0ae5a2bc2770134ae7df4df7c8233bafe661895a59`
and `f053369fc7798cebbb648edcde83d96a9ad8f82f6975f2f66729a2de3d8d2242`.
All receipts under `build/q8-nonvision-20261009/` are available only in this
validation workspace.

For the same 6,447-object denominator and unchanged gate, retained vision-Q8
and full-Q8 development runs had 36 and 51 bad objects (both FAIL). The new
17 bad objects occur in 16 prompt pairs; eight pairs overlap vision-Q8, 11
overlap full-Q8, and five occur in neither. Among its matched failures, score,
mask and box limits are crossed 10, seven and two times (not disjoint).
Keeping the vision encoder F32 improves this development metric, but it does
not isolate a causal module or establish performance benefit.

This candidate was selected after the prior v2 final campaign used its
evaluation split. Do not retroactively include it in that campaign. A newly
frozen independent holdout, whole-recipe arithmetic coverage and measured
performance against a qualified baseline remain necessary before any support
or deployment label. No evaluation or reserve inference was run for this
diagnostic, and the frozen gates and prior reports were not changed.

The isolated Python tool suite passed 169/169 tests, documentation and local
link checks passed 90 documents, and `git diff --check` passed. The current
verified build, usable GGUF, manifest and hash-bound quality evidence were
retained; temporary test directories were removed by the test harness.
