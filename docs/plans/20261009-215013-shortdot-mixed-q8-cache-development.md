# Short-dot CUDA mixed-Q8 cache development screen

Created: 2026-10-09 21:50:13, Asia/Shanghai.

## Scope

Continue the v2 per-precision acceptance work after the short-F32-dot
text/fusion/decoder Q8_0 recipe earned a scoped CUDA GPU-memory label. Test
whether adding the existing mixed-Q8_0 image-feature cache to that exact
recipe, and to its F32 parent, is a viable next complete candidate. This is
development screening only; no consumed final set or unopened reserve may be
used to select a recipe. Keep the frozen v2 gates and all earlier reports
unchanged.

## Approach and steps

1. Verify the current short-dot image binary, CUDA library, F32 and custom
   Q8_0 GGUF, development dataset/input/annotation identities and prior F32
   and Q8 development exports. Check available disk space and GPU occupancy.
2. Export the two additional mixed-Q8_0-cache recipes on the same 896-image
   development split, with exact F32 compute and CUDA backend. Save each
   output in a fresh ignored directory. The paired F32-cache exports and
   official checkpoint reference already exist under the same candidate
   root. Never overwrite them.
3. Score absolute quality against the official reference and incremental
   cache quality against the otherwise identical F32-cache recipe. The F32
   weights plus mixed-cache configuration uses the F32 absolute budget; the
   custom Q8 weights plus mixed-cache configuration uses the Q8 absolute
   budget. Both use the frozen mixed-cache incremental budget. Record each
   gate status and the actual failing items, not only the aggregate.
4. If either passes development screening, inspect cache storage/arithmetic
   evidence and run a small paired performance diagnostic before deciding
   whether to freeze a *new* campaign with genuinely unused final images.
   A fresh final evaluation and candidate-bound whole-recipe arithmetic are
   required before any new deployment claim. Keep the existing Q8-weight
   memory-only result separate.

## Verification

Use `rtk proxy` for commands, isolated `.venv-reference` for scoring, and the
current verified CUDA build without changing production source. Confirm
model/binary/library and dataset hashes before inference, check backend
placement and output completeness, and run applicable tests, documentation
links and `git diff --check` after edits. Clean unneeded generated intermediates
while preserving the current build, original models, manifests, final
reports and compact validation evidence.

## Results

The exact short-dot CUDA image probe, library, F32 GGUF and custom Q8 GGUF
rehash to their previous identities
`2a0a66b1e6ae0ce04a415731bffd20a2c32e2762ed270e8887b332abdf8a4d94`,
`dc0ebfc0cbf0055b78a8111b02e2ab7ba245002f54602347fe8514baa339f6de`,
`cb13ecd5012a2fe19b06d840049be6daa6b177b35256af8c4afeb12125352486`
and `f854afae5a2eb2c388ff2dd85804a4327b4525b45a09c1fdc9b5f5619c93142e`.
The 896-image development dataset and normalized-input manifest SHA-256
values remain `4ebcf322c8f8877849ee6f27382151d0ed59b58d58e0664c22835f97dad46883`
and `9496e2b7632ccd96ab62e1f1d13efaa9eec35ed2f15c24358bf8ca21abc33088`.
The existing official, F32 and Q8 development exports each contain all 896
images and 3,518 prompts on these inputs. At the start there was no other
GPU compute process and 620G remained available on the SSD.

The custom Q8 weights plus mixed-Q8 cache exported all **896 images / 3,518
prompts** with strict CUDA placement. Its recipe differs from the same-model
F32-cache parent only in `feature_cache`. The ignored
`development-q8-mixed-cache-screen/manifest.json` SHA-256 is
`2403ed80024345aa7cdc8f8784ad8afc82a867447150895fd9cbb0dd27fdf11e`.
The F32 weights plus mixed-Q8 cache likewise completed all 896/3,518 with
the same single recipe-field difference from its F32-cache parent;
`development-f32-mixed-cache-screen/manifest.json` SHA-256 is
`a609b8c9fd62d9900d3cf2576baf22163f2b1af39eb72faea8953084c20c5f6a`.
The first 100 paired outputs of each candidate have bit-equal scores and
boxes but changed mask payloads, so output parity alone cannot settle mask
quality.

The Q8-weight combination completed the independent absolute and incremental
v2 development quality evaluation, including 2,000 image-paired bootstrap
repetitions for each comparison. Both statuses are **INCONCLUSIVE solely
because 896 development images are below the frozen 1,024-image final
minimum**; all other applicable checks pass, while low-coverage checks remain
`NOT_APPLICABLE`.
This is a positive development screen, not a final PASS. The ignored
`quality-development-q8-mixed-cache-screen/metrics.json` SHA-256 is
`9f4e213005a4819417725dcf1d1735f83c1823890cfb278996d9be6d75739e8a`.
The F32-weight combination likewise completed both 2,000-repetition
comparisons. Its absolute and incremental statuses are **INCONCLUSIVE solely
because 896 images are below the frozen 1,024-image final minimum**; all
other applicable checks pass. Both comparisons have zero bad objects and
zero protected misses. The absolute AP-drop 95% upper bound is
`0.00005391663697618121` against `0.001`, and its mIoU-drop upper bound is
`0.00000773279276275931` against `0.001`. The corresponding incremental
bounds are `0.00005298549221521842` against `0.0025` and
`0.000007769274369480004` against `0.001`. The ignored
`quality-development-f32-mixed-cache-screen/metrics.json` SHA-256 is
`aeda6c39a068ad2d05f1cec7d0d6597b9d587e11fb5d47e6e094c8ddb81f3607`.
Neither development result is a final quality PASS.

The existing independent cache-block oracle was then run directly against
the *current short-dot library*, using the retained FPN 0/1 F32 inputs and
zero/tie/tail fixture. Its CUDA codec probe resolves the same library SHA as
the image export. All **829,491/829,491** Q8_0 blocks pass: zero scale,
integer and decoded-bit errors. The three ignored verifier receipts under
`cache-codec-shortdot/` have SHA-256 values
`75cec96c4b6d6424061fd9702da5c6cd7b2776603b7fccd3d6b2445edca17a7a`,
`4bc84004f9a475c16f6c37deb24aeb84c940f125cb56cdf45260739ed47aa21f`
and `04ffe510aaecaa77d01b3b7ebe3799fb4e2df470696dbb5d5e9140d5f130fc21`.
These are component checks on named inputs, not whole-recipe arithmetic.

An eight-process ABBA development latency diagnostic used the existing
`perf-source-middle` input, exact short-dot benchmark binary and CUDA
library, 5 warmups plus 20 measured iterations per phase, and no competing
GPU compute process. Each output passed the probe's strict CUDA placement
check. Relative to its same-weight F32-cache parent, the two-run median
Q8-weight mixed-cache ratios were `0.8763` for full-image and `0.7864` for
changed-prompt; the F32-weight ratios were `0.8754` and `0.7930`.
Repeated-result timings remained near parity. The measured cache payload was
`111,476,736` bytes for F32 cache and `33,509,376` bytes for mixed-Q8_0.
The ignored `development-cache-latency-screen-20261009/summary.json` SHA-256
is `660533b6f19bf196468c9174308e6621f36bebf37e7e0a96933e4e1eacd73f71`.
This is one development case, not the frozen multi-case latency or GPU-memory
gate. Neither recipe has candidate-bound whole-graph arithmetic for this
combination, and a new untouched final holdout is needed before a deployment
label.

Routine evidence cleanup rehashed every raw F32 tensor against the two
existing whitespace-session verification receipts, then removed 324
regenerable tensor files (4,069,036,032 allocated bytes) from the superseded
rounded candidate and current short-dot candidate. The unchanged receipts,
node inventories and compact outputs remain. On the same host, `df -hT`
showed 230G available on `/` and 623G on `/mnt/SSD1` afterward; the latter
also contains 299G of user Steam games outside this project's scope.
