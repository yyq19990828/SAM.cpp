# Bound video diagnostic tensor names

Created: 2026-10-03 11:46:04 Asia/Shanghai.
Status: complete; pre-commit review fix.

## Scope and approach

The diagnostic getter parses only the numeric prefix of object IDs, then caches
a tensor under the original name. Junk, signed and leading-zero aliases can
retain duplicate tensor copies for the same object until the next frame/reset.
Require the supplied ID text to equal its canonical decimal representation
before object lookup and caching. Keep all canonical names, graph arithmetic,
weight/storage policies and temporal behavior unchanged.

## Steps and verification

1. Add malformed-name rejection to the existing real video-session test, plus
   a valid canonical pointer/cache check. Compile it separately against the
   immutable Metal libraries and current old header, then run a real hybrid
   positive/negative short session to show RED without changing those builds.
2. Add the canonical ID guard. Recompile the same test separately and run GREEN.
   Use the existing weight-free CPU/Metal checks and isolated tools suite.
3. Record source-only diagnostic delta and recheck unaffected artifact/source
   identities. Reuse the sealed full numerical/long/performance acceptance;
   the new getter restriction cannot alter canonical dump names or inference.
4. Update the Unreleased changelog, then commit and merge local main as requested.

No full CPU/model acceptance or performance measurement is repeated.

## Results

The isolated old-header hybrid Metal test failed at the first malformed alias
with `expected video boundary rejection` (RED). After the three-line canonical
ID change, the same real six-push session test passed all five malformed-name
rejections, canonical pointer/cache, Model lifetime, isolation and reset checks
(GREEN). The current test also compiles/links against CPU-only immutable GGML
libraries and its help path passes, without loading CPU model weights.

Weight-free CTest passes 11/11 on each backend, the isolated Python suite passes
20/20, and all 38 locked reference packages match their installed versions and
pass compatibility checks. The review fix changes no graph, weights, temporal
policy or valid diagnostic lookup: reversing only the canonical ID block
reconstructs the exact pre-fix header. All 151 sealed acceptance receipt hashes
remain unchanged; the only changed file in their current source/artifact map
is this session header. The original completion snapshot retains its historical
identities; post-review readiness records the diagnostic-only delta.

Receipts: `build/video-validation/20261002-182848/tensor-name-{red,green}/`,
`tensor-name-cpu-compile/`, and `post-review-readiness.json`.
