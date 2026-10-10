# Frozen v2 mixed-Q8 cache arithmetic alignment

Created: 2026-10-09 03:42:41, Asia/Shanghai.

## Scope

Align independent CUDA Q8_0 feature-cache arithmetic evidence with the exact
GGML library and F32/mixed-Q8 recipe used by the frozen v2 final quality and
performance campaigns. The current v0.26.0 oracle/fixed-case run is a
different binary recipe and cannot itself upgrade the earlier results.
Preserve all frozen reports and thresholds. The target is the cache component
and its provenance, not an automatic whole-recipe arithmetic or deployment
PASS.

## Approach and steps

1. Confirm that the frozen image probe and retained cache codec probe load
   the same GGML CUDA library, and that their original binaries, libraries,
   inputs and payloads still match the frozen report hashes.
2. Apply the independent Q8_0 block verifier directly to the retained
   v0.25.3 CUDA cache payloads for original FPN 0/1 and a boundary fixture.
   Verify native placement, exact payload/layout and decoded F32 bits; do not
   use the GGML bridge as the oracle.
3. Reconcile those receipts against the frozen `f32-cache-mixed-q8-0` recipe,
   quality and performance reports. Record only a bound cache-component
   conclusion, including the source-input coverage and any remaining
   full-recipe arithmetic gap.
4. Run focused tests, documentation and whitespace checks. Keep the old
   machine-readable evidence immutable and remove disposable intermediates.

## Verification

Use the isolated reference Python environment and current verifier source.
Read and hash the original inputs, packed bytes, decoded outputs, native
reports, binary and loaded library. Recheck the frozen campaign identities
without changing the outputs or gates.

## Results

The retained v0.25.3 `sam_cache_codec_probe_v1` and the frozen v2
`sam_precision_image_probe` resolve the same
`libggml-cuda.so.0` (SHA-256
`1e4f954340ea2c30cb60deb49f4ccda6780993b2e09cb81bc8105aa053c925ef`).
The frozen candidate binary is SHA-256
`a2980263d888d21835b3644032311deabe4124d678d2042041b7c8794dd72a53`;
both hashes match the retained final-export manifests. Those manifests,
the frozen campaign, the final quality report and the performance report all
bind the same candidate recipe SHA-256
`6c7b0bd26b2b7311f270f5d95f16eff5152f790529b6b064b66868ba4c26947c`.
Its F32 parent differs only in `feature_cache` and uses the same binary and
library. The v2 gate and dataset hashes also agree.

The [independent Q8 block verifier](../../tools/validation/verify_precision_q8_cache.py)
was run directly on the *retained v0.25.3 native* FPN 0/1 and zero/tie/tail
payloads and their original F32 inputs, not on the current v0.26.0 copy.
All 829,491/829,491 blocks passed: zero scale, integer or decoded-bit errors;
one scale and 43 integer decisions used the existing bounded rounding-tie
allowance. Native probe reports identify CUDA compute and no CPU compute.
The earlier v1 codec report's FPN 0/1 `passed: false` fields remain unchanged:
they used an older expected-byte arithmetic contract. The new independent
verdict is a separate receipt bound to those exact old payload hashes.

The [reconciler](../../tools/validation/reconcile_precision_cache_arithmetic.py)
re-runs the independent verifier, checks the retained v1 payload identities,
resolves both binaries' loaded GGML CUDA library, and requires identical
frozen campaign/recipe/gate/dataset identities. Its immutable local result is
`build/precision-q8-cache-frozen-oracle-20261009/summary-v2.json`
(SHA-256 `a40a2c44c51bfddce8f0085f0aedc4a07e2d186539a32789c1d6f81ffa555050`).
The three per-case receipts and 28 bound files are under the same ignored
`build/` directory or their retained original locations, available only in
this validation workspace. Frozen quality is absolute+incremental **PASS**
and the recorded whole-image/new-prompt latency comparisons **PASS** under
their existing workload gates. The newly aligned **cache arithmetic
component** is **PASS** for these named inputs and the exact frozen GGML
library. No independent full-graph same-operand audit exists for the other
F32 model operators, so whole-recipe `arithmetic_status` and deployment
remain **NOT_RUN**. This receipt does not transfer the frozen 1024-image
quality/performance results to the different v0.26.0 binary recipe or prove
cache codec behavior for every one of those images.

Focused reconciliation tests passed, followed by the isolated Python tool
suite (169/169), documentation/link checks (87 documents) and
`git diff --check`. The original v1 and
v2 receipts, model outputs and gates were not modified.
