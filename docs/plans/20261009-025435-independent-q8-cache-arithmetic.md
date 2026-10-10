# Independent Q8 cache arithmetic acceptance

Created: 2026-10-09 02:54:35, Asia/Shanghai.

## Scope

Continue v2 quantization acceptance by closing the recorded arithmetic-evidence gap for the CUDA Q8_0 feature cache. The existing codec comparison derives expected CUDA bytes from the same pinned GGML CUDA routine; it verifies integration but is not an independent encoding oracle. Preserve the frozen v2 quality/performance reports and thresholds.

## Approach and steps

1. Inspect the current pinned GGML encoder and the native cache path, including the v0.26.0 upgrade boundary. Define an independent block oracle for 32 F32 values, half scale storage, signed integers, layout and decoded values. State a narrow allowance for fast-math rounding ties rather than relaxing ordinary values.
2. Add a reusable verifier and focused boundary tests. Check zero blocks, positive/negative ties, scale half boundaries, tail rows and real FPN tensors against native CUDA payloads. Bind new receipts to input, payload, binary, source and dependency identities.
3. Rebuild the relevant probe on the current pinned dependency, execute it on matching CUDA hardware, run appropriate test suites, and review any disagreement before claiming arithmetic PASS.
4. Record exactly what the evidence proves for the mixed Q8 cache recipe. Keep final quality/performance receipts immutable; update documentation and changelog only for established results.
5. On the current v0.26.0 binary, rerun the seven fixed original-reference cases for the F32 parent and mixed Q8 cache, checking both absolute and incremental v2 spatial gates. Keep these results distinct from the frozen 1024-image evaluation.

## Verification

Run focused verifier tests, native CUDA codec checks, the relevant CPU/CUDA CTest targets and `git diff --check`. Retain the current verified build, inputs, models and immutable evidence; remove only disposable intermediates after validation.

## Results

The [independent verifier](../../tools/validation/verify_precision_q8_cache.py) now checks every 32-value block against the original F32 input. It computes the scale and nearest signed value in F64, admits an adjacent binary16 scale or signed integer only when the exact value is within four binary32 ulps of the appropriate rounding midpoint, and requires the native decoded F32 bits to match the stored block exactly. It rejects non-finite input/scale, negative scale, `-128`, wrong lengths, wrong row layout and mismatched native CUDA placement. This allowance is limited to encoding arithmetic; it does not alter a mask, score, AP or mIoU gate.

The pinned GGML v0.26.0 CUDA probe was rebuilt and linked to the v0.26.0 GGML libraries. The original RTX 4090 FPN 0/1 inputs for the mixed Q8 cache were encoded anew, together with the 17-row zero/tie/tail fixture. All 829,491 blocks passed; the real FPN levels account for 829,440 of them. There were zero scale, integer or decoded-bit errors. One scale and 43 integer values used the explicitly bounded rounding-tie allowance. Native reports record CUDA compute nodes and zero CPU compute nodes. The current payload bytes equal the earlier v0.25.3 probe payloads for these three inputs; this equality was observed, not used as the oracle.

The immutable local summary is `build/precision-q8-cache-oracle-20261009/accepted/summary.json` (SHA-256 `0c7f9d58d2588a26aa678e974e2926d5cceab2b899458e765a14395ff00a1621`). Its per-case receipts bind the input, native payload, decoded output, native probe report, rebuilt binary, CUDA/base GGML libraries, patched encoder source and GGML provenance. The raw native output is retained under `build/precision-q8-cache-oracle-20261009/current/`. These Git-ignored paths are available only in the local workspace. Earlier exploratory and superseded receipts were removed; the retained receipts and payloads were not rewritten.

Focused boundary/tamper tests passed, the final isolated Python tool suite passed 156/156, and the current CUDA build passed the six selected quantization, host-tensor and precision CTest cases. `git diff --check` passed. The newly checked component is CUDA Q8_0 cache encode/decode for the named inputs and shapes. The frozen v2 final quality/performance receipts retain their original identity and statuses; the current GGML upgrade does not inherit a new dataset-wide quality or deployment result from this component test. Full-recipe arithmetic acceptance, W8A8 raw INT32-dot evidence, and FP8 arithmetic remain separate work.

The current GGML v0.26.0 `sam_precision_image_probe` also completed the seven fixed original-checkpoint cases using the same F32 GGUF. The uncompressed F32 parent passed all seven absolute v2 spatial checks. The `mixed-q8_0` feature-cache variant passed all seven absolute and all seven stricter incremental checks against that parent, with no failed case. The local manifests are `build/precision-q8-cache-oracle-20261009/fixed-v0260/f32/manifest.json` (SHA-256 `193b6f148126a98b403912c001060bc7622d5d3c23e19ef36b14fdece0981ca2`) and `build/precision-q8-cache-oracle-20261009/fixed-v0260/mixed-q8-0/manifest.json` (SHA-256 `5f71f54f0e378666bf28d68fa93cd57585989f022144995a8ee8010724aa3e38`). They retain the native case outputs, original reference binding, recipe, binary and source identities. These seven fixed cases have no COCO GT, so they cannot establish the protected-GT or 1024-image quality requirements. No frozen evaluation or reserve image was inferred anew in this step.
