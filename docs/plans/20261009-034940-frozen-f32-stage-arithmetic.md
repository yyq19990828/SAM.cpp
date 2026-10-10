# Frozen v2 F32 stage arithmetic diagnosis

Created: 2026-10-09 03:49:40, Asia/Shanghai.

## Scope

Measure stage-by-stage output differences between the frozen v2 CUDA F32
parent recipe and the retained official-checkpoint SAM 3 F32 reference on
the seven fixed image cases. Identify where deviations appear and whether
the available evidence can close any part of the remaining full-graph
arithmetic gap. Keep the frozen v2 gates, reports and reference tensors
unchanged. The official reference uses original checkpoint weights, so this
comparison alone is not a same-operand arithmetic acceptance gate.

## Approach and steps

1. Bind the frozen test driver, loaded GGML CUDA library, F32 GGUF and
   reference manifest to their hashes and verify each input and reference
   tensor file before comparison.
2. Run the frozen test driver on all seven reference PPMs, once per prompt,
   into fresh ignored `build/` directories. Confirm CUDA execution and
   tokenizer agreement with the official reference.
3. Measure every exported tensor's shape, finite values, relative L2 and
   maximum absolute difference using the existing validation arithmetic.
   Report the first diverging stage and distinguish weight/graph differences
   from evidence about the mixed-Q8 cache.
4. Independently compare each actual F32 GGUF weight payload with its original
   checkpoint tensor, accounting for the declared RoPE complex-pair and
   positional-embedding transformations. This determines whether weight
   conversion contributes to the stage differences.
5. Update documentation with precise conclusions and run focused validation,
   documentation and whitespace checks. Clean disposable intermediates while
   retaining the bound receipts.

## Verification

Compare only hash-bound files from the frozen binary/library/model, original
checkpoint and retained reference corpus. Record per-case and aggregate JSON evidence in
`build/precision-frozen-f32-stage-20261009/`; keep all old receipts immutable.

## Results

The frozen v2 `test_image` driver (SHA-256
`9c0190cd5b360a76f690001cd45add69c91a28acb0d926d7ad3b257f5d25a036`)
loads the same GGML CUDA library as the frozen image and cache probes
(`1e4f954340ea2c30cb60deb49f4ccda6780993b2e09cb81bc8105aa053c925ef`).
The F32 GGUF SHA-256 is
`cb13ecd5012a2fe19b06d840049be6daa6b177b35256af8c4afeb12125352486`;
its sidecar and the official reference both identify checkpoint SHA-256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
All seven fixed cases ran through CUDA, with no CPU/Metal graph nodes and
identical 32-token IDs to the reference. The reference inputs and all raw
reference and native tensors passed shape, finite-value and SHA-256 checks.

The maximum relative L2 across cases, per exported stage, was:

| Stage | Maximum relative L2 |
| --- | ---: |
| Preprocessed image | 0 (bit-identical) |
| Vision features 0 / 1 / 2 | 5.05e-6 / 1.11e-5 / 1.24e-5 |
| Text features | 1.72e-6 |
| Fusion features | 1.10e-5 |
| Predicted boxes | 6.02e-5 |
| Presence logits | 5.08e-6 |
| Class logits | 1.39e-4 |
| Mask logits | 1.13e-4 |

The first observed output difference is in `vision_features_0`; error grows
downstream, with the largest class/mask relative errors on the `truck` +
`purple elephant` negative prompt. These values are diagnostic: there is no
frozen 2e-5 limit on full-model stage outputs. An independent
[F32 weight verifier](../../tools/validation/verify_f32_gguf_weight_parity.py)
compared every actual GGUF payload with the original checkpoint: all 1,133
tensors and 842,343,734 values match byte-for-byte after the declared
positional-embedding slice and 32 complex-to-real-pair conversions. Its receipt
is `build/precision-frozen-f32-stage-20261009/weight-parity.json` (SHA-256
`45ef1391eb5f0b67c794f4482baedf04fc187843210f9f996f2142134b05aa10`,
available only in this validation workspace). Stored-weight rounding therefore
does not explain the stage differences. The Meta and native execution graphs
still may use different operations and intermediate operands; the final-stage
comparison cannot isolate each operator's arithmetic or establish a
same-operand PASS. No mixed-Q8
feature cache was used in these F32-parent runs. The bounded cache-component
PASS from the preceding plan and frozen final quality/performance PASS remain
unchanged; whole-recipe arithmetic and deployment remain **NOT_RUN**.

The [stage comparator](../../tools/validation/compare_f32_reference_stages.py)
records all seven case metrics and 184 file hashes in
`build/precision-frozen-f32-stage-20261009/summary.json` (SHA-256
`620e4051a385b20a097b020c7dc16744421a02825da15cdc1d8f4e6ff704b1fa`,
available only in this validation workspace). The frozen reference, old v2
receipts and gates were not modified.

The isolated Python tool suite passed 169/169, the documentation/link check
passed 88 documents, and `git diff --check` passed. All 184 stage-receipt
file hashes were independently rechecked after the comparison.
