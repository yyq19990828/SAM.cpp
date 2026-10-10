# Unified quantization configuration

## Scope and approach

Unify the image tools' weight storage, activation policy, compute policy and
image-feature cache settings in one versioned configuration. Conversion,
reference exports, precision inspection and paired performance measurements
must resolve the same settings, reject conflicting inputs before expensive
work, and report requests separately from observed execution. Preserve existing
CLI usage and immutable historical results. Historical v2/v3 policies remain
archived.

The initial configuration describes implemented paths only: dense or one
quantized weight format with module selection, backend-selected activations,
supported compute policies and the existing image-feature cache modes.
Independent activation precision and arbitrary per-module/per-tensor mixed
formats must fail explicitly. This work does not add kernels or claim new
hardware qualification.

## Steps

1. Commit the completed benchmark/precision/archive changes before new work.
2. Inventory conversion rules, runtime policy scopes and tool restrictions.
3. Add a dependency-light shared configuration parser/resolver and a read-only
   validation/capability entry point. Integrate conversion and image benchmark
   commands, including independent baseline/candidate configurations.
4. Bind resolved configuration to model provenance and reports; reject
   unsupported combinations and disagreement with an existing model.
5. Document supported custom combinations and limitations in both languages.
6. Write separate implementation plans for mixed weight formats and independent
   activation quantization, with bounded development checks and user-owned
   quality decisions.

## Verification

- Test configuration schema errors, unsupported combinations, explicit CLI/file
  conflicts, conversion options, model/config disagreement and benchmark recipe
  propagation using small synthetic fixtures and subprocess stubs.
- Run the isolated reference Python tool suite and documentation checks; run
  relevant CPU checks if native code changes.
- Check staged/working-tree whitespace, preserve historical receipts, and
  remove temporary outputs created by this cycle. No large GPU acceptance,
  dataset copying or new model conversion is required.

## Results

- Initial completed slice committed as `900b289` before implementation.
- Added the dependency-free `tools/quantize/quantization_config.py` resolver and
  read-only `validate` / `capabilities` entry point. Strict schema 1 covers all
  four axes with explicit image/backend/allocation scope. Unknown/duplicate
  fields, arbitrary mixed formats and independent activation modes fail.
- Extracted the existing allocation definitions/functions into
  `tools/quantize/weight_policy.py`; their function ASTs are identical to the
  committed implementation. Existing GGUF helpers re-export the same API.
- Conversion, quality exports, precision inspection and paired performance
  share the resolver. Performance takes two complete configuration files on
  one backend. File/CLI conflicts and weight/model disagreements fail before
  inference; both performance models are checked before environment queries.
- New conversion manifests bind the shared policy/resolver sources and record
  applied weight settings separately from intended activation/compute/cache
  settings. Runtime reports retain normalized configuration/source identities,
  requested policy and unavailable kernel evidence. Changed configuration
  files prevent completion; offline reports need no live configuration file.
- Corrected schema-3 vision recipe metadata to match the native probe's empty
  `quantization_modules` field. The resolved configuration still identifies
  the vision selection; schema-4 manifests require their explicit canonical
  module list. No existing GGUF or receipt was rewritten.
- Added four portable JSON examples and bilingual configuration/combination
  guides. Future work has separate [mixed-weight](20261010-155435-mixed-weight-quantization.md)
  and [native-activation](20261010-155435-native-activation-quantization.md)
  implementation plans; their proposed features remain unavailable.

### Verification

- Isolated `.venv-reference/bin/python -B tools/test_tools.py`: 215 tests,
  68.237 seconds, OK with one existing skip. This includes twelve configuration
  regressions, all sixty nonempty module/format combinations, real tiny CPU
  checkpoint-to-GGUF conversion/inspection, synthetic export/performance
  producers, early rejection, source mutation and offline receipt checks.
  Test `FAIL` lines are expected negative-fixture output; the suite succeeded.
- The read-only configuration CLI also runs with Python `-S`, proving that it
  needs no torch/numpy/gguf installation. Existing legacy CLI defaults and
  grouped/flat/module entry points passed the complete tool suite.
- Read-only real-model inspection matched the full Q4 allocation and all 1,133
  tensor headers to the existing conversion manifest. The requested runtime
  policy was CUDA F16 with mixed-Q8 image cache; `inference_executed=false` and
  kernel arithmetic remained `NOT_COLLECTED`. Evidence:
  `build/quantization-config/full-q4-resolved.json` and
  `build/quantization-config/full-q4-precision.json` (Git-ignored local files,
  unavailable in a clean checkout). They occupy 16 KiB together and remain
  immutable. No large model was converted or copied.
- Documentation/link checks: PASS across 155 Markdown documents. Working-tree
  whitespace and all twelve new files passed checks.
- Both archived policy hashes and the eight completed receipts listed in
  `build/v3-first/interruption-20261010-142235-gpu-loss/receipt.json`
  (Git-ignored local evidence) remain unchanged.
- No native source/build configuration changed, so no new C++ build was needed.
  The verified CPU/CUDA builds were retained. Temporary fixture checkpoints,
  outputs and subprocess directories were removed by their isolated tests.
  No GPU model inference, large acceptance campaign or dataset transfer ran.

The initial commit is complete; this subsequent configuration slice remains in
the working tree for review. No remote push was requested or performed.
