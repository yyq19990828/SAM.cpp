# Independent performance benchmarks and explicit precision evidence

Created: 2026-10-10 15:17:10 Asia/Shanghai.

## Scope

Continue the application benchmark plan: decouple performance measurement from
quality qualification, then describe weight/activation/compute/cache choices and
their execution evidence consistently. Archive v2/v3 policies and their command
implementations; retain no active historical qualification entry point. Preserve
historical result files unchanged. This follows the user's subsequent instruction
to archive historical policies rather than retaining them in the current flow. Do not
start large GPU acceptance or restore the interrupted campaign.

## Approach and steps

1. Add a standalone application performance command with explicit models,
   compute/cache settings and image/prompt cases. It must not require COCO,
   quality reports, a quality tier or a final campaign. Preserve independent
   latency/memory processes, paired order, immutable inputs and contamination
   reporting. Report raw latency/memory and optional benefit tags without a
   quality veto or a minimum improvement requirement.
2. Reuse policy-free measurement code and extract reusable protocol summaries.
   Move historical policy decisions into the archive namespace, outside current
   validation/benchmark entry points.
   Allow a small case selection and an offline summary path for existing raw
   records. Do not infer performance from quality-export wall time.
3. Centralize precision descriptions for quality and performance reports. Name
   requested configuration, resolved storage/runtime policy and observed
   evidence separately. Validate unsupported activation/backend combinations
   before inference. Include mixed weight exceptions and per-level image-cache
   types; distinguish a profile identifier from observed kernel arithmetic.
4. Improve opt-in graph diagnostics to expose tensor operand/output types and
   precision hints, with an explicit limitation that kernel-internal arithmetic
   remains untraced. Diagnostic observers must not run during timing workloads.
5. Add bilingual usage/capability documentation and changelog entries; cover
   quality-independent timing, slowdowns, contamination, invalid records,
   precision mismatches and unsupported configuration with behavioral tests.

## Verification

Run CPU-only synthetic process/graph tests, the isolated Python tools suite,
selected CPU CTest/build checks for changed C++ diagnostics, documentation
validation and `git diff --check`. Confirm frozen v2/v3 policy hashes and prior
campaign artifacts remain unchanged. Reuse the current verified build when
appropriate; preserve original checkpoints and useful generated models. Remove
unneeded test/build intermediates after checks.

No hardware-specific speedup or universal FP16 claim may follow from source,
configuration strings, synthetic records or CPU-only checks. Record whether
each result is declared, resolved from model/source contracts, observed at graph
boundaries or confirmed by kernel tracing. Kernel evidence unavailable in this
delivery must remain explicitly uncollected.

## Results

Completed the scoped tool/documentation changes:

- `quantization_benchmark.py performance` now accepts application images and
  baseline/candidate GGUF recipes directly. It calls no quality evaluator or
  historical policy loader. There is no campaign, COCO annotation, tier,
  passing-quality receipt, minimum case count beyond one, or benefit veto.
  AB/BA pairing and independent latency/memory child processes are retained.
  Users can select cases, iteration counts, process timeout and advisory benefit
  thresholds. Raw process identities, input/recipe/protocol hashes, per-case
  tails, pooled percentiles and maximum process memory peaks are retained.
  Slowdowns remain valid completed reports; contaminated runs cannot earn tags.
- `summarize-performance` rechecks saved input/record hashes and writes a fresh
  summary without loading weights or requiring the original executable. It
  describes stored provenance rather than recertifying a producer's current
  model, binary or hardware. Same-prompt result hits remain distinct from
  image encoding and changed-prompt inference.
- `inspect-precision` verifies model/sidecar hashes and bounded GGUF tensor
  headers without inference. Shared descriptions distinguish requested
  activation/compute/cache choices, weight type inventories, source-resolved
  paths and producer-reported runtime policy identifiers. CPU F16 weight
  promotion, CPU quantized-weight casts and mixed Q8 image-cache levels are
  explicit. Unsupported independent activation settings fail before loading.
- Opt-in `sam_profile_graph` now records operand source slots/types, output
  types and accumulation/RHS hints, and accepts the existing experimental
  image-cache selection. These observations are graph allocation boundaries;
  kernel-private representations and arithmetic remain `NOT_COLLECTED`.
  Performance measurement does not install this observer.
- Following the user's steering, moved twelve historical v2/v3 policy/helper
  implementations and both policy JSON files into
  `tools/archive/precision_v2_v3/`. Removed their active grouped commands and
  flat forwarders. Active tool modules have no imports of archived policy
  code. Moved policy prose into `docs/archive/`, updated current bilingual
  guides and changelog, and adjusted historical links without changing result
  receipts. Developer arithmetic/reference validators retain their separate
  scope; this delivery concerns application image quantization benchmarks,
  not a rewrite of the video qualification workflow.

Verification:

- Release CPU build using the existing pinned GGML source: profiler and
  benchmark probe plus four relevant test executables compiled successfully.
  Selected `sam_graph_profile`, `sam_graph_workspace`, `sam_host_tensor` and
  `sam_precision` CTests: 4/4 passed. A tiny real CPU matmul verifies that F16
  weight storage, F32 input/output and an F16 RHS hint are recorded separately;
  it does not certify GPU kernel arithmetic.
- Isolated Python tools suite: 203 tests, 78.066 seconds, OK (one CUDA-device
  test skipped in the CPU reference environment). The preceding cycle failed
  because phase-input preparation was relocated while the test process still
  held its old entry-point inventory; the completed rerun used the updated
  inventory. Subsequent descriptive CPU-F16 reporting checks passed all 15
  application quality/performance tests in 0.500 seconds.
- The new tests cover running with no quality evidence, valid slowdowns,
  alternating process order, instability/contamination, offline summaries
  after model/binary removal, changed inputs/records, missing/reordered records,
  invalid timing/PID/sampler data, precision mismatch and unsupported settings.
  A real lightweight CPU child exercises direct process observation; synthetic
  timings are test fixtures, not hardware measurements.
- Documentation validation: 150 Markdown documents passed bilingual table and
  local-link checks. `git diff --check` passed. Both archived policy hashes match
  their prior values, and all eight completed historical receipts listed in
  `build/v3-first/interruption-20261010-142235-gpu-loss/receipt.json` match their
  frozen hashes. That evidence is Git-ignored and available only in the retained
  local validation archive.

Read-only weight inspection produced
`build/precision-reporting/q4-f16-mixed.json` (Git-ignored local evidence). The
full Q4_K preset contains 316 Q4_K tensors, 32 Q8_0 fallback tensors and 785 F32
tensors. `inference_executed=false`: requesting CUDA F16/mixed-cache description
did not initialize CUDA or execute a model. This is storage evidence only.

No GPU model inference, large COCO acceptance, weight generation/download or
performance campaign ran. Original checkpoints, usable GGUF models, previous
CUDA build and historical receipts were retained. Temporary test directories
were removed by the tests; no download cache or partial generated model was
created. The current verified CPU build is about 48 MiB and the retained
precision description occupies about 8 KiB on disk.

Remaining capability boundaries: arbitrary per-layer mixed-format recipes and
independent deployable activation quantization are not implemented; actual
kernel-internal multiply/accumulate tracing remains future work. No new speedup,
quality qualification or GPU-stability conclusion follows from this delivery.
