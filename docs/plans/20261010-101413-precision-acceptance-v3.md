# Precision acceptance v3 implementation

Created: 2026-10-10 10:14:13, Asia/Shanghai.

## Scope

Implement an explicitly selected v3 image acceptance policy and its complete
export, evaluation, campaign, regression and performance workflow. Preserve
the frozen v2 gate file, historical receipts and their original decisions.
Do not reclassify old final evaluations as new independent v3 acceptance.
No runtime/kernel, model, public cache default or hardware support change is
part of this work. Existing unrelated documentation deletions stay untouched.

## Approach

- Separate implementation correctness, task-quality eligibility, diagnostic
  fidelity and measured performance benefit. Missing arithmetic/regression
  evidence must remain missing; lack of a speed/memory benefit label must not
  turn otherwise qualified quality into a failure.
- Select a quality tier explicitly for each complete v3 recipe, independently
  of weight, compute and cache storage. Start with high-fidelity, balanced and
  compact engineering budgets. Preserve ranked instance AP, union-mask mIoU,
  paired image confidence bounds, size/category coverage, negative-prompt
  checks and same-parent cache comparisons.
- Keep score/box and tight reference-mask differences in diagnostic reports.
  Gate substantial mask degradation and actual unmatched instances separately.
  Protect well-localized high-confidence reference objects, using a reference
  GT-IoU margin above the detection matching threshold; report threshold-edge
  flips separately without removing them from AP or other task metrics.
- Keep reproducible paired latency/memory measurements and the existing large
  benefit thresholds as optional labels. Report measurement validity,
  resource/latency non-regression and benefit independently. Use explicit
  statuses rather than inferring whole-model acceptance from a quality report.
- Version gate/recipe/campaign/report identities end to end; retain legacy
  entry points and v2 behavior. A fresh v3 final campaign requires independently
  unused evaluation inputs and a declared exclusion history, frozen before
  inference. Development re-evaluation is diagnostic, not new final evidence.

## Steps

1. Trace existing gate consumers and provenance checks; define the versioned
   v3 contract and document exact initial thresholds and their limitations.
2. Implement opt-in v3 policy selection across exports, campaign freezing,
   quality scoring, fixed regressions and paired performance reports.
3. Add behavior tests for boundary flips, real object loss, diagnostic-only
   score/box changes, explicit tiers, cache budgets, v2 compatibility,
   incomplete/tampered evidence and benefit-free valid configurations.
4. Update bilingual validation/quantization guidance, tools usage, README and
   changelog with runnable commands and honest qualification scope.
5. Run the isolated reference Python suite, relevant CLI/integration checks,
   documentation checks and whitespace checks. Record results and limitations.
   Clean disposable environments/output after verification while retaining
   any required evidence and usable existing artifacts.

## Frozen v3 contract

The machine-readable policy is
[sam3-precision-gates-v3.json](../../tests/data/sam3-precision-gates-v3.json),
SHA-256 `822a617b90dcd5b45153b777d5e720272f234d38ae8908b3422d27357d842281`.
The existing v2 file remains SHA-256
`4cf06bdc609e5a143b2d74b42f394480bbe39e7510726ecedfe44fde3212a802`.
These are engineering starting budgets selected before any v3 model evaluation,
not thresholds fitted to turn old final failures into passes. Application owners
must select the tier before development/final inference and may require stricter
task-specific acceptance beyond the prompted COCO subset.

| Tier | AP drop max | mIoU drop max | Task mask IoU min | Bad-object rate max | New negative rate max |
| --- | ---: | ---: | ---: | ---: | ---: |
| high-fidelity | 0.0025 | 0.002 | 0.90 | 0.005 | 0.001 |
| balanced | 0.010 | 0.005 | 0.85 | 0.010 | 0.003 |
| compact | 0.020 | 0.010 | 0.80 | 0.020 | 0.005 |

Drops use fractions: `0.01` is one percentage point. The AP/mIoU and rate
budgets reuse conservative F16/Q6/Q4 task budgets as explicit application
targets. They no longer follow the model's smallest dtype. The wider task-mask
floors distinguish substantial object degradation from small fidelity changes;
the aggregate and image-bootstrap AP/mIoU budgets still constrain their impact.
A tier name is a target, never an inference of quality from storage size.

All tiers retain ranked mask AP at IoU .50:.05:.95 (`maxDets=100` without a
deployment cutoff), positive image/prompt union mIoU at score > .5, 2,000
paired-image bootstrap replicates and a 95% upper bound, at least 1,024 final
images and 1,000 high-confidence reference objects. Size AP budgets are twice
the tier AP budget with minimum 50 images / 200 instances; category budgets
are max(twice tier AP budget, .01) with minimum 20 positive pairs / 50 instances.
Insufficient size/object coverage is INCONCLUSIVE; insufficient category
coverage is NOT_APPLICABLE and cannot support a per-category claim.

Spatial matching remains one-to-one with IoU >= .5. A counted matched pair
(either score >= .6) is a bad task object only when its task-mask floor fails;
unmatched high reference objects and unmatched high candidate objects also
count. Missing/extra rates each receive half the bad-object rate budget.
Seven fixed regressions allow zero task failures and no new detections on an
empty reference. Negative-prompt regressions cannot cancel across prompts.
Their extra-detection-count budget is twice the new-negative-pair rate budget.

Protected reference objects have score >= .9, area >= 1,024 pixels and a
one-to-one GT match at IoU >= .75. Every such GT must remain matched by a
candidate detection at IoU >= .5; zero protected misses. The .75 reference
margin prevents a barely valid original .50 match from creating a zero-tail
veto for a tiny change. All GT effects remain in AP/mIoU. The older .50
reference protection test is also reported as `gt_threshold_losses` for
diagnosis, including any severe losses; it is not removed from the report.

Tighter reference-mask IoUs (.95/.94/.90), score errors (.02/.03/.05) and
normalized box errors (.01/.015/.03) are diagnostic limits for the three tiers.
Object reports provide `task_passed` and `fidelity_passed`; their retained
`passed` field is the legacy fidelity result, not the v3 task decision.
The quality aggregate uses `task_passed`. A score change causing loss of an
actual deployment detection still affects task quality and mIoU.

Compressed caches must pass both their tier's absolute quality and a separate
comparison with exactly the same recipe using F32 cache:

| Cache | Incremental AP drop max | Incremental mIoU drop max | Task mask IoU min | Bad-object rate max |
| --- | ---: | ---: | ---: | ---: |
| f16 | .001 | .0005 | .95 | .0025 |
| mixed-q8_0 / all-q8_0 | .0025 | .001 | .90 | .005 |

These policy entries do not add runtime support. The existing CLI permits F32,
F16 and mixed Q8 cache choices; all-Q8 and activation prototypes remain outside
its runnable choices. Arithmetic correctness is unchanged: codec, operand,
operator, layout, finiteness, placement and complete graph evidence must be
bound to the actual recipe/build. A correct isolated dot product is not a
complete model qualification. Intermediate reference-tensor differences remain
diagnostic, and task tolerance cannot waive an arithmetic failure.

## Independent performance decisions

Quality-only campaigns need no benchmark executable, performance cases or
performance baseline. Compressed caches independently declare `cache_baseline`.
When performance is planned, freeze `performance_baseline`, cases and executable
before final inference. Both native recipes must pass final quality under the
same declared tier and backend. A failed/inconclusive baseline cannot lend a
benefit label to another recipe. The runner remains gated on passing final
quality; profiling during development uses the existing diagnostic tools.

Retain eight annotation-selected cases, three independent AB/BA/AB process
pairs, five warmups and twenty measured iterations. Latency and memory use
separate processes. Pooled statistics and every process pair must satisfy the
limits; an unstable crossing or observed interference is INCONCLUSIVE. Report
full-image, changed-prompt and repeated-result workloads separately.

- `measurement_status`: complete, uncontaminated observation or INCONCLUSIVE.
  Missing/malformed records are rejected, not filled in.
- `non_regression_status`: P50/P95 ratios <= 1.03, each case P95 <= 1.05,
  GPU process peak <= 1.03 and RSS peak <= 1.05; GPU metrics are inapplicable
  on CPU. This informs deployment choice separately from task quality.
- `benefit_status` and labels: latency P50 <= .90; GPU-memory <= .85;
  host-memory <= .85; combined latency/GPU-memory <= .90 for both. Existing
  tail and other-resource limits apply. A valid measurement without a label
  is NOT_DEMONSTRATED, not a task-quality failure. Smaller gains remain visible
  in measured ratios and milliseconds.

Qualification combines mandatory arithmetic, absolute quality, applicable
incremental quality and fixed regression evidence. Benefits are optional.
The quality, regression and performance runners each report their own evidence;
they deliberately do not synthesize a whole-model PASS without an independent
arithmetic adjudication. Missing whole-model evidence stays NOT_RUN. Known
quality/regression failures give FAIL. The status-combination helper must never
be interpreted as a verifier of files merely because strings were supplied.

Optimization policy: time-box each concrete bottleneck hypothesis, record
end-to-end measurements and archive an attempt after two independent checks
show no reproducible workload gain and profiling supplies no new explanation.
Resume only with a new mechanism or workload requirement. Do not spend repeated
weeks chasing the optional 10%/15% labels, and do not tune on final/reserve data.
This is a review practice; no timer or scheduler is introduced by these tools.

## Versioned workflow

Recipes/manifests, frozen selections/campaigns, quality decisions, native fixed
regression reports and performance reports use version 3 with the pinned gate
hash. Original ranked payloads, normalized inputs and dataset structure retain
version 2; the fixed original golden bundle also stays v2 as immutable raw
reference data. The evaluator rejects mixed-policy run manifests. Default CLI
selection remains v2 for compatibility; v3 always requires an explicit tier.

Use new directories and finish source/tool edits before freezing. Recover all
previous precision dataset manifests, including consumed follow-up final sets.
The exclusion history includes their previously used IDs/content and every
non-reserve split. Prior reserve remains closed and can be retained as reserve.
Declare all earlier v3 trials too. `complete_declaration: true` is an operator
assertion, not proof that no unrecorded external inference occurred. A source,
tier or recipe change after a frozen final attempt requires a new campaign and
fresh held-out inputs. Re-running development inputs is permitted.

Prepare a fresh dataset with the existing COCO selectors, using every prior
exclusion and verifying the selected files. `prepare_coco_fresh_holdout.py`
supports annotation-only train2017 selection plus content verification; inspect
its `select`/`finalize` help. Use the resulting merged annotation file matching
the dataset manifest. Never relabel old final outputs as fresh v3 acceptance.

Example commands below assume the declared dataset, source, BPE, converted
models and built probes already exist. They illustrate future real inference;
they were not run on final model data during this tool implementation.

```sh
.venv-reference/bin/python tools/validation/prepare_precision_inputs.py \
  --dataset build/v3/dataset.json --input-root models/coco \
  --phase development --output build/v3/inputs-dev
.venv-reference/bin/python tools/validation/export_precision_outputs.py \
  --policy-version 3 --quality-tier balanced --engine original \
  --dataset build/v3/dataset.json --inputs build/v3/inputs-dev --phase development \
  --sam3-source "$SAM3_SOURCE_DIR" --sam3-runtime-source build/reference-runtime/sam3-cuda \
  --checkpoint models/original/sam3.pt --bpe "$sam3_weights_dir/bpe_simple_vocab_16e6.txt.gz" \
  --output build/v3/original-dev
.venv-reference/bin/python tools/validation/export_precision_outputs.py \
  --policy-version 3 --quality-tier balanced --engine native \
  --dataset build/v3/dataset.json --inputs build/v3/inputs-dev --phase development \
  --binary build/cuda/examples/sam_precision_image_probe --model models/sam3-f32.gguf \
  --backend cuda --compute f32 --cache f32 --output build/v3/f32-dev
.venv-reference/bin/python tools/validation/evaluate_precision.py \
  --reference build/v3/original-dev --candidate build/v3/f32-dev \
  --annotations models/coco/annotations-v3.json --output build/v3/f32-dev-quality
```

Repeat the native export/evaluation for each candidate. A compressed-cache
evaluation additionally passes `--baseline` pointing to its same-tier F32-cache
parent's outputs. Regenerate development metrics after evaluator/source edits.
The native export target is `sam_precision_image_probe`; the performance target
is `sam_precision_benchmark_probe` (see `tools/CMakeLists.txt`).

A quality-only selection (replace the absolute paths and hash with actual
identities; add every historical dataset, not only the example entry):

```json
{
  "schema_version": 3,
  "kind": "sam3-precision-selection-v3",
  "evaluation_history": {
    "complete_declaration": true,
    "datasets": {"/absolute/history/dataset.json": "<sha256>"}
  },
  "runs": [
    {"id": "original", "development_run": "/absolute/v3/original-dev"},
    {"id": "f32", "development_run": "/absolute/v3/f32-dev",
     "development_quality": "/absolute/v3/f32-dev-quality/metrics.json"},
    {"id": "mixed", "development_run": "/absolute/v3/mixed-dev",
     "development_quality": "/absolute/v3/mixed-dev-quality/metrics.json",
     "cache_baseline": "f32"}
  ]
}
```

If comparing speed/memory, prepare cases with the command below, add top-level
`performance_cases` and `benchmark_binary` absolute paths, and set the mixed
row's `performance_baseline` to `f32`. Keep that baseline at the same tier.

```sh
.venv-reference/bin/python tools/benchmark/prepare_precision_performance.py \
  --policy-version 3 --dataset build/v3/dataset.json --inputs build/v3/inputs-dev \
  --annotations models/coco/annotations-v3.json --output build/v3/performance-cases.json
.venv-reference/bin/python tools/maintenance/freeze_precision_campaign.py \
  --selection build/v3/selection.json --output build/v3/campaign.json
.venv-reference/bin/python tools/validation/prepare_precision_inputs.py \
  --dataset build/v3/dataset.json --input-root models/coco \
  --phase evaluation --output build/v3/inputs-final
```

Export every frozen recipe again with `--phase evaluation`, `--inputs
build/v3/inputs-final`, `--campaign build/v3/campaign.json` and a fresh output
directory. Evaluation uses those final runs, still with the original reference
and applicable cache baseline. Final export exclusively claims one attempt per
recipe; keep the claim even when inference fails. Fixed regressions can use the
existing verified seven-case original ranked bundle:

```sh
.venv-reference/bin/python tools/validation/validate_precision_regression.py run \
  --policy-version 3 --quality-tier balanced \
  --binary build/cuda/examples/sam_precision_image_probe --model models/sam3-f32.gguf \
  --backend cuda --compute f32 --cache f32 \
  --reference models/reference/fixed-ranked-original --output build/v3/f32-fixed
.venv-reference/bin/python tools/benchmark/benchmark_precision.py \
  --campaign build/v3/campaign.json --candidate mixed \
  --candidate-quality build/v3/mixed-final-quality/metrics.json \
  --baseline-quality build/v3/f32-final-quality/metrics.json \
  --output build/v3/mixed-performance
```

For a compressed-cache fixed regression, add its `--cache` choice and
`--baseline build/v3/f32-fixed`. The optional benchmark command requires the
performance settings described above; it is intentionally unavailable for a
campaign that froze quality alone. This example does not replace the separate
whole-graph arithmetic evidence or supply missing original golden data.

## Verification

Use synthetic ranked-output/COCO fixtures for meaningful policy and CLI
integration coverage; do not consume final/reserve model data. Check the v2
gate/receipt hashes before and after. No CUDA/model qualification is claimed
by unit tests or tool delivery. This checkout currently has no `build/`
evidence or reference environment; inspect available runtimes and prepare an
isolated environment for required tool tests if necessary.

## Results

Implemented opt-in v3 throughout policy selection, ranked exports, evaluation,
campaign freezing, fixed regressions and paired performance. Updated the English
and Chinese guides, README, tools guide and changelog. No C++ inference/kernel or
public cache behavior changed. Kept unrelated PRD/TECH_SPEC deletions and the
concurrent platform-benchmark documentation changes intact.

Verification completed:

- Isolated Python 3.12.9 environment with Torch 2.10.0+cpu, NumPy 1.26.4,
  Pillow 11.2.1, pycocotools 2.0.10, GGUF 0.19.0, regex 2024.11.6 and ftfy
  6.1.1. This CPU tool-test environment is not the locked CUDA oracle runtime.
- `tools/test_tools.py`: 188 tests in 49.343 seconds, OK, 1 skipped because
  CUDA is unavailable to this CPU Torch environment (187 executed successfully).
  Includes 16 new v3 policy/provenance/integration tests. The preceding 171-test
  compatibility run also passed with the same CUDA skip. The current total also
  includes the concurrent documentation test addition.
- New coverage includes actual evaluator CLI execution on synthetic COCO/ranked
  files, explicit tiers independent of dtype, separate cache budgets, diagnostic
  score/box/tight-mask changes, GT-boundary flips, real protected/missing/extra
  objects, strict fixed regressions, incomplete evidence, performance without
  benefit, resource regressions/interference, immutable history exclusions,
  policy/tier tampering and a quality-only frozen campaign. Synthetic data is
  deliberately insufficient for full qualification and reports that limitation.
- All 12 grouped/legacy CLI help invocations pass from outside the checkout,
  including the `validate_precision_regression.py run` subcommand. Confirmed
  native probe names against CMake declarations.
- Byte comparisons against HEAD confirm the v2 gate and all four tracked v2
  JSON receipts are unchanged. Current v2/v3 gate hashes match their pinned
  constants. No historical receipt was regraded or overwritten.
- `git diff --check` passes. The unmodified full documentation check remains
  blocked by two pre-existing deleted targets, `docs/PRD.md` and
  `docs/TECH_SPEC.md`, linked by the older compiled-library plan. An exhaustive
  scan of 138 Markdown documents / 502 local links found exactly those two
  missing links and no other missing or Git-ignored links; the bilingual
  measurement tables match. The deleted files were not restored to hide this.
- No C++ build, CUDA arithmetic run, original-model inference or fresh v3 final
  campaign was performed. This delivers verified acceptance tools and policy,
  not a new model or hardware qualification.

Retained local verification evidence under `build/precision-v3-tools/`:
`tool-tests.log`, `historical-integrity.json`, `cli-checks.json`,
`documentation-checks.json` and `verification.json`. Disposable test directories
were cleaned by the tests; retained the usable isolated tool environment and
verified original checkpoint. No obsolete build or duplicate model was created.

Resources checked with the user's supplied locations:

- Docker volume `ai-annotation-platform_sam3_checkpoints` contains `sam3.pt`
  and a separate SAM 3.1 multiplex file. Copied only SAM 3 to
  `models/original/sam3.pt` (3,450,062,241 bytes), with source/copy SHA-256
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
  It matches the checkpoint identity recorded by the historical
  [mixed-cache plan](20261009-223907-shortdot-mixed-cache-final-acceptance.md).
  The local `models/original/sam3.pt.source.json` records copy integrity;
  upstream download revision/authentication was not independently reconstructed.
- SSH host `工作站2` exposes
  `/media/hehao/data1/yiqing/dataset/0_公开数据集/coco2017`, with `train2017`,
  `val2017`, and both instance annotation files (449 MiB / 20 MiB as listed).
  This is availability evidence; the full dataset was not transferred, decoded
  or used for inference. Prior final-usage manifests still need to be restored
  before any new real v3 campaign can assert a complete exclusion history.

These resources and all `build/`, `models/`, `.venv-reference/` paths in this
plan are local, Git-ignored artifacts and are unavailable in a clean checkout.
No final/reserve image was used by the tool tests.
