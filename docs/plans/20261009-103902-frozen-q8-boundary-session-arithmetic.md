# Frozen custom-Q8 boundary and session arithmetic

Created: 2026-10-09 10:39:02, Asia/Shanghai.

## Scope

Close, where feasible, the specific whole-recipe arithmetic gaps identified
by the eight-input scope audit for the frozen SAM 3 CUDA text-image
`text,fusion,decoder` Q8_0 allocation. Keep the campaign, GGUF, source and
binary/library identities and every numerical/quality/performance gate
unchanged. Do not use final-evaluation or unopened-reserve inputs to select
arithmetic cases. A concurrent exclusive-GPU benchmark has priority; launch
no diagnostic GPU process until it completes and its exclusivity is checked.

## Approach and steps

1. Predeclare a small input/path matrix from calibration/development data:
   a fresh-image prompt, a changed prompt on the same image, a repeated
   prompt/cache hit, representative small/large source geometry and
   token-length boundaries. Establish which cases share the existing graph
   topology and which execute a different node set. Reuse an existing
   audited node class only after checking exact shape, type, layout and
   backend placement; independently verify any newly active node.
2. Exercise the exact frozen GGML CUDA Q8_0 MMVQ and MMQ staging/dot paths
   with fixed zero, positive/negative tie, tail, outlier/saturation, maximum-K
   and representable-limit fixtures. Define expected behavior for F16 scale
   and sum overflow before running the native probe; distinguish invalid
   out-of-domain inputs from an accepted inference result. Check raw packed
   bytes, decoded scales, zero/nonfinite values and same-operand F64 dots.
   Pair Q8 dot cases with explicit bias/GELU or corresponding graph-epilogue
   checks at their boundaries.
3. Reconcile session outputs, token IDs, preprocessing/postprocessing,
   graph coverage and retained payload identities against the frozen native
   recipe. Do not elevate `whole_recipe_arithmetic_status` unless every
   required item has direct evidence; otherwise retain `NOT_RUN` and give
   the exact remaining failure or untested path.

The session-case choice is fixed before this plan's GPU runs:
`perf-source-small` uses development image `coco-000000001503` (320 x 240,
frozen PNG SHA-256
`ce22732ea8434e1a7ab4e292daf95381d0e50ab56f68f3fe14e0c5ca14438116`)
with `tv`, then its predeclared alternate `teddy bear`, then `teddy bear`
again. `perf-source-large` uses development image `coco-000000110884`
(640 x 640, SHA-256
`02ccc24021124db0395f49f0004f10c5faf70f09106954cd7c91db46d628ebee`)
with its predeclared `bowl`/`horse` pair. The fixed tokenizer boundary
prompts are empty/whitespace and a 33-times repeated `truck` phrase,
corresponding to the existing official-tokenizer goldens; malformed token
vectors are rejection checks, not candidate inference inputs. No case may
be swapped after seeing an arithmetic result.

The remaining Q8 staging fixtures are also fixed before execution. MMVQ
uses a 2 x 128 F32 input with a zero block, separate +2,000 and -2,000
outliers, and separate +65,504 and -65,504 singletons so F16 block sums
stay representable; all non-outlier positions are deterministic small
values. MMQ uses 65 x 160 F32 values (512-wide native padding), including
zero blocks, separate ±3e38 scale limits and ±2,000 outliers. The expected
valid outcome is finite stored scale/sum, signed endpoints within [-127,127],
independent staging PASS and no unclaimed tie allowance. A separate MMVQ
1 x 32 input has two +40,000 values: its mathematical block sum 80,000
exceeds F16. The independent validator **must reject** any resulting
nonfinite stored sum; this negative fixture is not counted as a passing
in-domain inference. The retained frozen-library `mlp-lin1` MMVQ/MMQ raw
dot plus F32 bias/GELU outputs are the epilogue cases, with the unchanged
2e-5 relative-L2 and 1e-6 zero-norm rules.

After the predeclared ±3e38 MMQ fixture produced nonfinite native scales,
the diagnostic-only follow-up is fixed to maximum magnitudes 1e30, 1e35,
1e37 and 1e38 in the same 65 x 160 shape and column locations. This sweep
will locate the overflow range and test the CUDA source-level explanation;
it will not replace the failed ±3e38 boundary or enter a passing acceptance
aggregate.

As a further post-failure diagnostic, scan the retained F32 RHS captures of
all eight predeclared real-image Q8 graphs for finite values and their largest
absolute magnitude. Rehash each bound payload while reading it. This
describes only observed activations; it cannot prove a bound for arbitrary
accepted images or prompts, and cannot turn the failed MMQ boundary into a
passing gate.

## Verification

Use only fresh Git-ignored output paths and the existing isolated reference
environment. Rehash frozen and new evidence, run relevant CUDA CTests,
`tools/test_tools.py`, the documentation/link checker and `git diff --check`
after changes. Store large captures on the separate evidence volume with a
space preflight. Preserve immutable receipts, original checkpoints, usable
GGUF and verified build; remove only disposable trial intermediates.

## Results

The separate formal benchmark finished before these diagnostic GPU probes.
It reports an uncontaminated GPU-memory optimization label for full-image
and changed-prompt inference; see the [exclusive performance plan](20261009-100528-exclusive-gpu-performance-readiness.md).
The frozen campaign/model/probe/CUDA-library SHA-256 identities remained
`0d7e79b83440b8b643ea89a06f65740e90aee4e904bece87c97bb152724d9a6d`,
`f854afae5a2eb2c388ff2dd85804a4327b4525b45a09c1fdc9b5f5619c93142e`,
`a2980263d888d21835b3644032311deabe4124d678d2042041b7c8794dd72a53`
and `1e4f954340ea2c30cb60deb49f4ccda6780993b2e09cb81bc8105aa053c925ef`.

The frozen image-session probe completed seven ordered calls in one process
and two alternate prompts in fresh processes. The combined calls covered
small and large source images, changed and repeated prompts, whitespace and
a long tokenizer boundary. Each output had 200 finite scores, 200 finite
boxes and a 32-token context. Exact non-runtime outputs matched prior
development exports, fresh alternate calls and repeated-call outputs. On
the small image, the vision build count stayed at one across changed and
repeated prompts; each changed prompt added 1,900 CUDA text/inference nodes,
while the repeated prompt added none. All calls remained on CUDA. The
read-only session auditor rehashed 22 files and passed; its ignored receipt
is `build/q8-integrated-arithmetic-20261009/session-boundary-audit.json`
(SHA-256 `0fadaf724b71bb1148fa76cc663d4579318fc6c741dcc47ae1cef32c7c7b0d37`).
This proves output/cache behavior on those calls, not an independent
same-operand audit of every newly active node.

The retained frozen-library `mlp-lin1` raw Q8 dots, already checked against
packed operands, were independently combined with the exact F32 bias and
GELU epilogue. Both MMVQ and MMQ outputs passed the unchanged 2e-5
relative-L2 gate: 317,312 outputs in all, worst relative L2
`2.8295821246e-7`. The ignored receipt
`build/q8-integrated-arithmetic-20261009/q8-epilogue-verification.json`
has SHA-256 `e2cadf80de53b112b0d45b53fbe3de6a689590abc5c478d7c7b37f07905d326c`.

The predeclared native Q8 boundary has a **failure**. The 2 x 128 MMVQ
finite/signed-outlier case passed all eight independently checked blocks.
The separate two-times-40,000 MMVQ block produced an F16 sum of `+inf` and
was correctly rejected as the expected negative case. But the finite
65 x 160 MMQ input with ±3e38 outliers produced 130 nonfinite stored
scales; the independent validator rejected it. The native RHS and failing
receipt remain unchanged. The post-failure diagnostic sweep of the same
shape passed at 1e30, 1e35 and 1e37, but produced 130 nonfinite scales at
1e38. These sweep passes do not replace the predeclared failure. The
ignored `build/q8-integrated-arithmetic-20261009/q8-boundary-audit.json`
(SHA-256 `c0343a5e9b694b6af03c8a5ca597cd7fe502c1e602fdf7dfbed206527d74fd42`)
rehashes native source, frozen probes, library, inputs and payloads, and
records `predeclared_boundary_status: FAIL` and
`whole_recipe_arithmetic_status: NOT_RUN`.

Frozen GGML `quantize.cu` computes `d_inv = 127.0f / amax` then
`d = 1.0f / d_inv`; the CUDA build enables fast math. Reciprocal
underflow/flush-to-zero during this division is **an inference** consistent
with the observed 1e37-to-1e38 transition, not a proven cause from the
source and results alone. No frozen source, library, model or gate was
patched to make this candidate pass.
An ensuing [isolated division diagnostic](20261009-111944-cuda-q8-mmq-extreme-scale-diagnosis.md)
reproduced inverse zero and infinite scale with the same GPU/compiler mode,
then passed the original extreme fixture with an explicitly rounded divide
in a separately compiled unit. This strengthens the causal explanation but
does not change this frozen candidate's failed boundary result.

The post-failure read-only scan rehashed all 2,616 retained Q8 RHS F32
payloads from the eight predeclared single-prompt images: 1,626,886,144
values (6,507,544,576 bytes), zero nonfinite values, and maximum absolute
value `105.8585663`. The ignored receipt
`build/q8-integrated-arithmetic-20261009/observed-q8-rhs-range.json`
(SHA-256 `cc9b13aa6aa69971bcca17d9edddfa2cd46be5540178e9f58243bb9b727dba94`)
describes only those observed activations, not a bound for every accepted
public input.

**Decision:** the eight-input active-graph arithmetic `PASS`, session output
parity and epilogue `PASS` remain bounded component evidence. The extreme
MMQ boundary is `FAIL`, and changed-prompt/cache paths still lack full
independent arithmetic capture. The frozen candidate's
`whole_recipe_arithmetic_status` and `deployment_status` remain `NOT_RUN`;
the quality and exclusive-GPU performance decisions are unchanged. A future
corrected CUDA implementation would need a separately identified recipe,
new boundary checks and fresh integrated quality/arithmetic/performance
evidence before deployment could be considered.

Verification after the diagnostic passed CUDA CTest 29/29, isolated
`tools/test_tools.py` 170/170, bilingual/local-link checks on 114 documents
and `git diff --check`. The remote-desktop service stayed inactive and no
other NVIDIA compute process was present. Temporary diagnostic compiler
objects were removed; the original GGUF, verified build and bound evidence
were retained.
