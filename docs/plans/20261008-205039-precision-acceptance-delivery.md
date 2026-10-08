# Precision acceptance v2 delivery

Created: 2026-10-08 20:50:39, Asia/Shanghai.

## Scope and approach

Commit the completed [v2 acceptance work](20261007-200954-precision-acceptance-gates-v2.md)
and merge the current branch into local `main`, as requested. Include the existing
`AGENTS.md` cleanup rule. The starting HEAD is `09c2c4a`; local `main` is
`0aa4971` and is an ancestor, so delivery can use a fast-forward merge. No remote
publication is included in this request.

The delivery review found that Quick checks does not install `pycocotools`, while
the registered COCO tests require it. Add version `2.0.10`, already pinned in both
reference lockfiles. No SciPy import exists in the tool test graph, so it does not
need an additional CI dependency. Also document the Linux requirement of the v2
Python native runners: their library inspection uses `ldd` and their process
memory observer uses Linux interfaces. Metal host integration for these wrappers
is not implemented; this does not change the existing runtime support table.

## Steps and verification

1. Reconcile the intended tracked/untracked files and verify the 139 frozen final
   source identities. Keep completed quality/performance receipts immutable.
2. Add the missing CI dependency and clarify the runner host requirement in both
   quantization guides. Do not change runtime, evaluator, thresholds or recipes.
3. Check the COCO test with the missing dependency reproduced and with the pinned
   installed dependency; run the full tool suite in `.venv-reference`, CLI help,
   documentation and whitespace checks. Reuse the matching CUDA 29/29 and CPU
   16/16 build evidence because compiled sources are unchanged.
4. Re-read HEAD and the staged tree, commit only the intended files, switch to
   `main`, merge with `--ff-only`, and verify branch tips and clean worktree.

## Results

The 139 frozen final source identities match. The dependency reproduction makes
the existing COCO test raise `ModuleNotFoundError` before assertions, confirming a
CI environment omission. Subsequent checks and delivery results are recorded below.

- Added only `pycocotools==2.0.10` to the CI install command, matching the existing
  lockfiles, and documented the v2 Python runner host boundary in both guides.
- `.venv-reference/bin/python -B tools/test_tools.py`: 144 tests passed in 35.940 s.
  The controlled missing-dependency reproduction failed before assertions as
  expected; the installed pinned dependency allows the full suite to complete.
- All nine new Python CLI `--help` commands returned exit code 0 with usage text.
- Documentation checks passed for 66 documents, including bilingual tables and
  local links. Existing CPU/CUDA build results are reused with frozen source
  identities checked, rather than represented as new builds.
- The bounded assumptions/security, composition/architecture, cascade and
  repeated-use reviews found no remaining blocker after the CI and host-scope
  corrections. No production source, gate, model or immutable receipt changed.
- Commit and fast-forward integration are recorded in Git history; final branch
  tips and worktree cleanliness are checked after those operations.
