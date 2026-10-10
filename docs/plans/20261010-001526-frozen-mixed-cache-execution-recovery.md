# Frozen mixed-cache execution recovery

Created: 2026-10-10 00:15:26, Asia/Shanghai.

## Scope and approach

Continue the [mixed-cache final acceptance](20261009-223907-shortdot-mixed-cache-final-acceptance.md)
without changing its frozen five recipes, v2 gates, source closure or unused
1,024-image final split. Prepare an ignored execution sidecar that can run the
remaining exports, final quality and paired performance on the bound RTX 4090
environment. Preserve the outstanding whole-recipe arithmetic and regression
requirements; a successful batch must not imply full deployment acceptance.

The current kernel exposes NVIDIA driver version 610.57.04, but this sandbox
has no `/dev/nvidia*` device nodes and `nvidia-smi` exits 9. Do not change host
drivers, device permissions or the frozen source tree to work around that
environment boundary. Continue the work that can be verified offline.

## Steps

1. Verify the frozen campaign, complete final inputs, annotation identity and
   existing one-attempt ledger before constructing executable commands.
2. Add a sidecar under ignored `build/mixed-cache-final-20261009/` for offline
   preflight, serial GPU export, resumable completed quality stages and paired
   performance. Invoke the existing frozen tools rather than replacing their
   numerical or statistical checks.
3. Require exact recipe/environment equality and an idle GPU before any new
   final inference. Reuse only completed, hash-verified artifacts; preserve and
   reject incomplete or previously claimed evaluation attempts.
4. Verify offline command coverage for all five exports, four quality reports
   and three non-self performance comparisons. Exercise tampered-campaign and
   missing-device failures, proving that they create no evaluation claims.
5. Record the execution readiness receipt, remaining gates, documentation
   checks and cleanup results. Continue final GPU work if the device becomes
   visible; otherwise keep the overall goal active.

## Verification

Use the isolated reference environment through `rtk proxy`. Verify all bound
artifact hashes and the normalized final image inventory. Check completed
output payload hashes before skipping stages. Do not write production sources
or gate files while the campaign is frozen. Use fresh paths for verification
receipts, and run the documentation checker and `git diff --check` afterward.

## Results

The ignored `build/mixed-cache-final-20261009/run_frozen_acceptance.py`
sidecar now covers the exact five final exports, four native quality reports
and three non-self performance comparisons. It invokes the frozen grouped
tools, validates the original checkpoint and same-weight cache parents,
keeps GPU exports serial, and skips only completed artifacts after source and
payload hash verification. Existing one-time claims and incomplete output
directories stop execution for inspection. Performance still requires passing
final quality through the existing runner; stage completion leaves deployment
`NOT_RUN` while whole-recipe arithmetic and regression are outstanding.
This sidecar and its receipts are ignored local artifacts, unavailable in a
clean checkout.

The offline preflight rehashed the frozen campaign identity closure and all
1,024 normalized final images, confirmed 4,729 prompts and 1,024 unopened
reserve images, and verified the reference environment packages. Its GPU
observation confirms loaded kernel driver 610.57.04, zero visible NVIDIA
device nodes and `nvidia-smi` exit 9. The command inventory is complete and
the final evaluation ledger is absent.

`verify_frozen_runner.py` exercised three actual subprocess failures: a
modified campaign is rejected before inference, a missing CUDA device creates
no final output/claim, and an incomplete export directory is preserved and
rejected. It also checked full command coverage, full-split export arguments,
both cache baseline dependencies, and absence of reserve inference arguments.
All checks passed, and the evaluation ledger was unchanged. No GPU inference
was started by these checks.

SHA-256 identities of the local runner, preflight receipt and boundary-check
receipt are respectively:

- `20550aa175a9908982d9a92a240baf4150acf6dd762eae0d7e5e4a5e88af36f6`
- `f020db5fc6cafb28b421e3b4d19fbf03eccd891605d7d1c14266694012449e30`
- `7814cdf48e411faab2e84763f40bb4c2570e4984344a2cf0be85371d959b35eb`

Once this same frozen CUDA environment is visible, continue with:

```sh
rtk proxy .venv-reference/bin/python -B build/mixed-cache-final-20261009/run_frozen_acceptance.py --stage quality
rtk proxy .venv-reference/bin/python -B build/mixed-cache-final-20261009/run_frozen_acceptance.py --stage performance
```

The second command requires all relevant final quality reports to pass.
Final quality, paired performance, candidate-bound whole-recipe arithmetic,
regression and final adjudication remain open. Frozen sources and gates were
not edited.

After the local environment's permissions were restored, the same RTX 4090
and driver 610.57.04 became visible with no competing compute process. The
exact frozen quality runner started successfully. Its official checkpoint
reference export completed **1,024 images / 4,729 prompts**; the four native
exports and subsequent quality scoring continue in the same serial runner.
The isolated Python regression passed **171/171** in 40.623 seconds. A separate
ignored queue waits for all five complete final manifests before beginning
the development-only [mixed-cache session arithmetic](20261010-005446-mixed-cache-session-arithmetic.md).
These are execution milestones, not final precision or deployment verdicts.
