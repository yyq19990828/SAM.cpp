# Preserve verified GGML archive upgrades

## Scope

Commit preparation found that the new CUDA patches reject the verified combined source archive accepted by parent commit `174cd650f4d966cdd85b584097702aa4a914afbd`. Preserve that exact input state without broadening source provenance acceptance or changing the final GGML source tree.

## Approach and steps

1. Reconstruct the historical combined tree by reversing the short-dot patch and the appended inverse-scale hunk; assert its known full-tree hash.
2. Prove the current preparer rejects that historical archive before the fix.
3. Accept the exact historical hash and apply only the missing inverse-scale hunk in the build-local copy before the short-dot patch. Verify the same final combined hash and unchanged caller input.
4. Exercise original, Metal-only, historical combined, rounded-scale combined and current combined inputs, including repeated preparation and existing path/symlink rejection tests.

## Verification

Run the preparation regression red/green, isolated Release CPU build/CTest, isolated reference Python tools, documentation and whitespace checks. Reuse the frozen CUDA numerical and recipe evidence only for its unchanged kernel source and tested build; no recipe acceptance is inferred for a new binary. Retain immutable evidence and clean only this verification run's disposable files.

## Results

The historical fixture reconstructs exactly `37f787b8a432f2399e9ee50270f8d025378c605ae23f5efaae6a6ae9148fd5cb`. Before the fix, preparation failed on that identity (exit 1). After the fix, the complete preparation regression passed from both the current combined tree and the reconstructed original tree (exit 0). Original, Metal-only, historical combined, rounded-scale combined and current combined inputs converge to the same pinned final tree; repeated historical preparation preserves the caller input. Existing overlap, symlink and dirty-source rejection checks also pass.

An isolated worktree at parent `174cd650f4d966cdd85b584097702aa4a914afbd` plus the complete intended patch passed:

- Release CPU configure/build, with Metal/CUDA/BLAS/native tuning disabled.
- CTest: 17/17, including the preparation regression and SDK consumer checks.
- `tools/test_tools.py` in the isolated reference environment: 171/171.
- Documentation: bilingual tables and local links in 134 documents; whitespace checks passed.
- Five additional diagnostic/preparation CLI `--help` invocations passed outside the source working directory.

Read-only security and architecture review, plus assumption, composition, cascade and abuse passes, found the historical archive regression; the fix was re-reviewed with no remaining findings.

Logs and check receipts are retained under `build/ship-check-20261010/` (Git-ignored and available only in the local verification workspace), including `preparation-red.log`, `preparation-green.log`, `preparation-original-green.log`, `checks.json` and `extra-cli-smoke.json`.

The existing frozen CUDA 31/31 test receipt and mixed-cache acceptance artifacts were reused for their recorded build only. Their evidence identities were rechecked unchanged. Of the 250 frozen producer files, only `cmake/prepare_ggml.cmake` and its preparation regression differ after this shipping fix; CUDA patches, C++ runtime and arithmetic oracles remain unchanged. The new preparer produces the same verified GGML content hash, but no new binary is assigned the frozen recipe's acceptance. Historical receipts remain immutable.
