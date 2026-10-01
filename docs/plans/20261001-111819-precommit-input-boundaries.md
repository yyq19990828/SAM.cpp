# Pre-commit input boundary corrections

Status: complete and verified for the authorized first local commit.

## Scope and evidence

The commit-readiness review found four reachable boundary mismatches. Custom
supplementary case IDs bypass the exporter's safe filename rule before the
validator opens logs. Dependency preparation resolves an output symlink before
recursively replacing that path. A checkpoint vector reshaped with a redundant
singleton dimension can be published as F16 although the runtime requires F32
for its canonical vector shape. Reference source preparation can stage a copy
inside the source subtree being copied when given a nested output path. The
accepted official checkpoint is unaffected.

This correction touches eight implementation/test files plus this plan,
the format documentation and changelog; it adds no dependency or public API.

## Approach and steps

1. Reuse one case-manifest validation rule for exporter and validator; validate
   unsafe/duplicate IDs before creating output or opening model payloads. Retain
   explicit supplementary authorization and the frozen official corpus gate.
2. Reject symbolic links at generated dependency destinations before REALPATH,
   including the temporary output name. Do not remove the links or their targets.
   Reject overlapping reference-source/output paths before creating staging or
   output parents, preserving the supplied source checkout.
3. Reject source-rank/storage combinations incompatible with SAM's canonical
   tensor contract before publication. Preserve the official conversion bytes,
   tokenizer, graph, backend policies and numerical tolerances.
4. Add focused regressions that first fail on the old implementation, then pass
   after correction. Use only disposable fixtures for destructive-path probes.
5. Refresh the affected tooling and CMake checks, verify original GGUFs remain
   admissible, and update the changelog before staging the completed baseline.

## Verification and recovery

Run the isolated Python tool suite and CPU/Metal CTest, including dependency
preparation. Preserve ordinary valid paths, tokenizer/precision roundtrip,
source checkout hashes, and existing-output refusal. Reuse the recent official
21-case evidence only after confirming unchanged graph/backend files and GGML
source/patch identities. Exclude weights and build artifacts from the commit.
No push, release, checkpoint rewrite or M2 implementation is included. Recovery
is a scoped source revert; disposable tests cannot reach user directories.

## Results

- Three focused Python regressions fail on the original code (exit 1), proving
  unsafe case IDs reach model access, an incompatible singleton model publishes,
  and nested/symlink-resolved outputs reach the copy step and create source
  parents. Disposable fixtures prevent actual recursive copying or user-file loss.
- The corrected isolated tool suite passes 10/10. Exporter and validator now
  share case validation; runtime-incompatible precision is rejected on readback
  before publication; resolved source/output overlap fails before staging.
- The original CMake test fails on disposable destination/temporary symlinks:
  its external target sentinel is deleted or its link removed. The new guard
  rejects both paths before REALPATH. The complete preparation regression passes,
  preserving links, target sentinels and the supplied source. Refreshed canonical
  CPU and Metal Release builds pass CTest 9/9 each (7.29 / 11.31 seconds).
- Original GGUF files and historical sidecars remain unchanged; current-tool
  admission checks pass both files with all 1,133 tensor records and seven
  original-reference cases checked. Model graphs and numerical gates are unchanged;
  the recent 21-case numerical evidence is reused, not rerun for these path guards.
- Logs and receipts are under `build/precommit-input-boundaries/` (tool red/green,
  official admission) and `build/precommit-boundaries/` (CMake red/green, builds,
  CTest and dependency identity). GGML revision, patch and source-tree hashes
  remain unchanged. An independent read-only pass confirms the reported issues
  are closed.
- Owned-source whitespace and documentation checks pass. The staged unified
  patch's single-space blank context lines are intentional and retained verbatim;
  its pinned SHA-256 still matches. No weights, credentials or build outputs are
  included in the commit.
