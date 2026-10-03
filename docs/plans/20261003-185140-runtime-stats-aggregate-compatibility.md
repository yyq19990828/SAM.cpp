# RuntimeStats aggregate compatibility before commit

Created: 2026-10-03T18:51:40.135674+08:00.
Baseline: main56a4cde plus the accepted CPU BLAS/ViT and P1/P2 candidate.
Status: complete, verified before local commit. No push authorized.

## Scope and approach

The new blas_nodes member currently precedes existing public RuntimeStats fields.
A valid legacy13-field aggregate initializer fails narrowing at its old timing
positions. Append the new member so existing positional fields retain meaning.
Keep backend scheduling, model graphs, weights, precision and preprocessing unchanged.
Also restore the archive verifier's promised complete inventory/byte-total checks:
reject files outside the recorded tree and inconsistent aggregate sizes. Copying,
qualification, model validation and performance recipes remain unchanged.

## Steps and verification

1. Reproduce baseline success and candidate compile failure in an isolated probe.
2. Add the legacy initializer/value checks to ordinary CTest, move the field to
   the end, and observe RED/GREEN rather than assuming compatibility.
3. Build/test the complete candidate from a clean base+patch source tree with
   pinned GGML for CPU/Metal; retain immutable measured builds and receipts.
4. Run focused real-model tensor/output parity and session checks. Reuse the
   completed heavy numerical/performance matrix for unchanged arithmetic, with
   an explicit metadata-only delta. Never relabel old binary receipts as new.
5. Refresh source inventory, reviewer disposition, documentation and whitespace;
   commit the intended files locally. No push, release or remote CI requested.

## Results

Before repair, the baseline aggregate probe compiles and runs; the candidate
fails at double11.25 narrowing to compute_buffer_bytes. Evidence is under
build/commit-preflight/20261003/aggregate-repro. Other review observations about
interrupted exclusive outputs are recovery limits: existing paths/partial receipts
are intentionally rejected, failed data preserved, and fresh output paths required.

The unchanged archived verifier admits an unlisted root file and an altered byte
total in two isolated RED probes. Current tests now reject both and still accept
the valid archive. Stats repair also compiles/runs the unchanged legacy initializer
in GREEN. A separate quantization plan appeared during preflight and is excluded
from this commit. A clean base+owned-patch source tree is verified byte-for-byte;
all runtime/model source changes in this repair are limited to the stats layout.
Added symlinks are rejected too, including links to external directories; the
archive regression exercises that case alongside extra files and byte totals.

Isolated CPU/Metal builds compile all targets and independent headers; CTest11/11
each passes. Current tools24/24 also passes. On both backends, all11 tensor/mask
files are bitwise equal to the preserved hybrid truck/PPM case, with exact public
detections; real hybrid VideoSession checks also pass. The baseline index records
the source hashes and focused reuse delta. Heavy matrix/timing receipts and their
archived implementations remain immutable, not relabeled for rebuilt binaries.
