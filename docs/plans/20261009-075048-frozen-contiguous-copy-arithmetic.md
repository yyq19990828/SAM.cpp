# Frozen contiguous-copy arithmetic

Created: 2026-10-09 07:50:48, Asia/Shanghai.

## Scope

Continue the exact frozen CUDA Q8 single-image arithmetic audit after all
other 2,897 active compute nodes passed component checks. Target all 564
`CONT` nodes (551 F32 and 13 F16). Preserve the frozen binary, CUDA
library, model, campaign, case, quality split and arithmetic limits. Do
not infer formal performance or deployment qualification.

## Approach and steps

1. Inventory each node's source/output type, logical shape and physical
   strides. Build an isolated preload observer that synchronizes at every
   graph node, reads each `CONT` source in logical row order before the
   kernel, and reads the output afterward. Compare every raw byte in
   memory and emit SHA-256 of both byte streams, mismatch count, first
   mismatch and descriptors. This avoids retaining roughly 15 GiB of
   duplicate tensors on the 3 GiB free volume.
2. Retain raw source/output dumps for a small prefix while comparing all
   nodes in the same run. Independently verify the raw bytes and hashes in
   Python. Reconcile node order, type, shape, strides, source and output
   digest equality, and final inference parity against the frozen graph
   and previous component receipts.
3. Account separately for the graph's metadata-only nodes and leaves;
   state the numerical coverage for this one case without promoting the
   frozen campaign's whole-recipe or deployment gates beyond their actual
   input scope.

## Verification

Compile C++17 with warnings as errors. Run trial and full comparator,
independent receipt verifier, isolated Python tool suite, documentation
checker and `git diff --check`. Remove superseded trial dumps after
their verification, retaining immutable receipts and current builds.

## Results

The first diagnostic observer stopped at a transposed F32 source because it
assumed an inner-contiguous stride. A revised observer handled arbitrary
inner strides but reached a view with millions of 8-byte rows and spent
excessive time issuing one device read per row; that run was interrupted
without a numerical failure. The final observer reads each tensor's
physical span once and gathers its logical elements by stride on the CPU.
All variants were isolated preload sidecars; no frozen binary, GGML
library, model, campaign or limit changed.

The complete run on frozen `coco-000000001503-p0`/`tv` checked all 564
active `CONT` nodes (551 F32, 13 F16). All 8,156,709,344 logical source
bytes matched output bytes exactly, with zero mismatches. Each node's
source/output SHA-256, type, shape and strides is in the immutable ledger.
The first three nodes also retained raw source/output dumps; an independent
Python verifier rehashed those six files, checked their sizes and matched
them to the ledger. The same verifier reconciled all 564 nodes to the
frozen graph inventory. Final masks, boxes and ranking matched the
unobserved export; synchronization changed `query_scores` by at most
1.19e-7.

A separate structural audit passed all 1,308 `RESHAPE`, 33 `TRANSPOSE`,
571 `PERMUTE` and 703 `VIEW` metadata nodes plus 1,155 leaves, checking
alias buffers, offsets, extents, shapes and strides. The one-case aggregate
now accounts for all 3,461 active CUDA compute nodes and 7,231 graph
tensors. Its status is `PASS` for the single development image and prompt.
The frozen campaign's whole-recipe arithmetic remains `NOT_RUN` because
this one-case audit does not cover its full input scope. Independent image
quality remains `PASS`; formal exclusive-GPU performance and deployment
remain `NOT_RUN`.

The following Git-ignored receipt paths are available only in this
workspace under `build/q8-integrated-arithmetic-20261009/`:

- `cont-full-verification.json`: SHA-256
  `404aee72e4cd83e0ade44b72534b29e07e0cc729d4cd848070f3c1d9b18a0175`.
- `cont-reconciled.json`: SHA-256
  `144a5a23a803e6258f925c421a30651457531e95130a0a7a08fdb7bdac690106`.
- `metadata-audit.json`: SHA-256
  `b5b06cce74ec3dd5c787aaa1b6821ddfaa002494fa4cdd9570c97e8474ac27af`.
- `one-case-active-graph-reconciled.json`: SHA-256
  `e4270ef25a021fe298b2f7636e39b2b658519b7fa1b4854ee5e36d8fccca35ec`.
- `cont-trial-v3/nodes.tsv`: SHA-256
  `2d81b1bdbe2534a71433b02876485399addeb6474855781c94645bbef53955bd`.
- `capture_cont.cpp` and `libcapture_cont.so`: SHA-256
  `fdf14203a50fe4791b3c15467777e97d1f67b94606ec89bd989e1e2da018fe69`
  and `6dc697dd20ad37607a01fa836be6677911c015af3a99f0e078e297b59a5cc102`.
- `verify_cont.py`: SHA-256
  `cf69310cbe579f0c8662284341f0590ff107e74ddf86c33c93f418b96a9f6532`.

The six raw prefix files and their SHA-256 values are listed in the
verification receipt. The aggregate receipt binds all prior component
receipts, the frozen campaign and the diagnostic GGUF. A non-campaign
desktop client still shares the RTX 4090, so the formal exclusive-GPU
performance gate cannot run in this session.

The isolated Python tool suite passed 170 tests; the documentation checker
passed 103 documents; `git diff --check` passed. The interrupted observer's
duplicate raw prefix dumps were removed after the corrected full run;
its partial ledger remains diagnostic evidence.
