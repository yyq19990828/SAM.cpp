# Benchmark and model catalog documentation

Created: 2026-10-01 01:45:55 Asia/Shanghai.
Status: documentation complete; no GGUF implementation or inference changes.

## Scope

Add root `BENCHMARK.md` and `MODEL_ZOO.md`, link them from README, and record
the completed documentation in `changelog.md`. Reuse verified local results;
this task does not require new inference or performance runs.

## Approach and steps

1. Check the final benchmark receipts, checkpoint manifests, converter options,
   runtime format reader, and official HF/GitHub source links.
2. Use models/precision as benchmark rows, with two header rows for hardware
   and backend/software versions. Record units, workload, warmup/repeats,
   storage versus compute precision, and memory semantics. Mark missing results
   explicitly; do not invent CUDA measurements or versions.
3. Document supported model sources, pinned checkpoint/code revisions, hashes,
   current conversion commands, and verification. Separate future model families
   from supported models. The existing converter/reader use custom `.ggml` v3;
   clarify the requested GGUF scope before advertising a GGUF build command.
   A GGUF runtime migration requires implementation and numerical acceptance,
   and is not implied by a filename change.
4. Keep README navigation concise and preserve historical acceptance records.

## Verification

Check table structure, numbers against receipts, source/local links, shell
syntax and actual CLI help. Check whitespace and documentation punctuation.
Do not rerun model suites for documentation-only changes. Record results below.

## Results

- Planning preceded document changes. Existing project files remain uncommitted.
- Added `BENCHMARK.md` with an HTML table containing exactly two header rows:
  hardware above backend/software, with model/precision rows. CPU/Metal median
  latency and peak RSS match the final five-run receipts. FP32 CPU is explicitly
  unmeasured for timing; FP32 Metal is unsupported. No CUDA values were invented.
- The host reports 14 CPU cores (10 performance + 4 efficiency), 20 GPU cores,
  48 GiB memory and macOS SDK 27.0. These identify the measured M4 Pro variant.
- Added `MODEL_ZOO.md` with official HF/GitHub sources, pinned revisions,
  checkpoint/BPE/output hashes, authenticating/downloading instructions and the
  existing FP32/FP16 conversion path. Explained custom GGML versus GGUF in
  response to the user's follow-up; GGUF remains explicitly unimplemented.
- HF metadata confirms the pinned revision. Official and community source pages
  resolve; pinned tokenizer/image raw URLs return HTTP 200. Documented CLI
  arguments match installed HF and converter help.
- Table values/hashes, two-row header structure, local Markdown links, shell
  syntax, punctuation and whitespace checks pass. `git diff --check` passes;
  changed untracked documents were checked directly as well. No model suite,
  conversion, download or benchmark was rerun for this documentation change.
- README links both guides; `changelog.md` records the additions. No commit/push.
