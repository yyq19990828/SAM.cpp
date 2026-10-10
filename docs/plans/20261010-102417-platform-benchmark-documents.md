# Split performance documents by operating system and hardware

Created: 2026-10-10 10:24:17, Asia/Shanghai.

## Scope and approach

Move the existing macOS M4 Pro and Linux RTX 4090 measurement sections from
the bilingual root performance pages into `benchmarks/MacOS-m4pro.md` and
`benchmarks/Linux-4090.md`, with matching `_zh.md` translations. Preserve the
measurement values, dates, qualification scope and environment descriptions.
Keep root `BENCHMARK.md` and `BENCHMARK_zh.md` as linked indexes and shared
measurement guidance, so existing user-guide and historical-plan links remain
usable. Keep ongoing precision-v3 work and unrelated deletions intact.

## Steps

1. Extract each platform's existing sections and fix relative links and
   references to the other platform.
2. Add platform indexes, update the changelog, and retain bilingual numeric-table
   checking for the moved measurement tables.
3. Check repository Markdown links, compare migrated table values with the
   original files, and run the focused documentation tests and whitespace check.

## Verification

This is a documentation organization change; no new performance measurement or
model/backend qualification is involved. Numeric table rows must match the
pre-change pages in order, and English/Chinese platform tables must agree.

## Results

- Created the two platform pages and their Chinese translations under
  `benchmarks/`. Both root pages now link to them and retain shared measurement
  guidance. Retained the legacy English Linux/video anchors used by historical
  plans without editing those plans.
- Compared the extracted tables against `HEAD:BENCHMARK.md` and
  `HEAD:BENCHMARK_zh.md`: all 49 table rows in each language are byte-identical
  and retain their order. Existing measurement dates and environments are
  preserved; this change introduces no new timing results.
- Extended `tools/maintenance/check_docs.py` to discover bilingual platform
  benchmark tables. All five focused documentation tests pass, including the
  new missing-translation and numeric-mismatch checks.
- The seven affected performance/plan documents pass all 50 local-link checks
  and both bilingual platform-table comparisons. `git diff --check` passes.
- Full-repository documentation checking is blocked by the pre-existing
  deletion of `docs/PRD.md`, still referenced from
  `docs/plans/20261008-211718-compiled-library-orca-dag.md`. Left this unrelated
  deletion and ongoing v3 work untouched. No build/model artifacts were created
  by this documentation change.
