# Plan document scope cleanup

Created: 2026-10-05 01:37:54 Asia/Shanghai.
Status: complete.

## Scope and approach

The user requested removing plans with little connection to this project while
the authorized local commit is being prepared. Keep implementation, numerical
acceptance, performance history and reference provenance. This work changes
documentation only; model, build and reference payloads are outside scope.

Remove the untracked obsolete workstation-space and iCloud-sync plans
`20261004-172458-obsolete-artifact-cleanup.md` and
`20261004-192303-icloud-local-storage-cleanup.md`. The project archive-retirement
record is relevant to acceptance provenance: compact
`20261004-174839-deep-local-artifact-cleanup.md` to its project-specific facts.
Keep its existing path because the immutable retirement index names it.

## Verification and results

Check Markdown links, the retirement index's unchanged bytes/source-plan path,
unchanged production fingerprints, and whitespace before the local commit.
Record the completed scope here; preserve other engineering plans and all raw
evidence. No push is authorized.

Reviewed all 39 existing plan topics. Removed the two untracked workstation
operation records above; compacted the archive-retirement plan while retaining
its path and project acceptance boundary. Implementation, numerical and
performance history remain intact. The retirement JSON and all model/build/
reference payloads were not edited. This cleanup record is project documentation
maintenance; it contains no workstation settings or disk inventories.

Verification passed: 52 documents' bilingual table/local-link checks,
`git diff --check`, the retirement source-plan target, its protected reference
manifest, and all 98 frozen production source paths. The retirement index is
byte-identical, SHA256
`de20169d8c13fe5435c977cd669f79398f81746fe3c41aee3211f0ac72467cdf`.
Independent read-only review found no dangling references or lost numerical
provenance. These two removed files were never tracked, so no public changelog
entry claims deletion of previously shipped documentation.
