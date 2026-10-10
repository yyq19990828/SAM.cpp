# Repair links to retired migration documents

Created: 2026-10-10 10:27:00, Asia/Shanghai.

## Scope and approach

The user confirmed that the migration PRD should remain deleted. Preserve the
existing deletions of `docs/PRD.md` and `docs/TECH_SPEC.md`, remove broken links
and stale reading requirements from the historical migration plan, and point
readers to the retained architecture and migration plan. Keep historical file
names as plain archive paths and retain unrelated ongoing changes.

## Steps and verification

1. Locate references to both retired documents and update the historical plan
   with an explicit retirement note.
2. Record the documentation retirement in the changelog.
3. Run the repository documentation checker and `git diff --check`.

## Results

- Preserved both document deletions. The historical Orca migration plan now
  labels their paths as retired and links to the retained architecture and
  migration plan instead of requiring deleted files.
- `python3 tools/maintenance/check_docs.py`: PASS, bilingual tables and local
  links in all 139 repository Markdown documents.
- `git diff --check`: PASS. No source/runtime changes or generated artifacts.
