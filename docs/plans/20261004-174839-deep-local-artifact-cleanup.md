# Historical validation payload retirement

Created: 2026-10-04 17:48:40 Asia/Shanghai.
Status: historical payloads retired; original references and metadata retained.
Compacted to project provenance on 2026-10-05.

## Scope and acceptance boundary

The earlier user-authorized project-artifact cleanup retired five superseded
full validation bundles. Original archive manifests, acceptance indexes and
metadata/source reports remain unchanged; their full historical byte payloads
are unavailable for complete archive verification. Retirement does not turn
historical acceptance into current acceptance.

The independent official SAM 3 video reference was retained at
`models/reference/official-sam3-video-20261004-174839`. Its manifest SHA256 is
`00e51649947020c3cab441f35b70ff69df6ff4c88fff9041f40cfee9ce2e4d46`.
The official checkpoint, tokenizer, image references, supported GGUF files and
then-current v8 evidence were protected. Project temporary-directory cleanup
also retained its original reference and metadata.

## Provenance

The unchanged [retirement index](../validation-baselines/archive-retirement-20261004.json)
records original manifest hashes, retained metadata/reference destinations and
the historical verification boundary. Its detailed local deletion ledger is
`build/deep-cleanup-20261004-174839.json`. This plan keeps its existing filename
so the index's `source_plan` remains valid.

Subsequent source-bound numerical qualification and current measurements are
recorded in the [complete performance plan](20261004-214547-latest-complete-model-performance-records.md).
Workstation settings, filesystem-space observations and redundant operational
inventories have been removed from this project plan. Raw evidence is unchanged.
