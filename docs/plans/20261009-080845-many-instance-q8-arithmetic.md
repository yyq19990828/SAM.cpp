# Frozen Q8 arithmetic on a many-instance development input

Created: 2026-10-09 08:08:45, Asia/Shanghai.

## Scope

Extend exact frozen CUDA Q8_0 component arithmetic to the predeclared
`perf-instances-many` development case, `coco-000000018380`, prompt `person`.
Use the unchanged model, library, executable, numerical gate and prior native
output. Keep the final evaluation and 1,024-image reserve out of tuning.
The sole GPU remains ineligible for exclusive performance acceptance.

## Approach and steps

1. Bind the image, prompt index, native output and frozen recipe to the
   campaign and performance-case manifest. Check durable disk capacity before
   capture. The repository volume has insufficient headroom; a separate
   mounted ext4 volume has 640 GiB free. Use a dedicated directory there
   through a Git-ignored `build/` symlink.
2. Capture the 327 Q8_0 matrix operations into a fresh evidence directory and
   independently verify their GGUF weight bytes, Q8_1 RHS staging, F64
   same-operand dots and final inference payload under the unchanged gate.
3. Retain all raw operands, outputs and staging bytes. Rehash every payload
   against the verifier's receipt; bind frozen model, binary, library and
   source identities. Compare the frozen Q8 weight and input-dependent RHS
   hashes with earlier cases. Do not promote three inputs to whole-recipe
   arithmetic PASS.

## Verification

Run the exact frozen binary and isolated reference verifier; rehash the
retained raw evidence after verification.
Run the relevant tool suite, docs checker and `git diff --check`; retain the
current verified build and required evidence. Leave other directories on the
mounted volume untouched.

## Results

The frozen `perf-instances-many` image and `person` prompt were bound to
the calibration/development dataset, performance manifest, Q8 native export
and exact model/binary recipe. The case receipt is
`build/q8-integrated-arithmetic-20261009/many-case-bound.json` (SHA-256
`4fd2ab3891edee6710ad727a1956bf0375f64ee78b250cb843040bea8833143b`).
The frozen binary captured all 327 Q8_0 matrix nodes. The independent
verifier checked 161,445,056 same-operand F64 dots and 2,612,156 Q8_1 staging
blocks under the unchanged gate; the worst relative L2 was
`1.327418055440376e-07`. Final inference payloads matched the prior
unobserved export exactly after removing runtime measurements. Its receipt is
`build/q8-integrated-arithmetic-20261009/many-capture-verification.json`
(SHA-256 `fa075bef50404f0b8ca69b733f1eb1ddf50d39830e2a71d67677653ab497d46b`).

The three-input reconciliation rehashed all 4,905 bound per-node payloads
and frozen identities. All 327 Q8_0 weight payloads were stable; relative
to each earlier input, 312 RHS activations and 312 dot outputs changed. Its
receipt is `build/q8-integrated-arithmetic-20261009/many-three-input-reconciled.json`
(SHA-256 `1d62c51427059f34f0ce0a5f8ace9d9d5b707d15e5db46a79e700837d3fdd3ca`).
The complete raw capture is reachable through the Git-ignored local path
`build/q8-integrated-arithmetic-20261009/many-external/many-capture`, backed
by the separate `/mnt/SSD1` volume; it is available only in this validation
workspace. The campaign, thresholds, evaluation and reserve were unchanged.
Whole-recipe arithmetic, exclusive-GPU performance and deployment remain
`NOT_RUN`. The follow-on [eight-input audit](20261009-081225-eight-performance-q8-arithmetic.md)
reuses this immutable evidence.
The completed validation cycle passed 170/170 tool tests, the bilingual/local
link checker on 106 documents, and `git diff --check`. Its raw capture and
staging data remain required evidence; no disposable new intermediate was
removed.
