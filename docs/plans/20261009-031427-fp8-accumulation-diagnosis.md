# FP8 accumulation diagnosis

Created 2026-10-09 03:14:27 Asia/Shanghai. This plan concerns the experimental CUDA linear probe only. The frozen v2 model gates and earlier receipts remain unchanged.

## Scope

Investigate why the RTX 4090/cuBLASLt E4M3 candidate fails its existing decoded-operand raw-dot checks with `FAST_ACCUM=0`. Establish whether a supported algorithm configuration can meet the current arithmetic gate, without weakening the threshold or claiming integrated SAM support.

## Approach and steps

1. Inspect the prior immutable F32/F16 failures and current cuBLASLt configuration. Compare NVIDIA's documented accumulation semantics with the gate's independent reference.
2. Add a focused, opt-in diagnostic only if needed to test available algorithm candidates and preserve exact packed operands, raw results and algorithm identity in a new output directory.
3. Recompute the relevant dots independently from the packed E4M3 bytes, distinguish scale/output rounding from accumulation behavior, and try any supported stricter candidate before deciding whether this cuBLASLt route can enter the next stage.
4. Record hashes, device/toolchain identity, findings and remaining limitations. Update user-facing quantization guidance or changelog only for actual behavior changes.

## Verification

Build the affected CUDA probe, run the pre-existing FP8 failure shapes and relevant small boundaries in fresh directories, run focused tool tests and `git diff --check`. Preserve the previous failed receipts. Clean unneeded intermediates while keeping the current verified build and immutable evidence.

## Results

NVIDIA's [cuBLASLt descriptor documentation](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t) says disabling `FAST_ACCUM` periodically promotes FP8 intermediate results to higher precision; it does not promise an exact decoded-operand sum. The [FP8 combination table](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul) requires `CUBLAS_COMPUTE_32F` for these FP8 kernels, so a pedantic compute-type switch is not an available stricter setting for this route. The numerical-implementation flags identify an F32 accumulator but do not override the measured discrepancy.

The opt-in CUDA probe flags `--fp8-algo-index=0..7` and `--dump-fp8` select one returned heuristic candidate and save packed E4M3 operands, raw pre-epilogue output and tensorwide scales, even when the existing gate rejects it. The normal fastest-candidate selection and its threshold are unchanged. The separate [FP8-byte verifier](../../tools/validation/verify_fp8_dots.py) decodes the saved bytes without CUDA, recomputes the same 512 sampled dots in F64, applies the original F32/F16 output rounding and tolerance, and confirms the first reported failure. It hashes every input and output before and after reading.

On RTX 4090 SM 8.9, driver 610.57.04, all **8/8 returned candidates failed** the existing QKV F32 gate at output 0: the independently decoded dot is `4235.715209960938`, versus the previously recorded GPU unit-scale output `4231.34375`; scaled values are approximately `0.11935859` expected and `0.11923541` actual. Independent verification found **429/512 sampled outputs outside tolerance for each candidate**. All eight report algorithm ID 35 and the same numerical flags, although their configurations need not be identical.

The regenerated nearest-even F16 boundary also failed under all **8/8 candidates** at output 451: decoded dot `39676.8671875`, scaled F16 expectation `18.578125`, GPU output `18.640625`. Independent verification found 11–22 failing sampled outputs per candidate. Across both shapes, **16/16 configurations reproduced failure; 3,542/8,192 checked positions exceeded the unchanged threshold**. This is a failure result, not acceptance. The sweep covers the eight candidates returned by the current heuristic and does not assert that every possible cuBLASLt configuration or GPU behaves this way.

The aggregate receipt is `build/fp8-algorithm-diagnosis-20261009/summary.json`, SHA-256 `414d5e3ec0322e496aaa4db0a3aa3e10d9e37b2661187d203dcaa39da862a8e0`. It binds the source, rebuilt binary, frozen QKV and generated boundary inputs, cuBLASLt/CUDA runtime libraries and 16 independent per-candidate receipts. These Git-ignored artifacts are available only in this local workspace. Prior FP8 failures and frozen v2 receipts were not changed. The current FP8 route remains excluded from integration; a different accumulation implementation would need fresh arithmetic, quality, latency and memory evaluation.

The affected CUDA target rebuilt. The default INT8 probe still passed on the small boundary input; an FP8-only selection flag was rejected in INT8 mode before output creation. The isolated Python tool suite passed **161/161**, relevant CPU/CUDA quantization CTest passed **2/2**, documentation tests passed **4/4**, and `git diff --check` passed. One focused test expectation was corrected before the final suite; the production verifier did not change after evidence generation. Unused generated synthetic inputs were removed, retaining the nearest-even input, current verified CUDA build and all required candidate evidence.
