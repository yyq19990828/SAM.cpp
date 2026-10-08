// Diagnostic bridge to the pinned GGML CUDA cache encoder. Calling the vendor
// routine preserves its compiled fast-math behavior at quantization boundaries.
#include "ggml-cuda/cpy-utils.cuh"
#include <cstddef>

namespace {

__global__ void encode_blocks(const float* values, block_q8_0* packed, std::size_t count) {
    const std::size_t block = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (block < count) quantize_f32_q8_0_block(values + block * QK8_0, packed + block);
}

} // namespace

extern "C" int sam_cache_bridge_abi() {
    return 1;
}

extern "C" int sam_cache_encode_q8_0(const float* values, void* packed, std::size_t elements, void* stream) {
    if (!values || !packed || elements == 0 || elements % QK8_0 || elements > (1u << 26))
        return int(cudaErrorInvalidValue);
    const auto blocks = elements / QK8_0;
    encode_blocks<<<(blocks + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        values, static_cast<block_q8_0*>(packed), blocks);
    return int(cudaGetLastError());
}
