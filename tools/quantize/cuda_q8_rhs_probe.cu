#include <cuda_runtime.h>
#include <ggml.h>
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// The pinned GGML CUDA library exports the same staging routine used by MMVQ.
static_assert(QK8_1 == 32 && sizeof(block_q8_1) == 36, "unsupported GGML Q8_1 layout");
void quantize_row_q8_1_cuda(const float* input, const int32_t* ids, void* output,
                            ggml_type weight_type, int64_t width, int64_t row_stride,
                            int64_t plane_stride, int64_t batch_stride, int64_t padded_width,
                            int64_t rows, int64_t planes, int64_t batches, cudaStream_t stream);

namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct DeviceBuffer {
    void* data = nullptr;
    explicit DeviceBuffer(std::size_t bytes) { check(cudaMalloc(&data, bytes)); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Stream {
    cudaStream_t data = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&data, cudaStreamNonBlocking)); }
    ~Stream() { if (data) cudaStreamDestroy(data); }
};

std::uint32_t dimension(const char* text) {
    std::size_t consumed = 0;
    const auto number = std::stoull(text, &consumed);
    if (consumed != std::string(text).size() || number == 0 || number > 16384)
        throw std::invalid_argument("Q8_1 probe dimensions must be in [1,16384]");
    return static_cast<std::uint32_t>(number);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument("usage: sam_cuda_q8_rhs_probe INPUT_F32 WIDTH ROWS NEW_DIR");
        const std::filesystem::path input_path = argv[1], output_directory = argv[4];
        const auto width = dimension(argv[2]), rows = dimension(argv[3]);
        if (width % 32 != 0 || std::uint64_t(width) * rows > (1u << 26))
            throw std::invalid_argument("Q8_1 probe needs 32-aligned bounded rows");
        const auto count = std::size_t(width) * rows;
        if (std::filesystem::file_size(input_path) != count * sizeof(float))
            throw std::invalid_argument("Q8_1 probe input length differs");
        std::vector<float> input(count);
        std::ifstream source(input_path, std::ios::binary);
        source.read(reinterpret_cast<char*>(input.data()), std::streamsize(count * sizeof(float)));
        if (!source || !std::all_of(input.begin(), input.end(), [](float value) { return std::isfinite(value); }))
            throw std::invalid_argument("Q8_1 probe input is truncated or non-finite");
        if (!std::filesystem::create_directory(output_directory))
            throw std::runtime_error("Q8_1 probe output must be new");

        check(cudaSetDevice(0));
        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, 0));
        const std::size_t packed_bytes = count / QK8_1 * sizeof(block_q8_1);
        DeviceBuffer in(count * sizeof(float)), out(packed_bytes);
        Stream stream;
        check(cudaMemcpyAsync(in.data, input.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream.data));
        quantize_row_q8_1_cuda(static_cast<const float*>(in.data), nullptr, out.data,
                                GGML_TYPE_Q8_0, width, width, width * rows, width * rows,
                                width, rows, 1, 1, stream.data);
        check(cudaGetLastError());
        std::vector<std::uint8_t> packed(packed_bytes);
        check(cudaMemcpyAsync(packed.data(), out.data, packed_bytes, cudaMemcpyDeviceToHost, stream.data));
        check(cudaStreamSynchronize(stream.data));
        std::ofstream payload(output_directory / "rhs.q8_1", std::ios::binary);
        payload.write(reinterpret_cast<const char*>(packed.data()), std::streamsize(packed.size()));
        if (!payload) throw std::runtime_error("failed to save Q8_1 staging payload");
        std::ofstream report(output_directory / "probe.json");
        report << "{\"schema_version\":1,\"complete\":true,\"diagnostic_only\":true,\"width\":" << width
               << ",\"rows\":" << rows << ",\"blocks\":" << count / 32
               << ",\"format\":\"Q8_1: half scale, half original-input sum, 32 signed bytes\""
               << ",\"weight_type\":\"q8_0\",\"staging_path\":\"MMVQ\",\"device\":\""
               << device.name << "\",\"compute_capability\":" << device.major * 10 + device.minor << "}\n";
        if (!report) throw std::runtime_error("failed to save Q8_1 staging report");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_cuda_q8_rhs_probe: " << error.what() << '\n';
        return 1;
    }
}
