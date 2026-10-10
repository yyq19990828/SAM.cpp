#include <cuda_runtime_api.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

void quantize_mmq_q8_1_cuda(const float* input, const int32_t* ids, void* output,
                             ggml_type weight_type, int64_t width, int64_t row_stride,
                             int64_t plane_stride, int64_t batch_stride, int64_t padded_width,
                             int64_t rows, int64_t planes, int64_t batches, cudaStream_t stream);

namespace {

struct Q8MmqBlock {
    float scales[4];
    std::int8_t values[128];
};
static_assert(sizeof(Q8MmqBlock) == 144, "unexpected Q8 MMQ staging layout");

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_boundary() {
    constexpr int width = 160, padded_width = 512, rows = 65;
    constexpr int blocks_per_row = padded_width / 128;
    std::vector<float> input(width * rows, 0.0f);
    input[0] = 3.0e38f;
    input[width + 32] = -3.0e38f;
    input[2 * width + 64] = 2000.0f;
    input[2 * width + 80] = -800.0f;
    input[3 * width + 128] = -2000.0f;
    std::vector<Q8MmqBlock> output(blocks_per_row * rows);

    float* device_input = nullptr;
    Q8MmqBlock* device_output = nullptr;
    cudaStream_t stream = nullptr;
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(float)));
        check(cudaMalloc(reinterpret_cast<void**>(&device_output), output.size() * sizeof(Q8MmqBlock)));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaMemcpyAsync(device_input, input.data(), input.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
        quantize_mmq_q8_1_cuda(device_input, nullptr, device_output, GGML_TYPE_Q8_0,
                               width, width, width * rows, width * rows,
                               padded_width, rows, 1, 1, stream);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(output.data(), device_output,
                              output.size() * sizeof(Q8MmqBlock), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));

        for (int block = 0; block < blocks_per_row; ++block) {
            for (int row = 0; row < rows; ++row) {
                const auto& packed = output[block * rows + row];
                for (int subblock = 0; subblock < 4; ++subblock) {
                    double maximum = 0.0;
                    for (int offset = 0; offset < 32; ++offset) {
                        const int column = block * 128 + subblock * 32 + offset;
                        if (column < width)
                            maximum = std::max(maximum, std::abs(double(input[row * width + column])));
                    }
                    const float scale = packed.scales[subblock];
                    require(std::isfinite(scale) && scale >= 0.0f,
                            "finite F32 input produced a nonfinite or negative MMQ scale");
                    const float expected = float(maximum / 127.0);
                    if (expected == 0.0f) {
                        require(scale == 0.0f, "zero MMQ block has a nonzero scale");
                    } else {
                        const float ulp = std::nextafter(expected, std::numeric_limits<float>::infinity()) - expected;
                        require(std::abs(double(scale) - maximum / 127.0) <= 4.0 * double(ulp),
                                "MMQ scale differs from the F64 reference by more than four F32 ulps");
                    }
                }
                for (std::int8_t value : packed.values)
                    require(value >= -127 && value <= 127, "MMQ staging integer is outside [-127,127]");
            }
        }
        require(output[0 * rows + 0].values[0] == 127, "positive finite extreme lost its endpoint");
        require(output[0 * rows + 1].values[32] == -127, "negative finite extreme lost its endpoint");
        require(output[0 * rows + 2].values[64] == 127 &&
                output[0 * rows + 2].values[80] == -51,
                "ordinary signed MMQ values changed their encoded integers");
        require(output[1 * rows + 3].values[0] == -127,
                "second MMQ block lost its negative endpoint");
        check(cudaStreamDestroy(stream));
        check(cudaFree(device_output));
        check(cudaFree(device_input));
    } catch (...) {
        if (stream) cudaStreamDestroy(stream);
        if (device_output) cudaFree(device_output);
        if (device_input) cudaFree(device_input);
        throw;
    }
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        check(cudaSetDevice(0));
        check_boundary();
        std::cout << "CUDA Q8 MMQ finite boundary PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA Q8 MMQ finite boundary FAIL: " << error.what() << '\n';
        return 1;
    }
}
