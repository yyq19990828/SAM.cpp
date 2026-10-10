#include "linear_probe.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cfloat>
#include <cstring>
#include <iostream>

namespace {

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLASLt status " + std::to_string(int(status)));
}

struct Buffer {
    void* data = nullptr;
    std::size_t bytes = 0;
    explicit Buffer(std::size_t size = 0) { reset(size); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (data) cudaFree(data); }
    void reset(std::size_t size) {
        if (data) cuda_check(cudaFree(data));
        data = nullptr; bytes = 0;
        if (size) cuda_check(cudaMalloc(&data, size));
        bytes = size;
    }
    template<class T> T* as() const { return static_cast<T*>(data); }
    template<class T> void upload(const std::vector<T>& values) {
        if (values.size() * sizeof(T) != bytes) throw std::runtime_error("upload length differs");
        cuda_check(cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice));
    }
    template<class T> std::vector<T> download() const {
        if (bytes % sizeof(T)) throw std::runtime_error("download length differs");
        std::vector<T> values(bytes / sizeof(T));
        cuda_check(cudaMemcpy(values.data(), data, bytes, cudaMemcpyDeviceToHost));
        return values;
    }
};

struct Stream {
    cudaStream_t handle = nullptr;
    Stream() { cuda_check(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking)); }
    ~Stream() { if (handle) cudaStreamDestroy(handle); }
    void synchronize() { cuda_check(cudaStreamSynchronize(handle)); }
};

struct Event {
    cudaEvent_t handle = nullptr;
    Event() { cuda_check(cudaEventCreate(&handle)); }
    ~Event() { if (handle) cudaEventDestroy(handle); }
    void record(cudaStream_t stream) { cuda_check(cudaEventRecord(handle, stream)); }
    double elapsed(const Event& start) const {
        float value = 0;
        cuda_check(cudaEventElapsedTime(&value, start.handle, handle));
        return value;
    }
};

__device__ float block_max(float value) {
    __shared__ float shared[256];
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (threadIdx.x < step) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + step]);
        __syncthreads();
    }
    return shared[0];
}

template<bool Weight>
__global__ void pack_int8_rows(const float* source, const float* channel_scale, int8_t* packed,
                              float* row_scale, int inner) {
    const int row = blockIdx.x;
    float maximum = 0;
    for (int col = threadIdx.x; col < inner; col += blockDim.x) {
        const float value = Weight ? source[row * inner + col] * channel_scale[col]
                                   : source[row * inner + col] / channel_scale[col];
        maximum = fmaxf(maximum, fabsf(value));
    }
    maximum = block_max(maximum);
    const float scale = maximum == 0 ? 1 : fmaxf(maximum / 127.0f, FLT_MIN);
    if (threadIdx.x == 0) row_scale[row] = scale;
    for (int col = threadIdx.x; col < inner; col += blockDim.x) {
        const float value = Weight ? source[row * inner + col] * channel_scale[col]
                                   : source[row * inner + col] / channel_scale[col];
        packed[row * inner + col] = int8_t(max(-127, min(127, __float2int_rn(value / scale))));
    }
}

__global__ void pack_static_int8(const float* source, const float* channel_scale, int8_t* packed,
                                 std::size_t count, int inner, float scale) {
    const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = source[index] / channel_scale[index % inner];
        // Clamp before converting: __float2int_rn overflows on large finite
        // ratios and would otherwise turn positive saturation into -127.
        packed[index] = int8_t(__float2int_rn(fminf(127.0f, fmaxf(-127.0f, value / scale))));
    }
}

template<bool Weight>
__global__ void pack_fp8(const float* source, const float* channel_scale, __nv_fp8_e4m3* packed,
                         std::size_t count, int inner, float scale) {
    const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = Weight ? source[index] * channel_scale[index % inner]
                                   : source[index] / channel_scale[index % inner];
        packed[index] = __nv_fp8_e4m3(value / scale);
    }
}

__device__ float finish(float value, float bias, bool gelu) {
    value += bias;
    return gelu ? 0.5f * value * (1.0f + erff(value * 0.7071067811865475f)) : value;
}

template<bool Half>
__global__ void finish_int8(int32_t* sums, void* output, const float* weight_scale, const float* input_scale,
                           const float* bias, std::size_t count, int rows, bool gelu) {
    const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = finish(float(sums[index]) * weight_scale[index % rows] * input_scale[index / rows],
                                   bias[index % rows], gelu);
        if (Half) static_cast<__half*>(output)[index] = __float2half_rn(value);
        else sums[index] = __float_as_int(value); // Same-width in-place conversion; no overlapping elements.
    }
}

template<bool Half>
__global__ void finish_fp8(void* values, const float* bias, std::size_t count, int rows, bool gelu) {
    const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = Half ? __half2float(static_cast<__half*>(values)[index]) : static_cast<float*>(values)[index];
        const float result = finish(value, bias[index % rows], gelu);
        if (Half) static_cast<__half*>(values)[index] = __float2half_rn(result);
        else static_cast<float*>(values)[index] = result;
    }
}

struct Gemm {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    bool fp8 = false;
    int algorithm_id = -1;
    int algorithm_index = -1;
    int heuristic_count = 0;
    uint64_t numerical_flags = 0;
    std::size_t required_workspace = 0;

    Gemm() { blas_check(cublasLtCreate(&handle)); }
    ~Gemm() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (handle) cublasLtDestroy(handle);
    }
    cublasStatus_t run(const void* weights, const void* input, void* output, const Buffer& workspace,
                       cudaStream_t stream, const cublasLtMatmulAlgo_t* selected = nullptr) {
        const float one = 1, zero = 0;
        const int32_t integer_one = 1, integer_zero = 0;
        return cublasLtMatmul(handle, operation, fp8 ? static_cast<const void*>(&one) : &integer_one,
            weights, a, input, b, fp8 ? static_cast<const void*>(&zero) : &integer_zero,
            output, c, output, c, selected ? selected : &algorithm, workspace.data, workspace.bytes, stream);
    }
    void prepare(const sam_probe::Input& input, bool use_fp8, bool half, const Buffer& scales,
                 const Buffer& weights, const Buffer& values, void* output, Buffer& workspace, Stream& stream,
                 int requested_algorithm_index = -1) {
        fp8 = use_fp8;
        blas_check(cublasLtMatmulDescCreate(&operation, fp8 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32I,
                                          fp8 ? CUDA_R_32F : CUDA_R_32I));
        const auto transpose = CUBLAS_OP_T;
        blas_check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
        if (fp8) {
            const void* weight_scale = scales.as<float>();
            const void* input_scale = scales.as<float>() + 1;
            const int8_t fast_accumulation = 0;
            blas_check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scale, sizeof(weight_scale)));
            blas_check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &input_scale, sizeof(input_scale)));
            blas_check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accumulation, sizeof(fast_accumulation)));
        }
        const auto type = fp8 ? CUDA_R_8F_E4M3 : CUDA_R_8I;
        const auto result_type = fp8 ? (half ? CUDA_R_16F : CUDA_R_32F) : CUDA_R_32I;
        blas_check(cublasLtMatrixLayoutCreate(&a, type, input.k, input.m, input.k));
        blas_check(cublasLtMatrixLayoutCreate(&b, type, input.k, input.n, input.k));
        blas_check(cublasLtMatrixLayoutCreate(&c, result_type, input.m, input.n, input.m));
        blas_check(cublasLtMatmulPreferenceCreate(&preference));
        const std::size_t limit = 16 * 1024 * 1024;
        blas_check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
        const uint64_t accumulator = fp8 ? CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F :
            CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I;
        const uint64_t implementation_mask = ~uint64_t(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) | accumulator;
        blas_check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_IMPL_MASK,
                                                       &implementation_mask, sizeof(implementation_mask)));
        cublasLtMatmulHeuristicResult_t choices[8]{};
        int count = 0;
        blas_check(cublasLtMatmulAlgoGetHeuristic(handle, operation, a, b, c, c, preference, 8, choices, &count));
        heuristic_count = count;
        if (!count) throw std::runtime_error("no cuBLASLt algorithm supports this data/layout/shape combination");
        if (requested_algorithm_index >= count) throw std::invalid_argument("FP8 heuristic index is unavailable");
        workspace.reset(limit);
        Event start, end;
        double best = std::numeric_limits<double>::infinity();
        for (int i = 0; i < count; ++i) {
            if (requested_algorithm_index >= 0 && i != requested_algorithm_index) continue;
            if (choices[i].state != CUBLAS_STATUS_SUCCESS) continue;
            uint64_t flags = 0;
            std::size_t flag_bytes = 0;
            blas_check(cublasLtMatmulAlgoCapGetAttribute(&choices[i].algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                                       &flags, sizeof(flags), &flag_bytes));
            if ((flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != accumulator) continue;
            if (run(weights.data, values.data, output, workspace, stream.handle, &choices[i].algo) != CUBLAS_STATUS_SUCCESS) continue;
            start.record(stream.handle);
            for (int repeat = 0; repeat < 5; ++repeat)
                blas_check(run(weights.data, values.data, output, workspace, stream.handle, &choices[i].algo));
            end.record(stream.handle);
            stream.synchronize();
            const auto elapsed = end.elapsed(start);
            if (elapsed < best) {
                best = elapsed;
                algorithm = choices[i].algo;
                algorithm_index = i;
                required_workspace = choices[i].workspaceSize;
                numerical_flags = flags;
            }
        }
        if (!std::isfinite(best)) throw std::runtime_error("selected cuBLASLt algorithm rejected execution");
        workspace.reset(required_workspace);
        std::size_t written = 0;
        blas_check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, CUBLASLT_ALGO_CONFIG_ID, &algorithm_id, sizeof(algorithm_id), &written));
    }
};

float decode_fp8(unsigned char byte) {
    __nv_fp8_e4m3 value;
    value.__x = byte;
    return float(value);
}

float host_finish(float value, float bias, bool gelu, bool half) {
    value += bias;
    if (gelu) value = 0.5f * value * (1 + std::erf(value * 0.7071067811865475f));
    return half ? __half2float(__float2half_rn(value)) : value;
}

void write_payload(const std::filesystem::path& path, const void* data, std::size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream) throw std::runtime_error("failed writing raw INT8 probe payload");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 9) throw std::invalid_argument("usage: sam_cuda_linear_probe INPUT int8-token-f32|int8-token-f16|int8-static-f32|int8-static-f16|fp8-f32|fp8-f16 NEW_DIR [ITERATIONS] [--summary-only] [--dump-int32] [--dump-fp8] [--fp8-algo-index=0..7]");
        const auto input = sam_probe::read_input(argv[1]);
        const std::string mode = argv[2];
        const bool fp8 = mode == "fp8-f32" || mode == "fp8-f16";
        const bool dynamic = mode == "int8-token-f32" || mode == "int8-token-f16";
        const bool fixed = mode == "int8-static-f32" || mode == "int8-static-f16";
        const bool half = mode.size() >= 3 && mode.substr(mode.size() - 3) == "f16";
        if (!fp8 && !dynamic && !fixed) throw std::invalid_argument("unsupported CUDA probe mode");
        int iterations = 50;
        bool have_iterations = false, summary_only = false, dump_int32 = false, dump_fp8 = false;
        int fp8_algorithm_index = -1;
        for (int argument = 4; argument < argc; ++argument) {
            const std::string option = argv[argument];
            if (option == "--summary-only" && !summary_only) summary_only = true;
            else if (option == "--dump-int32" && !dump_int32) dump_int32 = true;
            else if (option == "--dump-fp8" && !dump_fp8) dump_fp8 = true;
            else if (option.rfind("--fp8-algo-index=", 0) == 0 && fp8_algorithm_index < 0) {
                const std::string value = option.substr(std::strlen("--fp8-algo-index="));
                if (value.size() != 1 || value[0] < '0' || value[0] > '7')
                    throw std::invalid_argument("FP8 heuristic index must be in [0,7]");
                fp8_algorithm_index = value[0] - '0';
            }
            else if (!have_iterations && option.rfind("--", 0) != 0) {
                iterations = sam_probe::iterations(argv[argument]);
                have_iterations = true;
            } else throw std::invalid_argument("unsupported or duplicate probe argument");
        }
        if (dump_int32 && fp8) throw std::invalid_argument("raw INT32 dump requires an INT8 mode");
        if (!fp8 && (dump_fp8 || fp8_algorithm_index >= 0))
            throw std::invalid_argument("FP8 diagnostics require an FP8 mode");
        const std::filesystem::path output_directory = argv[3];
        if (!std::filesystem::create_directory(output_directory)) throw std::runtime_error("probe output must be new");
        cuda_check(cudaSetDevice(0));
        cudaDeviceProp device{};
        cuda_check(cudaGetDeviceProperties(&device, 0));
        if (fp8 && device.major * 10 + device.minor < 89) throw std::runtime_error("FP8 probe requires compute capability 8.9+");
        Stream stream;
        const std::size_t x_count = std::size_t(input.k) * input.n, w_count = std::size_t(input.k) * input.m;
        const std::size_t y_count = std::size_t(input.m) * input.n;
        // Even at the format's maximum K, 127 * 127 * K fits signed INT32.
        for (std::size_t i = 0; i < input.samples.size(); ++i)
            if (!std::isfinite(input.samples[i] / input.scale[i % input.k]))
                throw std::runtime_error("channel balancing overflowed an activation");
        Buffer x(x_count * 4), weights(w_count), packed(x_count), scale(input.k * 4), bias(input.m * 4);
        Buffer sw(input.m * 4), sx(input.n * 4), scalar_scales(8), result(y_count * (half ? 2 : 4));
        Buffer sums(!fp8 && half ? y_count * 4 : 0), workspace;
        void* raw = !fp8 && half ? sums.data : result.data;
        x.upload(input.expanded_input());
        scale.upload(input.scale);
        bias.upload(input.bias);
        float weight_maximum = 0;
        for (std::size_t i = 0; i < w_count; ++i) {
            const float value = input.weights[i] * input.scale[i % input.k];
            if (!std::isfinite(value)) throw std::runtime_error("channel balancing overflowed a weight");
            weight_maximum = std::max(weight_maximum, std::abs(value));
        }
        const float weight_step = weight_maximum == 0 ? 1 : std::max(weight_maximum / 448.0f, FLT_MIN);
        const float input_step = input.scaled_activation_max == 0 ? 1 :
            std::max(input.scaled_activation_max / (fp8 ? 448.0f : 127.0f), FLT_MIN);
        scalar_scales.upload(std::vector<float>{weight_step, input_step});
        if (fixed) sx.upload(std::vector<float>(input.n, input_step));
        {
            Buffer original(w_count * 4);
            original.upload(input.weights);
            if (fp8) pack_fp8<true><<<(w_count + 255) / 256, 256, 0, stream.handle>>>(original.as<float>(), scale.as<float>(), weights.as<__nv_fp8_e4m3>(), w_count, input.k, weight_step);
            else pack_int8_rows<true><<<input.m, 256, 0, stream.handle>>>(original.as<float>(), scale.as<float>(), weights.as<int8_t>(), sw.as<float>(), input.k);
            cuda_check(cudaGetLastError());
            stream.synchronize();
        }
        auto quantize = [&] {
            if (fp8) pack_fp8<false><<<(x_count + 255) / 256, 256, 0, stream.handle>>>(x.as<float>(), scale.as<float>(), packed.as<__nv_fp8_e4m3>(), x_count, input.k, input_step);
            else if (dynamic) pack_int8_rows<false><<<input.n, 256, 0, stream.handle>>>(x.as<float>(), scale.as<float>(), packed.as<int8_t>(), sx.as<float>(), input.k);
            else pack_static_int8<<<(x_count + 255) / 256, 256, 0, stream.handle>>>(x.as<float>(), scale.as<float>(), packed.as<int8_t>(), x_count, input.k, input_step);
        };
        auto epilogue = [&] {
            const auto blocks = (y_count + 255) / 256;
            if (fp8 && half) finish_fp8<true><<<blocks, 256, 0, stream.handle>>>(raw, bias.as<float>(), y_count, input.m, input.gelu);
            else if (fp8) finish_fp8<false><<<blocks, 256, 0, stream.handle>>>(raw, bias.as<float>(), y_count, input.m, input.gelu);
            else if (half) finish_int8<true><<<blocks, 256, 0, stream.handle>>>(static_cast<int32_t*>(raw), result.data, sw.as<float>(), sx.as<float>(), bias.as<float>(), y_count, input.m, input.gelu);
            else finish_int8<false><<<blocks, 256, 0, stream.handle>>>(static_cast<int32_t*>(raw), result.data, sw.as<float>(), sx.as<float>(), bias.as<float>(), y_count, input.m, input.gelu);
        };
        quantize();
        stream.synchronize();
        Gemm gemm;
        gemm.prepare(input, fp8, half, scalar_scales, weights, packed, raw, workspace, stream, fp8_algorithm_index);
        auto run = [&] {
            quantize();
            blas_check(gemm.run(weights.data, packed.data, raw, workspace, stream.handle));
            epilogue();
            cuda_check(cudaGetLastError());
        };
        const auto warmup = std::chrono::steady_clock::now();
        int warmup_calls = 0;
        do {
            for (int i = 0; i < 5; ++i) { run(); ++warmup_calls; }
            stream.synchronize();
        } while (warmup_calls < 10 || sam_probe::milliseconds(warmup) < 250);
        std::vector<double> timing, quant_ms, gemm_ms, epilogue_ms, gpu_ms;
        for (int i = 0; i < iterations; ++i) {
            const auto start = std::chrono::steady_clock::now();
            run();
            stream.synchronize();
            timing.push_back(sam_probe::milliseconds(start));
        }
        Event start, after_quantize, after_gemm, end;
        for (int i = 0; i < 10; ++i) {
            start.record(stream.handle);
            quantize();
            after_quantize.record(stream.handle);
            blas_check(gemm.run(weights.data, packed.data, raw, workspace, stream.handle));
            after_gemm.record(stream.handle);
            epilogue();
            end.record(stream.handle);
            stream.synchronize();
            quant_ms.push_back(after_quantize.elapsed(start));
            gemm_ms.push_back(after_gemm.elapsed(after_quantize));
            epilogue_ms.push_back(end.elapsed(after_gemm));
            gpu_ms.push_back(end.elapsed(start));
        }
        std::vector<float> values;
        if (half) {
            const auto half_values = result.download<__half>();
            values.resize(half_values.size());
            std::transform(half_values.begin(), half_values.end(), values.begin(), [](__half value) { return __half2float(value); });
        } else values = result.download<float>();
        const auto numerical = sam_probe::error(values, input.reference, input.reference.size());

        // Independent scalar checks after timing. Check raw INT32 dots exactly,
        // and FP8 raw accumulation and the final bias/erf result separately.
        quantize();
        blas_check(gemm.run(weights.data, packed.data, raw, workspace, stream.handle));
        stream.synchronize();
        const auto qw = weights.download<unsigned char>(), qa = packed.download<unsigned char>();
        const auto weight_scales = fp8 ? std::vector<float>() : sw.download<float>();
        const auto input_scales = fp8 ? std::vector<float>() : sx.download<float>();
        std::vector<unsigned char> raw_bytes(y_count * (fp8 && half ? 2 : 4));
        cuda_check(cudaMemcpy(raw_bytes.data(), raw, raw_bytes.size(), cudaMemcpyDeviceToHost));
        if (dump_fp8) {
            write_payload(output_directory / "weight.e4m3", qw.data(), qw.size());
            write_payload(output_directory / "activation.e4m3", qa.data(), qa.size());
            write_payload(output_directory / (half ? "raw-f16.bin" : "raw-f32.bin"), raw_bytes.data(), raw_bytes.size());
            write_payload(output_directory / "weight-scale.f32", &weight_step, sizeof(weight_step));
            write_payload(output_directory / "activation-scale.f32", &input_step, sizeof(input_step));
        }
        const auto checks = std::min<std::size_t>(y_count, 512);
        for (std::size_t check = 0; check < checks; ++check) {
            const std::size_t index = check == 1 ? y_count - 1 : (check * 104729) % y_count;
            const auto row = index % input.m, token = index / input.m;
            double sum = 0, absolute = 0;
            for (std::size_t k = 0; k < input.k; ++k) {
                const double w = fp8 ? decode_fp8(qw[row * input.k + k]) : int8_t(qw[row * input.k + k]);
                const double a = fp8 ? decode_fp8(qa[token * input.k + k]) : int8_t(qa[token * input.k + k]);
                sum += w * a;
                absolute += std::abs(w * a);
            }
            float raw_value = 0, expected = 0;
            if (!fp8) {
                int32_t actual = 0;
                std::memcpy(&actual, raw_bytes.data() + index * 4, 4);
                if (sum != double(actual)) throw std::runtime_error("INT32 dot differs from exact scalar products");
                raw_value = float(actual) * weight_scales[row] * input_scales[token];
                expected = float(sum) * weight_scales[row] * input_scales[token];
            } else {
                expected = float(sum * double(weight_step) * double(input_step));
                if (half) {
                    __half actual;
                    std::memcpy(&actual, raw_bytes.data() + index * 2, 2);
                    raw_value = __half2float(actual);
                    expected = __half2float(__float2half_rn(expected));
                } else std::memcpy(&raw_value, raw_bytes.data() + index * 4, 4);
                const double tolerance = (half ? 0.0015 : 2e-5) * (1 + std::abs(expected)) +
                    5e-7 * absolute * double(weight_step) * double(input_step);
                if (!std::isfinite(raw_value) || std::abs(raw_value - expected) > tolerance) {
                    float unscaled = 0;
                    if (!half) {
                        scalar_scales.upload(std::vector<float>{1, 1});
                        blas_check(gemm.run(weights.data, packed.data, raw, workspace, stream.handle));
                        stream.synchronize();
                        cuda_check(cudaMemcpy(&unscaled, static_cast<float*>(raw) + index, sizeof(float), cudaMemcpyDeviceToHost));
                    }
                    std::ofstream failure(output_directory / "kernel-check-failure.json");
                    failure << std::setprecision(12) << "{\"mode\":\"" << mode << "\",\"index\":" << index
                        << ",\"actual\":" << raw_value << ",\"expected\":" << expected << ",\"tolerance\":" << tolerance
                        << ",\"decoded_dot\":" << sum << ",\"sum_abs_products\":" << absolute
                        << ",\"weight_scale\":" << weight_step << ",\"input_scale\":" << input_step
                        << ",\"unscaled_fp32_actual\":" << unscaled
                        << ",\"algorithm_id\":" << gemm.algorithm_id << ",\"algorithm_index\":" << gemm.algorithm_index
                        << ",\"heuristic_count\":" << gemm.heuristic_count
                        << ",\"numerical_flags\":" << gemm.numerical_flags << "}\n";
                    throw std::runtime_error("FP8 raw product differs from decoded scalar reference; see kernel-check-failure.json");
                }
            }
            const auto finished = host_finish(raw_value, input.bias[row], input.gelu, half);
            if (std::abs(values[index] - finished) > (half ? 0.0015 : 2e-5) * (1 + std::abs(finished)))
                throw std::runtime_error("epilogue differs from independent bias/erf reference");
        }
        if (dump_int32) {
            write_payload(output_directory / "weight.i8", qw.data(), qw.size());
            write_payload(output_directory / "activation.i8", qa.data(), qa.size());
            write_payload(output_directory / "raw-int32.bin", raw_bytes.data(), raw_bytes.size());
            write_payload(output_directory / "weight-scale.f32", weight_scales.data(), weight_scales.size() * sizeof(float));
            write_payload(output_directory / "activation-scale.f32", input_scales.data(), input_scales.size() * sizeof(float));
        }
        if (!summary_only) sam_probe::write_output(output_directory, values);
        std::ofstream report(output_directory / "probe.json");
        sam_probe::write_common(report, input, mode, timing, numerical);
        report << ",\"complete\":true,\"output_saved\":" << (summary_only ? "false" : "true")
            << ",\"raw_int32_saved\":" << (dump_int32 ? "true" : "false")
            << ",\"raw_int32_layout\":\"weight[M,K], activation[N,K], dot[N,M], row-major\""
            << ",\"warmup_calls\":" << warmup_calls << ",\"scalar_checks\":" << checks << ",\"int32_dots_exact\":" << (fp8 ? "null" : "true")
            << ",\"output_type\":\"" << (half ? "f16" : "f32") << "\",\"fp8_fast_accum\":false,\"algorithm_id\":" << gemm.algorithm_id
            << ",\"algorithm_index\":" << gemm.algorithm_index << ",\"heuristic_count\":" << gemm.heuristic_count
            << ",\"raw_fp8_saved\":" << (dump_fp8 ? "true" : "false")
            << ",\"numerical_flags\":" << gemm.numerical_flags
            << ",\"workspace_bytes\":" << workspace.bytes << ",\"explicit_inference_buffer_bytes\":"
            << x.bytes + weights.bytes + packed.bytes + scale.bytes + bias.bytes + sw.bytes + sx.bytes + scalar_scales.bytes + result.bytes + sums.bytes + workspace.bytes
            << ",\"fp32_producer_input_bytes\":" << x.bytes << ",\"packed_activation_bytes\":" << packed.bytes
            << ",\"packed_weight_bytes\":" << weights.bytes << ",\"output_bytes\":" << result.bytes
            << ",\"additional_int32_temporary_bytes\":" << sums.bytes
            << ",\"memory_scope\":\"Explicit live inference buffers; excludes context/library pools and post-timing validation allocations\""
            << ",\"quantization\":\"F32 scale division, nearest-even, symmetric INT8 or saturating E4M3\""
            << ",\"gpu_event_scope\":\"Separate diagnostic pass with event overhead; full wall timing has no events\""
            << ",\"quantize_gpu_ms\":";
        sam_probe::write_array(report, quant_ms);
        report << ",\"gemm_gpu_ms\":"; sam_probe::write_array(report, gemm_ms);
        report << ",\"epilogue_gpu_ms\":"; sam_probe::write_array(report, epilogue_ms);
        report << ",\"chain_gpu_ms\":"; sam_probe::write_array(report, gpu_ms);
        report << ",\"cuda_runtime\":" << CUDART_VERSION << ",\"cublaslt_version\":" << cublasLtGetVersion()
            << ",\"compute_capability\":" << device.major * 10 + device.minor << "}\n";
        if (!report) throw std::runtime_error("failed writing CUDA probe report");
        std::cout << mode << " median " << sam_probe::quantile(timing, 0.5) << " ms, relative L2 " << numerical.relative_l2 << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_cuda_linear_probe: " << error.what() << '\n';
        return 1;
    }
}
