#include "runtime/ggml.hpp"
#include "backend_test_support.hpp"
#include "ggml-sam.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Capture {
    std::vector<char> payload;
    std::size_t calls = 0, allocation = 0;
    std::exception_ptr error;
    static void observe(const ggml_tensor*, const void* data, std::size_t bytes,
                        std::size_t allocation, void* stream, void* user) noexcept {
        auto& self = *static_cast<Capture*>(user);
        try {
            ++self.calls;
            self.allocation = allocation;
            self.payload.resize(bytes);
            const auto cuda_stream = static_cast<cudaStream_t>(stream);
            require(cudaMemcpyAsync(self.payload.data(), data, bytes, cudaMemcpyDeviceToHost, cuda_stream) == cudaSuccess,
                    "cannot copy actual native MMQ RHS scratch");
            require(cudaStreamSynchronize(cuda_stream) == cudaSuccess, "cannot synchronize actual MMQ capture");
        } catch (...) { self.error = std::current_exception(); }
    }
};

std::vector<float> check_and_decode(const std::vector<char>& packed, const std::vector<float>& source, int k, int n) {
    const int padded = (k + 511) / 512 * 512;
    require(packed.size() == std::size_t(padded / 128 * n * 144), "actual MMQ payload byte count differs");
    std::vector<float> result(source.size());
    for (int tile = 0; tile < padded / 128; ++tile) for (int row = 0; row < n; ++row)
        for (int group = 0; group < 4; ++group) {
            const auto* bytes = packed.data() + (tile * n + row) * 144;
            float scale;
            std::memcpy(&scale, bytes + group * 4, 4);
            require(std::isfinite(scale) && scale >= 0, "actual MMQ scale invalid");
            const int begin = tile * 128 + group * 32;
            double maximum = 0;
            for (int i = 0; i < 32; ++i) if (begin + i < k)
                maximum = std::max(maximum, std::abs(double(source[row * k + begin + i])));
            const double expected = maximum / 127.;
            const float nearest = static_cast<float>(expected);
            const double ulp = double(std::nextafter(nearest, std::numeric_limits<float>::infinity())) - nearest;
            require(std::abs(double(scale) - expected) <= 4 * ulp, "actual MMQ scale differs from independent reference");
            for (int i = 0; i < 32; ++i) {
                std::int8_t code;
                std::memcpy(&code, bytes + 16 + group * 32 + i, 1);
                require(code != -128, "actual MMQ code outside [-127,127]");
                const double value = begin + i < k ? source[row * k + begin + i] : 0.;
                const double target = maximum == 0 ? 0. : value * 127. / maximum;
                const double exact_code = std::round(target);
                const float nearest_target = static_cast<float>(target);
                const double code_ulp = std::abs(double(std::nextafter(nearest_target, std::numeric_limits<float>::infinity())) - nearest_target);
                const bool tie = std::abs(std::abs(target) - std::floor(std::abs(target)) - .5) <= 4 * code_ulp;
                require(code == exact_code || (tie && std::abs(code - exact_code) == 1), "actual MMQ code differs from independent rounding reference");
                if (begin + i < k) result[row * k + begin + i] = scale * code;
                else require(code == 0 && scale == 0, "actual MMQ padding is not zero");
            }
        }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("usage: test_native_activation [NEW_CAPTURE_DIR]");
        const std::filesystem::path directory = argc == 2 ? argv[1] : "";
        const auto save = [&](const std::string& name, const void* data, std::size_t bytes) {
            if (directory.empty()) return;
            std::ofstream file(directory / name, std::ios::binary);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        };
        using namespace sam::internal;
        if (!sam::test::cuda_available()) {
#if SAM_TEST_REQUIRE_CUDA_DEVICE
            throw std::runtime_error("CUDA device required for native activation check");
#else
            std::cout << "CUDA device unavailable\n";
            return 77;
#endif
        }
        const auto reg = ggml_backend_reg_by_name("CUDA");
        const auto configure = reinterpret_cast<ggml_sam_q8_rhs_configure>(ggml_backend_reg_get_proc_address(reg, "ggml_cuda_sam_q8_rhs_configure_v1"));
        const auto query = reinterpret_cast<ggml_sam_q8_rhs_query>(ggml_backend_reg_get_proc_address(reg, "ggml_cuda_sam_q8_rhs_query_v1"));
        const auto install = reinterpret_cast<ggml_sam_q8_rhs_install_capture>(ggml_backend_reg_get_proc_address(reg, "ggml_cuda_sam_q8_rhs_capture_v1"));
        require(configure && query && install, "native Q8 RHS versioned extension missing");
        {
            auto context = make_context(4);
            auto* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_Q8_0, 32, 32);
            auto* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 32, 1);
            if (!configure(ggml_mul_mat(context.get(), weight, input), 0)) {
                CudaActivationPolicy policy;
                bool rejected = false;
                try { policy.initialize(0); }
                catch (const std::runtime_error& error) {
                    rejected = std::string(error.what()).find("requires supported CUDA hardware") != std::string::npos;
                }
                require(rejected && !policy.enabled(), "unsupported native activation was not explicitly rejected");
                std::cout << "Native Q8 RHS unsupported by selected CUDA backend; runtime rejection verified\n";
                return 77;
            }
        }
        if (!directory.empty() && !std::filesystem::create_directory(directory))
            throw std::invalid_argument("native activation capture requires a new output directory");
        sam::BackendOptions options{sam::Backend::Cuda, 4};
        options.activation = sam::ActivationMode::CudaVisionQ8Rhs;
        GgmlRuntime runtime(options, false, true);
        double maximum_error = 0;
        std::size_t checked = 0;
        for (const int k : {32, 256, 4736}) for (const int n : {1, 3, 33}) {
            constexpr int m = 65;
            auto context = make_context(4);
            auto* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_Q8_0, k, m);
            BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), runtime.weights_backend()));
            require(bool(buffer), "cannot allocate native activation fixture weights");
            ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::vector<float> original(k * m), decoded(k * m), source(k * n);
            for (std::size_t i = 0; i < original.size(); ++i) original[i] = float(int(i * 17 % 97) - 48) * .0073f;
            const auto* traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
            std::vector<char> packed(ggml_nbytes(weight));
            traits->from_float_ref(original.data(), packed.data(), original.size());
            traits->to_float(packed.data(), decoded.data(), decoded.size());
            ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
            sam::RuntimeStats stats;
            GraphExecution graph(runtime, 16, stats);
            auto* input = input_tensor(graph.context(), "native_q8_rhs_input", k, n);
            auto* output = ggml_mul_mat(graph.context(), weight, input);
            mark_vision_linear(output);
            graph.output(output);
            graph.allocate();
            ggml_sam_q8_rhs_info info{};
            require(query(output, &info) && info.version == 1 && info.launches == 0 && info.path == 0,
                    "native execution evidence was fabricated before compute");
            for (int repeat = 0; repeat < 2; ++repeat) {
                for (std::size_t i = 0; i < source.size(); ++i)
                    source[i] = float(int(i * 7 % 79) - 39) * (repeat ? -.013f : .009f);
                if (n > 1) std::fill(source.begin() + k, source.begin() + 2 * k, 0.f);
                source[0] = 127.; source[1] = .5f; source[2] = -.5f;
                upload(input, source, stats, &runtime);
                Capture capture;
                install(Capture::observe, &capture);
                try { graph.compute(); } catch (...) { install(nullptr, nullptr); throw; }
                install(nullptr, nullptr);
                if (capture.error) std::rethrow_exception(capture.error);
                require(capture.calls == 1, "forced MMQ did not expose exactly one real packing boundary");
                require(query(output, &info) && info.version == 1 && info.path == 1 && info.launches == std::uint64_t(repeat + 1) &&
                        info.payload_bytes == capture.payload.size() && info.allocation_bytes == capture.allocation &&
                        info.allocation_bytes >= info.payload_bytes, "actual MMQ dispatch/scratch evidence differs");
                const auto rhs = check_and_decode(capture.payload, source, k, n);
                const auto values = download(output, stats, &runtime);
                const auto prefix = "k" + std::to_string(k) + "-n" + std::to_string(n) + "-r" + std::to_string(repeat);
                save(prefix + "-rhs.f32", source.data(), source.size() * sizeof(float));
                save(prefix + "-actual-mmq.bin", capture.payload.data(), capture.payload.size());
                save(prefix + "-weight.q8_0", packed.data(), packed.size());
                save(prefix + "-output.f32", values.data(), values.size() * sizeof(float));
                for (int col = 0; col < n; ++col) for (int row = 0; row < m; ++row) {
                    double expected = 0;
                    for (int i = 0; i < k; ++i) expected += double(decoded[row * k + i]) * rhs[col * k + i];
                    const double error = std::abs(double(values[col * m + row]) - expected);
                    require(std::isfinite(values[col * m + row]) && error <= 5e-4 + 5e-5 * std::abs(expected),
                            "forced MMQ differs from actual-packed-operand scalar dot");
                    maximum_error = std::max(maximum_error, error); ++checked;
                }
            }
            require(stats.cuda_nodes == 2 && stats.cpu_nodes == 0 && stats.metal_nodes == 0, "explicit MMQ escaped CUDA");
            require(stats.activation_requests == 2 && stats.activation_mmq_launches == 2 &&
                    stats.activation_payload_peak_bytes > 0, "runtime did not bind actual activation evidence");
            // Input rejection must precede packing/dispatch, even under FTZ.
            for (const float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
                    1.e31f, 1.e-31f, std::numeric_limits<float>::denorm_min()}) {
                std::fill(source.begin(), source.end(), bad);
                upload(input, source, stats, &runtime);
                bool rejected = false;
                try { graph.compute(); } catch (const std::runtime_error& e) {
                    rejected = std::string(e.what()).find("each nonzero block maximum") != std::string::npos;
                }
                require(rejected && query(output, &info) && info.launches == 2, "invalid dynamic RHS silently executed");
                graph.allocate(); // Exception releases the workspace; restore a valid binding.
            }
            std::fill(source.begin(), source.end(), 0.f);
            upload(input, source, stats, &runtime);
            graph.compute();
            require(query(output, &info) && info.launches == 3, "native policy failed to recover after rejected input");
            for (const auto value : download(output, stats, &runtime)) require(value == 0.f, "zero RHS did not stay zero");
            auto dense = *weight;
            dense.type = GGML_TYPE_F32;
            auto incompatible = *output;
            incompatible.src[0] = &dense;
            require(!configure(&incompatible, 0), "unsupported dense LHS silently accepted");
            require(!configure(output, -1), "invalid CUDA device silently accepted");
            std::vector<char> retained(packed.size());
            ggml_backend_tensor_get(weight, retained.data(), 0, retained.size());
            require(retained == packed, "explicit activation changed resident weights");
        }
        std::cout << "Explicit MMQ packing bytes, actual dispatch, scalar dots and fail-closed recovery passed; outputs="
                  << checked << " maximum_error=" << maximum_error << '\n';
        if (!directory.empty()) {
            std::ofstream report(directory / "report.json");
            report.exceptions(std::ios::badbit | std::ios::failbit);
            report << "{\"complete\":true,\"contract\":\"ggml-cuda-q8-mmq-d4-v1\",\"cases\":9,\"captured_pairs\":18,"
                      "\"checked_outputs\":" << checked << ",\"maximum_error\":" << maximum_error << "}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native_activation: " << error.what() << '\n';
        return 1;
    }
}
