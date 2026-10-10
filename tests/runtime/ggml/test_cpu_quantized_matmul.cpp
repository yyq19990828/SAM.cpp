#include "../../../tools/benchmark/cpu_quantized_matmul.hpp"
#include "../../../tools/benchmark/execution_cost.hpp"
#include <iostream>

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template<class Function>
void rejects(Function function) {
    try { function(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid native CPU precision request was accepted");
}

void check_case(ggml_type type, int width, int rows, int columns, int threads,
                const char* scenario = "ordinary", bool qkv = false, bool strided = false) {
    sam_cpu_probe::Fixture data(type, width, rows, columns, scenario, qkv, strided);
    auto observer = std::make_shared<sam_cost::Observer>();
    sam_cpu_probe::Matmul native(data, sam::CpuComputeMode::NativeQuantized, threads, observer);
    sam_cpu_probe::Matmul baseline(data, sam::CpuComputeMode::F32, threads);
    native.compute();
    baseline.compute();
    const auto accuracy = sam_cpu_probe::validate(data, native.read(), true);
    sam_cpu_probe::validate(data, baseline.read(), false);
    require(std::string(native.profile()) == "ggml-quantized-cpu-native-v1" &&
            std::string(baseline.profile()) == "ggml-quantized-weights-f32-v1", "CPU arithmetic identity differs");
    if (data.scenario == "outlier")
        require(accuracy.max_f32_activation_difference > 1e-4, "native CPU arithmetic silently retained F32 RHS behavior");
    std::size_t matmuls = 0;
    for (const auto& node : observer->nodes) {
        require(node.category != "quantized_to_f32", "native CPU graph dequantized a weight matrix to F32");
        if (node.category != "matmul" || !node.calls) continue;
        ++matmuls;
        require(node.backend == "CPU" && node.inputs.front().type == ggml_type_name(type) &&
                node.inputs.at(1).type == "f32" && node.output_type == "f32" &&
                node.cpu_rhs_dot_type == ggml_type_name(sam::internal::cpu_quantized_rhs_type(type)),
                "native matmul cost observation lost actual placement or source-resolved RHS traits");
    }
    require(matmuls == (qkv ? 3u : 1u), "native CPU observer omitted a matmul");
    native.check_resident_weights();
    baseline.check_resident_weights();
    // Reuse the same graph after changing activations: RHS packing is per compute.
    for (auto& value : data.rhs) value *= -.37f;
    data.pack_rhs(data.packed_rhs);
    data.decode_rhs();
    native.upload_rhs(data.rhs);
    native.compute();
    sam_cpu_probe::validate(data, native.read(), true);
    native.set_observer({});
    native.compute();
    sam_cpu_probe::validate(data, native.read(), true);
    for (const auto& node : observer->nodes)
        if (node.category == "matmul") require(node.calls == 2, "normal compute retained a diagnostic callback");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1 && (argc != 3 || std::string(argv[1]) != "--cost-report"))
            throw std::invalid_argument("usage: test_cpu_quantized_matmul [--cost-report NEW_FILE]");
        rejects([] { sam::internal::GgmlRuntime r(sam_cpu_probe::options(sam::CpuComputeMode::NativeQuantized, 1), true); });
        for (const auto backend : {sam::Backend::Auto, sam::Backend::Metal, sam::Backend::Cuda}) {
            rejects([backend] {
                auto options = sam_cpu_probe::options(sam::CpuComputeMode::NativeQuantized, 1);
                options.backend = backend;
                sam::internal::GgmlRuntime runtime(options, false, true);
            });
        }
        rejects([] {
            auto options = sam_cpu_probe::options(sam::CpuComputeMode::F32, 1);
            options.cpu_compute = static_cast<sam::CpuComputeMode>(99);
            sam::internal::GgmlRuntime runtime(options, false, true);
        });
        rejects([] { sam::internal::cpu_quantized_rhs_type(GGML_TYPE_Q4_0); });
        for (const auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K}) {
            for (const auto threads : {1, 4}) {
                for (const auto columns : {1, 8, 17}) check_case(type, 256, 65, columns, threads, "ordinary", false, columns == 17);
                check_case(type, 256, 99, 9, threads, "ordinary", true);
                check_case(type, 1024, 33, 3, threads, "outlier");
                check_case(type, 256, 33, 3, threads, "zero");
            }
        }
        check_case(GGML_TYPE_Q8_0, 32, 3, 3, 4);
        check_case(GGML_TYPE_Q8_0, 4736, 33, 3, 4, "outlier");
        // Reject incompatible layouts before entering GGML's asserting kernel.
        auto context = sam::internal::make_context(8);
        auto* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_Q8_0, 256, 32);
        auto* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F16, 256, 3);
        auto* product = ggml_mul_mat(context.get(), weight, input);
        rejects([&] { sam::internal::validate_cpu_quantized_matmul(product); });
        if (argc == 3) {
            if (std::filesystem::exists(argv[2])) throw std::invalid_argument("cost report needs a new file");
            sam_cpu_probe::Fixture fixture(GGML_TYPE_Q4_K, 256, 65, 8, "outlier");
            auto observer = std::make_shared<sam_cost::Observer>();
            sam_cpu_probe::Matmul native(fixture, sam::CpuComputeMode::NativeQuantized, 4, observer);
            native.compute();
            sam_cpu_probe::validate(fixture, native.read(), true);
            auto output = sam_example::output_file(argv[2]);
            observer->write(output);
            output << '\n';
        }
        std::cout << "Native CPU Q8/Q6/Q5/Q4, decoded scalar references, outliers, tails, QKV/strided views and graph reuse passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
