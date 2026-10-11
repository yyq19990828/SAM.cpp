#include "../../../tools/benchmark/cpu_quantized_matmul.hpp"
#include "../../../tools/benchmark/execution_cost.hpp"
#include <fstream>
#include <iostream>

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void check_case(ggml_type type, sam::CpuComputeMode mode, int threads, bool qkv,
                std::int64_t width, std::ostream* report) {
    const bool native = mode == sam::CpuComputeMode::NativeQuantized;
    sam_cpu_probe::Fixture fixture(type, width, qkv ? 99 : 65, 33, "outlier", qkv);
    sam::internal::GgmlRuntime runtime(sam_cpu_probe::options(mode, threads), false, true);
    auto observer = std::make_shared<sam_cost::Observer>();
    runtime.set_graph_observer(observer);
    auto weights = sam::internal::make_context(4);
    auto* packed = ggml_new_tensor_2d(weights.get(), type, fixture.width, fixture.rows);
    auto* floating = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, fixture.width, fixture.rows);
    ggml_set_name(packed, "resident_packed_weight");
    ggml_set_name(floating, "resident_f32_control");
    sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(weights.get(), runtime.weights_backend()));
    require(static_cast<bool>(buffer), "BLAS coexistence fixture allocation failed");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(packed, fixture.packed.data(), 0, fixture.packed.size());
    ggml_backend_tensor_set(floating, fixture.source.data(), 0, fixture.source.size() * sizeof(float));
    sam::RuntimeStats stats;
    sam::internal::GraphExecution execution(runtime, 64, stats);
    auto* context = execution.context();
    auto* input = sam::internal::input_tensor(context, "shared_f32_rhs", fixture.width, fixture.columns);
    auto* control = ggml_mul_mat(context, floating, input);
    ggml_set_name(control, "floating_control");
    execution.output(control);
    std::vector<ggml_tensor*> products;
    const int parts = qkv ? 3 : 1;
    const auto rows = fixture.rows / parts;
    for (int part = 0; part < parts; ++part) {
        auto* weight = qkv ? ggml_view_2d(context, packed, fixture.width, rows, packed->nb[1],
                                        part * rows * packed->nb[1]) : packed;
        auto* product = ggml_mul_mat(context, weight, input);
        ggml_set_name(product, ("packed_product_" + std::to_string(part)).c_str());
        products.push_back(product);
        execution.output(product);
    }
    execution.allocate();
    double max_quantized_error = 0, max_control_error = 0;
    for (int reuse = 0; reuse < 2; ++reuse) {
        if (reuse) {
            for (auto& value : fixture.rhs) value *= -.37f;
            fixture.pack_rhs(fixture.packed_rhs);
            fixture.decode_rhs();
        }
        sam::internal::upload(input, fixture.rhs, stats, &runtime);
        execution.compute();
        std::vector<float> actual(static_cast<std::size_t>(fixture.rows * fixture.columns));
        for (int part = 0; part < parts; ++part) {
            const auto data = sam::internal::download(products[part], stats, &runtime);
            for (std::int64_t column = 0; column < fixture.columns; ++column)
                std::copy_n(data.data() + column * rows, rows,
                            actual.data() + column * fixture.rows + part * rows);
        }
        const auto accuracy = sam_cpu_probe::validate(fixture, actual, native);
        max_quantized_error = std::max(max_quantized_error, accuracy.max_effective_error);
        const auto dense = sam::internal::download(control, stats, &runtime);
        for (std::int64_t column = 0; column < fixture.columns; ++column) {
            for (std::int64_t row = 0; row < fixture.rows; ++row) {
                double reference = 0;
                for (std::int64_t k = 0; k < fixture.width; ++k)
                    reference += static_cast<double>(fixture.source[row * fixture.width + k]) *
                                 fixture.rhs[column * fixture.width + k];
                const auto error = std::abs(dense[column * fixture.rows + row] - reference);
                require(std::isfinite(dense[column * fixture.rows + row]) &&
                        error <= 5e-4 + 5e-5 * std::abs(reference), "BLAS positive control differs from scalar reference");
                max_control_error = std::max(max_control_error, error);
            }
        }
    }
    std::size_t controls = 0, quantized_products = 0, casts = 0;
    for (const auto& node : observer->nodes) {
        if (node.category == "quantized_to_f32") ++casts;
        if (node.category != "matmul" || !node.calls) continue;
        require(node.calls == 2, "BLAS coexistence graph reuse was not observed");
        require(node.output_type == "f32" && node.inputs.at(1).type == "f32", "coexistence output/RHS types changed");
        if (node.name == "floating_control") {
            ++controls;
            require(node.backend == "BLAS" && node.inputs.front().type == "f32",
                    "BLAS positive control did not execute on a real BLAS backend");
        } else {
            ++quantized_products;
            require(node.backend == (native ? "CPU" : "BLAS") &&
                    node.inputs.front().type == (native ? ggml_type_name(type) : "f32"),
                    "packed Q placement or default decoded BLAS placement differs");
        }
    }
    require(controls == 1 && quantized_products == static_cast<std::size_t>(parts) &&
            casts == (native ? 0u : static_cast<std::size_t>(parts)), "coexistence graph operations differ");
    require(stats.blas_nodes >= 2 && stats.cpu_nodes >= stats.blas_nodes &&
            !stats.cuda_nodes && !stats.metal_nodes, "coexistence backend accounting differs");
    std::vector<char> resident(fixture.packed.size());
    ggml_backend_tensor_get(packed, resident.data(), 0, resident.size());
    require(resident == fixture.packed && packed->type == type, "BLAS coexistence changed packed resident weights");
    if (report) {
        *report << "{\"format\":" << sam_example::json_string(ggml_type_name(type))
                << ",\"mode\":" << sam_example::json_string(native ? "native-quantized" : "f32")
                << ",\"threads\":" << threads << ",\"width\":" << width
                << ",\"qkv\":" << (qkv ? "true" : "false")
                << ",\"max_quantized_scalar_error\":" << max_quantized_error
                << ",\"max_f32_control_scalar_error\":" << max_control_error << ",\"execution_cost\":";
        observer->write(*report);
        *report << '}';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1 && (argc != 3 || std::string(argv[1]) != "--cost-report"))
            throw std::invalid_argument("usage: test_cpu_quantized_blas [--cost-report NEW_FILE]");
        auto blas = sam::internal::make_cpu_blas_backend(2);
        require(static_cast<bool>(blas.handle), "BLAS coexistence requires an actual registered BLAS backend");
        std::ofstream output;
        if (argc == 3) {
            if (std::filesystem::exists(argv[2])) throw std::invalid_argument("coexistence report needs a new file");
            output.open(argv[2]);
            if (!output) throw std::runtime_error("cannot create coexistence report");
            output << "{\"schema_version\":1,\"complete\":true,\"blas_required\":true,\"cases\":[";
        }
        std::size_t cases = 0;
        for (const auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K})
            for (const auto mode : {sam::CpuComputeMode::F32, sam::CpuComputeMode::NativeQuantized})
                for (const auto threads : {1, 4})
                    for (const bool qkv : {false, true}) {
                        if (output.is_open() && cases) output << ',';
                        check_case(type, mode, threads, qkv, 256, output.is_open() ? &output : nullptr);
                        ++cases;
                    }
        for (const auto mode : {sam::CpuComputeMode::F32, sam::CpuComputeMode::NativeQuantized}) {
            if (output.is_open()) output << ',';
            check_case(GGML_TYPE_Q8_0, mode, 4, false, 4736, output.is_open() ? &output : nullptr);
            ++cases;
        }
        if (output.is_open()) { output << "]}\n"; output.close(); require(!output.fail(), "coexistence report write failed"); }
        std::cout << "CPU/BLAS coexistence passed " << cases
                  << " cases: four Q formats, QKV views, 1/4 threads, 4736-wide rows and changed-RHS graph reuse\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
