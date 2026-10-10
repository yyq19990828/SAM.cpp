#include "runtime/ggml.hpp"
#include "models/sam3/detector.hpp"
#include "../../../tools/benchmark/execution_cost.hpp"
#include <cmath>
#include <iostream>
#include <sstream>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        using namespace sam::internal;
        GgmlRuntime runtime({sam::Backend::Cpu, 2}, false, true);
        auto context = make_context(4);
        auto* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_Q8_0, 256, 2);
        ggml_set_name(weight, "cost_fixture_weight");
        BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), runtime.weights_backend()));
        require(static_cast<bool>(buffer), "cannot allocate cost fixture weights");
        std::vector<float> original(512, .5f), decoded(512);
        std::vector<char> packed(ggml_nbytes(weight));
        const auto* traits = ggml_get_type_traits(weight->type);
        traits->from_float_ref(original.data(), packed.data(), 512);
        traits->to_float(packed.data(), decoded.data(), 512);
        ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
        auto observer = std::make_shared<sam_cost::Observer>();
        observer->stage = "bounded_cpu_cost_test";
        runtime.set_graph_observer(observer);
        sam::RuntimeStats stats;
        GraphExecution execution(runtime, 16, stats);
        auto* input = input_tensor(execution.context(), "cost_fixture_input", 256, 3);
        auto* product = ggml_mul_mat(execution.context(), weight, input);
        ggml_set_name(product, "cost_fixture_product");
        auto* zero = input_tensor(execution.context(), "rpb_pres_zeros", 2, 3);
        auto* result = ggml_add(execution.context(), product, zero);
        execution.output(result);
        for (int repeat = 0; repeat < 2; ++repeat) {
            execution.allocate();
            upload(input, std::vector<float>(768, static_cast<float>(repeat + 1)), stats, &runtime);
            sam3::initialize_detector_zero_inputs(execution.context(), stats, &runtime);
            execution.compute();
            const auto output = download(result, stats, &runtime);
            for (std::size_t row = 0; row < output.size(); ++row) {
                double expected = 0;
                for (std::size_t k = 0; k < 256; ++k) expected += decoded[(row % 2) * 256 + k] * (repeat + 1);
                require(std::isfinite(output[row]) && std::abs(output[row] - expected) < .001,
                        "cost observation changed CPU numerical output");
            }
        }
        require(observer->graphs.size() == 2 && observer->graphs[0].compute_calls == 1 &&
                observer->graphs[1].reused && observer->graphs[1].compute_calls == 1,
                "cost observer lost a reused graph execution");
        std::size_t casts = 0, matmuls = 0;
        for (const auto& node : observer->nodes) {
            require(node.calls == 1 && node.wall_ms >= 0 && node.backend == "CPU", "invalid observed CPU node cost");
            if (node.category == "quantized_to_f32") {
                ++casts;
                require(node.inputs.front().type == "q8_0" && node.output_type == "f32", "cast report lost actual types");
            }
            if (node.category == "matmul") {
                ++matmuls;
                require(node.inputs.front().type == "f32" && node.accumulation_hint == "f32", "matmul precision was guessed");
            }
        }
        require(casts == 2 && matmuls == 2 && observer->transfers.size() == 6,
                "cost report omitted casts, matmuls or transfers");
        require(observer->graphs.front().arena_bytes > 0 && stats.host_upload_bytes == 2 * (768 + 6) * sizeof(float) &&
                stats.host_download_bytes == 2 * 6 * sizeof(float), "cost observation changed transfer accounting");
        runtime.set_graph_observer({});
        execution.compute(); // No bind: an evaluation callback must not outlive the observed compute.
        for (const auto& node : observer->nodes) require(node.calls == 1, "normal compute retained a diagnostic callback");
        std::ostringstream report;
        observer->write(report);
        require(report.str().find("\"performance_comparable\":false") != std::string::npos &&
                report.str().find("\"kernel_internal_rhs_packing\":\"NOT_COLLECTED\"") != std::string::npos,
                "cost report implied a benchmark or unobserved kernel arithmetic");
        std::cout << "CPU execution costs, graph reuse and callback lifetime passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
