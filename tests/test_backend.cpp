#include <sam/internal/runtime/ggml.hpp>

#include <iostream>

namespace {

void check_scheduled_attribution(sam::internal::GgmlRuntime& runtime, sam::Backend expected,
                                 bool weight_backed = false) {
    sam::RuntimeStats stats;
    sam::internal::ContextPtr weight_context;
    sam::internal::BufferPtr weight_buffer;
    sam::internal::GraphExecution graph(runtime, 32, stats);
    ggml_tensor* left = nullptr;
    if (weight_backed) {
        weight_context = sam::internal::make_context(2);
        left = ggml_new_tensor_2d(weight_context.get(), GGML_TYPE_F32, 256, 256);
        weight_buffer.reset(ggml_backend_alloc_ctx_tensors(weight_context.get(), runtime.weights_backend()));
        if (!weight_buffer) throw std::runtime_error("Could not allocate the weight-backed attribution probe");
        ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    } else {
        left = sam::internal::input_tensor(graph.context(), "left", 256, 256);
    }
    auto* right = sam::internal::input_tensor(graph.context(), "right", 256, 32);
    auto* product = ggml_mul_mat(graph.context(), left, right);
    graph.output(product);
    graph.allocate();
    sam::internal::upload(left, std::vector<float>(256 * 256, 0.25f), stats);
    sam::internal::upload(right, std::vector<float>(256 * 32, 0.5f), stats);
    graph.compute();
    for (const float value : sam::internal::download(product, stats)) {
        if (value != 32.0f) throw std::runtime_error("Scheduled matrix graph did not execute correctly");
    }
    const std::uint64_t expected_cpu = expected == sam::Backend::Cpu ? 1 : 0;
    const std::uint64_t expected_metal = expected == sam::Backend::Metal ? 1 : 0;
    if (stats.cpu_nodes != expected_cpu || stats.metal_nodes != expected_metal || !stats.graph_partitions) {
        throw std::runtime_error(std::string(weight_backed ? "Weight-backed" : "Host-input") +
            " graph on " + ggml_backend_name(runtime.weights_backend()) +
            ": expected CPU=" + std::to_string(expected_cpu) + " Metal=" + std::to_string(expected_metal) +
            ", actual CPU=" + std::to_string(stats.cpu_nodes) + " Metal=" + std::to_string(stats.metal_nodes) +
            " partitions=" + std::to_string(stats.graph_partitions));
    }
    std::cout << ggml_backend_name(runtime.weights_backend()) << (weight_backed ? " weights" : " host inputs")
              << ": scheduled CPU nodes=" << stats.cpu_nodes
              << ", Metal nodes=" << stats.metal_nodes << '\n';
}

void check_unknown_attribution(sam::internal::GgmlRuntime& runtime, ggml_backend_t foreign) {
    sam::RuntimeStats stats;
    for (auto* backend : {static_cast<ggml_backend_t>(nullptr), foreign}) {
        bool rejected = false;
        try { runtime.record_node(backend, stats); }
        catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("unknown backend") != std::string::npos;
        }
        if (!rejected || stats.cpu_nodes || stats.metal_nodes) {
            throw std::runtime_error("Foreign or null backend was silently attributed to a known driver");
        }
    }
}

} // namespace

int main() {
    try {
        sam::internal::GgmlRuntime cpu({sam::Backend::Cpu, 1}, false);
        sam::internal::GgmlRuntime auto_fp32({sam::Backend::Auto, 1}, true);
        if (cpu.backend() != sam::Backend::Cpu || auto_fp32.backend() != sam::Backend::Cpu ||
            !cpu.promote_f16_weights() || !auto_fp32.promote_f16_weights()) {
            throw std::runtime_error("CPU/FP32 Auto selection did not preserve FP16 values through promotion");
        }
        check_scheduled_attribution(cpu, sam::Backend::Cpu);
        // A live backend from another runtime has the same kind, but is not an
        // owned scheduler backend. Never fabricate an invalid pointer to test it.
        check_unknown_attribution(cpu, auto_fp32.weights_backend());
        bool rejected = false;
        try { sam::internal::GgmlRuntime unsupported({sam::Backend::Metal, 1}, true); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Unsupported FP32 Metal combination was accepted");

        bool metal_registered = false;
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto* device = ggml_backend_dev_get(i);
            const std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
            if (name == "MTL") metal_registered = true;
        }
        sam::internal::GgmlRuntime auto_fp16({sam::Backend::Auto, 1}, false);
        if (metal_registered) {
            sam::internal::GgmlRuntime metal({sam::Backend::Metal, 1}, false);
            if (metal.backend() != sam::Backend::Metal || auto_fp16.backend() != sam::Backend::Metal ||
                metal.promote_f16_weights() || auto_fp16.promote_f16_weights()) {
                throw std::runtime_error("Metal/FP16 Auto selection did not keep packed FP16 weights");
            }
            // Pinned GGML assigns graph inputs to its CPU/default fallback.
            // Production weights carry buffer affinity to the selected driver.
            check_scheduled_attribution(metal, sam::Backend::Cpu);
            check_scheduled_attribution(metal, sam::Backend::Metal, true);
        } else {
            if (auto_fp16.backend() != sam::Backend::Cpu || !auto_fp16.promote_f16_weights()) {
                throw std::runtime_error("FP16 Auto did not select promoted CPU weights when Metal was unavailable");
            }
            rejected = false;
            try { sam::internal::GgmlRuntime missing({sam::Backend::Metal, 1}, false); }
            catch (const std::runtime_error& error) {
                rejected = std::string(error.what()).find("Metal") != std::string::npos;
            }
            if (!rejected) throw std::runtime_error("Unavailable Metal backend was not clearly rejected");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "backend check: " << error.what() << '\n';
        return 1;
    }
}
