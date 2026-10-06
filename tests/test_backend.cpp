#include <sam/internal/runtime/ggml.hpp>
#include "backend_test_support.hpp"

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
    const std::uint64_t expected_cuda = expected == sam::Backend::Cuda ? 1 : 0;
    if (stats.cpu_nodes != expected_cpu || stats.metal_nodes != expected_metal ||
        stats.cuda_nodes != expected_cuda || !stats.graph_partitions) {
        throw std::runtime_error(std::string(weight_backed ? "Weight-backed" : "Host-input") +
            " graph on " + ggml_backend_name(runtime.weights_backend()) +
            ": expected CPU=" + std::to_string(expected_cpu) + " Metal=" + std::to_string(expected_metal) +
            ", actual CPU=" + std::to_string(stats.cpu_nodes) + " Metal=" + std::to_string(stats.metal_nodes) +
            " CUDA=" + std::to_string(stats.cuda_nodes) + " expected CUDA=" + std::to_string(expected_cuda) +
            " partitions=" + std::to_string(stats.graph_partitions));
    }
    const bool expects_blas = expected == sam::Backend::Cpu &&
        std::string(ggml_backend_name(runtime.backends().front())) == "BLAS";
    if (stats.blas_nodes != (expects_blas ? 1U : 0U) || stats.blas_nodes > stats.cpu_nodes)
        throw std::runtime_error("BLAS work was not attributed as a CPU subset");
    std::cout << ggml_backend_name(runtime.weights_backend()) << (weight_backed ? " weights" : " host inputs")
              << ": scheduled CPU nodes=" << stats.cpu_nodes
              << ", Metal nodes=" << stats.metal_nodes << ", CUDA nodes=" << stats.cuda_nodes << '\n';
}

void check_unknown_attribution(sam::internal::GgmlRuntime& runtime, ggml_backend_t foreign) {
    sam::RuntimeStats stats;
    for (auto* backend : {static_cast<ggml_backend_t>(nullptr), foreign}) {
        bool rejected = false;
        try { runtime.record_node(backend, stats); }
        catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("unknown backend") != std::string::npos;
        }
        if (!rejected || stats.cpu_nodes || stats.metal_nodes || stats.cuda_nodes) {
            throw std::runtime_error("Foreign or null backend was silently attributed to a known driver");
        }
    }
}

void check_cuda() {
    sam::internal::GgmlRuntime cuda({sam::Backend::Cuda, 1}, false);
    if (cuda.backend() != sam::Backend::Cuda || cuda.cuda_device() != 0 || cuda.device_name().empty() ||
        cuda.promote_f16_weights())
        throw std::runtime_error("Explicit CUDA selection did not retain device identity and packed F16 storage");
    check_scheduled_attribution(cuda, sam::Backend::Cuda);
    check_scheduled_attribution(cuda, sam::Backend::Cuda, true);
    sam::internal::GgmlRuntime cpu({sam::Backend::Cpu, 1}, false);
    check_unknown_attribution(cuda, cpu.weights_backend());
    auto* registration = ggml_backend_reg_by_name("CUDA");
    bool rejected = false;
    try {
        sam::internal::GgmlRuntime invalid({sam::Backend::Cuda, 1,
            static_cast<int>(ggml_backend_reg_dev_count(registration))}, false);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("out of range") != std::string::npos;
    }
    if (!rejected) throw std::runtime_error("CUDA accepted a non-visible device index");

    sam::RuntimeStats stats;
    auto workspace = std::make_shared<sam::internal::GraphWorkspace>(cuda, 32);
    {
        sam::internal::GraphExecution unsupported(cuda, 16, stats, workspace);
        auto* input = sam::internal::input_tensor(unsupported.context(), "cpu_only_input", 4);
        auto* output = ggml_map_custom1(unsupported.context(), input,
            [](ggml_tensor*, const ggml_tensor*, int, int, void*) {}, 1, nullptr);
        unsupported.output(output);
        rejected = false;
        try { unsupported.allocate(); }
        catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("MAP_CUSTOM1") != std::string::npos;
        }
        if (!rejected || workspace->allocated_bytes() || stats.cpu_nodes || stats.cuda_nodes)
            throw std::runtime_error("CUDA silently allocated or ran an unsupported CPU-only operator");
    }
    {
        sam::internal::GraphExecution graph(cuda, 16, stats, workspace);
        auto* input = sam::internal::input_tensor(graph.context(), "input", 4);
        auto* output = ggml_scale(graph.context(), input, 2.0f);
        graph.output(output);
        graph.allocate();
        sam::internal::upload(input, {1, 2, 3, 4}, stats);
        // A misplaced compute node must be rejected before launching any kernel.
        ggml_backend_sched_set_tensor_backend(workspace->scheduler(), output, cuda.backends().back());
        rejected = false;
        try { graph.compute(); }
        catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("fallback") != std::string::npos;
        }
        if (!rejected || stats.cpu_nodes || stats.cuda_nodes)
            throw std::runtime_error("CUDA placement guard executed a misplaced compute node");
        workspace->release_storage();
        graph.allocate();
        sam::internal::upload(input, {2, 3, 4, 5}, stats);
        graph.compute();
        if (sam::internal::download(output, stats) != std::vector<float>{4, 6, 8, 10})
            throw std::runtime_error("Placement rejection corrupted the reusable CUDA workspace");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (sam::test::cuda_requested(argc, argv)) {
            if (!sam::test::cuda_available()) return 77;
            check_cuda();
            return 0;
        }
        sam::internal::GgmlRuntime cpu({sam::Backend::Cpu, 1}, false);
        sam::internal::GgmlRuntime auto_fp32({sam::Backend::Auto, 1}, true);
        if (cpu.backend() != sam::Backend::Cpu || auto_fp32.backend() != sam::Backend::Cpu ||
            !cpu.promote_f16_weights() || !auto_fp32.promote_f16_weights()) {
            throw std::runtime_error("CPU/FP32 Auto selection did not preserve FP16 values through promotion");
        }
        check_scheduled_attribution(cpu, sam::Backend::Cpu);
        check_scheduled_attribution(cpu, sam::Backend::Cpu, true);
        if (std::string(ggml_backend_name(cpu.weights_backend())) != "CPU")
            throw std::runtime_error("CPU acceleration changed model weight ownership");
        // A live backend from another runtime has the same kind, but is not an
        // owned scheduler backend. Never fabricate an invalid pointer to test it.
        check_unknown_attribution(cpu, auto_fp32.weights_backend());
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
            sam::internal::GgmlRuntime metal_fp32({sam::Backend::Metal, 1}, true);
            if (metal_fp32.backend() != sam::Backend::Metal || metal_fp32.promote_f16_weights())
                throw std::runtime_error("Explicit FP32 Metal selection did not retain device weight storage");
            check_scheduled_attribution(metal_fp32, sam::Backend::Metal, true);
            // Pinned GGML assigns graph inputs to its CPU/default fallback.
            // Production weights carry buffer affinity to the selected driver.
            check_scheduled_attribution(metal, sam::Backend::Cpu);
            check_scheduled_attribution(metal, sam::Backend::Metal, true);
        } else {
            if (auto_fp16.backend() != sam::Backend::Cpu || !auto_fp16.promote_f16_weights()) {
                throw std::runtime_error("FP16 Auto did not select promoted CPU weights when Metal was unavailable");
            }
            for (const bool fp32 : {false, true}) {
                bool rejected = false;
                try { sam::internal::GgmlRuntime missing({sam::Backend::Metal, 1}, fp32); }
                catch (const std::runtime_error& error) {
                    rejected = std::string(error.what()).find("Metal") != std::string::npos;
                }
                if (!rejected) throw std::runtime_error("Unavailable Metal backend was not clearly rejected");
            }
        }
        if (!sam::test::cuda_available()) {
            bool rejected = false;
            try { sam::internal::GgmlRuntime missing({sam::Backend::Cuda, 1}, false); }
            catch (const std::runtime_error& error) {
                rejected = std::string(error.what()).find("CUDA") != std::string::npos;
            }
            if (!rejected) throw std::runtime_error("Unavailable CUDA was not clearly rejected");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "backend check: " << error.what() << '\n';
        return 1;
    }
}
