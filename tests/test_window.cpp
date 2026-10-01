#include <sam/internal/runtime/ggml.hpp>

#include <array>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

struct WindowResult {
    std::vector<float> partitioned;
    std::vector<float> restored;
};

WindowResult run_window(ggml_backend_t backend, int channels, int width, int height, int window) {
    auto context = sam::internal::make_context(8, 16);
    auto* input = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, channels, width, height);
    auto* partitioned = ggml_win_part(context.get(), input, window);
    auto* restored = ggml_win_unpart(context.get(), partitioned, width, height, window);
    if (!ggml_backend_supports_op(backend, partitioned) || !ggml_backend_supports_op(backend, restored)) {
        throw std::runtime_error("Selected backend does not support window partition/restoration");
    }
    auto* graph = ggml_new_graph_custom(context.get(), 16, false);
    ggml_build_forward_expand(graph, restored);
    sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) throw std::runtime_error("Could not allocate window buffers");
    std::vector<float> values(static_cast<std::size_t>(channels) * width * height);
    std::iota(values.begin(), values.end(), 1.0f);
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    // A direct backend graph cannot schedule an unsupported operator onto CPU.
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Window graph execution failed");
    }
    WindowResult result{
        std::vector<float>(static_cast<std::size_t>(ggml_nelements(partitioned))),
        std::vector<float>(values.size())};
    ggml_backend_tensor_get(partitioned, result.partitioned.data(), 0, ggml_nbytes(partitioned));
    ggml_backend_tensor_get(restored, result.restored.data(), 0, ggml_nbytes(restored));
    if (result.restored != values) throw std::runtime_error("Window restoration changed image data");
    return result;
}

void check_windows(ggml_backend_t cpu, ggml_backend_t metal) {
    // Handwritten golden layouts fix channel order, window order and zero padding.
    const auto rectangle = run_window(cpu, 2, 3, 2, 2);
    if (rectangle.partitioned != std::vector<float>{1, 2, 3, 4, 7, 8, 9, 10, 5, 6, 0, 0, 11, 12, 0, 0}) {
        throw std::runtime_error("Rectangular window layout differs from the contract");
    }
    const auto padded = run_window(cpu, 1, 3, 3, 2);
    if (padded.partitioned != std::vector<float>{1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 0, 0, 9, 0, 0, 0}) {
        throw std::runtime_error("Window edge padding differs from the contract");
    }
    // The C37 rectangle overdispatches both kernels into a partial final group.
    for (const auto& shape : std::array<std::array<int, 4>, 6>{{
            {{2, 3, 2, 2}}, {{1, 3, 3, 2}}, {{3, 1, 1, 4}},
            {{37, 7, 5, 4}}, {{33, 2, 5, 1}}, {{1024, 72, 72, 24}}}}) {
        const auto reference = run_window(cpu, shape[0], shape[1], shape[2], shape[3]);
        if (metal) {
            const auto actual = run_window(metal, shape[0], shape[1], shape[2], shape[3]);
            if (actual.partitioned != reference.partitioned || actual.restored != reference.restored) {
                throw std::runtime_error("Metal window layout differs from CPU");
            }
        }
        std::cout << ggml_backend_name(metal ? metal : cpu) << " windows C" << shape[0]
                  << ' ' << shape[1] << 'x' << shape[2] << " W" << shape[3] << ": exact CPU match\n";
    }
}

} // namespace

int main() {
    try {
        sam::internal::GgmlRuntime cpu({sam::Backend::Cpu, 1}, false);
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto* device = ggml_backend_dev_get(i);
            if (std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) == "MTL") {
                sam::internal::GgmlRuntime metal({sam::Backend::Metal, 1}, false);
                if (metal.backend() != sam::Backend::Metal) throw std::runtime_error("Window probe used a different backend");
                check_windows(cpu.weights_backend(), metal.weights_backend());
                return 0;
            }
        }
        check_windows(cpu.weights_backend(), nullptr);
        std::cout << "CPU window layouts passed; Metal backend not available\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "window check: " << error.what() << '\n';
        return 1;
    }
}
