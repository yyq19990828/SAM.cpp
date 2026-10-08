#include "backend_test_support.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void check_concat(ggml_backend_t backend, ggml_type type, int axis, bool padded, int planes = 5) {
    auto context = sam::internal::make_context(20, 16);
    std::array<int64_t, 4> a_shape{17, 3, 5, planes}, b_shape = a_shape;
    if (ggml_is_quantized(type)) a_shape[0] = b_shape[0] = ggml_blck_size(type) * 2;
    b_shape[axis] = axis == 0 && ggml_is_quantized(type) ? ggml_blck_size(type) : 7;
    if (planes > 65535) a_shape = b_shape = {1, 1, 1, planes};
    auto make = [&](const std::array<int64_t, 4>& shape) {
        auto* storage = ggml_new_tensor_4d(context.get(), type, shape[0], shape[1],
            shape[2] + (padded ? 2 : 0), shape[3]);
        auto* tensor = padded ? ggml_view_4d(context.get(), storage, shape[0], shape[1], shape[2], shape[3],
            storage->nb[1], storage->nb[2], storage->nb[3], 0) : storage;
        return std::make_pair(storage, tensor);
    };
    auto [as, a] = make(a_shape);
    auto [bs, b] = make(b_shape);
    auto* output = ggml_concat(context.get(), a, b, axis);
    if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("unsupported concat case");
    auto* graph = ggml_new_graph_custom(context.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) throw std::runtime_error("concat buffer allocation failed");
    std::vector<std::uint8_t> av(ggml_nbytes(as)), bv(ggml_nbytes(bs)), actual(ggml_nbytes(output));
    for (std::size_t i = 0; i < av.size(); ++i) av[i] = std::uint8_t((i * 37 + i / 97 + 11) % 251);
    for (std::size_t i = 0; i < bv.size(); ++i) bv[i] = std::uint8_t((i * 19 + i / 31 + 173) % 253);
    ggml_backend_tensor_set(as, av.data(), 0, av.size());
    ggml_backend_tensor_set(bs, bv.data(), 0, bv.size());
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("concat computation failed");
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size());
    // Independently select the source coordinates and compare every byte,
    // including compressed blocks, unusual float bit patterns and plane padding.
    const auto a_row_bytes = ggml_row_size(type, a->ne[0]);
    const auto row_bytes = ggml_row_size(type, output->ne[0]);
    for (int64_t i3 = 0; i3 < output->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < output->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < output->ne[1]; ++i1)
                for (std::size_t byte = 0; byte < row_bytes; ++byte) {
                    std::array<std::size_t, 4> index{byte, std::size_t(i1), std::size_t(i2), std::size_t(i3)};
                    const std::size_t boundary = axis == 0 ? a_row_bytes : std::size_t(a->ne[axis]);
                    const bool second = index[axis] >= boundary;
                    if (second) index[axis] -= boundary;
                    const auto* source = second ? b : a;
                    const auto& values = second ? bv : av;
                    const auto expected = values[index[0] + index[1] * source->nb[1] +
                        index[2] * source->nb[2] + index[3] * source->nb[3]];
                    const auto offset = byte + i1 * output->nb[1] + i2 * output->nb[2] + i3 * output->nb[3];
                    if (actual[offset] != expected) throw std::runtime_error("concat changed bytes or plane stride");
                }
    std::cout << ggml_type_name(type) << " concat axis=" << axis << " padded=" << padded
              << " planes=" << planes << " passed\n";
}
}

int main() {
    try {
        if (!sam::test::cuda_available()) return 77;
        sam::internal::GgmlRuntime runtime({sam::Backend::Cuda, 1}, true);
        for (auto type : {GGML_TYPE_I8, GGML_TYPE_I16, GGML_TYPE_I32, GGML_TYPE_I64,
                          GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K})
            for (int axis = 0; axis < 4; ++axis)
                for (bool padded : {false, true}) {
                    if (axis == 3 && padded && ggml_is_quantized(type)) continue;
                    check_concat(runtime.weights_backend(), type, axis, padded);
                }
        check_concat(runtime.weights_backend(), GGML_TYPE_I8, 1, true, 65537);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
