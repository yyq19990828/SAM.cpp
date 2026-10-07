#include "backend_test_support.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void encode(std::uint8_t* destination, ggml_type type, float value) {
    switch (type) {
        case GGML_TYPE_F32: std::memcpy(destination, &value, 4); break;
        case GGML_TYPE_F16: {
            const auto converted = ggml_fp32_to_fp16(value);
            std::memcpy(destination, &converted, 2); break;
        }
        case GGML_TYPE_BF16: {
            const auto converted = ggml_fp32_to_bf16(value);
            std::memcpy(destination, &converted, 2); break;
        }
        case GGML_TYPE_I32: {
            const auto converted = static_cast<std::int32_t>(value);
            std::memcpy(destination, &converted, 4); break;
        }
        default: throw std::runtime_error("unexpected conversion type");
    }
}

void check_copy(ggml_backend_t backend, ggml_type source_type, ggml_type destination_type,
                const std::array<int64_t, 4>& shape, const std::array<int, 4>& permutation,
                bool padded_source, bool padded_destination, bool flatten_destination) {
    auto context = sam::internal::make_context(16, 16);
    const auto source_size = ggml_type_size(source_type), destination_size = ggml_type_size(destination_type);
    auto* storage = ggml_new_tensor_4d(context.get(), source_type,
        shape[0] + (padded_source ? 3 : 0), shape[1], shape[2], shape[3]);
    const std::size_t source_offset = padded_source ? source_size : 0;
    auto* input = ggml_view_4d(context.get(), storage, shape[0], shape[1], shape[2], shape[3],
        storage->nb[1], storage->nb[2], storage->nb[3], source_offset);
    input = ggml_permute(context.get(), input, permutation[0], permutation[1], permutation[2], permutation[3]);
    std::array<int64_t, 4> output_shape{input->ne[0], input->ne[1], input->ne[2], input->ne[3]};
    if (flatten_destination) output_shape = {ggml_nelements(input), 1, 1, 1};
    auto* destination = ggml_new_tensor_4d(context.get(), destination_type,
        output_shape[0] + (padded_destination ? 3 : 0), output_shape[1], output_shape[2], output_shape[3]);
    const std::size_t destination_offset = padded_destination ? destination_size : 0;
    auto* target = ggml_view_4d(context.get(), destination, output_shape[0], output_shape[1],
        output_shape[2], output_shape[3], destination->nb[1], destination->nb[2], destination->nb[3], destination_offset);
    auto* output = ggml_cpy(context.get(), input, target);
    if (!ggml_backend_supports_op(backend, output))
        throw std::runtime_error(std::string("unsupported copy case: ") + ggml_type_name(source_type) +
            " to " + ggml_type_name(destination_type));
    auto* graph = ggml_new_graph_custom(context.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) throw std::runtime_error("copy allocation failed");
    std::vector<std::uint8_t> values(ggml_nbytes(storage));
    // Same-type copies preserve arbitrary representations, including NaNs,
    // infinities, signed zero and integer sign bits. Conversions use exactly
    // representable finite values so the host scalar conversion is a golden.
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = std::uint8_t((i * 37 + i / 97) % 256);
    if (source_type != destination_type)
        for (std::size_t i = 0; i < values.size() / source_size; ++i)
            encode(values.data() + i * source_size, source_type, float(int(i % 127) - 63));
    std::vector<std::uint8_t> expected(ggml_nbytes(destination), 0xa5), actual(expected.size());
    std::size_t index = 0;
    for (int64_t i3 = 0; i3 < input->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < input->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < input->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < input->ne[0]; ++i0, ++index) {
                    const auto from = source_offset + i0 * input->nb[0] + i1 * input->nb[1] +
                        i2 * input->nb[2] + i3 * input->nb[3];
                    const auto to = destination_offset + (flatten_destination ? index * destination_size :
                        i0 * target->nb[0] + i1 * target->nb[1] + i2 * target->nb[2] + i3 * target->nb[3]);
                    if (source_type == destination_type)
                        std::memcpy(expected.data() + to, values.data() + from, source_size);
                    else encode(expected.data() + to, destination_type, float(int((from / source_size) % 127) - 63));
                }
    ggml_backend_tensor_set(storage, values.data(), 0, values.size());
    std::fill(actual.begin(), actual.end(), 0xa5);
    ggml_backend_tensor_set(destination, actual.data(), 0, actual.size());
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("copy execution failed");
    ggml_backend_tensor_get(destination, actual.data(), 0, actual.size());
    if (actual != expected) throw std::runtime_error("copy changed values, logical order or destination padding");
}

} // namespace

int main() {
    try {
        if (!sam::test::cuda_available()) return 77;
        sam::internal::GgmlRuntime runtime({sam::Backend::Cuda, 1}, true);
        auto backend = runtime.weights_backend();
        int cases = 0;
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_I32}) {
            std::array<int, 4> permutation{0, 1, 2, 3};
            do {
                for (bool padded : {false, true}) {
                    check_copy(backend, type, type, {33, 65, 3, 2}, permutation, padded, false, false); ++cases;
                }
            } while (std::next_permutation(permutation.begin(), permutation.end()));
            for (const auto& permutation : {std::array<int, 4>{1, 2, 0, 3}, {2, 0, 1, 3}}) {
                check_copy(backend, type, type, {35, 37, 39, 9}, permutation, false, false, true); ++cases;
                check_copy(backend, type, type, {35, 37, 39, 9}, permutation, true, true, false); ++cases;
            }
        }
        for (auto from : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_I32})
            for (auto to : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_I32}) {
                if (from == to || (from == GGML_TYPE_I32 && to != GGML_TYPE_F32) ||
                    (to == GGML_TYPE_I32 && from != GGML_TYPE_F32)) continue;
                check_copy(backend, from, to, {31, 7, 5, 3}, {2, 0, 3, 1}, true, false, true); ++cases;
                check_copy(backend, from, to, {31, 7, 5, 3}, {2, 0, 3, 1}, true, true, false); ++cases;
            }
        // The tiled transpose cannot use CUDA grid.y beyond 65535. The copy
        // must still work using generic indexing, including a non-power-of-two divisor.
        check_copy(backend, GGML_TYPE_F32, GGML_TYPE_F32, {3, 65537 * 32LL, 1, 1},
            {1, 0, 2, 3}, false, false, false); ++cases;
        std::cout << cases << " CUDA copy layout/type/padding cases passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
