#include "backend_test_support.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {

void check_matrix(ggml_backend_t backend, ggml_type type, int batches, bool broadcast, bool fast = false) {
    constexpr int inner = 257, rows = 17, columns = 35;
    const int left_batches = broadcast ? 1 : batches;
    auto context = sam::internal::make_context(8, 16);
    auto* a = ggml_new_tensor_3d(context.get(), type, inner, rows, left_batches);
    auto* b = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, inner, columns, batches);
    auto* output = ggml_mul_mat(context.get(), a, b);
    ggml_prec_set_acc(output, GGML_PREC_F32);
    if (fast) sam::internal::configure_cuda_f16_node(output);
    if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("CUDA rejected batched precise matmul");
    auto* graph = ggml_new_graph_custom(context.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) throw std::runtime_error("matrix allocation failed");
    std::vector<float> left(ggml_nelements(a)), right(ggml_nelements(b));
    for (std::size_t i = 0; i < left.size(); ++i) left[i] = float(std::sin(0.071 * i) * 0.7 + 0.0003);
    for (std::size_t i = 0; i < right.size(); ++i) right[i] = float(std::cos(0.037 * i) * 0.8 - 0.0003);
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> stored(left.size());
        ggml_fp32_to_fp16_row(left.data(), stored.data(), int64_t(left.size()));
        ggml_backend_tensor_set(a, stored.data(), 0, stored.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(stored.data(), left.data(), int64_t(left.size()));
    } else ggml_backend_tensor_set(a, left.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, right.data(), 0, ggml_nbytes(b));
    if (fast) {
        for (auto& value : left) value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
        for (auto& value : right) value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("batched matrix execution failed");
    std::vector<float> actual(ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    double maximum = 0;
    for (int batch = 0; batch < batches; ++batch)
        for (int column = 0; column < columns; ++column)
            for (int row = 0; row < rows; ++row) {
                double expected = 0;
                for (int k = 0; k < inner; ++k)
                    expected += double(left[((broadcast ? 0 : batch) * rows + row) * inner + k]) *
                        double(right[(batch * columns + column) * inner + k]);
                const float value = actual[(batch * columns + column) * rows + row];
                if (!std::isfinite(value) || std::abs(double(value) - expected) > 2e-5 * (1 + std::abs(expected)))
                    throw std::runtime_error("batched precise matmul differs from double scalar dot products");
                maximum = std::max(maximum, std::abs(double(value) - expected));
            }
    std::cout << "CUDA " << ggml_type_name(type) << " batch=" << batches << " broadcast=" << broadcast
              << " reduced-inputs=" << fast << " matmul max abs=" << maximum << '\n';
}

struct Operand {
    ggml_tensor* storage;
    ggml_tensor* tensor;
    std::vector<float> values;
    int dimensions, tokens, heads, stride, offset;
    bool interleaved;

    Operand(ggml_context* context, int d, int t, int h, int batches, bool strided, double phase,
            bool interleaved_heads)
        : dimensions(d), tokens(t), heads(h), stride(d + (strided ? 8 : 0)), offset(strided ? 4 : 0),
          interleaved(interleaved_heads) {
        storage = ggml_new_tensor_4d(context, GGML_TYPE_F32, stride,
            interleaved ? heads : tokens, interleaved ? tokens : heads, batches);
        tensor = strided ? ggml_view_4d(context, storage, dimensions, storage->ne[1], storage->ne[2], batches,
            storage->nb[1], storage->nb[2], storage->nb[3], offset * sizeof(float)) : storage;
        if (interleaved) tensor = ggml_permute(context, tensor, 0, 2, 1, 3);
        values.resize(ggml_nelements(storage), -99.0f);
        for (int batch = 0; batch < batches; ++batch)
            for (int head = 0; head < heads; ++head)
                for (int token = 0; token < tokens; ++token)
                    for (int dimension = 0; dimension < dimensions; ++dimension) {
                        const auto index = index_of(batch, head, token, dimension);
                        values[index] = float(0.7 * std::sin(phase + 0.173 * token + 0.071 * dimension +
                            0.23 * head + 0.39 * batch));
                    }
    }
    int index_of(int batch, int head, int token, int dimension) const {
        return (interleaved ? (batch * tokens + token) * heads + head
                            : (batch * heads + head) * tokens + token) * stride + offset + dimension;
    }
    float at(int batch, int head, int token, int dimension) const {
        return values[index_of(batch, head, token, dimension)];
    }
};

void check_attention(ggml_backend_t backend, int dimensions, int queries, int keys, int heads,
                     int kv_heads, int batches, bool strided, int mask_heads, int mask_batches,
                     bool interleaved = false, bool fast = false) {
    auto context = sam::internal::make_context(24, 32);
    Operand q(context.get(), dimensions, queries, heads, batches, strided, 0.17, interleaved);
    Operand k(context.get(), dimensions, keys, kv_heads, batches, strided, 0.41, interleaved);
    Operand v(context.get(), dimensions, keys, kv_heads, batches, strided, 1.29, interleaved);
    ggml_tensor* mask = nullptr;
    std::vector<ggml_fp16_t> mask_values;
    const int mask_queries = ((queries + 63) / 64) * 64;
    if (mask_heads) {
        mask = ggml_new_tensor_4d(context.get(), GGML_TYPE_F16, keys, mask_queries, mask_heads, mask_batches);
        mask_values.resize(ggml_nelements(mask));
        for (int batch = 0; batch < mask_batches; ++batch)
            for (int head = 0; head < mask_heads; ++head)
                for (int query = 0; query < mask_queries; ++query)
                    for (int key = 0; key < keys; ++key) {
                        // Cover finite additive bias, masked tokens and a fully masked row.
                        const float bias = query == queries - 1 || (key + head + batch) % 7 == 0
                            ? -std::numeric_limits<float>::infinity() : float((key + query + head) % 5 - 2) * 0.125f;
                        mask_values[((batch * mask_heads + head) * mask_queries + query) * keys + key] = ggml_fp32_to_fp16(bias);
                    }
    }
    const float scale = 1.0f / std::sqrt(float(dimensions));
    auto* output = ggml_flash_attn_ext(context.get(), q.tensor, k.tensor, v.tensor, mask, scale, 0, 0);
    ggml_prec_set_acc(output, GGML_PREC_F32);
    if (fast) sam::internal::configure_cuda_f16_node(output);
    // A smaller sparse hint must not truncate the authoritative mask/key range.
    if (mask && !fast) ggml_flash_attn_ext_set_n_kv_max(output, keys / 2);
    if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("CUDA rejected precise attention boundary case");
    auto* graph = ggml_new_graph_custom(context.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) throw std::runtime_error("attention allocation failed");
    for (const auto* operand : {&q, &k, &v})
        ggml_backend_tensor_set(operand->storage, operand->values.data(), 0, ggml_nbytes(operand->storage));
    if (mask) ggml_backend_tensor_set(mask, mask_values.data(), 0, ggml_nbytes(mask));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("attention boundary execution failed");
    std::vector<float> actual(ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    std::vector<double> scores(keys);
    double maximum_error = 0;
    for (int batch = 0; batch < batches; ++batch)
        for (int head = 0; head < heads; ++head)
            for (int query = 0; query < queries; ++query) {
                const int kv_head = head / (heads / kv_heads);
                double maximum = -std::numeric_limits<double>::infinity();
                for (int key = 0; key < keys; ++key) {
                    double score = 0;
                    for (int dimension = 0; dimension < dimensions; ++dimension)
                        score += double(q.at(batch, head, query, dimension)) * k.at(batch, kv_head, key, dimension);
                    score *= scale;
                    if (mask) score += ggml_fp16_to_fp32(mask_values[
                        (((batch % mask_batches) * mask_heads + head % mask_heads) * mask_queries + query) * keys + key]);
                    scores[key] = score;
                    maximum = std::max(maximum, score);
                }
                double total = 0;
                for (auto& score : scores) {
                    score = maximum == -std::numeric_limits<double>::infinity() ? 0 : std::exp(score - maximum);
                    total += score;
                }
                for (int dimension = 0; dimension < dimensions; ++dimension) {
                    double expected = 0;
                    for (int key = 0; key < keys; ++key)
                        expected += scores[key] * v.at(batch, kv_head, key, dimension);
                    if (total > 0) expected /= total;
                    const float value = actual[((batch * queries + query) * heads + head) * dimensions + dimension];
                    if (!std::isfinite(value) || std::abs(double(value) - expected) > (fast && !mask ? 2e-3 : 5e-6)) {
                        std::cerr << "attention mismatch B" << batch << " H" << head << " Q" << query
                                  << " D" << dimension << " actual=" << value << " expected=" << expected << '\n';
                        throw std::runtime_error("CUDA attention differs from independent double masked softmax/value sum");
                    }
                    maximum_error = std::max(maximum_error, std::abs(double(value) - expected));
                }
            }
    std::cout << "CUDA attention D" << dimensions << " Q" << queries << " K" << keys << " H" << heads
              << " KV" << kv_heads << " B" << batches << " strides=" << strided
              << " interleaved=" << interleaved << " reduced-inputs=" << fast
              << " mask=" << mask_heads << 'x' << mask_batches << " max abs=" << maximum_error << '\n';
}

} // namespace

int main() {
    try {
        if (!sam::test::cuda_available()) return 77;
        sam::internal::GgmlRuntime runtime({sam::Backend::Cuda, 1}, false);
        auto* backend = runtime.weights_backend();
        for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16})
            for (const bool broadcast : {false, true}) check_matrix(backend, type, 3, broadcast);
        check_attention(backend, 32, 3, 1, 2, 1, 2, true, 0, 0);
        check_attention(backend, 64, 129, 137, 4, 2, 2, true, 4, 2);
        check_attention(backend, 32, 17, 131, 3, 1, 2, false, 1, 1);
        check_attention(backend, 64, 7, 5184, 2, 2, 1, false, 1, 1);
        // SAM reshapes [D*H,N,B] to [D,H,N,B], then permutes token/head axes.
        check_attention(backend, 64, 129, 137, 4, 2, 2, true, 4, 2, true);
        check_attention(backend, 32, 17, 131, 3, 1, 2, false, 1, 1, true);
        // Cross the query-tile boundary, including an all-masked final row.
        check_attention(backend, 32, 1025, 131, 4, 2, 2, true, 4, 2);
        check_attention(backend, 64, 1025, 137, 2, 1, 1, false, 1, 1, true);
        // More than one eight-head group, including masked and strided tails.
        check_attention(backend, 64, 129, 137, 9, 9, 2, true, 9, 2, true);
        check_attention(backend, 32, 1025, 131, 9, 9, 1, false, 1, 1);
        sam::internal::GgmlRuntime fast_runtime({sam::Backend::Cuda, 1, 0, sam::CudaComputeMode::F16}, false);
        for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16})
            for (const bool broadcast : {false, true}) check_matrix(fast_runtime.weights_backend(), type, 3, broadcast, true);
        check_attention(fast_runtime.weights_backend(), 64, 129, 137, 4, 2, 2, false, 1, 2, true, true);
        check_attention(fast_runtime.weights_backend(), 64, 1025, 137, 2, 1, 1, false, 0, 0, true, true);
        check_attention(fast_runtime.weights_backend(), 256, 129, 137, 1, 1, 3, false, 0, 0, false, true);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA numerics: " << error.what() << '\n';
        return 1;
    }
}
