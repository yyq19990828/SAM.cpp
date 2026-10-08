#ifndef SAM_CPP_SRC_MODELS_SAM3_VIDEO_ATTENTION_HPP
#define SAM_CPP_SRC_MODELS_SAM3_VIDEO_ATTENTION_HPP

#include "ggml.h"
#include "sam/internal/runtime/ggml/backend.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sam::internal::sam3 {

inline ggml_tensor* sam3_repeat_batch_3d(ggml_context* ctx, ggml_tensor* tensor, int64_t batch) {
    if (batch <= 0 || tensor->ne[3] != 1 || (tensor->ne[2] != 1 && tensor->ne[2] != batch))
        throw std::invalid_argument("SAM attention batch cannot be broadcast");
    if (tensor->ne[2] == batch) return tensor;
    return ggml_repeat_4d(ctx, tensor, tensor->ne[0], tensor->ne[1], batch, 1);
}

inline ggml_tensor* sam3_add_token_batch(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    if ((a->ne[2] != b->ne[2] && a->ne[2] != 1 && b->ne[2] != 1) || a->ne[3] != 1 || b->ne[3] != 1)
        throw std::invalid_argument("SAM token tensors have incompatible object batches");
    return a->ne[2] >= b->ne[2] ? ggml_add(ctx, a, b) : ggml_add(ctx, b, a);
}

// Single 256-wide head. Each query tile sees the complete key sequence;
// splitting keys would change the softmax denominator. The object batch is ne[2].
inline ggml_tensor* tiled_memory_attention(ggml_context* ctx, ggml_tensor* query,
                                         ggml_tensor* key, ggml_tensor* value,
                                         const AttentionExecutionPolicy& policy = {}) {
    const auto query_batch = query->ne[2];
    const auto key_batch = key->ne[2];
    const auto value_batch = value->ne[2];
    const auto batch = std::max(query_batch, std::max(key_batch, value_batch));
    if (query->type != GGML_TYPE_F32 || key->type != GGML_TYPE_F32 || value->type != GGML_TYPE_F32 ||
        query->ne[0] != 256 || key->ne[0] != 256 || value->ne[0] != 256 ||
        query->ne[1] <= 0 || key->ne[1] <= 0 || key->ne[1] != value->ne[1] ||
        batch <= 0 || (query_batch != 1 && query_batch != batch) ||
        (key_batch != 1 && key_batch != batch) || (value_batch != 1 && value_batch != batch) ||
        query->ne[3] != 1 || key->ne[3] != 1 || value->ne[3] != 1)
        throw std::invalid_argument("memory attention requires F32 single-head [256,N,B] inputs");
    if (policy.query_tile <= 0)
        throw std::invalid_argument("memory attention query tile must be positive");
    // GGML SET stores destination byte strides and offsets as signed int32.
    if (policy.assemble_in_place && !policy.fused_memory && query->ne[1] >
        std::numeric_limits<int>::max() / (256 * sizeof(float)) / batch)
        throw std::invalid_argument("memory attention output exceeds GGML SET addressing limits");
    query = sam3_repeat_batch_3d(ctx, query, batch);
    key = sam3_repeat_batch_3d(ctx, key, batch);
    value = sam3_repeat_batch_3d(ctx, value, batch);
    if (policy.fused_memory) {
        const auto queries = query->ne[1], keys = key->ne[1];
        // Memory attention has one head per object. Preserve the object axis
        // as the batch axis and let the backend select the arithmetic policy.
        auto reshape = [&](ggml_tensor* tensor, int64_t tokens) {
            return ggml_reshape_4d(ctx, ggml_is_contiguous(tensor) ? tensor : ggml_cont(ctx, tensor),
                                  256, tokens, 1, batch);
        };
        auto* output = ggml_flash_attn_ext(ctx, reshape(query, queries), reshape(key, keys),
                                         reshape(value, keys), nullptr, 1.0f / 16.0f, 0.0f, 0.0f);
        return ggml_reshape_3d(ctx, output, 256, queries, batch);
    }
    auto* transposed_value = ggml_cont(ctx, ggml_transpose(ctx, value));
    ggml_tensor* output = policy.assemble_in_place
        ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, query->ne[1], batch) : nullptr;
    for (int64_t start = 0; start < query->ne[1]; start += policy.query_tile) {
        const auto count = std::min<int64_t>(policy.query_tile, query->ne[1] - start);
        auto* tile = batch == 1
            ? ggml_view_2d(ctx, query, 256, count, query->nb[1], start * query->nb[1])
            : ggml_view_3d(ctx, query, 256, count, batch, query->nb[1], query->nb[2], start * query->nb[1]);
        auto* scores = ggml_mul_mat(ctx, key, tile);
        ggml_prec_set_acc(scores, GGML_PREC_F32);
        auto* probabilities = ggml_soft_max(ctx, ggml_scale(ctx, scores, 1.0f / 16.0f));
        auto* attended = ggml_mul_mat(ctx, transposed_value, probabilities);
        ggml_prec_set_acc(attended, GGML_PREC_F32);
        // Every destination tile is written exactly once. Chaining SET nodes
        // orders the writes without repeatedly copying a growing prefix.
        output = policy.assemble_in_place
            ? ggml_set_inplace(ctx, output, attended, output->nb[1], output->nb[2], output->nb[3],
                               static_cast<std::size_t>(start) * output->nb[1])
            : output ? ggml_concat(ctx, output, attended, 1) : attended;
    }
    return output;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_VIDEO_ATTENTION_HPP
