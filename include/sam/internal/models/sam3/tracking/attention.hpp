#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_ATTENTION_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_ATTENTION_HPP

#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sam::internal::sam3 {

// Single 256-wide head. Each query tile sees the complete key sequence;
// splitting keys would change the softmax denominator. Inputs use [C, N].
inline ggml_tensor* tiled_memory_attention(ggml_context* ctx, ggml_tensor* query,
                                         ggml_tensor* key, ggml_tensor* value) {
    if (query->type != GGML_TYPE_F32 || key->type != GGML_TYPE_F32 || value->type != GGML_TYPE_F32 ||
        query->ne[0] != 256 || key->ne[0] != 256 || value->ne[0] != 256 ||
        query->ne[1] <= 0 || key->ne[1] <= 0 || key->ne[1] != value->ne[1] ||
        query->ne[2] != 1 || key->ne[2] != 1 || value->ne[2] != 1 ||
        query->ne[3] != 1 || key->ne[3] != 1 || value->ne[3] != 1)
        throw std::invalid_argument("memory attention requires F32 single-head [256,N] inputs");
    auto* transposed_value = ggml_cont(ctx, ggml_transpose(ctx, value));
    ggml_tensor* output = nullptr;
    for (int64_t start = 0; start < query->ne[1]; start += 128) {
        const auto count = std::min<int64_t>(128, query->ne[1] - start);
        auto* tile = ggml_view_2d(ctx, query, 256, count, query->nb[1], start * query->nb[1]);
        auto* scores = ggml_mul_mat(ctx, key, tile);
        ggml_prec_set_acc(scores, GGML_PREC_F32);
        auto* probabilities = ggml_soft_max(ctx, ggml_scale(ctx, scores, 1.0f / 16.0f));
        auto* attended = ggml_mul_mat(ctx, transposed_value, probabilities);
        ggml_prec_set_acc(attended, GGML_PREC_F32);
        output = output ? ggml_concat(ctx, output, attended, 1) : attended;
    }
    return output;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_ATTENTION_HPP
