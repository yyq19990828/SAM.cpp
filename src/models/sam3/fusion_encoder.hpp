#ifndef SAM_CPP_SRC_MODELS_SAM3_FUSION_ENCODER_HPP
#define SAM_CPP_SRC_MODELS_SAM3_FUSION_ENCODER_HPP

// Selected SAM 3 image graph code adapted from PABannier/sam3.cpp
// revision 416186c501d060df7ca02989d49b38080f5f81f3.
/*
MIT License

Copyright (c) 2025-2026 Pierre-Antoine Bannier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "architecture.hpp"
#include "ops.hpp"
#include "ggml.h"
#include <cmath>
#include <cstdint>

namespace sam::internal::sam3 {

inline struct ggml_tensor* sam3_fenc_layer_forward(
    struct ggml_context* ctx,
    const sam3_fenc_layer& ly,
    struct ggml_tensor* x,
    struct ggml_tensor* prompt,
    struct ggml_tensor* pos,
    struct ggml_tensor* prompt_attn_bias,
    int n_heads) {
    // Self-attention: Q/K get positional encoding, V does not
    {
        auto* shortcut = x;
        auto* x_norm = sam3_layer_norm(ctx, x, ly.norm1_w, ly.norm1_b);
        auto* q_in = ggml_add(ctx, x_norm, pos);
        auto* k_in = ggml_add(ctx, x_norm, pos);

        const int64_t D = x->ne[0];

        auto* q_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], D * ly.sa_in_proj_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 2 * D * ly.sa_in_proj_w->nb[1]);

        auto* q_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 2 * D * sizeof(float));

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, k_in), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, x_norm), v_b);

        const int64_t N = x->ne[1];
        const int64_t B = x->ne[2];
        const int64_t HD = D / n_heads;

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N, B);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);

        float scale = 1.0f / sqrtf((float)HD);
        auto* sa_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        sa_out = ggml_reshape_3d(ctx, sa_out, D, N, B);

        sa_out = ggml_mul_mat(ctx, ly.sa_out_proj_w, sa_out);
        sa_out = ggml_add(ctx, sa_out, ly.sa_out_proj_b);

        x = ggml_add(ctx, shortcut, sa_out);
    }

    // Cross-attention: Q from image features, K/V from prompt tokens.
    // ca_q_w stores fused [D, 3*D] weights split as Q-proj, K-proj, V-proj.
    {
        auto* shortcut = x;
        auto* x_norm = sam3_layer_norm(ctx, x, ly.norm2_w, ly.norm2_b);
        const int64_t D = x->ne[0];

        auto* q_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], D * ly.ca_q_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 2 * D * ly.ca_q_w->nb[1]);

        auto* q_b = ggml_view_1d(ctx, ly.ca_q_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.ca_q_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.ca_q_b, D, 2 * D * sizeof(float));

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, x_norm), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, prompt), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, prompt), v_b);

        const int64_t N_q = x->ne[1];
        const int64_t N_kv = prompt->ne[1];
        const int64_t B = x->ne[2];
        const int64_t HD = D / n_heads;

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);

        auto* ca_mask = sam3_expand_token_attn_bias(ctx, prompt_attn_bias, N_q, n_heads, B);
        float scale = 1.0f / sqrtf((float)HD);
        auto* ca_out = ggml_flash_attn_ext(ctx, Q, K, V, ca_mask, scale, 0.0f, 0.0f);
        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);

        ca_out = ggml_mul_mat(ctx, ly.ca_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_out_b);

        x = ggml_add(ctx, shortcut, ca_out);
    }

    {
        auto* shortcut = x;
        auto* x_norm = sam3_layer_norm(ctx, x, ly.norm3_w, ly.norm3_b);

        auto* ffn = ggml_mul_mat(ctx, ly.ffn_fc1_w, x_norm);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc1_b);
        ffn = ggml_relu(ctx, ffn);
        ffn = ggml_mul_mat(ctx, ly.ffn_fc2_w, ffn);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc2_b);

        x = ggml_add(ctx, shortcut, ffn);
    }

    return x;
}

// Build full fusion encoder graph (6 layers).
// image_feats: [D, N, B] where N=5184 (72*72), D=256
// prompt_tokens: [D, T, B] text/exemplar features
// pos_enc: [D, N, B] sinusoidal positional encoding for image features
// Returns: conditioned_features [D, N, B]
inline struct ggml_tensor* sam3_build_fenc_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* image_feats,
    struct ggml_tensor* prompt_tokens,
    struct ggml_tensor* pos_enc,
    struct ggml_tensor* prompt_attn_bias = nullptr) {
    const auto& hp = model.hparams;
    auto* x = image_feats;

    for (int i = 0; i < hp.fenc_layers; ++i) {
        x = sam3_fenc_layer_forward(ctx, model.fenc.layers[i], x, prompt_tokens,
                                    pos_enc, prompt_attn_bias, hp.fenc_heads);
        sam3_name_tensorf(x, "fenc_layer%d_out", i);
    }

    return x;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_FUSION_ENCODER_HPP
