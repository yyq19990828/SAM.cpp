#ifndef SAM_CPP_INTERNAL_MODELS_SAM3_PROMPT_ENCODER_HPP
#define SAM_CPP_INTERNAL_MODELS_SAM3_PROMPT_ENCODER_HPP

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

// Empty-geometry encoder output: one learned CLS token in [D, 1, 1].
struct sam3_geom_result {
    struct ggml_tensor* geo_feats;  // [D, N_geo, 1]
    int n_tokens;                   // N_geo
};

inline sam3_geom_result sam3_build_geom_enc_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* img_feats,  // [D, N_img, 1] where N_img = H*H
    struct ggml_tensor* img_pe)     // [D, N_img, 1] sinusoidal PE
{
    const auto& ge = model.geom_enc;
    const int D = model.hparams.neck_dim;  // 256
    const int n_heads = 8;
    const int N_geo = 1;  // Empty geometry still has the learned CLS token.

    // Text-only prompts retain the learned CLS token. Project and normalize it
    // directly in the graph before the geometry transformer.
    auto* cls = ggml_reshape_3d(ctx, ge.cls_token, D, N_geo, 1);
    auto* x = ggml_add(ctx, ggml_mul_mat(ctx, ge.post_proj_w, cls), ge.post_proj_b);
    x = sam3_layer_norm(ctx, x, ge.norm_w, ge.norm_b);
    ggml_set_name(x, "geom_post_final_proj");

    // Transformer layers (3 layers: self-attn + cross-attn + FFN, pre-norm)
    for (int i = 0; i < (int)ge.layers.size(); ++i) {
        const auto& ly = ge.layers[i];

        // 1. Self-attention (pre-norm, pos_enc_at_attn=False)
        {
            auto* shortcut = x;
            auto* xn = sam3_layer_norm(ctx, x, ly.norm1_w, ly.norm1_b);

            // Q = K = V = norm(x) — no positional encoding at self-attention
            auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 0), xn),
                               ggml_view_1d(ctx, ly.sa_in_proj_b, D, 0));
            auto* K = ggml_add(ctx, ggml_mul_mat(ctx, ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], D * ly.sa_in_proj_w->nb[1]), xn),
                               ggml_view_1d(ctx, ly.sa_in_proj_b, D, D * sizeof(float)));
            auto* V = ggml_add(ctx, ggml_mul_mat(ctx, ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 2 * D * ly.sa_in_proj_w->nb[1]), xn),
                               ggml_view_1d(ctx, ly.sa_in_proj_b, D, 2 * D * sizeof(float)));

            const int64_t S = N_geo;
            const int64_t HD = D / n_heads;

            Q = ggml_reshape_4d(ctx, Q, HD, n_heads, S, 1);
            Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            K = ggml_reshape_4d(ctx, K, HD, n_heads, S, 1);
            K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
            V = ggml_reshape_4d(ctx, V, HD, n_heads, S, 1);
            V = ggml_permute(ctx, V, 0, 2, 1, 3);

            float scale = 1.0f / sqrtf((float)HD);
            auto* sa_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
            sa_out = ggml_reshape_3d(ctx, sa_out, D, S, 1);

            sa_out = ggml_mul_mat(ctx, ly.sa_out_proj_w, sa_out);
            sa_out = ggml_add(ctx, sa_out, ly.sa_out_proj_b);

            x = ggml_add(ctx, shortcut, sa_out);
        }

        // 2. Cross-attention (pre-norm, Q from x, K from img+PE, V from img)
        {
            auto* shortcut = x;
            auto* xn = sam3_layer_norm(ctx, x, ly.norm2_w, ly.norm2_b);

            // Q from normalized geometry tokens (no pos)
            // K from image features + PE
            // V from image features
            auto* q_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 0);
            auto* k_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], D * ly.ca_q_w->nb[1]);
            auto* v_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 2 * D * ly.ca_q_w->nb[1]);
            auto* q_b = ggml_view_1d(ctx, ly.ca_q_b, D, 0);
            auto* k_b = ggml_view_1d(ctx, ly.ca_q_b, D, D * sizeof(float));
            auto* v_b = ggml_view_1d(ctx, ly.ca_q_b, D, 2 * D * sizeof(float));

            auto* k_input = ggml_add(ctx, img_feats, img_pe);  // pos_enc_at_cross_attn_keys

            auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, xn), q_b);
            auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, k_input), k_b);
            auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, img_feats), v_b);

            const int64_t S_q = N_geo;
            const int64_t S_kv = img_feats->ne[1];
            const int64_t HD = D / n_heads;

            Q = ggml_reshape_4d(ctx, Q, HD, n_heads, S_q, 1);
            Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            K = ggml_reshape_4d(ctx, K, HD, n_heads, S_kv, 1);
            K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
            V = ggml_reshape_4d(ctx, V, HD, n_heads, S_kv, 1);
            V = ggml_permute(ctx, V, 0, 2, 1, 3);

            float scale = 1.0f / sqrtf((float)HD);
            auto* ca_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
            ca_out = ggml_reshape_3d(ctx, ca_out, D, S_q, 1);

            ca_out = ggml_mul_mat(ctx, ly.ca_out_w, ca_out);
            ca_out = ggml_add(ctx, ca_out, ly.ca_out_b);

            x = ggml_add(ctx, shortcut, ca_out);
        }

        // 3. FFN (pre-norm, ReLU)
        {
            auto* shortcut = x;
            auto* xn = sam3_layer_norm(ctx, x, ly.norm3_w, ly.norm3_b);
            auto* ffn = ggml_mul_mat(ctx, ly.ffn_fc1_w, xn);
            ffn = ggml_add(ctx, ffn, ly.ffn_fc1_b);
            ffn = ggml_relu(ctx, ffn);
            ffn = ggml_mul_mat(ctx, ly.ffn_fc2_w, ffn);
            ffn = ggml_add(ctx, ffn, ly.ffn_fc2_b);
            x = ggml_add(ctx, shortcut, ffn);
        }
        sam3_name_tensorf(x, "geom_layer%d_out", i);
    }

    // Final encode norm
    x = sam3_layer_norm(ctx, x, ge.encode_norm_w, ge.encode_norm_b);
    ggml_set_name(x, "geom_output");
    ggml_set_output(x);

    sam3_geom_result result;
    result.geo_feats = x;
    result.n_tokens = N_geo;
    return result;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_INTERNAL_MODELS_SAM3_PROMPT_ENCODER_HPP
