#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MEMORY_ATTENTION_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MEMORY_ATTENTION_HPP

// Adapted from PABannier/sam3.cpp, revision 416186c501d060df7ca02989d49b38080f5f81f3.
// Copyright (c) 2025-2026 Pierre-Antoine Bannier. MIT: licenses/sam3.cpp-MIT.txt.

#include "../architecture.hpp"
#include "../vision.hpp"
#include "../ops.hpp"
#include "attention.hpp"
#include <algorithm>

namespace sam::internal::sam3 {

inline struct ggml_tensor* sam3_build_mem_attn_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* curr_tokens,   // [D, N, B]
    struct ggml_tensor* src_pos,       // [D, N, B]
    struct ggml_tensor* prompt,        // [MD, M_total, B]
    struct ggml_tensor* prompt_pos,    // [MD, M_total, B]
    struct ggml_tensor* rope_freqs,    // [2, D/2, N], shared across B
    struct ggml_tensor* rope_k_freqs,  // [2, D/2, M_spatial], shared across B or nullptr
    int num_obj_ptr_tokens,
    const AttentionExecutionPolicy& attention_policy = {}) {
    const auto& ma = model.mem_attn;
    const int D = model.hparams.neck_dim;  // 256
    if (!curr_tokens || !src_pos || !prompt || !prompt_pos || !rope_freqs || num_obj_ptr_tokens < 0)
        throw std::invalid_argument("memory attention inputs are incomplete");
    const auto N = curr_tokens->ne[1];
    const auto M_total = prompt->ne[1];
    const auto M_spatial = M_total - num_obj_ptr_tokens;
    const auto B = std::max(curr_tokens->ne[2], prompt->ne[2]);
    if (curr_tokens->type != GGML_TYPE_F32 || src_pos->type != GGML_TYPE_F32 ||
        prompt->type != GGML_TYPE_F32 || prompt_pos->type != GGML_TYPE_F32 ||
        curr_tokens->ne[0] != D || N <= 0 || B <= 0 || curr_tokens->ne[3] != 1 ||
        (curr_tokens->ne[2] != 1 && curr_tokens->ne[2] != B) ||
        src_pos->ne[0] != D || src_pos->ne[1] != N ||
        (src_pos->ne[2] != 1 && src_pos->ne[2] != B) || src_pos->ne[3] != 1 ||
        prompt->ne[0] != model.hparams.mem_out_dim || M_total <= 0 ||
        (prompt->ne[2] != 1 && prompt->ne[2] != B) || prompt->ne[3] != 1 ||
        prompt_pos->ne[0] != prompt->ne[0] || prompt_pos->ne[1] != M_total ||
        (prompt_pos->ne[2] != 1 && prompt_pos->ne[2] != B) || prompt_pos->ne[3] != 1 || M_spatial < 0 ||
        rope_freqs->type != GGML_TYPE_F32 || rope_freqs->ne[0] != 2 || rope_freqs->ne[1] != D / 2 ||
        rope_freqs->ne[2] != N || rope_freqs->ne[3] != 1 ||
        (rope_k_freqs && (rope_k_freqs->type != GGML_TYPE_F32 || rope_k_freqs->ne[0] != 2 ||
                          rope_k_freqs->ne[1] != D / 2 || rope_k_freqs->ne[2] != M_spatial ||
                          rope_k_freqs->ne[3] != 1)))
        throw std::invalid_argument("memory attention inputs must use matching F32 object batches");

    // pos_enc_at_input: x = curr + 0.1 * src_pos
    auto* x = sam3_add_token_batch(ctx, curr_tokens, ggml_scale(ctx, src_pos, 0.1f));
    ggml_set_name(x, "phase7_mem_attn_input");

    for (int l = 0; l < (int)ma.layers.size(); ++l) {
        const auto& ly = ma.layers[l];

        // ── Self-attention with RoPE ──────────────────────────────────────
        {
            auto* x_norm = sam3_layer_norm(ctx, x, ly.norm1_w, ly.norm1_b);
            auto* q = ggml_add(ctx, ggml_mul_mat(ctx, ly.sa_q_w, x_norm), ly.sa_q_b);
            auto* k = ggml_add(ctx, ggml_mul_mat(ctx, ly.sa_k_w, x_norm), ly.sa_k_b);
            auto* v = ggml_add(ctx, ggml_mul_mat(ctx, ly.sa_v_w, x_norm), ly.sa_v_b);

            // Apply RoPE to Q and K (single-head, [D, N, 1])
            q = sam3_apply_rope(ctx, q, rope_freqs);
            k = sam3_apply_rope(ctx, k, rope_freqs);

            auto* sa_out = tiled_memory_attention(ctx, q, k, v, attention_policy);
            sa_out = ggml_add(ctx, ggml_mul_mat(ctx, ly.sa_out_w, sa_out), ly.sa_out_b);
            x = sam3_add_token_batch(ctx, x, sa_out);
            sam3_name_tensorf(x, "phase7_mem_attn_layer%d_after_sa", l);
        }

        // ── Cross-attention to memory (kv_dim=64→256) with RoPE ──────────
        {
            auto* x_norm = sam3_layer_norm(ctx, x, ly.norm2_w, ly.norm2_b);
            auto* q = ggml_add(ctx, ggml_mul_mat(ctx, ly.ca_q_w, x_norm), ly.ca_q_b);

            // K: project from (prompt + prompt_pos) — pos enc added before projection
            auto* kv_with_pos = sam3_add_token_batch(ctx, prompt, prompt_pos);
            auto* k = ggml_add(ctx, ggml_mul_mat(ctx, ly.ca_k_w, kv_with_pos), ly.ca_k_b);
            // V: project from prompt only (no pos enc)
            auto* v = ggml_add(ctx, ggml_mul_mat(ctx, ly.ca_v_w, prompt), ly.ca_v_b);

            // Apply RoPE to Q
            q = sam3_apply_rope(ctx, q, rope_freqs);

            // Apply RoPE to spatial K only (exclude num_obj_ptr_tokens tail keys)
            if (M_spatial > 0 && rope_k_freqs) {
                auto* k_spatial = ggml_cont(ctx, ggml_view_3d(ctx, k,
                                                              D, M_spatial, k->ne[2], k->nb[1], k->nb[2], 0));
                k_spatial = sam3_apply_rope(ctx, k_spatial, rope_k_freqs);
                if (num_obj_ptr_tokens > 0) {
                    auto* k_ptr = ggml_cont(ctx, ggml_view_3d(ctx, k,
                                                              D, num_obj_ptr_tokens, k->ne[2], k->nb[1], k->nb[2],
                                                              (size_t)M_spatial * k->nb[1]));
                    k = ggml_concat(ctx, k_spatial, k_ptr, 1);
                } else {
                    k = k_spatial;
                }
            }

            auto* ca_out = tiled_memory_attention(ctx, q, k, v, attention_policy);
            ca_out = ggml_add(ctx, ggml_mul_mat(ctx, ly.ca_out_w, ca_out), ly.ca_out_b);
            x = sam3_add_token_batch(ctx, x, ca_out);
            sam3_name_tensorf(x, "phase7_mem_attn_layer%d_after_ca", l);
        }

        // ── FFN ───────────────────────────────────────────────────────────
        {
            auto* x_norm = sam3_layer_norm(ctx, x, ly.norm3_w, ly.norm3_b);
            auto* ffn = ggml_add(ctx, ggml_mul_mat(ctx, ly.ffn_fc1_w, x_norm), ly.ffn_fc1_b);
            ffn = ggml_relu(ctx, ffn);
            ffn = ggml_add(ctx, ggml_mul_mat(ctx, ly.ffn_fc2_w, ffn), ly.ffn_fc2_b);
            x = sam3_add_token_batch(ctx, x, ffn);
            sam3_name_tensorf(x, "phase7_mem_attn_layer%d_after_ffn", l);
        }
    }

    // Final norm
    auto* norm_w = model.tensors.at("mem_attn.norm.weight");
    auto* norm_b = model.tensors.at("mem_attn.norm.bias");
    x = sam3_layer_norm(ctx, x, norm_w, norm_b);
    ggml_set_name(x, "phase7_mem_attn_output");

    return x;  // [D, N, 1]
}


} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MEMORY_ATTENTION_HPP
