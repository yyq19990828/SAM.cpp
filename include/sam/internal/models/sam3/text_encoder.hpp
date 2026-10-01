#ifndef SAM_CPP_INTERNAL_MODELS_SAM3_TEXT_ENCODER_HPP
#define SAM_CPP_INTERNAL_MODELS_SAM3_TEXT_ENCODER_HPP

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

/*****************************************************************************
** Text Encoder — graph building
*****************************************************************************/

// Build a causal (lower-triangular) attention mask for the text encoder.
// Returns: [L, L] F16 tensor. mask[kv][q] = 0 if kv <= q, -inf otherwise.
// Marked as input — caller must upload data via ggml_backend_tensor_set after alloc.
inline struct ggml_tensor* sam3_build_causal_mask(struct ggml_context* ctx, int L) {
    auto* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L, L);
    ggml_set_name(mask, "causal_mask");
    ggml_set_input(mask);
    return mask;
}

// Fill a pre-allocated causal mask buffer (host-side, F16).
// mask_data must hold L*L ggml_fp16_t values.
inline void sam3_fill_causal_mask(ggml_fp16_t* mask_data, int L) {
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neginf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < L; ++q) {
        for (int kv = 0; kv < L; ++kv) {
            mask_data[kv + q * L] = (kv <= q) ? zero : neginf;
        }
    }
}

// Single text encoder block forward pass.
// Input x: [E, L] where E=text_width=1024, L=seq_len (typically 32).
// causal_mask: [L, L] F16 additive mask for ggml_flash_attn_ext.
// Returns: [E, L]
inline struct ggml_tensor* sam3_text_block_forward(struct ggml_context* ctx,
                                                   struct ggml_tensor* x,
                                                   const sam3_text_block& blk,
                                                   const sam3_hparams& hp,
                                                   struct ggml_tensor* causal_mask,
                                                   int block_idx) {
    const int E = hp.text_width;   // 1024
    const int NH = hp.text_heads;  // 16
    const int HD = E / NH;         // 64
    const int64_t L = x->ne[1];    // sequence length

    auto* shortcut = x;
    x = sam3_layer_norm(ctx, x, blk.ln1_w, blk.ln1_b);
    sam3_name_tensorf(x, "text_block_%02d_after_ln1", block_idx);

    auto* qkv = ggml_mul_mat(ctx, blk.attn_in_proj_w, x);
    qkv = ggml_add(ctx, qkv, blk.attn_in_proj_b);
    sam3_name_tensorf(qkv, "text_block_%02d_qkv", block_idx);

    // [3*E, L] → [E, 3, L] → permute → [E, L, 3]
    qkv = ggml_reshape_3d(ctx, qkv, E, 3, L);
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 2, 1, 3));
    // qkv: [E, L, 3]

    auto* Q = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 0);
    auto* K = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 1 * qkv->nb[2]);
    auto* V = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 2 * qkv->nb[2]);

    // [E, L] → [HD, NH, L] → permute → [HD, L, NH, 1]
    Q = ggml_reshape_3d(ctx, Q, HD, NH, L);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    Q = ggml_reshape_4d(ctx, Q, HD, L, NH, 1);

    K = ggml_reshape_3d(ctx, K, HD, NH, L);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    K = ggml_reshape_4d(ctx, K, HD, L, NH, 1);

    V = ggml_reshape_3d(ctx, V, HD, NH, L);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);  // non-contiguous; flash_attn uses strides

    float scale = 1.0f / sqrtf((float)HD);
    auto* attn_out = ggml_flash_attn_ext(ctx, Q, K, V, causal_mask, scale, 0.0f, 0.0f);
    x = ggml_reshape_2d(ctx, attn_out, E, L);

    x = ggml_mul_mat(ctx, blk.attn_out_proj_w, x);
    x = ggml_add(ctx, x, blk.attn_out_proj_b);

    if (blk.ls1) {
        x = ggml_mul(ctx, x, blk.ls1);
    }
    sam3_name_tensorf(x, "text_block_%02d_attn_out", block_idx);

    x = ggml_add(ctx, shortcut, x);
    sam3_name_tensorf(x, "text_block_%02d_after_attn_residual", block_idx);

    shortcut = x;
    x = sam3_layer_norm(ctx, x, blk.ln2_w, blk.ln2_b);
    sam3_name_tensorf(x, "text_block_%02d_after_ln2", block_idx);

    x = ggml_mul_mat(ctx, blk.mlp_fc1_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc1_b);
    sam3_name_tensorf(x, "text_block_%02d_mlp_fc1", block_idx);
    x = ggml_gelu_erf(ctx, x);
    sam3_name_tensorf(x, "text_block_%02d_mlp_gelu", block_idx);
    x = ggml_mul_mat(ctx, blk.mlp_fc2_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc2_b);

    if (blk.ls2) {
        x = ggml_mul(ctx, x, blk.ls2);
    }
    sam3_name_tensorf(x, "text_block_%02d_mlp_out", block_idx);

    x = ggml_add(ctx, shortcut, x);
    sam3_name_tensorf(x, "text_block_%02d_out", block_idx);

    return x;
}

// Build the full text encoder computation graph.
// token_ids: [L] int32 tensor (BPE token IDs, padded to ctx_len with 0s).
//            Must be marked as input by caller; data uploaded after alloc.
// Returns: text_features tensor [text_out_dim, L] = [256, L].
// Also creates the causal mask internally (marked as input).
inline struct ggml_tensor* sam3_build_text_encoder_graph(struct ggml_context* ctx,
                                                         struct ggml_tensor* token_ids,
                                                         const sam3_model& model) {
    const auto& hp = model.hparams;
    const auto& enc = model.text_enc;
    const int L = hp.text_ctx_len;  // 32

    auto* x = ggml_get_rows(ctx, enc.token_embed_w, token_ids);
    ggml_set_name(x, "text_token_embed");

    x = ggml_add(ctx, x, enc.pos_embed);
    ggml_set_name(x, "text_after_pos_embed");

    auto* causal_mask = sam3_build_causal_mask(ctx, L);

    for (int i = 0; i < hp.text_layers; ++i) {
        x = sam3_text_block_forward(ctx, x, enc.blocks[i], hp, causal_mask, i);
    }

    x = sam3_layer_norm(ctx, x, enc.ln_final_w, enc.ln_final_b);
    ggml_set_name(x, "text_final_ln");

    // Resizer: project 1024 → 256
    x = ggml_mul_mat(ctx, enc.resizer_w, x);
    x = ggml_add(ctx, x, enc.resizer_b);
    ggml_set_name(x, "text_features_2d");

    return x;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_INTERNAL_MODELS_SAM3_TEXT_ENCODER_HPP
