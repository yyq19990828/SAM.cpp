#ifndef SAM_CPP_SRC_MODELS_SAM3_DETECTOR_HPP
#define SAM_CPP_SRC_MODELS_SAM3_DETECTOR_HPP

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
#include "sam/types.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace sam::internal::sam3 {

/*****************************************************************************
** DETR decoder — graph building (6 layers)
*****************************************************************************/

// inverse_sigmoid: log(x / (1 - x)), clamped to avoid inf
// Python reference uses eps=1e-3: x1 = x.clamp(min=eps), x2 = (1-x).clamp(min=eps)
inline struct ggml_tensor* sam3_inverse_sigmoid(struct ggml_context* ctx, struct ggml_tensor* x) {
    // Match the official independent numerator/denominator clamps, including
    // values below eps or above 1-eps.
    // This GGML revision implements clamp as an in-place view. Isolate both
    // clamps so neither the shared reference boxes nor the denominator change.
    x = ggml_clamp(ctx, ggml_dup(ctx, x), 0.0f, 1.0f);
    auto* log_x = ggml_log(ctx, ggml_clamp(ctx, ggml_dup(ctx, x), 1e-3f, 1.0f));
    // Compute (1 - x) as (-1)*x + 1.  We use ggml_scale_bias which takes float
    // scalars (no tensor allocation needed, safe in no_alloc contexts).
    auto* one_minus = ggml_scale_bias(ctx, x, -1.0f, 1.0f);
    auto* log_1mx = ggml_log(ctx, ggml_clamp(ctx, one_minus, 1e-3f, 1.0f));
    return ggml_sub(ctx, log_x, log_1mx);
}

// Build sinusoidal positional embedding for 4D reference points in the ggml graph.
// ref_boxes: [4, NQ, B] — (cx, cy, w, h) after sigmoid, B=1
// sine_dim_t: [1, 64] — pre-computed angle multipliers (2π / 10000^(2i/128))
// Returns: [512, NQ, B] sinusoidal embedding matching Python gen_sineembed_for_position
inline struct ggml_tensor* sam3_build_sine_pos_embed_4d(
    struct ggml_context* ctx,
    struct ggml_tensor* ref_boxes,     // [4, NQ, B]
    struct ggml_tensor* sine_dim_t) {  // [1, 64]
    const int64_t NQ = ref_boxes->ne[1];

    // Python output order: [cy, cx, w, h] → coord indices from boxes [cx(0),cy(1),w(2),h(3)]
    const int coord_order[4] = {1, 0, 2, 3};

    struct ggml_tensor* coord_embeds[4];

    for (int c = 0; c < 4; ++c) {
        int ci = coord_order[c];
        // Extract one coordinate: view into ref_boxes [4, NQ, 1] at element ci
        auto* coord = ggml_view_2d(ctx, ref_boxes, 1, NQ,
                                   ref_boxes->nb[1], ci * sizeof(float));  // [1, NQ]

        // Outer product: angles[i, q] = dim_t[i] * coord[q]
        // ggml_mul_mat(A=[1,64], B=[1,NQ]) = A^T @ B = [64,1]@[1,NQ] = [64, NQ]
        auto* angles = ggml_mul_mat(ctx, sine_dim_t, coord);  // [64, NQ]

        auto* sin_vals = ggml_sin(ctx, angles);  // [64, NQ]
        auto* cos_vals = ggml_cos(ctx, angles);  // [64, NQ]

        // Interleave: [sin_0, cos_0, sin_1, cos_1, ...]
        auto* sin_r = ggml_reshape_3d(ctx, sin_vals, 1, 64, NQ);
        auto* cos_r = ggml_reshape_3d(ctx, cos_vals, 1, 64, NQ);
        auto* interleaved = ggml_concat(ctx, sin_r, cos_r, 0);  // [2, 64, NQ]
        coord_embeds[c] = ggml_reshape_2d(ctx, interleaved, 128, NQ);
    }

    // Concatenate all 4 coordinates → [512, NQ]
    auto* embed = ggml_concat(ctx, coord_embeds[0], coord_embeds[1], 0);  // [256, NQ]
    embed = ggml_concat(ctx, embed, coord_embeds[2], 0);                  // [384, NQ]
    embed = ggml_concat(ctx, embed, coord_embeds[3], 0);                  // [512, NQ]

    return embed;
}

// Build query positional encoding from reference boxes via sine embed + ref_point_head MLP.
// ref_boxes: [4, NQ, 1] — after sigmoid
// sine_dim_t: [1, 64]
// Returns: [D, NQ+1, 1] (zeros for presence token at index 0, MLP output for object queries)
inline struct ggml_tensor* sam3_build_query_pos(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* ref_boxes,   // [4, NQ, 1]
    struct ggml_tensor* sine_dim_t,  // [1, 64]
    int layer_idx = -1) {
    const auto& tensors = model.tensors;
    const int64_t NQ = ref_boxes->ne[1];
    const int D = model.hparams.neck_dim;  // 256

    // 1. Sine positional embedding: [512, NQ]
    auto* sine_embed = sam3_build_sine_pos_embed_4d(ctx, ref_boxes, sine_dim_t);
    if (layer_idx == 0) {
        ggml_set_name(sine_embed, "ddec_query_sine_0");
    }

    // 2. ref_point_head MLP: 512 → 256 → 256
    // Layer 0: relu(W0 @ sine_embed + b0)
    auto* h = ggml_mul_mat(ctx, tensors.at("ddec.ref_point_head.layers.0.weight"), sine_embed);
    h = ggml_add(ctx, h, tensors.at("ddec.ref_point_head.layers.0.bias"));
    h = ggml_relu(ctx, h);
    // Layer 1: W1 @ h + b1 (no activation)
    auto* qpos_obj = ggml_mul_mat(ctx, tensors.at("ddec.ref_point_head.layers.1.weight"), h);
    qpos_obj = ggml_add(ctx, qpos_obj, tensors.at("ddec.ref_point_head.layers.1.bias"));
    // qpos_obj: [D, NQ]
    if (layer_idx == 0) {
        ggml_set_name(qpos_obj, "ddec_query_pos_0");
    }

    // 3. Reshape to 3D and prepend zeros for presence token
    qpos_obj = ggml_reshape_3d(ctx, qpos_obj, D, NQ, 1);
    auto* qpos_pres = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    ggml_set_name(qpos_pres, "ddec_query_pos_pres");
    ggml_set_input(qpos_pres);  // zeros — set by caller

    return ggml_concat(ctx, qpos_pres, qpos_obj, 1);  // [D, NQ+1, 1]
}

// Build box-relative positional bias for DETR cross-attention.
// ref_boxes: [4, N_q, B] — (cx, cy, w, h) in [0,1]
// rpb_coords: [feat_hw] — normalized coords [0/H, 1/H, ..., (H-1)/H] (input tensor)
// Returns: bias tensor [N_kv, N_q+1, n_heads, B] for ggml_flash_attn_ext mask
//
// Python _get_rpb_matrix with boxRPB="log":
//   1. boxes → xyxy
//   2. deltas_x[q,w,:2] = [coord_w - x0, coord_w - x1]
//   3. deltas_y[q,h,:2] = [coord_h - y0, coord_h - y1]
//   4. log transform: sign(d*8) * log2(|d*8|+1) / log2(8)
//   5. MLP: [2] → [256] → [n_heads]
//   6. outer sum: B[h,w] = delta_y[h] + delta_x[w]
inline struct ggml_tensor* sam3_compute_box_rpb(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* ref_boxes,   // [4, N_q, B]
    struct ggml_tensor* rpb_coords,  // [feat_hw] — pre-filled grid coordinates
    int feat_hw,
    int layer_idx = -1) {
    const int64_t NQ = ref_boxes->ne[1];
    const int NH = model.hparams.ddec_heads;  // 8
    const int W = feat_hw;
    const int H = feat_hw;
    const auto& tensors = model.tensors;

    // ── 1. Convert cxcywh → xyxy ─────────────────────────────────────────
    // ggml_view_2d on strided data is non-contiguous — ggml_scale requires contiguous.
    // Use ggml_cont to make each coordinate slice contiguous.
    auto* cx = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 0));
    auto* cy = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 1 * sizeof(float)));
    auto* bw = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 2 * sizeof(float)));
    auto* bh = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 3 * sizeof(float)));
    // x0 = cx - w/2, x1 = cx + w/2
    auto* half_w = ggml_scale(ctx, bw, 0.5f);
    auto* half_h = ggml_scale(ctx, bh, 0.5f);
    auto* x0 = ggml_sub(ctx, cx, half_w);  // [1, NQ]
    auto* x1 = ggml_add(ctx, cx, half_w);
    auto* y0 = ggml_sub(ctx, cy, half_h);
    auto* y1 = ggml_add(ctx, cy, half_h);

    // ── 2. Compute deltas via outer subtract ──────────────────────────────
    // coords: [W] → reshape to [W, 1] for outer subtract
    auto* cw = ggml_reshape_2d(ctx, rpb_coords, W, 1);  // [W, 1]

    // Outer subtract: delta[w, q] = coord[w] - edge[q]
    // Use ggml_mul_mat trick: not applicable for subtraction.
    // Instead: repeat coords to [W, NQ], repeat edge to [W, NQ], subtract.
    auto* shape_wn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, NQ);

    auto* cw_rep = ggml_repeat(ctx, cw, shape_wn);         // [W, NQ] (each column = coords)
    auto* x0_t = ggml_cont(ctx, ggml_transpose(ctx, x0));  // [NQ, 1]
    auto* x0_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, x0_t, 1, NQ), shape_wn);
    auto* x1_t = ggml_cont(ctx, ggml_transpose(ctx, x1));
    auto* x1_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, x1_t, 1, NQ), shape_wn);
    auto* y0_t = ggml_cont(ctx, ggml_transpose(ctx, y0));
    auto* y0_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, y0_t, 1, NQ), shape_wn);
    auto* y1_t = ggml_cont(ctx, ggml_transpose(ctx, y1));
    auto* y1_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, y1_t, 1, NQ), shape_wn);

    auto* dx0 = ggml_sub(ctx, cw_rep, x0_rep);  // [W, NQ]
    auto* dx1 = ggml_sub(ctx, cw_rep, x1_rep);
    auto* dy0 = ggml_sub(ctx, cw_rep, y0_rep);  // reusing coords for H (H==W)
    auto* dy1 = ggml_sub(ctx, cw_rep, y1_rep);

    // Stack into [2, W, NQ]: reshape each to [1, W, NQ], concat dim 0
    auto* dx0_r = ggml_reshape_3d(ctx, dx0, 1, W, NQ);
    auto* dx1_r = ggml_reshape_3d(ctx, dx1, 1, W, NQ);
    auto* deltas_x = ggml_concat(ctx, dx0_r, dx1_r, 0);  // [2, W, NQ]
    auto* dy0_r = ggml_reshape_3d(ctx, dy0, 1, H, NQ);
    auto* dy1_r = ggml_reshape_3d(ctx, dy1, 1, H, NQ);
    auto* deltas_y = ggml_concat(ctx, dy0_r, dy1_r, 0);  // [2, H, NQ]

    // ── 3. Log transform: sign(d*8) * log2(|d*8|+1) / log2(8) ────────────
    const float scale8 = 8.0f;
    const float inv_log2_8 = 1.0f / log2f(8.0f);  // = 1/3

    auto rpb_log = [&](struct ggml_tensor* d) -> struct ggml_tensor* {
        auto* d8 = ggml_scale(ctx, d, scale8);
        auto* sign_d = ggml_sgn(ctx, d8);
        auto* abs_d = ggml_abs(ctx, d8);
        auto* log_val = ggml_log(ctx, ggml_scale_bias(ctx, abs_d, 1.0f, 1.0f));
        // log2(x) = ln(x) / ln(2)
        log_val = ggml_scale(ctx, log_val, 1.0f / logf(2.0f));
        return ggml_mul(ctx, sign_d, ggml_scale(ctx, log_val, inv_log2_8));
    };

    deltas_x = rpb_log(deltas_x);  // [2, W, NQ]
    deltas_y = rpb_log(deltas_y);  // [2, H, NQ]

    // ── 4. MLP: [2, W*NQ] → [NH, W*NQ] ───────────────────────────────────
    // boxRPB_embed_x: MLP(2, 256, 8, 2) = Linear(2→256)+ReLU, Linear(256→8)
    // Reshape to [2, W*NQ] so matmul treats each (w, q) pair as a sample
    auto rpb_mlp = [&](struct ggml_tensor* d, const char* axis) -> struct ggml_tensor* {
        int64_t spatial = d->ne[1];
        int64_t nq = d->ne[2];
        auto* flat = ggml_reshape_2d(ctx, d, 2, spatial * nq);  // [2, W*NQ]
        auto wn0 = std::string("ddec.boxRPB_embed_") + axis + ".layers.0.weight";
        auto bn0 = std::string("ddec.boxRPB_embed_") + axis + ".layers.0.bias";
        auto wn1 = std::string("ddec.boxRPB_embed_") + axis + ".layers.1.weight";
        auto bn1 = std::string("ddec.boxRPB_embed_") + axis + ".layers.1.bias";
        flat = ggml_mul_mat(ctx, tensors.at(wn0), flat);
        flat = ggml_add(ctx, flat, tensors.at(bn0));
        flat = ggml_relu(ctx, flat);
        flat = ggml_mul_mat(ctx, tensors.at(wn1), flat);
        flat = ggml_add(ctx, flat, tensors.at(bn1));
        // flat: [NH, W*NQ] → reshape to [NH, spatial, NQ]
        return ggml_reshape_3d(ctx, flat, NH, spatial, nq);
    };

    auto* rpb_x = rpb_mlp(deltas_x, "x");  // [NH, W, NQ]
    auto* rpb_y = rpb_mlp(deltas_y, "y");  // [NH, H, NQ]

    // ── 5. Outer sum: B[nh, w, h, q] = rpb_y[nh, h, q] + rpb_x[nh, w, q] ─
    // Reshape for broadcasting:
    //   rpb_y → [NH, 1, H, NQ]
    //   rpb_x → [NH, W, 1, NQ]
    //
    // Keep W in ne[1] so reshaping [NH, W, H, NQ] → [NH, H*W, NQ, 1]
    // preserves Python's flatten(H, W) order where W is the fast spatial axis.
    auto* rpb_y_4d = ggml_reshape_4d(ctx, rpb_y, NH, 1, H, NQ);
    auto* rpb_x_4d = ggml_reshape_4d(ctx, rpb_x, NH, W, 1, NQ);

    // ggml_add broadcasts: where one dim is 1, the other is used
    auto* rpb_hw = ggml_repeat(ctx, rpb_y_4d,
                               ggml_new_tensor_4d(ctx, GGML_TYPE_F32, NH, W, H, NQ));
    auto* rpb_hw_x = ggml_repeat(ctx, rpb_x_4d,
                                 ggml_new_tensor_4d(ctx, GGML_TYPE_F32, NH, W, H, NQ));
    auto* rpb = ggml_add(ctx, rpb_hw, rpb_hw_x);  // [NH, W, H, NQ]

    // ── 6. Reshape to [H*W, NQ, NH, 1] for flash_attn_ext mask ───────────
    // Current: [NH, W, H, NQ]. Need: [N_kv=H*W, NQ, NH, B=1]
    rpb = ggml_reshape_4d(ctx, rpb, NH, H * W, NQ, 1);
    rpb = ggml_cont(ctx, ggml_permute(ctx, rpb, 2, 0, 1, 3));  // [H*W, NQ, NH, 1]

    // Prepend zeros for presence token: mask for presence token has no box-relative bias
    auto* pres_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H * W, 1, NH, 1);
    ggml_set_name(pres_mask, "rpb_pres_zeros");
    ggml_set_input(pres_mask);  // zeros — set by caller

    // [H*W, NQ+1, NH, 1]
    auto* full_rpb = ggml_concat(ctx, pres_mask, rpb, 1);
    full_rpb = ggml_cont(ctx, full_rpb);
    if (layer_idx == 0) {
        ggml_set_name(full_rpb, "ddec_rpb_mask_0");
    }

    return full_rpb;
}

// Single DETR decoder layer.
// queries: [D, N_q, B] where N_q = 201 (200 object queries + 1 presence token)
// query_pos: [D, N_q, B] positional encoding for queries
// enc_feats: [D, N_kv, B] conditioned image features from fusion encoder
// enc_pos: [D, N_kv, B] positional encoding for image features
// text_feats: [D, T, B] text features
// rpb_mask: [N_kv, N_q, n_heads, B] box-relative positional bias (or nullptr)
// Returns: updated queries [D, N_q, B]
inline struct ggml_tensor* sam3_ddec_layer_forward(
    struct ggml_context* ctx,
    const sam3_ddec_layer& ly,
    struct ggml_tensor* queries,
    struct ggml_tensor* query_pos,
    struct ggml_tensor* enc_feats,
    struct ggml_tensor* enc_pos,
    struct ggml_tensor* text_feats,
    int n_heads,
    struct ggml_tensor* text_attn_bias = nullptr,
    struct ggml_tensor* rpb_mask = nullptr,
    int layer_idx = -1) {
    const int64_t D = queries->ne[0];

    // Python decoder layer order (all post-norm):
    //   1. Self-attention → norm2 (post-norm)
    //   2. Text cross-attention (ca_text) → catext_norm (post-norm)
    //   3. Image cross-attention (cross_attn) → norm1 (post-norm)
    //   4. FFN → norm3 (post-norm)
    //
    // Norm weight mapping:
    //   ly.norm2_w  = ".norm2.weight"        = Python norm2 (post-SA)
    //   ly.norm3_w  = ".norm_ca_text.weight"  = Python catext_norm (post-text-CA)
    //   ly.norm1_w  = ".norm1.weight"         = Python norm1 (post-image-CA)
    //   ly.norm4_w  = ".norm3.weight"         = Python norm3 (post-FFN)

    // 1. Self-attention among queries (post-norm)
    {
        // Q = K = queries + query_pos, V = queries (no pos)
        auto* q_in = ggml_add(ctx, queries, query_pos);

        auto* q_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], D * ly.sa_in_proj_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 2 * D * ly.sa_in_proj_w->nb[1]);
        auto* q_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 2 * D * sizeof(float));

        const int64_t N = queries->ne[1];
        const int64_t B = queries->ne[2];
        const int64_t HD = D / n_heads;

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, q_in), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, queries), v_b);

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

        queries = ggml_add(ctx, queries, sa_out);
        // Post-norm: norm2 (Python's post-SA norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm2_w, ly.norm2_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_after_sa");
        }
    }

    // 2. Cross-attention to text tokens (post-norm)
    {
        // Q = queries + query_pos, K = V = text_feats
        auto* q_in = ggml_add(ctx, queries, query_pos);

        auto* q_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], D * ly.ca_text_q_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], 2 * D * ly.ca_text_q_w->nb[1]);
        auto* q_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, 2 * D * sizeof(float));

        const int64_t N_q = queries->ne[1];
        const int64_t N_kv = text_feats->ne[1];
        const int64_t B = queries->ne[2];
        const int64_t HD = D / n_heads;

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, text_feats), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, text_feats), v_b);

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);

        auto* text_mask = sam3_expand_token_attn_bias(ctx, text_attn_bias, N_q, n_heads, B);
        float scale = 1.0f / sqrtf((float)HD);
        auto* ca_out = ggml_flash_attn_ext(ctx, Q, K, V, text_mask, scale, 0.0f, 0.0f);
        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);
        ca_out = ggml_mul_mat(ctx, ly.ca_text_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_text_out_b);

        queries = ggml_add(ctx, queries, ca_out);
        // Post-norm: catext_norm (Python's post-text-CA norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm3_w, ly.norm3_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_after_text_ca");
        }
    }

    // 3. Cross-attention to conditioned image features (post-norm)
    {
        // Q = queries + query_pos, K = enc_feats + enc_pos, V = enc_feats
        auto* q_in = ggml_add(ctx, queries, query_pos);
        auto* k_in = ggml_add(ctx, enc_feats, enc_pos);

        auto* q_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], D * ly.ca_q_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 2 * D * ly.ca_q_w->nb[1]);
        auto* q_b = ggml_view_1d(ctx, ly.ca_q_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.ca_q_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.ca_q_b, D, 2 * D * sizeof(float));

        const int64_t N_q = queries->ne[1];
        const int64_t N_kv = enc_feats->ne[1];
        const int64_t B = queries->ne[2];
        const int64_t HD = D / n_heads;

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, k_in), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, enc_feats), v_b);

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);

        float scale = 1.0f / sqrtf((float)HD);
        struct ggml_tensor* ca_out = nullptr;

        if (rpb_mask) {
            // Keep the box-relative positional bias in fp32. The CPU flash-attn
            // kernel reads mask storage as fp16, which is good enough for token
            // padding masks but introduces avoidable drift here.
            auto* kq = ggml_mul_mat(ctx, K, Q);                      // [N_kv, N_q, NH, B]
            kq = ggml_soft_max_ext(ctx, kq, rpb_mask, scale, 0.0f);  // [N_kv, N_q, NH, B]

            auto* v_t = ggml_cont(ctx, ggml_transpose(ctx, V));  // [N_kv, HD, NH, B]
            ca_out = ggml_mul_mat(ctx, v_t, kq);                 // [HD, N_q, NH, B]
            ca_out = ggml_cont(ctx, ggml_permute(ctx, ca_out, 0, 2, 1, 3));
        } else {
            ca_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        }

        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);
        ca_out = ggml_mul_mat(ctx, ly.ca_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_out_b);

        queries = ggml_add(ctx, queries, ca_out);
        // Post-norm: norm1 (Python's post-image-CA norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm1_w, ly.norm1_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_after_img_ca");
        }
    }

    // 4. FFN (post-norm, ReLU)
    {
        auto* ffn = ggml_mul_mat(ctx, ly.ffn_fc1_w, queries);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc1_b);
        ffn = ggml_relu(ctx, ffn);
        ffn = ggml_mul_mat(ctx, ly.ffn_fc2_w, ffn);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc2_b);

        queries = ggml_add(ctx, queries, ffn);
        // Post-norm: norm3 (Python's post-FFN norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm4_w, ly.norm4_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_full_out");
        }
    }

    return queries;
}

// DotProductScoring: classify queries against text features via dot product.
//
// Python reference (DotProductScoring.forward):
//   1. prompt_mlp(prompt) → residual MLP + LN on text features
//   2. mean_pool_text(result, prompt_mask) → pooled [BS, D] (only valid tokens)
//   3. prompt_proj(pooled) → [BS, D]
//   4. hs_proj(hs) → [num_layer, BS, N_q, D]
//   5. matmul(proj_hs, proj_pooled.unsqueeze(-1)) → dot product → [num_layer, BS, N_q, 1]
//   6. scale by 1/sqrt(D)
//   7. clamp to [-12, 12]
//
// query_outputs: [D, N_q, B] — the 200 object query outputs
// text_features: [D, T, B] — text encoder output (already through resizer)
// text_valid_mask: [T, 1, B] — 1.0 for valid tokens, 0.0 for padding (or nullptr for all-valid)
// Returns: class_scores [N_q, B] (one score per query per batch)
inline struct ggml_tensor* sam3_dot_product_scoring(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* query_outputs,    // [D, N_q, B]
    struct ggml_tensor* text_features,    // [D, T, B]
    struct ggml_tensor* text_valid_mask)  // [T, 1, B] or nullptr
{
    const auto& tensors = model.tensors;
    const int64_t D = query_outputs->ne[0];  // 256
    const int64_t T = text_features->ne[1];
    const int64_t B = text_features->ne[2];

    // Step 1: Apply prompt_mlp on text features (residual MLP + LayerNorm)
    auto* text_mlp = text_features;  // [D, T, B]
    auto* orig_text = text_features;
    text_mlp = ggml_mul_mat(ctx, tensors.at("scoring.prompt_mlp.layers.0.weight"), text_mlp);
    text_mlp = ggml_add(ctx, text_mlp, tensors.at("scoring.prompt_mlp.layers.0.bias"));
    text_mlp = ggml_relu(ctx, text_mlp);
    text_mlp = ggml_mul_mat(ctx, tensors.at("scoring.prompt_mlp.layers.1.weight"), text_mlp);
    text_mlp = ggml_add(ctx, text_mlp, tensors.at("scoring.prompt_mlp.layers.1.bias"));
    text_mlp = ggml_add(ctx, text_mlp, orig_text);
    text_mlp = sam3_layer_norm(ctx, text_mlp,
                               tensors.at("scoring.prompt_mlp.out_norm.weight"),
                               tensors.at("scoring.prompt_mlp.out_norm.bias"));
    // text_mlp: [D, T, B]
    ggml_set_name(text_mlp, "scoring_prompt_mlp_out");

    // Step 2: Mean-pool over valid text tokens → [D, 1, B]
    // Python: pooled = (prompt * is_valid).sum(0) / num_valid
    struct ggml_tensor* text_pooled;
    if (text_valid_mask) {
        // Permute to [T, D, B] for masked pooling
        auto* tp = ggml_cont(ctx, ggml_permute(ctx, text_mlp, 1, 0, 2, 3));  // [T, D, B]
        // Mask: text_valid_mask is [T, 1, B] — broadcast multiply zeros out padding
        tp = ggml_mul(ctx, tp, text_valid_mask);  // [T, D, B] with padding zeroed
        // Sum over T dimension: pool_1d with SUM kernel=T
        // ggml_pool_1d AVG divides by T; we want SUM then divide by n_valid.
        // Use AVG and then scale by T/n_valid? Or use a manual approach.
        // Simpler: sum via pool_1d with AVG, then scale by T/n_valid.
        // But n_valid is dynamic. Instead: sum = mean * T, then divide by n_valid.
        // We pass n_valid as part of the mask: text_valid_mask sums to n_valid.
        // pool_1d(masked, AVG, T, T, 0) = sum(masked) / T. Multiply by T → sum(masked).
        // Then divide by n_valid. But n_valid is a scalar we know CPU-side.
        // For simplicity: compute AVG over ALL T positions (with padding zeroed out).
        // This gives sum(valid) / T. To get sum(valid) / n_valid, scale by T / n_valid.
        // We embed the scale factor into the mask: mask = (T / n_valid) for valid, 0 for pad.
        // Then AVG(mask * features) = sum(valid * T/n_valid) / T = sum(valid) / n_valid. ✓
        // Caller should set mask values to T/n_valid for valid tokens, 0 for padding.
        auto* pooled_t = ggml_pool_1d(ctx, tp, GGML_OP_POOL_AVG, (int)T, (int)T, 0);
        text_pooled = ggml_cont(ctx, ggml_permute(ctx, pooled_t, 1, 0, 2, 3));  // [D, 1, B]
    } else {
        // All tokens valid — simple mean
        auto* tp = ggml_cont(ctx, ggml_permute(ctx, text_mlp, 1, 0, 2, 3));
        auto* pooled_t = ggml_pool_1d(ctx, tp, GGML_OP_POOL_AVG, (int)T, (int)T, 0);
        text_pooled = ggml_cont(ctx, ggml_permute(ctx, pooled_t, 1, 0, 2, 3));
    }
    ggml_set_name(text_pooled, "scoring_pooled");

    // Step 3: Project pooled prompt through prompt_proj: D→D
    auto* proj_pooled = ggml_mul_mat(ctx, tensors.at("scoring.prompt_proj.weight"), text_pooled);
    proj_pooled = ggml_add(ctx, proj_pooled, tensors.at("scoring.prompt_proj.bias"));
    // proj_pooled: [D, 1, B]
    ggml_set_name(proj_pooled, "scoring_proj_pooled");

    // Step 4: Project queries through hs_proj: D→D
    auto* proj_hs = ggml_mul_mat(ctx, tensors.at("scoring.hs_proj.weight"), query_outputs);
    proj_hs = ggml_add(ctx, proj_hs, tensors.at("scoring.hs_proj.bias"));
    // proj_hs: [D, N_q, B]
    ggml_set_name(proj_hs, "scoring_proj_hs");

    // Step 5: Dot product — for each query, dot with pooled prompt
    // matmul(proj_hs, proj_pooled.unsqueeze(-1)) in Python = batched vector-matrix multiply
    // ggml_mul_mat(A, B) = A^T @ B
    // With A = proj_pooled [D, 1, B], B = proj_hs [D, N_q, B]:
    // result = [1, N_q, B] — each element is dot product of query with pooled prompt
    auto* scores = ggml_mul_mat(ctx, proj_pooled, proj_hs);  // [1, N_q, B]

    // Step 6: Scale by 1/sqrt(D)
    float scale = 1.0f / sqrtf((float)D);
    scores = ggml_scale(ctx, scores, scale);

    // Step 7: Clamp to [-12, 12]
    scores = ggml_clamp(ctx, scores, -12.0f, 12.0f);

    // Reshape to [N_q, B]
    const int64_t N_q = query_outputs->ne[1];
    scores = ggml_reshape_2d(ctx, scores, N_q, B);
    ggml_set_name(scores, "scoring_class_scores");

    return scores;
}

// Build full DETR decoder graph.
// enc_feats: [D, N_kv, B] conditioned features from fusion encoder (N_kv=5184)
// enc_pos: [D, N_kv, B] positional encoding
// text_feats: [D, T, B] text features
// Returns struct with:
//   queries: [D, 201, B] (all query outputs including presence token)
//   pred_boxes: [4, 200, B] (cx, cy, w, h in [0,1])
//   class_scores: [200, B]
//   presence_score: [1, B]
struct sam3_ddec_output {
    struct ggml_tensor* queries;         // [D, 201, B]
    struct ggml_tensor* presence_feats;  // [D, 1, B] pre-decoder-norm presence token
    struct ggml_tensor* pred_boxes;      // [4, 200, B]
    struct ggml_tensor* class_scores;    // [200, B]
    struct ggml_tensor* presence_score;  // [1, B]
};

inline sam3_ddec_output sam3_build_ddec_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* enc_feats,                  // [D, N_kv, B]
    struct ggml_tensor* enc_pos,                    // [D, N_kv, B]
    struct ggml_tensor* text_feats,                 // [D, T, B]
    struct ggml_tensor* sine_dim_t,                 // [1, 64] — pre-computed angle multipliers
    struct ggml_tensor* rpb_coords,                 // [feat_hw] — normalized grid coords (or nullptr)
    struct ggml_tensor* text_attn_bias = nullptr,   // [T, 1, B] additive text padding bias
    struct ggml_tensor* text_valid_mask = nullptr)  // [T, 1, B] for scoring (or nullptr)
{
    const auto& hp = model.hparams;
    const auto& tensors = model.tensors;
    const int D = hp.neck_dim;            // 256
    const int NQ = hp.ddec_num_queries;   // 200
    const int feat_hw = hp.n_img_embd();  // 72

    // ── Initialize queries from query_embed ──────────────────────────────
    auto* content = ggml_reshape_3d(ctx, model.ddec.query_embed, D, NQ, 1);
    auto* pres_tok = ggml_reshape_3d(ctx, model.ddec.presence_token, D, 1, 1);
    auto* queries = ggml_concat(ctx, pres_tok, content, 1);  // [D, NQ+1, B=1]

    // Reference points: sigmoid → initial anchor boxes
    auto* ref_pts_raw = tensors.at("ddec.reference_points.weight"); // [4, NQ]
    // The container stores these parameters in F16, but they become decoder
    // activations. Promote before sigmoid/sine/matmul, as upstream loading did.
    if (ref_pts_raw->type != GGML_TYPE_F32) ref_pts_raw = ggml_cast(ctx, ref_pts_raw, GGML_TYPE_F32);
    ref_pts_raw = ggml_cont(ctx, ref_pts_raw);
    auto* ref_boxes = ggml_sigmoid(ctx, ref_pts_raw);                                // [4, NQ]
    ref_boxes = ggml_reshape_3d(ctx, ref_boxes, 4, NQ, 1);                           // [4, NQ, 1]

    // ── Run decoder layers ───────────────────────────────────────────────
    // Per-layer: recompute query_pos from updated ref_boxes (matching Python exactly)
    struct ggml_tensor* last_presence = pres_tok;
    for (int i = 0; i < hp.ddec_layers; ++i) {
        // Recompute query_pos from current ref_boxes via sine embed + ref_point_head MLP
        auto* query_pos = sam3_build_query_pos(ctx, model, ref_boxes, sine_dim_t, i);

        // Compute box-relative positional bias for image cross-attention
        struct ggml_tensor* rpb_mask = nullptr;
        if (rpb_coords) {
            rpb_mask = sam3_compute_box_rpb(ctx, model, ref_boxes, rpb_coords, feat_hw, i);
        }

        queries = sam3_ddec_layer_forward(ctx, model.ddec.layers[i],
                                          queries, query_pos,
                                          enc_feats, enc_pos,
                                          text_feats, hp.ddec_heads,
                                          text_attn_bias,
                                          rpb_mask,
                                          i);

        // Box refinement after each layer (on object queries only, not presence token)
        auto* obj_q = ggml_view_3d(ctx, queries, D, NQ, 1,
                                   queries->nb[1], queries->nb[2], 1 * queries->nb[1]);
        obj_q = ggml_cont(ctx, obj_q);
        sam3_name_tensorf(obj_q, "ddec_layer%d_out", i);

        auto* pres_q = ggml_view_3d(ctx, queries, D, 1, 1,
                                    queries->nb[1], queries->nb[2], 0);
        pres_q = ggml_cont(ctx, pres_q);
        if (i == 0) {
            ggml_set_name(pres_q, "ddec_layer0_presence");
        }
        last_presence = pres_q;

        // Apply the final decoder norm before box refinement (use_normed_output_consistently)
        auto* obj_q_normed = sam3_layer_norm(ctx, obj_q,
                                             tensors.at("ddec.norm.weight"),
                                             tensors.at("ddec.norm.bias"));

        // Shared bbox_embed MLP
        auto* bd = obj_q_normed;
        for (int j = 0; j < 3; ++j) {
            auto wn = "ddec.bbox_embed.layers." + std::to_string(j) + ".weight";
            auto bn = "ddec.bbox_embed.layers." + std::to_string(j) + ".bias";
            bd = ggml_mul_mat(ctx, tensors.at(wn), bd);
            bd = ggml_add(ctx, bd, tensors.at(bn));
            if (j < 2) bd = ggml_relu(ctx, bd);
        }
        // bd: [4, NQ, 1]

        // ref_boxes = sigmoid(inverse_sigmoid(ref_boxes) + box_delta)
        auto* ref_inv_cur = sam3_inverse_sigmoid(ctx, ref_boxes);
        ref_boxes = ggml_sigmoid(ctx, ggml_add(ctx, ref_inv_cur, bd));
        sam3_name_tensorf(ref_boxes, "ddec_layer%d_refboxes", i);
    }

    // ── Final normalization ──────────────────────────────────────────────
    // Match Python: decoder.norm is applied to object queries only.
    auto* obj_queries = ggml_view_3d(ctx, queries, D, NQ, 1,
                                     queries->nb[1], queries->nb[2], 1 * queries->nb[1]);
    obj_queries = ggml_cont(ctx, obj_queries);
    obj_queries = sam3_layer_norm(ctx, obj_queries,
                                  tensors.at("ddec.norm.weight"),
                                  tensors.at("ddec.norm.bias"));
    ggml_set_name(obj_queries, "ddec_normed_output");

    auto* queries_for_seg = ggml_concat(ctx, last_presence, obj_queries, 1);

    auto* class_scores = sam3_dot_product_scoring(ctx, model, obj_queries, text_feats, text_valid_mask);
    // class_scores: [NQ, B]

    // ── Presence score ───────────────────────────────────────────────────
    // Presence token head: LN + 3-layer MLP (D→D→D→1)
    auto* pres_out = sam3_layer_norm(ctx, last_presence,
                                     tensors.at("ddec.presence_token_out_norm.weight"),
                                     tensors.at("ddec.presence_token_out_norm.bias"));

    for (int j = 0; j < 3; ++j) {
        auto wn = "ddec.presence_token_head.layers." + std::to_string(j) + ".weight";
        auto bn = "ddec.presence_token_head.layers." + std::to_string(j) + ".bias";
        pres_out = ggml_mul_mat(ctx, tensors.at(wn), pres_out);
        pres_out = ggml_add(ctx, pres_out, tensors.at(bn));
        if (j < 2) pres_out = ggml_relu(ctx, pres_out);
    }
    // Keep presence as raw logit (no sigmoid yet — applied during post-processing)
    auto* presence_score = ggml_reshape_2d(ctx, pres_out, 1, 1);
    // presence_score: [1, B] — raw logit

    sam3_ddec_output out;
    out.queries = queries_for_seg;        // [D, NQ+1, B]
    out.presence_feats = last_presence;   // [D, 1, B]
    out.pred_boxes = ref_boxes;           // [4, NQ, B]
    out.class_scores = class_scores;      // [NQ, B]
    out.presence_score = presence_score;  // [1, B]

    return out;
}

inline void initialize_detector_zero_inputs(ggml_context* context, RuntimeStats& stats) {
    // Each decoder layer creates its own inputs with repeated names. Initialize
    // every tensor rather than only one ggml_get_tensor() match.
    for (auto* tensor = ggml_get_first_tensor(context); tensor; tensor = ggml_get_next_tensor(context, tensor)) {
        const std::string name = ggml_get_name(tensor);
        if ((name == "ddec_query_pos_pres" || name == "rpb_pres_zeros") && tensor->buffer) {
            std::vector<float> zeros(static_cast<std::size_t>(ggml_nelements(tensor)), 0.0f);
            ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size() * sizeof(float));
            stats.host_upload_bytes += zeros.size() * sizeof(float);
        }
    }
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_DETECTOR_HPP
