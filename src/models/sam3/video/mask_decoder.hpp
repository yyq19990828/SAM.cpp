#ifndef SAM_CPP_SRC_MODELS_SAM3_VIDEO_MASK_DECODER_HPP
#define SAM_CPP_SRC_MODELS_SAM3_VIDEO_MASK_DECODER_HPP

// Adapted from PABannier/sam3.cpp, revision 416186c501d060df7ca02989d49b38080f5f81f3.
// Copyright (c) 2025-2026 Pierre-Antoine Bannier. MIT: licenses/sam3.cpp-MIT.txt.

#include "../architecture.hpp"
#include "../vision.hpp"
#include "../ops.hpp"
#include "attention.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sam::internal::sam3 {

inline ggml_tensor* sam3_add_spatial_batch(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    if (a->ne[3] != b->ne[3] && a->ne[3] != 1 && b->ne[3] != 1)
        throw std::invalid_argument("SAM spatial tensors have incompatible object batches");
    return a->ne[3] >= b->ne[3] ? ggml_add(ctx, a, b) : ggml_add(ctx, b, a);
}

inline struct ggml_tensor* sam3_sam_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q_in,  // [D, N_q, B]
    struct ggml_tensor* k_in,  // [D, N_kv, B]
    struct ggml_tensor* v_in,  // [D, N_kv, B]
    const sam3_sam_attn& attn,
    int n_heads) {
    if (!q_in || !k_in || !v_in || n_heads <= 0 || q_in->ne[2] <= 0 ||
        k_in->ne[2] <= 0 || v_in->ne[2] <= 0 ||
        q_in->ne[3] != 1 || k_in->ne[3] != 1 || v_in->ne[3] != 1 ||
        k_in->ne[1] != v_in->ne[1])
        throw std::invalid_argument("SAM attention requires matching token batches");
    const auto batch = std::max(q_in->ne[2], std::max(k_in->ne[2], v_in->ne[2]));
    q_in = sam3_repeat_batch_3d(ctx, q_in, batch);
    k_in = sam3_repeat_batch_3d(ctx, k_in, batch);
    v_in = sam3_repeat_batch_3d(ctx, v_in, batch);
    const int64_t N_q = q_in->ne[1];
    const int64_t B = q_in->ne[2];
    const int64_t N_kv = k_in->ne[1];

    // Project
    auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, attn.q_w, q_in), attn.q_b);
    auto* K = ggml_add(ctx, ggml_mul_mat(ctx, attn.k_w, k_in), attn.k_b);
    auto* V = ggml_add(ctx, ggml_mul_mat(ctx, attn.v_w, v_in), attn.v_b);

    // internal_dim = out_proj cols = attn.q_w->ne[1]
    const int64_t ID = attn.q_w->ne[1];
    const int64_t HD = ID / n_heads;

    // Reshape to multi-head: [HD, N, NH, B]
    Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // [HD, N_q, NH, B]

    K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));  // [HD, N_kv, NH, B]

    V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));  // [HD, N_kv, NH, B] contiguous

    // Attention
    float scale = 1.0f / sqrtf((float)HD);
    ggml_tensor* out;
    if (HD == 16) {
        // The pinned Metal flash kernel has no head-16 implementation. Keep
        // both operands and softmax in F32 on the same backend.
        auto* scores = ggml_mul_mat(ctx, K, Q);
        ggml_prec_set_acc(scores, GGML_PREC_F32);
        scores = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
        auto* values = ggml_cont(ctx, ggml_transpose(ctx, V));
        out = ggml_mul_mat(ctx, values, scores);
        ggml_prec_set_acc(out, GGML_PREC_F32);
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    } else {
        out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
    }
    // [HD, NH, N_q, B], matching flash_attn_ext's output layout.
    // Merge heads: [ID=HD*NH, N_q, B]
    auto* merged = ggml_reshape_3d(ctx, out, ID, N_q, B);

    // Output projection
    out = ggml_mul_mat(ctx, attn.out_w, merged);
    out = ggml_add(ctx, out, attn.out_b);

    return out;
}

inline void sam3_twoway_block_forward(
    struct ggml_context* ctx,
    struct ggml_tensor*& queries,  // [D, N_q, B] — modified in place
    struct ggml_tensor*& keys,     // [D, N_kv, B] — modified in place
    struct ggml_tensor* query_pe,  // [D, N_q, B]
    struct ggml_tensor* key_pe,    // [D, N_kv, B]
    const sam3_twoway_block& blk,
    int n_heads,
    bool skip_first_layer_pe) {
    // 1. Self-attention on queries
    if (skip_first_layer_pe) {
        // Python: queries = self.self_attn(q=queries, k=queries, v=queries)
        // No residual connection when skipping first layer PE
        queries = sam3_sam_attention(ctx, queries, queries, queries, blk.self_attn, n_heads);
    } else {
        auto* q = ggml_add(ctx, queries, query_pe);
        auto* attn_out = sam3_sam_attention(ctx, q, q, queries, blk.self_attn, n_heads);
        queries = ggml_add(ctx, queries, attn_out);
    }
    queries = sam3_layer_norm(ctx, queries, blk.norm1_w, blk.norm1_b);

    // 2. Cross-attention: tokens attending to image
    {
        auto* q = sam3_add_token_batch(ctx, queries, query_pe);
        auto* k = sam3_add_token_batch(ctx, keys, key_pe);
        auto* attn_out = sam3_sam_attention(ctx, q, k, keys, blk.ca_tok2img, n_heads);
        queries = sam3_add_token_batch(ctx, queries, attn_out);
        queries = sam3_layer_norm(ctx, queries, blk.norm2_w, blk.norm2_b);
    }

    // 3. MLP on queries (ReLU activation)
    {
        auto* mlp = ggml_mul_mat(ctx, blk.mlp_fc1_w, queries);
        mlp = ggml_add(ctx, mlp, blk.mlp_fc1_b);
        mlp = ggml_relu(ctx, mlp);
        mlp = ggml_mul_mat(ctx, blk.mlp_fc2_w, mlp);
        mlp = ggml_add(ctx, mlp, blk.mlp_fc2_b);
        queries = ggml_add(ctx, queries, mlp);
        queries = sam3_layer_norm(ctx, queries, blk.norm3_w, blk.norm3_b);
    }

    // 4. Cross-attention: image attending to tokens
    {
        auto* q = sam3_add_token_batch(ctx, queries, query_pe);
        auto* k = sam3_add_token_batch(ctx, keys, key_pe);
        // Note: q and k are swapped — image (k) attends to tokens (q)
        auto* attn_out = sam3_sam_attention(ctx, k, q, queries, blk.ca_img2tok, n_heads);
        keys = sam3_add_token_batch(ctx, keys, attn_out);
        keys = sam3_layer_norm(ctx, keys, blk.norm4_w, blk.norm4_b);
    }
}

// MLP forward: N layers with ReLU (except last), optional sigmoid on last
inline struct ggml_tensor* sam3_mlp_forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* const* weights,
    struct ggml_tensor* const* biases,
    int n_layers,
    bool sigmoid_output = false) {
    for (int i = 0; i < n_layers; ++i) {
        x = ggml_mul_mat(ctx, weights[i], x);
        x = ggml_add(ctx, x, biases[i]);
        if (i < n_layers - 1) {
            x = ggml_relu(ctx, x);
        }
    }
    if (sigmoid_output) {
        x = ggml_sigmoid(ctx, x);
    }
    return x;
}

// Full SAM mask decoder graph
// Inputs:
//   image_feats:  [D, H, H, B] — tracker neck features (scale 2 = 72×72)
//   image_pe:     [D, H, H, B] — dense positional encoding
//   sparse_emb:   [D, N_pts, B] — sparse prompt embeddings
//   dense_emb:    [D, H, H, B] — dense prompt embeddings (no_mask default)
//   feat_s0:      [256, H0, H0, B] — high-res features (scale 0 = 288×288)
//   feat_s1:      [256, H1, H1, B] — mid-res features (scale 1 = 144×144)
// Outputs: sam3_dec_result with masks, iou_pred, obj_score, sam_token_out
struct sam3_dec_result {
    struct ggml_tensor* masks;        // [H4*H4, N_masks, B], object-major when downloaded
    struct ggml_tensor* iou_pred;     // [N_masks, 1, B]
    struct ggml_tensor* obj_score;    // [1, 1, B]
    struct ggml_tensor* sam_token;    // [D, 1, B] — for object pointer
    struct ggml_tensor* mask_tokens;  // [D, N_masks, B] — raw SAM mask tokens
};

inline sam3_dec_result sam3_build_sam_dec_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* image_feats,  // [D, H, H, 1|B]
    struct ggml_tensor* image_pe,     // [D, H, H, 1|B]
    struct ggml_tensor* sparse_emb,   // [D, N_pts, B]
    struct ggml_tensor* dense_emb,    // [D, H, H, 1|B]
    struct ggml_tensor* feat_s0,      // [256, H*4, H*4, 1|B] high-res
    struct ggml_tensor* feat_s1,     // [256, H*2, H*2, 1|B] mid-res
    int eff_feat_size = 0)
{
    if (!image_feats || !image_pe || !sparse_emb || !dense_emb || !feat_s0 || !feat_s1)
        throw std::invalid_argument("SAM decoder inputs are incomplete");
    const auto& dec = model.sam_dec;
    const auto& hp = model.hparams;
    const int D = hp.sam_embed_dim;  // 256
    const int H = (eff_feat_size > 0) ? eff_feat_size : hp.n_img_embd();
    const int N_pts = (int)sparse_emb->ne[1];
    const int64_t B = sparse_emb->ne[2];
    const int n_heads = 8;                               // SAM uses 8 heads
    const int num_mask_tokens = hp.sam_n_multimask + 1;  // 4
    if (!image_feats || !image_pe || !sparse_emb || !dense_emb || !feat_s0 || !feat_s1 ||
        B <= 0 || image_feats->type != GGML_TYPE_F32 || image_feats->ne[0] != D ||
        image_feats->ne[1] != H || image_feats->ne[2] != H ||
        (image_feats->ne[3] != 1 && image_feats->ne[3] != B) ||
        image_pe->type != GGML_TYPE_F32 || image_pe->ne[0] != D || image_pe->ne[1] != H ||
        image_pe->ne[2] != H || (image_pe->ne[3] != 1 && image_pe->ne[3] != B) ||
        sparse_emb->type != GGML_TYPE_F32 || sparse_emb->ne[0] != D || sparse_emb->ne[1] <= 0 ||
        sparse_emb->ne[2] != B || sparse_emb->ne[3] != 1 ||
        dense_emb->type != GGML_TYPE_F32 || dense_emb->ne[0] != D || dense_emb->ne[1] != H ||
        dense_emb->ne[2] != H || (dense_emb->ne[3] != 1 && dense_emb->ne[3] != B) ||
        feat_s0->type != GGML_TYPE_F32 || feat_s0->ne[0] != 256 || feat_s0->ne[1] != H * 4 ||
        feat_s0->ne[2] != H * 4 || (feat_s0->ne[3] != 1 && feat_s0->ne[3] != B) ||
        feat_s1->type != GGML_TYPE_F32 || feat_s1->ne[0] != 256 || feat_s1->ne[1] != H * 2 ||
        feat_s1->ne[2] != H * 2 || (feat_s1->ne[3] != 1 && feat_s1->ne[3] != B))
        throw std::invalid_argument("SAM decoder inputs must use matching F32 spatial batches");

    // ── Concatenate output tokens ────────────────────────────────────────
    // When pred_obj_scores=True:  [obj_score(1,D), iou(1,D), masks(4,D)] = 6 tokens
    // When pred_obj_scores=False: [iou(1,D), masks(4,D)] = 5 tokens (older SAM2)
    const bool has_obj_score = (dec.obj_score_token != nullptr);
    const int n_special = (has_obj_score ? 6 : 5);

    struct ggml_tensor* output_tokens;
    if (has_obj_score) {
        output_tokens = ggml_concat(ctx, dec.obj_score_token, dec.iou_token, 1);
    } else {
        output_tokens = ggml_reshape_2d(ctx, dec.iou_token, D, 1);
    }
    output_tokens = ggml_concat(ctx, output_tokens, dec.mask_tokens, 1);
    output_tokens = ggml_reshape_3d(ctx, output_tokens, D, n_special, 1);
    output_tokens = sam3_repeat_batch_3d(ctx, output_tokens, B);
    auto* tokens = ggml_concat(ctx, output_tokens, sparse_emb, 1);
    ggml_set_name(tokens, "sam_dec_tokens_initial");

    const int N_tok = 6 + N_pts;

    auto* src = sam3_add_spatial_batch(ctx, image_feats, dense_emb);
    if (src->ne[3] != B) src = ggml_repeat_4d(ctx, src, D, H, H, B);
    src = ggml_reshape_3d(ctx, src, D, H * H, B);
    auto* pos_src = ggml_reshape_3d(ctx, image_pe, D, H * H, image_pe->ne[3]);
    if (pos_src->ne[2] != B) pos_src = sam3_repeat_batch_3d(ctx, pos_src, B);

    auto* queries = tokens;
    auto* keys = src;
    auto* query_pe = tokens;  // query PE = initial point embedding
    auto* key_pe = pos_src;

    for (int i = 0; i < hp.sam_dec_depth; ++i) {
        sam3_twoway_block_forward(ctx, queries, keys, query_pe, key_pe,
                                  dec.twoway_blocks[i], n_heads,
                                  /*skip_first_layer_pe=*/(i == 0));
        sam3_name_tensorf(queries, "sam_dec_block%d_queries", i);
        sam3_name_tensorf(keys, "sam_dec_block%d_keys", i);
    }

    // Final attention: tokens → image
    {
        auto* q = sam3_add_token_batch(ctx, queries, query_pe);
        auto* k = sam3_add_token_batch(ctx, keys, key_pe);
        auto* attn_out = sam3_sam_attention(ctx, q, k, keys, dec.final_attn, n_heads);
        queries = sam3_add_token_batch(ctx, queries, attn_out);
        queries = sam3_layer_norm(ctx, queries, dec.final_norm_w, dec.final_norm_b);
        ggml_set_name(queries, "sam_dec_final_queries");
    }

    // ── Extract output tokens ────────────────────────────────────────────
    // With pred_obj_scores=True (6 tokens):  obj(0), iou(1), masks(2..5)
    // With pred_obj_scores=False (5 tokens): iou(0), masks(1..4)
    const int s = has_obj_score ? 1 : 0;
    auto* iou_token_out = ggml_view_3d(ctx, queries, D, 1, B,
                                       queries->nb[1], queries->nb[2],
                                       s * queries->nb[1]);
    iou_token_out = ggml_cont(ctx, iou_token_out);  // [D, 1, B]

    auto* mask_tokens_out = ggml_view_3d(ctx, queries, D, num_mask_tokens, B,
                                         queries->nb[1], queries->nb[2],
                                         (s + 1) * queries->nb[1]);
    mask_tokens_out = ggml_cont(ctx, mask_tokens_out);  // [D, 4, B]
    ggml_set_name(mask_tokens_out, "sam_dec_mask_tokens");

    struct ggml_tensor* obj_in = nullptr;
    if (has_obj_score) {
        obj_in = ggml_view_3d(ctx, queries, D, 1, B,
                              queries->nb[1], queries->nb[2], 0);
        obj_in = ggml_cont(ctx, obj_in);  // [D, 1, B]
    }

    // SAM output token = first mask token, used for object pointer
    auto* sam_token = ggml_view_3d(ctx, queries, D, 1, B,
                                   queries->nb[1], queries->nb[2], (s + 1) * queries->nb[1]);
    sam_token = ggml_cont(ctx, sam_token);  // [D, 1, B]
    ggml_set_name(sam_token, "sam_dec_sam_token");

    // Upscale: [D, H*H, 1] → ConvTranspose → high-res masks
    auto* src_img = ggml_reshape_4d(ctx, keys, D, H, H, B);
    src_img = ggml_cont(ctx, ggml_permute(ctx, src_img, 2, 0, 1, 3));

    auto* up1 = sam3_deconv_2x2(ctx, dec.up1_w, src_img);
    up1 = ggml_add(ctx, up1, ggml_reshape_4d(ctx, dec.up1_b, 1, 1, ggml_nelements(dec.up1_b), 1));

    auto* fs1 = ggml_cont(ctx, ggml_permute(ctx, feat_s1, 2, 0, 1, 3));
    auto* hs1 = sam3_conv_2d_sk_p0(ctx, dec.conv_s1_w, fs1);             // [W, H, 64, B]
    hs1 = ggml_add(ctx, hs1, ggml_reshape_4d(ctx, dec.conv_s1_b, 1, 1, 64, 1));
    ggml_set_name(hs1, "sam_dec_feat_s1_proj");

    // Python: act1(ln1(dc1(src) + feat_s1)) — add before LayerNorm
    up1 = ggml_add(ctx, up1, hs1);
    up1 = ggml_cont(ctx, ggml_permute(ctx, up1, 1, 2, 0, 3));
    up1 = sam3_layer_norm_2d(ctx, up1, dec.up1_norm_w, dec.up1_norm_b);

    up1 = ggml_gelu_erf(ctx, up1);

    // Permute back to [W, H, C, B] for next deconv
    up1 = ggml_cont(ctx, ggml_permute(ctx, up1, 2, 0, 1, 3));  // [144, 144, 64, 1]

    // dc2: ConvTranspose2d(64, 32, k=2, s=2) → [288, 288, 32, 1]
    auto* up2 = sam3_deconv_2x2(ctx, dec.up2_w, up1);
    up2 = ggml_add(ctx, up2, ggml_reshape_4d(ctx, dec.up2_b, 1, 1, ggml_nelements(dec.up2_b), 1));

    // conv_s0: 1x1 conv on feat_s0 (256→32). feat_s0 is [C, W, H, B] — permute for conv.
    auto* fs0 = ggml_cont(ctx, ggml_permute(ctx, feat_s0, 2, 0, 1, 3));  // [W, H, C, B]
    auto* hs0 = sam3_conv_2d_sk_p0(ctx, dec.conv_s0_w, fs0);             // [W, H, 32, B]
    hs0 = ggml_add(ctx, hs0, ggml_reshape_4d(ctx, dec.conv_s0_b, 1, 1, 32, 1));
    ggml_set_name(hs0, "sam_dec_feat_s0_proj");

    // Python: act2(dc2(upscaled_embedding) + feat_s0) — no LayerNorm here
    up2 = ggml_add(ctx, up2, hs0);  // both [W, H, 32, B]

    // Permute to [C, W, H, B] for subsequent operations
    up2 = ggml_cont(ctx, ggml_permute(ctx, up2, 1, 2, 0, 3));  // [32, 288, 288, 1]

    // GELU activation (exact, matching Python nn.GELU)
    up2 = ggml_gelu_erf(ctx, up2);

    // up2: [32, 288, 288, 1] — this is our upscaled_embedding
    ggml_set_name(up2, "sam_dec_upscaled");

    // ── Hypernetwork: predict masks ──────────────────────────────────────
    // For each mask token i, pass through 3-layer MLP to get [32] vector
    // Then dot product with upscaled_embedding [32, (H*4)^2] to get mask
    const int H4 = H * 4;
    auto* up_flat = ggml_reshape_3d(ctx, up2, 32, H4 * H4, B);

    // Process each mask token through its hypernetwork MLP
    // mask_tokens_out: [D, 4, 1]
    struct ggml_tensor* mask_list[4];
    for (int m = 0; m < num_mask_tokens; ++m) {
        // Extract token m: [D, 1, 1]
        auto* tok = ggml_view_3d(ctx, mask_tokens_out, D, 1, B,
                                 mask_tokens_out->nb[1], mask_tokens_out->nb[2],
                                 m * mask_tokens_out->nb[1]);
        tok = ggml_cont(ctx, tok);  // [D, 1, B]

        // MLP: 3 layers, 256→256→256→32, ReLU on first two
        auto* hyper = sam3_mlp_forward(ctx, tok,
                                       dec.hyper_w[m], dec.hyper_b[m], 3);
        // hyper: [32, 1, B]

        // Dot product: hyper^T @ up_flat → [1, 288*288, 1]
        // Use mul_mat: up_flat^T [288*288, 32] @ hyper [32, 1] → [288*288, 1, 1]
        auto* mask = ggml_mul_mat(ctx, up_flat, hyper);  // [H4*H4, 1, B]
        mask_list[m] = mask;
    }

    // Stack masks: [H4*H4, 4, B]
    auto* masks = mask_list[0];
    for (int m = 1; m < num_mask_tokens; ++m) {
        masks = ggml_concat(ctx, masks, mask_list[m], 1);
    }
    ggml_set_name(masks, "sam_dec_masks");

    // ── IoU prediction ───────────────────────────────────────────────────
    // iou_token_out: [D, 1, B]
    auto* iou_pred = sam3_mlp_forward(ctx, iou_token_out,
                                      dec.iou_head_w, dec.iou_head_b, 3,
                                      /*sigmoid_output=*/true);
    // iou_pred: [4, 1, B]
    iou_pred = ggml_reshape_3d(ctx, iou_pred, num_mask_tokens, 1, B);
    ggml_set_name(iou_pred, "sam_dec_iou");

    // ── Object score ─────────────────────────────────────────────────────
    struct ggml_tensor* obj_score;
    if (has_obj_score) {
        // obj_in: [D, 1, B] → MLP → [1, 1, B]
        obj_score = sam3_mlp_forward(ctx, obj_in,
                                     dec.obj_head_w, dec.obj_head_b, 3);
        obj_score = ggml_reshape_3d(ctx, obj_score, 1, 1, B);
    } else {
        // No obj_score prediction — return raw logit 10.0 (sigmoid ≈ 1.0, object always present).
        obj_score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, B);
        ggml_set_name(obj_score, "sam_dec_obj_score");
        ggml_set_input(obj_score);
        // Mark that callers must set this to 10.0f before compute
    }

    sam3_dec_result res;
    res.masks = masks;
    res.iou_pred = iou_pred;
    res.obj_score = obj_score;
    res.sam_token = sam_token;
    res.mask_tokens = mask_tokens_out;
    return res;
}



} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_VIDEO_MASK_DECODER_HPP
