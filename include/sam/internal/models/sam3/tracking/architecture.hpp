#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_ARCHITECTURE_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_ARCHITECTURE_HPP

// Adapted from PABannier/sam3.cpp, revision 416186c501d060df7ca02989d49b38080f5f81f3.
// Copyright (c) 2025-2026 Pierre-Antoine Bannier. MIT: licenses/sam3.cpp-MIT.txt.

#include "ggml.h"
#include <vector>

namespace sam::internal::sam3 {

struct sam3_sam_prompt_enc {
    struct ggml_tensor* pe_gaussian         = nullptr;  // [2, 128]
    struct ggml_tensor* point_embed[4]      = {};       // neg, pos, box_tl, box_br
    struct ggml_tensor* not_a_point_embed   = nullptr;  // [256]
    struct ggml_tensor* no_mask_embed       = nullptr;  // [256]
    struct ggml_tensor* mask_ds_conv_w[3]   = {};
    struct ggml_tensor* mask_ds_conv_b[3]   = {};
    struct ggml_tensor* mask_ds_norm_w[2]   = {};
    struct ggml_tensor* mask_ds_norm_b[2]   = {};
};

/*
** ── SAM Mask Decoder (Tracker Path) ──────────────────────────────────────
*/

struct sam3_sam_attn {
    struct ggml_tensor* q_w   = nullptr;
    struct ggml_tensor* q_b   = nullptr;
    struct ggml_tensor* k_w   = nullptr;
    struct ggml_tensor* k_b   = nullptr;
    struct ggml_tensor* v_w   = nullptr;
    struct ggml_tensor* v_b   = nullptr;
    struct ggml_tensor* out_w = nullptr;
    struct ggml_tensor* out_b = nullptr;
};

struct sam3_twoway_block {
    sam3_sam_attn       self_attn;
    sam3_sam_attn       ca_tok2img;
    sam3_sam_attn       ca_img2tok;
    struct ggml_tensor* norm1_w   = nullptr;
    struct ggml_tensor* norm1_b   = nullptr;
    struct ggml_tensor* norm2_w   = nullptr;
    struct ggml_tensor* norm2_b   = nullptr;
    struct ggml_tensor* norm3_w   = nullptr;
    struct ggml_tensor* norm3_b   = nullptr;
    struct ggml_tensor* norm4_w   = nullptr;
    struct ggml_tensor* norm4_b   = nullptr;
    struct ggml_tensor* mlp_fc1_w = nullptr;
    struct ggml_tensor* mlp_fc1_b = nullptr;
    struct ggml_tensor* mlp_fc2_w = nullptr;
    struct ggml_tensor* mlp_fc2_b = nullptr;
};

struct sam3_sam_mask_dec {
    struct ggml_tensor*           iou_token       = nullptr;  // [1, 256]
    struct ggml_tensor*           mask_tokens     = nullptr;  // [4, 256]
    struct ggml_tensor*           obj_score_token = nullptr;  // [1, 256]

    std::vector<sam3_twoway_block> twoway_blocks;             // [2]

    sam3_sam_attn                 final_attn;
    struct ggml_tensor*           final_norm_w    = nullptr;
    struct ggml_tensor*           final_norm_b    = nullptr;

    // upscaling
    struct ggml_tensor* up1_w        = nullptr;
    struct ggml_tensor* up1_b        = nullptr;
    struct ggml_tensor* up1_norm_w   = nullptr;
    struct ggml_tensor* up1_norm_b   = nullptr;
    struct ggml_tensor* up2_w        = nullptr;
    struct ggml_tensor* up2_b        = nullptr;

    // high-res feature convolutions
    struct ggml_tensor* conv_s0_w    = nullptr;
    struct ggml_tensor* conv_s0_b    = nullptr;
    struct ggml_tensor* conv_s1_w    = nullptr;
    struct ggml_tensor* conv_s1_b    = nullptr;

    // hypernetwork MLPs: 4 masks x 3 layers
    struct ggml_tensor* hyper_w[4][3]  = {};
    struct ggml_tensor* hyper_b[4][3]  = {};

    // IoU prediction head (3 layers)
    struct ggml_tensor* iou_head_w[3]  = {};
    struct ggml_tensor* iou_head_b[3]  = {};

    // object score head (3 layers)
    struct ggml_tensor* obj_head_w[3]  = {};
    struct ggml_tensor* obj_head_b[3]  = {};
};

/*
** ── Memory Encoder ───────────────────────────────────────────────────────
*/

struct sam3_mem_enc {
    // mask downsampler (4 conv stages + final 1x1)
    struct ggml_tensor* ds_conv_w[5]      = {};
    struct ggml_tensor* ds_conv_b[5]      = {};
    struct ggml_tensor* ds_norm_w[4]      = {};
    struct ggml_tensor* ds_norm_b[4]      = {};
    // pixel feature projection
    struct ggml_tensor* pix_proj_w        = nullptr;
    struct ggml_tensor* pix_proj_b        = nullptr;
    // fuser (2 CXBlock layers)
    struct ggml_tensor* fuser_dw_w[2]     = {};
    struct ggml_tensor* fuser_dw_b[2]     = {};
    struct ggml_tensor* fuser_norm_w[2]   = {};
    struct ggml_tensor* fuser_norm_b[2]   = {};
    struct ggml_tensor* fuser_fc1_w[2]    = {};
    struct ggml_tensor* fuser_fc1_b[2]    = {};
    struct ggml_tensor* fuser_fc2_w[2]    = {};
    struct ggml_tensor* fuser_fc2_b[2]    = {};
    struct ggml_tensor* fuser_gamma[2]    = {};
    // output projection
    struct ggml_tensor* out_proj_w        = nullptr;
    struct ggml_tensor* out_proj_b        = nullptr;
    // temporal pos encodings
    struct ggml_tensor* tpos[7]           = {};
};

/*
** ── Memory Attention (Tracker Transformer) ───────────────────────────────
*/

struct sam3_mem_attn_layer {
    // self-attention (RoPE, 1 head, 256-dim)
    struct ggml_tensor* sa_q_w    = nullptr;
    struct ggml_tensor* sa_q_b    = nullptr;
    struct ggml_tensor* sa_k_w    = nullptr;
    struct ggml_tensor* sa_k_b    = nullptr;
    struct ggml_tensor* sa_v_w    = nullptr;
    struct ggml_tensor* sa_v_b    = nullptr;
    struct ggml_tensor* sa_out_w  = nullptr;
    struct ggml_tensor* sa_out_b  = nullptr;
    struct ggml_tensor* norm1_w   = nullptr;
    struct ggml_tensor* norm1_b   = nullptr;
    // cross-attention (RoPE, kv_dim=64)
    struct ggml_tensor* ca_q_w    = nullptr;
    struct ggml_tensor* ca_q_b    = nullptr;
    struct ggml_tensor* ca_k_w    = nullptr;  // [256, 64]
    struct ggml_tensor* ca_k_b    = nullptr;
    struct ggml_tensor* ca_v_w    = nullptr;  // [256, 64]
    struct ggml_tensor* ca_v_b    = nullptr;
    struct ggml_tensor* ca_out_w  = nullptr;
    struct ggml_tensor* ca_out_b  = nullptr;
    struct ggml_tensor* norm2_w   = nullptr;
    struct ggml_tensor* norm2_b   = nullptr;
    // FFN
    struct ggml_tensor* ffn_fc1_w = nullptr;
    struct ggml_tensor* ffn_fc1_b = nullptr;
    struct ggml_tensor* ffn_fc2_w = nullptr;
    struct ggml_tensor* ffn_fc2_b = nullptr;
    struct ggml_tensor* norm3_w   = nullptr;
    struct ggml_tensor* norm3_b   = nullptr;
};

struct sam3_mem_attn {
    std::vector<sam3_mem_attn_layer> layers;
};



} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_ARCHITECTURE_HPP
