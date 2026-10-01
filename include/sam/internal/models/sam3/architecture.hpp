#ifndef SAM_CPP_INTERNAL_MODELS_SAM3_ARCHITECTURE_HPP
#define SAM_CPP_INTERNAL_MODELS_SAM3_ARCHITECTURE_HPP

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

#include "ggml.h"
#include "tracking/architecture.hpp"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace sam::internal::sam3 {

struct sam3_hparams {
    int32_t img_size = 1008, patch_size = 14, vit_embed_dim = 1024, vit_depth = 32;
    int32_t vit_num_heads = 16, vit_mlp_dim = 4736, vit_window_size = 24, n_global_attn = 4;
    int32_t global_attn_idx[4] = {7,15,23,31};
    int32_t text_width = 1024, text_heads = 16, text_layers = 24, text_ctx_len = 32;
    int32_t text_vocab_size = 49408, text_out_dim = 256, neck_dim = 256;
    int32_t fenc_layers = 6, fenc_heads = 8, fenc_ffn_dim = 2048;
    int32_t ddec_layers = 6, ddec_heads = 8, ddec_ffn_dim = 2048, ddec_num_queries = 200;
    int32_t geom_layers = 3;
    int32_t sam_embed_dim = 256, sam_dec_depth = 2, sam_n_multimask = 3;
    int32_t mem_out_dim = 64, mem_attn_layers = 4, num_maskmem = 7;
    int32_t n_img_embd() const { return img_size / patch_size; }
    int32_t n_img_tokens() const { return n_img_embd() * n_img_embd(); }
    int32_t vit_head_dim() const { return vit_embed_dim / vit_num_heads; }
    bool is_global_attn(int layer) const {
        return std::find(std::begin(global_attn_idx), std::end(global_attn_idx), layer) != std::end(global_attn_idx);
    }
};
struct sam3_vit_block {
    struct ggml_tensor* norm1_w   = nullptr;
    struct ggml_tensor* norm1_b   = nullptr;
    struct ggml_tensor* qkv_w     = nullptr;
    struct ggml_tensor* qkv_b     = nullptr;
    struct ggml_tensor* proj_w    = nullptr;
    struct ggml_tensor* proj_b    = nullptr;
    struct ggml_tensor* norm2_w   = nullptr;
    struct ggml_tensor* norm2_b   = nullptr;
    struct ggml_tensor* mlp_fc1_w = nullptr;
    struct ggml_tensor* mlp_fc1_b = nullptr;
    struct ggml_tensor* mlp_fc2_w = nullptr;
    struct ggml_tensor* mlp_fc2_b = nullptr;
    struct ggml_tensor* freqs_cis = nullptr;  // [N, 32, 2] RoPE
};

struct sam3_vit {
    struct ggml_tensor*          patch_embed_w = nullptr;  // [patch, patch, 3, embed]
    struct ggml_tensor*          pos_embed     = nullptr;  // [embed, 24, 24, 1]
    struct ggml_tensor*          ln_pre_w      = nullptr;
    struct ggml_tensor*          ln_pre_b      = nullptr;
    std::vector<sam3_vit_block>  blocks;
};

/*
** ── Neck (SimpleFPN) ─────────────────────────────────────────────────────
*/

struct sam3_neck_scale {
    struct ggml_tensor* deconv1_w  = nullptr;
    struct ggml_tensor* deconv1_b  = nullptr;
    struct ggml_tensor* deconv2_w  = nullptr;  // only for 4x scale
    struct ggml_tensor* deconv2_b  = nullptr;
    struct ggml_tensor* conv1x1_w  = nullptr;
    struct ggml_tensor* conv1x1_b  = nullptr;
    struct ggml_tensor* conv3x3_w  = nullptr;
    struct ggml_tensor* conv3x3_b  = nullptr;
};

struct sam3_neck {
    sam3_neck_scale scales[4];
    struct ggml_tensor* norms_w[4] = {};
    struct ggml_tensor* norms_b[4] = {};
};

/*
** ── Text Encoder ─────────────────────────────────────────────────────────
*/

struct sam3_text_block {
    struct ggml_tensor* attn_in_proj_w  = nullptr;
    struct ggml_tensor* attn_in_proj_b  = nullptr;
    struct ggml_tensor* attn_out_proj_w = nullptr;
    struct ggml_tensor* attn_out_proj_b = nullptr;
    struct ggml_tensor* ln1_w           = nullptr;
    struct ggml_tensor* ln1_b           = nullptr;
    struct ggml_tensor* ln2_w           = nullptr;
    struct ggml_tensor* ln2_b           = nullptr;
    struct ggml_tensor* mlp_fc1_w       = nullptr;
    struct ggml_tensor* mlp_fc1_b       = nullptr;
    struct ggml_tensor* mlp_fc2_w       = nullptr;
    struct ggml_tensor* mlp_fc2_b       = nullptr;
    struct ggml_tensor* ls1             = nullptr;  // LayerScale
    struct ggml_tensor* ls2             = nullptr;
};

struct sam3_text_encoder {
    struct ggml_tensor* token_embed_w = nullptr;  // [vocab, width]
    struct ggml_tensor* pos_embed     = nullptr;  // [ctx_len, width]
    struct ggml_tensor* ln_final_w    = nullptr;
    struct ggml_tensor* ln_final_b    = nullptr;
    struct ggml_tensor* resizer_w     = nullptr;  // [out_dim, width]
    struct ggml_tensor* resizer_b     = nullptr;
    // Note: text_projection ([width, proj_dim]) exists in the checkpoint but is
    // intentionally not loaded. In SAM3, VETextEncoder discards the pooled output
    // that text_projection operates on — only the full token sequence (through
    // resizer) is used for downstream fusion/decoding.
    std::vector<sam3_text_block> blocks;
};

/*
** ── Fusion Encoder ───────────────────────────────────────────────────────
*/

struct sam3_fenc_layer {
    // self-attention
    struct ggml_tensor* sa_in_proj_w  = nullptr;
    struct ggml_tensor* sa_in_proj_b  = nullptr;
    struct ggml_tensor* sa_out_proj_w = nullptr;
    struct ggml_tensor* sa_out_proj_b = nullptr;
    struct ggml_tensor* norm1_w       = nullptr;
    struct ggml_tensor* norm1_b       = nullptr;
    // cross-attention to prompt tokens
    struct ggml_tensor* ca_q_w        = nullptr;
    struct ggml_tensor* ca_q_b        = nullptr;
    struct ggml_tensor* ca_kv_w       = nullptr;
    struct ggml_tensor* ca_kv_b       = nullptr;
    struct ggml_tensor* ca_out_w      = nullptr;
    struct ggml_tensor* ca_out_b      = nullptr;
    struct ggml_tensor* norm2_w       = nullptr;
    struct ggml_tensor* norm2_b       = nullptr;
    // FFN
    struct ggml_tensor* ffn_fc1_w     = nullptr;
    struct ggml_tensor* ffn_fc1_b     = nullptr;
    struct ggml_tensor* ffn_fc2_w     = nullptr;
    struct ggml_tensor* ffn_fc2_b     = nullptr;
    struct ggml_tensor* norm3_w       = nullptr;
    struct ggml_tensor* norm3_b       = nullptr;
};

struct sam3_fusion_encoder {
    std::vector<sam3_fenc_layer> layers;
};

/*
** ── DETR Decoder ─────────────────────────────────────────────────────────
*/

struct sam3_ddec_layer {
    // self-attention
    struct ggml_tensor* sa_in_proj_w   = nullptr;
    struct ggml_tensor* sa_in_proj_b   = nullptr;
    struct ggml_tensor* sa_out_proj_w  = nullptr;
    struct ggml_tensor* sa_out_proj_b  = nullptr;
    struct ggml_tensor* norm1_w        = nullptr;
    struct ggml_tensor* norm1_b        = nullptr;
    // cross-attention to image
    struct ggml_tensor* ca_q_w         = nullptr;
    struct ggml_tensor* ca_q_b         = nullptr;
    struct ggml_tensor* ca_kv_w        = nullptr;
    struct ggml_tensor* ca_kv_b        = nullptr;
    struct ggml_tensor* ca_out_w       = nullptr;
    struct ggml_tensor* ca_out_b       = nullptr;
    struct ggml_tensor* norm2_w        = nullptr;
    struct ggml_tensor* norm2_b        = nullptr;
    // cross-attention to text
    struct ggml_tensor* ca_text_q_w    = nullptr;
    struct ggml_tensor* ca_text_q_b    = nullptr;
    struct ggml_tensor* ca_text_kv_w   = nullptr;
    struct ggml_tensor* ca_text_kv_b   = nullptr;
    struct ggml_tensor* ca_text_out_w  = nullptr;
    struct ggml_tensor* ca_text_out_b  = nullptr;
    struct ggml_tensor* norm3_w        = nullptr;
    struct ggml_tensor* norm3_b        = nullptr;
    // FFN
    struct ggml_tensor* ffn_fc1_w      = nullptr;
    struct ggml_tensor* ffn_fc1_b      = nullptr;
    struct ggml_tensor* ffn_fc2_w      = nullptr;
    struct ggml_tensor* ffn_fc2_b      = nullptr;
    struct ggml_tensor* norm4_w        = nullptr;
    struct ggml_tensor* norm4_b        = nullptr;
    // box refinement MLP (3 layers)
    struct ggml_tensor* bbox_w[3]      = {};
    struct ggml_tensor* bbox_b[3]      = {};
};

struct sam3_detr_decoder {
    struct ggml_tensor*          query_embed      = nullptr;  // [num_queries, 512]
    struct ggml_tensor*          presence_token   = nullptr;  // [1, 256]
    // DotProductScoring MLP
    struct ggml_tensor*          score_mlp_w[2]   = {};
    struct ggml_tensor*          score_mlp_b[2]   = {};
    struct ggml_tensor*          score_ln_w       = nullptr;
    struct ggml_tensor*          score_ln_b       = nullptr;
    // Presence head
    struct ggml_tensor*          presence_head_w[2] = {};
    struct ggml_tensor*          presence_head_b[2] = {};
    std::vector<sam3_ddec_layer> layers;
};

/*
** ── Geometry / Exemplar Encoder ──────────────────────────────────────────
*/

struct sam3_geom_layer {
    struct ggml_tensor* sa_in_proj_w  = nullptr;
    struct ggml_tensor* sa_in_proj_b  = nullptr;
    struct ggml_tensor* sa_out_proj_w = nullptr;
    struct ggml_tensor* sa_out_proj_b = nullptr;
    struct ggml_tensor* norm1_w       = nullptr;
    struct ggml_tensor* norm1_b       = nullptr;
    struct ggml_tensor* ca_q_w        = nullptr;
    struct ggml_tensor* ca_q_b        = nullptr;
    struct ggml_tensor* ca_kv_w       = nullptr;
    struct ggml_tensor* ca_kv_b       = nullptr;
    struct ggml_tensor* ca_out_w      = nullptr;
    struct ggml_tensor* ca_out_b      = nullptr;
    struct ggml_tensor* norm2_w       = nullptr;
    struct ggml_tensor* norm2_b       = nullptr;
    struct ggml_tensor* ffn_fc1_w     = nullptr;
    struct ggml_tensor* ffn_fc1_b     = nullptr;
    struct ggml_tensor* ffn_fc2_w     = nullptr;
    struct ggml_tensor* ffn_fc2_b     = nullptr;
    struct ggml_tensor* norm3_w       = nullptr;
    struct ggml_tensor* norm3_b       = nullptr;
};

struct sam3_geom_encoder {
    // Direct projections
    struct ggml_tensor* point_proj_w      = nullptr;  // Linear(2, D)
    struct ggml_tensor* point_proj_b      = nullptr;
    struct ggml_tensor* box_proj_w        = nullptr;  // Linear(4, D)
    struct ggml_tensor* box_proj_b        = nullptr;
    // Pooling projections
    struct ggml_tensor* point_pool_proj_w = nullptr;  // Linear(D, D)
    struct ggml_tensor* point_pool_proj_b = nullptr;
    struct ggml_tensor* box_pool_proj_w   = nullptr;  // Conv2d(D, D, 7)
    struct ggml_tensor* box_pool_proj_b   = nullptr;
    // Positional encoding projections
    struct ggml_tensor* point_pos_proj_w  = nullptr;  // Linear(D, D)
    struct ggml_tensor* point_pos_proj_b  = nullptr;
    struct ggml_tensor* box_pos_proj_w    = nullptr;  // Linear(258, 256)
    struct ggml_tensor* box_pos_proj_b    = nullptr;
    // Label and CLS embeddings
    struct ggml_tensor* type_embed        = nullptr;  // Embedding(2, D)
    struct ggml_tensor* cls_token         = nullptr;  // Embedding(1, D)
    // Final projection + norms
    struct ggml_tensor* post_proj_w       = nullptr;  // Linear(D, D)
    struct ggml_tensor* post_proj_b       = nullptr;
    struct ggml_tensor* norm_w            = nullptr;  // LayerNorm final_proj
    struct ggml_tensor* norm_b            = nullptr;
    struct ggml_tensor* encode_norm_w     = nullptr;  // LayerNorm after xfmr
    struct ggml_tensor* encode_norm_b     = nullptr;
    struct ggml_tensor* img_pre_norm_w    = nullptr;  // LayerNorm before pool
    struct ggml_tensor* img_pre_norm_b    = nullptr;
    std::vector<sam3_geom_layer> layers;
};

/*
** ── Segmentation Head (MaskFormer) ───────────────────────────────────────
*/

struct sam3_seg_head {
    struct ggml_tensor* up_conv_w[3]      = {};
    struct ggml_tensor* up_conv_b[3]      = {};
    struct ggml_tensor* up_norm_w[3]      = {};
    struct ggml_tensor* up_norm_b[3]      = {};
    struct ggml_tensor* ca_prompt_q_w     = nullptr;
    struct ggml_tensor* ca_prompt_q_b     = nullptr;
    struct ggml_tensor* ca_prompt_kv_w    = nullptr;
    struct ggml_tensor* ca_prompt_kv_b    = nullptr;
    struct ggml_tensor* ca_prompt_out_w   = nullptr;
    struct ggml_tensor* ca_prompt_out_b   = nullptr;
    struct ggml_tensor* mask_embed_w      = nullptr;
    struct ggml_tensor* mask_embed_b      = nullptr;
};

/*
** ── Model Weight Bundle ─────────────────────────────────────────────────
*/

struct sam3_model {
    sam3_hparams hparams;
    ggml_type weight_type = GGML_TYPE_F16;
    sam3_vit vit;
    sam3_neck neck_det, neck_trk;
    sam3_sam_prompt_enc sam_pe;
    sam3_sam_mask_dec sam_dec;
    sam3_mem_enc mem_enc;
    sam3_mem_attn mem_attn;
    ggml_tensor* mem_attn_norm_w = nullptr;
    ggml_tensor* mem_attn_norm_b = nullptr;
    ggml_tensor* obj_ptr_proj_w[3] = {};
    ggml_tensor* obj_ptr_proj_b[3] = {};
    ggml_tensor* obj_ptr_tpos_w = nullptr;
    ggml_tensor* obj_ptr_tpos_b = nullptr;
    ggml_tensor* no_obj_ptr = nullptr;
    ggml_tensor* no_mem_embed = nullptr;
    ggml_tensor* no_mem_pos_enc = nullptr;
    ggml_tensor* no_obj_embed_spatial = nullptr;
    sam3_text_encoder text_enc;
    sam3_fusion_encoder fenc;
    sam3_detr_decoder ddec;
    sam3_geom_encoder geom_enc;
    sam3_seg_head seg_head;
    ggml_context* ctx = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::map<std::string, ggml_type> source_types;
    ggml_type source_type(const std::string& name, ggml_type fallback) const {
        const auto found = source_types.find(name);
        return found == source_types.end() ? fallback : found->second;
    }
};

} // namespace sam::internal::sam3

#endif // SAM_CPP_INTERNAL_MODELS_SAM3_ARCHITECTURE_HPP
