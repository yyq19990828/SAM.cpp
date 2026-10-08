#ifndef SAM_CPP_SRC_MODELS_SAM3_MASK_DECODER_HPP
#define SAM_CPP_SRC_MODELS_SAM3_MASK_DECODER_HPP

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
#include <cstdint>
#include <string>

namespace sam::internal::sam3 {

/*****************************************************************************
** Segmentation head (MaskFormer) — graph building
*****************************************************************************/

// Build pixel decoder: progressively upsample FPN features.
// fpn_feats[0]: [D, 288, 288, B] (highest res)
// fpn_feats[1]: [D, 144, 144, B]
// fpn_feats[2]: [D,  72,  72, B] (lowest res)
// Returns: [D, 288, 288, B] pixel features
//
// Python PixelDecoder.forward:
//   prev_fpn = backbone_feats[-1]  (lowest res)
//   for bb_feat in backbone_feats[:-1][::-1]:  (iterate from second-lowest to highest)
//       prev_fpn = bb_feat + F.interpolate(prev_fpn, size=bb_feat.shape[-2:], mode="nearest")
//       prev_fpn = conv_layers[i](prev_fpn)    # conv on the MERGED result
//       prev_fpn = F.relu(norms[i](prev_fpn))  # GroupNorm then ReLU
//
// Python uses GroupNorm(8, 256) — we use ggml_group_norm which normalizes ne[2]
// (the channel dim) in groups.  The conv output is [W, H, D, B] with D in ne[2].
inline struct ggml_tensor* sam3_pixel_decoder(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* fpn_feats[3],  // [D, W, H, B] at 3 scales
    ggml_type columns_type = GGML_TYPE_F32)
{
    const auto& seg = model.seg_head;

    // Start from lowest resolution
    auto* feat = fpn_feats[2];  // [D, 72, 72, B]

    // Iteration 0: merge with FPN[1] (144x144)
    // prev_fpn = FPN[1] + upsample(prev_fpn)
    // Permute to [W, H, D, B] for conv operations
    auto* prev = ggml_cont(ctx, ggml_permute(ctx, feat, 2, 0, 1, 3));          // [72, 72, D, B]
    prev = ggml_upscale(ctx, prev, 2, GGML_SCALE_MODE_NEAREST);                // [144, 144, D, B]
    auto* fpn1 = ggml_cont(ctx, ggml_permute(ctx, fpn_feats[1], 2, 0, 1, 3));  // [144, 144, D, B]
    prev = ggml_add(ctx, fpn1, prev);                                          // merged
    // Conv 3x3 on the MERGED result (not individual FPN feat)
    prev = sam3_conv_2d_s1_ph(ctx, seg.up_conv_w[0], prev, columns_type);
    {
        auto* b3d = ggml_reshape_3d(ctx, seg.up_conv_b[0], 1, 1, seg.up_conv_b[0]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, b3d, prev));
    }
    // GroupNorm(8, 256) then ReLU — prev is [W, H, D, B] with D in ne[2]
    prev = ggml_group_norm(ctx, prev, 8, 1e-5f);
    {
        auto* w3d = ggml_reshape_3d(ctx, seg.up_norm_w[0], 1, 1, seg.up_norm_w[0]->ne[0]);
        prev = ggml_mul(ctx, prev, ggml_repeat(ctx, w3d, prev));
        auto* bn3d = ggml_reshape_3d(ctx, seg.up_norm_b[0], 1, 1, seg.up_norm_b[0]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, bn3d, prev));
    }
    prev = ggml_relu(ctx, prev);
    ggml_set_name(prev, "seg_pixel_dec_stage0");

    // Iteration 1: merge with FPN[0] (288x288)
    prev = ggml_upscale(ctx, prev, 2, GGML_SCALE_MODE_NEAREST);                // [288, 288, D, B]
    auto* fpn0 = ggml_cont(ctx, ggml_permute(ctx, fpn_feats[0], 2, 0, 1, 3));  // [288, 288, D, B]
    prev = ggml_add(ctx, fpn0, prev);                                          // merged
    // Conv 3x3 on the MERGED result
    prev = sam3_conv_2d_s1_ph(ctx, seg.up_conv_w[1], prev, columns_type);
    {
        auto* b3d = ggml_reshape_3d(ctx, seg.up_conv_b[1], 1, 1, seg.up_conv_b[1]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, b3d, prev));
    }
    // GroupNorm(8, 256) then ReLU
    prev = ggml_group_norm(ctx, prev, 8, 1e-5f);
    {
        auto* w3d = ggml_reshape_3d(ctx, seg.up_norm_w[1], 1, 1, seg.up_norm_w[1]->ne[0]);
        prev = ggml_mul(ctx, prev, ggml_repeat(ctx, w3d, prev));
        auto* bn3d = ggml_reshape_3d(ctx, seg.up_norm_b[1], 1, 1, seg.up_norm_b[1]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, bn3d, prev));
    }
    prev = ggml_relu(ctx, prev);
    ggml_set_name(prev, "seg_pixel_dec_stage1");

    // Python PixelDecoder allocates 3 conv layers but only uses 2 (one per
    // upsample step). The 3rd conv (up_conv_w[2]) is unused.

    auto* out = ggml_cont(ctx, ggml_permute(ctx, prev, 1, 2, 0, 3));  // [D, 288, 288, B]
    return out;
}

// Build the full segmentation head graph.
//
// Python UniversalSegmentationHead.forward:
//   1. Cross-attend encoder_hidden_states to prompt → updated encoder
//   2. _embed_pixels: replace lowest-res FPN feat with spatial portion of encoder output
//   3. Run pixel decoder on modified FPN feats
//   4. instance_seg_head (Conv1x1)
//   5. mask_predictor: einsum(mask_embed(queries), instance_embeds)
//
// enc_hidden: [D, N_spatial, B] — fusion encoder output (cross-attended in step 1)
// fpn_feats[3]: the 3 FPN features at different resolutions
// query_outputs: [D, N, B] selected object query outputs
// text_features: [D, T, B] for cross-attention (prompt)
// Returns: mask_logits [W*H, N, B] (raw logits, not sigmoid)
inline struct ggml_tensor* sam3_build_seg_head_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* enc_hidden,     // [D, N_spatial, B] fusion encoder output
    struct ggml_tensor* fpn_feats[3],   // FPN features at 3 scales
    struct ggml_tensor* query_outputs,  // [D, N, B]
    struct ggml_tensor* text_features,  // [D, T, B] (for cross-attn, can be nullptr)
    struct ggml_tensor* text_attn_bias = nullptr,
    ggml_type columns_type = GGML_TYPE_F32) {
    const auto& seg = model.seg_head;
    const auto& tensors = model.tensors;
    const int64_t D = enc_hidden->ne[0];     // 256
    const int64_t B = enc_hidden->ne[2];     // 1

    auto* enc = enc_hidden;
    if (text_features) {
        auto* ca_norm = sam3_layer_norm(ctx, enc,
                                        tensors.at("seg.cross_attn_norm.weight"),
                                        tensors.at("seg.cross_attn_norm.bias"));

        auto* ca_mask = sam3_expand_token_attn_bias(ctx, text_attn_bias, enc->ne[1], 8, B);
        auto* ca_out = sam3_multihead_attn_fused(ctx, ca_norm, text_features,
                                                 seg.ca_prompt_q_w, seg.ca_prompt_q_b,
                                                 seg.ca_prompt_out_w, seg.ca_prompt_out_b,
                                                 8, ca_mask);
        enc = ggml_add(ctx, enc, ca_out);
    }
    // enc: [D, N_spatial, B]
    ggml_set_name(enc, "seg_enc_after_ca");

    // Replace lowest-res FPN feat with spatial portion of encoder output
    const int64_t feat_hw = model.hparams.n_img_embd();  // 72
    auto* enc_spatial = ggml_reshape_4d(ctx, enc, D, feat_hw, feat_hw, B);

    struct ggml_tensor* modified_fpn[3] = {
        fpn_feats[0],
        fpn_feats[1],
        enc_spatial,  // replaces original lowest-res FPN
    };

    auto* pixel_feats = sam3_pixel_decoder(ctx, model, modified_fpn, columns_type);

    const int64_t W = pixel_feats->ne[1];  // 288
    const int64_t H = pixel_feats->ne[2];  // 288

    // Instance segmentation head (Conv1x1)
    auto* pf_conv = ggml_cont(ctx, ggml_permute(ctx, pixel_feats, 2, 0, 1, 3));
    pf_conv = sam3_conv_2d_sk_p0(ctx, tensors.at("seg.instance_seg_head.weight"), pf_conv, columns_type);
    {
        auto* b3d = ggml_reshape_3d(ctx, tensors.at("seg.instance_seg_head.bias"),
                                    1, 1, tensors.at("seg.instance_seg_head.bias")->ne[0]);
        pf_conv = ggml_add(ctx, pf_conv, ggml_repeat(ctx, b3d, pf_conv));
    }
    auto* pixel_embed = ggml_cont(ctx, ggml_permute(ctx, pf_conv, 1, 2, 0, 3));  // [D, W, H, B]

    // Mask embedding MLP
    auto* mask_embed = query_outputs;
    for (int j = 0; j < 3; ++j) {
        auto wn = "seg.mask_predictor.mask_embed.layers." + std::to_string(j) + ".weight";
        auto bn = "seg.mask_predictor.mask_embed.layers." + std::to_string(j) + ".bias";
        mask_embed = ggml_mul_mat(ctx, tensors.at(wn), mask_embed);
        mask_embed = ggml_add(ctx, mask_embed, tensors.at(bn));
        if (j < 2) mask_embed = ggml_relu(ctx, mask_embed);
    }
    ggml_set_name(mask_embed, "seg_mask_embed");

    // Mask prediction: einsum('bqc,bchw->bqhw')
    auto* pe_flat = ggml_reshape_3d(ctx, pixel_embed, D, W * H, B);
    auto* masks = ggml_mul_mat(ctx, pe_flat, mask_embed);
    ggml_set_name(masks, "seg_mask_logits");

    return masks;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_MASK_DECODER_HPP
