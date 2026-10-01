#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MEMORY_ENCODER_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MEMORY_ENCODER_HPP

// Adapted from PABannier/sam3.cpp, revision 416186c501d060df7ca02989d49b38080f5f81f3.
// Copyright (c) 2025-2026 Pierre-Antoine Bannier. MIT: licenses/sam3.cpp-MIT.txt.

#include "../architecture.hpp"
#include "../vision.hpp"
#include "../ops.hpp"

namespace sam::internal::sam3 {

inline struct ggml_tensor* sam3_cxblock_forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,       // [D, H, W, B]
    struct ggml_tensor* dw_w,    // [7, 7, 1, D] depthwise
    struct ggml_tensor* dw_b,    // [D]
    struct ggml_tensor* norm_w,  // [D]
    struct ggml_tensor* norm_b,  // [D]
    struct ggml_tensor* fc1_w,   // [D, 1024]
    struct ggml_tensor* fc1_b,   // [1024]
    struct ggml_tensor* fc2_w,   // [1024, D]
    struct ggml_tensor* fc2_b,   // [D]
    struct ggml_tensor* gamma)   // [D]
{
    const int D = (int)x->ne[0];
    const int H = (int)x->ne[1];
    const int W = (int)x->ne[2];

    // ggml conv expects WHCB layout; internal feature maps are stored as CWHB.
    auto* x_whcb = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));

    // Depthwise conv (groups = D): use the direct depthwise path.
    // ggml_conv_2d_dw_direct only supports f32 kernel — cast if needed.
    auto* dw_w_f32 = (dw_w->type == GGML_TYPE_F32) ? dw_w : ggml_cast(ctx, dw_w, GGML_TYPE_F32);
    auto* h = ggml_conv_2d_dw_direct(ctx, dw_w_f32, x_whcb, 1, 1, 3, 3, 1, 1);
    h = ggml_add(ctx, h, ggml_reshape_4d(ctx, dw_b, 1, 1, D, 1));
    h = ggml_cont(ctx, ggml_permute(ctx, h, 1, 2, 0, 3));

    // LayerNorm2d
    h = sam3_layer_norm_2d(ctx, h, norm_w, norm_b);

    // Pointwise MLP: reshape to [D, H*W, B], apply FC, reshape back
    auto* flat = ggml_reshape_3d(ctx, h, D, H * W, 1);
    flat = ggml_add(ctx, ggml_mul_mat(ctx, fc1_w, flat), fc1_b);
    flat = ggml_gelu_erf(ctx, flat);
    flat = ggml_add(ctx, ggml_mul_mat(ctx, fc2_w, flat), fc2_b);
    h = ggml_reshape_4d(ctx, flat, D, H, W, 1);

    // Residual with learnable scaling: x + gamma * h
    auto* gamma_4d = ggml_reshape_4d(ctx, gamma, D, 1, 1, 1);
    h = ggml_mul(ctx, h, gamma_4d);
    return ggml_add(ctx, x, h);
}



// mask is [16W,16H,1,1] after resize/sigmoid/scale/bias preprocessing;
// pixels are tracker-neck [256,W,H,1] after the explicit BF16 transport.
inline ggml_tensor* build_memory_encoder(ggml_context* ctx, const sam3_model& model,
                                         ggml_tensor* mask, ggml_tensor* pixels,
                                         bool object_present) {
    const auto& weights = model.mem_enc;
    auto* downsampled = mask;
    for (int stage = 0; stage < 4; ++stage) {
        const auto channels = weights.ds_conv_w[stage]->ne[3];
        downsampled = sam3_conv_2d(ctx, weights.ds_conv_w[stage], downsampled, 2, 2, 1, 1);
        downsampled = ggml_add(ctx, downsampled,
            ggml_reshape_4d(ctx, weights.ds_conv_b[stage], 1, 1, channels, 1));
        downsampled = ggml_cont(ctx, ggml_permute(ctx, downsampled, 1, 2, 0, 3));
        downsampled = sam3_layer_norm_2d(ctx, downsampled, weights.ds_norm_w[stage], weights.ds_norm_b[stage]);
        downsampled = ggml_gelu_erf(ctx, downsampled);
        downsampled = ggml_cont(ctx, ggml_permute(ctx, downsampled, 2, 0, 1, 3));
    }
    downsampled = sam3_conv_2d_sk_p0(ctx, weights.ds_conv_w[4], downsampled);
    downsampled = ggml_add(ctx, downsampled, ggml_reshape_4d(ctx, weights.ds_conv_b[4], 1, 1, 256, 1));
    downsampled = ggml_cont(ctx, ggml_permute(ctx, downsampled, 1, 2, 0, 3));
    auto* spatial = ggml_cont(ctx, ggml_permute(ctx, pixels, 2, 0, 1, 3));
    spatial = sam3_conv_2d_sk_p0(ctx, weights.pix_proj_w, spatial);
    spatial = ggml_add(ctx, spatial, ggml_reshape_4d(ctx, weights.pix_proj_b, 1, 1, 256, 1));
    spatial = ggml_cont(ctx, ggml_permute(ctx, spatial, 1, 2, 0, 3));
    auto* fused = ggml_add(ctx, spatial, downsampled);
    for (int stage = 0; stage < 2; ++stage)
        fused = sam3_cxblock_forward(ctx, fused, weights.fuser_dw_w[stage], weights.fuser_dw_b[stage],
            weights.fuser_norm_w[stage], weights.fuser_norm_b[stage], weights.fuser_fc1_w[stage],
            weights.fuser_fc1_b[stage], weights.fuser_fc2_w[stage], weights.fuser_fc2_b[stage],
            weights.fuser_gamma[stage]);
    fused = ggml_cont(ctx, ggml_permute(ctx, fused, 2, 0, 1, 3));
    auto* output = sam3_conv_2d_sk_p0(ctx, weights.out_proj_w, fused);
    output = ggml_add(ctx, output, ggml_reshape_4d(ctx, weights.out_proj_b, 1, 1, 64, 1));
    output = ggml_cont(ctx, ggml_permute(ctx, output, 1, 2, 0, 3));
    if (!object_present)
        output = ggml_add(ctx, output, ggml_reshape_4d(ctx, model.no_obj_embed_spatial, 64, 1, 1, 1));
    return output;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MEMORY_ENCODER_HPP
