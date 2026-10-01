#ifndef SAM_CPP_INTERNAL_MODELS_SAM3_OPS_HPP
#define SAM_CPP_INTERNAL_MODELS_SAM3_OPS_HPP

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
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace sam::internal::sam3 {

inline void sam3_name_tensorf(struct ggml_tensor* t, const char* fmt, int index) {
    if (!t) {
        return;
    }
    char name[64];
    snprintf(name, sizeof(name), fmt, index);
    ggml_set_name(t, name);
}

inline struct ggml_tensor* sam3_layer_norm(struct ggml_context* ctx,
                                           struct ggml_tensor* x,
                                           struct ggml_tensor* w,
                                           struct ggml_tensor* b) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul_inplace(ctx, x, w);
    if (b) {
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

// GGML's convolution helper stages F32 activations in F16. Build the same
// graph with F32 im2col; promoting kernels preserves their stored F16 values.
inline ggml_tensor* sam3_conv_2d(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* input,
                                 int stride_x, int stride_y, int padding_x, int padding_y) {
    if (kernel->type != GGML_TYPE_F32) kernel = ggml_cast(ctx, kernel, GGML_TYPE_F32);
    auto* columns = ggml_im2col(ctx, kernel, input, stride_x, stride_y, padding_x, padding_y,
                               1, 1, true, GGML_TYPE_F32);
    auto* output = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, columns, columns->ne[0], columns->ne[3] * columns->ne[2] * columns->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1] * kernel->ne[2], kernel->ne[3]));
    output = ggml_reshape_4d(ctx, output, columns->ne[1], columns->ne[2], columns->ne[3], kernel->ne[3]);
    return ggml_cont(ctx, ggml_permute(ctx, output, 0, 1, 3, 2));
}

inline ggml_tensor* sam3_conv_2d_sk_p0(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* input) {
    return sam3_conv_2d(ctx, kernel, input, static_cast<int>(kernel->ne[0]), static_cast<int>(kernel->ne[1]), 0, 0);
}

inline ggml_tensor* sam3_conv_2d_s1_ph(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* input) {
    return sam3_conv_2d(ctx, kernel, input, 1, 1, static_cast<int>(kernel->ne[0] / 2), static_cast<int>(kernel->ne[1] / 2));
}

inline ggml_tensor* sam3_deconv_2x2(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x) {
    // Native F16 deconvolution also narrows activations. Non-overlapping 2x2
    // kernels are an exact matrix multiplication with F32 activations.
    const int64_t width = x->ne[0], height = x->ne[1], channels = w->ne[2];
    auto* matrix = ggml_reshape_2d(ctx, w, 4 * channels, w->ne[3]);
    matrix = ggml_cont(ctx, ggml_transpose(ctx, matrix));
    auto* input = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
    input = ggml_reshape_2d(ctx, input, x->ne[2], width * height);
    auto* out = ggml_mul_mat(ctx, matrix, input);
    out = ggml_reshape_4d(ctx, out, 4, channels, width, height);
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 3, 1, 2));
    out = ggml_reshape_4d(ctx, out, 2, 2, width, height * channels);
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    return ggml_reshape_4d(ctx, out, 2 * width, 2 * height, channels, 1);
}

// Tracker LayerNorm2d normalizes channels in CWH layout with epsilon 1e-6.
inline ggml_tensor* sam3_layer_norm_2d(ggml_context* ctx, ggml_tensor* x,
                                        ggml_tensor* weight, ggml_tensor* bias) {
    x = ggml_mul(ctx, ggml_norm(ctx, x, 1e-6f), weight);
    return bias ? ggml_add(ctx, x, bias) : x;
}

// Fused Q/K/V multihead attention: [D, N_q, B] attends to [D, N_kv, B].
inline struct ggml_tensor* sam3_multihead_attn_fused(
    struct ggml_context* ctx,
    struct ggml_tensor* q_in,        // [D, N_q, B]
    struct ggml_tensor* kv_in,       // [D, N_kv, B] (can be same as q_in for self-attn)
    struct ggml_tensor* in_proj_w,   // [D, 3*D] — fused QKV weights
    struct ggml_tensor* in_proj_b,   // [3*D]
    struct ggml_tensor* out_proj_w,  // [D, D]
    struct ggml_tensor* out_proj_b,  // [D]
    int n_heads,
    struct ggml_tensor* attn_mask = nullptr)  // [N_kv, N_q] or nullptr
{
    const int64_t D = q_in->ne[0];  // 256
    const int64_t N_q = q_in->ne[1];
    const int64_t B = q_in->ne[2];
    const int64_t N_kv = kv_in->ne[1];
    const int64_t HD = D / n_heads;

    auto* q_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], 0);
    auto* k_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], D * in_proj_w->nb[1]);
    auto* v_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], 2 * D * in_proj_w->nb[1]);

    auto* q_b = ggml_view_1d(ctx, in_proj_b, D, 0);
    auto* k_b = ggml_view_1d(ctx, in_proj_b, D, D * sizeof(float));
    auto* v_b = ggml_view_1d(ctx, in_proj_b, D, 2 * D * sizeof(float));

    auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
    auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, kv_in), k_b);
    auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, kv_in), v_b);

    Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // [HD, N_q, NH, B]

    K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));  // [HD, N_kv, NH, B]

    V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);  // [HD, N_kv, NH, B] non-contiguous; flash_attn uses strides

    float scale = 1.0f / sqrtf((float)HD);
    auto* attn_out = ggml_flash_attn_ext(ctx, Q, K, V, attn_mask, scale, 0.0f, 0.0f);

    auto* merged = ggml_reshape_3d(ctx, attn_out, D, N_q, B);
    merged = ggml_mul_mat(ctx, out_proj_w, merged);
    merged = ggml_add(ctx, merged, out_proj_b);

    return merged;
}

inline struct ggml_tensor* sam3_expand_token_attn_bias(
    struct ggml_context* ctx,
    struct ggml_tensor* token_bias,  // [T, 1, B] or nullptr
    int64_t n_q,
    int n_heads,
    int64_t batch) {
    if (!token_bias) {
        return nullptr;
    }

    const int64_t n_kv = token_bias->ne[0];
    auto* bias_4d = ggml_reshape_4d(ctx, token_bias, n_kv, 1, 1, batch);
    auto* full_bias = ggml_repeat(
        ctx,
        bias_4d,
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_q, n_heads, batch));

    // ggml_flash_attn_ext expects mask storage in fp16.
    return ggml_cont(ctx, ggml_cast(ctx, full_bias, GGML_TYPE_F16));
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_INTERNAL_MODELS_SAM3_OPS_HPP
