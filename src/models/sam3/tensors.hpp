#ifndef SAM_CPP_SRC_MODELS_SAM3_TENSORS_HPP
#define SAM_CPP_SRC_MODELS_SAM3_TENSORS_HPP

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
#include "ggml.h"
#include <cstdint>
#include <string>
#include <vector>

namespace sam::internal::sam3 {

inline void sam3_register_tensors(sam3_model& model, bool video = false) {
    const auto& hp = model.hparams;
    auto& tensors = model.tensors;
    auto ctx = model.ctx;

    auto T1 = [&](const std::string& name, int64_t d0) -> ggml_tensor* {
        auto* t = ggml_new_tensor_1d(ctx, model.source_type(name, GGML_TYPE_F32), d0);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    const ggml_type WTYPE = model.weight_type;
    const int64_t   WBLK  = ggml_blck_size(WTYPE);  // 1 for F32/F16, 32 for Q4/Q8

    auto T2 = [&](const std::string& name, int64_t d0, int64_t d1) -> ggml_tensor* {
        const ggml_type type = (d0 % WBLK == 0) ? WTYPE : GGML_TYPE_F32;
        auto* t = ggml_new_tensor_2d(ctx, model.source_type(name, type), d0, d1);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T4 = [&](const std::string& name, int64_t d0, int64_t d1, int64_t d2, int64_t d3) -> ggml_tensor* {
        const ggml_type type = (d0 % WBLK == 0) ? WTYPE : GGML_TYPE_F32;
        auto* t = ggml_new_tensor_4d(ctx, model.source_type(name, type), d0, d1, d2, d3);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T1f = T1;
    auto T2f = [&](const std::string& name, int64_t d0, int64_t d1) -> ggml_tensor* {
        auto* t = ggml_new_tensor_2d(ctx, model.source_type(name, GGML_TYPE_F32), d0, d1);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T3f = [&](const std::string& name, int64_t d0, int64_t d1, int64_t d2) -> ggml_tensor* {
        auto* t = ggml_new_tensor_3d(ctx, model.source_type(name, GGML_TYPE_F32), d0, d1, d2);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T4f = [&](const std::string& name, int64_t d0, int64_t d1, int64_t d2, int64_t d3) -> ggml_tensor* {
        auto* t = ggml_new_tensor_4d(ctx, model.source_type(name, GGML_TYPE_F32), d0, d1, d2, d3);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };

    const int E = hp.vit_embed_dim;      // 1024
    const int MD = hp.mem_out_dim;
    const int D = hp.neck_dim;           // 256
    const int TW = hp.text_width;        // 1024
    const int MLP = hp.vit_mlp_dim;      // 4736
    const int FFN = hp.fenc_ffn_dim;     // 2048
    const int NQ = hp.ddec_num_queries;  // 200

    // ── ViT backbone ─────────────────────────────────────────────────────
    model.vit.blocks.resize(hp.vit_depth);

    model.vit.patch_embed_w = T4("vit.patch_embed.proj.weight", hp.patch_size, hp.patch_size, 3, E);
    // pos_embed: The pretrained model stores [1, 24, 24, 1024] at pretrained resolution (no cls token).
    // Conversion script writes reversed dims → ggml [E, 24, 24, 1].
    // Tiled 3x at runtime to [E, 72, 72, 1].
    {
        const int pretrained_grid = hp.img_size / hp.patch_size / 3;  // 1008/14/3 = 24
        model.vit.pos_embed = T4f("vit.pos_embed", E, pretrained_grid, pretrained_grid, 1);
    }
    model.vit.ln_pre_w = T1f("vit.ln_pre.weight", E);
    model.vit.ln_pre_b = T1f("vit.ln_pre.bias", E);

    for (int i = 0; i < hp.vit_depth; ++i) {
        auto& blk = model.vit.blocks[i];
        auto p = "vit.blocks." + std::to_string(i);
        blk.norm1_w = T1f(p + ".norm1.weight", E);
        blk.norm1_b = T1f(p + ".norm1.bias", E);
        blk.qkv_w = T2(p + ".attn.qkv.weight", E, 3 * E);
        blk.qkv_b = T1f(p + ".attn.qkv.bias", 3 * E);
        blk.proj_w = T2(p + ".attn.proj.weight", E, E);
        blk.proj_b = T1f(p + ".attn.proj.bias", E);
        blk.norm2_w = T1f(p + ".norm2.weight", E);
        blk.norm2_b = T1f(p + ".norm2.bias", E);
        blk.mlp_fc1_w = T2(p + ".mlp.lin1.weight", E, MLP);
        blk.mlp_fc1_b = T1f(p + ".mlp.lin1.bias", MLP);
        blk.mlp_fc2_w = T2(p + ".mlp.lin2.weight", MLP, E);
        blk.mlp_fc2_b = T1f(p + ".mlp.lin2.bias", E);

        // RoPE freqs_cis: [N, 32, 2] where N=5184 for global, 576 for window
        int64_t rope_n = hp.is_global_attn(i) ? hp.n_img_tokens() : (hp.vit_window_size * hp.vit_window_size);
        blk.freqs_cis = T3f(p + ".attn.freqs_cis", 2, 32, rope_n);
    }

    // ── Neck (detector + tracker) ────────────────────────────────────────
    // ggml weight layout: conv2d [kW, kH, Cin, Cout], conv_transpose [kW, kH, Cout, Cin]
    auto register_neck = [&](sam3_neck& neck, const std::string& prefix) {
        // scale 0 (4x): ConvTranspose(E→512, k=2, s=2), GELU, ConvTranspose(512→D, k=2, s=2), Conv1x1(D→D), Conv3x3(D→D)
        neck.scales[0].deconv1_w = T4(prefix + "0.dconv_2x2_0.weight", 2, 2, 512, E);  // [kW, kH, Cout=512, Cin=E]
        neck.scales[0].deconv1_b = T1f(prefix + "0.dconv_2x2_0.bias", 512);
        neck.scales[0].deconv2_w = T4(prefix + "0.dconv_2x2_1.weight", 2, 2, D, 512);  // [kW, kH, Cout=D, Cin=512]
        neck.scales[0].deconv2_b = T1f(prefix + "0.dconv_2x2_1.bias", D);
        neck.scales[0].conv1x1_w = T4(prefix + "0.conv_1x1.weight", 1, 1, D, D);  // Conv2d(D→D)
        neck.scales[0].conv1x1_b = T1f(prefix + "0.conv_1x1.bias", D);
        neck.scales[0].conv3x3_w = T4(prefix + "0.conv_3x3.weight", 3, 3, D, D);  // Conv2d(D→D)
        neck.scales[0].conv3x3_b = T1f(prefix + "0.conv_3x3.bias", D);

        // scale 1 (2x): ConvTranspose(E→512, k=2, s=2), Conv1x1(512→D), Conv3x3(D→D)
        neck.scales[1].deconv1_w = T4(prefix + "1.dconv_2x2.weight", 2, 2, 512, E);  // ConvTranspose
        neck.scales[1].deconv1_b = T1f(prefix + "1.dconv_2x2.bias", 512);
        neck.scales[1].conv1x1_w = T4(prefix + "1.conv_1x1.weight", 1, 1, 512, D);  // Conv2d(512→D): Cin=512, Cout=D
        neck.scales[1].conv1x1_b = T1f(prefix + "1.conv_1x1.bias", D);
        neck.scales[1].conv3x3_w = T4(prefix + "1.conv_3x3.weight", 3, 3, D, D);
        neck.scales[1].conv3x3_b = T1f(prefix + "1.conv_3x3.bias", D);

        // scale 2 (1x): Conv1x1(E→D), Conv3x3(D→D)
        neck.scales[2].conv1x1_w = T4(prefix + "2.conv_1x1.weight", 1, 1, E, D);  // Conv2d(E→D): Cin=E, Cout=D
        neck.scales[2].conv1x1_b = T1f(prefix + "2.conv_1x1.bias", D);
        neck.scales[2].conv3x3_w = T4(prefix + "2.conv_3x3.weight", 3, 3, D, D);
        neck.scales[2].conv3x3_b = T1f(prefix + "2.conv_3x3.bias", D);

        // scale 3 (0.5x): MaxPool(k=2, s=2), Conv1x1(E→D), Conv3x3(D→D)
        neck.scales[3].conv1x1_w = T4(prefix + "3.conv_1x1.weight", 1, 1, E, D);
        neck.scales[3].conv1x1_b = T1f(prefix + "3.conv_1x1.bias", D);
        neck.scales[3].conv3x3_w = T4(prefix + "3.conv_3x3.weight", 3, 3, D, D);
        neck.scales[3].conv3x3_b = T1f(prefix + "3.conv_3x3.bias", D);
    };
    register_neck(model.neck_det, "neck.det.");
    if (video) register_neck(model.neck_trk, "neck.trk.");

    // Helper lambdas used by multiple sections (detector + tracker)
    auto reg = [&](const std::string& n, int64_t d0, int64_t d1, bool is_f32 = false) {
        const ggml_type rtype = (is_f32 || d0 % WBLK != 0) ? GGML_TYPE_F32 : WTYPE;
        auto* t = ggml_new_tensor_2d(ctx, model.source_type(n, rtype), d0, d1);
        ggml_set_name(t, n.c_str());
        tensors[n] = t;
        return t;
    };
    auto reg1 = [&](const std::string& n, int64_t d0) {
        auto* t = ggml_new_tensor_1d(ctx, model.source_type(n, GGML_TYPE_F32), d0);
        ggml_set_name(t, n.c_str());
        tensors[n] = t;
        return t;
    };
    auto reg4 = [&](const std::string& n, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
        const ggml_type rtype = (d0 % WBLK == 0) ? WTYPE : GGML_TYPE_F32;
        auto* t = ggml_new_tensor_4d(ctx, model.source_type(n, rtype), d0, d1, d2, d3);
        ggml_set_name(t, n.c_str());
        tensors[n] = t;
        return t;
    };

    // ── Detector-only tensors (skipped for visual-only models) ──────────

    // ── Text encoder ─────────────────────────────────────────────────────
    model.text_enc.blocks.resize(hp.text_layers);
    model.text_enc.token_embed_w = T2f("text.token_embed.weight", TW, hp.text_vocab_size);
    model.text_enc.pos_embed = T2f("text.pos_embed", TW, hp.text_ctx_len);
    model.text_enc.ln_final_w = T1f("text.ln_final.weight", TW);
    model.text_enc.ln_final_b = T1f("text.ln_final.bias", TW);
    model.text_enc.resizer_w = T2("text.resizer.weight", TW, hp.text_out_dim);
    model.text_enc.resizer_b = T1f("text.resizer.bias", hp.text_out_dim);
    // text.text_projection is intentionally not registered — the conversion
    // script skips it and the loader rejects unknown tensors. See struct comment.

    for (int i = 0; i < hp.text_layers; ++i) {
        auto& blk = model.text_enc.blocks[i];
        auto p = "text.blocks." + std::to_string(i);
        blk.attn_in_proj_w = T2(p + ".attn.in_proj.weight", TW, 3 * TW);
        blk.attn_in_proj_b = T1f(p + ".attn.in_proj.bias", 3 * TW);
        blk.attn_out_proj_w = T2(p + ".attn.out_proj.weight", TW, TW);
        blk.attn_out_proj_b = T1f(p + ".attn.out_proj.bias", TW);
        blk.ln1_w = T1f(p + ".ln_1.weight", TW);
        blk.ln1_b = T1f(p + ".ln_1.bias", TW);
        blk.ln2_w = T1f(p + ".ln_2.weight", TW);
        blk.ln2_b = T1f(p + ".ln_2.bias", TW);
        blk.mlp_fc1_w = T2(p + ".mlp.fc1.weight", TW, TW * 4);
        blk.mlp_fc1_b = T1f(p + ".mlp.fc1.bias", TW * 4);
        blk.mlp_fc2_w = T2(p + ".mlp.fc2.weight", TW * 4, TW);
        blk.mlp_fc2_b = T1f(p + ".mlp.fc2.bias", TW);
    }

    // ── Fusion encoder ───────────────────────────────────────────────────
    model.fenc.layers.resize(hp.fenc_layers);
    for (int i = 0; i < hp.fenc_layers; ++i) {
        auto& ly = model.fenc.layers[i];
        auto p = "fenc.layers." + std::to_string(i);
        // self-attention
        ly.sa_in_proj_w = T2(p + ".sa.in_proj_weight", D, 3 * D);
        ly.sa_in_proj_b = T1f(p + ".sa.in_proj_bias", 3 * D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);
        // cross-attention
        ly.ca_q_w = T2(p + ".ca.in_proj_weight", D, 3 * D);
        ly.ca_q_b = T1f(p + ".ca.in_proj_bias", 3 * D);
        ly.ca_kv_w = nullptr;  // fused in_proj for MHA
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);
        // FFN
        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm3_w = T1f(p + ".norm3.weight", D);
        ly.norm3_b = T1f(p + ".norm3.bias", D);
    }

    // ── DETR decoder ─────────────────────────────────────────────────────
    model.ddec.layers.resize(hp.ddec_layers);
    model.ddec.query_embed = T2f("ddec.query_embed.weight", D, NQ);
    model.ddec.presence_token = T2f("ddec.presence_token.weight", D, 1);

    // Reference points, norms, bbox embed, ref_point_head, boxRPB, presence_head
    // These use the exact checkpoint names after renaming
    tensors["ddec.reference_points.weight"] = ggml_new_tensor_2d(ctx, model.source_type("ddec.reference_points.weight", GGML_TYPE_F32), 4, NQ);
    ggml_set_name(tensors["ddec.reference_points.weight"], "ddec.reference_points.weight");
    tensors["ddec.norm.weight"] = ggml_new_tensor_1d(ctx, model.source_type("ddec.norm.weight", GGML_TYPE_F32), D);
    ggml_set_name(tensors["ddec.norm.weight"], "ddec.norm.weight");
    tensors["ddec.norm.bias"] = ggml_new_tensor_1d(ctx, model.source_type("ddec.norm.bias", GGML_TYPE_F32), D);
    ggml_set_name(tensors["ddec.norm.bias"], "ddec.norm.bias");

    // bbox_embed MLP (3 layers: 256→256→256→4)
    for (int j = 0; j < 3; ++j) {
        int out = (j == 2) ? 4 : D;
        auto bp = "ddec.bbox_embed.layers." + std::to_string(j);
        tensors[bp + ".weight"] = ggml_new_tensor_2d(ctx, model.source_type(bp + ".weight", WTYPE), D, out);
        ggml_set_name(tensors[bp + ".weight"], (bp + ".weight").c_str());
        tensors[bp + ".bias"] = ggml_new_tensor_1d(ctx, model.source_type(bp + ".bias", GGML_TYPE_F32), out);
        ggml_set_name(tensors[bp + ".bias"], (bp + ".bias").c_str());
    }

    // ref_point_head MLP (2 layers: 512→256→256)
    tensors["ddec.ref_point_head.layers.0.weight"] = ggml_new_tensor_2d(ctx, model.source_type("ddec.ref_point_head.layers.0.weight", WTYPE), 512, D);
    tensors["ddec.ref_point_head.layers.0.bias"] = ggml_new_tensor_1d(ctx, model.source_type("ddec.ref_point_head.layers.0.bias", GGML_TYPE_F32), D);
    tensors["ddec.ref_point_head.layers.1.weight"] = ggml_new_tensor_2d(ctx, model.source_type("ddec.ref_point_head.layers.1.weight", WTYPE), D, D);
    tensors["ddec.ref_point_head.layers.1.bias"] = ggml_new_tensor_1d(ctx, model.source_type("ddec.ref_point_head.layers.1.bias", GGML_TYPE_F32), D);
    for (auto& kv : std::vector<std::string>{
             "ddec.ref_point_head.layers.0.weight", "ddec.ref_point_head.layers.0.bias",
             "ddec.ref_point_head.layers.1.weight", "ddec.ref_point_head.layers.1.bias"})
        ggml_set_name(tensors[kv], kv.c_str());

    // boxRPB MLPs (x and y, each 2 layers)
    for (const auto& axis : {"x", "y"}) {
        auto bp = std::string("ddec.boxRPB_embed_") + axis;
        tensors[bp + ".layers.0.weight"] = ggml_new_tensor_2d(ctx, model.source_type(bp + ".layers.0.weight", (2 % WBLK == 0) ? WTYPE : GGML_TYPE_F32), 2, D);
        tensors[bp + ".layers.0.bias"] = ggml_new_tensor_1d(ctx, model.source_type(bp + ".layers.0.bias", GGML_TYPE_F32), D);
        tensors[bp + ".layers.1.weight"] = ggml_new_tensor_2d(ctx, model.source_type(bp + ".layers.1.weight", WTYPE), D, hp.ddec_heads);
        tensors[bp + ".layers.1.bias"] = ggml_new_tensor_1d(ctx, model.source_type(bp + ".layers.1.bias", GGML_TYPE_F32), hp.ddec_heads);
        for (int j = 0; j < 2; ++j) {
            auto l = bp + ".layers." + std::to_string(j);
            ggml_set_name(tensors[l + ".weight"], (l + ".weight").c_str());
            ggml_set_name(tensors[l + ".bias"], (l + ".bias").c_str());
        }
    }

    // presence_token_head MLP (3 layers: 256→256→256→1)
    for (int j = 0; j < 3; ++j) {
        int out = (j == 2) ? 1 : D;
        auto bp = "ddec.presence_token_head.layers." + std::to_string(j);
        tensors[bp + ".weight"] = ggml_new_tensor_2d(ctx, model.source_type(bp + ".weight", WTYPE), D, out);
        ggml_set_name(tensors[bp + ".weight"], (bp + ".weight").c_str());
        tensors[bp + ".bias"] = ggml_new_tensor_1d(ctx, model.source_type(bp + ".bias", GGML_TYPE_F32), out);
        ggml_set_name(tensors[bp + ".bias"], (bp + ".bias").c_str());
    }
    tensors["ddec.presence_token_out_norm.weight"] = ggml_new_tensor_1d(ctx, model.source_type("ddec.presence_token_out_norm.weight", GGML_TYPE_F32), D);
    tensors["ddec.presence_token_out_norm.bias"] = ggml_new_tensor_1d(ctx, model.source_type("ddec.presence_token_out_norm.bias", GGML_TYPE_F32), D);
    ggml_set_name(tensors["ddec.presence_token_out_norm.weight"], "ddec.presence_token_out_norm.weight");
    ggml_set_name(tensors["ddec.presence_token_out_norm.bias"], "ddec.presence_token_out_norm.bias");

    // DETR decoder layers
    for (int i = 0; i < hp.ddec_layers; ++i) {
        auto& ly = model.ddec.layers[i];
        auto p = "ddec.layers." + std::to_string(i);
        ly.sa_in_proj_w = T2(p + ".sa.in_proj_weight", D, 3 * D);
        ly.sa_in_proj_b = T1f(p + ".sa.in_proj_bias", 3 * D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);

        ly.ca_q_w = T2(p + ".ca.in_proj_weight", D, 3 * D);
        ly.ca_q_b = T1f(p + ".ca.in_proj_bias", 3 * D);
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);

        ly.ca_text_q_w = T2(p + ".ca_text.in_proj_weight", D, 3 * D);
        ly.ca_text_q_b = T1f(p + ".ca_text.in_proj_bias", 3 * D);
        ly.ca_text_out_w = T2(p + ".ca_text.out_proj.weight", D, D);
        ly.ca_text_out_b = T1f(p + ".ca_text.out_proj.bias", D);
        ly.norm3_w = T1f(p + ".norm_ca_text.weight", D);
        ly.norm3_b = T1f(p + ".norm_ca_text.bias", D);

        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm4_w = T1f(p + ".norm3.weight", D);
        ly.norm4_b = T1f(p + ".norm3.bias", D);
    }

    // ── DotProductScoring ────────────────────────────────────────────────
    reg("scoring.prompt_proj.weight", D, D);
    reg1("scoring.prompt_proj.bias", D);
    reg("scoring.hs_proj.weight", D, D);
    reg1("scoring.hs_proj.bias", D);
    reg("scoring.prompt_mlp.layers.0.weight", D, FFN);
    reg1("scoring.prompt_mlp.layers.0.bias", FFN);
    reg("scoring.prompt_mlp.layers.1.weight", FFN, D);
    reg1("scoring.prompt_mlp.layers.1.bias", D);
    reg1("scoring.prompt_mlp.out_norm.weight", D);
    reg1("scoring.prompt_mlp.out_norm.bias", D);

    // ── Geometry encoder ───────────────────────────────────────────────────
    model.geom_enc.layers.resize(hp.geom_layers);

    // Direct projections
    model.geom_enc.point_proj_w = T2("geom.points_direct_project.weight", 2, D);
    model.geom_enc.point_proj_b = T1f("geom.points_direct_project.bias", D);
    model.geom_enc.box_proj_w = T2("geom.boxes_direct_project.weight", 4, D);
    model.geom_enc.box_proj_b = T1f("geom.boxes_direct_project.bias", D);
    // Pooling projections
    model.geom_enc.point_pool_proj_w = T2("geom.points_pool_project.weight", D, D);
    model.geom_enc.point_pool_proj_b = T1f("geom.points_pool_project.bias", D);
    model.geom_enc.box_pool_proj_w = T4("geom.boxes_pool_project.weight", 7, 7, D, D);
    model.geom_enc.box_pool_proj_b = T1f("geom.boxes_pool_project.bias", D);
    // Positional encoding projections
    model.geom_enc.point_pos_proj_w = T2("geom.points_pos_enc_project.weight", D, D);
    model.geom_enc.point_pos_proj_b = T1f("geom.points_pos_enc_project.bias", D);
    model.geom_enc.box_pos_proj_w = T2("geom.boxes_pos_enc_project.weight", 258, D);
    model.geom_enc.box_pos_proj_b = T1f("geom.boxes_pos_enc_project.bias", D);
    // Label and CLS
    model.geom_enc.type_embed = T2f("geom.label_embed.weight", D, 2);
    model.geom_enc.cls_token = T2f("geom.cls_embed.weight", D, 1);
    // Final projection + norms
    model.geom_enc.post_proj_w = T2("geom.final_proj.weight", D, D);
    model.geom_enc.post_proj_b = T1f("geom.final_proj.bias", D);
    model.geom_enc.norm_w = T1f("geom.norm.weight", D);
    model.geom_enc.norm_b = T1f("geom.norm.bias", D);
    model.geom_enc.encode_norm_w = T1f("geom.encode_norm.weight", D);
    model.geom_enc.encode_norm_b = T1f("geom.encode_norm.bias", D);
    model.geom_enc.img_pre_norm_w = T1f("geom.img_pre_norm.weight", D);
    model.geom_enc.img_pre_norm_b = T1f("geom.img_pre_norm.bias", D);

    for (int i = 0; i < hp.geom_layers; ++i) {
        auto& ly = model.geom_enc.layers[i];
        auto p = "geom.layers." + std::to_string(i);
        ly.sa_in_proj_w = T2(p + ".sa.in_proj_weight", D, 3 * D);
        ly.sa_in_proj_b = T1f(p + ".sa.in_proj_bias", 3 * D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);
        ly.ca_q_w = T2(p + ".ca.in_proj_weight", D, 3 * D);
        ly.ca_q_b = T1f(p + ".ca.in_proj_bias", 3 * D);
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);
        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm3_w = T1f(p + ".norm3.weight", D);
        ly.norm3_b = T1f(p + ".norm3.bias", D);
    }

    // ── Segmentation head ────────────────────────────────────────────────
    // Pixel decoder (3 conv layers + norms)
    for (int i = 0; i < 3; ++i) {
        auto si = std::to_string(i);
        model.seg_head.up_conv_w[i] = T4("seg.pixel_decoder.conv_layers." + si + ".weight", 3, 3, D, D);
        model.seg_head.up_conv_b[i] = T1f("seg.pixel_decoder.conv_layers." + si + ".bias", D);
        model.seg_head.up_norm_w[i] = T1f("seg.pixel_decoder.norms." + si + ".weight", D);
        model.seg_head.up_norm_b[i] = T1f("seg.pixel_decoder.norms." + si + ".bias", D);
    }

    // Mask predictor (3-layer MLP: 256→256→256→256)
    for (int j = 0; j < 3; ++j) {
        auto bp = "seg.mask_predictor.mask_embed.layers." + std::to_string(j);
        model.seg_head.mask_embed_w = T2(bp + ".weight", D, D);  // overwritten but last one
        model.seg_head.mask_embed_b = T1f(bp + ".bias", D);
    }
    // Re-register properly: all 3 layers with unique names are already in tensors map
    // The struct only has one pointer — use the tensors map at runtime
    // For now, just ensure all 6 tensors are registered (they are via the loop above —
    // each T2/T1f call registers under unique names)

    // Cross-attention to prompt
    model.seg_head.ca_prompt_q_w = T2("seg.cross_attend_prompt.in_proj_weight", D, 3 * D);
    model.seg_head.ca_prompt_q_b = T1f("seg.cross_attend_prompt.in_proj_bias", 3 * D);
    model.seg_head.ca_prompt_out_w = T2("seg.cross_attend_prompt.out_proj.weight", D, D);
    model.seg_head.ca_prompt_out_b = T1f("seg.cross_attend_prompt.out_proj.bias", D);

    // Cross-attn norm
    reg1("seg.cross_attn_norm.weight", D);
    reg1("seg.cross_attn_norm.bias", D);

    // Instance and semantic seg heads (Conv 1x1)
    reg4("seg.instance_seg_head.weight", 1, 1, D, D);
    reg1("seg.instance_seg_head.bias", D);
    reg4("seg.semantic_seg_head.weight", 1, 1, D, 1);
    reg1("seg.semantic_seg_head.bias", 1);

    if (video) {
    // ── SAM prompt encoder ───────────────────────────────────────────────
    model.sam_pe.pe_gaussian = T2f("sam_pe.pe_gaussian", 128, 2);
    for (int i = 0; i < 4; ++i)
        model.sam_pe.point_embed[i] = T2f("sam_pe.point_embeddings." + std::to_string(i) + ".weight", D, 1);
    model.sam_pe.not_a_point_embed = T2f("sam_pe.not_a_point_embed.weight", D, 1);
    model.sam_pe.no_mask_embed = T2f("sam_pe.no_mask_embed.weight", D, 1);

    // mask_downscaling: sequential with numeric indices
    model.sam_pe.mask_ds_conv_w[0] = T4("sam_pe.mask_ds.0.weight", 2, 2, 1, 4);
    model.sam_pe.mask_ds_conv_b[0] = T1f("sam_pe.mask_ds.0.bias", 4);
    model.sam_pe.mask_ds_norm_w[0] = T1f("sam_pe.mask_ds.1.weight", 4);
    model.sam_pe.mask_ds_norm_b[0] = T1f("sam_pe.mask_ds.1.bias", 4);
    model.sam_pe.mask_ds_conv_w[1] = T4("sam_pe.mask_ds.3.weight", 2, 2, 4, 16);
    model.sam_pe.mask_ds_conv_b[1] = T1f("sam_pe.mask_ds.3.bias", 16);
    model.sam_pe.mask_ds_norm_w[1] = T1f("sam_pe.mask_ds.4.weight", 16);
    model.sam_pe.mask_ds_norm_b[1] = T1f("sam_pe.mask_ds.4.bias", 16);
    model.sam_pe.mask_ds_conv_w[2] = T4("sam_pe.mask_ds.6.weight", 1, 1, 16, D);
    model.sam_pe.mask_ds_conv_b[2] = T1f("sam_pe.mask_ds.6.bias", D);

    // ── SAM mask decoder ─────────────────────────────────────────────────
    model.sam_dec.iou_token = T2f("sam_dec.iou_token.weight", D, 1);
    model.sam_dec.mask_tokens = T2f("sam_dec.mask_tokens.weight", D, 4);
    model.sam_dec.obj_score_token = T2f("sam_dec.obj_score_token.weight", D, 1);

    model.sam_dec.twoway_blocks.resize(hp.sam_dec_depth);
    for (int i = 0; i < hp.sam_dec_depth; ++i) {
        auto& blk = model.sam_dec.twoway_blocks[i];
        auto p = "sam_dec.twoway." + std::to_string(i);

        auto reg_attn = [&](sam3_sam_attn& a, const std::string& pfx, int in_dim, int out_dim) {
            a.q_w = T2(pfx + ".q_proj.weight", in_dim, out_dim);
            a.q_b = T1f(pfx + ".q_proj.bias", out_dim);
            a.k_w = T2(pfx + ".k_proj.weight", in_dim, out_dim);
            a.k_b = T1f(pfx + ".k_proj.bias", out_dim);
            a.v_w = T2(pfx + ".v_proj.weight", in_dim, out_dim);
            a.v_b = T1f(pfx + ".v_proj.bias", out_dim);
            a.out_w = T2(pfx + ".out_proj.weight", out_dim, in_dim);
            a.out_b = T1f(pfx + ".out_proj.bias", in_dim);
        };

        reg_attn(blk.self_attn, p + ".sa", D, D);
        reg_attn(blk.ca_tok2img, p + ".cross_attn_token_to_image", D, 128);
        reg_attn(blk.ca_img2tok, p + ".cross_attn_image_to_token", D, 128);

        blk.norm1_w = T1f(p + ".norm1.weight", D);
        blk.norm1_b = T1f(p + ".norm1.bias", D);
        blk.norm2_w = T1f(p + ".norm2.weight", D);
        blk.norm2_b = T1f(p + ".norm2.bias", D);
        blk.norm3_w = T1f(p + ".norm3.weight", D);
        blk.norm3_b = T1f(p + ".norm3.bias", D);
        blk.norm4_w = T1f(p + ".norm4.weight", D);
        blk.norm4_b = T1f(p + ".norm4.bias", D);

        blk.mlp_fc1_w = T2(p + ".mlp.lin1.weight", D, FFN);
        blk.mlp_fc1_b = T1f(p + ".mlp.lin1.bias", FFN);
        blk.mlp_fc2_w = T2(p + ".mlp.lin2.weight", FFN, D);
        blk.mlp_fc2_b = T1f(p + ".mlp.lin2.bias", D);
    }

    // final attention
    auto reg_sam_attn = [&](sam3_sam_attn& a, const std::string& pfx, int in_dim, int out_dim) {
        a.q_w = T2(pfx + ".q_proj.weight", in_dim, out_dim);
        a.q_b = T1f(pfx + ".q_proj.bias", out_dim);
        a.k_w = T2(pfx + ".k_proj.weight", in_dim, out_dim);
        a.k_b = T1f(pfx + ".k_proj.bias", out_dim);
        a.v_w = T2(pfx + ".v_proj.weight", in_dim, out_dim);
        a.v_b = T1f(pfx + ".v_proj.bias", out_dim);
        a.out_w = T2(pfx + ".out_proj.weight", out_dim, in_dim);
        a.out_b = T1f(pfx + ".out_proj.bias", in_dim);
    };
    reg_sam_attn(model.sam_dec.final_attn, "sam_dec.final_attn", D, 128);
    model.sam_dec.final_norm_w = T1f("sam_dec.final_norm.weight", D);
    model.sam_dec.final_norm_b = T1f("sam_dec.final_norm.bias", D);

    // upscaling
    model.sam_dec.up1_w = T4("sam_dec.upscale.0.weight", 2, 2, 64, D);
    model.sam_dec.up1_b = T1f("sam_dec.upscale.0.bias", 64);
    model.sam_dec.up1_norm_w = T1f("sam_dec.upscale.1.weight", 64);
    model.sam_dec.up1_norm_b = T1f("sam_dec.upscale.1.bias", 64);
    model.sam_dec.up2_w = T4("sam_dec.upscale.3.weight", 2, 2, 32, 64);
    model.sam_dec.up2_b = T1f("sam_dec.upscale.3.bias", 32);

    // high-res feature convolutions
    model.sam_dec.conv_s0_w = T4("sam_dec.conv_s0.weight", 1, 1, D, 32);
    model.sam_dec.conv_s0_b = T1f("sam_dec.conv_s0.bias", 32);
    model.sam_dec.conv_s1_w = T4("sam_dec.conv_s1.weight", 1, 1, D, 64);
    model.sam_dec.conv_s1_b = T1f("sam_dec.conv_s1.bias", 64);

    // hypernetwork MLPs (4 × 3 layers: 256→256→256→32)
    for (int m = 0; m < 4; ++m) {
        for (int j = 0; j < 3; ++j) {
            int in_d = D, out_d = (j == 2) ? 32 : D;
            auto bp = "sam_dec.hyper." + std::to_string(m) + ".layers." + std::to_string(j);
            model.sam_dec.hyper_w[m][j] = T2(bp + ".weight", in_d, out_d);
            model.sam_dec.hyper_b[m][j] = T1f(bp + ".bias", out_d);
        }
    }

    // IoU prediction head (3 layers: 256→256→256→4)
    for (int j = 0; j < 3; ++j) {
        int out_d = (j == 2) ? 4 : D;
        auto bp = "sam_dec.iou_prediction_head.layers." + std::to_string(j);
        model.sam_dec.iou_head_w[j] = T2(bp + ".weight", D, out_d);
        model.sam_dec.iou_head_b[j] = T1f(bp + ".bias", out_d);
    }

    // object score head (3 layers: 256→256→256→1)
    for (int j = 0; j < 3; ++j) {
        int out_d = (j == 2) ? 1 : D;
        auto bp = "sam_dec.pred_obj_score_head.layers." + std::to_string(j);
        model.sam_dec.obj_head_w[j] = T2(bp + ".weight", D, out_d);
        model.sam_dec.obj_head_b[j] = T1f(bp + ".bias", out_d);
    }

    // ── Memory encoder ───────────────────────────────────────────────────
    // mask_downsampler: sequential encoder.{0,1,3,4,6,7,9,10,12}
    int ds_channels[] = {1, 4, 16, 64, 256};
    int ds_indices[] = {0, 3, 6, 9, 12};
    int norm_indices[] = {1, 4, 7, 10};
    for (int s = 0; s < 4; ++s) {
        auto si = std::to_string(ds_indices[s]);
        model.mem_enc.ds_conv_w[s] = T4("mem_enc.ds." + si + ".weight", 3, 3, ds_channels[s], ds_channels[s + 1]);
        model.mem_enc.ds_conv_b[s] = T1f("mem_enc.ds." + si + ".bias", ds_channels[s + 1]);
        auto ni = std::to_string(norm_indices[s]);
        model.mem_enc.ds_norm_w[s] = T1f("mem_enc.ds." + ni + ".weight", ds_channels[s + 1]);
        model.mem_enc.ds_norm_b[s] = T1f("mem_enc.ds." + ni + ".bias", ds_channels[s + 1]);
    }
    model.mem_enc.ds_conv_w[4] = T4("mem_enc.ds.12.weight", 1, 1, D, D);
    model.mem_enc.ds_conv_b[4] = T1f("mem_enc.ds.12.bias", D);

    model.mem_enc.pix_proj_w = T4("mem_enc.pix_feat_proj.weight", 1, 1, D, D);
    model.mem_enc.pix_proj_b = T1f("mem_enc.pix_feat_proj.bias", D);

    // fuser CXBlocks
    for (int i = 0; i < 2; ++i) {
        auto p = "mem_enc.fuser." + std::to_string(i);
        model.mem_enc.fuser_dw_w[i] = T4(p + ".dwconv.weight", 7, 7, 1, D);  // groups=256
        model.mem_enc.fuser_dw_b[i] = T1f(p + ".dwconv.bias", D);
        model.mem_enc.fuser_norm_w[i] = T1f(p + ".norm.weight", D);
        model.mem_enc.fuser_norm_b[i] = T1f(p + ".norm.bias", D);
        model.mem_enc.fuser_fc1_w[i] = T2(p + ".pwconv1.weight", D, 1024);
        model.mem_enc.fuser_fc1_b[i] = T1f(p + ".pwconv1.bias", 1024);
        model.mem_enc.fuser_fc2_w[i] = T2(p + ".pwconv2.weight", 1024, D);
        model.mem_enc.fuser_fc2_b[i] = T1f(p + ".pwconv2.bias", D);
        model.mem_enc.fuser_gamma[i] = T1f(p + ".gamma", D);
    }

    model.mem_enc.out_proj_w = T4("mem_enc.out_proj.weight", 1, 1, D, MD);
    model.mem_enc.out_proj_b = T1f("mem_enc.out_proj.bias", MD);

    // temporal pos encodings
    model.mem_enc.tpos[0] = T4f("mem_enc.tpos_enc", MD, 1, 1, hp.num_maskmem);

    // ── Memory attention ─────────────────────────────────────────────────
    model.mem_attn.layers.resize(hp.mem_attn_layers);
    model.mem_attn_norm_w = reg1("mem_attn.norm.weight", D);
    model.mem_attn_norm_b = reg1("mem_attn.norm.bias", D);

    for (int i = 0; i < hp.mem_attn_layers; ++i) {
        auto& ly = model.mem_attn.layers[i];
        auto p = "mem_attn.layers." + std::to_string(i);
        // self-attention (RoPE, 1 head, 256-dim)
        ly.sa_q_w = T2(p + ".sa.q_proj.weight", D, D);
        ly.sa_q_b = T1f(p + ".sa.q_proj.bias", D);
        ly.sa_k_w = T2(p + ".sa.k_proj.weight", D, D);
        ly.sa_k_b = T1f(p + ".sa.k_proj.bias", D);
        ly.sa_v_w = T2(p + ".sa.v_proj.weight", D, D);
        ly.sa_v_b = T1f(p + ".sa.v_proj.bias", D);
        ly.sa_out_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);
        // cross-attention (kv_in_dim=64) — renamed from cross_attn_image → ca
        ly.ca_q_w = T2(p + ".ca.q_proj.weight", D, D);
        ly.ca_q_b = T1f(p + ".ca.q_proj.bias", D);
        ly.ca_k_w = T2(p + ".ca.k_proj.weight", MD, D);
        ly.ca_k_b = T1f(p + ".ca.k_proj.bias", D);
        ly.ca_v_w = T2(p + ".ca.v_proj.weight", MD, D);
        ly.ca_v_b = T1f(p + ".ca.v_proj.bias", D);
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);
        // FFN
        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm3_w = T1f(p + ".norm3.weight", D);
        ly.norm3_b = T1f(p + ".norm3.bias", D);
    }

    // ── Object pointer projection ────────────────────────────────────────
    for (int j = 0; j < 3; ++j) {
        auto bp = "obj_ptr_proj.layers." + std::to_string(j);
        model.obj_ptr_proj_w[j] = T2(bp + ".weight", D, D);
        model.obj_ptr_proj_b[j] = T1f(bp + ".bias", D);
    }
    model.no_obj_ptr = T2f("no_obj_ptr", D, 1);
    model.obj_ptr_tpos_w = T2("obj_ptr_tpos_proj.weight", D, MD);
    model.obj_ptr_tpos_b = T1f("obj_ptr_tpos_proj.bias", MD);

    // standalone tracker parameters
    model.no_mem_embed         = T3f("no_mem_embed", D, 1, 1);
    model.no_mem_pos_enc       = T3f("no_mem_pos_enc", D, 1, 1);
    model.no_obj_embed_spatial = T2f("no_obj_embed_spatial", MD, 1);
    T4f("trk_mask_ds.weight", 4, 4, 1, 1);
    T1f("trk_mask_ds.bias", 1);
    }
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_TENSORS_HPP
