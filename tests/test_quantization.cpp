#include "sam/internal/models/sam3/weights.hpp"
#include "sam/internal/models/sam3/tensors.hpp"
#include "sam/internal/runtime/ggml.hpp"
#include "sam/internal/runtime/ggml/backends/cpu.hpp"
#include "sam/internal/runtime/ggml/backends/metal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
void rejects(Function function, const char* message) {
    try { function(); }
    catch (const std::runtime_error&) { return; }
    throw std::runtime_error(message);
}

std::vector<float> matrix_values(std::int64_t width, std::int64_t rows, int salt) {
    std::vector<float> values(static_cast<std::size_t>(width * rows));
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t column = 0; column < width; ++column) {
            const auto index = static_cast<std::size_t>(row * width + column);
            values[index] = row % 127 == 0 ? 0.0f :
                static_cast<float>((row * 17 + column * 13 + salt * 7) % 101 - 50) * 0.004f;
        }
    }
    return values;
}

std::vector<char> encode_rows(ggml_type type, std::int64_t width, std::int64_t rows,
                              const std::vector<float>& values) {
    const auto* traits = ggml_get_type_traits(type);
    require(traits && traits->is_quantized && traits->from_float_ref,
            "pinned GGML lacks a reference quantizer for a required profile type");
    require(values.size() == static_cast<std::size_t>(width * rows) && width % traits->blck_size == 0,
            "invalid arithmetic fixture shape");
    const auto row_bytes = static_cast<std::size_t>(width / traits->blck_size) * traits->type_size;
    std::vector<char> bytes(static_cast<std::size_t>(rows) * row_bytes);
    for (std::int64_t row = 0; row < rows; ++row)
        traits->from_float_ref(values.data() + row * width, bytes.data() + row * row_bytes, width);
    return bytes;
}

std::vector<float> decode_rows(ggml_type type, std::int64_t width, std::int64_t rows,
                               const std::vector<char>& bytes) {
    const auto* traits = ggml_get_type_traits(type);
    const auto row_bytes = static_cast<std::size_t>(width / traits->blck_size) * traits->type_size;
    std::vector<float> values(static_cast<std::size_t>(width * rows));
    for (std::int64_t row = 0; row < rows; ++row)
        traits->to_float(bytes.data() + row * row_bytes, values.data() + row * width, width);
    return values;
}

void check_quantized_payload_validation() {
    const std::array<ggml_type, 4> types{
        GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K};
    for (const auto type : types) {
        const auto* traits = ggml_get_type_traits(type);
        const auto width = traits->blck_size;
        std::vector<float> source(static_cast<std::size_t>(width));
        for (std::int64_t i = 0; i < width; ++i) source[i] = static_cast<float>(i - width / 2) / width;
        auto block = encode_rows(type, width, 1, source);
        sam::internal::sam3::validate_quantized_payload(type, {width, 1}, block.data(), block.size());

        // Q6_K stores its half scale at the end of the packed block. A value in
        // its quantized prefix that resembles a half infinity is still valid.
        if (type == GGML_TYPE_Q6_K) {
            const ggml_fp16_t infinity = 0x7c00;
            std::memcpy(block.data(), &infinity, sizeof(infinity));
            sam::internal::sam3::validate_quantized_payload(type, {width, 1}, block.data(), block.size());
        }

        const auto scale_offset = type == GGML_TYPE_Q6_K ? block.size() - sizeof(ggml_fp16_t) : 0;
        const ggml_fp16_t infinity = 0x7c00;
        std::memcpy(block.data() + scale_offset, &infinity, sizeof(infinity));
        rejects([&] {
            sam::internal::sam3::validate_quantized_payload(type, {width, 1}, block.data(), block.size());
        }, "non-finite packed block scale was accepted");

        if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            block = encode_rows(type, width, 1, source);
            std::memcpy(block.data() + sizeof(ggml_fp16_t), &infinity, sizeof(infinity));
            rejects([&] {
                sam::internal::sam3::validate_quantized_payload(type, {width, 1}, block.data(), block.size());
            }, "non-finite packed block minimum scale was accepted");
        }
    }
}

void check_profile_policy() {
    const auto* q8 = sam::internal::sam3::image_quantization_profile("image-linear-q8_0-v1");
    const auto* q6 = sam::internal::sam3::image_quantization_profile("image-linear-q6_k-v1");
    const auto* q5 = sam::internal::sam3::image_quantization_profile("image-linear-q5_k-v1");
    const auto* q4 = sam::internal::sam3::image_quantization_profile("image-linear-q4_k-v1");
    const auto* vision_q8 = sam::internal::sam3::image_quantization_profile("image-vision-linear-q8_0-v1");
    const auto* vision_q6 = sam::internal::sam3::image_quantization_profile("image-vision-linear-q6_k-v1");
    const auto* vision_q5 = sam::internal::sam3::image_quantization_profile("image-vision-linear-q5_k-v1");
    const auto* vision_q4 = sam::internal::sam3::image_quantization_profile("image-vision-linear-q4_k-v1");
    require(q8 && q6 && q5 && q4 && vision_q8 && vision_q6 && vision_q5 && vision_q4 &&
            q8->quantize_text_linear && q6->quantize_text_linear &&
            q5->quantize_text_linear && q4->quantize_text_linear &&
            !vision_q8->quantize_text_linear && !vision_q6->quantize_text_linear &&
            !vision_q5->quantize_text_linear && !vision_q4->quantize_text_linear &&
            !sam::internal::sam3::image_quantization_profile("q4_k"),
            "quantization profile tags are not exact and versioned");
    const std::vector<std::int64_t> k_row{4736, 1024};
    const std::vector<std::int64_t> regular_row{1024, 3072};
    require(sam::internal::sam3::image_quantized_tensor_type("vit.blocks.0.mlp.lin2.weight", k_row, *q6) == GGML_TYPE_Q8_0 &&
            sam::internal::sam3::image_quantized_tensor_type("vit.blocks.31.mlp.lin2.weight", k_row, *q4) == GGML_TYPE_Q8_0,
            "K profile did not assign Q8_0 to non-256-wide SAM rows");
    require(sam::internal::sam3::image_quantized_tensor_type("vit.blocks.0.mlp.lin2.weight", {4864, 1024}, *q6) == GGML_TYPE_Q6_K,
            "K profile used the Q8 fallback outside its exact 4736-wide inventory");
    rejects([&] {
        (void)sam::internal::sam3::image_quantized_tensor_type("vit.blocks.0.attn.qkv.weight", {4736, 3072}, *q6);
    }, "K profile used a generic non-block-aligned fallback");
    require(sam::internal::sam3::image_quantized_tensor_type("vit.blocks.0.attn.qkv.weight", regular_row, *q6) == GGML_TYPE_Q6_K &&
            sam::internal::sam3::image_quantized_tensor_type("text.blocks.23.mlp.fc2.weight", regular_row, *q5) == GGML_TYPE_Q5_K &&
            sam::internal::sam3::image_quantized_tensor_type("vit.blocks.0.attn.qkv.weight", regular_row, *q8) == GGML_TYPE_Q8_0,
            "profile did not select the requested packed type for eligible SAM matrices");
    require(sam::internal::sam3::image_quantized_tensor_type("text.blocks.0.attn.in_proj.weight", regular_row, *q8) == GGML_TYPE_Q8_0 &&
            sam::internal::sam3::image_quantized_tensor_type("text.blocks.0.attn.in_proj.weight", regular_row, *vision_q8) == GGML_TYPE_F32 &&
            sam::internal::sam3::image_quantized_tensor_type("text.blocks.23.mlp.fc2.weight", regular_row, *vision_q6) == GGML_TYPE_F32,
            "vision-only profile did not preserve the text encoder as F32");

    auto context = sam::internal::make_context(4096);
    sam::internal::sam3::sam3_model model;
    model.ctx = context.get();
    model.weight_type = GGML_TYPE_F32;
    sam::internal::sam3::sam3_register_tensors(model);
    int vision_quantized = 0, vision_q8_fallbacks = 0, text_quantized = 0, legacy_quantized = 0;
    for (const auto& entry : model.tensors) {
        const auto& name = entry.first;
        const auto* tensor = entry.second;
        const std::vector<std::int64_t> shape{tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]};
        const auto vision_type = sam::internal::sam3::image_quantized_tensor_type(name, shape, *vision_q6);
        const auto legacy_type = sam::internal::sam3::image_quantized_tensor_type(name, shape, *q6);
        if (vision_type != GGML_TYPE_F32) ++vision_quantized;
        if (vision_type == GGML_TYPE_Q8_0) ++vision_q8_fallbacks;
        if (name.rfind("text.blocks.", 0) == 0 && vision_type != GGML_TYPE_F32) ++text_quantized;
        if (legacy_type != GGML_TYPE_F32) ++legacy_quantized;
    }
    require(model.tensors.size() == 1133 && vision_quantized == 128 && vision_q8_fallbacks == 32 &&
            text_quantized == 0 && legacy_quantized == 224,
            "vision-only profile has an unexpected quantized tensor count or text policy");
    require(sam::internal::sam3::image_quantized_tensor_type("vit.blocks.32.attn.qkv.weight", regular_row, *q4) == GGML_TYPE_F32 &&
            sam::internal::sam3::image_quantized_tensor_type("vit.blocks.0.mlp.lin2.bias", {1024}, *q4) == GGML_TYPE_F32 &&
            sam::internal::sam3::image_quantized_tensor_type("neck.det.0.conv_1x1.weight", {256, 256, 1, 1}, *q8) == GGML_TYPE_F32,
            "quantization profile accepted a tensor outside its exact whitelist");
    sam::ModelInfo info{"sam3", "q8_0", sam::Backend::Cpu, 2, 1, 34, false};
    require(info.arithmetic_profile.empty() && info.quantization_modules.empty(),
            "trailing ModelInfo fields broke old aggregate initialization");
}

void check_modular_profile_policy() {
    using namespace sam::internal::sam3;
    const auto modules = parse_image_quantization_modules("vision,text,fusion,decoder");
    require(modules == std::vector<std::string>{"vision", "text", "fusion", "decoder"},
            "schema-4 canonical module list parsed incorrectly");
    for (const auto* invalid : {"", "text,vision", "vision,vision", "vision,text,",
                                "vision, text", "vision,cuda"}) {
        rejects([&] { (void)parse_image_quantization_modules(invalid); },
                "noncanonical schema-4 module list was accepted");
    }
    const auto* full_q8 = modular_image_quantization_profile("image-full-linear-q8_0-v1");
    const auto* module_q8 = modular_image_quantization_profile("image-modules-linear-q8_0-v1");
    const auto* full_q6 = modular_image_quantization_profile("image-full-linear-q6_k-v1");
    require(full_q8 && module_q8 && full_q6 && full_q8->full_preset && !module_q8->full_preset &&
            full_q8->type == GGML_TYPE_Q8_0 && full_q6->type == GGML_TYPE_Q6_K &&
            image_quantization_modules_match_profile(*full_q8, modules) &&
            image_quantization_modules_match_profile(*module_q8, modules) &&
            !image_quantization_modules_match_profile(*full_q8, {"vision"}) &&
            image_quantization_modules_match_profile(*module_q8, {"vision"}),
            "schema-4 profile names do not bind full versus configurable module selection");

    auto context = sam::internal::make_context(4096);
    sam::internal::sam3::sam3_model model;
    model.ctx = context.get();
    model.weight_type = GGML_TYPE_F32;
    sam3_register_tensors(model);

    struct ModuleCounts { int tensors = 0; std::size_t elements = 0; };
    std::map<std::string, ModuleCounts> counts;
    int full_q8_count = 0, full_q6_count = 0, k_q8_fallbacks = 0;
    std::size_t full_elements = 0;
    const std::vector<std::string> single_modules{"vision", "text", "fusion", "decoder"};
    for (const auto& entry : model.tensors) {
        const auto& name = entry.first;
        const auto* tensor = entry.second;
        std::vector<std::int64_t> shape{tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]};
        while (shape.size() > 1 && shape.back() == 1) shape.pop_back();
        for (const auto& module : single_modules) {
            const auto type = image_modular_quantized_tensor_type(name, shape, *full_q8, {module});
            if (type != GGML_TYPE_F32) {
                ++counts[module].tensors;
                counts[module].elements += static_cast<std::size_t>(ggml_nelements(tensor));
            }
        }
        const auto q8_type = image_modular_quantized_tensor_type(name, shape, *full_q8, modules);
        const auto q6_type = image_modular_quantized_tensor_type(name, shape, *full_q6, modules);
        if (q8_type != GGML_TYPE_F32) {
            ++full_q8_count;
            full_elements += static_cast<std::size_t>(ggml_nelements(tensor));
        }
        if (q6_type != GGML_TYPE_F32) ++full_q6_count;
        if (q6_type == GGML_TYPE_Q8_0) ++k_q8_fallbacks;
    }
    require(counts["vision"].tensors == 128 && counts["vision"].elements == 444596224 &&
            counts["text"].tensors == 97 && counts["text"].elements == 302252032 &&
            counts["fusion"].tensors == 36 && counts["fusion"].elements == 9437184 &&
            counts["decoder"].tensors == 87 && counts["decoder"].elements == 18027520 &&
            full_q8_count == 348 && full_elements == 774312960 &&
            full_q6_count == 348 && k_q8_fallbacks == 32,
            "schema-4 module whitelist counts, parameters or K fallback differ from the canonical image graph");

    const auto one_module = std::vector<std::string>{"decoder"};
    const auto q8 = modular_image_quantization_profile("image-modules-linear-q8_0-v1");
    const auto q6 = modular_image_quantization_profile("image-modules-linear-q6_k-v1");
    const std::vector<std::int64_t> rank2{256, 256};
    require(q8 && q6 &&
            image_modular_quantized_tensor_type("ddec.bbox_embed.layers.0.weight", rank2, *q8, one_module) == GGML_TYPE_Q8_0 &&
            image_modular_quantized_tensor_type("ddec.bbox_embed.layers.0.weight", rank2, *q6, one_module) == GGML_TYPE_Q6_K &&
            image_modular_quantized_tensor_type("ddec.presence_token_head.layers.2.weight", {256}, *q8, one_module) == GGML_TYPE_F32 &&
            image_modular_linear_weight("ddec.presence_token_head.layers.2.weight", "decoder") &&
            image_modular_quantized_tensor_type("ddec.query_embed.weight", rank2, *q8, one_module) == GGML_TYPE_F32 &&
            image_modular_quantized_tensor_type("ddec.reference_points.weight", {4, 200}, *q8, one_module) == GGML_TYPE_F32 &&
            image_modular_quantized_tensor_type("ddec.boxRPB_embed_x.layers.0.weight", {2, 256}, *q8, one_module) == GGML_TYPE_F32 &&
            image_modular_quantized_tensor_type("geom.boxes_direct_project.weight", {4, 256}, *q8, one_module) == GGML_TYPE_F32 &&
            image_modular_quantized_tensor_type("geom.boxes_pos_enc_project.weight", {258, 256}, *q8, one_module) == GGML_TYPE_F32 &&
            image_modular_linear_weight("geom.boxes_pos_enc_project.weight", "decoder") &&
            image_modular_quantized_tensor_type("geom.boxes_pool_project.weight", {7, 7, 256, 256}, *q8, one_module) == GGML_TYPE_F32 &&
            image_modular_quantized_tensor_type("seg.instance_seg_head.weight", {1, 1, 256, 256}, *q8, one_module) == GGML_TYPE_F32 &&
            image_modular_quantized_tensor_type("seg.mask_predictor.mask_embed.layers.0.weight", rank2, *q8, one_module) == GGML_TYPE_Q8_0 &&
            image_modular_quantized_tensor_type("text.resizer.weight", {1024, 256}, *q8, one_module) == GGML_TYPE_F32 &&
            image_modular_quantized_tensor_type("text.resizer.weight", {1024, 256}, *q8, {"text"}) == GGML_TYPE_Q8_0 &&
            image_modular_quantized_tensor_type("vit.blocks.0.mlp.lin2.weight", {4736, 1024}, *q6, {"vision"}) == GGML_TYPE_Q8_0 &&
            image_modular_quantized_tensor_type("vit.blocks.0.attn.qkv.weight", {4736, 3072}, *q6, {"vision"}) == GGML_TYPE_F32,
            "schema-4 module selection quantized an embedding, vector, unsupported row or convolution");
}

void check_quantized_backend_routing(bool metal_available) {
    sam::internal::GgmlRuntime automatic({sam::Backend::Auto, 2}, false, true);
    require(automatic.backend() == sam::Backend::Cpu && automatic.quantized_cpu_f32_weights() &&
            std::string(automatic.arithmetic_profile()) == "ggml-quantized-weights-f32-v1",
            "quantized Auto selection did not retain the CPU as its default backend");
    sam::internal::GgmlRuntime legacy_cpu({sam::Backend::Cpu, 2}, true, false);
    require(std::string(legacy_cpu.arithmetic_profile()).empty(),
            "legacy CPU arithmetic profile should remain unspecified");
    if (metal_available) {
        sam::internal::GgmlRuntime explicit_metal({sam::Backend::Metal, 2}, false, true);
        require(explicit_metal.backend() == sam::Backend::Metal && explicit_metal.backends().size() == 2 &&
                std::string(ggml_backend_name(explicit_metal.backends().back())).find("CPU") != std::string::npos &&
                std::string(explicit_metal.arithmetic_profile()) == "ggml-quantized-native-v1",
                "explicit quantized Metal did not keep the required CPU scheduler tail");
    }
}

void check_quantized_metal_scheduler(bool metal_available) {
    if (!metal_available) return;
    sam::internal::GgmlRuntime runtime({sam::Backend::Metal, 2}, false, true);
    auto weight_context = sam::internal::make_context(4);
    auto* weight = ggml_new_tensor_2d(weight_context.get(), GGML_TYPE_Q8_0, 32, 32);
    sam::internal::BufferPtr weight_buffer(
        ggml_backend_alloc_ctx_tensors(weight_context.get(), runtime.weights_backend()));
    if (!weight_buffer) throw std::runtime_error("failed to allocate explicit Metal quantized weight buffer");
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    sam::RuntimeStats stats;
    sam::internal::GraphExecution execution(runtime, 32, stats);
    auto* context = execution.context();
    auto* right = sam::internal::input_tensor(context, "quantized.scheduler.input", 32, 2);
    auto* product = ggml_mul_mat(context, weight, right);
    ggml_prec_set_acc(product, GGML_PREC_F32);
    execution.output(product);
    execution.allocate();
    const auto source = matrix_values(32, 32, 11);
    const auto packed = encode_rows(GGML_TYPE_Q8_0, 32, 32, source);
    const auto input = matrix_values(32, 2, 7);
    ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
    sam::internal::upload(right, input, stats);
    execution.compute();
    const auto output = sam::internal::download(product, stats);
    require(!output.empty() && std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); }),
            "quantized Metal scheduler returned an invalid result");
    require(stats.metal_nodes > 0 && stats.cpu_nodes == 0,
            "explicit quantized Metal execution assigned a graph node to the CPU tail");
}

void compare_product(ggml_backend_t backend, ggml_tensor* output, ggml_type type,
                     std::int64_t width, std::int64_t rows, const std::vector<char>& packed,
                     const std::vector<float>& right, std::int64_t columns, const char* label);

std::vector<float> positive_weights(std::int64_t width, std::int64_t rows, float adjustment) {
    std::vector<float> values(static_cast<std::size_t>(width * rows));
    for (std::int64_t row = 0; row < rows; ++row)
        for (std::int64_t column = 0; column < width; ++column)
            values[static_cast<std::size_t>(row * width + column)] =
                0.20f + static_cast<float>((row * 13 + column * 7) % 23) * 0.004f + adjustment;
    return values;
}

std::vector<float> small_rhs(std::int64_t width, bool outlier) {
    std::vector<float> values(static_cast<std::size_t>(width), 0.02f);
    if (outlier) values[0] = 128.0f;
    return values;
}

void check_quantized_cpu_f32_weight_cast(ggml_type type, std::int64_t width, std::int64_t rows) {
    sam::internal::GgmlRuntime runtime({sam::Backend::Cpu, 2}, false, true);
    require(runtime.quantized_cpu_f32_weights() && !runtime.quantized_native_metal_only(),
            "quantized CPU runtime did not select the F32-weight matmul policy");
    auto weight_context = sam::internal::make_context(4);
    auto* weight_a = ggml_new_tensor_2d(weight_context.get(), type, width, rows);
    auto* weight_b = ggml_new_tensor_2d(weight_context.get(), type, width, rows);
    sam::internal::BufferPtr weight_buffer(
        ggml_backend_alloc_ctx_tensors(weight_context.get(), runtime.weights_backend()));
    if (!weight_buffer) throw std::runtime_error("failed to allocate packed CPU model weights");
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    sam::RuntimeStats stats;
    sam::internal::GraphExecution execution(runtime, 32, stats);
    auto* context = execution.context();
    auto* rhs_small_tensor = sam::internal::input_tensor(context, "quantized.cpu.small", width);
    auto* rhs_outlier_tensor = sam::internal::input_tensor(context, "quantized.cpu.outlier", width);
    auto* rhs_other_tensor = sam::internal::input_tensor(context, "quantized.cpu.other", width);
    auto* small_product = ggml_mul_mat(context, weight_a, rhs_small_tensor);
    auto* outlier_product = ggml_mul_mat(context, weight_a, rhs_outlier_tensor);
    auto* other_product = ggml_mul_mat(context, weight_b, rhs_other_tensor);
    for (auto* product : {small_product, outlier_product, other_product})
        if (!ggml_prec_set_acc(product, GGML_PREC_F32))
            throw std::runtime_error("GGML rejected F32 accumulation on quantized CPU graph");
    execution.output(small_product);
    execution.output(outlier_product);
    execution.output(other_product);
    execution.allocate();

    const auto source_a = positive_weights(width, rows, 0.0f);
    const auto source_b = positive_weights(width, rows, 0.003f);
    const auto packed_a = encode_rows(type, width, rows, source_a);
    const auto packed_b = encode_rows(type, width, rows, source_b);
    const auto rhs_small_values = small_rhs(width, false);
    const auto rhs_outlier_values = small_rhs(width, true);
    ggml_backend_tensor_set(weight_a, packed_a.data(), 0, packed_a.size());
    ggml_backend_tensor_set(weight_b, packed_b.data(), 0, packed_b.size());
    sam::internal::upload(rhs_small_tensor, rhs_small_values, stats);
    sam::internal::upload(rhs_outlier_tensor, rhs_outlier_values, stats);
    sam::internal::upload(rhs_other_tensor, rhs_outlier_values, stats);

    execution.compute();
    compare_product(runtime.weights_backend(), small_product, type, width, rows, packed_a,
                    rhs_small_values, 1, "CPU-F32-cast-small");
    compare_product(runtime.weights_backend(), outlier_product, type, width, rows, packed_a,
                    rhs_outlier_values, 1, "CPU-F32-cast-outlier");
    compare_product(runtime.weights_backend(), other_product, type, width, rows, packed_b,
                    rhs_outlier_values, 1, "CPU-F32-cast-shared-weight-policy");

    std::vector<char> readback_a(packed_a.size()), readback_b(packed_b.size());
    ggml_backend_tensor_get(weight_a, readback_a.data(), 0, readback_a.size());
    ggml_backend_tensor_get(weight_b, readback_b.data(), 0, readback_b.size());
    require(readback_a == packed_a && readback_b == packed_b && weight_a->type == type && weight_b->type == type &&
            ggml_nbytes(weight_a) == packed_a.size() && ggml_nbytes(weight_b) == packed_b.size() &&
            ggml_backend_buffer_get_usage(weight_buffer.get()) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
            "CPU F32 casts modified or replaced the packed resident weights");
    const auto cast_bytes = static_cast<std::size_t>(width * rows) * sizeof(float);
    require(stats.cpu_nodes == 5 && stats.blas_nodes == 0 &&
            stats.compute_buffer_bytes >= cast_bytes && stats.compute_buffer_bytes < 2 * cast_bytes,
            "CPU graph did not share one F32 cast per weight or reuse cast storage by liveness");
    std::cout << "CPU " << ggml_type_name(type) << " K=" << width << " M=" << rows
              << " packed_weights_buffer_bytes=" << ggml_backend_buffer_get_size(weight_buffer.get())
              << " compute_buffer_bytes=" << stats.compute_buffer_bytes
              << " one_f32_weight_cast_bytes=" << cast_bytes
              << " compute_nodes=" << stats.cpu_nodes << '\n';
}

void check_quantized_cpu_f32_qkv_views(ggml_type type) {
    constexpr std::int64_t width = 256, rows = 3 * width, columns = 3;
    sam::internal::GgmlRuntime runtime({sam::Backend::Cpu, 2}, false, true);
    auto weight_context = sam::internal::make_context(4);
    auto* qkv = ggml_new_tensor_2d(weight_context.get(), type, width, rows);
    sam::internal::BufferPtr weight_buffer(
        ggml_backend_alloc_ctx_tensors(weight_context.get(), runtime.weights_backend()));
    if (!weight_buffer) throw std::runtime_error("failed to allocate packed QKV-view weights");
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    sam::RuntimeStats stats;
    sam::internal::GraphExecution execution(runtime, 128, stats);
    auto* context = execution.context();
    auto* right = sam::internal::input_tensor(context, "quantized.cpu.qkv-view.input", width, columns);
    std::array<ggml_tensor*, 3> outputs{};
    for (std::int64_t slice = 0; slice < 3; ++slice) {
        auto* view = ggml_view_2d(context, qkv, width, width, qkv->nb[1],
                                  static_cast<std::size_t>(slice * width) * qkv->nb[1]);
        outputs[slice] = ggml_mul_mat(context, view, right);
        if (!ggml_prec_set_acc(outputs[slice], GGML_PREC_F32))
            throw std::runtime_error("GGML rejected F32 accumulation for a packed QKV view");
        execution.output(outputs[slice]);
    }
    execution.allocate();

    const auto source = matrix_values(width, rows, 19);
    const auto packed = encode_rows(type, width, rows, source);
    const auto input = matrix_values(width, columns, 23);
    ggml_backend_tensor_set(qkv, packed.data(), 0, packed.size());
    sam::internal::upload(right, input, stats);
    execution.compute();

    const auto* traits = ggml_get_type_traits(type);
    const auto row_bytes = static_cast<std::size_t>(width / traits->blck_size) * traits->type_size;
    for (std::int64_t slice = 0; slice < 3; ++slice) {
        const auto begin = packed.begin() + static_cast<std::size_t>(slice * width) * row_bytes;
        const auto end = begin + static_cast<std::size_t>(width) * row_bytes;
        const std::vector<char> slice_packed(begin, end);
        compare_product(runtime.weights_backend(), outputs[slice], type, width, width, slice_packed,
                        input, columns, "CPU-F32-cast-QKV-view-D256");
    }
}

void compare_product(ggml_backend_t backend, ggml_tensor* output, ggml_type type,
                     std::int64_t width, std::int64_t rows, const std::vector<char>& packed,
                     const std::vector<float>& right, std::int64_t columns, const char* label) {
    const auto decoded = decode_rows(type, width, rows, packed);
    std::vector<float> actual(static_cast<std::size_t>(rows * columns));
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    double maximum_error = 0.0;
    for (std::int64_t column = 0; column < columns; ++column) {
        for (std::int64_t row = 0; row < rows; ++row) {
            double expected = 0.0;
            for (std::int64_t k = 0; k < width; ++k)
                expected += static_cast<double>(decoded[row * width + k]) * right[column * width + k];
            const auto index = static_cast<std::size_t>(column * rows + row);
            if (!std::isfinite(actual[index])) throw std::runtime_error("quantized matrix produced a non-finite result");
            const double error = std::abs(static_cast<double>(actual[index]) - expected);
            maximum_error = std::max(maximum_error, error);
            if (error > 0.025 + std::abs(expected) * 0.005) {
                throw std::runtime_error(std::string(ggml_backend_name(backend)) + " " + label +
                    " quantized matrix differs from its decoded scalar reference");
            }
        }
    }
    std::cout << ggml_backend_name(backend) << ' ' << label << ' ' << ggml_type_name(type)
              << " max_abs_error=" << maximum_error << '\n';
}

void check_backend_quantized_math(ggml_backend_t backend) {
    const std::array<ggml_type, 4> types{
        GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K};
    const std::array<std::int64_t, 2> widths{256, 1024};
    for (std::size_t type_index = 0; type_index < types.size(); ++type_index) {
        const auto type = types[type_index];
        for (const auto width : widths) {
            const std::int64_t rows = 3 * width, columns = 3, tail_rows = 257;
            auto context = sam::internal::make_context(32, 32);
            auto* qkv = ggml_new_tensor_2d(context.get(), type, width, rows);
            auto* right_tensor = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, columns);
            std::array<ggml_tensor*, 3> outputs{};
            for (std::int64_t slice = 0; slice < 3; ++slice) {
                auto* view = ggml_view_2d(context.get(), qkv, width, width, qkv->nb[1],
                                          static_cast<std::size_t>(slice * width) * qkv->nb[1]);
                outputs[slice] = ggml_mul_mat(context.get(), view, right_tensor);
                if (!ggml_prec_set_acc(outputs[slice], GGML_PREC_F32))
                    throw std::runtime_error("GGML rejected the requested F32 accumulation mode");
            }
            auto* tail = ggml_new_tensor_2d(context.get(), type, width, tail_rows);
            auto* tail_product = ggml_mul_mat(context.get(), tail, right_tensor);
            if (!ggml_prec_set_acc(tail_product, GGML_PREC_F32))
                throw std::runtime_error("GGML rejected the tail F32 accumulation mode");
            auto* graph = ggml_new_graph_custom(context.get(), 32, false);
            for (auto* output : outputs) {
                if (!ggml_backend_supports_op(backend, output))
                    throw std::runtime_error(std::string(ggml_backend_name(backend)) +
                                             " lacks direct support for a required quantized matrix type");
                ggml_build_forward_expand(graph, output);
            }
            if (!ggml_backend_supports_op(backend, tail_product))
                throw std::runtime_error(std::string(ggml_backend_name(backend)) +
                                         " lacks direct support for a quantized M-tail matrix");
            ggml_build_forward_expand(graph, tail_product);
            sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
            if (!buffer) throw std::runtime_error("failed to allocate direct quantized arithmetic buffers");

            const auto right_values = matrix_values(width, columns, 3);
            ggml_backend_tensor_set(right_tensor, right_values.data(), 0, right_values.size() * sizeof(float));
            const auto qkv_source = matrix_values(width, rows, static_cast<int>(type_index) + 1);
            const auto tail_source = matrix_values(width, tail_rows, static_cast<int>(type_index) + 7);
            const auto qkv_packed = encode_rows(type, width, rows, qkv_source);
            const auto tail_packed = encode_rows(type, width, tail_rows, tail_source);
            sam::internal::sam3::validate_quantized_payload(type, {width, rows}, qkv_packed.data(), qkv_packed.size());
            sam::internal::sam3::validate_quantized_payload(type, {width, tail_rows}, tail_packed.data(), tail_packed.size());
            ggml_backend_tensor_set(qkv, qkv_packed.data(), 0, qkv_packed.size());
            ggml_backend_tensor_set(tail, tail_packed.data(), 0, tail_packed.size());
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error(std::string(ggml_backend_name(backend)) + " direct quantized graph failed");

            const auto* traits = ggml_get_type_traits(type);
            const auto row_bytes = static_cast<std::size_t>(width / traits->blck_size) * traits->type_size;
            for (std::int64_t slice = 0; slice < 3; ++slice) {
                const auto begin = qkv_packed.begin() + static_cast<std::size_t>(slice * width) * row_bytes;
                const auto end = begin + static_cast<std::size_t>(width) * row_bytes;
                const std::vector<char> slice_packed(begin, end);
                compare_product(backend, outputs[slice], type, width, width, slice_packed,
                                right_values, columns, width == 256 ? "QKV-strided-view-D256" : "QKV-strided-view");
            }
            compare_product(backend, tail_product, type, width, tail_rows, tail_packed,
                            right_values, columns, "M-tail");
        }
    }
}

void check_backend_q8_k_fallback(ggml_backend_t backend) {
    constexpr std::int64_t width = 4736, rows = 17, columns = 3;
    auto context = sam::internal::make_context(8, 8);
    auto* weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_Q8_0, width, rows);
    auto* right_tensor = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, columns);
    auto* output = ggml_mul_mat(context.get(), weight, right_tensor);
    if (!ggml_prec_set_acc(output, GGML_PREC_F32) || !ggml_backend_supports_op(backend, output))
        throw std::runtime_error(std::string(ggml_backend_name(backend)) +
                                 " lacks direct Q8_0 support for the SAM 4736-wide matrix");
    auto* graph = ggml_new_graph_custom(context.get(), 8, false);
    ggml_build_forward_expand(graph, output);
    sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) throw std::runtime_error("failed to allocate Q8_0 fallback arithmetic buffers");
    const auto right = matrix_values(width, columns, 13);
    const auto source = matrix_values(width, rows, 17);
    const auto packed = encode_rows(GGML_TYPE_Q8_0, width, rows, source);
    sam::internal::sam3::validate_quantized_payload(GGML_TYPE_Q8_0, {width, rows}, packed.data(), packed.size());
    ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(right_tensor, right.data(), 0, right.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error(std::string(ggml_backend_name(backend)) + " Q8_0 4736-wide graph failed");
    compare_product(backend, output, GGML_TYPE_Q8_0, width, rows, packed, right, columns, "Q8-4736-K-tail");
}

} // namespace

int main() {
    try {
        check_profile_policy();
        check_modular_profile_policy();
        check_quantized_payload_validation();
        auto cpu = sam::internal::make_cpu_backend(2);
        auto metal = sam::internal::make_metal_backend();
        check_quantized_backend_routing(static_cast<bool>(metal.handle));
        check_quantized_metal_scheduler(static_cast<bool>(metal.handle));
        check_quantized_cpu_f32_weight_cast(GGML_TYPE_Q8_0, 4736, 1024);
        check_quantized_cpu_f32_weight_cast(GGML_TYPE_Q6_K, 1024, 4736);
        check_quantized_cpu_f32_qkv_views(GGML_TYPE_Q8_0);
        check_quantized_cpu_f32_qkv_views(GGML_TYPE_Q6_K);
        check_backend_quantized_math(cpu.handle.get());
        check_backend_q8_k_fallback(cpu.handle.get());
        if (metal.handle) {
            check_backend_quantized_math(metal.handle.get());
            check_backend_q8_k_fallback(metal.handle.get());
        }
        else std::cout << "Metal device unavailable; direct Metal quantized arithmetic skipped\n";
        std::cout << "SAM quantization checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
