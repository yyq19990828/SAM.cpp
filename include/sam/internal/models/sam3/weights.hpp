#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_WEIGHTS_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_WEIGHTS_HPP

// Tensor precision and tokenizer validation adapted from PABannier/sam3.cpp,
// revision 416186c501d060df7ca02989d49b38080f5f81f3.
// Copyright (c) 2025-2026 Pierre-Antoine Bannier. MIT: licenses/sam3.cpp-MIT.txt.

#include "tokenizer.hpp"
#include "sam/internal/io/gguf_reader.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sam::internal::sam3 {

using TensorInfo = GgufTensor;

struct WeightFile {
    std::string path;
    std::uint64_t file_size = 0;
    std::int32_t ftype = 0;
    bool video = false;
    bool quantized = false;
    bool modular_quantized = false;
    std::string precision;
    std::string storage_profile;
    std::vector<std::string> quantization_modules;
    std::vector<TensorInfo> tensors;
    TokenizerData tokenizer;
    std::unique_ptr<GgufReader> reader;
};

struct ImageQuantizationProfile {
    const char* name;
    const char* precision;
    ggml_type type;
    std::uint32_t file_type;
    bool quantize_text_linear;
};

struct ModularImageQuantizationProfile {
    const char* name;
    const char* precision;
    ggml_type type;
    std::uint32_t file_type;
    bool full_preset;
};

inline const ImageQuantizationProfile* image_quantization_profile(const std::string& name) {
    static constexpr std::array<ImageQuantizationProfile, 8> profiles{{
        {"image-linear-q8_0-v1", "q8_0", GGML_TYPE_Q8_0, 7, true},
        {"image-linear-q6_k-v1", "q6_k", GGML_TYPE_Q6_K, 18, true},
        {"image-linear-q5_k-v1", "q5_k", GGML_TYPE_Q5_K, 16, true},
        {"image-linear-q4_k-v1", "q4_k", GGML_TYPE_Q4_K, 14, true},
        {"image-vision-linear-q8_0-v1", "q8_0", GGML_TYPE_Q8_0, 7, false},
        {"image-vision-linear-q6_k-v1", "q6_k", GGML_TYPE_Q6_K, 18, false},
        {"image-vision-linear-q5_k-v1", "q5_k", GGML_TYPE_Q5_K, 16, false},
        {"image-vision-linear-q4_k-v1", "q4_k", GGML_TYPE_Q4_K, 14, false},
    }};
    for (const auto& profile : profiles) if (name == profile.name) return &profile;
    return nullptr;
}

inline const ModularImageQuantizationProfile* modular_image_quantization_profile(const std::string& name) {
    static constexpr std::array<ModularImageQuantizationProfile, 8> profiles{{
        {"image-full-linear-q8_0-v1", "q8_0", GGML_TYPE_Q8_0, 7, true},
        {"image-full-linear-q6_k-v1", "q6_k", GGML_TYPE_Q6_K, 18, true},
        {"image-full-linear-q5_k-v1", "q5_k", GGML_TYPE_Q5_K, 16, true},
        {"image-full-linear-q4_k-v1", "q4_k", GGML_TYPE_Q4_K, 14, true},
        {"image-modules-linear-q8_0-v1", "q8_0", GGML_TYPE_Q8_0, 7, false},
        {"image-modules-linear-q6_k-v1", "q6_k", GGML_TYPE_Q6_K, 18, false},
        {"image-modules-linear-q5_k-v1", "q5_k", GGML_TYPE_Q5_K, 16, false},
        {"image-modules-linear-q4_k-v1", "q4_k", GGML_TYPE_Q4_K, 14, false},
    }};
    for (const auto& profile : profiles) if (name == profile.name) return &profile;
    return nullptr;
}

inline const std::array<const char*, 4>& image_quantization_module_order() {
    static constexpr std::array<const char*, 4> modules{{"vision", "text", "fusion", "decoder"}};
    return modules;
}

inline std::vector<std::string> parse_image_quantization_modules(const std::string& csv) {
    if (csv.empty()) throw std::runtime_error("schema-4 quantization module list is empty");
    std::vector<std::string> result;
    int previous = -1;
    std::size_t begin = 0;
    while (begin <= csv.size()) {
        const auto comma = csv.find(',', begin);
        const auto end = comma == std::string::npos ? csv.size() : comma;
        const auto token = csv.substr(begin, end - begin);
        int index = -1;
        const auto& ordered = image_quantization_module_order();
        for (std::size_t i = 0; i < ordered.size(); ++i)
            if (token == ordered[i]) index = static_cast<int>(i);
        if (index < 0 || index <= previous)
            throw std::runtime_error("schema-4 quantization modules must be a nonempty canonical CSV list");
        previous = index;
        result.push_back(token);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return result;
}

inline bool image_quantization_modules_match_profile(const ModularImageQuantizationProfile& profile,
                                                     const std::vector<std::string>& modules) {
    if (modules.empty()) return false;
    if (!profile.full_preset) return true;
    const auto& canonical = image_quantization_module_order();
    if (modules.size() != canonical.size()) return false;
    for (std::size_t i = 0; i < canonical.size(); ++i)
        if (modules[i] != canonical[i]) return false;
    return true;
}

inline bool is_image_quantized_linear_weight(const std::string& name, bool quantize_text_linear) {
    for (int block = 0; block < 32; ++block) {
        const auto prefix = "vit.blocks." + std::to_string(block) + ".";
        for (const char* suffix : {"attn.qkv.weight", "attn.proj.weight", "mlp.lin1.weight", "mlp.lin2.weight"})
            if (name == prefix + suffix) return true;
    }
    if (!quantize_text_linear) return false;
    for (int block = 0; block < 24; ++block) {
        const auto prefix = "text.blocks." + std::to_string(block) + ".";
        for (const char* suffix : {"attn.in_proj.weight", "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"})
            if (name == prefix + suffix) return true;
    }
    return false;
}

inline bool is_image_q8_k_fallback_weight(const std::string& name) {
    for (int block = 0; block < 32; ++block)
        if (name == "vit.blocks." + std::to_string(block) + ".mlp.lin2.weight") return true;
    return false;
}

inline ggml_type image_quantized_tensor_type(const std::string& name,
                                            const std::vector<std::int64_t>& dimensions,
                                            const ImageQuantizationProfile& profile) {
    if (!is_image_quantized_linear_weight(name, profile.quantize_text_linear)) return GGML_TYPE_F32;
    const auto block = static_cast<std::int64_t>(ggml_blck_size(profile.type));
    if (block == 256 && dimensions.size() && dimensions[0] == 4736 && is_image_q8_k_fallback_weight(name))
        return GGML_TYPE_Q8_0;
    if (block == 256 && (dimensions.empty() || dimensions[0] % block != 0))
        throw std::runtime_error("K quantization profile has a noncanonical row width: " + name);
    return profile.type;
}

template<std::size_t N>
inline bool image_indexed_layer_weight(const std::string& name, const std::string& prefix,
                                       int layer_count, const std::array<const char*, N>& suffixes) {
    for (int layer = 0; layer < layer_count; ++layer) {
        const auto layer_prefix = prefix + std::to_string(layer) + ".";
        for (const auto* suffix : suffixes)
            if (name == layer_prefix + suffix) return true;
    }
    return false;
}

inline bool image_modular_linear_weight(const std::string& name, const std::string& module) {
    if (module == "vision") return is_image_quantized_linear_weight(name, false);
    if (module == "text") {
        if (name == "text.resizer.weight") return true;
        static constexpr std::array<const char*, 4> suffixes{{
            "attn.in_proj.weight", "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"}};
        return image_indexed_layer_weight(name, "text.blocks.", 24, suffixes);
    }
    if (module == "fusion") {
        static constexpr std::array<const char*, 6> suffixes{{
            "sa.in_proj_weight", "sa.out_proj.weight", "ca.in_proj_weight",
            "ca.out_proj.weight", "linear1.weight", "linear2.weight"}};
        return image_indexed_layer_weight(name, "fenc.layers.", 6, suffixes);
    }
    if (module != "decoder") return false;

    static constexpr std::array<const char*, 8> decoder_layer_suffixes{{
        "sa.in_proj_weight", "sa.out_proj.weight", "ca.in_proj_weight", "ca.out_proj.weight",
        "ca_text.in_proj_weight", "ca_text.out_proj.weight", "linear1.weight", "linear2.weight"}};
    if (image_indexed_layer_weight(name, "ddec.layers.", 6, decoder_layer_suffixes)) return true;

    static constexpr std::array<const char*, 6> geometry_layer_suffixes{{
        "sa.in_proj_weight", "sa.out_proj.weight", "ca.in_proj_weight", "ca.out_proj.weight",
        "linear1.weight", "linear2.weight"}};
    if (image_indexed_layer_weight(name, "geom.layers.", 3, geometry_layer_suffixes)) return true;

    for (const auto* head : {"ddec.bbox_embed.layers.", "ddec.presence_token_head.layers.",
                             "seg.mask_predictor.mask_embed.layers."}) {
        for (int layer = 0; layer < 3; ++layer)
            if (name == std::string(head) + std::to_string(layer) + ".weight") return true;
    }
    if (image_indexed_layer_weight(name, "ddec.ref_point_head.layers.", 2,
                                   std::array<const char*, 1>{{"weight"}})) return true;
    for (const auto* axis : {"x", "y"})
        if (name == std::string("ddec.boxRPB_embed_") + axis + ".layers.0.weight" ||
            name == std::string("ddec.boxRPB_embed_") + axis + ".layers.1.weight") return true;
    for (const auto* fixed : {"scoring.prompt_proj.weight", "scoring.hs_proj.weight",
                              "scoring.prompt_mlp.layers.0.weight", "scoring.prompt_mlp.layers.1.weight",
                              "geom.points_direct_project.weight", "geom.boxes_direct_project.weight",
                              "geom.boxes_pos_enc_project.weight", "geom.points_pool_project.weight",
                              "geom.points_pos_enc_project.weight",
                              "geom.final_proj.weight", "seg.cross_attend_prompt.in_proj_weight",
                              "seg.cross_attend_prompt.out_proj.weight"})
        if (name == fixed) return true;
    return false;
}

inline ggml_type image_modular_quantized_tensor_type(const std::string& name,
                                                      const std::vector<std::int64_t>& dimensions,
                                                      const ModularImageQuantizationProfile& profile,
                                                      const std::vector<std::string>& modules) {
    if (dimensions.size() != 2) return GGML_TYPE_F32;
    const auto selected = [&](const char* module) {
        return std::find(modules.begin(), modules.end(), module) != modules.end();
    };
    bool eligible = false;
    for (const auto& module : image_quantization_module_order())
        eligible = eligible || (selected(module) && image_modular_linear_weight(name, module));
    if (!eligible) return GGML_TYPE_F32;

    const auto row_width = dimensions[0];
    if (row_width <= 0 || row_width % 32 != 0) return GGML_TYPE_F32;
    if (profile.type == GGML_TYPE_Q8_0) return profile.type;
    if (row_width % 256 == 0) return profile.type;
    if (row_width == 4736 && is_image_q8_k_fallback_weight(name)) return GGML_TYPE_Q8_0;
    return GGML_TYPE_F32;
}

inline void validate_quantized_payload(ggml_type type, const std::vector<std::int64_t>& dimensions,
                                       const char* payload, std::size_t payload_size) {
    const auto* traits = ggml_get_type_traits(type);
    if (!traits || !traits->is_quantized || !traits->to_float || dimensions.empty() || dimensions[0] <= 0 ||
        traits->blck_size <= 0 || dimensions[0] % traits->blck_size != 0 || traits->type_size == 0)
        throw std::runtime_error("invalid quantized SAM tensor layout");
    std::size_t rows = 1;
    for (std::size_t i = 1; i < dimensions.size(); ++i) {
        if (dimensions[i] <= 0 || rows > std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(dimensions[i]))
            throw std::runtime_error("quantized SAM tensor row count overflows");
        rows *= static_cast<std::size_t>(dimensions[i]);
    }
    const auto row_elements = static_cast<std::size_t>(dimensions[0]);
    const auto blocks = row_elements / static_cast<std::size_t>(traits->blck_size);
    if (blocks > std::numeric_limits<std::size_t>::max() / traits->type_size)
        throw std::runtime_error("quantized SAM tensor row size overflows");
    const auto row_bytes = blocks * traits->type_size;
    if (rows > std::numeric_limits<std::size_t>::max() / row_bytes || rows * row_bytes != payload_size ||
        (!payload && payload_size != 0))
        throw std::runtime_error("quantized SAM tensor payload size does not match its shape");
    std::vector<float> decoded(row_elements);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto* encoded = payload + row * row_bytes;
        for (std::size_t block = 0; block < blocks; ++block) {
            ggml_fp16_t half = 0;
            const auto* encoded_block = encoded + block * traits->type_size;
            const auto scale_offset = type == GGML_TYPE_Q6_K ? traits->type_size - sizeof(half) : 0;
            std::memcpy(&half, encoded_block + scale_offset, sizeof(half));
            const float scale = ggml_fp16_to_fp32(half);
            if (!std::isfinite(scale)) throw std::runtime_error("non-finite quantized SAM block scale");
            if ((type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) && traits->type_size >= 4) {
                std::memcpy(&half, encoded_block + sizeof(half), sizeof(half));
                if (!std::isfinite(ggml_fp16_to_fp32(half)))
                    throw std::runtime_error("non-finite quantized SAM block minimum scale");
            }
        }
        traits->to_float(encoded, decoded.data(), static_cast<std::int64_t>(row_elements));
        if (!std::all_of(decoded.begin(), decoded.end(), [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("quantized SAM tensor dequantizes to a non-finite value");
    }
}

inline std::array<std::pair<const char*, std::uint32_t>, 22> sam3_image_parameters() {
    return {{{"sam3.vision.image_size", 1008}, {"sam3.vision.patch_size", 14},
             {"sam3.vision.embedding_length", 1024}, {"sam3.vision.block_count", 32},
             {"sam3.vision.attention.head_count", 16}, {"sam3.vision.feed_forward_length", 4736},
             {"sam3.vision.window_size", 24}, {"sam3.text.embedding_length", 1024},
             {"sam3.text.attention.head_count", 16}, {"sam3.text.block_count", 24},
             {"sam3.text.context_length", 32}, {"sam3.text.vocab_size", 49408},
             {"sam3.text.output_length", 256}, {"sam3.neck.embedding_length", 256},
             {"sam3.fusion.block_count", 6}, {"sam3.fusion.attention.head_count", 8},
             {"sam3.fusion.feed_forward_length", 2048}, {"sam3.decoder.block_count", 6},
             {"sam3.decoder.attention.head_count", 8}, {"sam3.decoder.feed_forward_length", 2048},
             {"sam3.decoder.query_count", 200}, {"sam3.geometry.block_count", 3}}};
}

inline bool tensor_kept_f32(const std::string& name, std::size_t rank, bool hybrid = false) {
    if (rank == 1) return true;
    if (hybrid) for (const char* prefix : {"vit.", "neck.trk.", "mem_attn.", "mem_enc.", "sam_pe.", "sam_dec.",
                                         "obj_ptr_proj.", "obj_ptr_tpos_proj.", "trk_mask_ds."})
        if (name.rfind(prefix, 0) == 0) return true;
    for (const char* part : {"embed", "tpos", "pe_gaussian", "token", "no_obj", "no_mem", "gamma", "freqs_cis"})
        if (name.find(part) != std::string::npos) return true;
    return false;
}

inline std::array<std::int64_t, 4> canonical_dimensions(const std::vector<std::int64_t>& dimensions) {
    std::array<std::int64_t, 4> result{1, 1, 1, 1};
    if (dimensions.empty() || dimensions.size() > result.size()) throw std::runtime_error("invalid tensor rank");
    std::copy(dimensions.begin(), dimensions.end(), result.begin());
    return result;
}

inline std::string utf8_codepoint(int value) {
    if (value < 128) { return std::string(1, static_cast<char>(value)); }
    std::string result;
    result.push_back(static_cast<char>(0xc0 | (value >> 6)));
    result.push_back(static_cast<char>(0x80 | (value & 63)));
    return result;
}

inline void validate_tokenizer_vocabulary(const TokenizerData& tokenizer) {
    std::vector<int> bytes;
    for (int byte = 33; byte <= 126; ++byte) { bytes.push_back(byte); }
    for (int byte = 161; byte <= 172; ++byte) { bytes.push_back(byte); }
    for (int byte = 174; byte <= 255; ++byte) { bytes.push_back(byte); }
    std::vector<int> codepoints = bytes;
    int extended = 256;
    for (int byte = 0; byte < 256; ++byte) {
        if (std::find(bytes.begin(), bytes.end(), byte) == bytes.end()) {
            bytes.push_back(byte);
            codepoints.push_back(extended++);
        }
    }
    for (std::size_t i = 0; i < codepoints.size(); ++i) {
        const auto symbol = utf8_codepoint(codepoints[i]);
        const auto base = tokenizer.vocabulary.find(symbol);
        const auto ending = tokenizer.vocabulary.find(symbol + "</w>");
        if (base == tokenizer.vocabulary.end() || base->second != static_cast<std::int32_t>(i) ||
            ending == tokenizer.vocabulary.end() || ending->second != static_cast<std::int32_t>(i + 256)) {
            throw std::runtime_error("SAM tokenizer byte vocabulary does not match the official asset");
        }
    }
    const auto sot = tokenizer.vocabulary.find("<start_of_text>");
    const auto eot = tokenizer.vocabulary.find("<end_of_text>");
    if (sot == tokenizer.vocabulary.end() || sot->second != Tokenizer::sot_token ||
        eot == tokenizer.vocabulary.end() || eot->second != Tokenizer::eot_token) {
        throw std::runtime_error("SAM tokenizer special tokens do not match the official asset");
    }
}

inline WeightFile inspect_weights(const std::string& path) {
    WeightFile result;
    result.path = path;
    result.reader = std::make_unique<GgufReader>(path);
    const auto& reader = *result.reader;
    const auto* metadata = reader.metadata();
    result.file_size = reader.size();
    auto require_string = [&](const char* key, const char* expected) {
        if (reader.string(key) != expected) throw std::runtime_error(std::string("unsupported SAM GGUF metadata: ") + key);
    };
    require_string("general.architecture", "sam3");
    const auto schema = reader.u32("sam.schema_version");
    if (schema != 1 && schema != 2 && schema != 3 && schema != 4)
        throw std::runtime_error("unsupported SAM GGUF schema version");
    result.video = schema == 2;
    result.quantized = schema == 3 || schema == 4;
    result.modular_quantized = schema == 4;
    if (result.quantized && gguf_find_key(metadata, "sam.storage_profile") < 0)
        throw std::runtime_error("quantized SAM GGUF is missing its storage profile");
    require_string("sam.task", result.video ? "text_video" : "text_image");
    if (result.video) {
        const std::array<std::pair<const char*, std::uint32_t>, 9> parameters{{
            {"sam3.tracker.embedding_length", 256}, {"sam3.tracker.memory_length", 64},
            {"sam3.tracker.attention.block_count", 4}, {"sam3.tracker.attention.head_count", 1},
            {"sam3.tracker.attention.head_length", 256}, {"sam3.tracker.memory_position_count", 7},
            {"sam3.tracker.conditioning_frame_count", 4}, {"sam3.tracker.pointer_candidate_count", 16},
            {"sam3.tracker.mask_memory_size", 1152}}};
        for (const auto& parameter : parameters)
            if (reader.u32(parameter.first) != parameter.second)
                throw std::runtime_error(std::string("unsupported SAM 3 tracker parameter: ") + parameter.first);
        require_string("sam3.tracker.policy", "meta-sam3-temporal-v1");
        require_string("sam3.tracker.feature_storage", "bf16");
        require_string("sam3.tracker.memory_storage", "bf16");
        require_string("sam3.video.preprocessing", "pillow-bicubic-f16-normalize-v1");
    }
    const auto ftype = reader.u32("general.file_type");
    result.ftype = static_cast<std::int32_t>(ftype);
    const ImageQuantizationProfile* quant_profile = nullptr;
    const ModularImageQuantizationProfile* modular_profile = nullptr;
    if (result.quantized) {
        result.storage_profile = reader.string("sam.storage_profile");
        if (result.modular_quantized) {
            modular_profile = modular_image_quantization_profile(result.storage_profile);
            if (!modular_profile) throw std::runtime_error("unsupported modular SAM GGUF quantization profile");
            result.quantization_modules = parse_image_quantization_modules(
                reader.string("sam.quantization.modules", 64));
            if (!image_quantization_modules_match_profile(*modular_profile, result.quantization_modules))
                throw std::runtime_error("SAM modular profile does not match its canonical module list");
            if (ftype != modular_profile->file_type)
                throw std::runtime_error("SAM GGUF file type does not match its modular profile");
            result.precision = modular_profile->precision;
        } else {
            quant_profile = image_quantization_profile(result.storage_profile);
            if (!quant_profile) throw std::runtime_error("unsupported SAM GGUF quantization profile");
            if (ftype != quant_profile->file_type)
                throw std::runtime_error("SAM GGUF file type does not match its quantization profile");
            result.precision = quant_profile->precision;
        }
        if (reader.u32("general.quantization_version") != 2)
            throw std::runtime_error("unsupported GGML quantization version");
    } else {
        if (ftype > 1) throw std::runtime_error("SAM GGUF precision must be F32 or mixed F16");
        if (gguf_find_key(metadata, "sam.storage_profile") >= 0) {
            result.storage_profile = reader.string("sam.storage_profile");
            if (!result.video || ftype != 1 || result.storage_profile != "visual-tracker-f32-v1")
                throw std::runtime_error("unsupported SAM GGUF storage profile");
        }
        result.precision = !result.storage_profile.empty() ? "hybrid" : (ftype == 0 ? "f32" : "f16");
    }
    if (gguf_find_key(metadata, "general.name") >= 0) (void) reader.string("general.name");
    const auto source_hash = reader.string("sam.source.checkpoint_sha256", 64);
    if (source_hash.size() != 64 || !std::all_of(source_hash.begin(), source_hash.end(), [](char c) {
            return ascii_digit(c) || (c >= 'a' && c <= 'f');
        })) throw std::runtime_error("invalid SAM source checkpoint SHA-256");
    require_string("sam.source.code_revision", "2345a4ad109ac29c569da749c91d84f10dc08c40");
    require_string("sam.tokenizer.sha256", "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a");
    for (const auto& parameter : sam3_image_parameters())
        if (reader.u32(parameter.first) != parameter.second)
            throw std::runtime_error(std::string("unsupported SAM 3 image parameter: ") + parameter.first);
    const auto globals = reader.array("sam3.vision.global_attention_blocks", GGUF_TYPE_UINT32, 4);
    std::array<std::uint32_t, 4> indices{};
    std::memcpy(indices.data(), gguf_get_arr_data(metadata, globals), sizeof(indices));
    const std::array<std::uint32_t, 4> expected_indices{7, 15, 23, 31};
    if (indices != expected_indices)
        throw std::runtime_error("unsupported SAM 3 global attention block order");
    result.tensors = reader.tensors();
    if (result.tensors.size() != (result.video ? 1464 : 1133))
        throw std::runtime_error("SAM 3 GGUF tensor count does not match its task profile");
    for (const auto& tensor : result.tensors) {
        if (!std::all_of(tensor.name.begin(), tensor.name.end(), [](char c) {
                return ascii_letter(c) || ascii_digit(c) || c == '_' || c == '.';
            })) throw std::runtime_error("invalid SAM 3 tensor name: " + tensor.name);
        std::int32_t expected_type = 0;
        if (result.modular_quantized) {
            expected_type = static_cast<std::int32_t>(image_modular_quantized_tensor_type(
                tensor.name, tensor.dimensions, *modular_profile, result.quantization_modules));
        } else if (result.quantized) {
            expected_type = static_cast<std::int32_t>(image_quantized_tensor_type(
                tensor.name, tensor.dimensions, *quant_profile));
        } else {
            expected_type = result.ftype == 1 &&
                !tensor_kept_f32(tensor.name, tensor.dimensions.size(), !result.storage_profile.empty()) ? 1 : 0;
        }
        if (tensor.type != expected_type) throw std::runtime_error("incompatible SAM 3 tensor precision: " + tensor.name);
    }
    require_string("tokenizer.ggml.model", "clip");
    if (reader.u32("tokenizer.ggml.bos_token_id") != std::uint32_t(Tokenizer::sot_token) ||
        reader.u32("tokenizer.ggml.eos_token_id") != std::uint32_t(Tokenizer::eot_token) ||
        reader.u32("tokenizer.ggml.padding_token_id") != 0)
        throw std::runtime_error("incompatible SAM tokenizer special token IDs");
    const auto tokens = reader.array("tokenizer.ggml.tokens", GGUF_TYPE_STRING, Tokenizer::vocabulary_size);
    std::size_t tokenizer_bytes = 0;
    for (std::int32_t i = 0; i < Tokenizer::vocabulary_size; ++i) {
        auto token = GgufReader::bounded_string(gguf_get_arr_str(metadata, tokens, i), 2048, "tokenizer token");
        if (token.size() > GgufReader::metadata_limit - tokenizer_bytes ||
            !result.tokenizer.vocabulary.emplace(token, i).second)
            throw std::runtime_error("duplicate or oversized SAM tokenizer vocabulary");
        tokenizer_bytes += token.size();
    }
    validate_tokenizer_vocabulary(result.tokenizer);
    constexpr std::size_t merge_count = 48894;
    const auto merge_key = reader.array("tokenizer.ggml.merges", GGUF_TYPE_STRING, merge_count);
    std::unordered_set<std::string> merges;
    result.tokenizer.merges.reserve(merge_count);
    for (std::size_t rank = 0; rank < merge_count; ++rank) {
        const auto entry = GgufReader::bounded_string(gguf_get_arr_str(metadata, merge_key, rank), 4097, "tokenizer merge");
        const auto separator = entry.find(' ');
        if (separator == std::string::npos || separator == 0 || separator + 1 == entry.size() ||
            entry.find(' ', separator + 1) != std::string::npos || separator > 2048 ||
            entry.size() - separator - 1 > 2048 || entry.size() > GgufReader::metadata_limit - tokenizer_bytes)
            throw std::runtime_error("invalid or oversized SAM tokenizer merge");
        const auto first = entry.substr(0, separator), second = entry.substr(separator + 1);
        const auto combined = result.tokenizer.vocabulary.find(first + second);
        if (!result.tokenizer.vocabulary.count(first) || !result.tokenizer.vocabulary.count(second) ||
            combined == result.tokenizer.vocabulary.end() || combined->second != static_cast<std::int32_t>(512 + rank) ||
            !merges.insert(entry).second)
            throw std::runtime_error("invalid or duplicate SAM tokenizer merge sequence");
        tokenizer_bytes += entry.size();
        result.tokenizer.merges.emplace_back(first, second);
    }
    return result;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_WEIGHTS_HPP
