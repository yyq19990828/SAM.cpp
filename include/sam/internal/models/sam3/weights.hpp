#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_WEIGHTS_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_WEIGHTS_HPP

// Tensor precision and tokenizer validation adapted from PABannier/sam3.cpp,
// revision 416186c501d060df7ca02989d49b38080f5f81f3.
// Copyright (c) 2025-2026 Pierre-Antoine Bannier. MIT: licenses/sam3.cpp-MIT.txt.

#include "tokenizer.hpp"
#include "sam/internal/io/gguf_reader.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
    std::vector<TensorInfo> tensors;
    TokenizerData tokenizer;
    std::unique_ptr<GgufReader> reader;
};

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

inline bool tensor_kept_f32(const std::string& name, std::size_t rank) {
    if (rank == 1) return true;
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
    require_string("sam.task", "text_image");
    if (reader.u32("sam.schema_version") != 1) throw std::runtime_error("unsupported SAM GGUF schema version");
    const auto ftype = reader.u32("general.file_type");
    if (ftype > 1) throw std::runtime_error("SAM GGUF precision must be F32 or mixed F16");
    result.ftype = static_cast<std::int32_t>(ftype);
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
    if (result.tensors.size() != 1133) throw std::runtime_error("SAM 3 image GGUF requires exactly 1133 tensors");
    for (const auto& tensor : result.tensors) {
        if (!std::all_of(tensor.name.begin(), tensor.name.end(), [](char c) {
                return ascii_letter(c) || ascii_digit(c) || c == '_' || c == '.';
            })) throw std::runtime_error("invalid SAM 3 tensor name: " + tensor.name);
        const auto expected_type = result.ftype == 1 && !tensor_kept_f32(tensor.name, tensor.dimensions.size()) ? 1 : 0;
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
