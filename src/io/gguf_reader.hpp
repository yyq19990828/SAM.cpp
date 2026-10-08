#ifndef SAM_CPP_SRC_IO_GGUF_READER_HPP
#define SAM_CPP_SRC_IO_GGUF_READER_HPP

#include "gguf.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sam::internal {

struct GgufDeleter {
    void operator()(gguf_context* context) const { if (context) gguf_free(context); }
};
using GgufPtr = std::unique_ptr<gguf_context, GgufDeleter>;

struct GgufTensor {
    std::string name;
    std::vector<std::int64_t> dimensions;
    std::int32_t type = 0;
    std::uint64_t offset = 0;
    std::uint64_t size_bytes = 0;
};

class GgufReader {
public:
    static constexpr std::size_t metadata_limit = 16 * 1024 * 1024;

    explicit GgufReader(const std::string& path) : stream_(path, std::ios::binary) {
        if (!stream_) throw std::runtime_error("cannot open SAM checkpoint: " + path);
        stream_.seekg(0, std::ios::end);
        const auto length = stream_.tellg();
        if (length < 0) throw std::runtime_error("cannot determine SAM checkpoint size");
        size_ = static_cast<std::uint64_t>(length);
        std::array<unsigned char, 24> header{};
        read(0, reinterpret_cast<char*>(header.data()), std::min<std::uint64_t>(size_, header.size()));
        if (size_ < 4) throw std::runtime_error("truncated GGUF checkpoint header");
        if (std::string(reinterpret_cast<char*>(header.data()), 4) != GGUF_MAGIC)
            throw std::runtime_error("unsupported checkpoint: expected GGUF; reconvert the original .pt checkpoint "
                                     "with tools/convert_sam3.py, rather than loading a legacy .ggml file");
        if (size_ < header.size()) throw std::runtime_error("truncated GGUF checkpoint header");
        auto integer = [&](std::size_t offset, std::size_t bytes) {
            std::uint64_t result = 0;
            for (std::size_t i = 0; i < bytes; ++i) result |= std::uint64_t(header[offset + i]) << (8 * i);
            return result;
        };
        if (integer(4, 4) != 3) throw std::runtime_error("SAM checkpoints require little-endian GGUF v3");
        const auto tensor_count = integer(8, 8), key_count = integer(16, 8);
        if (tensor_count == 0 || tensor_count > 4096 || key_count > 256)
            throw std::runtime_error("GGUF tensor or metadata key count exceeds checkpoint limits");
        const auto budget = std::min<std::uint64_t>(size_, metadata_limit);
        context_.reset(gguf_init_from_callback(&read_callback, this, 65536, budget, {true, nullptr}));
        if (!context_) throw std::runtime_error("invalid, truncated or oversized GGUF metadata");
        if (gguf_get_alignment(context_.get()) != 32)
            throw std::runtime_error("SAM GGUF checkpoints require 32-byte alignment");
        data_offset_ = gguf_get_data_offset(context_.get());
        if (data_offset_ > size_) throw std::runtime_error("truncated GGUF data section");
        const auto available = size_ - data_offset_;
        std::uint64_t expected_offset = 0;
        tensors_.reserve(static_cast<std::size_t>(tensor_count));
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(tensor_count); ++i) {
            GgufTensor tensor;
            tensor.name = gguf_get_tensor_name(context_.get(), i);
            if (tensor.name.empty()) throw std::runtime_error("empty GGUF tensor name");
            const auto* dimensions = gguf_get_tensor_ne(context_.get(), i);
            std::uint64_t elements = 1;
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (dimensions[d] <= 0 || elements > std::uint64_t(INT64_MAX) / std::uint64_t(dimensions[d]))
                    throw std::runtime_error("invalid or overflowing GGUF tensor dimensions: " + tensor.name);
                elements *= static_cast<std::uint64_t>(dimensions[d]);
                tensor.dimensions.push_back(dimensions[d]);
            }
            while (tensor.dimensions.size() > 1 && tensor.dimensions.back() == 1) tensor.dimensions.pop_back();
            const auto type = gguf_get_tensor_type(context_.get(), i);
            const auto block = static_cast<std::uint64_t>(ggml_blck_size(type));
            const auto bytes = static_cast<std::uint64_t>(ggml_type_size(type));
            if (block > 1 && static_cast<std::uint64_t>(dimensions[0]) % block != 0)
                throw std::runtime_error("GGUF tensor row width is not quantization-block aligned: " + tensor.name);
            if (block == 0 || bytes == 0 || elements % block != 0 || elements / block > UINT64_MAX / bytes)
                throw std::runtime_error("invalid or overflowing GGUF tensor byte count: " + tensor.name);
            tensor.type = static_cast<std::int32_t>(type);
            tensor.size_bytes = elements / block * bytes;
            const auto relative = static_cast<std::uint64_t>(gguf_get_tensor_offset(context_.get(), i));
            const auto padding = (32 - tensor.size_bytes % 32) % 32;
            if (relative != expected_offset || relative > available || tensor.size_bytes > available - relative ||
                padding > available - relative - tensor.size_bytes)
                throw std::runtime_error("overlapping, overflowing or truncated GGUF tensor range: " + tensor.name);
            if (gguf_get_tensor_size(context_.get(), i) != tensor.size_bytes)
                throw std::runtime_error("incompatible GGUF tensor storage: " + tensor.name);
            tensor.offset = data_offset_ + relative;
            if (padding) {
                std::array<char, 31> padding_bytes{};
                read(tensor.offset + tensor.size_bytes, padding_bytes.data(), static_cast<std::size_t>(padding));
                if (std::any_of(padding_bytes.begin(), padding_bytes.begin() + padding,
                                [](char value) { return value != 0; }))
                    throw std::runtime_error("nonzero GGUF tensor padding: " + tensor.name);
            }
            expected_offset = relative + tensor.size_bytes + padding;
            tensors_.push_back(std::move(tensor));
        }
        if (expected_offset != available) throw std::runtime_error("unexpected data after GGUF tensor blob");
        validate_canonical_metadata();
    }

    const gguf_context* metadata() const { return context_.get(); }
    std::uint64_t size() const { return size_; }
    const std::vector<GgufTensor>& tensors() const { return tensors_; }

    void read(std::uint64_t offset, char* output, std::size_t bytes) {
        if (offset > size_ || bytes > size_ - offset ||
            bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error("invalid or truncated GGUF file range");
        stream_.clear();
        stream_.seekg(static_cast<std::streamoff>(offset));
        stream_.read(output, static_cast<std::streamsize>(bytes));
        if (!stream_) throw std::runtime_error("GGUF checkpoint changed or became unreadable");
    }

    std::int64_t key(const char* name, gguf_type type) const {
        const auto id = gguf_find_key(context_.get(), name);
        if (id < 0 || gguf_get_kv_type(context_.get(), id) != type)
            throw std::runtime_error(std::string("missing or incompatible GGUF metadata: ") + name);
        return id;
    }
    std::uint32_t u32(const char* name) const { return gguf_get_val_u32(context_.get(), key(name, GGUF_TYPE_UINT32)); }
    std::string string(const char* name, std::size_t maximum = 2048) const {
        return bounded_string(gguf_get_val_str(context_.get(), key(name, GGUF_TYPE_STRING)), maximum, name);
    }
    std::int64_t array(const char* name, gguf_type type, std::size_t count) const {
        const auto id = key(name, GGUF_TYPE_ARRAY);
        if (gguf_get_arr_type(context_.get(), id) != type || gguf_get_arr_n(context_.get(), id) != count)
            throw std::runtime_error(std::string("incompatible GGUF array: ") + name);
        return id;
    }
    static std::string bounded_string(const char* value, std::size_t maximum, const char* name) {
        std::size_t length = 0;
        while (length <= maximum && value[length]) ++length;
        if (length == 0 || length > maximum)
            throw std::runtime_error(std::string("invalid or oversized GGUF string: ") + name);
        return std::string(value, length);
    }

private:
    static std::size_t read_callback(void* user, void* output, std::uint64_t offset, std::size_t bytes) noexcept {
        auto& reader = *static_cast<GgufReader*>(user);
        try {
            reader.read(offset, static_cast<char*>(output), bytes);
            return bytes;
        } catch (...) { return 0; }
    }

    void validate_canonical_metadata() {
        for (std::int64_t i = 0; i < gguf_get_n_kv(context_.get()); ++i)
            if (gguf_find_key(context_.get(), gguf_get_key(context_.get(), i)) != i)
                throw std::runtime_error("ambiguous GGUF metadata key");
        GgufPtr canonical(gguf_init_empty());
        gguf_set_kv(canonical.get(), context_.get());
        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            ggml_tensor tensor{};
            tensor.type = static_cast<ggml_type>(tensors_[i].type);
            ggml_set_name(&tensor, tensors_[i].name.c_str());
            const auto* ne = gguf_get_tensor_ne(context_.get(), static_cast<std::int64_t>(i));
            std::copy(ne, ne + GGML_MAX_DIMS, tensor.ne);
            tensor.nb[0] = ggml_type_size(tensor.type);
            tensor.nb[1] = tensor.nb[0] * (tensor.ne[0] / ggml_blck_size(tensor.type));
            for (int d = 2; d < GGML_MAX_DIMS; ++d) tensor.nb[d] = tensor.nb[d - 1] * tensor.ne[d - 1];
            gguf_add_tensor(canonical.get(), &tensor);
        }
        const auto bytes = gguf_get_meta_size(canonical.get());
        if (bytes != data_offset_ || bytes > metadata_limit)
            throw std::runtime_error("noncanonical GGUF metadata or tensor dimensions");
        std::vector<char> expected(bytes), actual(bytes);
        gguf_get_meta_data(canonical.get(), expected.data());
        read(0, actual.data(), actual.size());
        if (actual != expected) throw std::runtime_error("noncanonical GGUF metadata, string or tensor name");
    }

    std::ifstream stream_;
    GgufPtr context_;
    std::uint64_t size_ = 0, data_offset_ = 0;
    std::vector<GgufTensor> tensors_;
};

} // namespace sam::internal

#endif // SAM_CPP_SRC_IO_GGUF_READER_HPP
