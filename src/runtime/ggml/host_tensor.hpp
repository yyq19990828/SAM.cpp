#ifndef SAM_CPP_SRC_RUNTIME_GGML_HOST_TENSOR_HPP
#define SAM_CPP_SRC_RUNTIME_GGML_HOST_TENSOR_HPP

#include "graph.hpp"
#include "common/input_validation.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sam::internal {

// An owned, contiguous host snapshot. It retains one representation; callers
// explicitly materialize F32 only for diagnostics or an F32 consumer.
class HostTensor {
public:
    static HostTensor download(ggml_tensor* tensor, RuntimeStats& stats, GgmlRuntime* runtime = nullptr) {
        if (!tensor || !tensor->buffer || !ggml_is_contiguous(tensor) || ggml_nelements(tensor) <= 0 ||
            (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_Q8_0))
            throw std::runtime_error("invalid SAM cached tensor allocation, layout or type");
        HostTensor result;
        result.type_ = tensor->type;
        result.elements_ = static_cast<std::size_t>(ggml_nelements(tensor));
        result.row_elements_ = tensor->ne[0];
        result.payload_.resize(ggml_nbytes(tensor));
        observe_transfer(runtime, tensor, "download", result.payload_.size(), [&] {
            ggml_backend_tensor_get(tensor, result.payload_.data(), 0, result.payload_.size());
        });
        stats.host_download_bytes += result.payload_.size();
        result.validate_scales();
        return result;
    }

    ggml_type type() const { return type_; }
    std::size_t elements() const { return elements_; }
    std::size_t size_bytes() const { return payload_.size(); }
    bool empty() const { return payload_.empty(); }

    void validate_shape(int64_t d0, int64_t d1, int64_t d2, int64_t d3) const {
        std::size_t count = 1;
        for (auto dimension : {d0, d1, d2, d3}) {
            if (dimension <= 0) throw std::invalid_argument("invalid SAM cache input dimension");
            count = checked_product(count, static_cast<std::size_t>(dimension), "cached tensor shape");
        }
        // Flattening spatial dimensions is permitted; changing row layout is not.
        if (empty() || count != elements_ || d0 != row_elements_)
            throw std::invalid_argument("SAM cache input shape or row layout differs");
    }

    void upload_to(ggml_tensor* tensor, RuntimeStats& stats, GgmlRuntime* runtime = nullptr) const {
        if (!tensor || !tensor->buffer || !ggml_is_contiguous(tensor) || tensor->type != type_ ||
            ggml_nbytes(tensor) != payload_.size())
            throw std::runtime_error("invalid SAM cached input allocation, layout or type");
        validate_shape(tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        observe_transfer(runtime, tensor, "upload", payload_.size(), [&] {
            ggml_backend_tensor_set(tensor, payload_.data(), 0, payload_.size());
        });
        stats.host_upload_bytes += payload_.size();
    }

    std::vector<float> as_f32() const {
        if (empty()) throw std::runtime_error("no SAM cached tensor is available");
        std::vector<float> result(elements_);
        if (type_ == GGML_TYPE_F32) {
            std::memcpy(result.data(), payload_.data(), payload_.size());
        } else {
            const auto* traits = ggml_get_type_traits(type_);
            if (!traits || !traits->to_float) throw std::runtime_error("SAM cache type cannot be decoded");
            traits->to_float(payload_.data(), result.data(), static_cast<int64_t>(elements_));
        }
        return result;
    }

private:
    void validate_scales() const {
        // Overflow must not become an apparently valid compact snapshot. F16
        // stores each value directly; Q8_0 stores one half scale per block.
        if (type_ == GGML_TYPE_F32) return;
        const auto stride = type_ == GGML_TYPE_F16 ? sizeof(ggml_fp16_t) : ggml_type_size(type_);
        for (std::size_t offset = 0; offset < payload_.size(); offset += stride) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, payload_.data() + offset, sizeof(bits));
            if ((bits & 0x7c00u) == 0x7c00u || (type_ == GGML_TYPE_Q8_0 && (bits & 0x8000u)))
                throw std::runtime_error("non-finite or invalid SAM compact cache value/scale");
        }
    }

    ggml_type type_ = GGML_TYPE_F32;
    int64_t row_elements_ = 0;
    std::size_t elements_ = 0;
    std::vector<std::uint8_t> payload_;
};

struct CachedInput {
    ggml_tensor* stored = nullptr;
    ggml_tensor* values = nullptr;
};

inline CachedInput cached_input(ggml_context* ctx, const char* name, const HostTensor& cache,
                                int64_t d0, int64_t d1 = 1, int64_t d2 = 1, int64_t d3 = 1) {
    cache.validate_shape(d0, d1, d2, d3);
    auto* stored = input_tensor(ctx, name, d0, d1, d2, d3, cache.type());
    auto* values = cache.type() == GGML_TYPE_F32 ? stored : ggml_cast(ctx, stored, GGML_TYPE_F32);
    return {stored, values};
}

} // namespace sam::internal

#endif // SAM_CPP_SRC_RUNTIME_GGML_HOST_TENSOR_HPP
