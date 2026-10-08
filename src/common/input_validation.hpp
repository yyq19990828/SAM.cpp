#ifndef SAM_CPP_SRC_COMMON_INPUT_VALIDATION_HPP
#define SAM_CPP_SRC_COMMON_INPUT_VALIDATION_HPP

#include "sam/types.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace sam::internal {

inline std::size_t checked_product(std::size_t a, std::size_t b, const char* what) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::invalid_argument(std::string(what) + " overflows size_t");
    }
    return a * b;
}

inline void validate_backend_options(const BackendOptions& options) {
    if (options.backend != Backend::Auto && options.backend != Backend::Cpu &&
        options.backend != Backend::Metal && options.backend != Backend::Cuda) {
        throw std::invalid_argument("unknown SAM backend");
    }
    if (options.threads <= 0) {
        throw std::invalid_argument("CPU thread count must be positive");
    }
    if (options.cuda_device < 0 || (options.backend != Backend::Cuda && options.cuda_device != 0)) {
        throw std::invalid_argument("CUDA device must be nonnegative and requires an explicit CUDA backend");
    }
    if (options.cuda_compute != CudaComputeMode::F32 && options.cuda_compute != CudaComputeMode::F16)
        throw std::invalid_argument("unknown CUDA compute mode");
    if (options.cuda_compute != CudaComputeMode::F32 && options.backend != Backend::Cuda)
        throw std::invalid_argument("reduced CUDA compute requires an explicit CUDA backend");
}

inline void validate_score_threshold(float threshold) {
    if (!std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("score threshold must be finite and in [0, 1]");
    }
}

inline void validate_image(const ImageView& image) {
    if (image.width <= 0 || image.height <= 0 || image.data == nullptr) {
        throw std::invalid_argument("image requires RGB data and positive dimensions");
    }
    const auto row_bytes = checked_product(static_cast<std::size_t>(image.width), 3, "RGB row");
    if (image.row_stride < row_bytes) {
        throw std::invalid_argument("RGB row stride is smaller than width * 3");
    }
    const auto last_row = checked_product(static_cast<std::size_t>(image.height - 1),
                                          image.row_stride, "RGB buffer");
    if (last_row > std::numeric_limits<std::size_t>::max() - row_bytes ||
        image.size_bytes < last_row + row_bytes) {
        throw std::invalid_argument("RGB buffer is too small or its extent overflows size_t");
    }
}

} // namespace sam::internal

#endif // SAM_CPP_SRC_COMMON_INPUT_VALIDATION_HPP
