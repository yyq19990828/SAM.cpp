#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_PREPROCESSING_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_PREPROCESSING_HPP

#include "sam/types.hpp"
#include "sam/internal/input_validation.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sam::internal::sam3 {

struct BicubicAxis {
    std::vector<int> first;
    std::vector<std::vector<std::int32_t>> coefficients;
};

inline double bicubic_kernel(double distance) {
    const double x = std::abs(distance);
    if (x < 1.0) return ((1.5 * x - 2.5) * x) * x + 1.0;
    if (x < 2.0) return (((-0.5 * x + 2.5) * x - 4.0) * x) + 2.0;
    return 0.0;
}

// Pillow's RGB8 route rounds/clips after each separable pass using signed
// 22-bit fixed-point filter coefficients. Floating intermediate rows differ.
inline BicubicAxis bicubic_axis(int source, int target) {
    if (source <= 0 || target <= 0) throw std::invalid_argument("invalid bicubic resize dimensions");
    BicubicAxis axis;
    axis.first.resize(target);
    axis.coefficients.resize(target);
    const double scale = static_cast<double>(source) / target;
    const double filter_scale = std::max(1.0, scale), support = 2.0 * filter_scale;
    for (int i = 0; i < target; ++i) {
        const double center = (i + 0.5) * scale;
        const int first = static_cast<int>(std::clamp(center - support + 0.5, 0.0, static_cast<double>(source)));
        const int end = static_cast<int>(std::clamp(center + support + 0.5, 0.0, static_cast<double>(source)));
        axis.first[i] = first;
        std::vector<double> weights;
        double sum = 0.0;
        for (int pixel = first; pixel < end; ++pixel) {
            const auto weight = bicubic_kernel((pixel - center + 0.5) / filter_scale);
            weights.push_back(weight);
            sum += weight;
        }
        for (const auto weight : weights) {
            const auto normalized = weight / sum;
            axis.coefficients[i].push_back(static_cast<std::int32_t>(
                normalized * (1 << 22) + (normalized < 0.0 ? -0.5 : 0.5)));
        }
    }
    return axis;
}

inline std::uint8_t bicubic_byte(std::int64_t accumulator) {
    // Division is spelled explicitly to preserve arithmetic right-shift
    // semantics for negative values under C++17.
    const auto rounded = accumulator >= 0 ? accumulator / (1 << 22) :
        -((-accumulator + (1 << 22) - 1) / (1 << 22));
    return static_cast<std::uint8_t>(std::clamp<std::int64_t>(rounded, 0, 255));
}

inline std::vector<std::uint8_t> resize_video_rgb(const ImageView& image, int target_width, int target_height) {
    validate_image(image);
    const auto horizontal = bicubic_axis(image.width, target_width);
    const auto vertical = bicubic_axis(image.height, target_height);
    const auto row_bytes = checked_product(target_width, 3, "video resize row");
    std::vector<std::uint8_t> rows(checked_product(image.height, row_bytes, "video resize intermediate"));
    for (int y = 0; y < image.height; ++y) {
        const auto* source = image.data + static_cast<std::size_t>(y) * image.row_stride;
        for (int x = 0; x < target_width; ++x) for (int channel = 0; channel < 3; ++channel) {
            std::int64_t sum = 1 << 21;
            for (std::size_t k = 0; k < horizontal.coefficients[x].size(); ++k)
                sum += static_cast<std::int64_t>(horizontal.coefficients[x][k]) *
                    source[(horizontal.first[x] + k) * 3 + channel];
            rows[static_cast<std::size_t>(y) * row_bytes + x * 3 + channel] = bicubic_byte(sum);
        }
    }
    std::vector<std::uint8_t> output(checked_product(target_height, row_bytes, "video resized image"));
    for (int y = 0; y < target_height; ++y) for (int x = 0; x < target_width; ++x)
        for (int channel = 0; channel < 3; ++channel) {
            std::int64_t sum = 1 << 21;
            for (std::size_t k = 0; k < vertical.coefficients[y].size(); ++k)
                sum += static_cast<std::int64_t>(vertical.coefficients[y][k]) *
                    rows[(vertical.first[y] + k) * row_bytes + x * 3 + channel];
            output[static_cast<std::size_t>(y) * row_bytes + x * 3 + channel] = bicubic_byte(sum);
        }
    return output;
}

inline float round_f16(float value) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)); }
inline float round_bf16(float value) { return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value)); }

inline std::vector<float> preprocess_video_frame(const ImageView& image, int size = 1008) {
    const auto pixels = resize_video_rgb(image, size, size);
    const auto count = checked_product(size, size, "video normalized pixels");
    std::vector<float> output(pixels.size());
    for (std::size_t pixel = 0; pixel < count; ++pixel) for (int channel = 0; channel < 3; ++channel) {
        const auto scaled = round_f16(static_cast<float>(pixels[pixel * 3 + channel] / 255.0));
        const auto centered = round_f16(scaled - 0.5f);
        output[channel * count + pixel] = round_f16(centered / 0.5f);
    }
    return output;
}

inline void round_bf16_storage(std::vector<float>& values) {
    for (auto& value : values) value = round_bf16(value);
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_PREPROCESSING_HPP
