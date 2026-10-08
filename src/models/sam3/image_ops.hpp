#ifndef SAM_CPP_SRC_MODELS_SAM3_IMAGE_OPS_HPP
#define SAM_CPP_SRC_MODELS_SAM3_IMAGE_OPS_HPP

#include "sam/types.hpp"
#include "common/input_validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam::internal::sam3 {

struct ResizeAxis {
    std::vector<int> first;
    std::vector<std::vector<float>> weights;
};

// Match torchvision's ARM/GPU path: cast RGB8 to float for antialiased triangle
// filtering, then round the final image to RGB8 (nearest even). Intermediate
// byte rounding would change up to one byte step in both resize directions.
inline ResizeAxis resize_axis(int source_size, int target_size) {
    ResizeAxis axis;
    axis.first.resize(target_size);
    axis.weights.resize(target_size);
    const float scale = static_cast<float>(source_size) / target_size;
    const float support = std::max(1.0f, scale);
    const float inverse_scale = scale >= 1.0f ? 1.0f / scale : 1.0f;
    for (int i = 0; i < target_size; ++i) {
        const int first = static_cast<int>(std::max<std::int64_t>(0,
            static_cast<std::int64_t>(std::fma(scale, i + 0.5f, -support) + 0.5f)));
        const int end = static_cast<int>(std::min<std::int64_t>(source_size,
            static_cast<std::int64_t>(std::fma(scale, i + 0.5f, support) + 0.5f)));
        axis.first[i] = first;
        auto& weights = axis.weights[i];
        float sum = 0.0f;
        // CUDA contracts this subtraction with the scale multiplication. Keep
        // the relative coordinate precise even on hosts without contraction:
        // rounding an absolute center first can change the final RGB8 byte.
        const float relative_center = std::fma(-scale, i + 0.5f, static_cast<float>(first));
        for (int source = first; source < end; ++source) {
            const float weight = std::max(0.0f,
                1.0f - std::abs((source - first + relative_center + 0.5f) * inverse_scale));
            weights.push_back(weight);
            sum += weight;
        }
        for (float& weight : weights) {
            weight /= sum;
        }
    }
    return axis;
}

inline float round_nearest_even_byte(float value) {
    const float base = std::floor(value);
    const float fraction = value - base;
    const bool increment = fraction > 0.5f || (fraction == 0.5f && static_cast<int>(base) % 2 != 0);
    return std::clamp(base + (increment ? 1.0f : 0.0f), 0.0f, 255.0f);
}

inline std::vector<float> preprocess_image(const ImageView& image, int image_size = 1008) {
    validate_image(image);
    if (image_size <= 0) {
        throw std::invalid_argument("target image size must be positive");
    }
    const auto pixel_count = checked_product(static_cast<std::size_t>(image_size), image_size,
                                             "preprocessed image");
    std::vector<float> result(checked_product(pixel_count, 3, "preprocessed RGB image"));
    const auto horizontal = resize_axis(image.width, image_size);
    const auto vertical = resize_axis(image.height, image_size);
    std::vector<float> temporary(checked_product(
        checked_product(static_cast<std::size_t>(image.height), image_size, "resize buffer"),
        3, "resize RGB buffer"));
    for (int y = 0; y < image.height; ++y) {
        const auto* source = image.data + static_cast<std::size_t>(y) * image.row_stride;
        for (int x = 0; x < image_size; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                float value = 0.0f;
                for (std::size_t k = 0; k < horizontal.weights[x].size(); ++k) {
                    value = std::fma(horizontal.weights[x][k],
                        static_cast<float>(source[(static_cast<std::size_t>(horizontal.first[x]) + k) * 3 + channel]), value);
                }
                temporary[(static_cast<std::size_t>(y) * image_size + x) * 3 + channel] = value;
            }
        }
    }
    for (int y = 0; y < image_size; ++y) {
        for (int x = 0; x < image_size; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                float value = 0.0f;
                for (std::size_t k = 0; k < vertical.weights[y].size(); ++k) {
                    value = std::fma(vertical.weights[y][k], temporary[
                        ((static_cast<std::size_t>(vertical.first[y]) + k) * image_size + x) * 3 + channel], value);
                }
                const auto byte = round_nearest_even_byte(value);
                // Keep torchvision's dtype conversion and Normalize operations
                // separate so contraction cannot fuse the reciprocal multiply.
                const float scaled = byte * (1.0f / 255.0f);
                const float centered = scaled - 0.5f;
                result[static_cast<std::size_t>(channel) * pixel_count +
                       static_cast<std::size_t>(y) * image_size + x] =
                    centered / 0.5f;
            }
        }
    }
    return result;
}

inline float sigmoid(float value) {
    return value >= 0.0f ? 1.0f / (1.0f + std::exp(-value)) :
                          std::exp(value) / (1.0f + std::exp(value));
}

inline Result postprocess_detections(const std::vector<float>& boxes,
                                    const std::vector<float>& class_logits,
                                    float presence_logit,
                                    const std::vector<float>& mask_logits,
                                    int mask_width, int mask_height,
                                    int image_width, int image_height,
                                    float threshold) {
    validate_score_threshold(threshold);
    if (mask_width <= 0 || mask_height <= 0 || image_width <= 0 || image_height <= 0) {
        throw std::runtime_error("invalid detection output dimensions");
    }
    const auto queries = class_logits.size();
    if (queries > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("detection query count exceeds the result index range");
    }
    const auto mask_elements = checked_product(mask_width, mask_height, "mask logits");
    if (boxes.size() != checked_product(queries, 4, "detection boxes") ||
        mask_logits.size() != checked_product(queries, mask_elements, "detection mask logits")) {
        throw std::runtime_error("detection output tensor size mismatch");
    }
    auto finite = [](float value) { return std::isfinite(value); };
    if (!std::isfinite(presence_logit) || !std::all_of(boxes.begin(), boxes.end(), finite) ||
        !std::all_of(class_logits.begin(), class_logits.end(), finite) ||
        !std::all_of(mask_logits.begin(), mask_logits.end(), finite)) {
        throw std::runtime_error("model returned non-finite detection outputs");
    }
    Result result;
    for (std::size_t query = 0; query < queries; ++query) {
        const float score = sigmoid(class_logits[query]) * sigmoid(presence_logit);
        if (score <= threshold) {
            continue;
        }
        const auto* box = boxes.data() + query * 4;
        Detection detection;
        detection.score = score;
        detection.query_index = static_cast<int>(query);
        detection.box = {(box[0] - box[2] * 0.5f) * image_width,
                         (box[1] - box[3] * 0.5f) * image_height,
                         (box[0] + box[2] * 0.5f) * image_width,
                         (box[1] + box[3] * 0.5f) * image_height};
        detection.mask.width = image_width;
        detection.mask.height = image_height;
        detection.mask.data.resize(checked_product(image_width, image_height, "output mask"));
        const auto* source = mask_logits.data() + query * mask_elements;
        for (int y = 0; y < image_height; ++y) {
            const float fy = std::max(0.0f, (y + 0.5f) * mask_height / image_height - 0.5f);
            const int y0 = std::min(static_cast<int>(fy), mask_height - 1);
            const int y1 = std::min(y0 + 1, mask_height - 1);
            const float wy = fy - y0;
            for (int x = 0; x < image_width; ++x) {
                const float fx = std::max(0.0f, (x + 0.5f) * mask_width / image_width - 0.5f);
                const int x0 = std::min(static_cast<int>(fx), mask_width - 1);
                const int x1 = std::min(x0 + 1, mask_width - 1);
                const float wx = fx - x0;
                const auto row0 = static_cast<std::size_t>(y0) * mask_width;
                const auto row1 = static_cast<std::size_t>(y1) * mask_width;
                const float logit = (1.0f - wy) * ((1.0f - wx) * source[row0 + x0] +
                                                  wx * source[row0 + x1]) +
                                    wy * ((1.0f - wx) * source[row1 + x0] +
                                          wx * source[row1 + x1]);
                detection.mask.data[static_cast<std::size_t>(y) * image_width + x] = logit > 0.0f ? 1 : 0;
            }
        }
        result.detections.push_back(std::move(detection));
    }
    return result;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_IMAGE_OPS_HPP
