#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MASK_OPS_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MASK_OPS_HPP

#include "../image_ops.hpp"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace sam::internal::sam3 {

// align_corners=false, planar single-channel input. Reuse the image triangle
// filter for Torch's antialiased path without its RGB byte rounding.
inline std::vector<float> resize_mask(const std::vector<float>& input, int width, int height,
                                      int target_width, int target_height, bool antialias = false) {
    if (width <= 0 || height <= 0 || target_width <= 0 || target_height <= 0 ||
        input.size() != checked_product(width, height, "mask input"))
        throw std::invalid_argument("invalid mask resize dimensions");
    if (width == target_width && height == target_height) return input;
    const auto count = checked_product(target_width, target_height, "resized mask");
    std::vector<float> output(count);
    if (antialias) {
        const auto x_axis = resize_axis(width, target_width), y_axis = resize_axis(height, target_height);
        std::vector<float> temporary(checked_product(target_width, height, "mask resize scratch"));
        for (int y = 0; y < height; ++y) for (int x = 0; x < target_width; ++x)
            for (std::size_t k = 0; k < x_axis.weights[x].size(); ++k)
                temporary[static_cast<std::size_t>(y) * target_width + x] += x_axis.weights[x][k] *
                    input[static_cast<std::size_t>(y) * width + x_axis.first[x] + k];
        for (int y = 0; y < target_height; ++y) for (int x = 0; x < target_width; ++x)
            for (std::size_t k = 0; k < y_axis.weights[y].size(); ++k)
                output[static_cast<std::size_t>(y) * target_width + x] += y_axis.weights[y][k] *
                    temporary[(static_cast<std::size_t>(y_axis.first[y]) + k) * target_width + x];
    } else {
        for (int y = 0; y < target_height; ++y) {
            const float fy = std::max(0.0f, (y + 0.5f) * height / target_height - 0.5f);
            const int y0 = std::min(static_cast<int>(fy), height - 1), y1 = std::min(y0 + 1, height - 1);
            const float wy = fy - y0;
            for (int x = 0; x < target_width; ++x) {
                const float fx = std::max(0.0f, (x + 0.5f) * width / target_width - 0.5f);
                const int x0 = std::min(static_cast<int>(fx), width - 1), x1 = std::min(x0 + 1, width - 1);
                const float wx = fx - x0;
                output[static_cast<std::size_t>(y) * target_width + x] =
                    (1 - wy) * ((1 - wx) * input[static_cast<std::size_t>(y0) * width + x0] +
                                wx * input[static_cast<std::size_t>(y0) * width + x1]) +
                    wy * ((1 - wx) * input[static_cast<std::size_t>(y1) * width + x0] +
                          wx * input[static_cast<std::size_t>(y1) * width + x1]);
            }
        }
    }
    return output;
}

inline bool mask_nonempty(const std::vector<float>& mask) {
    return std::any_of(mask.begin(), mask.end(), [](float value) { return value > 0; });
}

inline float mask_overlap(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("mask IoU dimensions differ");
    std::size_t intersection = 0, total = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        intersection += a[i] > 0 && b[i] > 0;
        total += a[i] > 0 || b[i] > 0;
    }
    return total ? static_cast<float>(intersection) / total : 0.0f;
}

// Meta's CPU fallback uses 8-connectivity, including boundary components.
// Padding to even dimensions adds only zeros and does not change these labels.
inline void clean_mask_components(std::vector<float>& mask, int width, int height, int area = 16) {
    if (width <= 0 || height <= 0 || mask.size() != checked_product(width, height, "component mask"))
        throw std::invalid_argument("invalid component mask dimensions");
    if (area <= 0) return;
    for (const bool foreground : {false, true}) {
        const auto foreground_count = std::count_if(mask.begin(), mask.end(), [](float value) { return value > 0; });
        const auto limit = foreground ? std::min<std::size_t>(area, foreground_count / 2) : static_cast<std::size_t>(area);
        std::vector<std::uint8_t> visited(mask.size());
        std::vector<std::size_t> component;
        for (std::size_t start = 0; start < mask.size(); ++start) {
            if (visited[start] || (mask[start] > 0) != foreground) continue;
            component.clear(); component.push_back(start); visited[start] = 1;
            for (std::size_t cursor = 0; cursor < component.size(); ++cursor) {
                const auto pixel = component[cursor];
                const int x = static_cast<int>(pixel % width), y = static_cast<int>(pixel / width);
                for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                    const auto next = static_cast<std::size_t>(ny) * width + nx;
                    if (!visited[next] && (mask[next] > 0) == foreground) {
                        visited[next] = 1; component.push_back(next);
                    }
                }
            }
            if (component.size() <= limit)
                for (const auto pixel : component) mask[pixel] = foreground ? -0.1f : 0.1f;
        }
    }
}

inline std::vector<std::vector<float>> pixel_nonoverlap(const std::vector<std::vector<float>>& masks) {
    auto output = masks;
    if (masks.size() < 2) return output;
    for (const auto& mask : masks) if (mask.size() != masks.front().size())
        throw std::invalid_argument("non-overlap mask dimensions differ");
    for (std::size_t pixel = 0; pixel < masks.front().size(); ++pixel) {
        std::size_t winner = 0;
        for (std::size_t i = 1; i < masks.size(); ++i)
            if (masks[i][pixel] > masks[winner][pixel]) winner = i;
        for (std::size_t i = 0; i < masks.size(); ++i)
            if (i != winner) output[i][pixel] = std::min(output[i][pixel], -10.0f);
    }
    return output;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MASK_OPS_HPP
