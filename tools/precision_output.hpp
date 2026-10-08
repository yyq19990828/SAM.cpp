#ifndef SAM_CPP_TOOLS_PRECISION_OUTPUT_HPP
#define SAM_CPP_TOOLS_PRECISION_OUTPUT_HPP

#include <models/sam3/image_ops.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace sam_probe {

inline std::vector<int> ranked_queries(const std::vector<float>& scores, std::size_t count = 100) {
    if (scores.size() != 200 || count > scores.size() ||
        !std::all_of(scores.begin(), scores.end(), [](float value) { return std::isfinite(value) && value >= 0 && value <= 1; }))
        throw std::invalid_argument("invalid ranked query scores");
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    indices.resize(count);
    return indices;
}

// COCO's compressed RLE uses column-major binary runs and signed five-bit
// deltas to the run two positions earlier. Only one full-size mask is needed.
inline std::string compressed_rle(const sam::Mask& mask) {
    if (mask.width <= 0 || mask.height <= 0 || mask.data.size() !=
        sam::internal::checked_product(mask.width, mask.height, "RLE mask"))
        throw std::invalid_argument("invalid RLE mask dimensions");
    std::vector<std::int64_t> runs;
    std::int64_t count = 0;
    std::uint8_t previous = 0;
    for (int x = 0; x < mask.width; ++x) {
        for (int y = 0; y < mask.height; ++y) {
            const auto value = mask.data[static_cast<std::size_t>(y) * mask.width + x];
            if (value > 1) throw std::invalid_argument("RLE requires a binary mask");
            if (value == previous) ++count;
            else { runs.push_back(count); count = 1; previous = value; }
        }
    }
    runs.push_back(count);
    std::string result;
    for (std::size_t index = 0; index < runs.size(); ++index) {
        std::int64_t value = runs[index] - (index > 2 ? runs[index - 2] : 0);
        bool more;
        do {
            auto byte = static_cast<unsigned>(static_cast<std::uint64_t>(value) & 31U);
            value = value >= 0 ? value / 32 : (value - 31) / 32;
            more = (byte & 16U) ? value != -1 : value != 0;
            if (more) byte |= 32U;
            result.push_back(static_cast<char>(byte + 48U));
        } while (more);
    }
    return result;
}

inline sam::Mask query_mask(const std::vector<float>& logits, int query, int mask_width, int mask_height,
                            int width, int height) {
    if (mask_width <= 0 || mask_height <= 0 || width <= 0 || height <= 0)
        throw std::invalid_argument("query mask dimensions must be positive");
    const auto size = sam::internal::checked_product(mask_width, mask_height, "query mask");
    if (query < 0 || static_cast<std::size_t>(query) >= logits.size() / size || logits.size() % size)
        throw std::invalid_argument("query mask index or shape differs");
    const auto first = logits.begin() + static_cast<std::size_t>(query) * size;
    std::vector<float> one(first, first + size);
    // Select one mask with synthetic finite logits, then use the production
    // resize/threshold implementation. Actual query scores are exported apart.
    auto result = sam::internal::sam3::postprocess_detections({0, 0, 0, 0}, {0}, 0, one,
                                                              mask_width, mask_height, width, height, 0);
    return std::move(result.detections.front().mask);
}

} // namespace sam_probe

#endif // SAM_CPP_TOOLS_PRECISION_OUTPUT_HPP
