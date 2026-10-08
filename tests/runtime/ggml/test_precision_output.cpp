#include "../../../tools/validation/precision_output.hpp"
#include "../../../tools/validation/precision_output.hpp"

#include <iostream>
#include <limits>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function&& function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid export input was accepted");
}
}

int main() {
    try {
        // Bytes independently generated with pinned pycocotools 2.0.10.
        sam::Mask mask;
        mask.width = 4;
        mask.height = 3;
        mask.data.assign(12, 0);
        require(sam_probe::compressed_rle(mask) == "<", "zero RLE differs");
        mask.data.assign(12, 1);
        require(sam_probe::compressed_rle(mask) == "0<", "foreground RLE differs");
        mask.data = {0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1};
        require(sam_probe::compressed_rle(mask) == "11110O0010", "column-major RLE differs");
        mask.data[0] = 2;
        rejects([&] { sam_probe::compressed_rle(mask); });
        std::vector<float> scores(200, 0);
        scores[150] = .9f;
        scores[40] = .9f;
        const auto ranked = sam_probe::ranked_queries(scores);
        require(ranked.size() == 100 && ranked[0] == 40 && ranked[1] == 150 && ranked[2] == 0,
                "ranked export lost stable ties or low-score candidates");
        scores[2] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { sam_probe::ranked_queries(scores); });
        const std::vector<float> logits = {-1, 1, 1, -1, 2, -2, 0, 3};
        const auto selected = sam_probe::query_mask(logits, 1, 2, 2, 2, 2);
        require(selected.data == std::vector<std::uint8_t>({1, 0, 0, 1}), "query selection or mask threshold differs");
        const auto resized = sam_probe::query_mask(logits, 0, 2, 2, 4, 4);
        const auto production = sam::internal::sam3::postprocess_detections({0, 0, 0, 0}, {0}, 0,
                                                                          {-1, 1, 1, -1}, 2, 2, 4, 4, 0);
        require(resized.data == production.detections.front().mask.data, "ranked resize differs from deployed resize");
        rejects([&] { sam_probe::query_mask(logits, 0, 0, 2, 4, 4); });
        rejects([&] { sam_probe::query_mask(logits, 2, 2, 2, 4, 4); });
        std::cout << "precision export checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
