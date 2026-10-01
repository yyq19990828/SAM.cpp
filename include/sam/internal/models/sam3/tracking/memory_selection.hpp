#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MEMORY_SELECTION_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MEMORY_SELECTION_HPP

// Forward selectors follow Meta SAM 3 at 2345a4ad109ac29c569da749c91d84f10dc08c40.
// See licenses/SAM-model-license.txt. No reverse or interactive-edit policy.
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace sam::internal::sam3 {

struct MemoryFrame {
    int frame = 0;
    bool conditioning = false;
    float group_quality = 0.0f;
};

struct SelectedMemory {
    int frame = 0;
    int position = 0;
};

struct MemorySelection {
    std::vector<SelectedMemory> spatial, pointers;
};

// Call before processing current_frame. discarded_conditioning counts old
// conditioning records already pruned. It preserves the official branch and
// token order when the retained dictionary alone contains at most four entries.
inline MemorySelection select_memory(const std::map<int, MemoryFrame>& history, int current_frame,
                                     int frame_count, int discarded_conditioning = 0) {
    if (current_frame < 1 || current_frame >= frame_count || discarded_conditioning < 0)
        throw std::invalid_argument("invalid forward memory-selection frame");
    std::vector<int> conditioning;
    for (const auto& item : history) {
        if (item.first != item.second.frame || item.first < 0 || item.first >= current_frame || !std::isfinite(item.second.group_quality))
            throw std::invalid_argument("memory history is outside the accepted forward prefix");
        if (item.second.conditioning) conditioning.push_back(item.first);
    }
    if (conditioning.size() + static_cast<std::size_t>(discarded_conditioning) > 4) {
        std::reverse(conditioning.begin(), conditioning.end());
        if (conditioning.size() > 4) conditioning.resize(4);
    }
    std::set<int> selected(conditioning.begin(), conditioning.end());
    MemorySelection output;
    for (const auto frame : conditioning) {
        output.spatial.push_back({frame, 0});
        output.pointers.push_back({frame, current_frame - frame});
    }
    const int maximum = std::min(frame_count, 16);
    std::vector<int> valid;
    for (auto item = history.rbegin(); item != history.rend(); ++item) {
        if (item->first == 0 || item->second.conditioning || item->second.group_quality <= 0.01f) continue;
        valid.push_back(item->first);
        if (static_cast<int>(valid.size()) >= maximum - 1) break;
    }
    std::reverse(valid.begin(), valid.end());
    if (std::find(valid.begin(), valid.end(), current_frame - 1) == valid.end())
        valid.push_back(current_frame - 1);
    auto available = [&](int frame) {
        return history.count(frame) && !selected.count(frame);
    };
    for (int position = 1; position < 7; ++position) {
        const int relative = 7 - position;
        if (relative > static_cast<int>(valid.size())) continue;
        const int frame = valid[valid.size() - relative];
        if (available(frame)) output.spatial.push_back({frame, position});
    }
    // The strict inequality is deliberate: the official pointer selector
    // excludes the earliest valid entry, even when fewer than 16 are available.
    for (int relative = 1; relative < maximum && relative < static_cast<int>(valid.size()); ++relative) {
        const int frame = valid[valid.size() - relative];
        if (available(frame)) output.pointers.push_back({frame, relative});
    }
    return output;
}

// After processing a frame, preserve all mutable hotstart group history. Once
// the group is fixed, retain four conditioning records, the eight-position
// correction window, and 15 older quality-eligible non-conditioning records.
// The caller owns tensor records and keeps the discarded conditioning count.
inline std::set<int> retained_memory_frames(const std::map<int, MemoryFrame>& history,
                                          int current_frame, int birth_frame) {
    if (birth_frame < 0 || current_frame < birth_frame)
        throw std::invalid_argument("invalid memory-retention lifetime");
    std::set<int> keep;
    int conditioning = 0, older_eligible = 0;
    for (auto item = history.rbegin(); item != history.rend(); ++item) {
        if (item->first != item->second.frame || item->first < birth_frame || item->first > current_frame || !std::isfinite(item->second.group_quality))
            throw std::invalid_argument("memory record is outside its group lifetime");
        if (current_frame - birth_frame < 15) { keep.insert(item->first); continue; }
        if (item->second.conditioning) {
            if (conditioning++ < 4) keep.insert(item->first);
        } else if (item->first >= current_frame - 7) {
            keep.insert(item->first);
        } else if (item->first > 0 && item->second.group_quality > 0.01f && older_eligible++ < 15) {
            keep.insert(item->first);
        }
    }
    return keep;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_MEMORY_SELECTION_HPP
