#ifndef SAM_CPP_SRC_MODELS_SAM3_VIDEO_MEMORY_PAYLOAD_HPP
#define SAM_CPP_SRC_MODELS_SAM3_VIDEO_MEMORY_PAYLOAD_HPP

#include "memory_selection.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <vector>

namespace sam::internal::sam3 {

// One object's retained tracker record: BF16 spatial memory plus the F32
// object pointer that conditions the next propagation.
struct TrackerRecord {
    std::vector<ggml_bf16_t> memory;
    std::vector<float> pointer;
    float object_logit = 0, iou = 0;
};

struct TrackerPropagationInput {
    const std::map<int, TrackerRecord>* records = nullptr;
    const MemorySelection* selection = nullptr;
};

// Packed model inputs for a propagation batch. The memory plane holds spatial
// BF16 records converted in place and pointer tokens; the pointer positions
// carry the projected temporal embedding prepared by the pointer graph.
struct TrackerMemoryPayload {
    std::vector<float> memory, pointer_position;
};

inline std::vector<float> tracker_sine_positions(const std::vector<TrackerPropagationInput>& inputs,
                                                 int frame_count) {
    std::vector<float> values;
    const auto pointers = inputs.front().selection->pointers.size();
    if (!pointers) return values;
    const int maximum = std::min(frame_count, 16);
    if (maximum <= 1) throw std::runtime_error("tracker pointer temporal range is invalid");
    values.reserve(inputs.size() * pointers * 256);
    for (const auto& input : inputs) for (const auto& selected : input.selection->pointers) {
        const float relative = static_cast<float>(selected.position) / (maximum - 1);
        for (int phase = 0; phase < 2; ++phase) for (int k = 0; k < 128; ++k) {
            const float angle = relative / std::pow(10000.0f, 2.0f * (k / 2) / 128);
            values.push_back(phase ? std::cos(angle) : std::sin(angle));
        }
    }
    return values;
}

inline std::vector<float> tracker_slice_pointer_positions(const std::vector<float>& source,
        int pointers, std::size_t begin, std::size_t count) {
    if (!pointers) {
        if (!source.empty()) throw std::runtime_error("tracker pointer-position payload is unexpectedly nonempty");
        return {};
    }
    const auto row = static_cast<std::size_t>(pointers) * 64;
    if (source.size() % row || begin > source.size() / row || count > source.size() / row - begin)
        throw std::runtime_error("tracker pointer-position slice is out of range");
    const auto first = begin * row;
    return std::vector<float>(source.begin() + first, source.begin() + first + count * row);
}

// payload_peak_bytes tracks the retained capacity high-water mark that the
// session diagnostics report; the coordinator owns the counter.
inline TrackerMemoryPayload tracker_make_memory_payload(
        const std::vector<TrackerPropagationInput>& inputs,
        const std::vector<float>& projected, std::size_t* payload_peak_bytes) {
    const auto spatial_count = inputs.front().selection->spatial.size();
    const auto pointer_count = inputs.front().selection->pointers.size();
    const auto batch = inputs.size();
    const auto tokens = spatial_count * 5184 + pointer_count * 4;
    TrackerMemoryPayload result;
    result.memory.reserve(batch * tokens * 64);
    result.pointer_position.reserve(batch * pointer_count * 4 * 64);
    for (std::size_t b = 0; b < batch; ++b) {
        const auto& input = inputs[b];
        for (const auto& selected : input.selection->spatial) {
            const auto& record = input.records->at(selected.frame);
            for (const auto value : record.memory) result.memory.push_back(ggml_bf16_to_fp32(value));
        }
        for (std::size_t i = 0; i < input.selection->pointers.size(); ++i) {
            const auto& selected = input.selection->pointers[i];
            const auto& pointer = input.records->at(selected.frame).pointer;
            result.memory.insert(result.memory.end(), pointer.begin(), pointer.end());
            const auto projected_offset = (b * pointer_count + i) * 64;
            for (int token = 0; token < 4; ++token)
                result.pointer_position.insert(result.pointer_position.end(),
                    projected.begin() + projected_offset, projected.begin() + projected_offset + 64);
        }
    }
    if (payload_peak_bytes) {
        const auto bytes = (result.memory.capacity() + result.pointer_position.capacity()) * sizeof(float);
        *payload_peak_bytes = std::max(*payload_peak_bytes, bytes);
    }
    return result;
}

inline std::vector<float> tracker_memory_position_payload(
        const std::vector<TrackerPropagationInput>& inputs, const TrackerMemoryPayload& payload,
        const std::vector<float>& position_bank) {
    if (inputs.empty()) return {};
    const auto spatial_count = inputs.front().selection->spatial.size();
    const auto pointer_count = inputs.front().selection->pointers.size();
    const auto batch = inputs.size();
    constexpr std::size_t memory_plane_size = 64u * 5184;
    const auto pointer_row = pointer_count * 4 * 64;
    std::vector<float> values;
    values.reserve(batch * (spatial_count * memory_plane_size + pointer_row));
    for (std::size_t b = 0; b < batch; ++b) {
        for (const auto& selected : inputs[b].selection->spatial) {
            if (selected.position < 0 || selected.position > 6)
                throw std::runtime_error("tracker memory position is outside its resident table");
            const auto position = static_cast<std::size_t>(6 - selected.position);
            const auto first = position_bank.begin() + position * memory_plane_size;
            values.insert(values.end(), first, first + memory_plane_size);
        }
        if (pointer_row) {
            const auto first = payload.pointer_position.begin() + b * pointer_row;
            values.insert(values.end(), first, first + pointer_row);
        }
    }
    return values;
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_VIDEO_MEMORY_PAYLOAD_HPP
