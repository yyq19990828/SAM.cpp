#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_POLICY_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_POLICY_HPP

// Ordered forward rules from pinned Meta Sam3VideoBase, SAM model license.
#include "mask_ops.hpp"
#include "../execution.hpp"
#include <map>
#include <set>

namespace sam::internal::sam3 {

struct VideoDetection {
    int query = 0;
    float score = 0;
    std::vector<float> mask;
};

inline std::vector<VideoDetection> video_detections(const Prediction& prediction) {
    constexpr int pixels = 288 * 288;
    std::vector<VideoDetection> candidates;
    for (int query = 0; query < 200; ++query) {
        const float probability = sigmoid(prediction.class_logits.at(query)) * sigmoid(prediction.presence_logit);
        // The video builder supervises the joint score, using inverse_sigmoid
        // with eps=1e-3 and logit clipping before its final sigmoid.
        const float score = sigmoid(std::clamp(std::log(std::max(probability, 0.001f) /
            std::max(1.0f - probability, 0.001f)), -10.0f, 10.0f));
        if (score > 0.5f) candidates.push_back({query, score,
            {prediction.mask_logits.begin() + query * pixels, prediction.mask_logits.begin() + (query + 1) * pixels}});
    }
    std::vector<std::size_t> order(candidates.size()); std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return candidates[a].score > candidates[b].score; });
    std::set<std::size_t> kept;
    for (const auto candidate : order) {
        bool suppressed = false;
        for (const auto previous : kept)
            if (mask_overlap(candidates[candidate].mask, candidates[previous].mask) > 0.1f) { suppressed = true; break; }
        if (!suppressed) kept.insert(candidate);
    }
    std::vector<VideoDetection> output;
    for (const auto index : kept) output.push_back(std::move(candidates[index]));
    return output;
}

struct Association {
    std::vector<std::size_t> births, unmatched, empty;
    std::vector<std::vector<std::size_t>> matches;
    std::map<std::size_t, std::size_t> recondition;
};

inline Association associate_video(const std::vector<VideoDetection>& detections,
                                    const std::vector<std::vector<float>>& masks) {
    Association result;
    result.matches.resize(detections.size());
    if (masks.empty()) {
        // Meta admits every post-NMS >0.5 detection when there are no tracks.
        for (std::size_t i = 0; i < detections.size(); ++i) result.births.push_back(i);
        return result;
    }
    std::vector<std::vector<float>> overlaps(detections.size(), std::vector<float>(masks.size()));
    std::vector<int> track_high(masks.size());
    for (std::size_t d = 0; d < detections.size(); ++d)
        for (std::size_t t = 0; t < masks.size(); ++t) {
            overlaps[d][t] = mask_overlap(detections[d].mask, masks[t]);
            track_high[t] += overlaps[d][t] >= 0.8f;
        }
    for (std::size_t t = 0; t < masks.size(); ++t) {
        if (!mask_nonempty(masks[t])) { result.empty.push_back(t); continue; }
        bool matched = false;
        for (const auto& overlap : overlaps) matched = matched || overlap[t] >= 0.5f;
        if (!matched) result.unmatched.push_back(t);
    }
    for (std::size_t d = 0; d < detections.size(); ++d) {
        bool new_object = detections[d].score >= 0.7f;
        int detection_high = 0;
        for (const auto overlap : overlaps[d]) {
            new_object = new_object && overlap < 0.1f;
            detection_high += overlap >= 0.8f;
        }
        if (new_object) result.births.push_back(d);
        float best = -1; std::size_t best_track = 0;
        for (std::size_t t = 0; t < masks.size(); ++t) {
            const float overlap = detection_high > 1 || track_high[t] > 1 ? 0.0f : overlaps[d][t];
            if (overlap >= 0.1f) result.matches[d].push_back(t);
            if (overlap > best) { best = overlap; best_track = t; }
        }
        if (!new_object && detections[d].score >= 0.8f && best >= 0.8f) result.recondition[best_track] = d;
    }
    return result;
}

struct ObjectPolicy {
    int birth = 0, keep_alive = 30, unmatched = 0, last_occluded = -1;
    float score = 0;
};

struct TemporalPolicy {
    std::map<std::uint64_t, ObjectPolicy> objects;
    std::map<std::pair<std::uint64_t, std::uint64_t>, int> duplicates;

    std::set<std::uint64_t> update(int frame, const std::vector<std::uint64_t>& old_ids,
                                  const Association& association) {
        std::set<std::uint64_t> removed, matched;
        for (const auto& match : association.matches)
            for (const auto index : match) matched.insert(old_ids.at(index));
        for (const auto id : matched) objects.at(id).keep_alive = std::min(30, objects.at(id).keep_alive + 1);
        for (const auto index : association.unmatched) {
            auto& state = objects.at(old_ids.at(index));
            state.unmatched = std::min(8, state.unmatched + 1);
            state.keep_alive = std::max(-1, state.keep_alive - 1);
        }
        for (const auto& [id, state] : objects)
            if (state.birth > frame - 15 && state.unmatched >= 8) removed.insert(id);
        for (const auto& match : association.matches) {
            if (match.size() < 2) continue;
            const auto first_index = *std::min_element(match.begin(), match.end(), [&](auto a, auto b) {
                return objects.at(old_ids[a]).birth < objects.at(old_ids[b]).birth;
            });
            for (const auto index : match) if (index != first_index) {
                auto& count = duplicates[{old_ids[first_index], old_ids[index]}];
                count = std::min(8, count + 1);
            }
        }
        for (const auto& [pair, count] : duplicates)
            if (objects.at(pair.second).birth > frame - 15 && count >= 8) removed.insert(pair.second);
        return removed;
    }

    void retire(const std::set<std::uint64_t>& removed) {
        for (const auto id : removed) objects.erase(id);
        for (auto item = duplicates.begin(); item != duplicates.end(); )
            if (!objects.count(item->first.first) || !objects.count(item->first.second)) item = duplicates.erase(item);
            else ++item;
    }
};

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_POLICY_HPP
