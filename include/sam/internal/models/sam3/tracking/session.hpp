#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_SESSION_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_SESSION_HPP

#include "execution.hpp"
#include "policy.hpp"
#include "sam/internal/model_interface.hpp"
#include <deque>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>

namespace sam::internal::sam3 {

struct TrackerObject {
    std::uint64_t id = 0;
    std::map<int, TrackerRecord> records;
};

struct TrackerGroup {
    int birth = 0, discarded_conditioning = 0;
    std::vector<TrackerObject> objects;
    std::map<int, MemoryFrame> history;
    std::map<int, std::set<std::uint64_t>> inputs;
    std::set<int> consolidated_frames;
};

struct PendingObject {
    std::uint64_t id = 0;
    float score = 0, tracker_score = 0;
    std::vector<float> mask;
};

struct PendingFrame {
    int frame = 0;
    std::vector<PendingObject> objects;
};

class VideoSession final : public sam::internal::TextVideoSessionImplementation {
public:
    VideoSession(std::shared_ptr<ModelState> model, int frame_count, VideoOptions options)
        : model_(std::move(model)), tokenizer_(model_->tokenizer), options_(options) {
        if (model_->model_info.task != "text_video") throw std::runtime_error("video session requires full video GGUF weights");
        if (options.max_objects <= 0) throw std::invalid_argument("max_objects must be positive");
        reset(frame_count);
    }

    void reset(int frame_count) override {
        if (frame_count <= 0) throw std::invalid_argument("frame_count must be positive");
        frame_count_ = frame_count; next_frame_ = 0; next_id_ = 0;
        width_ = height_ = 0; failed_ = false;
        tokens_.clear(); text_.clear(); groups_.clear(); policy_ = {}; pending_.clear();
        image_ = {}; prediction_ = {}; debug_.clear(); snapshots_.clear(); trace_ = "{}";
        stats_ = {}; stats_.runtime.weight_buffer_bytes = ggml_backend_buffer_get_size(model_->buffer.get());
    }

    void set_text(std::string_view prompt) override {
        set_tokens(tokenizer_.encode(std::string(prompt)));
    }

    void set_tokens(const std::vector<std::int32_t>& tokens) override {
        tokenizer_.validate_tokens(tokens);
        if (failed_) throw std::runtime_error("failed video session requires reset");
        if (next_frame_ != 0) throw std::runtime_error("video prompt must be set before the first frame");
        tokens_ = tokens;
    }

    std::vector<VideoFrameResult> push_frame(int frame_index, ImageView image) override {
        validate_image(image);
        if (frame_index != next_frame_ || frame_index < 0 || frame_index >= frame_count_)
            throw std::invalid_argument("video frames must be pushed exactly once in order");
        if (width_ && (image.width != width_ || image.height != height_))
            throw std::invalid_argument("video frame resolution changed");
        if (failed_) throw std::runtime_error("failed video session requires reset");
        if (tokens_.empty()) throw std::runtime_error("video prompt is not set");
        const auto start = std::chrono::steady_clock::now();
        auto pixels = preprocess_video_frame(image);
        std::lock_guard<std::mutex> lock(model_->execution_mutex);
        try {
            width_ = image.width; height_ = image.height;
            debug_.clear(); snapshots_.clear();
            image_ = {}; prediction_ = {};
            if (!execution_) execution_ = std::make_unique<TrackerExecution>(*model_, stats_.runtime);
            if (text_.empty()) text_ = model_->definition.encode_text(*model_->runtime, tokens_, stats_.runtime);
            image_ = model_->definition.encode_image(*model_->runtime, std::move(pixels), width_, height_, stats_.runtime, true);
            prediction_ = model_->definition.predict(*model_->runtime, image_, tokens_, text_, stats_.runtime);
            process(frame_index, video_detections(prediction_));
            ++next_frame_; ++stats_.accepted_frames;
            std::vector<VideoFrameResult> output;
            while (!pending_.empty() && (next_frame_ == frame_count_ || pending_.size() >= 15)) {
                output.push_back(emit(pending_.front())); pending_.pop_front(); ++stats_.emitted_frames;
            }
            update_stats(); stats_.frame_ms = elapsed_ms(start);
            return output;
        } catch (...) {
            failed_ = true;
            throw;
        }
    }

    const VideoStats& stats() const override { return stats_; }
    const std::vector<std::int32_t>& token_ids() const override { return tokens_; }
    std::string trace_json() const override { return trace_; }
    std::vector<std::string> tensor_names() const override {
        std::vector<std::string> names{"preprocessed_image", "text_features", "fusion_features", "class_logits",
                                     "presence_logits", "mask_logits"};
        for (const char* prefix : {"detector_neck_fpn", "tracker_neck_fpn"})
            for (int i = 0; i < 3; ++i) names.push_back(std::string(prefix) + std::to_string(i));
        for (const auto& [id, data] : debug_) {
            for (const char* suffix : {"conditioned_features", "decoder_masks", "decoder_iou", "pointer", "object_logit", "decoder_object_logit", "memory_features", "memory_mask"})
                names.push_back("object." + std::to_string(id) + "." + suffix);
            if (!data.propagated_mask.empty()) names.push_back("object." + std::to_string(id) + ".propagated_mask_logits");
            if (!data.propagated_conditioned.empty()) names.push_back("object." + std::to_string(id) + ".propagated_conditioned_features");
        }
        return names;
    }

    const TensorData& tensor(const std::string& name) const override {
        if (!next_frame_ || failed_) throw std::runtime_error("no completed video frame tensors");
        if (const auto found = snapshots_.find(name); found != snapshots_.end()) return found->second;
        TensorData value;
        if (name == "preprocessed_image") value = {{1, 3, 1008, 1008}, image_.preprocessed, "NCHW"};
        else if (name == "text_features") value = {{32, 1, 256}, text_, "LNC"};
        else if (name == "fusion_features") value = feature_tensor(prediction_.fusion, 256, 72, 72);
        else if (name == "class_logits") value = {{1, 200, 1}, prediction_.class_logits, "NQC"};
        else if (name == "presence_logits") value = {{1, 1}, {prediction_.presence_logit}, "NC"};
        else if (name == "mask_logits") value = {{1, 200, 288, 288}, prediction_.mask_logits, "NQHW"};
        else if (name.size() == 18 && name.substr(0, 17) == "detector_neck_fpn" && name.back() >= '0' && name.back() <= '2') {
            const int scale = name.back() - '0'; value = feature_tensor(image_.vision[scale], 256, 288 >> scale, 288 >> scale);
        } else if (name.size() == 17 && name.substr(0, 16) == "tracker_neck_fpn" && name.back() >= '0' && name.back() <= '2') {
            const int scale = name.back() - '0'; value = feature_tensor(image_.tracker[scale], 256, 288 >> scale, 288 >> scale);
        } else if (name.substr(0, 7) == "object.") {
            const auto separator = name.find('.', 7);
            if (separator == std::string::npos) throw std::invalid_argument("invalid tracker tensor name");
            const auto id_text = name.substr(7, separator - 7);
            const auto id = static_cast<std::uint64_t>(std::stoull(id_text));
            if (id_text != std::to_string(id)) throw std::invalid_argument("invalid tracker tensor name");
            const auto& data = debug_.at(id);
            const auto key = name.substr(separator + 1);
            if (key == "conditioned_features") value = feature_tensor(data.prediction.conditioned, 256, 72, 72);
            else if (key == "decoder_masks") value = {{1, 4, 288, 288}, data.prediction.decoder_masks, "NQHW"};
            else if (key == "decoder_iou") value = {{1, 4}, data.prediction.decoder_iou, "NQ"};
            else if (key == "pointer") value = {{1, 256}, data.prediction.pointer, "NC"};
            else if (key == "object_logit") value = {{1, 1}, {data.prediction.object_logit}, "NC"};
            else if (key == "decoder_object_logit") value = {{1, 1}, {data.prediction.decoder_object_logit}, "NC"};
            else if (key == "memory_features") value = feature_tensor(data.memory, 64, 72, 72);
            else if (key == "memory_mask") value = {{1, 1, 1152, 1152}, data.memory_mask, "NCHW"};
            else if (key == "propagated_mask_logits") value = {{1, 1, 288, 288}, data.propagated_mask, "NCHW"};
            else if (key == "propagated_conditioned_features") value = feature_tensor(data.propagated_conditioned, 256, 72, 72);
            else throw std::invalid_argument("unknown tracker tensor name");
        } else throw std::invalid_argument("unknown video tensor name");
        return snapshots_.emplace(name, std::move(value)).first->second;
    }

private:
    struct DebugObject {
        TrackerPrediction prediction;
        std::vector<float> memory, propagated_mask, propagated_conditioned, memory_mask;
    };

    TrackerPrediction seed(const VideoDetection& detection) {
        const auto start = std::chrono::steady_clock::now();
        auto binary = resize_mask(detection.mask, 288, 288, 1152, 1152);
        for (auto& value : binary) value = value > 0 ? 1.0f : 0.0f;
        auto result = execution_->decode(image_, image_.tracker[2], &binary); ++stats_.tracker_calls;
        const bool present = mask_nonempty(binary);
        if (!present) result.pointer = execution_->no_object_pointer();
        result.object_logit = present ? 10.0f : -10.0f; result.iou = 1.0f;
        auto video = resize_mask(binary, 1152, 1152, width_, height_, true);
        for (auto& value : video) value = value > 0.5f ? 1024.0f : -1024.0f;
        result.mask = std::move(video);
        stats_.tracker_ms += elapsed_ms(start);
        return result;
    }

    static float quality(const TrackerGroup& group, int frame) {
        float sum = 0, object_sum = 0, iou_sum = 0;
        for (const auto& object : group.objects) {
            const auto& record = object.records.at(frame);
            const float probability = record.object_logit > 0 ? sigmoid(record.object_logit) * 2 - 1 : 0.0f;
            sum += probability * record.iou; object_sum += probability; iou_sum += record.iou;
        }
        // Normal track_step has IoU [B] and logits [B,1], which Meta broadcasts
        // to [B,B]. Consolidated correction/birth records have IoU [B,1].
        const float count = static_cast<float>(group.objects.size());
        return group.consolidated_frames.count(frame) ? sum / count : object_sum * iou_sum / (count * count);
    }

    static void json_ids(std::ostream& output, const std::vector<std::uint64_t>& ids) {
        output << '[';
        for (std::size_t i = 0; i < ids.size(); ++i) { if (i) output << ','; output << ids[i]; }
        output << ']';
    }

    void process(int frame, const std::vector<VideoDetection>& detections) {
        const auto tracker_start = std::chrono::steady_clock::now();
        std::vector<std::uint64_t> old_ids;
        std::vector<std::vector<float>> masks;
        std::vector<float> tracker_scores;
        std::ostringstream trace;
        std::ostringstream propagation;
        propagation << std::setprecision(9);
        bool first_propagation = true;
        trace << std::setprecision(9) << "{\"frame_index\":" << frame << ",\"groups\":[";
        bool first_group = true;
        for (auto& group : groups_) {
            const auto selected = select_memory(group.history, frame, frame_count_, group.discarded_conditioning);
            if (!first_group) trace << ','; first_group = false;
            trace << "{\"birth\":" << group.birth << ",\"ids\":[";
            for (std::size_t i = 0; i < group.objects.size(); ++i) { if (i) trace << ','; trace << group.objects[i].id; }
            trace << "],\"spatial\":[";
            for (std::size_t i = 0; i < selected.spatial.size(); ++i) {
                if (i) trace << ','; trace << '[' << selected.spatial[i].frame << ',' << selected.spatial[i].position << ']';
            }
            trace << "],\"pointers\":[";
            for (std::size_t i = 0; i < selected.pointers.size(); ++i) {
                if (i) trace << ','; trace << '[' << selected.pointers[i].frame << ',' << selected.pointers[i].position << ']';
            }
            trace << "]}";
            for (auto& object : group.objects) {
                auto conditioned = execution_->condition(image_, object.records, selected, frame_count_);
                auto prediction = execution_->decode(image_, std::move(conditioned)); ++stats_.tracker_calls;
                if (!first_propagation) propagation << ',';
                first_propagation = false;
                propagation << "{\"id\":" << object.id << ",\"iou\":[";
                for (std::size_t i = 0; i < prediction.decoder_iou.size(); ++i) {
                    if (i) propagation << ','; propagation << prediction.decoder_iou[i];
                }
                propagation << "],\"mask_index\":" << prediction.mask_index
                            << ",\"pointer_index\":" << prediction.pointer_index << '}';
                debug_[object.id].propagated_mask = prediction.mask;
                debug_[object.id].propagated_conditioned = prediction.conditioned;
                object.records[frame] = {{}, prediction.pointer, prediction.object_logit, prediction.iou};
                auto cleaned = prediction.mask; clean_mask_components(cleaned, 288, 288);
                old_ids.push_back(object.id); masks.push_back(std::move(cleaned));
                tracker_scores.push_back(sigmoid(prediction.object_logit));
                debug_[object.id].prediction = std::move(prediction);
            }
            group.history[frame] = {frame, false, quality(group, frame)};
        }
        stats_.tracker_ms = elapsed_ms(tracker_start);
        const auto association = associate_video(detections, masks);
        auto births = association.births;
        const auto capacity = static_cast<std::size_t>(options_.max_objects) - old_ids.size();
        if (births.size() > capacity) {
            stats_.rejected_new_objects += births.size() - capacity;
            std::stable_sort(births.begin(), births.end(), [&](auto a, auto b) { return detections[a].score > detections[b].score; });
            births.resize(capacity);
        }
        const auto removed = policy_.update(frame, old_ids, association);
        // Periodic correction updates the group's records/pointers, while Meta
        // still emits this frame's pre-correction propagated masks.
        if (frame % 16 == 0) {
            for (const auto& [index, detection] : association.recondition) {
                const auto id = old_ids[index];
                if (debug_.at(id).prediction.object_logit <= 0.8f) continue;
                auto replacement = seed(detections[detection]);
                replacement.mask = resize_mask(replacement.mask, width_, height_, 288, 288, true);
                debug_[id].prediction = std::move(replacement);
                for (auto& group : groups_) for (auto& object : group.objects) if (object.id == id) {
                    const auto& corrected = debug_[id].prediction;
                    object.records[frame].pointer = corrected.pointer;
                    object.records[frame].object_logit = corrected.object_logit;
                    object.records[frame].iou = corrected.iou;
                    group.inputs[frame].insert(id); group.history[frame].conditioning = true;
                    group.consolidated_frames.insert(frame);
                }
            }
            for (auto& group : groups_) group.history[frame].group_quality = quality(group, frame);
        }
        // Suppress the more recently occluded member of each high-IoU pair.
        std::vector<bool> suppressed(old_ids.size());
        for (std::size_t i = 0; i < old_ids.size(); ++i) for (std::size_t j = i + 1; j < old_ids.size(); ++j) {
            const auto a = policy_.objects.at(old_ids[i]).last_occluded, b = policy_.objects.at(old_ids[j]).last_occluded;
            if (mask_overlap(masks[i], masks[j]) < 0.7f) continue;
            if (a > b && b > -1) suppressed[i] = true;
            if (b > a && a > -1) suppressed[j] = true;
        }
        for (std::size_t i = 0; i < old_ids.size(); ++i) {
            if (!mask_nonempty(masks[i]) || suppressed[i]) policy_.objects.at(old_ids[i]).last_occluded = frame;
            if (suppressed[i]) std::fill(masks[i].begin(), masks[i].end(), -10.0f);
        }
        const auto memory_start = std::chrono::steady_clock::now();
        const auto tracker_before_memory = stats_.tracker_ms;
        std::vector<std::vector<float>> high_masks;
        for (const auto& mask : masks) high_masks.push_back(resize_mask(mask, 288, 288, 1152, 1152));
        const auto nonoverlap = pixel_nonoverlap(high_masks);
        for (std::size_t i = 0; i < high_masks.size(); ++i) {
            const auto before = std::count_if(high_masks[i].begin(), high_masks[i].end(), [](float v) { return v > 0; });
            const auto after = std::count_if(nonoverlap[i].begin(), nonoverlap[i].end(), [](float v) { return v > 0; });
            if (static_cast<float>(after) / std::max<std::ptrdiff_t>(1, before) < 0.3f)
                for (auto& value : high_masks[i]) value = std::min(value, -10.0f);
            const bool present = mask_nonempty(high_masks[i]);
            for (auto& value : high_masks[i]) value = sigmoid(value) * 20 - 10;
            auto memory = execution_->encode_memory(image_, high_masks[i], present, &debug_[old_ids[i]].memory);
            debug_[old_ids[i]].memory_mask = std::move(high_masks[i]);
            for (auto& group : groups_) for (auto& object : group.objects)
                if (object.id == old_ids[i]) object.records[frame].memory = std::move(memory);
        }
        PendingFrame pending{frame, {}};
        for (std::size_t i = 0; i < old_ids.size(); ++i)
            if (!removed.count(old_ids[i])) pending.objects.push_back({old_ids[i], policy_.objects.at(old_ids[i]).score,
                tracker_scores[i], std::move(masks[i])});
        trace << "],\"births\":[";
        TrackerGroup new_group; new_group.birth = frame;
        std::vector<std::vector<float>> seed_videos;
        for (std::size_t i = 0; i < births.size(); ++i) {
            const auto& detection = detections[births[i]];
            const auto id = next_id_++;
            if (i) trace << ',';
            trace << "{\"id\":" << id << ",\"query_index\":" << detection.query << '}';
            policy_.objects[id] = {frame, 30, 0, -1, detection.score};
            auto predicted = seed(detection);
            // Later annotations remove their overlap only from earlier inputs.
            for (auto& previous : seed_videos) for (std::size_t pixel = 0; pixel < previous.size(); ++pixel)
                if (predicted.mask[pixel] > 0) previous[pixel] = -1024.0f;
            seed_videos.push_back(std::move(predicted.mask));
            predicted.mask.clear();
            new_group.objects.push_back({id, {{frame, {{}, predicted.pointer, predicted.object_logit, 1.0f}}}});
            new_group.inputs[frame].insert(id);
            debug_[id].prediction = std::move(predicted);
            auto output_mask = detection.mask; clean_mask_components(output_mask, 288, 288);
            pending.objects.push_back({id, detection.score, detection.score, std::move(output_mask)});
        }
        if (!new_group.objects.empty()) {
            new_group.consolidated_frames.insert(frame);
            std::vector<std::vector<float>> initial_high;
            for (std::size_t i = 0; i < seed_videos.size(); ++i) {
                auto low = resize_mask(seed_videos[i], width_, height_, 288, 288, true);
                auto high = resize_mask(low, 288, 288, 1008, 1008);
                debug_[new_group.objects[i].id].prediction.mask = std::move(low);
                initial_high.push_back(std::move(high));
            }
            initial_high = pixel_nonoverlap(initial_high);
            for (std::size_t i = 0; i < initial_high.size(); ++i) {
                for (auto& value : initial_high[i]) value = value > 0 ? 10.0f : -10.0f;
                auto memory_mask = resize_mask(initial_high[i], 1008, 1008, 1152, 1152, true);
                auto& object = new_group.objects[i];
                auto& record = object.records.at(frame);
                record.memory = execution_->encode_memory(image_, memory_mask, record.object_logit > 0, &debug_[object.id].memory);
                debug_[object.id].memory_mask = std::move(memory_mask);
            }
            new_group.history[frame] = {frame, true, quality(new_group, frame)};
            groups_.push_back(std::move(new_group));
        }
        stats_.memory_ms = elapsed_ms(memory_start) - (stats_.tracker_ms - tracker_before_memory);
        trace << "],\"removed\":";
        json_ids(trace, {removed.begin(), removed.end()});
        trace << ",\"propagation\":[" << propagation.str() << "]}"; trace_ = trace.str();
        for (auto& previous : pending_) previous.objects.erase(std::remove_if(previous.objects.begin(), previous.objects.end(),
            [&](const auto& object) { return removed.count(object.id); }), previous.objects.end());
        policy_.retire(removed);
        for (auto& group : groups_) {
            group.objects.erase(std::remove_if(group.objects.begin(), group.objects.end(),
                [&](const auto& object) { return removed.count(object.id); }), group.objects.end());
            if (group.objects.empty()) continue;
            for (auto& [index, ids] : group.inputs) {
                for (const auto id : removed) ids.erase(id);
                if (ids.empty()) group.history.at(index).conditioning = false;
            }
            for (auto& [index, metadata] : group.history) metadata.group_quality = quality(group, index);
            const auto retained = retained_memory_frames(group.history, frame, group.birth);
            for (auto item = group.history.begin(); item != group.history.end(); ) {
                if (!retained.count(item->first)) {
                    group.discarded_conditioning += item->second.conditioning;
                    for (auto& object : group.objects) object.records.erase(item->first);
                    group.inputs.erase(item->first); group.consolidated_frames.erase(item->first);
                    item = group.history.erase(item);
                } else ++item;
            }
        }
        groups_.erase(std::remove_if(groups_.begin(), groups_.end(), [](const auto& group) { return group.objects.empty(); }), groups_.end());
        pending_.push_back(std::move(pending));
        stats_.pending_high_water = std::max(stats_.pending_high_water, pending_.size());
    }

    VideoFrameResult emit(const PendingFrame& frame) const {
        VideoFrameResult result; result.frame_index = frame.frame;
        std::vector<float> tracker_scores;
        for (const auto& pending : frame.objects) {
            const auto scaled = resize_mask(pending.mask, 288, 288, width_, height_);
            TrackedObject object; object.id = pending.id; object.score = pending.score;
            object.mask = {width_, height_, std::vector<std::uint8_t>(scaled.size())};
            int x0 = width_, y0 = height_, x1 = -1, y1 = -1;
            for (std::size_t i = 0; i < scaled.size(); ++i) if (scaled[i] > 0) {
                object.mask.data[i] = 1;
                const int x = static_cast<int>(i % width_), y = static_cast<int>(i / width_);
                x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y);
            }
            if (x1 < 0) continue;
            object.box = {static_cast<float>(x0), static_cast<float>(y0), static_cast<float>(x1), static_cast<float>(y1)};
            result.objects.push_back(std::move(object)); tracker_scores.push_back(pending.tracker_score);
        }
        // Meta computes boxes before resolving output overlap by tracker score.
        for (std::size_t pixel = 0; pixel < checked_product(width_, height_, "video output"); ++pixel) {
            std::size_t winner = result.objects.size(); float best = 0;
            for (std::size_t i = 0; i < result.objects.size(); ++i)
                if (result.objects[i].mask.data[pixel] && tracker_scores[i] > best) { best = tracker_scores[i]; winner = i; }
            for (std::size_t i = 0; i < result.objects.size(); ++i)
                if (i != winner) result.objects[i].mask.data[pixel] = 0;
        }
        return result;
    }

    void update_stats() {
        stats_.active_objects = policy_.objects.size(); stats_.pending_frames = pending_.size();
        stats_.retained_records = stats_.retained_memory_bytes = 0;
        for (const auto& group : groups_) for (const auto& object : group.objects) {
            stats_.retained_records += object.records.size();
            for (const auto& [frame, record] : object.records)
                stats_.retained_memory_bytes += record.memory.size() * sizeof(ggml_bf16_t) + record.pointer.size() * sizeof(float);
        }
        stats_.retained_records_high_water = std::max(stats_.retained_records_high_water, stats_.retained_records);
    }

    std::shared_ptr<ModelState> model_;
    Tokenizer tokenizer_;
    VideoOptions options_;
    VideoStats stats_;
    std::unique_ptr<TrackerExecution> execution_;
    std::vector<std::int32_t> tokens_;
    std::vector<float> text_;
    ImageFeatures image_;
    Prediction prediction_;
    TemporalPolicy policy_;
    std::vector<TrackerGroup> groups_;
    std::deque<PendingFrame> pending_;
    std::map<std::uint64_t, DebugObject> debug_;
    mutable std::map<std::string, TensorData> snapshots_;
    std::string trace_ = "{}";
    int frame_count_ = 0, next_frame_ = 0, width_ = 0, height_ = 0;
    std::uint64_t next_id_ = 0;
    bool failed_ = false;
};

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_SESSION_HPP
