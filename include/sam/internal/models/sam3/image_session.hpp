#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_IMAGE_SESSION_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_IMAGE_SESSION_HPP

#include "state.hpp"
#include "tokenizer.hpp"
#include "image_ops.hpp"
#include "sam/internal/model_interface.hpp"
#include "sam/internal/input_validation.hpp"

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sam::internal::sam3 {

// A session retains its model and owns image/prompt caches. The same session
// must not be called concurrently; separate sessions may share a Model.
class ImageSession final : public sam::internal::TextImageSessionImplementation {
public:
    explicit ImageSession(std::shared_ptr<ModelState> model)
        : model_(std::move(model)), tokenizer_(model_->tokenizer) {
        stats_.weight_buffer_bytes = ggml_backend_buffer_get_size(model_->buffer.get());
    }
    ImageSession(const ImageSession&) = delete;
    ImageSession& operator=(const ImageSession&) = delete;

    void set_image(ImageView image) override {
        auto pixels = preprocess_image(image);
        std::lock_guard<std::mutex> lock(model_->execution_mutex);
        // Invalidate before execution so a failed replacement cannot expose the
        // preceding image as if the new image had been prepared successfully.
        has_image_ = false;
        has_prediction_ = false;
        last_tokens_.clear();
        debug_tensors_.clear();
        prediction_ = {};
        image_ = {};
        image_ = model_->definition.encode_image(*model_->runtime, std::move(pixels), image.width, image.height, stats_);
        has_image_ = true;
    }

    Result segment_text(std::string_view prompt, float threshold = 0.5f) override {
        validate_score_threshold(threshold);
        return segment_tokens(tokenizer_.encode(std::string(prompt)), threshold);
    }

    Result segment_tokens(const std::vector<std::int32_t>& tokens, float threshold = 0.5f) override {
        validate_score_threshold(threshold);
        tokenizer_.validate_tokens(tokens);
        if (!has_image_) throw std::runtime_error("set_image must succeed before segmentation");
        if (!has_prediction_ || tokens != last_tokens_) {
            std::lock_guard<std::mutex> lock(model_->execution_mutex);
            has_prediction_ = false;
            debug_tensors_.clear();
            const auto start = std::chrono::steady_clock::now();
            auto text = model_->definition.encode_text(*model_->runtime, tokens, stats_);
            prediction_ = model_->definition.predict(*model_->runtime, image_, tokens, std::move(text), stats_);
            stats_.inference_ms = elapsed_ms(start);
            last_tokens_ = tokens;
            has_prediction_ = true;
        }
        const int mask_size = model_->definition.weights.hparams.n_img_embd() * 4;
        return postprocess_detections(prediction_.boxes, prediction_.class_logits,
            prediction_.presence_logit, prediction_.mask_logits, mask_size, mask_size,
            image_.width, image_.height, threshold);
    }

    const RuntimeStats& stats() const override { return stats_; }
    const std::vector<std::int32_t>& token_ids() const override { return last_tokens_; }

    // Internal experiment accounting; debug F32 snapshots are excluded.
    std::size_t feature_cache_bytes() const {
        std::size_t bytes = 0;
        for (const auto& feature : image_.vision) bytes += feature.size_bytes();
        return bytes;
    }

    // Numerical validation hook. Native GGML buffers are never exposed. These
    // owned snapshots are materialized only when requested, then invalidated
    // by the next image or changed prompt.
    const TensorData& tensor(const std::string& name) const override {
        if (!has_image_) throw std::runtime_error("no encoded image tensor is available");
        const auto existing = debug_tensors_.find(name);
        if (existing != debug_tensors_.end()) return existing->second;
        TensorData value;
        const int d = model_->definition.weights.hparams.neck_dim;
        const int h = model_->definition.weights.hparams.n_img_embd();
        if (name == "preprocessed_image") value = {{1, 3, 1008, 1008}, image_.preprocessed, "NCHW"};
        else if (name == "vision_features_0") value = feature_tensor(image_.vision[0], d, h * 4, h * 4);
        else if (name == "vision_features_1") value = feature_tensor(image_.vision[1], d, h * 2, h * 2);
        else if (name == "vision_features_2") value = feature_tensor(image_.vision[2], d, h, h);
        else {
            if (!has_prediction_) throw std::runtime_error("no prompt prediction tensor is available");
            if (name == "text_features") value = {{32, 1, d}, prediction_.text, "LNC"};
            else if (name == "fusion_features") value = feature_tensor(prediction_.fusion, d, h, h);
            else if (name == "pred_boxes") value = {{1, 200, 4}, prediction_.boxes, "NQC"};
            else if (name == "class_logits") value = {{1, 200, 1}, prediction_.class_logits, "NQC"};
            else if (name == "presence_logits") value = {{1, 1}, {prediction_.presence_logit}, "NC"};
            else if (name == "mask_logits") value = {{1, 200, h * 4, h * 4}, prediction_.mask_logits, "NQHW"};
            else throw std::invalid_argument("unknown SAM tensor: " + name);
        }
        return debug_tensors_.emplace(name, std::move(value)).first->second;
    }
private:
    std::shared_ptr<ModelState> model_;
    Tokenizer tokenizer_;
    ImageFeatures image_;
    Prediction prediction_;
    std::vector<std::int32_t> last_tokens_;
    RuntimeStats stats_;
    bool has_image_ = false, has_prediction_ = false;
    mutable std::map<std::string, TensorData> debug_tensors_;
};

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_IMAGE_SESSION_HPP
