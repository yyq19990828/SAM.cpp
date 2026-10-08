#ifndef SAM_CPP_SRC_MODELS_SAM3_VIDEO_FRAME_STORAGE_HPP
#define SAM_CPP_SRC_MODELS_SAM3_VIDEO_FRAME_STORAGE_HPP

#include "../state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace sam::internal::sam3 {

// Tracks which resident frame inputs were already uploaded in the current
// frame, so repeated stage calls do not repeat host-to-device copies.
struct TrackerResidentUploadState {
    template<class Upload>
    void ensure_core(Upload&& upload) {
        if (core_uploaded_) return;
        upload();
        core_uploaded_ = true;
    }

    template<class Upload>
    void ensure_high(Upload&& upload) {
        if (high_uploaded_) return;
        upload();
        high_uploaded_ = true;
    }

    void reset_frame() noexcept { core_uploaded_ = high_uploaded_ = false; }
    void release_high() noexcept { high_uploaded_ = false; }
    bool core_uploaded() const noexcept { return core_uploaded_; }
    bool high_uploaded() const noexcept { return high_uploaded_; }

private:
    bool core_uploaded_ = false;
    bool high_uploaded_ = false;
};

// Owns the current frame's resident inputs: tracker neck features, positional
// tables and the fixed prompt. Frames are released at frame boundaries; cached
// graph metadata survives because it only refers to these tensors by pointer.
class TrackerFrameStorage {
public:
    explicit TrackerFrameStorage(GgmlRuntime& runtime) : runtime_(runtime) { make_resident_context(); }

    void begin_frame() {
        high_resident_ = high_residency_allowed_;
        none_resident_ = false;
        resident_uploads_.reset_frame();
    }

    // The coordinator releases the shared workspace before this call when
    // cached graphs may still reference the resident tensors.
    void end_frame() {
        if (resident_buffer_) {
            clear_context(resident_context_.get());
            resident_buffer_.reset();
        }
        if (high_buffer_) {
            clear_context(high_context_.get());
            high_buffer_.reset();
        }
        resident_uploads_.reset_frame();
        high_resident_ = high_residency_allowed_;
        none_resident_ = false;
    }

    bool high_resident() const { return high_resident_; }
    bool none_resident() const { return none_resident_; }
    bool high_residency_allowed() const { return high_residency_allowed_; }

    ggml_tensor* tracker(int index) const { return resident_tracker_[index]; }
    ggml_tensor* position() const { return resident_position_; }
    ggml_tensor* memory_position_bank() const { return resident_memory_position_bank_; }
    ggml_tensor* dense_position() const { return resident_dense_position_; }
    ggml_tensor* rope() const { return resident_rope_; }
    ggml_tensor* no_point() const { return resident_no_point_; }

    std::size_t resident_bytes() const {
        return (resident_buffer_ ? ggml_backend_buffer_get_size(resident_buffer_.get()) : 0) +
            (high_buffer_ ? ggml_backend_buffer_get_size(high_buffer_.get()) : 0);
    }

    std::size_t context_buffer_bytes(const ggml_context* context) const {
        const auto buffer_type = ggml_backend_get_default_buffer_type(runtime_.weights_backend());
        return ggml_backend_alloc_ctx_tensors_from_buft_size(const_cast<ggml_context*>(context), buffer_type);
    }

    // Selects the resident mode for the incoming budget. A none-resident
    // decision is returned to the coordinator because it also affects cached
    // graphs and batch policy; the high-resolution downgrade is local.
    enum class ResidentDecision { keep, demote_none };
    ResidentDecision select_resident_mode(std::size_t budget) {
        if (none_resident_ || resident_buffer_) return ResidentDecision::keep;
        const auto core_bytes = context_buffer_bytes(resident_context_.get());
        const auto high_bytes = high_residency_allowed_ ? context_buffer_bytes(high_context_.get()) : 0;
        if (core_bytes > budget) return ResidentDecision::demote_none;
        if (high_residency_allowed_ && high_bytes > budget - core_bytes) {
            high_resident_ = false;
            high_residency_allowed_ = false;
            ++resident_demotions_;
        } else {
            high_resident_ = high_residency_allowed_;
        }
        return ResidentDecision::keep;
    }

    void require_frame_buffers() {
        if (none_resident_) return;
        if (!resident_buffer_) {
            auto buffer = ggml_backend_alloc_ctx_tensors(resident_context_.get(), runtime_.weights_backend());
            if (!buffer) throw std::runtime_error("failed to allocate tracker resident frame inputs");
            resident_buffer_.reset(buffer);
            ggml_backend_buffer_set_usage(resident_buffer_.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        if (high_resident_ && !high_buffer_) {
            auto high_buffer = ggml_backend_alloc_ctx_tensors(high_context_.get(), runtime_.weights_backend());
            if (!high_buffer) throw std::runtime_error("failed to allocate tracker high-resolution frame inputs");
            high_buffer_.reset(high_buffer);
            ggml_backend_buffer_set_usage(high_buffer_.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        resident_peak_bytes_ = std::max(resident_peak_bytes_, resident_bytes());
    }

    template<class Upload>
    void ensure_core_uploaded(Upload&& upload) {
        if (!resident_buffer_) throw std::runtime_error("tracker core resident frame was not allocated");
        resident_uploads_.ensure_core(std::forward<Upload>(upload));
    }

    template<class Upload>
    void ensure_high_uploaded(Upload&& upload) {
        if (!high_buffer_) throw std::runtime_error("tracker high-resolution resident frame was not allocated");
        resident_uploads_.ensure_high(std::forward<Upload>(upload));
    }

    // Drops uploaded frame storage without touching the residency policy; used
    // before workspace probes and frame transitions.
    void release_storage() noexcept {
        if (resident_buffer_) {
            clear_context(resident_context_.get());
            resident_buffer_.reset();
        }
        if (high_buffer_) {
            clear_context(high_context_.get());
            high_buffer_.reset();
        }
        resident_uploads_.reset_frame();
    }

    // Demotion primitives: the coordinator invalidates cached graphs and
    // output pointers around them, the workspace policy records the decision.
    void drop_high_residency() noexcept {
        if (!high_resident_) return;
        clear_context(high_context_.get());
        high_buffer_.reset();
        resident_uploads_.release_high();
        high_resident_ = false;
        high_residency_allowed_ = false;
        ++resident_demotions_;
    }

    void force_none_residency() noexcept {
        if (none_resident_) return;
        clear_context(resident_context_.get());
        clear_context(high_context_.get());
        resident_buffer_.reset();
        high_buffer_.reset();
        resident_uploads_.reset_frame();
        high_resident_ = false;
        none_resident_ = true;
        ++none_resident_fallbacks_;
    }

    // Sticky none-resident decisions mark the flags without releasing buffers
    // that were already dropped by a demotion or were never allocated.
    void mark_none_residency() noexcept {
        high_resident_ = false;
        none_resident_ = true;
    }

    std::size_t resident_peak_bytes() const { return resident_peak_bytes_; }
    std::size_t resident_demotions() const { return resident_demotions_; }
    std::size_t none_resident_fallbacks() const { return none_resident_fallbacks_; }

private:
    void make_resident_context() {
        resident_context_ = make_context(6);
        high_context_ = make_context(2);
        for (int i = 0; i < 2; ++i) {
            const int size = 288 >> i;
            resident_tracker_[i] = ggml_new_tensor_4d(high_context_.get(), GGML_TYPE_F32, 256, size, size, 1);
            ggml_set_name(resident_tracker_[i], ("tracker_frame_" + std::to_string(i)).c_str());
        }
        resident_tracker_[2] = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32, 256, 72, 72, 1);
        ggml_set_name(resident_tracker_[2], "tracker_frame_2");
        resident_position_ = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32, 256, 5184, 1, 1);
        resident_memory_position_bank_ = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32,
                                                            64, 5184, 7, 1);
        resident_dense_position_ = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32, 256, 72, 72, 1);
        resident_rope_ = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32, 2, 128, 5184, 1);
        resident_no_point_ = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32, 256, 2, 1, 1);
        ggml_set_name(resident_position_, "tracker_frame_position");
        ggml_set_name(resident_memory_position_bank_, "tracker_memory_position_bank");
        ggml_set_name(resident_dense_position_, "tracker_dense_position");
        ggml_set_name(resident_rope_, "tracker_rope_frequencies");
        ggml_set_name(resident_no_point_, "tracker_no_point_prompt");
    }

    static void clear_context(ggml_context* context) noexcept {
        if (!context) return;
        for (auto* tensor = ggml_get_first_tensor(context); tensor; tensor = ggml_get_next_tensor(context, tensor)) {
            tensor->data = nullptr;
            tensor->buffer = nullptr;
            tensor->extra = nullptr;
        }
    }

    GgmlRuntime& runtime_;
    ContextPtr resident_context_;
    BufferPtr resident_buffer_;
    ContextPtr high_context_;
    BufferPtr high_buffer_;
    TrackerResidentUploadState resident_uploads_;
    std::array<ggml_tensor*, 3> resident_tracker_{};
    ggml_tensor* resident_position_ = nullptr;
    ggml_tensor* resident_memory_position_bank_ = nullptr;
    ggml_tensor* resident_dense_position_ = nullptr;
    ggml_tensor* resident_rope_ = nullptr;
    ggml_tensor* resident_no_point_ = nullptr;

    bool high_resident_ = true, high_residency_allowed_ = true, none_resident_ = false;
    std::size_t resident_peak_bytes_ = 0, resident_demotions_ = 0, none_resident_fallbacks_ = 0;
};

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_VIDEO_FRAME_STORAGE_HPP
