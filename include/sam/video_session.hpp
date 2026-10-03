#ifndef SAM_CPP_SAM_VIDEO_SESSION_HPP
#define SAM_CPP_SAM_VIDEO_SESSION_HPP

#include "model.hpp"
#include <memory>
#include <string_view>

namespace sam {

// Forward-only, fixed prompt/resolution/count. Returned batches own their masks.
// Separate sessions share model weights and the existing model execution lock.
class VideoSession {
public:
    VideoSession(const Model& model, int frame_count, VideoOptions options = {})
        : implementation_(model.implementation_->create_text_video_session(frame_count, options)) {}
    VideoSession(const VideoSession&) = delete;
    VideoSession& operator=(const VideoSession&) = delete;
    void set_text(std::string_view prompt) { implementation_->set_text(prompt); }
    void set_tokens(const std::vector<std::int32_t>& tokens) { implementation_->set_tokens(tokens); }
    std::vector<VideoFrameResult> push_frame(int frame_index, ImageView image) {
        return implementation_->push_frame(frame_index, image);
    }
    void reset(int frame_count) { implementation_->reset(frame_count); }
    const VideoStats& stats() const { return implementation_->stats(); }
    const std::vector<std::int32_t>& token_ids() const { return implementation_->token_ids(); }
    const TensorData& tensor(const std::string& name) const { return implementation_->tensor(name); }
    std::string trace_json() const { return implementation_->trace_json(); }
    std::vector<std::string> tensor_names() const { return implementation_->tensor_names(); }
private:
    std::unique_ptr<internal::TextVideoSessionImplementation> implementation_;
};

} // namespace sam

#endif // SAM_CPP_SAM_VIDEO_SESSION_HPP
