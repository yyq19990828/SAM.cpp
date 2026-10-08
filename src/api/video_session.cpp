#include "sam/video_session.hpp"

#include "sam/internal/model_interface.hpp"

namespace sam {

VideoSession::VideoSession(const Model& model, int frame_count, VideoOptions options)
    : implementation_(model.implementation_->create_text_video_session(frame_count, options)) {}

VideoSession::~VideoSession() = default;

void VideoSession::set_text(std::string_view prompt) { implementation_->set_text(prompt); }

void VideoSession::set_tokens(const std::vector<std::int32_t>& tokens) { implementation_->set_tokens(tokens); }

std::vector<VideoFrameResult> VideoSession::push_frame(int frame_index, ImageView image) {
    return implementation_->push_frame(frame_index, image);
}

void VideoSession::reset(int frame_count) { implementation_->reset(frame_count); }

const VideoStats& VideoSession::stats() const { return implementation_->stats(); }

const std::vector<std::int32_t>& VideoSession::token_ids() const { return implementation_->token_ids(); }

const TensorData& VideoSession::tensor(const std::string& name) const { return implementation_->tensor(name); }

std::string VideoSession::trace_json() const { return implementation_->trace_json(); }

std::vector<std::string> VideoSession::tensor_names() const { return implementation_->tensor_names(); }

} // namespace sam
