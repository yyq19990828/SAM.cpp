#include "sam/image_session.hpp"

#include "contracts/model.hpp"

namespace sam {

ImageSession::ImageSession(const Model& model)
    : implementation_(model.implementation_->create_text_image_session()) {}

ImageSession::~ImageSession() = default;

void ImageSession::set_image(ImageView image) { implementation_->set_image(image); }

Result ImageSession::segment_text(std::string_view prompt, float threshold) {
    return implementation_->segment_text(prompt, threshold);
}

Result ImageSession::segment_tokens(const std::vector<std::int32_t>& tokens, float threshold) {
    return implementation_->segment_tokens(tokens, threshold);
}

const RuntimeStats& ImageSession::stats() const { return implementation_->stats(); }

const std::vector<std::int32_t>& ImageSession::token_ids() const { return implementation_->token_ids(); }

const TensorData& ImageSession::tensor(const std::string& name) const { return implementation_->tensor(name); }

} // namespace sam
