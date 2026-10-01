#ifndef SAM_CPP_SAM_IMAGE_SESSION_HPP
#define SAM_CPP_SAM_IMAGE_SESSION_HPP

#include "model.hpp"
#include "internal/model_interface.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sam {

// A session retains its model and owns image/prompt caches. The same session
// must not be called concurrently; separate sessions may share a Model.
class ImageSession {
public:
    explicit ImageSession(const Model& model)
        : implementation_(model.implementation_->create_text_image_session()) {}
    ImageSession(const ImageSession&) = delete;
    ImageSession& operator=(const ImageSession&) = delete;

    void set_image(ImageView image) { implementation_->set_image(image); }
    Result segment_text(std::string_view prompt, float threshold = 0.5f) {
        return implementation_->segment_text(prompt, threshold);
    }
    Result segment_tokens(const std::vector<std::int32_t>& tokens, float threshold = 0.5f) {
        return implementation_->segment_tokens(tokens, threshold);
    }
    const RuntimeStats& stats() const { return implementation_->stats(); }
    const std::vector<std::int32_t>& token_ids() const { return implementation_->token_ids(); }
    const TensorData& tensor(const std::string& name) const { return implementation_->tensor(name); }

private:
    std::unique_ptr<internal::TextImageSessionImplementation> implementation_;
};

} // namespace sam

#endif // SAM_CPP_SAM_IMAGE_SESSION_HPP
