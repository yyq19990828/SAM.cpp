#ifndef SAM_CPP_SAM_IMAGE_SESSION_HPP
#define SAM_CPP_SAM_IMAGE_SESSION_HPP

#include "model.hpp"
#include "sam/export.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sam {

namespace internal {
class TextImageSessionImplementation;
}

// A session retains its model and owns image/prompt caches. The same session
// must not be called concurrently; separate sessions may share a Model.
class SAM_API ImageSession {
public:
    explicit ImageSession(const Model& model);
    ImageSession(const ImageSession&) = delete;
    ImageSession& operator=(const ImageSession&) = delete;
    ~ImageSession();

    void set_image(ImageView image);
    Result segment_text(std::string_view prompt, float threshold = 0.5f);
    Result segment_tokens(const std::vector<std::int32_t>& tokens, float threshold = 0.5f);
    const RuntimeStats& stats() const;
    const std::vector<std::int32_t>& token_ids() const;
    const TensorData& tensor(const std::string& name) const;

private:
    std::unique_ptr<internal::TextImageSessionImplementation> implementation_;
};

} // namespace sam

#endif // SAM_CPP_SAM_IMAGE_SESSION_HPP
