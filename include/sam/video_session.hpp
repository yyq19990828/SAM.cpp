#ifndef SAM_CPP_SAM_VIDEO_SESSION_HPP
#define SAM_CPP_SAM_VIDEO_SESSION_HPP

#include "model.hpp"
#include "sam/export.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sam {

namespace internal {
class TextVideoSessionImplementation;
}

// Forward-only, fixed prompt/resolution/count. Returned batches own their masks.
// Separate sessions share model weights and the existing model execution lock.
class SAM_API VideoSession {
public:
    VideoSession(const Model& model, int frame_count, VideoOptions options = {});
    VideoSession(const VideoSession&) = delete;
    VideoSession& operator=(const VideoSession&) = delete;
    ~VideoSession();

    void set_text(std::string_view prompt);
    void set_tokens(const std::vector<std::int32_t>& tokens);
    std::vector<VideoFrameResult> push_frame(int frame_index, ImageView image);
    void reset(int frame_count);
    const VideoStats& stats() const;
    const std::vector<std::int32_t>& token_ids() const;
    const TensorData& tensor(const std::string& name) const;
    std::string trace_json() const;
    std::vector<std::string> tensor_names() const;
private:
    std::unique_ptr<internal::TextVideoSessionImplementation> implementation_;
};

} // namespace sam

#endif // SAM_CPP_SAM_VIDEO_SESSION_HPP
