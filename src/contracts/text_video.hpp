#ifndef SAM_CPP_SRC_CONTRACTS_TEXT_VIDEO_HPP
#define SAM_CPP_SRC_CONTRACTS_TEXT_VIDEO_HPP

#include "sam/types.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sam::internal {

// Task contract for text-prompted video tracking. Each model family owns its
// own tracking policy; this interface only fixes the public session vocabulary.
class TextVideoSessionImplementation {
public:
    virtual ~TextVideoSessionImplementation() = default;
    virtual void set_text(std::string_view prompt) = 0;
    virtual void set_tokens(const std::vector<std::int32_t>& tokens) = 0;
    virtual std::vector<VideoFrameResult> push_frame(int frame_index, ImageView image) = 0;
    virtual void reset(int frame_count) = 0;
    virtual const VideoStats& stats() const = 0;
    virtual const std::vector<std::int32_t>& token_ids() const = 0;
    virtual const TensorData& tensor(const std::string& name) const = 0;
    virtual std::string trace_json() const = 0;
    virtual std::vector<std::string> tensor_names() const = 0;
};

} // namespace sam::internal

#endif // SAM_CPP_SRC_CONTRACTS_TEXT_VIDEO_HPP
