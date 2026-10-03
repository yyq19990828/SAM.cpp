#ifndef SAM_CPP_SAM_INTERNAL_MODEL_INTERFACE_HPP
#define SAM_CPP_SAM_INTERNAL_MODEL_INTERFACE_HPP

#include "sam/types.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sam::internal {

// A task contract, not a requirement that every model provides text prompts.
// Point/box and video tasks will define separate contracts when implemented.
class TextImageSessionImplementation {
public:
    virtual ~TextImageSessionImplementation() = default;
    virtual void set_image(ImageView image) = 0;
    virtual Result segment_text(std::string_view prompt, float threshold) = 0;
    virtual Result segment_tokens(const std::vector<std::int32_t>& tokens, float threshold) = 0;
    virtual const RuntimeStats& stats() const = 0;
    virtual const std::vector<std::int32_t>& token_ids() const = 0;
    virtual const TensorData& tensor(const std::string& name) const = 0;
};

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

class ModelImplementation {
public:
    virtual ~ModelImplementation() = default;
    virtual const ModelInfo& info() const = 0;
    virtual std::unique_ptr<TextImageSessionImplementation> create_text_image_session() {
        throw std::runtime_error("loaded model does not support text-prompted image segmentation");
    }
    virtual std::unique_ptr<TextVideoSessionImplementation> create_text_video_session(int, VideoOptions) {
        throw std::runtime_error("loaded model does not support text-prompted video tracking");
    }
};

} // namespace sam::internal

#endif // SAM_CPP_SAM_INTERNAL_MODEL_INTERFACE_HPP
