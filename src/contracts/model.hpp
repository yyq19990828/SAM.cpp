#ifndef SAM_CPP_SRC_CONTRACTS_MODEL_HPP
#define SAM_CPP_SRC_CONTRACTS_MODEL_HPP

#include "contracts/text_image.hpp"
#include "contracts/text_video.hpp"

#include <memory>
#include <stdexcept>

namespace sam::internal {

// A task contract, not a requirement that every model provides text prompts.
// Point/box and video tasks define separate contracts when implemented.
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

#endif // SAM_CPP_SRC_CONTRACTS_MODEL_HPP
