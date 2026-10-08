#ifndef SAM_CPP_SRC_CONTRACTS_TEXT_IMAGE_HPP
#define SAM_CPP_SRC_CONTRACTS_TEXT_IMAGE_HPP

#include "sam/types.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace sam::internal {

// Task contract for text-prompted image segmentation. Point/box tasks will
// define separate contracts when implemented; this one stays text-specific.
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

} // namespace sam::internal

#endif // SAM_CPP_SRC_CONTRACTS_TEXT_IMAGE_HPP
