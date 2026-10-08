#ifndef SAM_CPP_SAM_MODEL_HPP
#define SAM_CPP_SAM_MODEL_HPP

#include "types.hpp"
#include "sam/export.hpp"

#include <memory>
#include <string>

namespace sam {

class ImageSession;
class VideoSession;

namespace internal {
class ModelImplementation;
}

// Owns shared model state. Copying a Model shares that state; the definition
// of the private implementation stays in the compiled library.
class SAM_API Model {
public:
    static Model load(const std::string& path, BackendOptions options = {});
    Backend backend() const;
    const ModelInfo& info() const;
private:
    explicit Model(std::shared_ptr<internal::ModelImplementation> implementation);
    std::shared_ptr<internal::ModelImplementation> implementation_;
    friend class ImageSession;
    friend class VideoSession;
};

} // namespace sam

#endif // SAM_CPP_SAM_MODEL_HPP
