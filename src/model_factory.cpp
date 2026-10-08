#include "model_factory.hpp"

#include "models/sam3/model.hpp"

namespace sam::internal {

// Explicit branch instead of a global mutable registry: static linking must
// not drop model implementations, and adding a model stays a visible change
// in this assembly point.
std::shared_ptr<ModelImplementation> load_model(const std::string& path, BackendOptions options) {
    return sam3::load_model(path, options);
}

} // namespace sam::internal
