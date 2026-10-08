#ifndef SAM_CPP_SRC_MODEL_FACTORY_HPP
#define SAM_CPP_SRC_MODEL_FACTORY_HPP

#include "sam/types.hpp"

#include <memory>
#include <string>

namespace sam {
namespace internal {

class ModelImplementation;

// Explicit assembly point for the implemented model adapters. Only this
// factory needs to know which model implementations exist; callers receive
// the private task contract.
std::shared_ptr<ModelImplementation> load_model(const std::string& path, BackendOptions options);

} // namespace internal
} // namespace sam

#endif // SAM_CPP_SRC_MODEL_FACTORY_HPP
