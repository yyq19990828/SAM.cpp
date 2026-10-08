#include "sam/model.hpp"

#include "model_factory.hpp"
#include "contracts/model.hpp"

#include <utility>

namespace sam {

Model Model::load(const std::string& path, BackendOptions options) {
    return Model(internal::load_model(path, options));
}

Backend Model::backend() const { return implementation_->info().backend; }

const ModelInfo& Model::info() const { return implementation_->info(); }

Model::Model(std::shared_ptr<internal::ModelImplementation> implementation)
    : implementation_(std::move(implementation)) {}

} // namespace sam
