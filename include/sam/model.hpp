#ifndef SAM_CPP_SAM_MODEL_HPP
#define SAM_CPP_SAM_MODEL_HPP

#include "types.hpp"
#include "internal/model_interface.hpp"
#include "internal/models/sam3/model.hpp"

#include <memory>
#include <string>
#include <utility>

namespace sam {

class ImageSession;
class VideoSession;

class Model {
public:
    static Model load(const std::string& path, BackendOptions options = {}) {
        return Model(internal::sam3::load_model(path, options));
    }
    Backend backend() const { return implementation_->info().backend; }
    const ModelInfo& info() const { return implementation_->info(); }
private:
    explicit Model(std::shared_ptr<internal::ModelImplementation> implementation)
        : implementation_(std::move(implementation)) {}
    std::shared_ptr<internal::ModelImplementation> implementation_;
    friend class ImageSession;
    friend class VideoSession;
};

} // namespace sam

#endif // SAM_CPP_SAM_MODEL_HPP
