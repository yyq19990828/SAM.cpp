#include <sam/types.hpp>
#include <sam/types.hpp>
#include <contracts/model.hpp>
#include <contracts/model.hpp>

#if defined(GGML_H) || defined(GGML_BACKEND_H) || defined(GGML_ALLOC_H) || defined(GGML_CPU_H)
#error "Public values and task contracts must not include GGML"
#endif

#include <iostream>

namespace {

class ModelWithoutTextTask final : public sam::internal::ModelImplementation {
public:
    const sam::ModelInfo& info() const override { return info_; }
private:
    sam::ModelInfo info_;
};

} // namespace

int main() {
    const std::shared_ptr<sam::internal::ModelImplementation> model = std::make_shared<ModelWithoutTextTask>();
    try {
        (void) model->create_text_image_session();
        std::cerr << "Unsupported text-image task was accepted\n";
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("text") != std::string::npos) return 0;
        std::cerr << "Unsupported task error did not identify the text task\n";
    }
    return 1;
}
