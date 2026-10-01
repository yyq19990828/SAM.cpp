#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_MODEL_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_MODEL_HPP

#include "sam/types.hpp"
#include "sam/internal/model_interface.hpp"
#include "sam/internal/input_validation.hpp"
#include "state.hpp"
#include "image_session.hpp"
#include "tensors.hpp"
#include "weights.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sam::internal::sam3 {

inline std::vector<float> promote_f16(const std::vector<char>& bytes) {
    if (bytes.size() % sizeof(ggml_fp16_t)) throw std::runtime_error("invalid F16 tensor byte count");
    std::vector<ggml_fp16_t> half(bytes.size() / sizeof(ggml_fp16_t));
    std::memcpy(half.data(), bytes.data(), bytes.size());
    std::vector<float> values(half.size());
    ggml_fp16_to_fp32_row(half.data(), values.data(), static_cast<int64_t>(half.size()));
    return values;
}

inline std::shared_ptr<ModelState> load_state(const std::string& path, BackendOptions options) {
    validate_backend_options(options);
    auto file = inspect_weights(path); // Validate every file range before allocating weights.
    auto state = std::make_shared<ModelState>();
    state->context = make_context(4096);
    auto& definition = state->definition.weights;
    definition.ctx = state->context.get();
    definition.weight_type = file.ftype == 0 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    for (const auto& tensor : file.tensors)
        definition.source_types.emplace(tensor.name, static_cast<ggml_type>(tensor.type));
    sam3_register_tensors(definition, file.video);
    std::map<std::string, const TensorInfo*> inventory;
    for (const auto& tensor : file.tensors) inventory.emplace(tensor.name, &tensor);
    for (const auto& required : definition.tensors) {
        const auto found = inventory.find(required.first);
        if (found == inventory.end()) throw std::runtime_error("missing SAM 3 tensor: " + required.first);
        const auto expected = required.second;
        const int required_type = file.ftype == 1 && !tensor_kept_f32(required.first, ggml_n_dims(expected)) ? 1 : 0;
        if (found->second->type != required_type)
            throw std::runtime_error("incompatible SAM 3 tensor precision: " + required.first);
        const auto shape = canonical_dimensions(found->second->dimensions);
        if (!std::equal(shape.begin(), shape.end(), expected->ne))
            throw std::runtime_error("incompatible SAM 3 tensor shape: " + required.first);
        if (expected->type != static_cast<ggml_type>(found->second->type) ||
            ggml_nbytes(expected) != found->second->size_bytes)
            throw std::runtime_error("incompatible SAM 3 tensor storage: " + required.first);
    }
    for (const auto& tensor : file.tensors) {
        if (!definition.tensors.count(tensor.name))
            throw std::runtime_error("unknown SAM 3 tensor: " + tensor.name);
    }
    state->runtime = std::make_unique<GgmlRuntime>(options, file.ftype == 0);
    if (file.ftype == 1 && state->runtime->promote_f16_weights()) {
        // The CPU F16 dot path narrows activations to F16. Preserve the exact
        // stored values while using F32 arithmetic, after the source schema
        // has passed validation. Only the allocated representation changes.
        state->context = make_context(4096);
        state->definition = ModelDefinition{};
        definition.ctx = state->context.get();
        definition.weight_type = GGML_TYPE_F32;
        sam3_register_tensors(definition, file.video);
    }
    state->buffer.reset(ggml_backend_alloc_ctx_tensors(state->context.get(), state->runtime->weights_backend()));
    if (!state->buffer) throw std::runtime_error("failed to allocate SAM 3 model weights");
    ggml_backend_buffer_set_usage(state->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::size_t weight_bytes = 0;
    for (const auto& tensor : file.tensors) {
        const auto found = definition.tensors.find(tensor.name);
        if (found == definition.tensors.end()) continue;
        std::vector<char> bytes(static_cast<std::size_t>(tensor.size_bytes));
        file.reader->read(tensor.offset, bytes.data(), bytes.size());
        if (tensor.type == GGML_TYPE_F16 && found->second->type == GGML_TYPE_F32) {
            const auto values = promote_f16(bytes);
            ggml_backend_tensor_set(found->second, values.data(), 0, values.size() * sizeof(float));
        } else {
            ggml_backend_tensor_set(found->second, bytes.data(), 0, bytes.size());
        }
        weight_bytes += ggml_nbytes(found->second);
    }
    state->tokenizer = std::move(file.tokenizer);
    state->model_info = {"sam3", file.ftype == 0 ? "f32" : "f16", state->runtime->backend(), options.threads,
                   definition.tensors.size(), weight_bytes, false};
    state->model_info.task = file.video ? "text_video" : "text_image";
    state->model_info.profile = file.video ? "meta-sam3-temporal-v1" : "";
    return state;
}

class ModelAdapter final : public sam::internal::ModelImplementation {
public:
    explicit ModelAdapter(std::shared_ptr<ModelState> state) : state_(std::move(state)) {}
    const ModelInfo& info() const override { return state_->model_info; }
    std::unique_ptr<sam::internal::TextImageSessionImplementation> create_text_image_session() override {
        return std::make_unique<ImageSession>(state_);
    }
private:
    std::shared_ptr<ModelState> state_;
};

inline std::shared_ptr<sam::internal::ModelImplementation> load_model(const std::string& path, BackendOptions options) {
    return std::make_shared<ModelAdapter>(load_state(path, options));
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_MODEL_HPP
