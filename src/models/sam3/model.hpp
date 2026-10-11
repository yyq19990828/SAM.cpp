#ifndef SAM_CPP_SRC_MODELS_SAM3_MODEL_HPP
#define SAM_CPP_SRC_MODELS_SAM3_MODEL_HPP

#include "sam/types.hpp"
#include "contracts/model.hpp"
#include "common/input_validation.hpp"
#include "state.hpp"
#include "image_session.hpp"
#include "video/session.hpp"
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

inline std::shared_ptr<ModelState> load_state(const std::string& path, BackendOptions options,
                                             FeatureCacheMode cache = FeatureCacheMode::F32) {
    validate_backend_options(options);
    auto file = inspect_weights(path); // Validate every file range before allocating weights.
    if (file.video && cache != FeatureCacheMode::F32)
        throw std::invalid_argument("experimental image feature caches require an image model");
    auto state = std::make_shared<ModelState>();
    state->context = make_context(4096);
    auto& definition = state->definition.weights;
    definition.ctx = state->context.get();
    // Register the canonical F32 graph first. Validate all untrusted shapes
    // before source_types can request a quantized tensor with an incompatible
    // ne[0] and trigger GGML's tensor-construction assertions.
    definition.weight_type = GGML_TYPE_F32;
    sam3_register_tensors(definition, file.video);
    std::map<std::string, const TensorInfo*> inventory;
    for (const auto& tensor : file.tensors) inventory.emplace(tensor.name, &tensor);
    for (const auto& required : definition.tensors) {
        const auto found = inventory.find(required.first);
        if (found == inventory.end()) throw std::runtime_error("missing SAM 3 tensor: " + required.first);
        const auto expected = required.second;
        const auto shape = canonical_dimensions(found->second->dimensions);
        if (!std::equal(shape.begin(), shape.end(), expected->ne))
            throw std::runtime_error("incompatible SAM 3 tensor shape: " + required.first);
        std::int32_t required_type = 0;
        if (file.mixed_quantized) {
            required_type = static_cast<std::int32_t>(image_mixed_quantized_tensor_type(
                required.first, found->second->dimensions, {file.base_precision, file.policy_sha256, file.module_precisions, file.tensor_precisions, file.storage_profile}));
        } else if (file.modular_quantized) {
            required_type = static_cast<std::int32_t>(image_modular_quantized_tensor_type(
                required.first, found->second->dimensions,
                *modular_image_quantization_profile(file.storage_profile), file.quantization_modules));
        } else if (file.quantized) {
            required_type = static_cast<std::int32_t>(image_quantized_tensor_type(
                required.first, found->second->dimensions, *image_quantization_profile(file.storage_profile)));
        } else {
            required_type = file.ftype == 1 &&
                !tensor_kept_f32(required.first, ggml_n_dims(expected), !file.storage_profile.empty()) ? 1 : 0;
        }
        if (found->second->type != required_type)
            throw std::runtime_error("incompatible SAM 3 tensor precision: " + required.first);
    }
    for (const auto& tensor : file.tensors) {
        if (!definition.tensors.count(tensor.name))
            throw std::runtime_error("unknown SAM 3 tensor: " + tensor.name);
    }
    if (options.activation == ActivationMode::CudaVisionQ8Rhs) {
        if (file.video || cache != FeatureCacheMode::F32)
            throw std::invalid_argument("explicit Q8 RHS supports image models with F32 cache only");
        for (std::size_t i = 0; i < definition.vit.blocks.size(); ++i)
            for (const auto* suffix : {".attn.qkv.weight", ".attn.proj.weight", ".mlp.lin1.weight", ".mlp.lin2.weight"}) {
                const auto name = "vit.blocks." + std::to_string(i) + suffix;
                if (inventory.at(name)->type != GGML_TYPE_Q8_0)
                    throw std::invalid_argument("explicit Q8 RHS requires actual Q8_0 vision linear weights: " + name);
            }
    }
    const bool packed_weights = std::any_of(file.tensors.begin(), file.tensors.end(), [](const TensorInfo& tensor) {
        return ggml_is_quantized(static_cast<ggml_type>(tensor.type));
    });
    const bool f32_weights = std::all_of(file.tensors.begin(), file.tensors.end(), [](const TensorInfo& tensor) {
        return tensor.type == GGML_TYPE_F32;
    });
    state->runtime = std::make_unique<GgmlRuntime>(options, f32_weights, packed_weights, cache);
    if (file.ftype == 1 || file.quantized) {
        state->definition = ModelDefinition{};
        state->context = make_context(4096);
        definition.ctx = state->context.get();
        definition.weight_type = GGML_TYPE_F32;
        for (const auto& tensor : file.tensors)
            definition.source_types.emplace(tensor.name, state->runtime->allocated_weight_type(static_cast<ggml_type>(tensor.type)));
        sam3_register_tensors(definition, file.video);
    }
    for (const auto& required : definition.tensors) {
        const auto* tensor = inventory.at(required.first);
        const auto stored_type = static_cast<ggml_type>(tensor->type);
        const auto allocated_type = state->runtime->allocated_weight_type(stored_type);
        const auto expected_bytes = stored_type == GGML_TYPE_F16 && allocated_type == GGML_TYPE_F32 ?
            tensor->size_bytes / sizeof(ggml_fp16_t) * sizeof(float) : tensor->size_bytes;
        if (required.second->type != allocated_type || ggml_nbytes(required.second) != expected_bytes)
            throw std::runtime_error("incompatible SAM 3 tensor storage: " + required.first);
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
        if (tensor.type == GGML_TYPE_F16) validate_f16_payload(bytes.data(), bytes.size());
        if (file.quantized && ggml_is_quantized(static_cast<ggml_type>(tensor.type)))
            validate_quantized_payload(static_cast<ggml_type>(tensor.type), tensor.dimensions, bytes.data(), bytes.size());
        if (tensor.type == GGML_TYPE_F16 && found->second->type == GGML_TYPE_F32) {
            const auto values = promote_f16(bytes);
            ggml_backend_tensor_set(found->second, values.data(), 0, values.size() * sizeof(float));
        } else {
            ggml_backend_tensor_set(found->second, bytes.data(), 0, bytes.size());
        }
        weight_bytes += ggml_nbytes(found->second);
    }
    state->tokenizer = std::move(file.tokenizer);
    state->model_info = {"sam3", file.precision, state->runtime->backend(), options.threads,
                   definition.tensors.size(), weight_bytes, false};
    state->model_info.task = file.video ? "text_video" : "text_image";
    state->model_info.profile = file.video ? "meta-sam3-temporal-v1" : "";
    state->model_info.storage_profile = file.storage_profile;
    state->model_info.arithmetic_profile = state->runtime->arithmetic_profile();
    state->model_info.activation_policy = state->runtime->activation_name();
    state->model_info.device_name = state->runtime->device_name();
    state->model_info.cuda_device = state->runtime->cuda_device();
    if (file.modular_quantized || file.mixed_quantized)
        state->model_info.quantization_modules = std::move(file.quantization_modules);
    state->model_info.base_precision = std::move(file.base_precision);
    state->model_info.module_precisions = std::move(file.module_precisions);
    state->model_info.policy_sha256 = std::move(file.policy_sha256);
    state->model_info.tensor_precisions = std::move(file.tensor_precisions);
    return state;
}

class ModelAdapter final : public sam::internal::ModelImplementation {
public:
    explicit ModelAdapter(std::shared_ptr<ModelState> state) : state_(std::move(state)) {}
    const ModelInfo& info() const override { return state_->model_info; }
    std::unique_ptr<sam::internal::TextImageSessionImplementation> create_text_image_session() override {
        return std::make_unique<ImageSession>(state_);
    }
    std::unique_ptr<sam::internal::TextVideoSessionImplementation> create_text_video_session(int count, VideoOptions options) override {
        return std::make_unique<VideoSession>(state_, count, options);
    }
private:
    std::shared_ptr<ModelState> state_;
};

inline std::shared_ptr<sam::internal::ModelImplementation> load_model(const std::string& path, BackendOptions options) {
    return std::make_shared<ModelAdapter>(load_state(path, options));
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_MODEL_HPP
