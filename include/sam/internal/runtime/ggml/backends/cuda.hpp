#ifndef SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKENDS_CUDA_HPP
#define SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKENDS_CUDA_HPP

#include "../backend.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam::internal {

// Check caller-owned GGML too: linking a CUDA target does not prove operator
// availability or that its dense/attention kernels preserve F32 activations.
inline void validate_cuda_precision(ggml_backend_t backend) {
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        auto context = make_context(8, 8);
        auto* left = ggml_new_tensor_2d(context.get(), type, 257, 65);
        auto* right = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 257, 33);
        auto* output = ggml_mul_mat(context.get(), left, right);
        ggml_prec_set_acc(output, GGML_PREC_F32);
        if (!ggml_backend_supports_op(backend, output))
            throw std::runtime_error("GGML CUDA lacks required F32 matrix operations");
        auto* graph = ggml_new_graph_custom(context.get(), 8, false);
        ggml_build_forward_expand(graph, output);
        BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        if (!buffer) throw std::runtime_error("failed to allocate CUDA precision check");
        const std::vector<float> lhs(257 * 65, 1.0003f), rhs(257 * 33, 1.0003f);
        if (type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> stored(lhs.size());
            ggml_fp32_to_fp16_row(lhs.data(), stored.data(), static_cast<int64_t>(stored.size()));
            ggml_backend_tensor_set(left, stored.data(), 0, stored.size() * sizeof(ggml_fp16_t));
        } else ggml_backend_tensor_set(left, lhs.data(), 0, lhs.size() * sizeof(float));
        ggml_backend_tensor_set(right, rhs.data(), 0, rhs.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CUDA matrix compatibility check failed to execute");
        std::vector<float> values(65 * 33);
        ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
        const double expected = 257.0 * (type == GGML_TYPE_F16 ? 1.0 : double(lhs[0])) * double(rhs[0]);
        for (const float value : values)
            if (!std::isfinite(value) || std::abs(double(value) - expected) > 0.002)
                throw std::runtime_error("GGML CUDA lacks required F32 matrix precision; use the SAM CUDA patch");
    }
    for (const int dimensions : {32, 64}) {
        auto context = make_context(8, 8);
        auto* q = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimensions, 1, 1);
        auto* k = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimensions, 130, 1);
        auto* v = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dimensions, 130, 1);
        const float scale = 1.0f / std::sqrt(float(dimensions));
        auto* output = ggml_flash_attn_ext(context.get(), q, k, v, nullptr, scale, 0, 0);
        ggml_prec_set_acc(output, GGML_PREC_F32);
        if (!ggml_backend_supports_op(backend, output))
            throw std::runtime_error("GGML CUDA lacks required head-32/64 F32 attention; use the SAM CUDA patch");
        auto* graph = ggml_new_graph_custom(context.get(), 8, false);
        ggml_build_forward_expand(graph, output);
        BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        if (!buffer) throw std::runtime_error("failed to allocate CUDA attention check");
        const std::vector<float> query(dimensions, 1.0003f);
        std::vector<float> keys(dimensions * 130), values(dimensions * 130);
        for (int token = 0; token < 130; ++token)
            for (int dimension = 0; dimension < dimensions; ++dimension) {
                keys[token * dimensions + dimension] = token < 65 ? 0.0625f : -0.0625f;
                values[token * dimensions + dimension] = token < 65 ? 1.0f : -1.0f;
            }
        ggml_backend_tensor_set(q, query.data(), 0, query.size() * sizeof(float));
        ggml_backend_tensor_set(k, keys.data(), 0, keys.size() * sizeof(float));
        ggml_backend_tensor_set(v, values.data(), 0, values.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CUDA attention compatibility check failed to execute");
        std::vector<float> actual(dimensions);
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        const double expected = std::tanh(dimensions * 0.0625 * double(scale) * double(query[0]));
        for (const float value : actual)
            if (!std::isfinite(value) || std::abs(double(value) - expected) > 1e-5)
                throw std::runtime_error("GGML CUDA lacks required F32 attention precision; use the SAM CUDA patch");
    }
}

inline BackendDriver make_cuda_backend(int device_index) {
    BackendDriver driver{Backend::Cuda, BackendPtr{}, false, &RuntimeStats::cuda_nodes};
    auto* registration = ggml_backend_reg_by_name("CUDA");
    if (!registration)
        throw std::runtime_error("requested GGML CUDA backend is not registered; build with GGML_CUDA=ON");
    const auto count = ggml_backend_reg_dev_count(registration);
    if (count == 0) throw std::runtime_error("requested GGML CUDA backend has no visible devices");
    if (device_index < 0 || static_cast<std::size_t>(device_index) >= count)
        throw std::runtime_error("requested CUDA device index is out of range for visible devices");
    auto* device = ggml_backend_reg_dev_get(registration, static_cast<std::size_t>(device_index));
    driver.handle.reset(ggml_backend_dev_init(device, nullptr));
    if (!driver.handle) throw std::runtime_error("requested GGML CUDA backend failed initialization");
    driver.device_name = ggml_backend_dev_description(device);
    driver.cuda_device = device_index;
    validate_cuda_precision(driver.handle.get());
    return driver;
}

} // namespace sam::internal

#endif // SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKENDS_CUDA_HPP
