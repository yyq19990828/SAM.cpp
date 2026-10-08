#ifndef SAM_CPP_SRC_RUNTIME_GGML_BACKENDS_METAL_HPP
#define SAM_CPP_SRC_RUNTIME_GGML_BACKENDS_METAL_HPP

#include "../backend.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam::internal {

inline void validate_metal_precision(ggml_backend_t backend) {
    auto context = make_context(8, 8);
    auto* a = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 256, 256);
    auto* b = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 256, 32);
    auto* output = ggml_mul_mat(context.get(), a, b);
    ggml_prec_set_acc(output, GGML_PREC_F32);
    auto* graph = ggml_new_graph_custom(context.get(), 8, false);
    ggml_build_forward_expand(graph, output);
    BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) throw std::runtime_error("failed to allocate Metal precision check");
    const std::vector<float> left(256 * 256, 1.0003f), right(256 * 32, 1.0003f);
    ggml_backend_tensor_set(a, left.data(), 0, left.size() * sizeof(float));
    ggml_backend_tensor_set(b, right.data(), 0, right.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Metal precision compatibility check failed to execute");
    float sample = 0.0f;
    ggml_backend_tensor_get(output, &sample, 0, sizeof(sample));
    const float expected = 256.0f * 1.0003f * 1.0003f;
    if (!std::isfinite(sample) || std::abs(sample - expected) > std::abs(expected) * 3.0e-6f)
        throw std::runtime_error("GGML Metal lacks required F32 matrix precision; use the SAM precision patch");
}

inline BackendDriver make_metal_backend() {
    BackendDriver driver{Backend::Metal, BackendPtr{}, false, &RuntimeStats::metal_nodes};
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto* device = ggml_backend_dev_get(i);
        const std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        if (name == "MTL") {
            driver.handle.reset(ggml_backend_dev_init(device, nullptr));
            driver.device_name = ggml_backend_dev_description(device);
            break;
        }
    }
    if (driver.handle) validate_metal_precision(driver.handle.get());
    return driver;
}

} // namespace sam::internal

#endif // SAM_CPP_SRC_RUNTIME_GGML_BACKENDS_METAL_HPP
