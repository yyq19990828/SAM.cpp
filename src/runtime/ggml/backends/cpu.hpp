#ifndef SAM_CPP_SRC_RUNTIME_GGML_BACKENDS_CPU_HPP
#define SAM_CPP_SRC_RUNTIME_GGML_BACKENDS_CPU_HPP

#include "../backend.hpp"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <stdexcept>

namespace sam::internal {

// These traits describe the pinned CPU dot path, not a graph activation dtype
// or a measurement of the dispatched SIMD/tiled kernel's internal operations.
inline ggml_type cpu_quantized_rhs_type(ggml_type weight_type) {
    if (weight_type != GGML_TYPE_Q8_0 && weight_type != GGML_TYPE_Q6_K &&
        weight_type != GGML_TYPE_Q5_K && weight_type != GGML_TYPE_Q4_K)
        throw std::invalid_argument("native CPU matmul supports Q8_0/Q6_K/Q5_K/Q4_K weights only");
    const auto* traits = ggml_get_type_traits_cpu(weight_type);
    const auto expected = weight_type == GGML_TYPE_Q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K;
    if (!traits || !traits->vec_dot || traits->vec_dot_type != expected ||
        !ggml_get_type_traits_cpu(expected)->from_float)
        throw std::runtime_error("pinned GGML CPU quantized matmul traits are unavailable");
    return expected;
}

inline void validate_cpu_quantized_matmul(ggml_tensor* node) {
    if (node->op != GGML_OP_MUL_MAT || !node->src[0] || !ggml_is_quantized(node->src[0]->type)) return;
    const auto* weight = node->src[0];
    const auto* rhs = node->src[1];
    cpu_quantized_rhs_type(weight->type);
    if (!rhs || rhs->type != GGML_TYPE_F32 || rhs->nb[0] != sizeof(float) ||
        weight->nb[0] != ggml_type_size(weight->type) || node->type != GGML_TYPE_F32)
        throw std::invalid_argument("native CPU quantized matmul requires row-contiguous packed weights and F32 RHS/output");
}

inline BackendDriver make_cpu_backend(int threads) {
    BackendDriver driver{Backend::Cpu, BackendPtr{}, true, &RuntimeStats::cpu_nodes};
    auto* cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_device) throw std::runtime_error("GGML CPU backend is unavailable");
    driver.handle.reset(ggml_backend_dev_init(cpu_device, nullptr));
    if (!driver.handle) throw std::runtime_error("GGML CPU backend initialization failed");
    driver.device_name = ggml_backend_dev_description(cpu_device);
    auto* registration = ggml_backend_dev_backend_reg(cpu_device);
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(registration, "ggml_backend_set_n_threads"));
    if (!set_threads) throw std::runtime_error("GGML CPU backend lacks thread configuration");
    set_threads(driver.handle.get(), threads);
    return driver;
}

inline BackendDriver make_cpu_blas_backend(int threads) {
    BackendDriver driver{Backend::Cpu, BackendPtr{}, true, &RuntimeStats::blas_nodes};
    auto* registration = ggml_backend_reg_by_name("BLAS");
    if (!registration || ggml_backend_reg_dev_count(registration) == 0) return driver;
    auto* device = ggml_backend_reg_dev_get(registration, 0);
    driver.handle.reset(ggml_backend_dev_init(device, nullptr));
    if (!driver.handle) throw std::runtime_error("GGML BLAS backend initialization failed");
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(registration, "ggml_backend_set_n_threads"));
    if (!set_threads) throw std::runtime_error("GGML BLAS backend lacks thread configuration");
    set_threads(driver.handle.get(), threads);
    return driver;
}

} // namespace sam::internal

#endif // SAM_CPP_SRC_RUNTIME_GGML_BACKENDS_CPU_HPP
