#ifndef SAM_CPP_SRC_RUNTIME_GGML_BACKENDS_CUDA_ACTIVATION_HPP
#define SAM_CPP_SRC_RUNTIME_GGML_BACKENDS_CUDA_ACTIVATION_HPP

#include "../linear_role.hpp"
#include "../resources.hpp"
#include "ggml-backend.h"
#include "sam/types.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#if __has_include("ggml-sam.h")
#include "ggml-sam.h"
#define SAM_CPP_HAS_Q8_RHS_ABI 1
#endif

namespace sam::internal {

class CudaActivationPolicy {
public:
    void initialize(int device) {
#ifdef SAM_CPP_HAS_Q8_RHS_ABI
        const auto reg = ggml_backend_reg_by_name("CUDA");
        configure_ = reinterpret_cast<ggml_sam_q8_rhs_configure>(ggml_backend_reg_get_proc_address(reg, "ggml_cuda_sam_q8_rhs_configure_v1"));
        query_ = reinterpret_cast<ggml_sam_q8_rhs_query>(ggml_backend_reg_get_proc_address(reg, "ggml_cuda_sam_q8_rhs_query_v1"));
        if (!configure_ || !query_) throw std::runtime_error("GGML lacks explicit Q8 RHS version-1 capability");
        device_ = device;
        auto context = make_context(4);
        auto* a = ggml_new_tensor_2d(context.get(), GGML_TYPE_Q8_0, 32, 32);
        auto* b = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 32, 1);
        if (!configure_(ggml_mul_mat(context.get(), a, b), device_))
            throw std::runtime_error("explicit Q8 RHS requires supported CUDA hardware (compute capability >= 8.9)");
        enabled_ = true;
#else
        (void)device;
        throw std::runtime_error("GGML lacks explicit Q8 RHS version-1 capability");
#endif
    }
    bool selected(const ggml_tensor* node) const { return enabled_ && is_vision_linear(node); }
    void configure(ggml_tensor* node) const {
#ifdef SAM_CPP_HAS_Q8_RHS_ABI
        if (selected(node) && !configure_(node, device_))
            throw std::runtime_error("explicit Q8 RHS cannot honor vision linear shape/storage");
#else
        (void)node;
#endif
    }
    std::uint64_t launches(const ggml_tensor* node) const {
#ifdef SAM_CPP_HAS_Q8_RHS_ABI
        ggml_sam_q8_rhs_info info{};
        if (!query_(node, &info) || info.version != 1) throw std::runtime_error("explicit Q8 RHS execution identity missing");
        return info.launches;
#else
        (void)node;
        throw std::runtime_error("explicit Q8 RHS capability unavailable");
#endif
    }
    void record(const ggml_tensor* node, std::uint64_t before, RuntimeStats& stats) const {
#ifdef SAM_CPP_HAS_Q8_RHS_ABI
        ggml_sam_q8_rhs_info info{};
        if (!query_(node, &info) || info.version != 1 || before == UINT64_MAX || info.launches != before + 1 ||
            info.path != 1 || !info.payload_bytes || info.allocation_bytes < info.payload_bytes)
            throw std::runtime_error("explicit Q8 RHS request did not execute exactly one actual MMQ dispatch");
        ++stats.activation_requests;
        ++stats.activation_mmq_launches;
        stats.activation_payload_peak_bytes = std::max(stats.activation_payload_peak_bytes, info.payload_bytes);
        stats.activation_allocation_peak_bytes = std::max(stats.activation_allocation_peak_bytes, info.allocation_bytes);
#else
        (void)node; (void)before; (void)stats;
#endif
    }
    bool enabled() const { return enabled_; }
private:
    bool enabled_ = false;
#ifdef SAM_CPP_HAS_Q8_RHS_ABI
    int device_ = -1;
    ggml_sam_q8_rhs_configure configure_ = nullptr;
    ggml_sam_q8_rhs_query query_ = nullptr;
#endif
};

} // namespace sam::internal

#ifdef SAM_CPP_HAS_Q8_RHS_ABI
#undef SAM_CPP_HAS_Q8_RHS_ABI
#endif
#endif // SAM_CPP_SRC_RUNTIME_GGML_BACKENDS_CUDA_ACTIVATION_HPP
