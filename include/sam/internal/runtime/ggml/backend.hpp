#ifndef SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKEND_HPP
#define SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKEND_HPP

#include "resources.hpp"
#include "sam/types.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sam::internal {

// Internal experiments only; no public model-loading option selects these.
enum class FeatureCacheMode { F32, F16, Q8_0 };

inline ggml_type feature_cache_storage_type(FeatureCacheMode mode) {
    switch (mode) {
        case FeatureCacheMode::F32: return GGML_TYPE_F32;
        case FeatureCacheMode::F16: return GGML_TYPE_F16;
        case FeatureCacheMode::Q8_0: return GGML_TYPE_Q8_0;
    }
    throw std::invalid_argument("unknown SAM feature cache mode");
}

struct AttentionExecutionPolicy {
    int query_tile = 128;
    bool assemble_in_place = false;
    bool fused_memory = false;
};

// Each implemented device supplies ownership and initialization-time policies.
// The counter points to its existing public statistics field; scheduling never
// guesses a device kind from an unrecognized GGML handle.
struct BackendDriver {
    Backend kind = Backend::Auto;
    BackendPtr handle;
    bool promote_f16_weights = false;
    std::uint64_t RuntimeStats::* node_counter = nullptr;
    std::string device_name;
    int cuda_device = -1;
    // Device-owned operator policy; shared model graphs never select CUDA kernels.
    void (*configure_node)(ggml_tensor*) = nullptr;
    bool reduced_precision = false;
    ggml_type convolution_columns_type = GGML_TYPE_F32;
    ggml_type feature_cache_type = GGML_TYPE_F32;
    AttentionExecutionPolicy attention;
    bool combine_graph_stages = false;
};

inline bool is_compute_node(const ggml_tensor* tensor) {
    return tensor->op != GGML_OP_NONE && tensor->op != GGML_OP_VIEW &&
        tensor->op != GGML_OP_RESHAPE && tensor->op != GGML_OP_PERMUTE && tensor->op != GGML_OP_TRANSPOSE;
}

} // namespace sam::internal

#endif // SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKEND_HPP
