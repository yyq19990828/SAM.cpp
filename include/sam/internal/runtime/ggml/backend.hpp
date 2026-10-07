#ifndef SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKEND_HPP
#define SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKEND_HPP

#include "resources.hpp"
#include "sam/types.hpp"
#include <cstdint>
#include <string>

namespace sam::internal {

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
    AttentionExecutionPolicy attention;
    bool combine_graph_stages = false;
};

inline bool is_compute_node(const ggml_tensor* tensor) {
    return tensor->op != GGML_OP_NONE && tensor->op != GGML_OP_VIEW &&
        tensor->op != GGML_OP_RESHAPE && tensor->op != GGML_OP_PERMUTE && tensor->op != GGML_OP_TRANSPOSE;
}

} // namespace sam::internal

#endif // SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKEND_HPP
