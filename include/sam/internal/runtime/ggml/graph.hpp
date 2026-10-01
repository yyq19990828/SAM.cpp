#ifndef SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_GRAPH_HPP
#define SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_GRAPH_HPP

#include "resources.hpp"
#include "runtime.hpp"
#include "sam/types.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam::internal {

class GraphExecution {
public:
    GraphExecution(GgmlRuntime& runtime, std::size_t graph_size, RuntimeStats& stats)
        : context_(make_context(graph_size * 2, graph_size)), runtime_(runtime), stats_(stats) {
        graph_ = ggml_new_graph_custom(context_.get(), graph_size, false);
        auto backends = runtime_.backends();
        scheduler_.reset(ggml_backend_sched_new(backends.data(), nullptr, static_cast<int>(backends.size()),
                                              graph_size, false, true));
        if (!scheduler_) throw std::runtime_error("failed to create GGML scheduler");
    }
    ggml_context* context() const { return context_.get(); }
    void output(ggml_tensor* tensor) {
        ggml_set_output(tensor);
        ggml_build_forward_expand(graph_, tensor);
    }
    void allocate() {
        // The scheduler asserts when no backend accepts a node. Check first so
        // incompatible operators remain a runtime error at the library boundary.
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            auto* node = ggml_graph_node(graph_, i);
            if (node->op == GGML_OP_MUL_MAT) ggml_prec_set_acc(node, GGML_PREC_F32);
            if (node->op == GGML_OP_FLASH_ATTN_EXT) ggml_prec_set_acc(node, GGML_PREC_F32);
            bool supported = false;
            for (auto* backend : runtime_.backends()) supported = supported || ggml_backend_supports_op(backend, node);
            if (!supported) {
                std::string message = std::string("no backend supports SAM operation ") +
                    ggml_op_name(node->op) + " tensor " + ggml_get_name(node);
                for (int source = 0; source < 2; ++source) if (node->src[source])
                    message += " source" + std::to_string(source) + "=" + ggml_get_name(node->src[source]) +
                        "/" + ggml_type_name(node->src[source]->type);
                throw std::runtime_error(message);
            }
        }
        if (!ggml_backend_sched_alloc_graph(scheduler_.get(), graph_))
            throw std::runtime_error("failed to allocate SAM graph");
    }
    void compute() {
        if (ggml_backend_sched_graph_compute(scheduler_.get(), graph_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("SAM graph execution failed");
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            auto* tensor = ggml_graph_node(graph_, i);
            if (tensor->op == GGML_OP_NONE || tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE ||
                tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_TRANSPOSE) continue;
            auto* backend = ggml_backend_sched_get_tensor_backend(scheduler_.get(), tensor);
            runtime_.record_node(backend, stats_);
        }
        stats_.graph_partitions += ggml_backend_sched_get_n_splits(scheduler_.get());
        std::size_t bytes = 0;
        for (auto* backend : runtime_.backends()) bytes += ggml_backend_sched_get_buffer_size(scheduler_.get(), backend);
        stats_.compute_buffer_bytes = std::max(stats_.compute_buffer_bytes, bytes);
    }
private:
    ContextPtr context_;
    SchedulerPtr scheduler_;
    GgmlRuntime& runtime_;
    RuntimeStats& stats_;
    ggml_cgraph* graph_ = nullptr;
};

inline ggml_tensor* input_tensor(ggml_context* ctx, const char* name,
                                int64_t d0, int64_t d1 = 1, int64_t d2 = 1, int64_t d3 = 1,
                                ggml_type type = GGML_TYPE_F32) {
    auto* tensor = ggml_new_tensor_4d(ctx, type, d0, d1, d2, d3);
    ggml_set_name(tensor, name);
    ggml_set_input(tensor);
    return tensor;
}

inline void upload(ggml_tensor* tensor, const std::vector<float>& values, RuntimeStats& stats) {
    if (!tensor->buffer || ggml_nbytes(tensor) != values.size() * sizeof(float))
        throw std::runtime_error("invalid SAM graph input allocation or shape");
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    stats.host_upload_bytes += values.size() * sizeof(float);
}
inline std::vector<float> download(ggml_tensor* tensor, RuntimeStats& stats) {
    if (!tensor->buffer || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor))
        throw std::runtime_error("invalid SAM graph output allocation or type");
    std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    stats.host_download_bytes += values.size() * sizeof(float);
    return values;
}

} // namespace sam::internal

#endif // SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_GRAPH_HPP
