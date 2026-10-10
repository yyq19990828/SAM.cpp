#ifndef SAM_CPP_SRC_RUNTIME_GGML_GRAPH_HPP
#define SAM_CPP_SRC_RUNTIME_GGML_GRAPH_HPP

#include "resources.hpp"
#include "runtime.hpp"
#include "workspace.hpp"
#include "transfer.hpp"
#include "sam/types.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sam::internal {

class GraphExecution {
public:
    GraphExecution(GgmlRuntime& runtime, std::size_t graph_size, RuntimeStats& stats,
                   std::shared_ptr<GraphWorkspace> workspace = {})
        : graph_capacity_(expanded_graph_capacity(graph_size, runtime)),
          context_(make_context(graph_capacity_ * 2, graph_capacity_)), runtime_(runtime), stats_(stats),
          workspace_(workspace ? std::move(workspace) : std::make_shared<GraphWorkspace>(runtime, graph_capacity_)) {
        if (&workspace_->runtime() != &runtime_)
            throw std::invalid_argument("SAM graph workspace belongs to another runtime");
        graph_ = ggml_new_graph_custom(context_.get(), graph_capacity_, false);
        workspace_->record_build();
    }
    ~GraphExecution() { workspace_->discard(context_.get()); }
    ggml_context* context() const { return context_.get(); }
    void output(ggml_tensor* tensor) {
        if (graph_validated_) throw std::runtime_error("cannot extend an allocated SAM graph");
        ggml_set_output(tensor);
        roots_.push_back(tensor);
        ggml_build_forward_expand(graph_, tensor);
    }
    void allocate(std::size_t arena_limit = std::numeric_limits<std::size_t>::max()) {
        prepare();
        if (arena_limit != std::numeric_limits<std::size_t>::max()) {
            if (buffer_requirements_.empty()) required_workspace_bytes();
            workspace_->fit_allocation_plan(buffer_requirements_, arena_limit);
        }
        workspace_->bind(context_.get(), graph_);
    }
    std::size_t required_workspace_bytes() {
        prepare();
        return workspace_->required_bytes(context_.get(), graph_, &buffer_requirements_);
    }
    const GraphDiagnostics& diagnostics() const { return workspace_->diagnostics(); }
private:
    void prepare() {
        if (graph_validated_) return;
        prepare_quantized_cpu_matmuls();
        // The scheduler asserts when no backend accepts a node. Check first so
        // incompatible operators remain a runtime error at the library boundary.
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            auto* node = ggml_graph_node(graph_, i);
            if (node->op == GGML_OP_MUL_MAT) ggml_prec_set_acc(node, GGML_PREC_F32);
            if (node->op == GGML_OP_FLASH_ATTN_EXT) ggml_prec_set_acc(node, GGML_PREC_F32);
            runtime_.configure_node(node);
            const auto required_backend = runtime_.required_compute_backend(node);
            bool supported = required_backend ? ggml_backend_supports_op(required_backend, node) : false;
            if (!required_backend)
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
        graph_validated_ = true;
    }
public:
    void compute() {
        workspace_->compute(context_.get(), graph_);
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            auto* tensor = ggml_graph_node(graph_, i);
            if (!is_compute_node(tensor)) continue;
            auto* backend = ggml_backend_sched_get_tensor_backend(workspace_->scheduler(), tensor);
            runtime_.record_node(backend, stats_);
        }
        stats_.graph_partitions += ggml_backend_sched_get_n_splits(workspace_->scheduler());
        stats_.compute_buffer_bytes = std::max(stats_.compute_buffer_bytes, workspace_->buffer_bytes());
    }
private:
    static std::size_t expanded_graph_capacity(std::size_t graph_size, const GgmlRuntime& runtime) {
        if (graph_size == 0 || graph_size > std::numeric_limits<std::size_t>::max() / 4)
            throw std::invalid_argument("invalid SAM graph capacity");
        const auto capacity = graph_size * (runtime.quantized_cpu_f32_weights() ? 2 : 1);
        if (capacity > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("SAM graph capacity exceeds GGML scheduler limits");
        return capacity;
    }

    void prepare_quantized_cpu_matmuls() {
        if (!runtime_.quantized_cpu_f32_weights() || graph_prepared_) {
            graph_prepared_ = true;
            return;
        }
        std::vector<ggml_tensor*> original_nodes;
        const auto node_count = ggml_graph_n_nodes(graph_);
        original_nodes.reserve(static_cast<std::size_t>(node_count));
        for (int i = 0; i < node_count; ++i) original_nodes.push_back(ggml_graph_node(graph_, i));

        std::unordered_map<ggml_tensor*, ggml_tensor*> f32_weights;
        for (auto* node : original_nodes) {
            if (node->op != GGML_OP_MUL_MAT || !node->src[0] || !ggml_is_quantized(node->src[0]->type)) continue;
            auto* source = node->src[0];
            auto found = f32_weights.find(source);
            if (found == f32_weights.end())
                found = f32_weights.emplace(source, ggml_cast(context_.get(), source, GGML_TYPE_F32)).first;
            node->src[0] = found->second;
        }

        if (!f32_weights.empty()) {
            ggml_graph_clear(graph_);
            for (auto* root : roots_) {
                ggml_set_output(root);
                ggml_build_forward_expand(graph_, root);
            }
        }
        graph_prepared_ = true;
    }

    std::size_t graph_capacity_ = 0;
    ContextPtr context_;
    GgmlRuntime& runtime_;
    RuntimeStats& stats_;
    std::shared_ptr<GraphWorkspace> workspace_;
    ggml_cgraph* graph_ = nullptr;
    std::vector<ggml_tensor*> roots_;
    std::vector<std::size_t> buffer_requirements_;
    bool graph_prepared_ = false;
    bool graph_validated_ = false;
};

inline ggml_tensor* input_tensor(ggml_context* ctx, const char* name,
                                int64_t d0, int64_t d1 = 1, int64_t d2 = 1, int64_t d3 = 1,
                                ggml_type type = GGML_TYPE_F32) {
    auto* tensor = ggml_new_tensor_4d(ctx, type, d0, d1, d2, d3);
    ggml_set_name(tensor, name);
    ggml_set_input(tensor);
    return tensor;
}

inline void upload(ggml_tensor* tensor, const std::vector<float>& values, RuntimeStats& stats,
                   GgmlRuntime* runtime = nullptr) {
    if (!tensor->buffer || ggml_nbytes(tensor) != values.size() * sizeof(float))
        throw std::runtime_error("invalid SAM graph input allocation or shape");
    observe_transfer(runtime, tensor, "upload", values.size() * sizeof(float), [&] {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    });
    stats.host_upload_bytes += values.size() * sizeof(float);
}
inline std::vector<float> download(ggml_tensor* tensor, RuntimeStats& stats, GgmlRuntime* runtime = nullptr) {
    if (!tensor->buffer || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor))
        throw std::runtime_error("invalid SAM graph output allocation or type");
    std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
    observe_transfer(runtime, tensor, "download", values.size() * sizeof(float), [&] {
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    });
    stats.host_download_bytes += values.size() * sizeof(float);
    return values;
}

} // namespace sam::internal

#endif // SAM_CPP_SRC_RUNTIME_GGML_GRAPH_HPP
