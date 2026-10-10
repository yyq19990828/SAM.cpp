#ifndef SAM_CPP_TOOLS_BENCHMARK_GRAPH_PROFILE_HPP
#define SAM_CPP_TOOLS_BENCHMARK_GRAPH_PROFILE_HPP

#include <runtime/ggml/observer.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sam_profile {

struct Buffer {
    std::string name;
    std::size_t bytes = 0;
    bool host = false, weights = false;
};

struct Tensor {
    std::string name, type, operation, backend;
    std::array<std::int64_t, GGML_MAX_DIMS> shape{};
    std::array<std::size_t, GGML_MAX_DIMS> strides{};
    std::vector<int> sources;
    std::vector<int> source_slots;
    std::string accumulation_hint = "NOT_APPLICABLE", rhs_representation_hint = "NOT_APPLICABLE";
    int buffer = -1, view_source = -1, allocation_root = -1, node = -1;
    int first = 0, last = 0;
    std::size_t offset = 0, bytes = 0, view_offset = 0;
    bool owned = false, input = false, output = false;
};

inline std::string precision_hint(std::int32_t value) {
    switch (value) {
        case GGML_PREC_UNDEFINED: return "backend-selected";
        case GGML_PREC_F32: return "f32";
        case GGML_PREC_F16: return "f16";
        case GGML_PREC_BF16: return "bf16";
        case GGML_PREC_Q8: return "q8";
        case GGML_PREC_Q4: return "q4";
        default: return "unknown-" + std::to_string(value);
    }
}

inline void capture_precision_hints(const ggml_tensor* tensor, Tensor& row) {
    // Pinned GGML ggml_prec_set_acc/src op_params layout. These are hints,
    // not observations of backend-private staging, multiplication or sums.
    auto parameter = [&](std::size_t index) {
        std::int32_t value;
        std::memcpy(&value, reinterpret_cast<const char*>(tensor->op_params) + index * sizeof(value), sizeof(value));
        return precision_hint(value);
    };
    if (tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID) {
        row.accumulation_hint = parameter(0);
        row.rhs_representation_hint = parameter(3);
    } else if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
        row.accumulation_hint = parameter(3);
    }
}

struct Span {
    int buffer;
    std::size_t begin, end;
};

inline std::size_t union_bytes(std::vector<Span> spans) {
    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) {
        return a.buffer < b.buffer || (a.buffer == b.buffer && a.begin < b.begin);
    });
    std::size_t total = 0, end = 0;
    int buffer = -1;
    for (const auto& span : spans) {
        if (span.end < span.begin) throw std::runtime_error("invalid memory span");
        const auto begin = span.buffer == buffer ? std::max(end, span.begin) : span.begin;
        const auto added = span.end > begin ? span.end - begin : 0;
        if (added > std::numeric_limits<std::size_t>::max() - total)
            throw std::overflow_error("memory span sum overflow");
        total += added;
        end = span.buffer == buffer ? std::max(end, span.end) : span.end;
        buffer = span.buffer;
    }
    return total;
}

struct GraphSnapshot {
    std::vector<Tensor> tensors;
    std::vector<Buffer> buffers;
    int nodes = 0, peak_node = 0;
    std::size_t graph_live_span_peak_bytes = 0;
    std::size_t graph_host_span_peak_bytes = 0, graph_device_span_peak_bytes = 0;
    std::vector<int> peak_roots;
    std::vector<double> compute_ms;
};

// This is an operator-boundary live address-span estimate, not CUDA process
// memory. A view pins its entire source allocation until its last consumer.
// In-place results and other overlapping arena addresses count only once.
// Backend padding, scheduler copies not owned by this context, scratch pools
// and external resident buffers are reported separately, never added as if
// all independent high-water marks occurred simultaneously.
inline void analyze_lifetimes(GraphSnapshot& snapshot) {
    auto& tensors = snapshot.tensors;
    std::vector<int> roots;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        auto& tensor = tensors[i];
        int root = static_cast<int>(i);
        std::size_t depth = 0;
        while (tensors.at(root).view_source >= 0) {
            root = tensors.at(root).view_source;
            if (++depth > tensors.size()) throw std::runtime_error("cyclic tensor view");
        }
        tensor.allocation_root = root;
        tensor.first = std::max(0, tensor.node);
        tensor.last = tensor.output ? snapshot.nodes : tensor.first;
        if (root == static_cast<int>(i)) roots.push_back(root);
    }
    for (const auto& consumer : tensors) {
        const auto at = std::max(0, consumer.node);
        for (const int source : consumer.sources)
            tensors.at(source).last = std::max(tensors.at(source).last, at);
    }
    for (const auto& tensor : tensors) {
        auto& root = tensors.at(tensor.allocation_root);
        root.last = std::max(root.last, tensor.last);
    }
    std::vector<std::vector<int>> born(snapshot.nodes + 1), dead(snapshot.nodes + 2);
    for (const int id : roots) {
        const auto& tensor = tensors[id];
        if (!tensor.owned || tensor.buffer < 0 || snapshot.buffers.at(tensor.buffer).weights) continue;
        born.at(tensor.first).push_back(id);
        dead.at(tensor.last + 1).push_back(id);
    }
    std::set<int> active;
    for (int node = 0; node <= snapshot.nodes; ++node) {
        for (const auto id : dead[node]) active.erase(id);
        for (const auto id : born[node]) active.insert(id);
        std::vector<Span> host, device, all;
        for (const int id : active) {
            const auto& tensor = tensors[id];
            if (tensor.bytes > std::numeric_limits<std::size_t>::max() - tensor.offset)
                throw std::overflow_error("tensor address span overflow");
            const Span span{tensor.buffer, tensor.offset, tensor.offset + tensor.bytes};
            all.push_back(span);
            (snapshot.buffers[tensor.buffer].host ? host : device).push_back(span);
        }
        const auto total = union_bytes(all);
        snapshot.graph_host_span_peak_bytes = std::max(snapshot.graph_host_span_peak_bytes, union_bytes(host));
        snapshot.graph_device_span_peak_bytes = std::max(snapshot.graph_device_span_peak_bytes, union_bytes(device));
        if (total > snapshot.graph_live_span_peak_bytes) {
            snapshot.graph_live_span_peak_bytes = total;
            snapshot.peak_node = node;
            snapshot.peak_roots.assign(active.begin(), active.end());
        }
    }
}

inline GraphSnapshot capture_graph(ggml_context* context, ggml_cgraph* graph,
                                    ggml_backend_sched_t scheduler) {
    GraphSnapshot result;
    result.nodes = ggml_graph_n_nodes(graph);
    std::set<const ggml_tensor*> owned;
    for (auto* tensor = ggml_get_first_tensor(context); tensor; tensor = ggml_get_next_tensor(context, tensor))
        owned.insert(tensor);
    std::vector<ggml_tensor*> tensors;
    std::unordered_map<ggml_tensor*, int> ids;
    auto add = [&](ggml_tensor* tensor) {
        if (tensor && ids.emplace(tensor, static_cast<int>(tensors.size())).second) tensors.push_back(tensor);
    };
    for (int i = 0; i < result.nodes; ++i) add(ggml_graph_node(graph, i));
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        auto* tensor = tensors[i];
        for (auto* source : tensor->src) add(source);
        add(tensor->view_src);
    }
    std::unordered_map<ggml_tensor*, int> node_ids;
    for (int i = 0; i < result.nodes; ++i) node_ids[ggml_graph_node(graph, i)] = i;
    std::map<ggml_backend_buffer_t, int> buffers;
    for (auto* tensor : tensors) {
        Tensor row;
        row.name = ggml_get_name(tensor);
        row.type = ggml_type_name(tensor->type);
        row.operation = ggml_op_name(tensor->op);
        capture_precision_hints(tensor, row);
        row.owned = owned.count(tensor) != 0;
        row.input = tensor->flags & GGML_TENSOR_FLAG_INPUT;
        row.output = tensor->flags & GGML_TENSOR_FLAG_OUTPUT;
        std::copy(std::begin(tensor->ne), std::end(tensor->ne), row.shape.begin());
        std::copy(std::begin(tensor->nb), std::end(tensor->nb), row.strides.begin());
        row.bytes = ggml_nbytes(tensor);
        const auto node = node_ids.find(tensor);
        if (node != node_ids.end()) row.node = node->second;
        if (auto* backend = ggml_backend_sched_get_tensor_backend(scheduler, tensor))
            row.backend = ggml_backend_name(backend);
        if (tensor->view_src) row.view_source = ids.at(tensor->view_src);
        row.view_offset = tensor->view_offs;
        for (int slot = 0; slot < GGML_MAX_SRC; ++slot) {
            if (tensor->src[slot]) {
                row.sources.push_back(ids.at(tensor->src[slot]));
                row.source_slots.push_back(slot);
            }
        }
        if (tensor->buffer && tensor->data) {
            const auto found = buffers.emplace(tensor->buffer, static_cast<int>(result.buffers.size()));
            row.buffer = found.first->second;
            if (found.second) result.buffers.push_back({ggml_backend_buffer_name(tensor->buffer),
                ggml_backend_buffer_get_size(tensor->buffer), ggml_backend_buffer_is_host(tensor->buffer),
                ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS});
            const auto base = reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
            const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
            if (address < base || address - base > result.buffers[row.buffer].bytes ||
                row.bytes > result.buffers[row.buffer].bytes - (address - base))
                throw std::runtime_error("tensor span is outside its backend buffer");
            row.offset = address - base;
        }
        result.tensors.push_back(std::move(row));
    }
    analyze_lifetimes(result);
    return result;
}

} // namespace sam_profile

#endif // SAM_CPP_TOOLS_BENCHMARK_GRAPH_PROFILE_HPP
