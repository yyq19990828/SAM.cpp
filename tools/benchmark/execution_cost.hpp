#ifndef SAM_CPP_TOOLS_BENCHMARK_EXECUTION_COST_HPP
#define SAM_CPP_TOOLS_BENCHMARK_EXECUTION_COST_HPP

#include "graph_profile.hpp"
#include "../../support/image_io/image_support.hpp"
#include <runtime/ggml/runtime.hpp>
#include <chrono>
#include <exception>
#include <map>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sam_cost {

struct Operand {
    std::string name, type;
    std::size_t bytes = 0;
};

struct Node {
    std::size_t graph = 0, calls = 0, output_bytes = 0;
    std::string name, operation, category, backend, output_type;
    std::vector<Operand> inputs;
    std::string accumulation_hint, rhs_hint;
    std::string cpu_rhs_dot_type; // Pinned type traits, not a capture of kernel scratch.
    double wall_ms = 0;
};

struct Graph {
    std::string stage;
    double bind_ms = 0, compute_wall_ms = 0;
    std::size_t arena_bytes = 0, compute_calls = 0;
    bool reused = false;
};

struct Transfer {
    std::string stage, tensor, direction, type;
    std::size_t bytes = 0;
    double wall_ms = 0;
};

class Observer final : public sam::internal::GraphObserver {
public:
    std::string stage = "unspecified";
    std::vector<Node> nodes;
    std::vector<Graph> graphs;
    std::vector<Transfer> transfers;

    void allocated(ggml_context* context, ggml_cgraph* graph, ggml_backend_sched_t scheduler) override {
        active_context_ = context;
        active_graph_ = graph;
        scheduler_ = scheduler;
        active_nodes_.clear();
        graphs.push_back({stage});
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) capture(ggml_graph_node(graph, i));
    }

    void bound(double milliseconds, std::size_t bytes, bool reused) override {
        if (graphs.empty()) throw std::runtime_error("cost observer received an unbound graph");
        graphs.back().bind_ms = milliseconds;
        graphs.back().arena_bytes = bytes;
        graphs.back().reused = reused;
    }

    void computing(ggml_context* context, ggml_cgraph* graph, ggml_backend_sched_t scheduler) override {
        if (active_context_ != context || active_graph_ != graph || graphs.empty())
            throw std::runtime_error("cost observer must be attached before graph allocation");
        callback_error_ = {};
        ggml_backend_sched_set_eval_callback(scheduler, evaluate, this);
    }

    void computed(ggml_context* context, ggml_cgraph* graph, double milliseconds) override {
        if (callback_error_) std::rethrow_exception(callback_error_);
        if (context != active_context_ || graph != active_graph_)
            throw std::runtime_error("cost observer computed an unknown graph");
        ++graphs.back().compute_calls;
        graphs.back().compute_wall_ms += milliseconds;
    }

    void transferred(ggml_tensor* tensor, const char* direction, std::size_t bytes, double milliseconds) override {
        transfers.push_back({stage, ggml_get_name(tensor), direction, ggml_type_name(tensor->type), bytes, milliseconds});
    }

    void write(std::ostream& output) const {
        using sam_example::json_string;
        std::map<std::string, std::pair<std::size_t, double>> categories;
        for (const auto& node : nodes) {
            auto& value = categories[node.category];
            value.first += node.calls;
            value.second += node.wall_ms;
        }
        double graph_ms = 0, bind_ms = 0;
        std::size_t arena_peak = 0;
        for (const auto& graph : graphs) {
            graph_ms += graph.compute_wall_ms;
            bind_ms += graph.bind_ms;
            arena_peak = std::max(arena_peak, graph.arena_bytes);
        }
        std::map<std::string, Transfer> transfer_totals;
        for (const auto& transfer : transfers) {
            auto& total = transfer_totals[transfer.direction];
            total.bytes += transfer.bytes;
            total.wall_ms += transfer.wall_ms;
        }
        output << "{\"schema_version\":1,\"kind\":\"sam-cpu-execution-cost-v1\",\"complete\":true,"
                  "\"diagnostic_only\":true,\"performance_comparable\":false,\"backend\":\"cpu\","
                  "\"observation_mode\":\"serialized scheduler evaluation callback; synchronization and dispatch overhead included\","
                  "\"timing_scope\":\"steady-clock synchronous wall time; node times are included in graph time; do not add both\","
                  "\"kernel_internal_rhs_packing\":\"NOT_COLLECTED\",\"kernel_internal_arithmetic\":\"NOT_COLLECTED\","
                  "\"weight_load_transfers\":\"NOT_COLLECTED\",\"host_allocation_and_preprocessing\":\"NOT_COLLECTED\","
                  "\"graph_build_and_reserve\":\"NOT_COLLECTED\","
                  "\"graph_compute_wall_ms\":" << graph_ms << ",\"graph_bind_wall_ms\":" << bind_ms
               << ",\"graph_arena_peak_bytes\":" << arena_peak << ",\"categories\":{";
        bool first = true;
        for (const auto& [category, value] : categories) {
            if (!first) output << ',';
            first = false;
            output << json_string(category) << ":{\"calls\":" << value.first << ",\"wall_ms\":" << value.second << '}';
        }
        output << "},\"transfer_totals\":{";
        first = true;
        for (const auto& [direction, total] : transfer_totals) {
            if (!first) output << ',';
            first = false;
            output << json_string(direction) << ":{\"bytes\":" << total.bytes << ",\"wall_ms\":" << total.wall_ms << '}';
        }
        output << "},\"graphs\":[";
        for (std::size_t i = 0; i < graphs.size(); ++i) {
            if (i) output << ',';
            const auto& graph = graphs[i];
            output << "{\"id\":" << i << ",\"stage\":" << json_string(graph.stage) << ",\"bind_wall_ms\":" << graph.bind_ms
                   << ",\"compute_wall_ms\":" << graph.compute_wall_ms << ",\"compute_calls\":" << graph.compute_calls
                   << ",\"arena_bytes\":" << graph.arena_bytes << ",\"reused\":" << (graph.reused ? "true" : "false") << '}';
        }
        output << "],\"nodes\":[";
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (i) output << ',';
            const auto& node = nodes[i];
            output << "{\"graph\":" << node.graph << ",\"tensor\":" << json_string(node.name)
                   << ",\"operation\":" << json_string(node.operation) << ",\"category\":" << json_string(node.category)
                   << ",\"backend\":" << json_string(node.backend) << ",\"output_type\":" << json_string(node.output_type)
                   << ",\"output_bytes\":" << node.output_bytes << ",\"calls\":" << node.calls << ",\"wall_ms\":" << node.wall_ms
                   << ",\"cpu_rhs_dot_type\":" << json_string(node.cpu_rhs_dot_type.empty() ? "NOT_APPLICABLE" : node.cpu_rhs_dot_type)
                   << ",\"cpu_rhs_dot_type_evidence\":" << json_string(node.cpu_rhs_dot_type.empty() ? "NOT_APPLICABLE" : "source-resolved pinned CPU type traits; not kernel scratch capture")
                   << ",\"accumulation_hint\":" << json_string(node.accumulation_hint)
                   << ",\"rhs_representation_hint\":" << json_string(node.rhs_hint) << ",\"inputs\":[";
            for (std::size_t j = 0; j < node.inputs.size(); ++j) {
                if (j) output << ',';
                const auto& input = node.inputs[j];
                output << "{\"tensor\":" << json_string(input.name) << ",\"type\":" << json_string(input.type)
                       << ",\"bytes\":" << input.bytes << '}';
            }
            output << "]}";
        }
        output << "],\"transfers\":[";
        for (std::size_t i = 0; i < transfers.size(); ++i) {
            if (i) output << ',';
            const auto& transfer = transfers[i];
            output << "{\"stage\":" << json_string(transfer.stage) << ",\"tensor\":" << json_string(transfer.tensor)
                   << ",\"direction\":" << json_string(transfer.direction) << ",\"type\":" << json_string(transfer.type)
                   << ",\"bytes\":" << transfer.bytes << ",\"wall_ms\":" << transfer.wall_ms << '}';
        }
        output << "]}";
    }

private:
    static std::string category(const ggml_tensor* tensor) {
        const bool copy = tensor->op == GGML_OP_DUP || tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_CONT;
        if (copy && tensor->src[0] && ggml_is_quantized(tensor->src[0]->type) && tensor->type == GGML_TYPE_F32)
            return "quantized_to_f32";
        if (copy && tensor->src[0] && ggml_is_quantized(tensor->type) && !ggml_is_quantized(tensor->src[0]->type))
            return "quantization";
        if (tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID) return "matmul";
        if (copy) return "cast_or_copy";
        if (!sam::internal::is_compute_node(tensor)) return "metadata";
        return "other";
    }

    std::size_t capture(ggml_tensor* tensor) {
        const auto found = active_nodes_.find(tensor);
        if (found != active_nodes_.end()) return found->second;
        const auto backend = ggml_backend_sched_get_tensor_backend(scheduler_, tensor);
        Node row;
        row.graph = graphs.size() - 1;
        row.name = ggml_get_name(tensor);
        row.operation = ggml_op_name(tensor->op);
        row.category = category(tensor);
        row.backend = backend ? ggml_backend_name(backend) : "UNASSIGNED";
        if (row.backend != "CPU" && row.backend != "BLAS" && row.backend != "UNASSIGNED")
            throw std::runtime_error("cost observer currently supports CPU/BLAS nodes only");
        row.output_type = ggml_type_name(tensor->type);
        if (row.backend == "CPU" && tensor->op == GGML_OP_MUL_MAT && tensor->src[0] &&
            ggml_is_quantized(tensor->src[0]->type))
            row.cpu_rhs_dot_type = ggml_type_name(sam::internal::cpu_quantized_rhs_type(tensor->src[0]->type));
        row.output_bytes = ggml_nbytes(tensor);
        sam_profile::Tensor hints;
        sam_profile::capture_precision_hints(tensor, hints);
        row.accumulation_hint = hints.accumulation_hint;
        row.rhs_hint = hints.rhs_representation_hint;
        for (auto* input : tensor->src) if (input)
            row.inputs.push_back({ggml_get_name(input), ggml_type_name(input->type), ggml_nbytes(input)});
        const auto index = nodes.size();
        nodes.push_back(std::move(row));
        active_nodes_.emplace(tensor, index);
        return index;
    }

    static bool evaluate(ggml_tensor* tensor, bool ask, void* user) noexcept {
        auto& self = *static_cast<Observer*>(user);
        try {
            if (ask) {
                self.active_node_ = self.capture(tensor);
                self.node_start_ = std::chrono::steady_clock::now();
            } else {
                const auto milliseconds = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - self.node_start_).count();
                auto& row = self.nodes.at(self.active_node_);
                ++row.calls;
                row.wall_ms += milliseconds;
            }
            // Request every node: no neighboring operation can be silently
            // included in this node's timing. This deliberately disables fusion.
            return true;
        } catch (...) {
            self.callback_error_ = std::current_exception();
            return ask;
        }
    }

    ggml_context* active_context_ = nullptr;
    ggml_cgraph* active_graph_ = nullptr;
    ggml_backend_sched_t scheduler_ = nullptr;
    std::unordered_map<ggml_tensor*, std::size_t> active_nodes_;
    std::size_t active_node_ = 0;
    std::chrono::steady_clock::time_point node_start_;
    std::exception_ptr callback_error_;
};

} // namespace sam_cost

#endif // SAM_CPP_TOOLS_BENCHMARK_EXECUTION_COST_HPP
