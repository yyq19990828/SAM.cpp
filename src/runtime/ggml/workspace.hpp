#ifndef SAM_CPP_SRC_RUNTIME_GGML_WORKSPACE_HPP
#define SAM_CPP_SRC_RUNTIME_GGML_WORKSPACE_HPP

#include "resources.hpp"
#include "runtime.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <vector>

namespace sam::internal {

struct GraphDiagnostics {
    std::size_t graph_builds = 0;
    std::size_t graph_reuses = 0;
    std::size_t graph_binds = 0;
    std::size_t reserve_probes = 0;
    std::size_t compute_calls = 0;
    std::size_t workspace_peak_bytes = 0;
    double bind_ms = 0;
    double reserve_ms = 0;
    double compute_ms = 0; // Synchronized wall time, not a GPU timestamp.
};

// A session may retain graph metadata while sharing one backend allocation.
// Graphs and external inputs must outlive their active binding to this workspace.
class GraphWorkspace {
public:
    GraphWorkspace(GgmlRuntime& runtime, std::size_t graph_capacity)
        : runtime_(runtime), backends_(runtime.backends()), capacity_(graph_capacity) {
        if (!capacity_ || capacity_ > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2)
            throw std::invalid_argument("invalid SAM workspace graph capacity");
        scheduler_.reset(ggml_backend_sched_new(backends_.data(), nullptr,
            static_cast<int>(backends_.size()), static_cast<int>(capacity_ * 2), false, true));
        if (!scheduler_) throw std::runtime_error("failed to create GGML scheduler");
    }
    ~GraphWorkspace() { release(); }
    GraphWorkspace(const GraphWorkspace&) = delete;
    GraphWorkspace& operator=(const GraphWorkspace&) = delete;

    ggml_backend_sched_t scheduler() const { return scheduler_.get(); }
    GgmlRuntime& runtime() const { return runtime_; }
    const GraphDiagnostics& diagnostics() const { return diagnostics_; }
    void record_build() { ++diagnostics_.graph_builds; }

    // Release the backend arena at a frame boundary while retaining this shared
    // owner and its cached graph metadata. The next bind lazily recreates it.
    void release_storage() noexcept {
        release();
        scheduler_.reset();
    }

    void bind(ggml_context* context, ggml_cgraph* graph) {
        validate(graph);
        ensure_scheduler();
        if (active_context_ == context && active_graph_ == graph) {
            ++diagnostics_.graph_reuses;
            if (const auto& observer = runtime_.graph_observer())
                observer->allocated(context, graph, scheduler_.get());
            return;
        }
        const auto start = std::chrono::steady_clock::now();
        release();
        auto sources = save_sources(context);
        assign_primary_compute(scheduler_.get(), graph);
        if (!ggml_backend_sched_alloc_graph(scheduler_.get(), graph)) {
            restore_sources(sources);
            clear_allocations(context);
            ggml_backend_sched_reset(scheduler_.get());
            throw std::runtime_error("failed to allocate SAM graph");
        }
        active_context_ = context;
        active_graph_ = graph;
        active_sources_ = std::move(sources);
        ++diagnostics_.graph_binds;
        diagnostics_.bind_ms += elapsed_ms(start);
        diagnostics_.workspace_peak_bytes = std::max(diagnostics_.workspace_peak_bytes, allocated_bytes());
        if (const auto& observer = runtime_.graph_observer())
            observer->allocated(context, graph, scheduler_.get());
    }

    std::size_t required_bytes(ggml_context* context, ggml_cgraph* graph,
                               std::vector<std::size_t>* buffer_sizes = nullptr) {
        validate(graph);
        const auto start = std::chrono::steady_clock::now();
        release();
        const auto sources = save_sources(context);
        std::vector<std::size_t> sizes(backends_.size());
        // The pinned allocator's size-only reserve also changes its allocation
        // plan and can free existing buffers. Probe on a metadata-only scheduler
        // so sizing cannot invalidate the session's reusable compute arena.
        SchedulerPtr probe(ggml_backend_sched_new(backends_.data(), nullptr,
            static_cast<int>(backends_.size()), static_cast<int>(capacity_ * 2), false, true));
        if (!probe) throw std::runtime_error("failed to create SAM workspace probe");
        assign_primary_compute(probe.get(), graph);
        ggml_backend_sched_reserve_size(probe.get(), graph, sizes.data());
        restore_sources(sources);
        clear_allocations(context);
        if (scheduler_) ggml_backend_sched_reset(scheduler_.get());
        ++diagnostics_.reserve_probes;
        diagnostics_.reserve_ms += elapsed_ms(start);
        if (buffer_sizes) *buffer_sizes = sizes;
        std::size_t total = 0;
        std::vector<ggml_backend_buffer_type_t> counted;
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            const auto type = ggml_backend_get_default_buffer_type(backends_[i]);
            if (std::find(counted.begin(), counted.end(), type) != counted.end()) continue;
            counted.push_back(type);
            const auto size = sizes[i];
            if (size > std::numeric_limits<std::size_t>::max() - total)
                throw std::overflow_error("SAM workspace size overflow");
            total += size;
        }
        return total;
    }

    void fit_allocation_plan(const std::vector<std::size_t>& sizes, std::size_t limit) {
        if (sizes.size() != backends_.size())
            throw std::invalid_argument("SAM workspace allocation plan has incorrect backend count");
        struct BufferSize { ggml_backend_buffer_type_t type; std::size_t bytes; };
        std::vector<BufferSize> buffers;
        for (std::size_t i = 0; i < backends_.size(); ++i) {
            const auto type = ggml_backend_get_default_buffer_type(backends_[i]);
            const auto retained = scheduler_ ? ggml_backend_sched_get_buffer_size(scheduler_.get(), backends_[i]) : 0;
            const auto bytes = std::max(retained, sizes[i]);
            auto found = std::find_if(buffers.begin(), buffers.end(),
                [type](const BufferSize& buffer) { return buffer.type == type; });
            if (found == buffers.end()) buffers.push_back({type, bytes});
            else found->bytes = std::max(found->bytes, bytes);
        }
        std::size_t projected = 0;
        for (const auto& buffer : buffers) {
            if (buffer.bytes > limit - std::min(projected, limit)) {
                release_storage();
                return;
            }
            projected += buffer.bytes;
        }
    }

    void compute(ggml_context* context, ggml_cgraph* graph) {
        if (!scheduler_ || active_context_ != context || active_graph_ != graph)
            throw std::runtime_error("SAM graph must be allocated before compute");
        if (runtime_.requires_primary_compute()) {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto* node = ggml_graph_node(graph, i);
                if (is_compute_node(node) &&
                    ggml_backend_sched_get_tensor_backend(scheduler_.get(), node) != runtime_.weights_backend())
                    throw std::runtime_error(std::string("SAM ") + ggml_backend_name(runtime_.weights_backend()) +
                        " graph attempted compute fallback for " + ggml_op_name(node->op));
            }
        }
        const auto start = std::chrono::steady_clock::now();
        const auto status = ggml_backend_sched_graph_compute(scheduler_.get(), graph);
        ++diagnostics_.compute_calls;
        const auto compute_ms = elapsed_ms(start);
        diagnostics_.compute_ms += compute_ms;
        if (status != GGML_STATUS_SUCCESS) {
            release();
            throw std::runtime_error("SAM graph execution failed");
        }
        if (const auto& observer = runtime_.graph_observer()) observer->computed(context, graph, compute_ms);
    }

    void discard(ggml_context* context) {
        if (active_context_ == context) release();
    }
    std::size_t buffer_bytes() const {
        if (!scheduler_) return 0;
        // Preserve RuntimeStats' per-backend sum. Use allocated_bytes() for
        // budgets: CPU and BLAS can share one physical arena.
        std::size_t bytes = 0;
        for (auto* backend : backends_) bytes += ggml_backend_sched_get_buffer_size(scheduler_.get(), backend);
        return bytes;
    }
    std::size_t allocated_bytes() const {
        if (!scheduler_) return 0;
        std::vector<ggml_backend_buffer_type_t> counted;
        std::size_t bytes = 0;
        for (auto* backend : backends_) {
            const auto type = ggml_backend_get_default_buffer_type(backend);
            if (std::find(counted.begin(), counted.end(), type) != counted.end()) continue;
            counted.push_back(type);
            bytes += ggml_backend_sched_get_buffer_size(scheduler_.get(), backend);
        }
        return bytes;
    }

private:
    void assign_primary_compute(ggml_backend_sched_t scheduler, ggml_cgraph* graph) const {
        if (!runtime_.requires_primary_compute()) return;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto* node = ggml_graph_node(graph, i);
            if (!is_compute_node(node)) continue;
            if (!ggml_backend_supports_op(runtime_.weights_backend(), node))
                throw std::runtime_error(std::string("primary backend does not support SAM operation ") +
                    ggml_op_name(node->op));
            ggml_backend_sched_set_tensor_backend(scheduler, node, runtime_.weights_backend());
        }
    }
    struct Sources {
        ggml_tensor* tensor;
        std::array<ggml_tensor*, GGML_MAX_SRC> sources;
    };
    static std::vector<Sources> save_sources(ggml_context* context) {
        std::vector<Sources> result;
        for (auto* tensor = ggml_get_first_tensor(context); tensor;
             tensor = ggml_get_next_tensor(context, tensor)) {
            Sources entry{tensor, {}};
            std::copy(std::begin(tensor->src), std::end(tensor->src), entry.sources.begin());
            result.push_back(entry);
        }
        return result;
    }
    static void restore_sources(const std::vector<Sources>& sources) {
        // Split graphs can replace sources with copies in the scheduler's scratch
        // context. Restore them before that context is reused by another graph.
        for (const auto& entry : sources)
            std::copy(entry.sources.begin(), entry.sources.end(), std::begin(entry.tensor->src));
    }
    static double elapsed_ms(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    void ensure_scheduler() {
        if (scheduler_) return;
        scheduler_.reset(ggml_backend_sched_new(backends_.data(), nullptr,
            static_cast<int>(backends_.size()), static_cast<int>(capacity_ * 2), false, true));
        if (!scheduler_) throw std::runtime_error("failed to recreate SAM shared workspace scheduler");
    }
    static void clear_allocations(ggml_context* context) {
        // sched_reset does not clear tensor allocation pointers. Only visit the
        // graph's own context: model weights and resident frame inputs are external.
        for (auto* tensor = ggml_get_first_tensor(context); tensor;
             tensor = ggml_get_next_tensor(context, tensor)) {
            tensor->data = nullptr;
            tensor->buffer = nullptr;
            tensor->extra = nullptr;
        }
    }
    void validate(ggml_cgraph* graph) const {
        // A GGML graph has up to size nodes plus size leaves; its scheduler hash
        // therefore needs twice that capacity even when the node list fits.
        if (static_cast<std::size_t>(ggml_graph_size(graph)) > capacity_)
            throw std::runtime_error("SAM graph exceeds shared workspace capacity");
    }
    void release() {
        if (!scheduler_) {
            restore_sources(active_sources_);
            active_sources_.clear();
            if (active_context_) clear_allocations(active_context_);
            active_context_ = nullptr;
            active_graph_ = nullptr;
            return;
        }
        ggml_backend_sched_synchronize(scheduler_.get());
        restore_sources(active_sources_);
        active_sources_.clear();
        if (active_context_) clear_allocations(active_context_);
        active_context_ = nullptr;
        active_graph_ = nullptr;
        ggml_backend_sched_reset(scheduler_.get());
    }

    GgmlRuntime& runtime_;
    std::vector<ggml_backend_t> backends_;
    std::size_t capacity_;
    SchedulerPtr scheduler_;
    ggml_context* active_context_ = nullptr;
    ggml_cgraph* active_graph_ = nullptr;
    std::vector<Sources> active_sources_;
    GraphDiagnostics diagnostics_;
};

} // namespace sam::internal

#endif // SAM_CPP_SRC_RUNTIME_GGML_WORKSPACE_HPP
