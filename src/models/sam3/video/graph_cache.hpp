#ifndef SAM_CPP_SRC_MODELS_SAM3_VIDEO_GRAPH_CACHE_HPP
#define SAM_CPP_SRC_MODELS_SAM3_VIDEO_GRAPH_CACHE_HPP

#include "../state.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam::internal::sam3 {

// Tracker graph construction is bounded on purpose: each stage owns one slot
// and rebuilds only when its shape key or graph metadata size changes.
inline constexpr int sam3_tracker_graph_batch_limit = 8;
inline constexpr std::size_t sam3_tracker_fused_graph_capacity = 4096;
inline constexpr std::size_t sam3_tracker_decoder_graph_capacity = 1024;
inline constexpr std::size_t sam3_tracker_memory_graph_capacity = 512;
inline constexpr std::size_t sam3_tracker_pointer_graph_capacity = 128;

inline std::size_t sam3_tracker_workspace_capacity(const GgmlRuntime& runtime) {
    return sam3_tracker_fused_graph_capacity * (runtime.quantized_cpu_f32_weights() ? 2 : 1);
}

// Shape key: rebuilt graphs must be numerically interchangeable, so every
// input layout, storage/arithmetic profile and residency decision is part of
// the identity of a cached graph.
struct TrackerGraphShape {
    enum class Stage : std::uint8_t { pointer_time, propagation, condition, decoder, memory_encoder } stage;
    Backend backend = Backend::Cpu;
    ggml_type activation_type = GGML_TYPE_F32;
    std::string storage_profile;
    std::string arithmetic_profile;
    std::vector<std::int64_t> dimensions;
    std::vector<std::size_t> strides;
    std::vector<int> memory_position_order;
    int spatial_count = 0, pointer_count = 0, batch = 1;
    bool seed = false, present = true, diagnostics = true, unrounded_output = false;
    bool high_resident = false;
    bool none_resident = false;

    bool operator==(const TrackerGraphShape& other) const {
        return stage == other.stage && backend == other.backend && activation_type == other.activation_type &&
            storage_profile == other.storage_profile &&
            arithmetic_profile == other.arithmetic_profile && dimensions == other.dimensions &&
            strides == other.strides && memory_position_order == other.memory_position_order &&
            spatial_count == other.spatial_count &&
            pointer_count == other.pointer_count && batch == other.batch && seed == other.seed &&
            present == other.present && diagnostics == other.diagnostics &&
            unrounded_output == other.unrounded_output && high_resident == other.high_resident &&
            none_resident == other.none_resident;
    }
};

inline void tracker_shape_tensor(TrackerGraphShape& shape, std::initializer_list<std::int64_t> dimensions) {
    std::size_t stride = sizeof(float);
    for (const auto dimension : dimensions) {
        shape.dimensions.push_back(dimension);
        shape.strides.push_back(stride);
        if (dimension > 0 && stride <= std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(dimension))
            stride *= static_cast<std::size_t>(dimension);
        else
            stride = 0;
    }
    shape.dimensions.push_back(0); // Tensor boundary; prevents ambiguous flattened layouts.
    shape.strides.push_back(0);
}

// One cached graph slot. Metadata lives until its shape changes or a residency
// demotion invalidates it; the actual arena is owned by GraphWorkspace.
class TrackerGraphSlot {
public:
    GraphExecution& ensure(GgmlRuntime& runtime, RuntimeStats& stats,
                           const std::shared_ptr<GraphWorkspace>& workspace,
                           const TrackerGraphShape& shape, std::size_t graph_size,
                           bool& built) {
        built = !execution_ || graph_size_ != graph_size || !(shape_ == shape);
        if (built) {
            execution_.reset();
            required_workspace_bytes_.reset();
            const auto start = std::chrono::steady_clock::now();
            execution_ = std::make_unique<GraphExecution>(runtime, graph_size, stats, workspace);
            build_ms_ += elapsed_ms(start);
            graph_size_ = graph_size;
            shape_ = shape;
        }
        return *execution_;
    }

    void reset() { execution_.reset(); graph_size_ = 0; shape_ = {}; required_workspace_bytes_.reset(); }
    GraphExecution* get() const { return execution_.get(); }
    std::size_t required_workspace_bytes() {
        if (!execution_) throw std::runtime_error("tracker graph slot is empty");
        if (!required_workspace_bytes_) required_workspace_bytes_ = execution_->required_workspace_bytes();
        return *required_workspace_bytes_;
    }
    void add_build_ms(double milliseconds) { build_ms_ += milliseconds; }
    double build_ms() const { return build_ms_; }

private:
    static double elapsed_ms(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    std::unique_ptr<GraphExecution> execution_;
    std::size_t graph_size_ = 0;
    TrackerGraphShape shape_{};
    std::optional<std::size_t> required_workspace_bytes_;
    double build_ms_ = 0;
};

// The tracker execution coordinator owns one cache; stage code asks for the
// named slot and keeps building the graph body itself so the call order and
// failure behavior stay in one place.
class TrackerGraphCache {
public:
    TrackerGraphSlot& pointer() { return pointer_; }
    TrackerGraphSlot& propagation() { return propagation_; }
    TrackerGraphSlot& decoder() { return decoder_; }
    TrackerGraphSlot& memory() { return memory_; }

    void reset_propagation() { propagation_.reset(); }
    void reset_decoder() { decoder_.reset(); }
    void reset_pointer() { pointer_.reset(); }
    void reset_memory() { memory_.reset(); }

    // Standalone probe graphs are not cached; their build time still belongs to
    // the tracker diagnostics total.
    void add_auxiliary_build_ms(double milliseconds) { auxiliary_build_ms_ += milliseconds; }
    double graph_build_ms() const {
        return pointer_.build_ms() + propagation_.build_ms() + decoder_.build_ms() + memory_.build_ms() +
            auxiliary_build_ms_;
    }

private:
    TrackerGraphSlot pointer_, propagation_, decoder_, memory_;
    double auxiliary_build_ms_ = 0;
};

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_VIDEO_GRAPH_CACHE_HPP
