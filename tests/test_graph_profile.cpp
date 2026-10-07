#include "../tools/graph_profile.hpp"
#include <sam/internal/runtime/ggml/graph.hpp>
#include "backend_test_support.hpp"
#include <iostream>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

class Observer final : public sam::internal::GraphObserver {
public:
    sam_profile::GraphSnapshot snapshot;
    int allocations = 0, computes = 0;
    void allocated(ggml_context* context, ggml_cgraph* graph, ggml_backend_sched_t scheduler) override {
        snapshot = sam_profile::capture_graph(context, graph, scheduler);
        ++allocations;
    }
    void computed(ggml_context*, ggml_cgraph*, double milliseconds) override {
        require(milliseconds >= 0, "invalid observed elapsed time");
        ++computes;
    }
};

void check(sam::Backend backend) {
    require(sam_profile::union_bytes({{0, 10, 30}, {0, 15, 20}, {0, 25, 40}, {1, 10, 20}}) == 40,
            "overlapping or distinct buffer spans counted incorrectly");
    sam::internal::GgmlRuntime runtime({backend, 1}, true);
    auto observer = std::make_shared<Observer>();
    runtime.set_graph_observer(observer);
    auto external_context = sam::internal::make_context(2);
    auto* external = ggml_new_tensor_1d(external_context.get(), GGML_TYPE_F32, 64);
    sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(external_context.get(), runtime.weights_backend()));
    require(bool(buffer), "external allocation failed");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    sam::RuntimeStats stats;
    sam::internal::upload(external, std::vector<float>(64, 3), stats);
    auto workspace = std::make_shared<sam::internal::GraphWorkspace>(runtime, 64);
    sam::internal::GraphExecution graph(runtime, 64, stats, workspace);
    auto* input = sam::internal::input_tensor(graph.context(), "input", 64);
    auto* sum = ggml_add(graph.context(), input, external);
    ggml_set_name(sum, "sum");
    auto* view = ggml_view_1d(graph.context(), sum, 32, 16 * sizeof(float));
    ggml_set_name(view, "middle_view");
    auto* doubled = ggml_scale(graph.context(), view, 2);
    graph.output(doubled);
    graph.allocate();
    sam::internal::upload(input, std::vector<float>(64, 2), stats);
    graph.compute();
    const auto values = sam::internal::download(doubled, stats);
    for (const auto value : values) require(value == 10, "observer changed graph output");
    const auto& snapshot = observer->snapshot;
    require(snapshot.graph_live_span_peak_bytes > 0 && snapshot.graph_live_span_peak_bytes <= workspace->allocated_bytes(),
            "live spans exceed their arena or are empty");
    bool found_view = false, found_weight = false;
    for (const auto& tensor : snapshot.tensors) {
        if (tensor.name == "middle_view") {
            found_view = true;
            const auto& root = snapshot.tensors.at(tensor.allocation_root);
            require(root.name == "sum" && root.last >= tensor.last, "view did not extend its root lifetime");
            require(tensor.offset == root.offset + 16 * sizeof(float), "view offset was lost");
        }
        if (tensor.buffer >= 0 && snapshot.buffers.at(tensor.buffer).weights) {
            found_weight = true;
            require(!tensor.owned, "external weight counted as graph allocation");
        }
    }
    require(found_view && found_weight, "profile omitted view or resident weight");
    graph.allocate();
    graph.compute();
    require(observer->allocations == 2 && observer->computes == 2, "cached binding observation lost a run");
    workspace->release_storage();
    graph.allocate();
    sam::internal::upload(input, std::vector<float>(64, 4), stats);
    graph.compute();
    for (const auto value : sam::internal::download(doubled, stats)) require(value == 14, "rebound graph changed output");
    runtime.set_graph_observer({});
    graph.compute();
    require(observer->computes == 3, "disabled observer still called");
    require(!snapshot.tensors.empty(), "snapshot lifetime depended on graph binding");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (sam::test::cuda_requested(argc, argv)) {
            if (!sam::test::cuda_available()) return 77;
            check(sam::Backend::Cuda);
        } else check(sam::Backend::Cpu);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "graph profiling check: " << error.what() << '\n';
        return 1;
    }
}
