#include <sam/internal/runtime/ggml/graph.hpp>
#include "backend_test_support.hpp"
#include <iostream>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Probe {
    sam::RuntimeStats stats;
    sam::internal::GraphExecution graph;
    ggml_tensor* input;
    ggml_tensor* result;
    Probe(sam::internal::GgmlRuntime& runtime, std::shared_ptr<sam::internal::GraphWorkspace> workspace,
          ggml_tensor* external, int columns)
        : graph(runtime, 64, stats, std::move(workspace)) {
        input = sam::internal::input_tensor(graph.context(), "input", 256, columns);
        result = ggml_add(graph.context(), ggml_mul_mat(graph.context(), external, input),
                          ggml_scale(graph.context(), input, 2.0f));
        graph.output(result);
    }
    std::vector<float> run(float value) {
        graph.allocate();
        sam::internal::upload(input, std::vector<float>(ggml_nelements(input), value), stats);
        graph.compute();
        return sam::internal::download(result, stats);
    }
};

struct ArenaProbe {
    sam::RuntimeStats stats;
    sam::internal::GraphExecution graph;
    ggml_tensor* input;
    ggml_tensor* result;
    ArenaProbe(sam::internal::GgmlRuntime& runtime,
               const std::shared_ptr<sam::internal::GraphWorkspace>& workspace, bool large_device)
        : graph(runtime, 64, stats, workspace) {
        input = sam::internal::input_tensor(graph.context(), "arena_input", 256, large_device ? 1 : 8192);
        if (large_device) {
            auto* shape = ggml_new_tensor_2d(graph.context(), GGML_TYPE_F32, 256, 32768);
            result = ggml_repeat(graph.context(), input, shape);
        } else {
            result = ggml_scale(graph.context(), input, 2.0f);
        }
        graph.output(result);
    }
    std::vector<float> run(float value, std::size_t limit = std::numeric_limits<std::size_t>::max()) {
        graph.allocate(limit);
        sam::internal::upload(input, std::vector<float>(ggml_nelements(input), value), stats);
        graph.compute();
        return sam::internal::download(result, stats);
    }
};

void check_cuda_arena_budget() {
    sam::internal::GgmlRuntime runtime({sam::Backend::Cuda, 1}, true);
    std::size_t limit = 0;
    for (const bool large_device : {true, false}) {
        auto workspace = std::make_shared<sam::internal::GraphWorkspace>(runtime, 64);
        ArenaProbe probe(runtime, workspace, large_device);
        const auto required = probe.graph.required_workspace_bytes();
        probe.run(1.0f);
        require(required > 0 && workspace->allocated_bytes() >= required, "fresh arena has an invalid allocation plan");
        limit = std::max(limit, workspace->allocated_bytes());
    }
    {
        auto workspace = std::make_shared<sam::internal::GraphWorkspace>(runtime, 64);
        ArenaProbe device(runtime, workspace, true), host(runtime, workspace, false);
        device.run(1.0f);
        host.run(1.0f);
        require(workspace->allocated_bytes() > limit,
                "budget fixture did not retain opposing host and device peaks");
    }
    auto workspace = std::make_shared<sam::internal::GraphWorkspace>(runtime, 64);
    ArenaProbe device(runtime, workspace, true), host(runtime, workspace, false);
    const auto saved = device.run(3.0f, limit);
    require(workspace->allocated_bytes() <= limit, "first graph exceeds its arena budget");
    for (const float value : host.run(4.0f, limit))
        require(value == 8.0f, "host-heavy graph lost inputs after arena replacement");
    require(workspace->allocated_bytes() <= limit, "opposing arena growth exceeds the budget");
    for (const float value : device.run(-2.0f, limit))
        require(value == -2.0f, "device-heavy graph failed after budgeted arena rebinding");
    require(workspace->diagnostics().workspace_peak_bytes <= limit,
            "budgeted graph transiently allocated an oversized arena");
    for (const float value : saved) require(value == 3.0f, "owned output changed after arena replacement");
    require(device.stats.cuda_nodes && host.stats.cuda_nodes && !device.stats.cpu_nodes && !host.stats.cpu_nodes,
            "budgeted CUDA graphs executed host compute");
}

void check(sam::Backend backend) {
    sam::internal::GgmlRuntime runtime({backend, 1}, true);
    auto context = sam::internal::make_context(2);
    auto* weights = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 256, 256);
    sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), runtime.weights_backend()));
    require(bool(buffer), "external allocation failed");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    sam::RuntimeStats stats;
    sam::internal::upload(weights, std::vector<float>(256 * 256, 0.25f), stats);
    auto workspace = std::make_shared<sam::internal::GraphWorkspace>(runtime, 128);
    Probe a(runtime, workspace, weights, 32);
    const auto bytes_a = a.graph.required_workspace_bytes();
    const auto saved = a.run(0.5f);
    const auto small_live_bytes = workspace->allocated_bytes();
    const auto resident_live_bytes = ggml_backend_buffer_get_size(buffer.get());
    const auto small_stage_cap = small_live_bytes + resident_live_bytes;
    for (const auto value : saved) require(value == 33.0f, "first graph differs");
    require(a.run(1.0f).front() == 66.0f, "same graph reused stale input");
    {
        Probe b(runtime, workspace, weights, 65);
        require(b.graph.required_workspace_bytes() > 0 && bytes_a > 0, "workspace probe is empty");
        require(b.run(2.0f).front() == 132.0f, "shape switch differs");
        require(workspace->allocated_bytes() + resident_live_bytes > small_stage_cap,
                "large graph did not exceed the measured small-stage plus resident cap");
        bool inactive_rejected = false;
        try { a.graph.compute(); } catch (const std::runtime_error&) { inactive_rejected = true; }
        require(inactive_rejected, "inactive graph executed another graph's workspace");
        require(a.run(3.0f).front() == 198.0f, "shape return used stale graph or copies");
        // Destroy an inactive graph, then execute the surviving active graph.
    }
    require(a.run(4.0f).front() == 264.0f, "inactive graph destruction invalidated active graph");
    workspace->release_storage();
    require(workspace->allocated_bytes() == 0, "large retained arena survived the release request");
    require(a.run(4.5f).front() == 297.0f, "small graph failed after releasing a larger retained arena");
    require(workspace->allocated_bytes() + resident_live_bytes <= small_stage_cap,
            "released smaller graph plus resident input exceeds its measured stage cap");
    {
        Probe b(runtime, workspace, weights, 17);
        b.run(5.0f);
        // Destroy the active graph before rebinding another retained graph.
    }
    const auto actual = a.run(-0.25f);
    Probe fresh(runtime, {}, weights, 32);
    require(actual == fresh.run(-0.25f), "retained graph differs from fresh execution");
    for (const auto value : saved) require(value == 33.0f, "copied output lost ownership");
    require(weights->buffer == buffer.get() && weights->data, "external tensor allocation was cleared");
    {
        sam::internal::GraphExecution resident(runtime, 32, stats, workspace);
        auto* normalized = ggml_norm(resident.context(), weights, 1e-5f);
        resident.output(normalized);
        resident.allocate();
        resident.compute(); resident.compute();
        for (const auto value : sam::internal::download(weights, stats))
            require(value == 0.25f, "arena aliasing modified a resident external input");
    }
    a.run(0.5f);
    const auto* input_metadata = a.input;
    const auto* result_metadata = a.result;
    const auto owned_output = a.run(0.75f);
    require(workspace->allocated_bytes() > 0, "active shared arena was not allocated");
    workspace->release_storage();
    require(workspace->allocated_bytes() == 0, "frame-boundary storage release retained the shared arena");
    require(a.input == input_metadata && a.result == result_metadata,
            "storage release invalidated retained graph metadata");
    require(weights->buffer == buffer.get() && weights->data,
            "storage release invalidated an external model tensor");
    bool inactive_rejected = false;
    try { a.graph.compute(); } catch (const std::runtime_error&) { inactive_rejected = true; }
    require(inactive_rejected, "released graph executed without rebinding its arena");
    require(a.run(-0.25f).front() == -16.5f, "graph failed after lazy arena recreation");
    for (const auto value : owned_output) require(value == 49.5f, "owned output changed after arena release");
    {
        Probe b(runtime, workspace, weights, 65);
        const auto b_output = b.run(0.125f);
        require(b_output.front() == 8.25f, "graph B failed after arena recreation");
        workspace->release_storage();
        require(workspace->allocated_bytes() == 0, "second arena release retained storage");
        require(a.run(0.25f).front() == 16.5f, "A-to-B-to-A rebinding failed after storage release");
        for (const auto value : b_output) require(value == 8.25f, "graph B output borrowed released arena memory");
    }
    {
        auto frame_context = sam::internal::make_context(2);
        auto* frame_features = ggml_new_tensor_2d(frame_context.get(), GGML_TYPE_F32, 256, 256);
        sam::internal::BufferPtr frame_buffer(
            ggml_backend_alloc_ctx_tensors(frame_context.get(), runtime.weights_backend()));
        require(bool(frame_buffer), "resident frame allocation failed");
        ggml_backend_buffer_set_usage(frame_buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        sam::internal::upload(frame_features, std::vector<float>(256 * 256, 0.25f), stats);
        Probe frame_graph(runtime, workspace, frame_features, 32);
        require(frame_graph.run(0.5f).front() == 33.0f, "resident frame source failed initially");
        workspace->release_storage();
        frame_features->data = nullptr;
        frame_features->buffer = nullptr;
        frame_features->extra = nullptr;
        frame_buffer.reset();
        frame_buffer.reset(ggml_backend_alloc_ctx_tensors(frame_context.get(), runtime.weights_backend()));
        require(bool(frame_buffer), "resident frame reallocation failed");
        ggml_backend_buffer_set_usage(frame_buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        sam::internal::upload(frame_features, std::vector<float>(256 * 256, 0.5f), stats);
        require(frame_graph.run(0.5f).front() == 65.0f,
                "cached graph did not read fresh data from the surviving frame tensor metadata");
        workspace->release_storage();
    }
    bool rejected = false;
    try { sam::internal::upload(a.input, {1.0f}, stats); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected && a.run(1.0f).front() == 66.0f, "input error corrupted reusable graph");
    // A size probe invalidates an active binding but must leave the graph reusable.
    a.graph.required_workspace_bytes();
    rejected = false;
    try { a.graph.compute(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected && a.run(2.0f).front() == 132.0f, "reserve recovery differs");
    sam::internal::GgmlRuntime other({sam::Backend::Cpu, 1}, true);
    rejected = false;
    try { sam::internal::GraphExecution foreign(other, 32, stats, workspace); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "workspace accepted a foreign runtime");
    auto small = std::make_shared<sam::internal::GraphWorkspace>(runtime, 1);
    sam::internal::GraphExecution oversized(runtime, 8, stats, small);
    auto* value = sam::internal::input_tensor(oversized.context(), "value", 1);
    oversized.output(ggml_scale(oversized.context(), value, 2.0f));
    rejected = false;
    try { oversized.allocate(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "workspace capacity mismatch reached a scheduler assertion");
    if (backend == sam::Backend::Cuda) {
        require(a.stats.cuda_nodes > 0 && !a.stats.cpu_nodes && !a.stats.metal_nodes,
                "reused CUDA workspace executed compute on another backend");
        require(fresh.stats.cuda_nodes > 0 && !fresh.stats.cpu_nodes && !fresh.stats.metal_nodes,
                "fresh CUDA workspace executed compute on another backend");
    }
}
}

int main(int argc, char** argv) {
    try {
        if (sam::test::cuda_requested(argc, argv)) {
            if (!sam::test::cuda_available()) return 77;
            check(sam::Backend::Cuda);
            check_cuda_arena_budget();
            return 0;
        }
        check(sam::Backend::Cpu);
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto* device = ggml_backend_dev_get(i);
            if (std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) == "MTL") {
                check(sam::Backend::Metal);
                break;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "workspace check: " << error.what() << '\n';
        return 1;
    }
}
