#include "graph_profile.hpp"
#include "../../support/image_io/image_support.hpp"
#include <models/sam3/image_session.hpp>
#include <models/sam3/model.hpp>
#include <iostream>
#include <memory>
#include <utility>

namespace {

template<class Values> void array(std::ostream& out, const Values& values) {
    out << '[';
    bool first = true;
    for (const auto value : values) {
        if (!first) out << ',';
        first = false;
        out << value;
    }
    out << ']';
}

struct Record {
    std::string stage;
    sam_profile::GraphSnapshot graph;
    std::vector<std::pair<std::string, std::size_t>> arenas;
    std::int64_t allocated_us = 0, computed_us = 0;
};

class Observer final : public sam::internal::GraphObserver {
public:
    explicit Observer(sam::internal::GgmlRuntime& runtime) : runtime_(runtime) {}
    std::string stage;
    std::vector<Record> records;

    void allocated(ggml_context* context, ggml_cgraph* graph, ggml_backend_sched_t scheduler) override {
        Record record{stage, sam_profile::capture_graph(context, graph, scheduler), {}};
        std::set<ggml_backend_buffer_type_t> counted;
        for (auto* backend : runtime_.backends()) {
            if (counted.insert(ggml_backend_get_default_buffer_type(backend)).second)
                record.arenas.emplace_back(ggml_backend_name(backend),
                    ggml_backend_sched_get_buffer_size(scheduler, backend));
        }
        records.push_back(std::move(record));
        records.back().allocated_us = ggml_time_us();
        active_context_ = context;
        active_graph_ = graph;
    }

    void computed(ggml_context* context, ggml_cgraph* graph, double milliseconds) override {
        if (context != active_context_ || graph != active_graph_ || records.empty())
            throw std::runtime_error("observed graph computed without an allocation snapshot");
        records.back().graph.compute_ms.push_back(milliseconds);
        records.back().computed_us = ggml_time_us();
    }
private:
    sam::internal::GgmlRuntime& runtime_;
    ggml_context* active_context_ = nullptr;
    ggml_cgraph* active_graph_ = nullptr;
};

void write_record(std::ostream& out, const Record& record) {
    using sam_example::json_string;
    const auto& graph = record.graph;
    out << "{\"stage\":" << json_string(record.stage) << ",\"nodes\":" << graph.nodes
        << ",\"allocated_us\":" << record.allocated_us << ",\"computed_us\":" << record.computed_us
        << ",\"graph_live_span_peak_bytes\":" << graph.graph_live_span_peak_bytes
        << ",\"graph_host_span_peak_bytes\":" << graph.graph_host_span_peak_bytes
        << ",\"graph_device_span_peak_bytes\":" << graph.graph_device_span_peak_bytes
        << ",\"peak_node\":" << graph.peak_node << ",\"peak_roots\":";
    array(out, graph.peak_roots);
    out << ",\"compute_wall_ms\":";
    array(out, graph.compute_ms);
    out << ",\"arenas\":[";
    for (std::size_t i = 0; i < record.arenas.size(); ++i) {
        if (i) out << ',';
        out << "{\"backend\":" << json_string(record.arenas[i].first)
            << ",\"bytes\":" << record.arenas[i].second << '}';
    }
    out << "],\"buffers\":[";
    for (std::size_t i = 0; i < graph.buffers.size(); ++i) {
        if (i) out << ',';
        const auto& buffer = graph.buffers[i];
        out << "{\"name\":" << json_string(buffer.name) << ",\"bytes\":" << buffer.bytes
            << ",\"host\":" << (buffer.host ? "true" : "false")
            << ",\"weights\":" << (buffer.weights ? "true" : "false") << '}';
    }
    out << "],\"tensors\":[";
    for (std::size_t i = 0; i < graph.tensors.size(); ++i) {
        if (i) out << ',';
        const auto& tensor = graph.tensors[i];
        out << "{\"id\":" << i << ",\"name\":" << json_string(tensor.name)
            << ",\"type\":" << json_string(tensor.type) << ",\"operation\":" << json_string(tensor.operation)
            << ",\"backend\":" << json_string(tensor.backend) << ",\"shape\":";
        array(out, tensor.shape);
        out << ",\"strides\":"; array(out, tensor.strides);
        out << ",\"sources\":"; array(out, tensor.sources);
        out << ",\"operands\":[";
        for (std::size_t j = 0; j < tensor.sources.size(); ++j) {
            if (j) out << ',';
            const auto id = tensor.sources[j];
            out << "{\"slot\":" << tensor.source_slots[j] << ",\"tensor\":" << id
                << ",\"storage_type\":" << json_string(graph.tensors.at(id).type) << '}';
        }
        out << "],\"precision_hints\":{\"accumulation\":" << json_string(tensor.accumulation_hint)
            << ",\"rhs_representation\":" << json_string(tensor.rhs_representation_hint) << '}';
        out << ",\"buffer\":" << tensor.buffer << ",\"offset\":" << tensor.offset
            << ",\"bytes\":" << tensor.bytes << ",\"view_source\":" << tensor.view_source
            << ",\"view_offset\":" << tensor.view_offset << ",\"allocation_root\":" << tensor.allocation_root
            << ",\"node\":" << tensor.node << ",\"first\":" << tensor.first << ",\"last\":" << tensor.last
            << ",\"owned\":" << (tensor.owned ? "true" : "false")
            << ",\"input\":" << (tensor.input ? "true" : "false")
            << ",\"output\":" << (tensor.output ? "true" : "false") << '}';
    }
    out << "]}";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string cache = "f32";
        std::vector<char*> arguments{argv[0]};
        bool cache_set = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--feature-cache") {
                if (cache_set || ++i == argc) throw std::invalid_argument("--feature-cache requires one value");
                cache = argv[i];
                cache_set = true;
            } else arguments.push_back(argv[i]);
        }
        if (cache != "f32" && cache != "f16" && cache != "mixed-q8_0")
            throw std::invalid_argument("--feature-cache must be f32, f16 or mixed-q8_0");
        const auto options = sam_example::parse_options(static_cast<int>(arguments.size()), arguments.data(), false);
        if (options.help) {
            std::cout << "Usage: sam_profile_graph --model FILE --image FILE --text PROMPT --output NEW_DIR\n"
                         "  [--backend cpu|metal|cuda] [--cuda-device N] [--cuda-compute f32|f16] [--threads N]\n"
                         "  [--feature-cache f32|f16|mixed-q8_0]\n"
                         "Diagnostic tensor types and precision hints; kernel-internal arithmetic is not traced.\n"
                         "Allocation snapshots perturb execution; timings are not performance benchmarks.\n";
            return 0;
        }
        sam_example::OutputDirectory output(options.output);
        const auto image = sam_example::read_image(options.image);
        using sam::internal::FeatureCacheMode;
        const auto cache_mode = cache == "f32" ? FeatureCacheMode::F32 : cache == "f16" ? FeatureCacheMode::F16 : FeatureCacheMode::Q8_0;
        auto state = sam::internal::sam3::load_state(options.model.string(), options.backend, cache_mode);
        if (state->model_info.task != "text_image") throw std::invalid_argument("graph profiler requires image weights");
        auto observer = std::make_shared<Observer>(*state->runtime);
        state->runtime->set_graph_observer(observer);
        sam::internal::sam3::ImageSession session(state);
        observer->stage = "image_encoding";
        session.set_image(sam_example::image_view(image));
        observer->stage = "text_and_prediction";
        const auto result = session.segment_text(options.text, options.score_threshold);
        state->runtime->set_graph_observer({});
        auto json = sam_example::output_file(options.output / "graphs.json");
        json << "{\"schema_version\":1,\"complete\":true,\"diagnostic_only\":true,\"scope\":"
             << sam_example::json_string("Operator-boundary owned allocation spans; views pin their root; overlapping ranges count once. Excludes backend scratch pools, padding, external storage and context. Per-stage peaks must not be summed.")
             << ",\"model\":" << sam_example::json_string(options.model.string())
             << ",\"image\":" << sam_example::json_string(options.image.string())
             << ",\"prompt\":" << sam_example::json_string(options.text)
             << ",\"feature_cache\":" << sam_example::json_string(cache)
             << ",\"cuda_compute\":" << sam_example::json_string(options.backend.cuda_compute == sam::CudaComputeMode::F16 ? "f16" : "f32")
             << ",\"precision_evidence\":{\"tensor_types\":\"observed at graph allocation boundaries\","
                "\"precision_hints\":\"requested op parameters; backend may select another kernel representation\","
                "\"kernel_internal_arithmetic\":\"NOT_COLLECTED\"}";
        sam_example::write_model_profile(json, state->model_info);
        json << ",\"runtime\":";
        sam_example::write_runtime_stats(json, session.stats());
        json << ",\"graphs\":[";
        for (std::size_t i = 0; i < observer->records.size(); ++i) {
            if (i) json << ',';
            write_record(json, observer->records[i]);
        }
        json << "],\"detections\":";
        sam_example::write_detections(json, result, options.output, false);
        json << "}\n";
        json.close();
        output.complete();
        std::cout << "Captured " << observer->records.size() << " graphs in " << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_profile_graph: " << error.what() << '\n';
        return 1;
    }
}
