#include "../examples/image_support.hpp"
#include "linear_probe.hpp"
#include <sam/internal/models/sam3/model.hpp>
#include <iostream>
#include <sstream>

namespace {

struct Case {
    std::string id, image, prompt, alternate;
};

std::vector<Case> read_cases(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open cache case list");
    std::vector<Case> cases;
    std::set<std::string> ids;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(row, field, '\t')) fields.push_back(field);
        if (fields.size() < 3 || fields.size() > 4 || fields[0].empty() || fields[1].empty() || fields[2].empty() ||
            fields[0].find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos ||
            !ids.insert(fields[0]).second || cases.size() >= 8192)
            throw std::runtime_error("case list requires unique safe IDs, image paths and prompts in three or four TSV columns");
        cases.push_back({fields[0], fields[1], fields[2], fields.size() == 4 ? fields[3] : ""});
    }
    if (!input.eof() || cases.empty()) throw std::runtime_error("invalid or empty cache case list");
    return cases;
}

void write_numbers(std::ostream& stream, const std::vector<double>& values) {
    stream << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) stream << ',';
        stream << values[i];
    }
    stream << ']';
}

void dump_tensors(sam::internal::sam3::ImageSession& session, const std::filesystem::path& output) {
    auto manifest = sam_example::output_file(output / "tensors.json");
    manifest << "{\"schema_version\":1,\"byte_order\":\"little\",\"tensors\":{";
    bool first = true;
    for (const char* name : {"preprocessed_image", "vision_features_0", "vision_features_1", "vision_features_2",
                            "text_features", "fusion_features", "pred_boxes", "presence_logits", "class_logits", "mask_logits"}) {
        const auto& tensor = session.tensor(name);
        const auto filename = std::string(name) + ".bin";
        sam_example::write_tensor(output / filename, tensor);
        if (!first) manifest << ',';
        first = false;
        manifest << sam_example::json_string(name) << ":{\"file\":" << sam_example::json_string(filename)
                 << ",\"dtype\":\"float32\",\"shape\":[";
        for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
            if (i) manifest << ',';
            manifest << tensor.shape[i];
        }
        manifest << "],\"layout\":" << sam_example::json_string(tensor.layout) << '}';
    }
    manifest << "}}\n";
    manifest.close();
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 6 || argc > 8)
            throw std::invalid_argument("usage: sam_cache_image_probe MODEL CASES.tsv f32|f16|q8_0 f32|f16 NEW_DIR [ITERATIONS|0] [dump]");
        const std::string cache_name = argv[3], compute_name = argv[4];
        using sam::internal::FeatureCacheMode;
        const auto cache = cache_name == "f32" ? FeatureCacheMode::F32 :
            (cache_name == "f16" ? FeatureCacheMode::F16 : FeatureCacheMode::Q8_0);
        if ((cache_name != "f32" && cache_name != "f16" && cache_name != "q8_0") ||
            (compute_name != "f32" && compute_name != "f16"))
            throw std::invalid_argument("unsupported cache or CUDA compute mode");
        const int iterations = argc >= 7 && std::string(argv[6]) != "0" ? sam_probe::iterations(argv[6]) : 0;
        const bool dump = argc == 8 && std::string(argv[7]) == "dump";
        if (argc == 8 && !dump) throw std::invalid_argument("unknown final argument");
        if (dump && iterations) throw std::invalid_argument("diagnostic dumps and performance runs must be separate");
        const auto cases = read_cases(argv[2]);
        if (iterations && (cases.size() != 1 || cases[0].alternate.empty() || cases[0].alternate == cases[0].prompt))
            throw std::invalid_argument("performance requires one case and a different alternate prompt");
        sam_example::OutputDirectory destination(argv[5], true);
        const std::filesystem::path output_root = argv[5];
        sam::BackendOptions options{sam::Backend::Cuda, 4};
        options.cuda_compute = compute_name == "f32" ? sam::CudaComputeMode::F32 : sam::CudaComputeMode::F16;
        const auto load_start = sam_example::Clock::now();
        const auto state = sam::internal::sam3::load_state(argv[1], options, cache);
        const auto load_ms = sam_example::elapsed_ms(load_start);
        sam::internal::sam3::ImageSession session(state);
        std::string last_image;
        sam_example::Image image;
        std::size_t index = 0;
        for (const auto& item : cases) {
            const auto case_start = sam_example::Clock::now();
            const bool image_reused = item.image == last_image;
            if (!image_reused) {
                image = sam_example::read_image(item.image);
                session.set_image(sam_example::image_view(image));
                last_image = item.image;
            }
            auto result = session.segment_text(item.prompt, 0.5f);
            const auto first_call_ms = sam_example::elapsed_ms(case_start);
            std::vector<double> full_ms, changed_prompt_ms, replay_ms;
            const auto initial_stats = session.stats();
            if (iterations) {
                for (int run = 0; run < iterations + 5; ++run) {
                    const auto start = sam_example::Clock::now();
                    session.set_image(sam_example::image_view(image));
                    result = session.segment_text(item.prompt, 0.5f);
                    if (run >= 5) full_ms.push_back(sam_example::elapsed_ms(start));
                }
                for (int run = 0; run < iterations + 5; ++run) {
                    const auto start = sam_example::Clock::now();
                    result = session.segment_text(run % 2 == 0 ? item.alternate : item.prompt, 0.5f);
                    if (run >= 5) changed_prompt_ms.push_back(sam_example::elapsed_ms(start));
                }
                result = session.segment_text(item.prompt, 0.5f);
                for (int run = 0; run < iterations; ++run) {
                    const auto start = sam_example::Clock::now();
                    result = session.segment_text(item.prompt, 0.5f);
                    replay_ms.push_back(sam_example::elapsed_ms(start));
                }
            }
            const auto measured_stats = session.stats();
            if (measured_stats.cpu_nodes || measured_stats.metal_nodes || !measured_stats.cuda_nodes)
                throw std::runtime_error("cache image experiment escaped strict CUDA compute");
            const auto directory = output_root / item.id;
            if (!std::filesystem::create_directory(directory)) throw std::runtime_error("case output already exists");
            auto report = sam_example::output_file(directory / "results.json");
            report << "{\"schema_version\":1,\"complete\":true,\"experimental\":true,\"width\":" << image.width
                   << ",\"height\":" << image.height << ",\"prompt\":" << sam_example::json_string(item.prompt)
                   << ",\"feature_cache\":" << sam_example::json_string(cache_name)
                   << ",\"feature_cache_bytes\":" << session.feature_cache_bytes()
                   << ",\"cuda_compute\":" << sam_example::json_string(compute_name)
                   << ",\"image_reused\":" << (image_reused ? "true" : "false")
                   << ",\"first_call_ms\":" << first_call_ms << ",\"token_ids\":[";
            const auto& tokens = session.token_ids();
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                if (i) report << ',';
                report << tokens[i];
            }
            sam_example::write_model_profile(report << ']', state->model_info);
            report << ",\"runtime\":";
            sam_example::write_runtime_stats(report, measured_stats);
            report << ",\"first_runtime\":";
            sam_example::write_runtime_stats(report, initial_stats);
            report << ",\"timing_ms\":{\"warmed_full_image\":";
            write_numbers(report, full_ms);
            report << ",\"changed_prompt\":";
            write_numbers(report, changed_prompt_ms);
            report << ",\"repeated_result_cache\":";
            write_numbers(report, replay_ms);
            report << "},\"query_scores\":[";
            const auto& scores = session.tensor("class_logits").values;
            const auto presence = session.tensor("presence_logits").values.front();
            const auto& boxes = session.tensor("pred_boxes").values;
            for (std::size_t query = 0; query < scores.size(); ++query) {
                if (query) report << ',';
                report << sam::internal::sam3::sigmoid(scores[query]) * sam::internal::sam3::sigmoid(presence);
            }
            report << "],\"query_boxes\":[";
            for (std::size_t query = 0; query < scores.size(); ++query) {
                if (query) report << ',';
                const auto* box = boxes.data() + 4 * query;
                report << '[' << (box[0] - box[2] * 0.5f) * image.width << ','
                       << (box[1] - box[3] * 0.5f) * image.height << ','
                       << (box[0] + box[2] * 0.5f) * image.width << ','
                       << (box[1] + box[3] * 0.5f) * image.height << ']';
            }
            report << "],\"detections\":";
            sam_example::write_detections(report, result, directory, false);
            report << "}\n";
            report.close();
            if (dump) dump_tensors(session, directory);
            std::cout << ++index << '/' << cases.size() << ' ' << item.id << " cache=" << session.feature_cache_bytes() << '\n';
        }
        auto receipt = sam_example::output_file(output_root / "run.json");
        receipt << "{\"complete\":true,\"experimental\":true,\"cases\":" << cases.size()
                << ",\"feature_cache\":" << sam_example::json_string(cache_name)
                << ",\"cuda_compute\":" << sam_example::json_string(compute_name)
                << ",\"threads\":4,\"model_load_ms\":" << load_ms << ",\"diagnostic_dump\":" << (dump ? "true" : "false")
                << ",\"iterations\":" << iterations << "}\n";
        receipt.close();
        destination.complete();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_cache_image_probe: " << error.what() << '\n';
        return 1;
    }
}
