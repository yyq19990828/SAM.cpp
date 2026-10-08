#include "../../support/image_io/image_support.hpp"
#include "precision_output.hpp"
#include <models/sam3/model.hpp>

#include <atomic>
#include <exception>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>

namespace {

struct Case { std::string id, image, prompt; };

std::vector<Case> read_cases(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open precision case list");
    std::vector<Case> rows;
    std::set<std::string> ids;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream parser(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(parser, field, '\t')) fields.push_back(field);
        if (fields.size() != 3 || fields[0].empty() || fields[1].empty() || fields[2].empty() ||
            fields[0].find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos ||
            !ids.insert(fields[0]).second || rows.size() >= 32768)
            throw std::invalid_argument("precision cases need safe unique IDs and three nonempty TSV fields");
        rows.push_back({fields[0], fields[1], fields[2]});
    }
    if (!input.eof() || rows.empty()) throw std::invalid_argument("invalid precision case list");
    return rows;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 7)
            throw std::invalid_argument("usage: sam_precision_image_probe MODEL CASES.tsv cpu|metal|cuda f32|f16|mixed-q8_0 f32|f16 NEW_DIR");
        const std::string backend = argv[3], cache = argv[4], compute = argv[5];
        if ((backend != "cpu" && backend != "metal" && backend != "cuda") ||
            (cache != "f32" && cache != "f16" && cache != "mixed-q8_0") || (compute != "f32" && compute != "f16"))
            throw std::invalid_argument("unsupported precision probe mode");
        sam::BackendOptions options{backend == "cuda" ? sam::Backend::Cuda : backend == "metal" ? sam::Backend::Metal : sam::Backend::Cpu, 4};
        options.cuda_compute = compute == "f16" ? sam::CudaComputeMode::F16 : sam::CudaComputeMode::F32;
        using sam::internal::FeatureCacheMode;
        const auto mode = cache == "f32" ? FeatureCacheMode::F32 : cache == "f16" ? FeatureCacheMode::F16 : FeatureCacheMode::Q8_0;
        if (backend != "cuda" && (mode != FeatureCacheMode::F32 || compute != "f32"))
            throw std::invalid_argument("compressed cache and F16 arithmetic require CUDA");
        const auto cases = read_cases(argv[2]);
        sam_example::OutputDirectory destination(argv[6], true);
        const std::filesystem::path root = argv[6];
        const auto state = sam::internal::sam3::load_state(argv[1], options, mode);
        sam::internal::sam3::ImageSession session(state);
        sam_example::Image image;
        std::string last_image;
        std::size_t completed = 0;
        for (const auto& item : cases) {
            if (item.image != last_image) {
                image = sam_example::read_image(item.image);
                session.set_image(sam_example::image_view(image));
                last_image = item.image;
            }
            session.segment_text(item.prompt, 1.0f); // Export below; avoid a duplicate set of deployed masks.
            const auto& class_logits = session.tensor("class_logits").values;
            const auto presence = session.tensor("presence_logits").values.front();
            const auto& boxes = session.tensor("pred_boxes").values;
            const auto& mask_tensor = session.tensor("mask_logits");
            std::vector<float> scores;
            for (float value : class_logits)
                scores.push_back(sam::internal::sam3::sigmoid(value) * sam::internal::sam3::sigmoid(presence));
            const auto ranked = sam_probe::ranked_queries(scores);
            auto required = ranked;
            for (std::size_t query = 0; query < scores.size(); ++query)
                if (scores[query] > 0.5f) required.push_back(static_cast<int>(query));
            std::sort(required.begin(), required.end());
            required.erase(std::unique(required.begin(), required.end()), required.end());
            std::vector<std::string> masks(required.size());
            std::atomic<std::size_t> next{0};
            std::exception_ptr failure;
            std::mutex error_mutex;
            auto worker = [&] {
                try {
                    for (auto index = next.fetch_add(1); index < required.size(); index = next.fetch_add(1)) {
                        auto mask = sam_probe::query_mask(mask_tensor.values, required[index], static_cast<int>(mask_tensor.shape[3]),
                                                         static_cast<int>(mask_tensor.shape[2]), image.width, image.height);
                        masks[index] = sam_probe::compressed_rle(mask);
                    }
                } catch (...) { std::lock_guard<std::mutex> lock(error_mutex); failure = std::current_exception(); }
            };
            std::vector<std::future<void>> workers;
            for (int i = 0; i < 3; ++i) workers.push_back(std::async(std::launch::async, worker));
            worker();
            for (auto& future : workers) future.get();
            if (failure) std::rethrow_exception(failure);
            const auto& stats = session.stats();
            if (backend == "cuda" && (!stats.cuda_nodes || stats.cpu_nodes || stats.metal_nodes || stats.blas_nodes))
                throw std::runtime_error("precision probe escaped strict CUDA compute");
            if (backend == "metal" && (!stats.metal_nodes || stats.cpu_nodes || stats.cuda_nodes || stats.blas_nodes))
                throw std::runtime_error("precision probe escaped strict Metal compute");
            auto report = sam_example::output_file(root / (item.id + ".json"));
            report << "{\"schema_version\":2,\"width\":" << image.width << ",\"height\":" << image.height
                   << ",\"prompt\":" << sam_example::json_string(item.prompt) << ",\"token_ids\":[";
            const auto& tokens = session.token_ids();
            for (std::size_t i = 0; i < tokens.size(); ++i) { if (i) report << ','; report << tokens[i]; }
            report << "],\"query_scores\":[";
            for (std::size_t i = 0; i < scores.size(); ++i) { if (i) report << ','; report << scores[i]; }
            report << "],\"query_boxes\":[";
            for (std::size_t i = 0; i < scores.size(); ++i) {
                if (i) report << ',';
                const auto* box = boxes.data() + 4 * i;
                report << '[' << (box[0] - box[2] * .5f) * image.width << ',' << (box[1] - box[3] * .5f) * image.height << ','
                       << (box[0] + box[2] * .5f) * image.width << ',' << (box[1] + box[3] * .5f) * image.height << ']';
            }
            report << "],\"ranked_queries\":[";
            for (std::size_t i = 0; i < ranked.size(); ++i) { if (i) report << ','; report << ranked[i]; }
            report << "],\"masks\":[";
            for (std::size_t i = 0; i < required.size(); ++i) {
                if (i) report << ',';
                report << "{\"query_index\":" << required[i] << ",\"mask\":{\"size\":[" << image.height << ',' << image.width
                       << "],\"counts\":" << sam_example::json_string(masks[i]) << "}}";
            }
            report << "],\"runtime\":";
            sam_example::write_runtime_stats(report, stats);
            sam_example::write_model_profile(report, state->model_info);
            report << ",\"feature_cache\":" << sam_example::json_string(cache)
                   << ",\"feature_cache_bytes\":" << session.feature_cache_bytes()
                   << ",\"cuda_compute\":" << sam_example::json_string(compute) << "}\n";
            report.close();
            std::cout << ++completed << '/' << cases.size() << ' ' << item.id << std::endl;
        }
        auto receipt = sam_example::output_file(root / "run.json");
        receipt << "{\"schema_version\":2,\"complete\":true,\"kind\":\"sam3-ranked-native-export\",\"cases\":" << cases.size()
                << ",\"backend\":" << sam_example::json_string(backend) << ",\"feature_cache\":" << sam_example::json_string(cache)
                << ",\"cuda_compute\":" << sam_example::json_string(compute) << ",\"threads\":4}\n";
        receipt.close();
        destination.complete();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_precision_image_probe: " << error.what() << '\n';
        return 1;
    }
}
