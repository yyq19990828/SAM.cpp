#include "image_support.hpp"

#include <iostream>
#include <optional>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void same_result(const sam::Result& actual, const sam::Result& expected) {
    require(actual.detections.size() == expected.detections.size(), "Session detection count changed");
    for (std::size_t i = 0; i < actual.detections.size(); ++i) {
        const auto& a = actual.detections[i];
        const auto& b = expected.detections[i];
        require(a.query_index == b.query_index && a.score == b.score &&
                a.box.x0 == b.box.x0 && a.box.y0 == b.box.y0 &&
                a.box.x1 == b.box.x1 && a.box.y1 == b.box.y1 &&
                a.mask.width == b.mask.width && a.mask.height == b.mask.height && a.mask.data == b.mask.data,
                "Session results changed after another session or a cache hit");
    }
}

void same_computation_counts(const sam::RuntimeStats& a, const sam::RuntimeStats& b) {
    require(a.vision_encodes == b.vision_encodes && a.text_encodes == b.text_encodes &&
            a.inferences == b.inferences && a.compute_buffer_bytes == b.compute_buffer_bytes &&
            a.cpu_nodes == b.cpu_nodes && a.metal_nodes == b.metal_nodes && a.cuda_nodes == b.cuda_nodes &&
            a.blas_nodes == b.blas_nodes &&
            a.graph_partitions == b.graph_partitions && a.host_upload_bytes == b.host_upload_bytes &&
            a.host_download_bytes == b.host_download_bytes,
            "A cached prompt unexpectedly executed or allocated a graph");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: test_session MODEL IMAGE_A IMAGE_B PROMPT_A PROMPT_B cpu|metal|cuda THREADS NEW_DIR\n";
            return 0;
        }
        if (argc != 9) throw std::invalid_argument("Use --help for the eight positional arguments");
        std::vector<std::string> arguments = {"test_session", "--model", argv[1], "--image", argv[2],
            "--text", argv[4], "--backend", argv[6], "--threads", argv[7], "--output", argv[8]};
        std::vector<char*> pointers;
        for (auto& argument : arguments) pointers.push_back(argument.data());
        const auto options = sam_example::parse_options(static_cast<int>(pointers.size()), pointers.data(), false);
        require(options.backend.backend != sam::Backend::Auto, "Session check requires an explicit backend");
        require(options.text != argv[5], "Session check requires two distinct prompts");
        require(!std::filesystem::exists(options.output), "Output directory already exists");
        const auto image_a = sam_example::read_image(options.image);
        const auto image_b = sam_example::read_image(argv[3]);
        require(image_a.rgb != image_b.rgb, "Session check requires two distinct images");
        std::optional<sam::Model> model(sam::Model::load(options.model.string(), options.backend));
        const auto info = model->info();
        sam::ImageSession first(*model), second(*model);
        model.reset(); // Sessions must retain the model after its last public owner is destroyed.
        first.set_image(sam_example::image_view(image_a));
        const auto result_a = first.segment_text(options.text);
        const auto tokens_a = first.token_ids();
        const auto initial = first.stats();
        auto start = sam_example::Clock::now();
        const auto result_b = first.segment_text(argv[5]);
        const double new_prompt_ms = sam_example::elapsed_ms(start);
        require(first.token_ids() != tokens_a, "Prompts must tokenize differently");
        require(first.stats().vision_encodes == initial.vision_encodes &&
                first.stats().text_encodes == initial.text_encodes + 1 &&
                first.stats().inferences == initial.inferences + 1,
                "A changed prompt failed to reuse vision or failed to recompute predictions");
        const auto cached = first.stats();
        same_result(first.segment_text(argv[5]), result_b);
        same_computation_counts(first.stats(), cached);

        second.set_image(sam_example::image_view(image_b));
        const auto second_result = second.segment_text(options.text);
        const auto second_cached = second.stats();
        same_result(first.segment_text(argv[5]), result_b);
        same_computation_counts(first.stats(), cached);
        same_result(second.segment_text(options.text), second_result);
        same_computation_counts(second.stats(), second_cached);

        if (info.backend == sam::Backend::Cuda) {
            require(!info.device_name.empty() && info.cuda_device == 0, "Session CUDA device identity is missing");
            for (const auto* stats : {&first.stats(), &second.stats()})
                require(stats->cuda_nodes > 0 && stats->cpu_nodes == 0 && stats->metal_nodes == 0 && stats->blas_nodes == 0,
                        "Session used compute outside the selected CUDA device");
        }

        first.set_image(sam_example::image_view(image_b));
        same_result(first.segment_text(options.text), second_result);
        require(first.stats().vision_encodes == cached.vision_encodes + 1 &&
                first.stats().text_encodes == cached.text_encodes + 1 &&
                first.stats().inferences == cached.inferences + 1,
                "Replacing the image failed to invalidate predictions");
        first.set_image(sam_example::image_view(image_a));
        same_result(first.segment_text(options.text), result_a);

        // Warm both prompt shapes before observing repeated allocations.
        same_result(first.segment_text(argv[5]), result_b);
        const auto plateau = first.stats().compute_buffer_bytes;
        const auto rss_before = sam_example::process_peak_rss_bytes();
        std::vector<double> changed_prompt_ms;
        std::vector<double> cache_hit_ms;
        for (int run = 0; run < 5; ++run) {
            {
                start = sam_example::Clock::now();
                const auto actual = first.segment_text(options.text);
                changed_prompt_ms.push_back(sam_example::elapsed_ms(start));
                same_result(actual, result_a);
            }
            {
                start = sam_example::Clock::now();
                const auto actual = first.segment_text(argv[5]);
                changed_prompt_ms.push_back(sam_example::elapsed_ms(start));
                same_result(actual, result_b);
            }
            const auto before_cache = first.stats();
            {
                start = sam_example::Clock::now();
                const auto actual = first.segment_text(argv[5]);
                cache_hit_ms.push_back(sam_example::elapsed_ms(start));
                same_result(actual, result_b);
            }
            same_computation_counts(first.stats(), before_cache);
            require(first.stats().compute_buffer_bytes == plateau, "Compute-buffer high-water mark grew after warm-up");
        }
        same_result(second.segment_text(options.text), second_result);
        same_computation_counts(second.stats(), second_cached);

        sam_example::OutputDirectory output(options.output);
        auto json = sam_example::output_file(options.output / "results.json");
        json << "{\"schema_version\":1,\"passed\":true,\"backend\":"
             << sam_example::json_string(sam_example::backend_name(info.backend))
             << ",\"model\":" << sam_example::json_string(options.model.string())
             << ",\"image_a\":" << sam_example::json_string(options.image.string())
             << ",\"image_b\":" << sam_example::json_string(argv[3])
             << ",\"prompt_a\":" << sam_example::json_string(options.text)
             << ",\"prompt_b\":" << sam_example::json_string(argv[5])
             << ",\"threads\":" << info.threads
             << ",\"precision\":" << sam_example::json_string(info.precision)
             << ",\"device_name\":" << sam_example::json_string(info.device_name)
             << ",\"cuda_device\":" << info.cuda_device
             << ",\"tokenizer_compatibility_repaired\":"
             << (info.tokenizer_compatibility_repaired ? "true" : "false")
             << ",\"changed_prompt_runs\":10,\"new_prompt_cached_vision_ms\":" << new_prompt_ms
             << ",\"changed_prompt_median_ms\":" << sam_example::median(changed_prompt_ms)
             << ",\"prediction_cache_median_ms\":" << sam_example::median(cache_hit_ms)
             << ",\"compute_buffer_high_water_bytes\":" << plateau
             << ",\"process_peak_rss_before_repeats_bytes\":" << rss_before
             << ",\"first_runtime\":";
        sam_example::write_runtime_stats(json, first.stats());
        json << ",\"second_runtime\":";
        sam_example::write_runtime_stats(json, second.stats());
        json << "}\n";
        json.close();
        output.complete();
        std::cout << "Session behavior checks passed on " << sam_example::backend_name(info.backend) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "session check: " << error.what() << '\n';
        return 1;
    }
}
