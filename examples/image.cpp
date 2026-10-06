#include "image_support.hpp"

#include <iostream>

int main(int argc, char** argv) {
    try {
        const auto options = sam_example::parse_options(argc, argv);
        if (options.help) {
            std::cout << "Usage: sam_image --model FILE --image FILE --text PROMPT --output NEW_DIR\n"
                         "  [--backend auto|cpu|metal|cuda] [--cuda-device N] [--threads N] [--score-threshold 0.5] [--repeat N]\n"
                         "Repeat measures warmed full-image and repeated-result-cache calls separately.\n";
            return 0;
        }
        if (std::filesystem::exists(options.output)) {
            throw std::runtime_error("Output directory already exists: " + options.output.string());
        }
        const auto cold_start = sam_example::Clock::now();
        auto start = sam_example::Clock::now();
        const auto image = sam_example::read_image(options.image);
        const double decode_ms = sam_example::elapsed_ms(start);
        start = sam_example::Clock::now();
        const auto model = sam::Model::load(options.model.string(), options.backend);
        const double model_load_ms = sam_example::elapsed_ms(start);
        sam::ImageSession session(model);
        session.set_image(sam_example::image_view(image));
        auto result = session.segment_text(options.text, options.score_threshold);
        const double cold_start_ms = sam_example::elapsed_ms(cold_start);

        std::vector<double> full_image_ms;
        std::vector<sam::RuntimeStats> full_image_stats;
        std::vector<double> repeated_result_cache_ms;
        for (int run = 0; run < options.repeat; ++run) {
            start = sam_example::Clock::now();
            session.set_image(sam_example::image_view(image));
            result = session.segment_text(options.text, options.score_threshold);
            full_image_ms.push_back(sam_example::elapsed_ms(start));
            full_image_stats.push_back(session.stats());
        }
        for (int run = 0; run < options.repeat; ++run) {
            start = sam_example::Clock::now();
            result = session.segment_text(options.text, options.score_threshold);
            repeated_result_cache_ms.push_back(sam_example::elapsed_ms(start));
        }

        const auto& info = model.info();
        const auto& stats = session.stats();
        sam_example::OutputDirectory output(options.output);
        auto json = sam_example::output_file(options.output / "results.json");
        json << "{\n\"schema_version\":1,\"width\":" << image.width << ",\"height\":" << image.height
             << ",\"prompt\":" << sam_example::json_string(options.text)
             << ",\"score_threshold\":" << options.score_threshold
             << ",\"model\":" << sam_example::json_string(options.model.string())
             << ",\"image\":" << sam_example::json_string(options.image.string())
             << ",\"architecture\":" << sam_example::json_string(info.architecture);
        sam_example::write_model_profile(json, info);
        json << ",\"tokenizer_compatibility_repaired\":" << (info.tokenizer_compatibility_repaired ? "true" : "false")
             << ",\"backend\":" << sam_example::json_string(sam_example::backend_name(model.backend()))
             << ",\"threads\":" << info.threads << ",\"repeat\":" << options.repeat
             << ",\"tensor_count\":" << info.tensor_count << ",\"weight_bytes\":" << info.weight_bytes
             << ",\"timing_ms\":{\"decode\":" << decode_ms << ",\"model_load\":" << model_load_ms
             << ",\"cold_start\":" << cold_start_ms
             << ",\"warmed_full_image_median\":" << sam_example::median(full_image_ms)
             << ",\"repeated_result_cache_median\":" << sam_example::median(repeated_result_cache_ms)
             << ",\"last_computed_image\":" << stats.image_ms << ",\"last_computed_text\":" << stats.text_ms
             << ",\"last_computed_inference\":" << stats.inference_ms
             << ",\"warmed_full_image_runs\":[";
        for (std::size_t run = 0; run < full_image_ms.size(); ++run) {
            if (run) json << ',';
            const auto& sample = full_image_stats[run];
            json << "{\"total\":" << full_image_ms[run] << ",\"image\":" << sample.image_ms
                 << ",\"text\":" << sample.text_ms << ",\"inference\":" << sample.inference_ms << '}';
        }
        json << "],\"repeated_result_cache_runs\":[";
        for (std::size_t run = 0; run < repeated_result_cache_ms.size(); ++run) {
            if (run) json << ',';
            json << repeated_result_cache_ms[run];
        }
        json << "]},\"runtime\":";
        sam_example::write_runtime_stats(json, stats);
        json << ",\"detections\":";
        sam_example::write_detections(json, result, options.output, false);
        json << "\n}\n";
        json.close();
        output.complete();
        std::cout << "Wrote " << result.detections.size() << " detections to " << options.output.string()
                  << " using " << sam_example::backend_name(model.backend()) << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_image: " << error.what() << '\n';
        return 1;
    }
}
