#include "image_support.hpp"

#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const auto options = sam_example::parse_options(argc, argv, false);
        if (options.help) {
            std::cout << "Usage: test_image --model FILE --image FILE --text PROMPT --output NEW_DIR\n"
                         "  [--backend auto|cpu|metal] [--threads N] [--score-threshold 0.5]\n";
            return 0;
        }
        if (std::filesystem::exists(options.output)) {
            throw std::runtime_error("Output directory already exists: " + options.output.string());
        }
        const auto image = sam_example::read_image(options.image);
        const auto model = sam::Model::load(options.model.string(), options.backend);
        sam::ImageSession session(model);
        session.set_image(sam_example::image_view(image));
        const auto result = session.segment_text(options.text, options.score_threshold);
        const auto computed_stats = session.stats();
        (void) session.segment_text(options.text, options.score_threshold);
        if (session.stats().vision_encodes != computed_stats.vision_encodes ||
            session.stats().text_encodes != computed_stats.text_encodes ||
            session.stats().inferences != computed_stats.inferences) {
            throw std::runtime_error("Repeated prompt failed to reuse cached image/text predictions");
        }

        sam_example::OutputDirectory output(options.output);
        auto manifest = sam_example::output_file(options.output / "tensors.json");
        manifest << "{\"schema_version\":1,\"byte_order\":\"little\",\"token_ids\":[";
        const auto& tokens = session.token_ids();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (i) manifest << ',';
            manifest << tokens[i];
        }
        manifest << "],\"tensors\":{";
        bool first = true;
        for (const char* name : {"preprocessed_image", "vision_features_0", "vision_features_1",
                                "vision_features_2", "text_features", "fusion_features", "pred_boxes", "presence_logits",
                                "class_logits", "mask_logits"}) {
            const auto& tensor = session.tensor(name);
            const std::string filename = std::string(name) + ".bin";
            sam_example::write_tensor(options.output / filename, tensor);
            if (!first) manifest << ',';
            first = false;
            manifest << sam_example::json_string(name) << ":{\"file\":"
                     << sam_example::json_string(filename) << ",\"dtype\":\"float32\",\"shape\":[";
            for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
                if (i) manifest << ',';
                manifest << tensor.shape[i];
            }
            manifest << "],\"layout\":" << sam_example::json_string(tensor.layout) << "}";
        }
        manifest << "}}\n";
        manifest.close();
        auto results = sam_example::output_file(options.output / "results.json");
        results << "{\"schema_version\":1,\"width\":" << image.width << ",\"height\":" << image.height
                << ",\"prompt\":" << sam_example::json_string(options.text)
                << ",\"score_threshold\":" << options.score_threshold
                << ",\"backend\":" << sam_example::json_string(sam_example::backend_name(model.backend()))
                << ",\"precision\":" << sam_example::json_string(model.info().precision)
                << ",\"storage_profile\":" << sam_example::json_string(model.info().storage_profile)
                << ",\"tokenizer_compatibility_repaired\":"
                << (model.info().tokenizer_compatibility_repaired ? "true" : "false")
                << ",\"runtime\":";
        sam_example::write_runtime_stats(results, session.stats());
        results << ",\"detections\":";
        sam_example::write_detections(results, result, options.output, true);
        results << "}\n";
        results.close();
        output.complete();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_image: " << error.what() << '\n';
        return 1;
    }
}
