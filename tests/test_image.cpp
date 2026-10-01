#include "image_support.hpp"

#include <cstring>
#include <iostream>

namespace {

void write_tensor(const std::filesystem::path& path, const sam::TensorData& tensor) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "Reference dumps require IEEE-754 binary32");
    std::size_t count = 1;
    if (tensor.shape.empty()) throw std::runtime_error("Tensor shape is empty");
    for (const auto dimension : tensor.shape) {
        if (dimension <= 0 || static_cast<std::uint64_t>(dimension) >
                                  std::numeric_limits<std::size_t>::max() / count) {
            throw std::runtime_error("Tensor shape is invalid");
        }
        count *= static_cast<std::size_t>(dimension);
    }
    if (count != tensor.values.size()) throw std::runtime_error("Tensor size does not match shape");
    auto file = sam_example::output_file(path, true);
    const std::uint16_t endian = 1;
    if (*reinterpret_cast<const std::uint8_t*>(&endian) == 1) {
        file.write(reinterpret_cast<const char*>(tensor.values.data()),
                   static_cast<std::streamsize>(count * sizeof(float)));
    } else {
        for (const float value : tensor.values) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            const char bytes[4] = {static_cast<char>(bits), static_cast<char>(bits >> 8),
                                   static_cast<char>(bits >> 16), static_cast<char>(bits >> 24)};
            file.write(bytes, sizeof(bytes));
        }
    }
    file.close();
}

} // namespace

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
            write_tensor(options.output / filename, tensor);
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
