#include "image_support.hpp"

#include <iostream>
#include <sstream>

namespace {

sam_example::Options parse(std::vector<std::string> arguments, bool allow_repeat = true) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return sam_example::parse_options(static_cast<int>(argv.size()), argv.data(), allow_repeat);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        const std::vector<std::string> valid = {"sam_image", "--model", "model.gguf", "--image", "image.png",
                                                "--text", "truck", "--output", "output"};
        auto arguments = valid;
        arguments.insert(arguments.end(), {"--backend", "metal", "--threads", "2", "--repeat", "5",
                                          "--score-threshold", "0.6"});
        const auto options = parse(arguments);
        require(options.backend.backend == sam::Backend::Metal && options.backend.threads == 2 &&
                options.repeat == 5 && std::abs(options.score_threshold - 0.6f) < 1e-6f,
                "CLI options were not parsed correctly");
        for (const auto& suffix : std::vector<std::vector<std::string>>{
                 {"--threads", "0"}, {"--repeat", "-1"}, {"--repeat", "2junk"},
                 {"--score-threshold", "nan"}, {"--score-threshold", "1.1"},
                 {"--score-threshold", " 0.5"}, {"--score-threshold", "1e-999"},
                 {"--backend", "unknown"}, {"--cuda-device", "0"}, {"--backend", "cuda", "--cuda-device", "-1"},
                 {"--backend", "cuda", "--cuda-device", "1junk"}, {"--backend", "cuda", "--cuda-device", "2147483648"},
                 {"--model", "other.gguf"}, {"--wat", "x"}, {"--threads"}}) {
            arguments = valid;
            arguments.insert(arguments.end(), suffix.begin(), suffix.end());
            bool rejected = false;
            try { (void) parse(arguments); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Invalid CLI arguments were accepted");
        }
        for (const auto& index : {"0", "2"}) {
            arguments = valid;
            arguments.insert(arguments.end(), {"--cuda-device", index, "--backend", "cuda"});
            const auto cuda = parse(arguments);
            require(cuda.backend.backend == sam::Backend::Cuda && cuda.backend.cuda_device == std::stoi(index),
                    "CUDA device selection was not preserved");
        }
        require(parse({"sam_image", "--help"}).help, "Help should work without model input");
        bool rejected = false;
        try { (void) parse({"sam_image"}); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Required arguments were not enforced");
        require(sam_example::json_string("\"\\\n") == "\"\\\"\\\\\\u000a\"", "JSON escaping failed");
        sam::RuntimeStats timings;
        timings.image_ms = 1.25; timings.text_ms = 2.5; timings.inference_ms = 3.75; timings.cuda_nodes = 17;
        std::ostringstream timing_json;
        sam_example::write_runtime_stats(timing_json, timings);
        for (const auto& field : {"\"image_ms\":1.25", "\"text_ms\":2.5", "\"inference_ms\":3.75", "\"cuda_nodes\":17"})
            require(timing_json.str().find(field) != std::string::npos, "Runtime encoding timer was omitted from JSON");

        const auto directory = std::filesystem::temp_directory_path() /
            ("sam-image-io-" + std::to_string(sam_example::Clock::now().time_since_epoch().count()));
        {
            sam_example::OutputDirectory output(directory);
            rejected = false;
            try { sam_example::OutputDirectory duplicate(directory); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "Existing output directory was accepted");
            const std::vector<std::uint8_t> mask = {0, 1, 1, 0, 0, 1};
            sam_example::write_mask_png(directory / "mask.png", 3, 2, mask);
            const auto decoded = sam_example::read_image(directory / "mask.png");
            require(decoded.width == 3 && decoded.height == 2 && decoded.rgb.size() == 18,
                    "PNG dimensions did not round trip");
            for (std::size_t i = 0; i < mask.size(); ++i) {
                for (int channel = 0; channel < 3; ++channel) {
                    require(decoded.rgb[3 * i + channel] == (mask[i] ? 255 : 0),
                            "Binary PNG mask did not round trip");
                }
            }
            std::ostringstream json;
            sam_example::write_detections(json, {}, directory, false);
            require(json.str() == "[]", "Empty detections should serialize successfully");
            sam::Result invalid;
            invalid.detections.push_back({{}, std::numeric_limits<float>::quiet_NaN(), {1, 1, {0}}, 0});
            rejected = false;
            try { sam_example::write_detections(json, invalid, directory, false); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "Non-finite detections should not become JSON");
        }
        require(!std::filesystem::exists(directory), "Incomplete output directory was not removed");
        try {
            sam_example::OutputDirectory output(directory);
            (void) sam_example::read_image(directory / "missing.png");
            throw std::logic_error("Missing image should fail decoding");
        } catch (const std::runtime_error&) {}
        require(!std::filesystem::exists(directory), "Failed decoding leaked an output directory");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "image CLI check: " << error.what() << '\n';
        return 1;
    }
}
