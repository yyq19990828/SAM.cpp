#include "image_support.hpp"
#include <iostream>
#include <sstream>

namespace {

std::string frame_name(int frame) {
    std::ostringstream text; text << std::setw(6) << std::setfill('0') << frame; return text.str();
}

void write_video_stats(std::ostream& output, const sam::VideoStats& stats) {
    output << "{\"accepted_frames\":" << stats.accepted_frames << ",\"emitted_frames\":" << stats.emitted_frames
           << ",\"tracker_calls\":" << stats.tracker_calls << ",\"active_objects\":" << stats.active_objects
           << ",\"rejected_new_objects\":" << stats.rejected_new_objects
           << ",\"retained_records\":" << stats.retained_records << ",\"retained_memory_bytes\":" << stats.retained_memory_bytes
           << ",\"retained_records_high_water\":" << stats.retained_records_high_water
           << ",\"pending_frames\":" << stats.pending_frames << ",\"pending_high_water\":" << stats.pending_high_water
           << ",\"frame_ms\":" << stats.frame_ms << ",\"tracker_ms\":" << stats.tracker_ms << ",\"memory_ms\":" << stats.memory_ms
           << ",\"runtime\":";
    sam_example::write_runtime_stats(output, stats.runtime); output << '}';
}

void write_frame(const sam::VideoFrameResult& frame, int emitted_after, const std::filesystem::path& root) {
    const auto directory = root / frame_name(frame.frame_index);
    std::filesystem::create_directory(directory);
    auto json = sam_example::output_file(directory / "results.json");
    json << "{\"schema_version\":1,\"frame_index\":" << frame.frame_index
         << ",\"emitted_after_frame\":" << emitted_after << ",\"objects\":[";
    for (std::size_t i = 0; i < frame.objects.size(); ++i) {
        if (i) json << ',';
        const auto& object = frame.objects[i];
        const auto filename = "object-" + std::to_string(object.id) + ".bin";
        auto mask = sam_example::output_file(directory / filename, true);
        mask.write(reinterpret_cast<const char*>(object.mask.data.data()), object.mask.data.size()); mask.close();
        json << "{\"id\":" << object.id << ",\"score\":" << object.score << ",\"box\":["
             << object.box.x0 << ',' << object.box.y0 << ',' << object.box.x1 << ',' << object.box.y1
             << "],\"mask\":{\"file\":" << sam_example::json_string(filename)
             << ",\"dtype\":\"uint8\",\"shape\":[" << object.mask.height << ',' << object.mask.width << "]}}";
    }
    json << "]}\n";
}

void write_tensors(const sam::VideoSession& session, const std::filesystem::path& directory) {
    std::filesystem::create_directory(directory);
    auto json = sam_example::output_file(directory / "tensors.json");
    json << "{\"schema_version\":1,\"byte_order\":\"little\",\"tensors\":{";
    bool first = true;
    for (const auto& name : session.tensor_names()) {
        const auto& tensor = session.tensor(name);
        const auto filename = name + ".bin";
        sam_example::write_tensor(directory / filename, tensor);
        if (!first) json << ','; first = false;
        json << sam_example::json_string(name) << ":{\"file\":" << sam_example::json_string(filename)
             << ",\"dtype\":\"float32\",\"shape\":[";
        for (std::size_t i = 0; i < tensor.shape.size(); ++i) { if (i) json << ','; json << tensor.shape[i]; }
        json << "],\"layout\":" << sam_example::json_string(tensor.layout) << '}';
    }
    json << "}}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> arguments{"sam_video"};
        sam::VideoOptions video_options;
        bool dump = false, dump_all = false, has_frames = false, has_cap = false;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--dump-tensors") { dump = true; continue; }
            if (argument == "--dump-all-tensors") { dump = dump_all = true; continue; }
            if (argument == "--max-objects") {
                if (has_cap || ++i >= argc) throw std::invalid_argument("--max-objects requires one positive integer");
                has_cap = true; video_options.max_objects = sam_example::positive_integer(argv[i], argument); continue;
            }
            if (argument == "--image" || argument == "--score-threshold") throw std::invalid_argument("use --frames for video input");
            if (argument == "--frames") { has_frames = true; arguments.push_back("--image"); }
            else arguments.push_back(argument);
        }
        std::vector<char*> pointers;
        for (auto& argument : arguments) pointers.push_back(argument.data());
        const auto options = sam_example::parse_options(pointers.size(), pointers.data(), false);
        if (options.help) {
            std::cout << "Usage: sam_video --model FILE --frames PNG_DIRECTORY --text PROMPT --output NEW_DIR\n"
                         "  [--backend auto|cpu|metal|cuda] [--cuda-device N] [--cuda-compute f32|f16]\n"
                         "  [--cpu-compute f32|native-quantized] (native-quantized requires quantized weights)\n"
                         "  [--threads N] [--max-objects 8]\n"
                         "  [--dump-tensors | --dump-all-tensors]\n"
                         "Frames must be contiguous numeric PNG files (000000.png onward).\n";
            return 0;
        }
        if (!has_frames || !std::filesystem::is_directory(options.image)) throw std::invalid_argument("--frames must name a PNG directory");
        if (std::filesystem::exists(options.output)) throw std::runtime_error("Output directory already exists");
        std::size_t count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(options.image))
            if (entry.path().extension() == ".png") ++count;
        if (!count || count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("video requires a positive INT32 frame count");
        const int frame_count = static_cast<int>(count);
        for (int i = 0; i < frame_count; ++i)
            if (!std::filesystem::is_regular_file(options.image / (frame_name(i) + ".png")))
                throw std::invalid_argument("PNG frames are missing or not contiguously numbered");
        const auto load_start = sam_example::Clock::now();
        const auto model = sam::Model::load(options.model.string(), options.backend);
        const auto load_ms = sam_example::elapsed_ms(load_start);
        sam::VideoSession session(model, frame_count, video_options); session.set_text(options.text);
        sam_example::OutputDirectory output(options.output, true);
        std::filesystem::create_directory(options.output / "trace");
        std::filesystem::create_directory(options.output / "tensors");
        int width = 0, height = 0;
        const auto write_manifest = [&](bool complete) {
            auto manifest = sam_example::output_file(options.output / "manifest.json");
            manifest << "{\"schema_version\":1,\"task\":\"text_video\",\"complete\":" << (complete ? "true" : "false")
                     << ",\"frame_count\":" << frame_count << ",\"max_objects\":" << video_options.max_objects
                     << ",\"width\":" << width << ",\"height\":" << height
                     << ",\"prompt\":" << sam_example::json_string(options.text)
                     << ",\"model\":" << sam_example::json_string(options.model.string())
                     << ",\"precision\":" << sam_example::json_string(model.info().precision)
                     << ",\"storage_profile\":" << sam_example::json_string(model.info().storage_profile)
                     << ",\"arithmetic_profile\":" << sam_example::json_string(model.info().arithmetic_profile)
                     << ",\"backend\":" << sam_example::json_string(sam_example::backend_name(model.backend()))
                     << ",\"device_name\":" << sam_example::json_string(model.info().device_name)
                     << ",\"cuda_device\":" << model.info().cuda_device
                     << ",\"threads\":" << model.info().threads << ",\"model_load_ms\":" << load_ms;
            sam_example::write_compute_policy(manifest, options.backend);
            manifest << ",\"token_ids\":[";
            for (std::size_t i = 0; i < session.token_ids().size(); ++i) { if (i) manifest << ','; manifest << session.token_ids()[i]; }
            manifest << "],\"stats\":"; write_video_stats(manifest, session.stats()); manifest << "}\n";
        };
        write_manifest(false);
        for (int i = 0; i < frame_count; ++i) {
            const auto image = sam_example::read_image(options.image / (frame_name(i) + ".png"));
            if (i == 0) { width = image.width; height = image.height; }
            const auto results = session.push_frame(i, sam_example::image_view(image));
            auto trace = sam_example::output_file(options.output / "trace" / (frame_name(i) + ".json"));
            trace << session.trace_json() << '\n'; trace.close();
            auto sample = sam_example::output_file(options.output / "trace" / (frame_name(i) + "-stats.json"));
            write_video_stats(sample, session.stats()); sample << '\n'; sample.close();
            if (dump && (dump_all || i == 0 || i == 1 || i == 16 || i == frame_count - 1))
                write_tensors(session, options.output / "tensors" / frame_name(i));
            for (const auto& result : results) write_frame(result, i, options.output);
            std::cout << "Processed " << i + 1 << '/' << frame_count << ", active " << session.stats().active_objects
                      << ", emitted " << session.stats().emitted_frames << '\n' << std::flush;
        }
        if (session.stats().emitted_frames != count || session.stats().pending_frames != 0)
            throw std::runtime_error("video output was not completely drained");
        write_manifest(true); output.complete();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_video: " << error.what() << '\n'; return 1;
    }
}
