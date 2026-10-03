#include "image_support.hpp"
#include <iostream>
#include <optional>
#include <sstream>

namespace {

constexpr int frame_count = 64;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string frame_name(int frame) {
    std::ostringstream text;
    text << std::setw(6) << std::setfill('0') << frame;
    return text.str();
}

std::uint64_t fingerprint(const sam::VideoFrameResult& frame) {
    std::uint64_t value = 14695981039346656037ULL;
    const auto bytes = [&](const void* data, std::size_t size) {
        const auto* begin = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) value = (value ^ begin[i]) * 1099511628211ULL;
    };
    bytes(&frame.frame_index, sizeof(frame.frame_index));
    const auto count = frame.objects.size();
    bytes(&count, sizeof(count));
    for (const auto& object : frame.objects) {
        bytes(&object.id, sizeof(object.id));
        bytes(&object.score, sizeof(object.score));
        for (const auto coordinate : {object.box.x0, object.box.y0, object.box.x1, object.box.y1})
            bytes(&coordinate, sizeof(coordinate));
        bytes(&object.mask.width, sizeof(object.mask.width));
        bytes(&object.mask.height, sizeof(object.mask.height));
        bytes(object.mask.data.data(), object.mask.data.size());
    }
    return value;
}

void write_stats(std::ostream& output, const sam::VideoStats& stats) {
    output << "{\"accepted_frames\":" << stats.accepted_frames << ",\"emitted_frames\":" << stats.emitted_frames
           << ",\"tracker_calls\":" << stats.tracker_calls << ",\"active_objects\":" << stats.active_objects
           << ",\"rejected_new_objects\":" << stats.rejected_new_objects
           << ",\"retained_records\":" << stats.retained_records << ",\"retained_memory_bytes\":" << stats.retained_memory_bytes
           << ",\"retained_records_high_water\":" << stats.retained_records_high_water
           << ",\"pending_frames\":" << stats.pending_frames << ",\"pending_high_water\":" << stats.pending_high_water
           << ",\"frame_ms\":" << stats.frame_ms << ",\"tracker_ms\":" << stats.tracker_ms << ",\"memory_ms\":" << stats.memory_ms
           << ",\"runtime\":";
    sam_example::write_runtime_stats(output, stats.runtime);
    output << '}';
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
        mask.write(reinterpret_cast<const char*>(object.mask.data.data()), object.mask.data.size());
        mask.close();
        json << "{\"id\":" << object.id << ",\"score\":" << object.score << ",\"box\":["
             << object.box.x0 << ',' << object.box.y0 << ',' << object.box.x1 << ',' << object.box.y1
             << "],\"mask\":{\"file\":" << sam_example::json_string(filename)
             << ",\"dtype\":\"uint8\",\"shape\":[" << object.mask.height << ',' << object.mask.width << "]}}";
    }
    json << "]}\n";
}

void unchanged(const sam::VideoStats& actual, const sam::VideoStats& before) {
    require(actual.accepted_frames == before.accepted_frames && actual.emitted_frames == before.emitted_frames &&
            actual.active_objects == before.active_objects && actual.retained_records == before.retained_records &&
            actual.retained_memory_bytes == before.retained_memory_bytes && actual.pending_frames == before.pending_frames &&
            actual.runtime.vision_encodes == before.runtime.vision_encodes &&
            actual.runtime.text_encodes == before.runtime.text_encodes &&
            actual.runtime.inferences == before.runtime.inferences &&
            actual.runtime.cpu_nodes == before.runtime.cpu_nodes && actual.runtime.metal_nodes == before.runtime.metal_nodes &&
            actual.runtime.compute_buffer_bytes == before.runtime.compute_buffer_bytes,
            "pushing one video changed the other video's state or graph counters");
}

void bounded(const sam::VideoStats& stats, int frame, sam::Backend backend) {
    const auto emitted = frame == frame_count - 1 ? frame_count : std::max(0, frame - 13);
    require(stats.accepted_frames == static_cast<std::uint64_t>(frame + 1) &&
            stats.emitted_frames == static_cast<std::uint64_t>(emitted) &&
            stats.pending_frames == static_cast<std::size_t>(frame + 1 - emitted) && stats.pending_high_water <= 15 &&
            stats.retained_records <= 27 * stats.active_objects,
            "video exceeded frame, drain or retained-state bounds");
    require(stats.runtime.vision_encodes == static_cast<std::uint64_t>(frame + 1) &&
            stats.runtime.inferences == static_cast<std::uint64_t>(frame + 1) && stats.runtime.text_encodes == 1,
            "a session recomputed text or reused another session's vision cache");
    require(backend == sam::Backend::Metal
                ? stats.runtime.metal_nodes > 0 && stats.runtime.cpu_nodes == 0
                : stats.runtime.cpu_nodes > 0 && stats.runtime.metal_nodes == 0,
            "long video used the wrong backend or fallback");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: test_video_long_session VIDEO_MODEL ENTRY_64_PNG_DIRECTORY NEGATIVE_IMAGE cpu|metal THREADS NEW_DIR\n";
            return 0;
        }
        if (argc != 7) throw std::invalid_argument("Use --help for the six positional arguments");
        std::vector<std::string> arguments{"test_video_long_session", "--model", argv[1], "--image", argv[2],
            "--text", "truck", "--backend", argv[4], "--threads", argv[5], "--output", argv[6]};
        std::vector<char*> pointers;
        for (auto& argument : arguments) pointers.push_back(argument.data());
        const auto options = sam_example::parse_options(pointers.size(), pointers.data(), false);
        require(options.backend.backend != sam::Backend::Auto, "long video check requires an explicit backend");
        require(!std::filesystem::exists(options.output), "output already exists");
        for (int i = 0; i < frame_count; ++i)
            require(std::filesystem::is_regular_file(options.image / (frame_name(i) + ".png")),
                    "the positive sequence must contain all 64 numbered PNG frames");
        const auto negative_image = sam_example::read_image(argv[3]);
        std::optional<sam::Model> model(sam::Model::load(options.model.string(), options.backend));
        const auto info = model->info();
        sam_example::OutputDirectory output(options.output, true);
        const auto positive_directory = options.output / "positive", negative_directory = options.output / "negative";
        for (const auto& directory : {positive_directory, negative_directory})
            std::filesystem::create_directories(directory / "trace");
        std::optional<sam::VideoFrameResult> owned_positive, owned_negative;
        std::uint64_t positive_hash = 0, negative_hash = 0;
        const auto check_owned = [&] {
            require(!owned_positive || fingerprint(*owned_positive) == positive_hash, "owned positive result changed");
            require(!owned_negative || fingerprint(*owned_negative) == negative_hash, "owned negative result changed");
        };
        sam::VideoStats positive_stats, negative_stats;
        {
            sam::VideoSession positive(*model, frame_count), negative(*model, frame_count);
            model.reset(); // Both live sessions must outlast the last public Model handle.
            positive.set_text("truck");
            negative.set_text("purple elephant");
            for (int frame = 0; frame < frame_count; ++frame) {
                const auto image = sam_example::read_image(options.image / (frame_name(frame) + ".png"));
                const auto negative_before = negative.stats();
                auto foreground = positive.push_frame(frame, sam_example::image_view(image));
                unchanged(negative.stats(), negative_before);
                bounded(positive.stats(), frame, info.backend);
                check_owned();
                const auto positive_before = positive.stats();
                auto background = negative.push_frame(frame, sam_example::image_view(negative_image));
                unchanged(positive.stats(), positive_before);
                bounded(negative.stats(), frame, info.backend);
                check_owned();
                require(negative.stats().active_objects == 0 && negative.stats().retained_records == 0 &&
                        negative.stats().retained_memory_bytes == 0, "negative session retained positive objects or memory");
                const auto save = [&](const sam::VideoSession& session, const std::filesystem::path& directory) {
                    auto trace = sam_example::output_file(directory / "trace" / (frame_name(frame) + ".json"));
                    trace << session.trace_json() << '\n';
                    auto stats = sam_example::output_file(directory / "trace" / (frame_name(frame) + "-stats.json"));
                    write_stats(stats, session.stats());
                    stats << '\n';
                };
                save(positive, positive_directory);
                save(negative, negative_directory);
                for (const auto& result : foreground) write_frame(result, frame, positive_directory);
                for (const auto& result : background) {
                    require(result.objects.empty(), "negative session emitted a positive object");
                    write_frame(result, frame, negative_directory);
                }
                // Move the actual returned result; retaining a deep copy would hide aliasing.
                if (!owned_positive && !foreground.empty()) {
                    require(!foreground.front().objects.empty(), "entry sequence lost its initial object");
                    positive_hash = fingerprint(foreground.front());
                    owned_positive = std::move(foreground.front());
                }
                if (!owned_negative && !background.empty()) {
                    negative_hash = fingerprint(background.front());
                    owned_negative = std::move(background.front());
                }
                std::cout << "Interleaved " << frame + 1 << '/' << frame_count << ", positive active "
                          << positive.stats().active_objects << '\n' << std::flush;
            }
            positive_stats = positive.stats();
            negative_stats = negative.stats();
        }
        require(owned_positive.has_value() && owned_negative.has_value(), "long video produced no owned results");
        check_owned(); // Results must also survive both sessions' destruction.
        auto json = sam_example::output_file(options.output / "results.json");
        json << "{\"schema_version\":1,\"passed\":true,\"frame_count_per_session\":64,\"total_pushes\":128"
             << ",\"backend\":" << sam_example::json_string(sam_example::backend_name(info.backend))
             << ",\"precision\":" << sam_example::json_string(info.precision)
             << ",\"storage_profile\":" << sam_example::json_string(info.storage_profile)
             << ",\"threads\":" << info.threads
             << ",\"owned_results\":true,\"model_lifetime\":true,\"independent_sessions\":true"
             << ",\"positive_stats\":";
        write_stats(json, positive_stats);
        json << ",\"negative_stats\":";
        write_stats(json, negative_stats);
        json << "}\n";
        json.close();
        output.complete();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "long video session check: " << error.what() << '\n';
        return 1;
    }
}
