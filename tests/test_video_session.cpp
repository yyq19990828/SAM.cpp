#include "image_support.hpp"
#include <iostream>
#include <optional>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Error, class Function> void rejects(Function function) {
    try { function(); } catch (const Error&) { return; }
    throw std::runtime_error("expected video boundary rejection");
}

void same_frames(const std::vector<sam::VideoFrameResult>& actual, const std::vector<sam::VideoFrameResult>& expected) {
    require(actual.size() == expected.size(), "video reset changed emitted frame count");
    for (std::size_t frame = 0; frame < actual.size(); ++frame) {
        const auto& a = actual[frame]; const auto& b = expected[frame];
        require(a.frame_index == b.frame_index && a.objects.size() == b.objects.size(), "video reset changed IDs/counts");
        for (std::size_t i = 0; i < a.objects.size(); ++i) {
            const auto& x = a.objects[i]; const auto& y = b.objects[i];
            require(x.id == y.id && x.score == y.score && x.box.x0 == y.box.x0 && x.box.y0 == y.box.y0 &&
                    x.box.x1 == y.box.x1 && x.box.y1 == y.box.y1 && x.mask.data == y.mask.data,
                    "video results changed after reset or another session");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: test_video_session VIDEO_MODEL POSITIVE_IMAGE NEGATIVE_IMAGE cpu|metal THREADS NEW_DIR\n";
            return 0;
        }
        if (argc != 7) throw std::invalid_argument("Use --help for the six positional arguments");
        std::vector<std::string> arguments{"test_video_session", "--model", argv[1], "--image", argv[2],
            "--text", "truck", "--backend", argv[4], "--threads", argv[5], "--output", argv[6]};
        std::vector<char*> pointers;
        for (auto& argument : arguments) pointers.push_back(argument.data());
        const auto options = sam_example::parse_options(pointers.size(), pointers.data(), false);
        require(options.backend.backend != sam::Backend::Auto, "video check requires an explicit backend");
        require(!std::filesystem::exists(options.output), "output already exists");
        const auto positive = sam_example::read_image(options.image), negative = sam_example::read_image(argv[3]);
        std::optional<sam::Model> model(sam::Model::load(options.model.string(), options.backend));
        const auto info = model->info();
        sam::VideoSession first(*model, 2), second(*model, 1);
        model.reset();
        first.set_text("truck"); second.set_text("purple elephant");
        require(first.push_frame(0, sam_example::image_view(positive)).empty(), "short video emitted before final draining");
        const auto& pointer = first.tensor("object.0.pointer");
        require(pointer.shape == std::vector<std::int64_t>{1, 256} && pointer.values.size() == 256,
                "canonical video pointer tensor is unavailable");
        for (const char* name : {"object.0junk.pointer", "object.00.pointer", "object.+0.pointer",
                                 "object. 0.pointer", "object.-0.pointer"})
            rejects<std::invalid_argument>([&] { (void) first.tensor(name); });
        require(&first.tensor("object.0.pointer") == &pointer, "canonical tensor did not reuse its snapshot");
        const auto view = sam_example::image_view(positive);
        rejects<std::invalid_argument>([&] { first.push_frame(0, view); });
        rejects<std::invalid_argument>([&] { first.push_frame(-1, view); });
        rejects<std::invalid_argument>([&] { first.push_frame(2, view); });
        auto changed_resolution = view; --changed_resolution.height;
        rejects<std::invalid_argument>([&] { first.push_frame(1, changed_resolution); });
        rejects<std::runtime_error>([&] { first.set_text("wheel"); });
        require(first.stats().accepted_frames == 1 && first.stats().runtime.vision_encodes == 1,
                "rejected video input changed counters or executed a graph");
        const auto unrelated = second.push_frame(0, sam_example::image_view(negative));
        require(unrelated.size() == 1 && unrelated[0].objects.empty(), "independent negative video retained objects");
        const auto expected = first.push_frame(1, sam_example::image_view(positive));
        require(expected.size() == 2 && !expected[0].objects.empty() &&
                expected[0].objects[0].id == expected[1].objects[0].id, "video propagation lost its original object");
        const auto saved = expected;
        rejects<std::invalid_argument>([&] { first.push_frame(2, view); });
        require(first.stats().runtime.vision_encodes == 2 && first.stats().runtime.text_encodes == 1 &&
                first.stats().runtime.inferences == 2 && first.stats().pending_frames == 0,
                "video recomputed text/vision or failed final draining");
        first.reset(2);
        require(first.token_ids().empty() && first.stats().accepted_frames == 0 &&
                first.stats().retained_records == 0 && first.stats().active_objects == 0,
                "video reset retained prompt, counters or temporal state");
        first.set_text("truck");
        require(first.push_frame(0, sam_example::image_view(positive)).empty(), "reset changed output delay");
        same_frames(first.push_frame(1, sam_example::image_view(positive)), expected);
        same_frames(expected, saved);
        second.reset(1); second.set_text("purple elephant");
        same_frames(second.push_frame(0, sam_example::image_view(negative)), unrelated);
        for (const auto* session : {&first, &second}) {
            const auto& stats = session->stats();
            require(stats.retained_records <= 27 * stats.active_objects && stats.pending_frames == 0,
                    "video reset exceeded state bounds");
            if (info.backend == sam::Backend::Metal)
                require(stats.runtime.metal_nodes > 0 && stats.runtime.cpu_nodes == 0, "video used CPU graph fallback");
        }
        sam_example::OutputDirectory output(options.output);
        auto json = sam_example::output_file(options.output / "results.json");
        json << "{\"passed\":true,\"backend\":" << sam_example::json_string(sam_example::backend_name(info.backend))
             << ",\"precision\":" << sam_example::json_string(info.precision) << ",\"threads\":" << info.threads
             << ",\"owned_results\":true,\"model_lifetime\":true,\"independent_sessions\":true,\"reset\":true}\n";
        json.close(); output.complete();
        std::cout << "Video session behavior checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "video session check: " << error.what() << '\n'; return 1;
    }
}
