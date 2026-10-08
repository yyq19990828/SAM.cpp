#include <models/sam3/video/session.hpp>
#include <iostream>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Error, class Function> void rejects(Function function) {
    try { function(); } catch (const Error&) { return; }
    throw std::runtime_error("expected video boundary failure");
}

void check_masks_and_policy() {
    using namespace sam::internal::sam3;
    std::vector<float> diagonal(100, -1); diagonal[22] = diagonal[33] = 1;
    clean_mask_components(diagonal, 10, 10);
    require(diagonal[22] > 0 && diagonal[33] > 0 &&
            std::count_if(diagonal.begin(), diagonal.end(), [](float v) { return v > 0; }) == 2,
            "8-connected tiny object was incorrectly removed");
    std::vector<float> hole(110, 1); hole[55] = -1;
    clean_mask_components(hole, 10, 11);
    require(hole[55] == 0.1f, "odd-size one-pixel hole was not filled");
    require(mask_overlap({-1, -1}, {-1, -1}) == 0, "empty mask pair became a match");
    const std::vector<float> left{1, 1, -1, -1}, right{-1, -1, 1, 1};
    const std::vector<VideoDetection> detections{{0, 0.6f, right}};
    require(associate_video(detections, {}).births == std::vector<std::size_t>{0},
            "initial detection used the later admission threshold");
    require(associate_video(detections, {left}).births.empty(), "later low-confidence detection became a birth");
    const auto association = associate_video({{0, 0.9f, left}}, {left, left});
    require(association.unmatched.empty() && association.matches[0].empty() && association.recondition.empty(),
            "ambiguous high-IoU association changed ordered Meta rules");

    TemporalPolicy policy;
    policy.objects[0] = {0};
    Association missing; missing.unmatched = {0};
    for (int frame = 1; frame < 8; ++frame) require(policy.update(frame, {0}, missing).empty(), "hotstart retired too early");
    require(policy.update(8, {0}, missing) == std::set<std::uint64_t>{0}, "eighth unmatched hotstart frame did not retire");
    policy = {}; policy.objects[0] = {0};
    for (int frame = 20; frame < 65; ++frame)
        require(policy.update(frame, {0}, missing).empty(), "keep-alive incorrectly deletes old empty/unmatched tracks");
    require(policy.objects.at(0).keep_alive == -1, "keep-alive counter is not bounded");
    policy = {}; policy.objects[0] = {0}; policy.objects[1] = {1};
    Association duplicate; duplicate.matches = {{0, 1}};
    for (int frame = 2; frame < 9; ++frame) require(policy.update(frame, {0, 1}, duplicate).empty(), "duplicate retired too early");
    require(policy.update(9, {0, 1}, duplicate) == std::set<std::uint64_t>{1}, "duplicate did not preserve older ID");
    policy.retire({1}); require(policy.duplicates.empty(), "retired object left unbounded duplicate metadata");
}

void check_session_failure_boundaries() {
    using namespace sam::internal::sam3;
    auto model = std::make_shared<ModelState>();
    model->model_info.task = "text_video";
    model->tokenizer.vocabulary = {{"<start_of_text>", 49406}, {"<end_of_text>", 49407}};
    model->runtime = std::make_unique<sam::internal::GgmlRuntime>(sam::BackendOptions{sam::Backend::Cpu, 1}, true);
    model->buffer.reset(ggml_backend_alloc_buffer(model->runtime->weights_backend(), 32));
    rejects<std::invalid_argument>([&] { VideoSession invalid(model, 0, {}); });
    rejects<std::invalid_argument>([&] { VideoSession invalid(model, 2, {0}); });
    std::weak_ptr<ModelState> retained = model;
    VideoSession session(model, 2, {}); model.reset();
    require(!retained.expired(), "video session does not retain model ownership");
    std::vector<std::int32_t> tokens(32); tokens[0] = 49406; tokens[1] = 4629; tokens[2] = 49407;
    std::uint8_t rgb[] = {127, 127, 127};
    const sam::ImageView image{rgb, 3, 1, 1, 3};
    rejects<std::invalid_argument>([&] { session.push_frame(1, image); });
    rejects<std::runtime_error>([&] { session.push_frame(0, image); });
    session.set_tokens(tokens);
    rejects<std::invalid_argument>([&] { session.push_frame(0, {rgb, 2, 1, 1, 3}); });
    require(session.token_ids() == tokens && session.stats().accepted_frames == 0, "bad argument changed session state");
    // Missing graph weights deliberately simulate an execution setup failure.
    rejects<std::runtime_error>([&] { session.push_frame(0, image); });
    rejects<std::runtime_error>([&] { session.set_tokens(tokens); });
    rejects<std::invalid_argument>([&] { session.reset(0); });
    rejects<std::runtime_error>([&] { session.set_tokens(tokens); });
    session.reset(1);
    require(session.token_ids().empty() && session.stats().accepted_frames == 0 && session.stats().pending_frames == 0,
            "reset did not clear failed sequence state");
    session.set_tokens(tokens);
    session.reset(std::numeric_limits<int>::max());
    require(session.stats().retained_records == 0 && session.stats().pending_frames == 0,
            "declared frame count preallocated temporal state");
}

} // namespace

int main() {
    try { check_masks_and_policy(); check_session_failure_boundaries(); return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
