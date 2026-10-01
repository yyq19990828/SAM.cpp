#include <sam/internal/models/sam3/tensors.hpp>
#include <sam/internal/models/sam3/tracking/attention.hpp>
#include <sam/internal/models/sam3/tracking/preprocessing.hpp>
#include <sam/internal/models/sam3/tracking/memory_selection.hpp>
#include <sam/internal/models/sam3/weights.hpp>
#include <sam/internal/runtime/ggml.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

#include "data/video_preprocessing_goldens.inc"

void check_preprocessing() {
    const std::vector<std::vector<int>> shapes{{19,11,13,7}, {3,5,11,9}, {1,7,5,3}, {11,11,11,11}};
    const std::vector<std::vector<std::uint8_t>> expected{
        resize_golden_0, resize_golden_1, resize_golden_2, resize_golden_3};
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        const auto& shape = shapes[i];
        const std::size_t stride = shape[0] * 3 + 7;
        std::vector<std::uint8_t> padded((shape[1] - 1) * stride + shape[0] * 3, 255);
        for (int y = 0; y < shape[1]; ++y) for (int x = 0; x < shape[0] * 3; ++x) {
            const auto index = y * shape[0] * 3 + x;
            padded[y * stride + x] = (index * 73 + (index / 7) * 19) % 256;
        }
        const sam::ImageView image{padded.data(), padded.size(), shape[0], shape[1], stride};
        require(sam::internal::sam3::resize_video_rgb(image, shape[2], shape[3]) == expected[i],
                "Video bicubic bytes differ from Pillow goldens");
    }
    std::vector<std::uint8_t> bytes(256 * 3);
    for (int i = 0; i < 256; ++i) for (int c = 0; c < 3; ++c) bytes[i * 3 + c] = i;
    const auto normalized = sam::internal::sam3::preprocess_video_frame(
        {bytes.data(), bytes.size(), 256, 1, bytes.size()}, 256);
    for (int c = 0; c < 3; ++c) for (int y = 0; y < 256; ++y) for (int x = 0; x < 256; ++x)
        require(normalized[c * 256 * 256 + y * 256 + x] == normalization_golden[x],
                "Video F16 normalization differs for an RGB byte");
    using sam::internal::sam3::round_bf16;
    require(round_bf16(1.00390625f) == 1.0f && round_bf16(1.01171875f) == 1.015625f &&
            round_bf16(-1.00390625f) == -1.0f, "BF16 halfway values do not round to even");
    using sam::internal::sam3::round_f16;
    require(round_f16(1.00048828125f) == 1.0f && round_f16(1.00146484375f) == 1.001953125f,
            "F16 halfway values do not round to even");
}

void check_inventory() {
    for (const bool video : {false, true}) for (const bool half : {false, true}) {
        auto context = sam::internal::make_context(4096);
        sam::internal::sam3::sam3_model model;
        model.ctx = context.get();
        model.weight_type = half ? GGML_TYPE_F16 : GGML_TYPE_F32;
        {
            std::ifstream inventory(SAM_VIDEO_INVENTORY);
            std::string line;
            while (std::getline(inventory, line)) {
                if (line.empty() || line.front() == '#') continue;
                std::istringstream record(line);
                std::string name;
                int rank = 0;
                record >> name >> rank;
                model.source_types[name] = half && !sam::internal::sam3::tensor_kept_f32(name, rank) ?
                    GGML_TYPE_F16 : GGML_TYPE_F32;
            }
        }
        sam::internal::sam3::sam3_register_tensors(model, video);
        require(model.tensors.size() == (video ? 1464 : 1133), "Task tensor inventory is incomplete");
        std::ifstream source(SAM_VIDEO_INVENTORY);
        require(source.good(), "Frozen official tensor inventory cannot be opened");
        std::string line;
        std::size_t matched = 0;
        while (std::getline(source, line)) {
            if (line.empty() || line.front() == '#') continue;
            std::istringstream record(line);
            std::string name;
            int rank = 0;
            record >> name >> rank;
            const auto found = model.tensors.find(name);
            if (found == model.tensors.end()) {
                require(!video, "Video tensor was not registered");
                continue;
            }
            ++matched;
            std::vector<std::int64_t> shape(rank);
            for (auto& size : shape) record >> size;
            const auto canonical = sam::internal::sam3::canonical_dimensions(shape);
            if (!std::equal(canonical.begin(), canonical.end(), found->second->ne))
                throw std::runtime_error("Registered tensor differs from official shape: " + name);
            const auto expected_type = half && !sam::internal::sam3::tensor_kept_f32(name, rank) ?
                GGML_TYPE_F16 : GGML_TYPE_F32;
            if (found->second->type != expected_type)
                throw std::runtime_error("Tensor dtype does not preserve storage policy: " + name);

        }
        require(matched == model.tensors.size(), "Registered tensor is absent from the official schema");
    }
}

void check_attention() {
    sam::internal::GgmlRuntime runtime({sam::Backend::Cpu, 1}, true);
    // Tails straddle both the 128-query tile and the complete-key boundary.
    for (const int query_count : {1, 128, 129, 257}) {
        const int key_count = 131, channels = 256;
        sam::RuntimeStats stats;
        sam::internal::GraphExecution graph(runtime, 256, stats);
        auto* ctx = graph.context();
        auto* q = sam::internal::input_tensor(ctx, "queries", channels, query_count);
        auto* k = sam::internal::input_tensor(ctx, "keys", channels, key_count);
        auto* v = sam::internal::input_tensor(ctx, "values", channels, key_count);
        auto* output = sam::internal::sam3::tiled_memory_attention(ctx, q, k, v);
        graph.output(output);
        graph.allocate();
        std::vector<float> queries(channels * query_count), keys(channels * key_count), values(keys.size());
        for (std::size_t i = 0; i < queries.size(); ++i) queries[i] = std::sin(i * 0.37) * 0.1f;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            keys[i] = std::cos(i * 0.19) * 0.2f;
            values[i] = i / channels == key_count - 1 ? 3.0f : std::sin(i * 0.13) * 0.7f;
        }
        sam::internal::upload(q, queries, stats);
        sam::internal::upload(k, keys, stats);
        sam::internal::upload(v, values, stats);
        graph.compute();
        const auto actual = sam::internal::download(output, stats);
        for (int query = 0; query < query_count; ++query) {
            std::vector<double> scores(key_count);
            for (int key = 0; key < key_count; ++key) {
                double score = 0.0;
                for (int c = 0; c < channels; ++c) score += static_cast<double>(queries[query * channels + c]) * keys[key * channels + c];
                scores[key] = score / 16.0;
            }
            const auto maximum = *std::max_element(scores.begin(), scores.end());
            double sum = 0.0;
            for (auto& score : scores) { score = std::exp(score - maximum); sum += score; }
            for (int c = 0; c < channels; ++c) {
                double expected = 0.0;
                for (int key = 0; key < key_count; ++key) expected += scores[key] * values[key * channels + c] / sum;
                require(std::abs(actual[query * channels + c] - expected) < 2e-6,
                        "Tiled attention changed full-key softmax or output layout");
            }
        }
        require(stats.cpu_nodes > 0, "Attention test did not execute a backend graph");
    }
}

void check_memory_selection() {
    using namespace sam::internal::sam3;
    std::ifstream goldens(SAM_MEMORY_GOLDENS);
    require(goldens.good(), "Cannot open the pinned official selector goldens");
    std::string line;
    while (goldens.peek() == '#') std::getline(goldens, line);
    for (int scenario = 0; scenario < 3; ++scenario) {
        const int birth = scenario == 2 ? 37 : 0;
        const int count = scenario == 1 ? 64 : 1000;
        std::map<int, MemoryFrame> complete, bounded;
        int discarded_conditioning = 0;
        for (int frame = birth; frame < count; ++frame) {
            for (auto* history : {&complete, &bounded}) {
                if (frame - birth == 11) for (auto& item : *history)
                    if (item.first >= frame - 7 && item.first < frame) item.second.conditioning = false;
                if (scenario == 2 && frame - birth == 12) for (auto& item : *history)
                    if (!item.second.conditioning) item.second.group_quality = 0.0f;
            }
            if (frame > birth) {
                int expected_scenario = -1, expected_frame = -1, size = -1;
                goldens >> expected_scenario >> expected_frame >> size;
                require(expected_scenario == scenario && expected_frame == frame && size >= 0,
                        "Selector fixture frame order is invalid");
                MemorySelection expected;
                expected.spatial.resize(size);
                for (auto& item : expected.spatial) goldens >> item.frame >> item.position;
                goldens >> size;
                require(size >= 0, "Invalid pointer fixture count");
                expected.pointers.resize(size);
                for (auto& item : expected.pointers) goldens >> item.frame >> item.position;
                require(goldens.good(), "Selector fixture is truncated");
                const auto same = [](const std::vector<SelectedMemory>& left, const std::vector<SelectedMemory>& right) {
                    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(),
                        [](const SelectedMemory& a, const SelectedMemory& b) {
                            return a.frame == b.frame && a.position == b.position;
                        });
                };
                for (const auto& actual : {select_memory(complete, frame, count),
                                          select_memory(bounded, frame, count, discarded_conditioning)}) {
                    if (!same(actual.spatial, expected.spatial) || !same(actual.pointers, expected.pointers))
                        throw std::runtime_error("Memory indices or pointer order differ from Meta at scenario " +
                            std::to_string(scenario) + " frame " + std::to_string(frame));
                }
            }
            const MemoryFrame record{frame, (frame - birth) % 16 == 0 ||
                (frame - birth < 15 && (frame - birth) % 4 == 0),
                scenario == 1 && frame > 1 ? 0.0f : (frame * 37 % 31) / 100.0f};
            complete.emplace(frame, record);
            bounded.emplace(frame, record);
            const auto keep = retained_memory_frames(bounded, frame, birth);
            require(keep.size() <= 27, "Retained memory exceeded the declared record bound");
            for (auto item = bounded.begin(); item != bounded.end();) {
                if (keep.count(item->first)) { ++item; continue; }
                if (item->second.conditioning) ++discarded_conditioning;
                item = bounded.erase(item);
            }
        }
    }
    std::string trailing;
    require(!(goldens >> trailing), "Selector fixture contains unconsumed cases");
}

} // namespace

int main() {
    try { check_preprocessing(); check_inventory(); check_attention(); check_memory_selection(); return 0; }
    catch (const std::exception& error) { std::cerr << "tracking math: " << error.what() << '\n'; return 1; }
}
