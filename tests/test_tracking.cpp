#include "backend_test_support.hpp"
#include <models/sam3/tensors.hpp>
#include <models/sam3/video/attention.hpp>
#include <models/sam3/video/memory_attention.hpp>
#include <models/sam3/video/mask_decoder.hpp>
#include <models/sam3/video/execution.hpp>
#include <models/sam3/video/preprocessing.hpp>
#include <models/sam3/video/memory_selection.hpp>
#include <models/sam3/weights.hpp>
#include <runtime/ggml.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_backend_placement(sam::Backend backend, const sam::RuntimeStats& stats) {
    if (backend == sam::Backend::Cuda)
        require(stats.cuda_nodes > 0 && stats.cpu_nodes == 0 && stats.metal_nodes == 0 && stats.blas_nodes == 0,
                "Tracking subgraph used compute outside CUDA");
    else require(stats.cpu_nodes > 0 && stats.cuda_nodes == 0 && stats.metal_nodes == 0,
                 "Tracking subgraph did not execute on CPU");
}

void check_mutable_tracker_chunk_limit() {
    int limit = 2;
    std::vector<int> order;
    std::vector<int> counts;
    sam::internal::sam3::sam3_for_each_tracker_chunk(9, limit, [&](int begin, int count) {
        counts.push_back(count);
        for (int index = begin; index < begin + count; ++index) order.push_back(index);
        if (begin == 0) limit = sam::internal::sam3::sam3_tracker_graph_batch_limit;
    });
    require(order == std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8} &&
            counts == std::vector<int>{2, 7},
            "tracker chunking skipped objects when recursive fallback changed the next limit");
}

void check_none_serial_policy_key() {
    int runtime_a = 0, runtime_b = 0;
    const sam::internal::sam3::TrackerNoneSerialPolicy policy{
        &runtime_a, sam::Backend::Cpu, "f32:video-v1", "f32-accumulation-v1", 6, 16, 207257600};
    const auto matches = [&](const void* runtime, sam::Backend backend, const std::string& storage,
                             const std::string& arithmetic, int spatial, int pointers, std::size_t budget) {
        return sam::internal::sam3::tracker_none_serial_policy_matches(
            std::optional<sam::internal::sam3::TrackerNoneSerialPolicy>{policy},
            runtime, backend, storage, arithmetic, spatial, pointers, budget);
    };
    require(matches(&runtime_a, sam::Backend::Cpu, "f32:video-v1", "f32-accumulation-v1", 6, 16, 207257600),
            "matching none-serial propagation shape did not retain its decision");
    require(!matches(&runtime_b, sam::Backend::Cpu, "f32:video-v1", "f32-accumulation-v1", 6, 16, 207257600) &&
            !matches(&runtime_a, sam::Backend::Metal, "f32:video-v1", "f32-accumulation-v1", 6, 16, 207257600) &&
            !matches(&runtime_a, sam::Backend::Cpu, "f16:video-v1", "f32-accumulation-v1", 6, 16, 207257600) &&
            !matches(&runtime_a, sam::Backend::Cpu, "f32:video-v1", "f16-accumulation-v1", 6, 16, 207257600) &&
            !matches(&runtime_a, sam::Backend::Cpu, "f32:video-v1", "f32-accumulation-v1", 5, 16, 207257600) &&
            !matches(&runtime_a, sam::Backend::Cpu, "f32:video-v1", "f32-accumulation-v1", 6, 15, 207257600) &&
            !matches(&runtime_a, sam::Backend::Cpu, "f32:video-v1", "f32-accumulation-v1", 6, 16, 207257601),
            "none-serial decision survived a runtime, profile, shape, or budget change");
}

void check_deferred_resident_upload_lifecycle() {
    sam::internal::sam3::TrackerResidentUploadState uploads;
    int core_uploads = 0, high_uploads = 0;
    const auto upload_core = [&] { ++core_uploads; };
    const auto upload_high = [&] { ++high_uploads; };

    uploads.ensure_core(upload_core);
    uploads.ensure_core(upload_core);
    uploads.ensure_high(upload_high);
    uploads.ensure_high(upload_high);
    require(core_uploads == 1 && high_uploads == 1,
            "resident inputs were uploaded more than once in one frame");

    uploads.release_high();
    uploads.ensure_core(upload_core);
    uploads.ensure_high(upload_high);
    require(core_uploads == 1 && high_uploads == 2 && uploads.core_uploaded(),
            "high-residency demotion discarded still-live core input data");

    uploads.reset_frame();
    uploads.ensure_core(upload_core);
    uploads.ensure_high(upload_high);
    require(core_uploads == 2 && high_uploads == 3,
            "resident inputs were not marked fresh at the next frame boundary");

    uploads.reset_frame();
    bool threw = false;
    try {
        uploads.ensure_core([] { throw std::runtime_error("upload failed"); });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw && !uploads.core_uploaded(),
            "failed resident upload was marked complete instead of remaining retryable");
    uploads.ensure_core(upload_core);
    require(core_uploads == 3, "resident upload could not recover after a failed attempt");
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

void check_mask_resize() {
    // Independent torch 2.10.0 CUDA interpolation results for a synthetic
    // signed mask. Rounding here can affect a later binary initialization.
    constexpr int width = 129, height = 83, target = 71;
    std::vector<float> input(width * height);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = ((i * 17 + 3) % 8) < 3 ? 1024.0f : -1024.0f;
    const auto bilinear = sam::internal::sam3::resize_mask(input, width, height, target, target);
    const auto antialiased = sam::internal::sam3::resize_mask(input, width, height, target, target, true);
    require(std::abs(bilinear[69 * target + 68] - 440.2027587890625f) <= 0.0001f,
            "mask half-pixel coordinates differ from the independent CUDA fixture");
    require(antialiased[35 * target + 23] == -944.8809204101562f &&
            antialiased[69 * target + 68] == 171.0402374267578f,
            "mask antialias accumulation lost its required float contraction");
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

void check_attention(sam::Backend backend, sam::CudaComputeMode compute = sam::CudaComputeMode::F32) {
    sam::internal::GgmlRuntime runtime({backend, 1, 0, compute}, true);
    // Tails straddle both backend query tiles and the complete-key boundary.
    for (const int query_count : {1, 128, 129, 257, 513}) {
        const int key_count = 131, channels = 256;
        const std::array<std::pair<int, int>, 6> batches{{{1, 1}, {2, 2}, {4, 4}, {8, 8}, {1, 8}, {8, 1}}};
        const auto case_count = query_count == 129 ? batches.size() : std::size_t(3);
        for (std::size_t batch_case = 0; batch_case < case_count; ++batch_case) {
            const auto [query_batch, memory_batch] = batches[batch_case];
            const int batch = std::max(query_batch, memory_batch);
            sam::RuntimeStats stats;
            sam::internal::GraphExecution graph(runtime, 256, stats);
            auto* ctx = graph.context();
            auto* q = sam::internal::input_tensor(ctx, "queries", channels, query_count, query_batch);
            auto* k = sam::internal::input_tensor(ctx, "keys", channels, key_count, memory_batch);
            auto* v = sam::internal::input_tensor(ctx, "values", channels, key_count, memory_batch);
            auto* output = sam::internal::sam3::tiled_memory_attention(ctx, q, k, v, runtime.attention_policy());
            graph.output(output);
            graph.allocate();
            std::vector<float> queries(channels * query_count * query_batch);
            std::vector<float> keys(channels * key_count * memory_batch), values(keys.size());
            for (int b = 0; b < query_batch; ++b) for (int query = 0; query < query_count; ++query)
                for (int c = 0; c < channels; ++c) {
                    const auto index = (b * query_count + query) * channels + c;
                    queries[index] = std::sin(index * 0.37 + b * 0.29) * 0.1f;
                }
            for (int b = 0; b < memory_batch; ++b) for (int key = 0; key < key_count; ++key)
                for (int c = 0; c < channels; ++c) {
                    const auto index = (b * key_count + key) * channels + c;
                    keys[index] = std::cos(index * 0.19 + b * 0.23) * 0.2f;
                    values[index] = key == key_count - 1 ? 3.0f + b : std::sin(index * 0.13) * 0.7f;
                }
            sam::internal::upload(q, queries, stats);
            sam::internal::upload(k, keys, stats);
            sam::internal::upload(v, values, stats);
            graph.compute();
            check_backend_placement(backend, stats);
            const auto actual = sam::internal::download(output, stats);
            for (int b = 0; b < batch; ++b) for (int query = 0; query < query_count; ++query) {
                const int qb = query_batch == 1 ? 0 : b;
                const int mb = memory_batch == 1 ? 0 : b;
                std::vector<double> scores(key_count);
                for (int key = 0; key < key_count; ++key) {
                    double score = 0.0;
                    for (int c = 0; c < channels; ++c) {
                        const auto qi = (qb * query_count + query) * channels + c;
                        const auto ki = (mb * key_count + key) * channels + c;
                        score += static_cast<double>(queries[qi]) * keys[ki];
                    }
                    scores[key] = score / 16.0;
                }
                const auto maximum = *std::max_element(scores.begin(), scores.end());
                double sum = 0.0;
                for (auto& score : scores) { score = std::exp(score - maximum); sum += score; }
                for (int c = 0; c < channels; ++c) {
                    double expected = 0.0;
                    for (int key = 0; key < key_count; ++key) {
                        const auto vi = (mb * key_count + key) * channels + c;
                        expected += scores[key] * values[vi] / sum;
                    }
                    const auto oi = (b * query_count + query) * channels + c;
                    const double tolerance = compute == sam::CudaComputeMode::F16 ? 2e-3 : 2e-6;
                    require(std::isfinite(actual[oi]) && std::abs(actual[oi] - expected) < tolerance,
                            "Batched tiled attention mixed objects or changed full-key softmax");
                }
            }
            require(output->ne[0] == channels && output->ne[1] == query_count && output->ne[2] == batch,
                    "Batched tiled attention returned a transposed object layout");
        }
    }
}

std::vector<float> run_memory_attention(const std::vector<float>& prompt_values,
                                        const std::vector<float>& prompt_positions,
                                        int batch, sam::Backend backend) {
    constexpr int channels = 256, memory_channels = 64, ffn_channels = 1024;
    constexpr int current_tokens = 3, memory_tokens = 3, spatial_memory_tokens = 2;
    sam::internal::GgmlRuntime runtime({backend, 1}, true);
    sam::RuntimeStats stats;
    sam::internal::GraphExecution graph(runtime, 1024, stats);
    auto* ctx = graph.context();
    auto* identity = sam::internal::input_tensor(ctx, "memory_identity", channels, channels);
    auto* zero = sam::internal::input_tensor(ctx, "memory_zero", channels);
    auto* norm = sam::internal::input_tensor(ctx, "memory_norm", channels);
    auto* kv_projection = sam::internal::input_tensor(ctx, "memory_kv_projection", memory_channels, channels);
    auto* ffn_first = sam::internal::input_tensor(ctx, "memory_ffn_first", channels, ffn_channels);
    auto* ffn_second = sam::internal::input_tensor(ctx, "memory_ffn_second", ffn_channels, channels);

    sam::internal::sam3::sam3_model model;
    model.hparams.neck_dim = channels;
    model.hparams.mem_out_dim = memory_channels;
    sam::internal::sam3::sam3_mem_attn_layer layer{};
    layer.sa_q_w = layer.sa_k_w = layer.sa_v_w = layer.sa_out_w = identity;
    layer.sa_q_b = layer.sa_k_b = layer.sa_v_b = layer.sa_out_b = zero;
    layer.norm1_w = layer.norm2_w = layer.norm3_w = norm;
    layer.norm1_b = layer.norm2_b = layer.norm3_b = zero;
    layer.ca_q_w = layer.ca_out_w = identity;
    layer.ca_q_b = layer.ca_k_b = layer.ca_v_b = layer.ca_out_b = zero;
    layer.ca_k_w = layer.ca_v_w = kv_projection;
    layer.ffn_fc1_w = ffn_first;
    layer.ffn_fc1_b = sam::internal::input_tensor(ctx, "memory_ffn_first_bias", ffn_channels);
    layer.ffn_fc2_w = ffn_second;
    layer.ffn_fc2_b = zero;
    model.mem_attn.layers.push_back(layer);
    model.tensors.emplace("mem_attn.norm.weight", norm);
    model.tensors.emplace("mem_attn.norm.bias", zero);

    auto* current = sam::internal::input_tensor(ctx, "memory_current", channels, current_tokens);
    auto* current_position = sam::internal::input_tensor(ctx, "memory_current_position", channels, current_tokens);
    auto* prompt = sam::internal::input_tensor(ctx, "memory_prompt", memory_channels, memory_tokens, batch);
    auto* prompt_position = sam::internal::input_tensor(ctx, "memory_prompt_position", memory_channels, memory_tokens, batch);
    auto* frequencies = sam::internal::input_tensor(ctx, "memory_rope", 2, channels / 2, current_tokens);
    auto* key_frequencies = sam::internal::input_tensor(ctx, "memory_key_rope", 2, channels / 2, spatial_memory_tokens);
    auto* output = sam::internal::sam3::sam3_build_mem_attn_graph(
        ctx, model, current, current_position, prompt, prompt_position,
        frequencies, key_frequencies, 1, runtime.attention_policy());
    graph.output(output);
    graph.allocate();

    std::vector<float> eye(channels * channels), norm_values(channels, 1.0f);
    std::vector<float> projection(memory_channels * channels);
    for (int i = 0; i < channels; ++i) eye[i * channels + i] = 1.0f;
    for (int i = 0; i < memory_channels; ++i) projection[i * memory_channels + i] = 0.5f;
    std::vector<float> zeros(channels), zeros_first(channels * ffn_channels), zeros_second(ffn_channels * channels);
    std::vector<float> current_values(channels * current_tokens), current_positions(current_values.size());
    for (std::size_t i = 0; i < current_values.size(); ++i) {
        current_values[i] = std::sin(i * 0.17) * 0.2f;
        current_positions[i] = std::cos(i * 0.11) * 0.1f;
    }
    std::vector<float> rope(2 * (channels / 2) * current_tokens);
    std::vector<float> key_rope(2 * (channels / 2) * spatial_memory_tokens);
    for (int token = 0; token < current_tokens; ++token) for (int k = 0; k < channels / 2; ++k) {
        rope[token * channels + 2 * k] = 1.0f;
        rope[token * channels + 2 * k + 1] = 0.0f;
    }
    for (int token = 0; token < spatial_memory_tokens; ++token) for (int k = 0; k < channels / 2; ++k) {
        key_rope[token * channels + 2 * k] = 1.0f;
        key_rope[token * channels + 2 * k + 1] = 0.0f;
    }
    sam::internal::upload(identity, eye, stats);
    sam::internal::upload(zero, zeros, stats);
    sam::internal::upload(norm, norm_values, stats);
    sam::internal::upload(kv_projection, projection, stats);
    sam::internal::upload(ffn_first, zeros_first, stats);
    sam::internal::upload(layer.ffn_fc1_b, std::vector<float>(ffn_channels), stats);
    sam::internal::upload(ffn_second, zeros_second, stats);
    sam::internal::upload(current, current_values, stats);
    sam::internal::upload(current_position, current_positions, stats);
    sam::internal::upload(prompt, prompt_values, stats);
    sam::internal::upload(prompt_position, prompt_positions, stats);
    sam::internal::upload(frequencies, rope, stats);
    sam::internal::upload(key_frequencies, key_rope, stats);
    graph.compute();
    check_backend_placement(backend, stats);
    require(output->ne[0] == channels && output->ne[1] == current_tokens && output->ne[2] == batch,
            "Memory attention lost its object batch axis");
    return sam::internal::download(output, stats);
}

void check_memory_attention_batch(sam::Backend backend) {
    constexpr int memory_channels = 64, memory_tokens = 3, batch = 4;
    std::vector<float> prompt(memory_channels * memory_tokens * batch);
    std::vector<float> position(prompt.size());
    for (int object = 0; object < batch; ++object) for (int token = 0; token < memory_tokens; ++token)
        for (int c = 0; c < memory_channels; ++c) {
            const auto i = (object * memory_tokens + token) * memory_channels + c;
            prompt[i] = std::sin(i * 0.071 + object * 0.31f) * 0.4f + (token == 2 ? object * 0.25f : 0.0f);
            position[i] = std::cos(i * 0.037 + object * 0.23f) * 0.05f;
        }
    const auto batched = run_memory_attention(prompt, position, batch, backend);
    constexpr std::size_t values_per_object = 256 * 3;
    for (int object = 0; object < batch; ++object) {
        const auto begin = static_cast<std::size_t>(object) * memory_channels * memory_tokens;
        const std::vector<float> prompt_slice(prompt.begin() + begin, prompt.begin() + begin + memory_channels * memory_tokens);
        const std::vector<float> position_slice(position.begin() + begin, position.begin() + begin + memory_channels * memory_tokens);
        const auto serial = run_memory_attention(prompt_slice, position_slice, 1, backend);
        for (std::size_t i = 0; i < values_per_object; ++i)
            require(std::abs(batched[static_cast<std::size_t>(object) * values_per_object + i] - serial[i]) < 2e-5f,
                    "Memory attention batch differs from independent per-object execution");
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

void check_sam_cross_attention(sam::Backend backend) {
    sam::internal::GgmlRuntime runtime({backend, 1}, true);
    sam::RuntimeStats stats;
    sam::internal::GraphExecution graph(runtime, 256, stats);
    auto* ctx = graph.context();
    constexpr int channels = 32, queries = 3, keys = 11, heads = 2, head_size = 16;
    auto* q = sam::internal::input_tensor(ctx, "sam_q", channels, queries);
    auto* k = sam::internal::input_tensor(ctx, "sam_k", channels, keys);
    auto* v = sam::internal::input_tensor(ctx, "sam_v", channels, keys);
    auto* identity = sam::internal::input_tensor(ctx, "identity", channels, channels);
    auto* zero = sam::internal::input_tensor(ctx, "zero", channels);
    sam::internal::sam3::sam3_sam_attn weights{};
    weights.q_w = weights.k_w = weights.v_w = weights.out_w = identity;
    weights.q_b = weights.k_b = weights.v_b = weights.out_b = zero;
    auto* out = sam::internal::sam3::sam3_sam_attention(ctx, q, k, v, weights, heads);
    graph.output(out); graph.allocate();
    std::vector<float> matrix(channels * channels), q_values(channels * queries), k_values(channels * keys), v_values(channels * keys);
    for (int c = 0; c < channels; ++c) matrix[c * channels + c] = 1;
    for (std::size_t i = 0; i < q_values.size(); ++i) q_values[i] = std::sin(i * 0.7) * 0.2f;
    for (std::size_t i = 0; i < k_values.size(); ++i) {
        k_values[i] = std::cos(i * 0.4) * 0.3f; v_values[i] = std::sin(i * 0.2) + i / channels;
    }
    sam::internal::upload(identity, matrix, stats); sam::internal::upload(zero, std::vector<float>(channels), stats);
    sam::internal::upload(q, q_values, stats); sam::internal::upload(k, k_values, stats); sam::internal::upload(v, v_values, stats);
    graph.compute();
    check_backend_placement(backend, stats);
    const auto actual = sam::internal::download(out, stats);
    for (int query = 0; query < queries; ++query) for (int head = 0; head < heads; ++head) {
        std::vector<double> scores(keys);
        for (int key = 0; key < keys; ++key) for (int c = 0; c < head_size; ++c)
            scores[key] += static_cast<double>(q_values[query * channels + head * head_size + c]) *
                k_values[key * channels + head * head_size + c] / 4.0;
        double sum = 0;
        for (auto& score : scores) { score = std::exp(score); sum += score; }
        for (int c = 0; c < head_size; ++c) {
            double expected = 0;
            for (int key = 0; key < keys; ++key)
                expected += scores[key] / sum * v_values[key * channels + head * head_size + c];
            require(std::abs(actual[query * channels + head * head_size + c] - expected) < 2e-6,
                    "SAM head-16 attention changed softmax/head layout");
        }
    }
}

void check_sam_cross_attention_batch(sam::Backend backend) {
    sam::internal::GgmlRuntime runtime({backend, 1}, true);
    constexpr int channels = 32, queries = 3, keys = 11, heads = 2, batch = 8;
    std::vector<float> matrix(channels * channels), q_values(channels * queries);
    std::vector<float> k_values(channels * keys * batch), v_values(k_values.size());
    for (int c = 0; c < channels; ++c) matrix[c * channels + c] = 1;
    for (std::size_t i = 0; i < q_values.size(); ++i) q_values[i] = std::sin(i * 0.7) * 0.2f;
    for (int b = 0; b < batch; ++b) for (int key = 0; key < keys; ++key)
        for (int c = 0; c < channels; ++c) {
            const auto i = (b * keys + key) * channels + c;
            k_values[i] = std::cos(i * 0.4 + b * 0.13) * 0.3f;
            v_values[i] = std::sin(i * 0.2) + b * 0.7f + key / 3.0f;
        }

    constexpr int changed_batch = 5;
    auto changed_values = v_values;
    for (int key = 0; key < keys; ++key) for (int c = 0; c < channels; ++c)
        changed_values[(changed_batch * keys + key) * channels + c] += 0.75f;
    const auto run_with_update = [&](const std::vector<float>& initial, const std::vector<float>& updated) {
        sam::RuntimeStats stats;
        sam::internal::GraphExecution graph(runtime, 512, stats);
        auto* ctx = graph.context();
        auto* q = sam::internal::input_tensor(ctx, "batched_sam_q", channels, queries);
        auto* k = sam::internal::input_tensor(ctx, "batched_sam_k", channels, keys, batch);
        auto* v = sam::internal::input_tensor(ctx, "batched_sam_v", channels, keys, batch);
        auto* identity = sam::internal::input_tensor(ctx, "batched_identity", channels, channels);
        auto* zero = sam::internal::input_tensor(ctx, "batched_zero", channels);
        sam::internal::sam3::sam3_sam_attn weights{};
        weights.q_w = weights.k_w = weights.v_w = weights.out_w = identity;
        weights.q_b = weights.k_b = weights.v_b = weights.out_b = zero;
        auto* out = sam::internal::sam3::sam3_sam_attention(ctx, q, k, v, weights, heads);
        graph.output(out); graph.allocate();
        sam::internal::upload(identity, matrix, stats);
        sam::internal::upload(zero, std::vector<float>(channels), stats);
        sam::internal::upload(q, q_values, stats);
        sam::internal::upload(k, k_values, stats);
        sam::internal::upload(v, initial, stats);
        graph.compute();
        check_backend_placement(backend, stats);
        require(out->ne[0] == channels && out->ne[1] == queries && out->ne[2] == batch,
                "SAM attention did not retain object-major batch output");
        auto before = sam::internal::download(out, stats);
        // A cached graph receives fresh inputs for every call. Change only one V
        // slice; unchanged Q/K slices must still produce isolated object outputs.
        sam::internal::upload(q, q_values, stats);
        sam::internal::upload(k, k_values, stats);
        sam::internal::upload(v, updated, stats);
        graph.compute();
        check_backend_placement(backend, stats);
        return std::pair{std::move(before), sam::internal::download(out, stats)};
    };

    auto [actual, changed] = run_with_update(v_values, changed_values);
    for (int b = 0; b < batch; ++b) for (int query = 0; query < queries; ++query)
        for (int head = 0; head < heads; ++head) {
            std::vector<double> scores(keys);
            for (int key = 0; key < keys; ++key) for (int c = 0; c < channels / heads; ++c)
                scores[key] += static_cast<double>(q_values[query * channels + head * (channels / heads) + c]) *
                    k_values[(b * keys + key) * channels + head * (channels / heads) + c] / 4.0;
            const auto maximum = *std::max_element(scores.begin(), scores.end());
            double sum = 0;
            for (auto& score : scores) { score = std::exp(score - maximum); sum += score; }
            for (int c = 0; c < channels / heads; ++c) {
                double expected = 0;
                for (int key = 0; key < keys; ++key)
                    expected += scores[key] / sum * v_values[(b * keys + key) * channels + head * (channels / heads) + c];
                const auto i = (b * queries + query) * channels + head * (channels / heads) + c;
                require(std::abs(actual[i] - expected) < 2e-6,
                        "Batched SAM attention mixed objects or changed head layout");
            }
        }

    bool target_changed = false;
    float max_other_delta = 0.0f;
    int affected_batch = -1;
    for (int b = 0; b < batch; ++b) for (int query = 0; query < queries; ++query)
        for (int c = 0; c < channels; ++c) {
            const auto i = (b * queries + query) * channels + c;
            if (b == changed_batch) target_changed = target_changed || std::abs(changed[i] - actual[i]) > 1e-5f;
            else if (const auto delta = std::abs(changed[i] - actual[i]); delta > max_other_delta) {
                max_other_delta = delta;
                affected_batch = b;
            }
        }
    require(max_other_delta < 1e-7f,
            (std::string("Changing one SAM object altered another output slice by ") +
             std::to_string(max_other_delta) + " at object " + std::to_string(affected_batch)).c_str());
    require(target_changed, "Changing one SAM object did not alter its own output slice");
}

std::vector<float> run_deconv_batch(const std::vector<float>& inputs, sam::Backend backend) {
    sam::internal::GgmlRuntime runtime({backend, 1}, true);
    sam::RuntimeStats stats;
    sam::internal::GraphExecution graph(runtime, 128, stats);
    auto* ctx = graph.context();
    auto* weights = sam::internal::input_tensor(ctx, "deconv_weights", 2, 2, 1, 1);
    auto* input = sam::internal::input_tensor(ctx, "deconv_input", 1, 1, 1, inputs.size());
    auto* output = sam::internal::sam3::sam3_deconv_2x2(ctx, weights, input);
    graph.output(output); graph.allocate();
    const std::vector<float> kernel{0.25f, -0.5f, 1.25f, 2.0f};
    sam::internal::upload(weights, kernel, stats);
    sam::internal::upload(input, inputs, stats);
    graph.compute();
    check_backend_placement(backend, stats);
    return sam::internal::download(output, stats);
}

void check_deconv_batch(sam::Backend backend) {
    const std::vector<float> inputs{1.0f, -0.5f, 2.0f, 3.0f};
    const auto batched = run_deconv_batch(inputs, backend);
    require(batched.size() == inputs.size() * 4, "Batched deconvolution returned the wrong spatial layout");
    for (std::size_t batch = 0; batch < inputs.size(); ++batch) {
        const auto single = run_deconv_batch({inputs[batch]}, backend);
        require(single.size() == 4, "Single-object deconvolution changed its output layout");
        for (std::size_t pixel = 0; pixel < single.size(); ++pixel)
            require(std::abs(batched[batch * single.size() + pixel] - single[pixel]) < 1e-7f,
                    "Batched deconvolution mixed or reordered object outputs");
    }
}

void check_mask_decoder_batch(sam::Backend backend) {
    constexpr int D = 256, H = 2, batch = 4, feature_channels = 256;
    constexpr int high = H * 4, middle = H * 2;
    sam::internal::GgmlRuntime runtime({backend, 1}, true);
    sam::RuntimeStats stats;
    sam::internal::GraphExecution graph(runtime, 16384, stats);
    auto* ctx = graph.context();
    std::vector<std::pair<ggml_tensor*, std::vector<float>>> uploads;
    auto add_input = [&](ggml_tensor* tensor, std::vector<float> values) {
        uploads.emplace_back(tensor, std::move(values));
        return tensor;
    };
    auto vector_input = [&](const char* name, int size, std::vector<float> values) {
        return add_input(sam::internal::input_tensor(ctx, name, size), std::move(values));
    };
    auto matrix_input = [&](const char* name, int width, int rows, std::vector<float> values) {
        return add_input(sam::internal::input_tensor(ctx, name, width, rows), std::move(values));
    };
    auto tensor4_input = [&](const char* name, int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                             std::vector<float> values) {
        return add_input(sam::internal::input_tensor(ctx, name, d0, d1, d2, d3), std::move(values));
    };
    auto values = [](std::size_t count, float scale, float phase) {
        std::vector<float> result(count);
        for (std::size_t i = 0; i < count; ++i) result[i] = std::sin(i * 0.071 + phase) * scale;
        return result;
    };
    auto identity = [](int size) {
        std::vector<float> result(static_cast<std::size_t>(size) * size);
        for (int i = 0; i < size; ++i) result[static_cast<std::size_t>(i) * size + i] = 1.0f;
        return result;
    };
    auto projection = [](int input, int output) {
        std::vector<float> result(static_cast<std::size_t>(input) * output);
        for (int i = 0; i < std::min(input, output); ++i)
            result[static_cast<std::size_t>(i) * input + i] = 0.1f;
        return result;
    };
    const auto zero_d = std::vector<float>(D);
    const auto zero_64 = std::vector<float>(64);
    const auto zero_32 = std::vector<float>(32);
    const auto zero_256 = std::vector<float>(feature_channels);
    const auto eye_d_values = identity(D);
    const auto eye_256_values = identity(feature_channels);
    auto* eye_d = matrix_input("decoder_eye", D, D, eye_d_values);
    auto* zero_matrix = matrix_input("decoder_zero_matrix", D, D, std::vector<float>(D * D));
    auto* zero_d_tensor = vector_input("decoder_zero_d", D, zero_d);
    auto* norm_d_tensor = vector_input("decoder_norm_d", D, std::vector<float>(D, 1.0f));

    sam::internal::sam3::sam3_model model;
    model.hparams.sam_embed_dim = D;
    model.hparams.sam_dec_depth = 1;
    model.hparams.sam_n_multimask = 3;
    auto& dec = model.sam_dec;
    dec.obj_score_token = matrix_input("decoder_obj_token", D, 1, values(D, 0.2f, 0.1f));
    dec.iou_token = matrix_input("decoder_iou_token", D, 1, values(D, 0.15f, 0.7f));
    dec.mask_tokens = matrix_input("decoder_mask_tokens", D, 4, values(D * 4, 0.17f, 1.1f));

    auto fill_attention = [&](sam::internal::sam3::sam3_sam_attn& attn, const char* prefix) {
        (void)prefix;
        attn.q_w = attn.k_w = attn.v_w = attn.out_w = eye_d;
        attn.q_b = attn.k_b = attn.v_b = attn.out_b = zero_d_tensor;
    };
    dec.twoway_blocks.resize(1);
    auto& block = dec.twoway_blocks.front();
    fill_attention(block.self_attn, "self");
    fill_attention(block.ca_tok2img, "tok2img");
    fill_attention(block.ca_img2tok, "img2tok");
    block.norm1_w = block.norm2_w = block.norm3_w = block.norm4_w = norm_d_tensor;
    block.norm1_b = block.norm2_b = block.norm3_b = block.norm4_b = zero_d_tensor;
    block.mlp_fc1_w = eye_d;
    block.mlp_fc1_b = zero_d_tensor;
    block.mlp_fc2_w = zero_matrix;
    block.mlp_fc2_b = zero_d_tensor;
    fill_attention(dec.final_attn, "final");
    dec.final_norm_w = norm_d_tensor;
    dec.final_norm_b = zero_d_tensor;

    dec.up1_w = tensor4_input("decoder_up1_w", 2, 2, 64, D, std::vector<float>(4 * 64 * D));
    dec.up1_b = vector_input("decoder_up1_b", 64, zero_64);
    dec.up1_norm_w = vector_input("decoder_up1_norm_w", 64, std::vector<float>(64, 1.0f));
    dec.up1_norm_b = vector_input("decoder_up1_norm_b", 64, zero_64);
    dec.up2_w = tensor4_input("decoder_up2_w", 2, 2, 32, 64, std::vector<float>(4 * 32 * 64));
    dec.up2_b = vector_input("decoder_up2_b", 32, zero_32);

    std::vector<float> conv_s1(static_cast<std::size_t>(feature_channels) * 64);
    std::vector<float> conv_s0(static_cast<std::size_t>(feature_channels) * 32);
    for (int c = 0; c < 64; ++c) conv_s1[static_cast<std::size_t>(c) * feature_channels + c] = 1.0f;
    for (int c = 0; c < 32; ++c) conv_s0[static_cast<std::size_t>(c) * feature_channels + c] = 1.0f;
    dec.conv_s1_w = tensor4_input("decoder_conv_s1_w", 1, 1, feature_channels, 64, std::move(conv_s1));
    dec.conv_s1_b = vector_input("decoder_conv_s1_b", 64, zero_64);
    dec.conv_s0_w = tensor4_input("decoder_conv_s0_w", 1, 1, feature_channels, 32, std::move(conv_s0));
    dec.conv_s0_b = vector_input("decoder_conv_s0_b", 32, zero_32);

    const auto project_d32 = projection(D, 32);
    const auto project_d4 = projection(D, 4);
    const auto project_d1 = projection(D, 1);
    for (int mask = 0; mask < 4; ++mask) {
        dec.hyper_w[mask][0] = dec.hyper_w[mask][1] = eye_d;
        dec.hyper_b[mask][0] = dec.hyper_b[mask][1] = zero_d_tensor;
        const auto index = std::to_string(mask);
        dec.hyper_w[mask][2] = matrix_input(("decoder_hyper_last_" + index).c_str(), D, 32, project_d32);
        dec.hyper_b[mask][2] = vector_input(("decoder_hyper_bias_" + index).c_str(), 32, zero_32);
    }
    dec.iou_head_w[0] = dec.iou_head_w[1] = eye_d;
    dec.iou_head_b[0] = dec.iou_head_b[1] = zero_d_tensor;
    dec.iou_head_w[2] = matrix_input("decoder_iou_last", D, 4, project_d4);
    dec.iou_head_b[2] = vector_input("decoder_iou_bias", 4, std::vector<float>(4));
    dec.obj_head_w[0] = dec.obj_head_w[1] = eye_d;
    dec.obj_head_b[0] = dec.obj_head_b[1] = zero_d_tensor;
    dec.obj_head_w[2] = matrix_input("decoder_object_last", D, 1, project_d1);
    dec.obj_head_b[2] = vector_input("decoder_object_bias", 1, std::vector<float>(1));

    std::vector<float> pointer_first(static_cast<std::size_t>(D) * feature_channels);
    for (int c = 0; c < D; ++c) pointer_first[static_cast<std::size_t>(c) * D + c] = 1.0f;
    model.obj_ptr_proj_w[0] = matrix_input("decoder_pointer_first", D, feature_channels, std::move(pointer_first));
    model.obj_ptr_proj_w[1] = model.obj_ptr_proj_w[2] = matrix_input(
        "decoder_pointer_identity", feature_channels, feature_channels, eye_256_values);
    for (auto& bias : model.obj_ptr_proj_b)
        bias = vector_input("decoder_pointer_bias", feature_channels, zero_256);

    auto* image_batch = tensor4_input("decoder_image_batch", D, H, H, batch, [&] {
        auto base = values(D * H * H, 0.12f, 0.2f);
        std::vector<float> result(base.size() * batch);
        for (int b = 0; b < batch; ++b)
            std::copy(base.begin(), base.end(), result.begin() + static_cast<std::size_t>(b) * base.size());
        return result;
    }());
    auto* position = tensor4_input("decoder_position_shared", D, H, H, 1, values(D * H * H, 0.07f, 0.5f));
    auto* dense_batch = tensor4_input("decoder_dense_batch", D, H, H, batch, [&] {
        auto base = values(D * H * H, 0.05f, 1.3f);
        std::vector<float> result(base.size() * batch);
        for (int b = 0; b < batch; ++b)
            std::copy(base.begin(), base.end(), result.begin() + static_cast<std::size_t>(b) * base.size());
        return result;
    }());
    auto* sparse = sam::internal::input_tensor(ctx, "decoder_sparse_batch", D, 2, batch);
    std::vector<float> sparse_values(static_cast<std::size_t>(D) * 2 * batch);
    for (int b = 0; b < batch; ++b) for (int token = 0; token < 2; ++token)
        for (int c = 0; c < D; ++c) {
            const auto i = (static_cast<std::size_t>(b) * 2 + token) * D + c;
            sparse_values[i] = std::sin(i * 0.031 + b * 0.71 + token * 0.23) * 0.3f;
        }
    uploads.emplace_back(sparse, std::move(sparse_values));
    auto* feat_s0 = tensor4_input("decoder_highres_shared", feature_channels, high, high, 1,
                                  values(static_cast<std::size_t>(feature_channels) * high * high, 0.09f, 0.4f));
    auto* feat_s1 = tensor4_input("decoder_midres_shared", feature_channels, middle, middle, 1,
                                  values(static_cast<std::size_t>(feature_channels) * middle * middle, 0.08f, 0.9f));

    const auto batched = sam::internal::sam3::sam3_build_sam_dec_graph(
        ctx, model, image_batch, position, sparse, dense_batch, feat_s0, feat_s1, H);
    auto* batched_pointer = sam::internal::sam3::sam3_mlp_forward(
        ctx, batched.mask_tokens, model.obj_ptr_proj_w, model.obj_ptr_proj_b, 3);
    graph.output(batched.masks); graph.output(batched.iou_pred); graph.output(batched.obj_score);
    graph.output(batched.sam_token); graph.output(batched.mask_tokens); graph.output(batched_pointer);

    std::vector<sam::internal::sam3::sam3_dec_result> serial;
    std::vector<ggml_tensor*> serial_pointers;
    for (int b = 0; b < batch; ++b) {
        auto* image = ggml_view_4d(ctx, image_batch, D, H, H, 1,
            image_batch->nb[1], image_batch->nb[2], image_batch->nb[3], b * image_batch->nb[3]);
        auto* dense = ggml_view_4d(ctx, dense_batch, D, H, H, 1,
            dense_batch->nb[1], dense_batch->nb[2], dense_batch->nb[3], b * dense_batch->nb[3]);
        auto* object_sparse = ggml_view_3d(ctx, sparse, D, 2, 1,
            sparse->nb[1], sparse->nb[2], b * sparse->nb[2]);
        serial.push_back(sam::internal::sam3::sam3_build_sam_dec_graph(
            ctx, model, image, position, object_sparse, dense, feat_s0, feat_s1, H));
        serial_pointers.push_back(sam::internal::sam3::sam3_mlp_forward(
            ctx, serial.back().mask_tokens, model.obj_ptr_proj_w, model.obj_ptr_proj_b, 3));
        graph.output(serial.back().masks); graph.output(serial.back().iou_pred); graph.output(serial.back().obj_score);
        graph.output(serial.back().sam_token); graph.output(serial.back().mask_tokens); graph.output(serial_pointers.back());
    }

    graph.allocate();
    for (const auto& input : uploads) sam::internal::upload(input.first, input.second, stats);
    graph.compute();
    check_backend_placement(backend, stats);

    constexpr std::size_t mask_pixels = high * high;
    require(batched.masks->ne[0] == mask_pixels && batched.masks->ne[1] == 4 && batched.masks->ne[2] == batch &&
            batched.iou_pred->ne[0] == 4 && batched.iou_pred->ne[1] == 1 && batched.iou_pred->ne[2] == batch &&
            batched.obj_score->ne[0] == 1 && batched.obj_score->ne[1] == 1 && batched.obj_score->ne[2] == batch &&
            batched.sam_token->ne[0] == D && batched.sam_token->ne[1] == 1 && batched.sam_token->ne[2] == batch &&
            batched.mask_tokens->ne[0] == D && batched.mask_tokens->ne[1] == 4 && batched.mask_tokens->ne[2] == batch &&
            batched_pointer->ne[0] == feature_channels && batched_pointer->ne[1] == 4 && batched_pointer->ne[2] == batch,
            "Full decoder did not return object-major mask/iou/object/pointer shapes");
    const auto batched_masks = sam::internal::download(batched.masks, stats);
    const auto batched_iou = sam::internal::download(batched.iou_pred, stats);
    const auto batched_object = sam::internal::download(batched.obj_score, stats);
    const auto batched_token = sam::internal::download(batched.sam_token, stats);
    const auto batched_mask_tokens = sam::internal::download(batched.mask_tokens, stats);
    const auto batched_pointers = sam::internal::download(batched_pointer, stats);
    for (int b = 0; b < batch; ++b) {
        const auto& single = serial[static_cast<std::size_t>(b)];
        const auto single_masks = sam::internal::download(single.masks, stats);
        const auto single_iou = sam::internal::download(single.iou_pred, stats);
        const auto single_object = sam::internal::download(single.obj_score, stats);
        const auto single_token = sam::internal::download(single.sam_token, stats);
        const auto single_mask_tokens = sam::internal::download(single.mask_tokens, stats);
        const auto single_pointer = sam::internal::download(serial_pointers[static_cast<std::size_t>(b)], stats);
        const auto compare = [](const std::vector<float>& batch_values, std::size_t offset,
                                const std::vector<float>& single_values, const char* name) {
            require(offset + single_values.size() <= batch_values.size(), "Decoder output slice is out of range");
            for (std::size_t i = 0; i < single_values.size(); ++i)
                if (std::abs(batch_values[offset + i] - single_values[i]) > 2e-4f)
                    throw std::runtime_error(std::string("Batched decoder ") + name + " differs from serial slice");
        };
        compare(batched_masks, static_cast<std::size_t>(b) * 4 * mask_pixels, single_masks, "mask");
        compare(batched_iou, static_cast<std::size_t>(b) * 4, single_iou, "IoU");
        compare(batched_object, static_cast<std::size_t>(b), single_object, "object score");
        compare(batched_token, static_cast<std::size_t>(b) * D, single_token, "SAM token");
        compare(batched_mask_tokens, static_cast<std::size_t>(b) * 4 * D, single_mask_tokens, "mask token");
        compare(batched_pointers, static_cast<std::size_t>(b) * 4 * feature_channels, single_pointer, "pointer");
    }
    (void)zero_256;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto backend = sam::test::cuda_requested(argc, argv) ? sam::Backend::Cuda : sam::Backend::Cpu;
        if (backend == sam::Backend::Cuda && !sam::test::cuda_available()) {
            std::cout << "SKIP: CUDA tracking subgraphs require a visible GPU\n";
            return 77;
        }
        check_mutable_tracker_chunk_limit();
        check_none_serial_policy_key();
        check_deferred_resident_upload_lifecycle();
        check_preprocessing(); check_mask_resize(); check_inventory(); check_attention(backend); check_sam_cross_attention(backend);
        check_sam_cross_attention_batch(backend); check_deconv_batch(backend); check_memory_attention_batch(backend);
        if (backend == sam::Backend::Cuda) check_attention(backend, sam::CudaComputeMode::F16);
        check_mask_decoder_batch(backend);
        check_memory_selection(); return 0;
    }
    catch (const std::exception& error) { std::cerr << "tracking math: " << error.what() << '\n'; return 1; }
}
