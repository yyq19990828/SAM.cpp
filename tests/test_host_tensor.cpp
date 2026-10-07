#include <sam/internal/runtime/ggml/host_tensor.hpp>
#include "backend_test_support.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Callable> void rejects(Callable&& callable, const char* message) {
    bool rejected = false;
    try { callable(); } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}

void check(sam::Backend backend) {
    using namespace sam::internal;
    GgmlRuntime runtime({backend, 2}, true);
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        sam::RuntimeStats stats;
        std::vector<float> values(32 * 3 * 5);
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto position = i % 32;
            values[i] = position == 0 ? 127.0f : (position == 1 ? -127.0f :
                (position == 2 ? 1.0003f : static_cast<float>(int(position) - 16)));
        }
        HostTensor cached;
        {
            GraphExecution encoder(runtime, 32, stats);
            auto* input = input_tensor(encoder.context(), "features", 32, 3, 5);
            auto* output = type == GGML_TYPE_F32 ? input : ggml_cast(encoder.context(), input, type);
            encoder.output(output);
            encoder.allocate();
            upload(input, values, stats);
            encoder.compute();
            cached = HostTensor::download(output, stats);
        } // The snapshot must outlive every graph tensor and allocation.
        const std::size_t expected_bytes = type == GGML_TYPE_F32 ? values.size() * 4 :
            (type == GGML_TYPE_F16 ? values.size() * 2 : values.size() / 32 * 34);
        require(cached.size_bytes() == expected_bytes && cached.elements() == values.size(),
                "cached payload size changed");
        require(stats.host_download_bytes == expected_bytes, "download accounting used decoded bytes");
        const auto host_values = cached.as_f32();
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto expected = type != GGML_TYPE_F32 && i % 32 == 2 ? 1.0f : values[i];
            require(host_values[i] == expected, "cache diagnostic decode changed row layout or values");
        }

        GraphExecution consumer(runtime, 64, stats);
        auto input = cached_input(consumer.context(), "reused_features", cached, 32, 15);
        auto* shift = input_tensor(consumer.context(), "shift", 32, 15);
        auto* output = ggml_add(consumer.context(), input.values, shift);
        consumer.output(output);
        consumer.allocate();
        const auto before_upload = stats.host_upload_bytes;
        cached.upload_to(input.stored, stats);
        require(stats.host_upload_bytes - before_upload == expected_bytes,
                "upload accounting used decoded bytes");
        upload(shift, std::vector<float>(values.size(), 0.25f), stats);
        consumer.compute();
        const auto actual = download(output, stats);
        for (std::size_t i = 0; i < actual.size(); ++i)
            require(actual[i] == host_values[i] + 0.25f, "decoded cache consumer received wrong values");
        if (backend == sam::Backend::Cuda)
            require(stats.cuda_nodes > 0 && stats.cpu_nodes == 0 && stats.metal_nodes == 0,
                    "CUDA cache consumer fell back to CPU");

        rejects([&] { cached_input(consumer.context(), "wrong_size", cached, 32, 14); }, "cache accepted wrong element count");
        rejects([&] { cached_input(consumer.context(), "wrong_rows", cached, 16, 30); }, "cache accepted a changed row layout");
        rejects([&] { cached_input(consumer.context(), "negative", cached, -32, 15); }, "cache accepted a negative dimension");
        rejects([&] { cached.validate_shape(32, std::numeric_limits<int64_t>::max(), 15, 1); }, "cache accepted shape overflow");
        rejects([&] { cached.upload_to(nullptr, stats); }, "cache accepted null upload");
        if (type != GGML_TYPE_F32)
            rejects([&] { cached.upload_to(shift, stats); }, "cache accepted a mismatched storage type");
    }

    sam::RuntimeStats stats;
    auto context = make_context(16);
    auto* contiguous = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 32, 5);
    auto* transposed = ggml_transpose(context.get(), contiguous);
    auto* integers = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 32);
    auto* halves = ggml_new_tensor_1d(context.get(), GGML_TYPE_F16, 32);
    auto* blocks = ggml_new_tensor_1d(context.get(), GGML_TYPE_Q8_0, 32);
    BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), runtime.weights_backend()));
    require(bool(buffer), "invalid-payload fixture allocation failed");
    rejects([&] { HostTensor::download(nullptr, stats); }, "cache accepted null output");
    rejects([&] { HostTensor::download(transposed, stats); }, "cache accepted a noncontiguous view");
    rejects([&] { HostTensor::download(integers, stats); }, "cache accepted unsupported integer storage");
    for (const auto bits : {std::uint16_t(0x7c00), std::uint16_t(0x7e00)}) {
        std::vector<std::uint16_t> payload(32, bits);
        ggml_backend_tensor_set(halves, payload.data(), 0, payload.size() * sizeof(std::uint16_t));
        rejects([&] { HostTensor::download(halves, stats); }, "cache accepted non-finite half values");
    }
    for (const auto bits : {std::uint16_t(0x7c00), std::uint16_t(0xbc00)}) {
        std::vector<std::uint8_t> payload(34, 0);
        std::memcpy(payload.data(), &bits, sizeof(bits));
        ggml_backend_tensor_set(blocks, payload.data(), 0, payload.size());
        rejects([&] { HostTensor::download(blocks, stats); }, "cache accepted an invalid Q8 scale");
    }
    require(stats.host_download_bytes == 2 * 64 + 2 * 34,
            "accounting omitted bytes transferred before payload validation failed");
    rejects([&] { HostTensor{}.as_f32(); }, "empty cache was decoded");
    rejects([&] { GgmlRuntime invalid({sam::Backend::Cpu, 1}, true, false, FeatureCacheMode::F16); },
            "experimental cache silently selected an unverified backend");
    rejects([&] { GgmlRuntime invalid({backend, 1}, true, false, static_cast<FeatureCacheMode>(99)); },
            "unknown cache mode was accepted");
    if (backend == sam::Backend::Cuda) {
        GgmlRuntime half({backend, 1}, true, false, FeatureCacheMode::F16);
        GgmlRuntime quantized({backend, 1}, true, false, FeatureCacheMode::Q8_0);
        require(half.feature_cache_type() == GGML_TYPE_F16 && quantized.feature_cache_type() == GGML_TYPE_Q8_0,
                "native cache capability checks did not select requested storage");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (sam::test::cuda_requested(argc, argv)) {
            if (!sam::test::cuda_available()) return 77;
            check(sam::Backend::Cuda);
        } else check(sam::Backend::Cpu);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "host tensor check: " << error.what() << '\n';
        return 1;
    }
}
