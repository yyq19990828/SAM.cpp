#include "linear_probe.hpp"
#include <sam/internal/runtime/ggml/graph.hpp>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>

namespace {

std::size_t positive_size(const char* text, std::size_t limit) {
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (consumed != std::strlen(text) || value == 0 || value > limit)
        throw std::invalid_argument("invalid probe dimension");
    return static_cast<std::size_t>(value);
}

void write_bytes(const std::filesystem::path& path, const void* data, std::size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream) throw std::runtime_error("failed writing cache probe payload");
}

void write_timing(std::ostream& stream, const std::vector<double>& samples) {
    stream << "{\"median_ms\":" << sam_probe::quantile(samples, 0.5)
           << ",\"p95_ms\":" << sam_probe::quantile(samples, 0.95) << ",\"samples_ms\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i) stream << ',';
        stream << samples[i];
    }
    stream << "]}";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 7 || argc > 8)
            throw std::invalid_argument("usage: sam_cache_codec_probe INPUT f32|f16|q8_0 cpu|cuda ROWS CHANNELS NEW_DIR [ITERATIONS]");
        const std::string mode = argv[2], backend = argv[3];
        if ((mode != "f32" && mode != "f16" && mode != "q8_0") || (backend != "cpu" && backend != "cuda"))
            throw std::invalid_argument("unsupported cache codec or backend");
        const auto rows = positive_size(argv[4], 1u << 20);
        const auto channels = positive_size(argv[5], 16384);
        if (rows > (1u << 26) / channels || (mode == "q8_0" && channels % 32))
            throw std::invalid_argument("cache shape exceeds bounds or violates the Q8_0 block size");
        const auto elements = rows * channels;
        const std::filesystem::path input_path = argv[1], output = argv[6];
        if (!std::filesystem::is_regular_file(input_path) || std::filesystem::file_size(input_path) != elements * sizeof(float))
            throw std::invalid_argument("cache probe input length differs");
        const int iterations = argc == 8 ? sam_probe::iterations(argv[7]) : 20;
        std::vector<float> values(elements);
        std::ifstream input_file(input_path, std::ios::binary);
        input_file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(elements * sizeof(float)));
        if (!input_file || std::any_of(values.begin(), values.end(), [](float value) { return !std::isfinite(value); }))
            throw std::invalid_argument("cache probe requires finite F32 input");
        if (!std::filesystem::create_directory(output)) throw std::runtime_error("cache probe output must be new");

        const auto type = mode == "q8_0" ? GGML_TYPE_Q8_0 : (mode == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32);
        const auto device = backend == "cuda" ? sam::Backend::Cuda : sam::Backend::Cpu;
        sam::internal::GgmlRuntime runtime({device, 4}, true);
        sam::RuntimeStats stats;
        auto context = sam::internal::make_context(4);
        auto* original = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, channels, rows);
        auto* transferred = ggml_new_tensor_2d(context.get(), type, channels, rows);
        sam::internal::BufferPtr resident(ggml_backend_alloc_ctx_tensors(context.get(), runtime.weights_backend()));
        if (!resident) throw std::runtime_error("cache probe allocation failed");
        sam::internal::upload(original, values, stats);

        std::unique_ptr<sam::internal::GraphExecution> encoder, decoder;
        ggml_tensor* packed = original;
        ggml_tensor* decoded = transferred;
        if (type != GGML_TYPE_F32) {
            encoder = std::make_unique<sam::internal::GraphExecution>(runtime, 32, stats);
            packed = ggml_cast(encoder->context(), original, type);
            encoder->output(packed);
            encoder->allocate();
            decoder = std::make_unique<sam::internal::GraphExecution>(runtime, 32, stats);
            decoded = ggml_cast(decoder->context(), transferred, GGML_TYPE_F32);
            decoder->output(decoded);
            decoder->allocate();
        }
        std::vector<unsigned char> payload(ggml_nbytes(packed));
        std::vector<double> encode_ms, decode_ms;
        auto run = [&](bool measured) {
            const auto encode_start = std::chrono::steady_clock::now();
            if (encoder) encoder->compute();
            ggml_backend_tensor_get(packed, payload.data(), 0, payload.size());
            const auto encoded_ms = sam_probe::milliseconds(encode_start);
            const auto decode_start = std::chrono::steady_clock::now();
            ggml_backend_tensor_set(transferred, payload.data(), 0, payload.size());
            if (decoder) decoder->compute();
            ggml_backend_synchronize(runtime.weights_backend());
            const auto decoded_ms = sam_probe::milliseconds(decode_start);
            if (measured) {
                encode_ms.push_back(encoded_ms);
                decode_ms.push_back(decoded_ms);
            }
        };
        const auto warmup_start = std::chrono::steady_clock::now();
        int warmup_calls = 0;
        do { run(false); ++warmup_calls; } while (warmup_calls < 10 || sam_probe::milliseconds(warmup_start) < 250);
        for (int i = 0; i < iterations; ++i) run(true);
        if (backend == "cuda" && (stats.cpu_nodes || stats.metal_nodes || (type != GGML_TYPE_F32 && !stats.cuda_nodes)))
            throw std::runtime_error("cache codec used compute outside CUDA");
        const auto actual = sam::internal::download(decoded, stats);
        const auto error = sam_probe::error(actual, values, values.size());
        write_bytes(output / "packed.bin", payload.data(), payload.size());
        write_bytes(output / "decoded.f32", actual.data(), actual.size() * sizeof(float));
        std::ofstream report(output / "probe.json");
        report << std::setprecision(17) << "{\"schema_version\":1,\"complete\":true,\"diagnostic_only\":true"
            << ",\"scope\":\"Resident F32 input: encode plus pageable-host download, then upload plus decode. No SAM quality or end-to-end claim.\""
            << ",\"mode\":" << std::quoted(mode) << ",\"backend\":" << std::quoted(backend)
            << ",\"device\":" << std::quoted(ggml_backend_name(runtime.weights_backend()))
            << ",\"rows\":" << rows << ",\"channels\":" << channels << ",\"warmup_calls\":" << warmup_calls
            << ",\"original_payload_bytes\":" << values.size() * sizeof(float)
            << ",\"packed_payload_bytes\":" << payload.size()
            << ",\"relative_l2\":" << error.relative_l2 << ",\"max_abs\":" << error.max_abs
            << ",\"cuda_nodes\":" << stats.cuda_nodes << ",\"cpu_nodes\":" << stats.cpu_nodes
            << ",\"resident_probe_bytes\":" << ggml_backend_buffer_get_size(resident.get())
            << ",\"encode_download\":";
        write_timing(report, encode_ms);
        report << ",\"upload_decode\":";
        write_timing(report, decode_ms);
        report << "}\n";
        if (!report) throw std::runtime_error("cache probe report failed");
        std::cout << mode << ' ' << backend << " payload " << payload.size() << " bytes; encode/download "
            << sam_probe::quantile(encode_ms, 0.5) << " ms; upload/decode " << sam_probe::quantile(decode_ms, 0.5) << " ms\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_cache_codec_probe: " << error.what() << '\n';
        return 1;
    }
}
