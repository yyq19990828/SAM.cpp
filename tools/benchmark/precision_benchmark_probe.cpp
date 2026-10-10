#include "../../support/image_io/image_support.hpp"
#include <models/sam3/model.hpp>

#include <iostream>

namespace {

void mark_phase(const std::filesystem::path& directory, const char* phase) {
    const auto temporary = directory / "phase.next";
    auto file = sam_example::output_file(temporary);
    file << phase << '\n';
    file.close();
    std::filesystem::rename(temporary, directory / "phase.txt");
}

void write_times(std::ostream& output, const std::vector<double>& times) {
    output << '[';
    for (std::size_t i = 0; i < times.size(); ++i) {
        if (i) output << ',';
        output << times[i];
    }
    output << ']';
}

int parse_count(const char* argument) {
    const std::string text(argument);
    std::size_t consumed = 0;
    const auto value = std::stoll(text, &consumed);
    if (consumed != text.size() || value < 0 || value > 10000)
        throw std::invalid_argument("benchmark counts must be integers in [0,10000]");
    return static_cast<int>(value);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 10 && argc != 12)
            throw std::invalid_argument("usage: sam_precision_benchmark_probe MODEL IMAGE PROMPT ALTERNATE cpu|cuda CACHE COMPUTE latency|memory NEW_DIR [WARMUPS ITERATIONS]");
        const std::string prompt = argv[3], alternate = argv[4], backend = argv[5], cache = argv[6], compute = argv[7], kind = argv[8];
        if (prompt.empty() || alternate.empty() || prompt == alternate || (backend != "cpu" && backend != "cuda") ||
            (cache != "f32" && cache != "f16" && cache != "mixed-q8_0") ||
            (compute != "f32" && compute != "f16" && compute != "native-quantized") ||
            (kind != "latency" && kind != "memory") ||
            (backend == "cpu" && (cache != "f32" || compute == "f16")) ||
            (backend != "cpu" && compute == "native-quantized"))
            throw std::invalid_argument("invalid precision benchmark recipe/workload");
        const int warmups = argc == 12 ? parse_count(argv[10]) : 5;
        const int iterations = argc == 12 ? parse_count(argv[11]) : 20;
        if (warmups < 0 || warmups > 10000 || iterations < 1 || iterations > 10000)
            throw std::invalid_argument("benchmark requires warmups in [0,10000] and iterations in [1,10000]");
        sam_example::OutputDirectory destination(argv[9], true);
        const std::filesystem::path directory = argv[9];
        mark_phase(directory, "load");
        const auto image = sam_example::read_image(argv[2]);
        sam::BackendOptions options{backend == "cuda" ? sam::Backend::Cuda : sam::Backend::Cpu, 4};
        options.cuda_compute = compute == "f16" ? sam::CudaComputeMode::F16 : sam::CudaComputeMode::F32;
        options.cpu_compute = compute == "native-quantized" ? sam::CpuComputeMode::NativeQuantized : sam::CpuComputeMode::F32;
        using sam::internal::FeatureCacheMode;
        const auto mode = cache == "f32" ? FeatureCacheMode::F32 : cache == "f16" ? FeatureCacheMode::F16 : FeatureCacheMode::Q8_0;
        const auto load_start = sam_example::Clock::now();
        const auto state = sam::internal::sam3::load_state(argv[1], options, mode);
        const double load_ms = sam_example::elapsed_ms(load_start);
        const auto load_rss = sam_example::process_peak_rss_bytes();
        sam::internal::sam3::ImageSession session(state);
        mark_phase(directory, "first_inference");
        session.set_image(sam_example::image_view(image));
        auto result = session.segment_text(prompt, .5f);
        const auto initial_stats = session.stats();
        std::vector<double> full, changed, repeated;
        auto measure = [&](const char* phase, std::vector<double>& times, auto&& operation) {
            mark_phase(directory, phase);
            for (int i = 0; i < warmups + iterations; ++i) {
                const auto started = sam_example::Clock::now();
                operation(i);
                const auto elapsed = sam_example::elapsed_ms(started);
                if (i >= warmups) times.push_back(elapsed);
            }
        };
        measure("full_image", full, [&](int) {
            session.set_image(sam_example::image_view(image));
            result = session.segment_text(prompt, .5f);
        });
        measure("changed_prompt", changed, [&](int i) {
            result = session.segment_text(i % 2 == 0 ? alternate : prompt, .5f);
        });
        result = session.segment_text(prompt, .5f);
        measure("repeated_result", repeated, [&](int) { result = session.segment_text(prompt, .5f); });
        const auto& stats = session.stats();
        if (backend == "cuda" && (!stats.cuda_nodes || stats.cpu_nodes || stats.metal_nodes || stats.blas_nodes))
            throw std::runtime_error("benchmark escaped strict CUDA placement");
        auto output = sam_example::output_file(directory / "result.json");
        output << "{\"schema_version\":2,\"complete\":true,\"kind\":" << sam_example::json_string(kind)
               << ",\"backend\":" << sam_example::json_string(backend) << ",\"feature_cache\":" << sam_example::json_string(cache)
               << ",\"prompt\":" << sam_example::json_string(prompt)
               << ",\"alternate\":" << sam_example::json_string(alternate) << ",\"threads\":4,\"warmups\":" << warmups
               << ",\"iterations\":" << iterations << ",\"model_load_ms\":" << load_ms << ",\"load_peak_rss_bytes\":" << load_rss
               << ",\"rss_peak_bytes\":" << sam_example::process_peak_rss_bytes() << ",\"timing_ms\":{\"full_image\":";
        write_times(output, full);
        write_times(output << ",\"changed_prompt\":", changed);
        write_times(output << ",\"repeated_result\":", repeated);
        output << "},\"runtime\":";
        sam_example::write_runtime_stats(output, stats);
        output << ",\"first_runtime\":";
        sam_example::write_runtime_stats(output, initial_stats);
        sam_example::write_model_profile(output, state->model_info);
        sam_example::write_compute_policy(output, options);
        output << ",\"feature_cache_bytes\":" << session.feature_cache_bytes() << "}\n";
        output.close();
        mark_phase(directory, "complete");
        destination.complete();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_precision_benchmark_probe: " << error.what() << '\n';
        return 1;
    }
}
