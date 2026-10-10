#include "cpu_quantized_matmul.hpp"
#include "../../support/image_io/image_support.hpp"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>

namespace {

using Clock = std::chrono::steady_clock;

double quantile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const auto position = q * (values.size() - 1);
    const auto low = static_cast<std::size_t>(position), high = std::min(low + 1, values.size() - 1);
    return values[low] + (position - low) * (values[high] - values[low]);
}

template<class Function>
double measure(Function function, int repeats = 1) {
    const auto start = Clock::now();
    for (int i = 0; i < repeats; ++i) function();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count() / repeats;
}

void samples(std::ostream& output, const std::vector<double>& values) {
    output << "{\"p50_ms\":" << quantile(values, .5) << ",\"p95_ms\":" << quantile(values, .95) << ",\"samples_ms\":[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) output << ',';
        output << values[i];
    }
    output << "]}";
}

void accuracy(std::ostream& output, const sam_cpu_probe::Accuracy& result) {
    output << "{\"sampled_outputs\":" << result.checked << ",\"max_decoded_scalar_error\":" << result.max_effective_error
           << ",\"max_difference_from_f32_activations\":" << result.max_f32_activation_difference
           << ",\"rmse_difference_from_f32_activations\":" << std::sqrt(result.squared_difference / result.checked) << '}';
}

void run_case(std::ostream& output, const sam_cpu_probe::Fixture& data, int threads, int warmups, int iterations) {
    sam_cpu_probe::Matmul baseline(data, sam::CpuComputeMode::F32, threads);
    sam_cpu_probe::Matmul native(data, sam::CpuComputeMode::NativeQuantized, threads);
    for (int i = 0; i < warmups; ++i) { baseline.compute(); native.compute(); }
    const auto slow_ms = std::max(measure([&] { baseline.compute(); }), measure([&] { native.compute(); }));
    const int repeats = std::clamp(static_cast<int>(std::ceil(2.0 / std::max(slow_ms, .001))), 1, 64);
    std::vector<double> old_times, native_times, packing_times;
    auto packed_rhs = data.packed_rhs;
    for (int i = 0; i < iterations; ++i) {
        double old_ms, native_ms;
        if (i % 2 == 0) {
            old_ms = measure([&] { baseline.compute(); }, repeats);
            native_ms = measure([&] { native.compute(); }, repeats);
        } else {
            native_ms = measure([&] { native.compute(); }, repeats);
            old_ms = measure([&] { baseline.compute(); }, repeats);
        }
        old_times.push_back(old_ms);
        native_times.push_back(native_ms);
    }
    // Separate single-thread packing experiment. It cannot be subtracted from
    // multithreaded MUL_MAT times to infer the kernel's private timing split.
    for (int i = 0; i < warmups; ++i) data.pack_rhs(packed_rhs);
    for (int i = 0; i < iterations; ++i)
        packing_times.push_back(measure([&] { data.pack_rhs(packed_rhs); }, repeats));
    const auto old_accuracy = sam_cpu_probe::validate(data, baseline.read(), false, false);
    const auto native_accuracy = sam_cpu_probe::validate(data, native.read(), true, false);
    baseline.check_resident_weights();
    native.check_resident_weights();
    using sam_example::json_string;
    output << "{\"weight_type\":" << json_string(ggml_type_name(data.type))
           << ",\"shape\":{\"k\":" << data.width << ",\"m\":" << data.rows << ",\"n\":" << data.columns << '}'
           << ",\"qkv_views\":" << (data.qkv ? "true" : "false")
           << ",\"scenario\":" << json_string(data.scenario) << ",\"threads\":" << threads
           << ",\"repeats_per_sample\":" << repeats << ",\"resident_weight_bytes_per_variant\":" << data.packed.size()
           << ",\"graph_input_type\":\"f32\",\"graph_output_type\":\"f32\",\"standard_cpu_rhs_type\":"
           << json_string(ggml_type_name(sam::internal::cpu_quantized_rhs_type(data.type)))
           << ",\"rhs_packed_payload_bytes\":" << data.packed_rhs.size()
           << ",\"source_precision_evidence\":\"pinned CPU type traits and decoded independent reference; kernel scratch not captured\""
           << ",\"baseline\":{\"arithmetic_profile\":" << json_string(baseline.profile())
           << ",\"actual_graph_lhs_type\":" << json_string(ggml_type_name(baseline.left_type()))
           << ",\"scheduler_arena_peak_bytes\":" << baseline.diagnostics().workspace_peak_bytes
           << ",\"blas_nodes\":" << baseline.stats().blas_nodes << ",\"timing\":";
    samples(output, old_times);
    output << ",\"accuracy\":";
    accuracy(output, old_accuracy);
    output << "},\"native\":{\"arithmetic_profile\":" << json_string(native.profile())
           << ",\"actual_graph_lhs_type\":" << json_string(ggml_type_name(native.left_type()))
           << ",\"quantized_matmul_backend\":\"CPU (enforced and checked by GraphWorkspace)\""
           << ",\"scheduler_arena_peak_bytes\":" << native.diagnostics().workspace_peak_bytes
           << ",\"blas_nodes\":" << native.stats().blas_nodes << ",\"timing\":";
    samples(output, native_times);
    output << ",\"accuracy\":";
    accuracy(output, native_accuracy);
    output << "},\"rhs_packing_single_thread_diagnostic\":";
    samples(output, packing_times);
    output << ",\"speedup_baseline_over_native\":" << quantile(old_times, .5) / quantile(native_times, .5) << '}';
    std::cout << ggml_type_name(data.type) << " k=" << data.width << " m=" << data.rows << " n=" << data.columns
              << " CPU threads=" << threads << " checked; median speedup=" << quantile(old_times, .5) / quantile(native_times, .5) << '\n';
}

const char* environment(const char* name) {
    const auto value = std::getenv(name);
    return value ? value : "UNSET";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path output;
        int threads = 4, warmups = 3, iterations = 15;
        std::set<std::string> seen;
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: sam_cpu_quantized_matmul_probe --output NEW_DIR [--threads 1..8] [--warmups 1..20] [--iterations 3..100]\n"
                         "Bounded synthetic CPU matrices only; alternates F32 decoding/native Q matmul; no GPU or whole-model inference.\n";
            return 0;
        }
        for (int i = 1; i < argc; i += 2) {
            const std::string name = argv[i];
            if (i + 1 >= argc || !seen.insert(name).second) throw std::invalid_argument("missing or duplicate probe argument");
            if (name == "--output") output = argv[i + 1];
            else if (name == "--threads") threads = sam_example::positive_integer(argv[i + 1], name);
            else if (name == "--warmups") warmups = sam_example::positive_integer(argv[i + 1], name);
            else if (name == "--iterations") iterations = sam_example::positive_integer(argv[i + 1], name);
            else throw std::invalid_argument("unknown CPU probe argument: " + name);
        }
        if (output.empty() || threads > 8 || warmups > 20 || iterations < 3 || iterations > 100)
            throw std::invalid_argument("CPU probe requires output and bounded threads/warmups/iterations; see --help");
        sam_example::OutputDirectory destination(output, true);
        auto cpu = sam::internal::make_cpu_backend(threads);
        std::ostringstream cases;
        cases << std::setprecision(12);
        bool first = true;
        const std::vector<std::tuple<int, int, int, bool>> shapes{
            {256, 257, 1, false}, {256, 257, 8, false}, {1024, 256, 16, false},
            {1024, 1024, 64, false}, {256, 768, 9, true}};
        for (const auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K}) {
            for (const auto& [k, m, n, qkv] : shapes) {
                if (!first) cases << ',';
                first = false;
                run_case(cases, sam_cpu_probe::Fixture(type, k, m, n, "ordinary", qkv), threads, warmups, iterations);
            }
        }
        cases << ',';
        run_case(cases, sam_cpu_probe::Fixture(GGML_TYPE_Q8_0, 4736, 65, 3, "outlier"), threads, warmups, iterations);
        auto file = sam_example::output_file(output / "report.json");
        file << "{\"schema_version\":1,\"kind\":\"sam-cpu-quantized-matmul-study-v1\",\"complete\":true,"
                "\"scope\":\"synthetic bounded operator study; no checkpoint, SAM image quality or whole-model speed claim\","
                "\"quality_gate\":false,\"graph_observer\":false,\"timing_order\":\"alternating AB/BA after warmup; same packed weights/F32 inputs\","
                "\"timing_scope\":\"synchronous reused GraphExecution.compute wall time; F32 baseline decodes each compute; native includes internal RHS packing\","
                "\"accuracy_scope\":\"boundary/middle rows and columns; independent scalar dots over decoded packed weights and CPU-traits-packed RHS; F32 activation delta is separate\","
                "\"packing_scope\":\"separate single-thread CPU packing experiment; not kernel instrumentation; do not subtract from MUL_MAT\","
                "\"memory_scope\":\"scheduler tensor arena, not RSS/total peak; excludes weights, source/decoded oracle arrays, threadpool/tiled scratch and other backend allocations\","
                "\"kernel_internal_arithmetic\":\"NOT_COLLECTED\",\"checkpoint\":null,\"generator\":\"deterministic-synthetic-v1\","
                "\"cpu\":" << sam_example::json_string(cpu.device_name)
             << ",\"ggml_version\":" << sam_example::json_string(ggml_version())
             << ",\"ggml_commit\":" << sam_example::json_string(ggml_commit())
             << ",\"configured_options\":{\"GGML_BLAS\":\"" << SAM_PROBE_GGML_BLAS
             << "\",\"GGML_LLAMAFILE\":\"" << SAM_PROBE_GGML_LLAMAFILE << "\",\"GGML_NATIVE\":\"" << SAM_PROBE_GGML_NATIVE << "\"}"
             << ",\"linked_cpu_isa\":{\"avx2\":" << ggml_cpu_has_avx2() << ",\"avx_vnni\":" << ggml_cpu_has_avx_vnni()
             << ",\"avx512_vnni\":" << ggml_cpu_has_avx512_vnni() << '}'
             << ",\"environment\":{\"GGML_CPU_TILED_MM\":" << sam_example::json_string(environment("GGML_CPU_TILED_MM"))
             << ",\"GGML_CPU_TILED_MM_FORCE\":" << sam_example::json_string(environment("GGML_CPU_TILED_MM_FORCE")) << '}'
             << ",\"threads\":" << threads << ",\"warmups\":" << warmups << ",\"iterations\":" << iterations
             << ",\"cases\":[" << cases.str() << "]}\n";
        file.close();
        destination.complete();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_cpu_quantized_matmul_probe: " << error.what() << '\n';
        return 1;
    }
}
