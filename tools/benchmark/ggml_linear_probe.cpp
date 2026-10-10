#include "linear_probe.hpp"
#include <runtime/ggml/graph.hpp>
#include <cstring>
#include <iostream>

namespace {

void write_payload(const std::filesystem::path& path, const void* data, std::size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream) throw std::runtime_error("failed writing GGML raw probe payload");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 7) throw std::invalid_argument("usage: sam_ggml_linear_probe INPUT ggml-q8|ggml-f16 NEW_DIR [ITERATIONS] [--summary-only] [--dump-q8-operands]");
        const auto input = sam_probe::read_input(argv[1]);
        const std::string mode = argv[2];
        if (mode != "ggml-q8" && mode != "ggml-f16") throw std::invalid_argument("unsupported GGML probe mode");
        int iterations = 50;
        bool have_iterations = false, summary_only = false, dump_q8_operands = false;
        for (int argument = 4; argument < argc; ++argument) {
            const std::string option = argv[argument];
            if (option == "--summary-only" && !summary_only) summary_only = true;
            else if (option == "--dump-q8-operands" && !dump_q8_operands) dump_q8_operands = true;
            else if (!have_iterations && option.rfind("--", 0) != 0) {
                iterations = sam_probe::iterations(argv[argument]);
                have_iterations = true;
            } else throw std::invalid_argument("unsupported or duplicate probe argument");
        }
        if (dump_q8_operands && mode != "ggml-q8")
            throw std::invalid_argument("raw Q8 operands require ggml-q8 mode");
        const std::filesystem::path output = argv[3];
        if (!std::filesystem::create_directory(output)) throw std::runtime_error("probe output must be new");
        const auto type = mode == "ggml-q8" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        if (input.k % ggml_blck_size(type)) throw std::invalid_argument("inner dimension violates weight block size");
        sam::internal::GgmlRuntime runtime({sam::Backend::Cuda, 4, 0, sam::CudaComputeMode::F16}, true);
        sam::RuntimeStats stats;
        auto weights_context = sam::internal::make_context(4);
        auto* weight = ggml_new_tensor_2d(weights_context.get(), type, input.k, input.m);
        auto* bias = ggml_new_tensor_1d(weights_context.get(), GGML_TYPE_F32, input.m);
        sam::internal::BufferPtr weights(ggml_backend_alloc_ctx_tensors(weights_context.get(), runtime.weights_backend()));
        if (!weights) throw std::runtime_error("cannot allocate probe weights");
        ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<unsigned char> packed(ggml_nbytes(weight));
        if (type == GGML_TYPE_Q8_0) {
            const auto bytes = ggml_quantize_chunk(type, input.weights.data(), packed.data(), 0, input.m, input.k, nullptr);
            if (bytes != packed.size()) throw std::runtime_error("incomplete Q8 weight packing");
        } else {
            std::vector<ggml_fp16_t> half(input.weights.size());
            ggml_fp32_to_fp16_row(input.weights.data(), half.data(), int64_t(half.size()));
            std::memcpy(packed.data(), half.data(), packed.size());
        }
        ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
        sam::internal::upload(bias, input.bias, stats);
        // Model-internal activations already live on the device. A scheduler
        // input defaults to host memory and would add a PCIe copy per call.
        auto input_context = sam::internal::make_context(2);
        auto* x = ggml_new_tensor_2d(input_context.get(), GGML_TYPE_F32, input.k, input.n);
        sam::internal::BufferPtr input_buffer(ggml_backend_alloc_ctx_tensors(input_context.get(), runtime.weights_backend()));
        if (!input_buffer || ggml_backend_buffer_is_host(input_buffer.get()))
            throw std::runtime_error("probe input must reside on CUDA");
        sam::internal::upload(x, input.expanded_input(), stats);
        sam::internal::GraphExecution graph(runtime, 128, stats);
        auto* y = ggml_add(graph.context(), ggml_mul_mat(graph.context(), weight, x), bias);
        if (input.gelu) y = ggml_gelu_erf(graph.context(), y);
        graph.output(y);
        graph.allocate();
        const auto warmup = std::chrono::steady_clock::now();
        int warmup_calls = 0;
        do { graph.compute(); ++warmup_calls; } while (warmup_calls < 10 || sam_probe::milliseconds(warmup) < 250);
        std::vector<double> timing;
        for (int i = 0; i < iterations; ++i) {
            const auto start = std::chrono::steady_clock::now();
            graph.compute();
            timing.push_back(sam_probe::milliseconds(start));
        }
        if (stats.cpu_nodes || stats.metal_nodes || !stats.cuda_nodes) throw std::runtime_error("probe did not use CUDA exclusively");
        const auto result = sam::internal::download(y, stats);
        const auto error = sam_probe::error(result, input.reference, input.reference.size());
        sam::RuntimeStats raw_stats;
        if (dump_q8_operands) {
            sam::internal::GraphExecution raw_graph(runtime, 64, raw_stats);
            auto* raw_dot = ggml_mul_mat(raw_graph.context(), weight, x);
            raw_graph.output(raw_dot);
            raw_graph.allocate();
            raw_graph.compute();
            if (raw_stats.cpu_nodes || raw_stats.metal_nodes || !raw_stats.cuda_nodes)
                throw std::runtime_error("raw Q8 dot did not execute on CUDA exclusively");
            const auto raw_values = sam::internal::download(raw_dot, raw_stats);
            const auto expanded = input.expanded_input();
            write_payload(output / "weight.q8_0", packed.data(), packed.size());
            write_payload(output / "rhs.f32", expanded.data(), expanded.size() * sizeof(float));
            write_payload(output / "raw-dot.f32", raw_values.data(), raw_values.size() * sizeof(float));
        }
        if (!summary_only) sam_probe::write_output(output, result);
        std::ofstream report(output / "probe.json");
        sam_probe::write_common(report, input, mode, timing, error);
        report << ",\"complete\":true,\"output_saved\":" << (summary_only ? "false" : "true")
            << ",\"raw_q8_operands_saved\":" << (dump_q8_operands ? "true" : "false")
            << ",\"raw_q8_layout\":\"weight[M,K] Q8_0, rhs[N,K] F32, dot[N,M] F32\""
            << ",\"raw_cuda_nodes\":" << raw_stats.cuda_nodes
            << ",\"warmup_calls\":" << warmup_calls << ",\"output_type\":\"f32\",\"cuda_nodes\":" << stats.cuda_nodes
            << ",\"cpu_nodes\":" << stats.cpu_nodes << ",\"weight_buffer_bytes\":" << ggml_backend_buffer_get_size(weights.get())
            << ",\"resident_input_bytes\":" << ggml_backend_buffer_get_size(input_buffer.get())
            << ",\"graph_workspace_bytes\":" << stats.compute_buffer_bytes
            << ",\"memory_scope\":\"GGML weights and arena only; CUDA RHS staging/pools are additional\"}\n";
        if (!report) throw std::runtime_error("failed writing probe report");
        std::cout << mode << " median " << sam_probe::quantile(timing, 0.5) << " ms, relative L2 " << error.relative_l2 << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_ggml_linear_probe: " << error.what() << '\n';
        return 1;
    }
}
