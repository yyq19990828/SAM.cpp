#include <sam/internal/models/sam3/detector.hpp>
#include <sam/internal/models/sam3/model.hpp>
#include <sam/internal/models/sam3/ops.hpp>
#include <sam/internal/runtime/ggml.hpp>

#include <cmath>
#include <iostream>

int main() {
    try {
        const std::vector<std::uint16_t> half_bits = {0, 0x8000, 0x3c00, 0xbc00, 1, 0x03ff, 0x0400, 0x7bff};
        std::vector<char> half_bytes;
        for (const auto bits : half_bits) {
            half_bytes.push_back(static_cast<char>(bits & 0xff));
            half_bytes.push_back(static_cast<char>(bits >> 8));
        }
        const auto promoted = sam::internal::sam3::promote_f16(half_bytes);
        const std::vector<float> expected_promoted = {0, -0.0f, 1, -1, std::ldexp(1.0f, -24),
            std::ldexp(1023.0f, -24), std::ldexp(1.0f, -14), 65504};
        if (promoted != expected_promoted || !std::signbit(promoted.at(1)) ||
            promoted.size() * sizeof(float) != 2 * half_bytes.size()) {
            throw std::runtime_error("FP16 promotion changed stored values or buffer size");
        }
        bool odd_bytes_rejected = false;
        try { (void) sam::internal::sam3::promote_f16({0}); }
        catch (const std::runtime_error&) { odd_bytes_rejected = true; }
        if (!odd_bytes_rejected) throw std::runtime_error("FP16 promotion accepted a partial value");
        sam::internal::GgmlRuntime runtime({sam::Backend::Cpu, 1}, true);
        sam::RuntimeStats stats;
        sam::internal::GraphExecution graph(runtime, 128, stats);
        auto* weights = sam::internal::input_tensor(graph.context(), "weights", 2, 2, 1, 1);
        auto* image = sam::internal::input_tensor(graph.context(), "image", 2, 1, 1, 1);
        auto* deconv = sam::internal::sam3::sam3_deconv_2x2(graph.context(), weights, image);
        graph.output(deconv);
        auto* endpoints = sam::internal::input_tensor(graph.context(), "endpoints", 2);
        auto* inverse_sigmoid = sam::internal::sam3::sam3_inverse_sigmoid(graph.context(), endpoints);
        graph.output(inverse_sigmoid);

        auto* conv_kernel = sam::internal::input_tensor(graph.context(), "conv_kernel", 3, 3, 1, 1);
        auto* half_kernel = ggml_new_tensor_4d(graph.context(), GGML_TYPE_F16, 3, 3, 1, 1);
        ggml_set_input(half_kernel);
        auto* conv_image = sam::internal::input_tensor(graph.context(), "conv_image", 4, 4, 1, 1);
        const std::vector<ggml_tensor*> convolutions = {
            sam::internal::sam3::sam3_conv_2d_sk_p0(graph.context(), conv_kernel, conv_image),
            sam::internal::sam3::sam3_conv_2d_s1_ph(graph.context(), conv_kernel, conv_image),
            sam::internal::sam3::sam3_conv_2d_sk_p0(graph.context(), half_kernel, conv_image),
            sam::internal::sam3::sam3_conv_2d_s1_ph(graph.context(), half_kernel, conv_image)};
        for (auto* convolution : convolutions) graph.output(convolution);

        auto* first = sam::internal::input_tensor(graph.context(), "ddec_query_pos_pres", 4);
        auto* second = sam::internal::input_tensor(graph.context(), "ddec_query_pos_pres", 4);
        auto* presence = sam::internal::input_tensor(graph.context(), "rpb_pres_zeros", 4);
        auto* zero_sum = ggml_add(graph.context(), ggml_add(graph.context(), first, second), presence);
        graph.output(zero_sum);
        graph.allocate();
        const std::vector<float> kernel_values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        sam::internal::upload(conv_kernel, kernel_values, stats);
        std::vector<ggml_fp16_t> half_kernel_values(kernel_values.size());
        ggml_fp32_to_fp16_row(kernel_values.data(), half_kernel_values.data(), kernel_values.size());
        ggml_backend_tensor_set(half_kernel, half_kernel_values.data(), 0,
                                half_kernel_values.size() * sizeof(ggml_fp16_t));
        const float image_value = 1.0003f; // FP16 staging rounds this to one.
        sam::internal::upload(conv_image, std::vector<float>(16, image_value), stats);
        sam::internal::upload(weights, {1, 2, 3, 4}, stats);
        sam::internal::upload(image, {10, 20}, stats);
        sam::internal::upload(endpoints, {0, 1}, stats);
        sam::internal::upload(first, {7, 7, 7, 7}, stats);
        sam::internal::upload(second, {9, 9, 9, 9}, stats);
        sam::internal::upload(presence, {11, 11, 11, 11}, stats);
        sam::internal::sam3::initialize_detector_zero_inputs(graph.context(), stats);
        graph.compute();

        for (std::size_t index = 0; index < convolutions.size(); ++index) {
            const bool padded = index % 2 == 1;
            const int width = padded ? 4 : 1;
            const auto values = sam::internal::download(convolutions[index], stats);
            if (values.size() != static_cast<std::size_t>(width * width)) {
                throw std::runtime_error("Convolution output shape regression");
            }
            for (int y = 0; y < width; ++y) {
                for (int x = 0; x < width; ++x) {
                    double expected = 0;
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            const int iy = y + ky - (padded ? 1 : 0);
                            const int ix = x + kx - (padded ? 1 : 0);
                            if (ix >= 0 && ix < 4 && iy >= 0 && iy < 4) {
                                expected += kernel_values[ky * 3 + kx] * static_cast<double>(image_value);
                            }
                        }
                    }
                    const float actual = values[y * width + x];
                    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-5) {
                        throw std::runtime_error("Convolution narrowed FP32 activations or changed padding layout");
                    }
                }
            }
        }

        const auto logits = sam::internal::download(inverse_sigmoid, stats);
        if (logits.size() != 2 || !std::isfinite(logits[0]) || !std::isfinite(logits[1]) ||
            std::abs(logits[0] + std::log(1000.0f)) > 1e-5f ||
            std::abs(logits[1] - std::log(1000.0f)) > 1e-5f) {
            throw std::runtime_error("Inverse sigmoid endpoints differ from official clamping: " +
                                     std::to_string(logits.at(0)) + ", " + std::to_string(logits.at(1)));
        }

        const auto actual = sam::internal::download(deconv, stats);
        const std::vector<float> expected = {10, 20, 20, 40, 30, 40, 60, 80};
        if (actual.size() != expected.size()) throw std::runtime_error("Deconvolution shape mismatch");
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-6f) {
                throw std::runtime_error("FP32 2x2 deconvolution layout regression");
            }
        }
        for (const float value : sam::internal::download(zero_sum, stats)) {
            if (value != 0) throw std::runtime_error("Repeated-name presence inputs were not all initialized");
        }
        if (!stats.cpu_nodes || !stats.graph_partitions) throw std::runtime_error("Graph was not executed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "graph check: " << error.what() << '\n';
        return 1;
    }
}
