#include <models/sam3/vision.hpp>
#include <sam/internal/runtime/ggml.hpp>
#include "backend_test_support.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int head_dim = 64;
constexpr int half_head_dim = head_dim / 2;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

enum class FrequencyPattern { arbitrary, identity, quarter_turn, trigonometric };

std::vector<float> make_input(int tokens, int heads, int batch, float salt) {
    const auto head_batches = heads * batch;
    std::vector<float> values(static_cast<std::size_t>(head_dim) * tokens * head_batches);
    for (int head_batch = 0; head_batch < head_batches; ++head_batch) {
        for (int token = 0; token < tokens; ++token) {
            for (int pair = 0; pair < half_head_dim; ++pair) {
                const auto index = (static_cast<std::size_t>(head_batch) * tokens + token) * head_dim + pair * 2;
                const double phase = 0.17 * (head_batch + 1) + 0.011 * (token + 1) +
                    0.031 * (pair + 1) + salt;
                values[index] = static_cast<float>(0.1234567 + 0.7 * std::sin(phase) + 0.01 * head_batch);
                values[index + 1] = static_cast<float>(-0.7654321 + 0.6 * std::cos(phase * 0.73) - 0.005 * head_batch);
            }
        }
    }
    values[0] = 0.1234567f + salt;
    values[1] = -0.7654321f - salt;
    return values;
}

std::vector<float> make_frequencies(int tokens, FrequencyPattern pattern, float salt) {
    std::vector<float> values(static_cast<std::size_t>(2) * half_head_dim * tokens);
    for (int token = 0; token < tokens; ++token) {
        for (int pair = 0; pair < half_head_dim; ++pair) {
            const auto index = (static_cast<std::size_t>(token) * half_head_dim + pair) * 2;
            if (pattern == FrequencyPattern::identity) {
                values[index] = 1.0f;
                values[index + 1] = 0.0f;
            } else if (pattern == FrequencyPattern::quarter_turn) {
                values[index] = 0.0f;
                values[index + 1] = 1.0f;
            } else if (pattern == FrequencyPattern::arbitrary) {
                const double phase = 0.19 * (pair + 1) + 0.027 * (token + 1) +
                    0.013 * ((pair + 3) * (token + 5) % 19) + salt;
                values[index] = static_cast<float>(0.71 + 0.13 * std::cos(phase));
                values[index + 1] = static_cast<float>(-0.23 + 0.17 * std::sin(phase * 0.61));
            } else {
                const double phase = 0.21 * (pair + 1) + 0.0007 * (token + 1) +
                    0.003 * ((pair + 2) * (token % 97 + 1)) + salt;
                values[index] = static_cast<float>(std::cos(phase));
                values[index + 1] = static_cast<float>(std::sin(phase));
            }
        }
    }
    return values;
}

void check_scalar_result(const std::vector<float>& input, const std::vector<float>& frequencies,
                         const std::vector<float>& actual, int tokens, int heads, int batch) {
    require(actual.size() == input.size(), "RoPE output changed the input element count");
    const int head_batches = heads * batch;
    for (int head_batch = 0; head_batch < head_batches; ++head_batch) {
        for (int token = 0; token < tokens; ++token) {
            for (int pair = 0; pair < half_head_dim; ++pair) {
                const auto x_index = (static_cast<std::size_t>(head_batch) * tokens + token) * head_dim + pair * 2;
                const auto frequency_index = (static_cast<std::size_t>(token) * half_head_dim + pair) * 2;
                const double real = input[x_index];
                const double imag = input[x_index + 1];
                const double cosine = frequencies[frequency_index];
                const double sine = frequencies[frequency_index + 1];
                const double expected_real = real * cosine - imag * sine;
                const double expected_imag = real * sine + imag * cosine;
                const double tolerance = 2e-6 * (1.0 + std::max(std::abs(expected_real), std::abs(expected_imag)));
                if (!std::isfinite(actual[x_index]) || !std::isfinite(actual[x_index + 1]) ||
                    std::abs(actual[x_index] - expected_real) > tolerance ||
                    std::abs(actual[x_index + 1] - expected_imag) > tolerance) {
                    throw std::runtime_error("RoPE differs from the independent scalar complex formula");
                }
            }
        }
    }
}

void check_frequency_variation(const std::vector<float>& frequencies) {
    const auto at = [&](int token, int pair, int component) {
        return frequencies[(static_cast<std::size_t>(token) * half_head_dim + pair) * 2 + component];
    };
    require(at(0, 0, 0) != at(0, 1, 0) && at(0, 0, 1) != at(0, 1, 1),
            "RoPE test frequencies do not vary across pairs");
    require(at(0, 0, 0) != at(1, 0, 0) && at(0, 0, 1) != at(1, 0, 1),
            "RoPE test frequencies do not vary across tokens");
}

void check_reused_shape(sam::internal::GgmlRuntime& runtime, int tokens, int heads, int batch,
                        const std::vector<FrequencyPattern>& patterns) {
    sam::RuntimeStats stats;
    auto input_context = sam::internal::make_context(2);
    auto* input = ggml_new_tensor_3d(input_context.get(), GGML_TYPE_F32, head_dim, tokens, heads * batch);
    auto* frequencies = ggml_new_tensor_3d(input_context.get(), GGML_TYPE_F32, 2, half_head_dim, tokens);
    ggml_set_name(input, "rope_input");
    ggml_set_name(frequencies, "rope_frequencies");
    ggml_set_input(input);
    ggml_set_input(frequencies);
    sam::internal::BufferPtr input_buffer(
        ggml_backend_alloc_ctx_tensors(input_context.get(), runtime.weights_backend()));
    require(static_cast<bool>(input_buffer), "could not allocate external RoPE inputs on the selected backend");
    ggml_backend_buffer_set_usage(input_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    sam::internal::GraphExecution graph(runtime, 64, stats);
    auto* output = sam::internal::sam3::sam3_apply_rope(graph.context(), input, frequencies);
    graph.output(output);
    graph.allocate();

    std::vector<float> previous;
    for (std::size_t pass = 0; pass < patterns.size(); ++pass) {
        const float salt = static_cast<float>(pass) * 0.137f;
        auto input_values = make_input(tokens, heads, batch, salt);
        auto frequency_values = make_frequencies(tokens, patterns[pass], salt);
        if (patterns[pass] == FrequencyPattern::arbitrary || patterns[pass] == FrequencyPattern::trigonometric)
            check_frequency_variation(frequency_values);
        sam::internal::upload(input, input_values, stats);
        sam::internal::upload(frequencies, frequency_values, stats);
        graph.compute();
        auto actual = sam::internal::download(output, stats);
        check_scalar_result(input_values, frequency_values, actual, tokens, heads, batch);
        if (!previous.empty()) {
            bool changed = false;
            for (std::size_t i = 0; i < actual.size(); ++i)
                changed = changed || std::abs(actual[i] - previous[i]) > 1e-5f;
            require(changed, "fresh same-shape RoPE inputs did not change the cached graph output");
        }
        previous = std::move(actual);
    }
    if (runtime.backend() == sam::Backend::Metal) {
        require(stats.metal_nodes > 0, "RoPE graph did not execute compute nodes on Metal");
        require(stats.cpu_nodes == 0, "RoPE graph fell back to CPU with Metal-resident inputs");
    } else if (runtime.backend() == sam::Backend::Cuda) {
        require(stats.cuda_nodes > 0, "RoPE graph did not execute compute nodes on CUDA");
        require(stats.cpu_nodes == 0 && stats.metal_nodes == 0, "CUDA RoPE graph used another backend");
    } else {
        require(stats.cpu_nodes > 0, "RoPE graph did not execute compute nodes on CPU");
    }
}

sam::Backend parse_backend(int argc, char** argv) {
    if (argc > 2) throw std::invalid_argument("usage: test_rope [cpu|metal|cuda]");
    if (argc == 1 || std::string(argv[1]) == "cpu") return sam::Backend::Cpu;
    if (std::string(argv[1]) == "metal") return sam::Backend::Metal;
    if (std::string(argv[1]) == "cuda") return sam::Backend::Cuda;
    throw std::invalid_argument("usage: test_rope [cpu|metal|cuda]");
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto selected = parse_backend(argc, argv);
        if (selected == sam::Backend::Cuda && !sam::test::cuda_available()) return 77;
        sam::internal::GgmlRuntime runtime({selected, 1}, false);
        check_reused_shape(runtime, 1, 1, 1, {
            FrequencyPattern::identity, FrequencyPattern::quarter_turn,
        });
        // Odd/tail token count, multiple heads and batches, changing frequency
        // tables and inputs on one cached graph, plus exact identity/quarter-turn.
        check_reused_shape(runtime, 37, 3, 2, {
            FrequencyPattern::arbitrary,
            FrequencyPattern::arbitrary,
            FrequencyPattern::identity,
            FrequencyPattern::quarter_turn,
        });
        // Representative SAM 3 window-attention token count.
        check_reused_shape(runtime, 576, 16, 1, {FrequencyPattern::trigonometric});
        // Representative SAM 3 global-attention token count.
        check_reused_shape(runtime, 5184, 16, 1, {FrequencyPattern::trigonometric});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RoPE check: " << error.what() << '\n';
        return 1;
    }
}
