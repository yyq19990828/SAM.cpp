#include <runtime/ggml.hpp>
#include "backend_test_support.hpp"

#include <cmath>
#include <iostream>

namespace {

void check_matrix(ggml_backend_t backend, ggml_type left_type, int k, int m, int n) {
    auto context = sam::internal::make_context(16, 64);
    auto* a = ggml_new_tensor_2d(context.get(), left_type, k, m);
    auto* b = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, k, n);
    auto* precise = ggml_mul_mat(context.get(), a, b);
    if (!ggml_prec_set_acc(precise, GGML_PREC_F32)) {
        throw std::runtime_error("Matrix precision request was rejected");
    }
    auto* default_precision = ggml_mul_mat(context.get(), a, b);
    if (!ggml_backend_supports_op(backend, precise) || !ggml_backend_supports_op(backend, default_precision)) {
        throw std::runtime_error("Selected backend does not support the precision probe");
    }
    auto* graph = ggml_new_graph_custom(context.get(), 64, false);
    ggml_build_forward_expand(graph, precise);
    ggml_build_forward_expand(graph, default_precision);
    sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) throw std::runtime_error("Could not allocate precision probe buffers");
    const float value = 1.0003f;
    const std::vector<float> a_values(k * m, value);
    const std::vector<float> b_values(k * n, value);
    double stored_a_value = value;
    if (left_type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half_values(a_values.size());
        ggml_fp32_to_fp16_row(a_values.data(), half_values.data(), static_cast<int64_t>(a_values.size()));
        ggml_backend_tensor_set(a, half_values.data(), 0, half_values.size() * sizeof(ggml_fp16_t));
        stored_a_value = 1.0; // 1.0003f rounds to exactly one in the stored FP16 operand.
    } else {
        ggml_backend_tensor_set(a, a_values.data(), 0, a_values.size() * sizeof(float));
    }
    ggml_backend_tensor_set(b, b_values.data(), 0, b_values.size() * sizeof(float));
    // Direct backend execution prevents a GPU probe from falling back to CPU.
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Precision probe graph execution failed");
    }
    std::vector<float> precise_values(m * n), default_values(m * n);
    ggml_backend_tensor_get(precise, precise_values.data(), 0, precise_values.size() * sizeof(float));
    ggml_backend_tensor_get(default_precision, default_values.data(), 0, default_values.size() * sizeof(float));
    const double expected = k * stored_a_value * static_cast<double>(value);
    double precise_error = 0;
    double default_error = 0;
    for (std::size_t i = 0; i < precise_values.size(); ++i) {
        if (!std::isfinite(precise_values[i]) || !std::isfinite(default_values[i])) {
            throw std::runtime_error("Precision probe produced a non-finite value");
        }
        precise_error = std::max(precise_error, std::abs(precise_values[i] - expected));
        default_error = std::max(default_error, std::abs(default_values[i] - expected));
    }
    // A bound for this arithmetic probe's FP32 summation, not a model tolerance.
    // Half staging loses about 0.154 (F32/F32) or 0.077 (F16/F32).
    if (precise_error > 0.002) {
        throw std::runtime_error(std::string(ggml_backend_name(backend)) +
            " does not honor GGML_PREC_F32; maximum absolute error " + std::to_string(precise_error));
    }
    std::cout << ggml_backend_name(backend) << ' ' << ggml_type_name(left_type)
              << ' ' << k << 'x' << m << 'x' << n << ": precise max error=" << precise_error
              << ", default max error=" << default_error << '\n';
}

void check_attention(ggml_backend_t backend, int dimensions, int queries, int keys) {
    auto context = sam::internal::make_context(16, 64);
    auto* q = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, dimensions, queries, 1, 1);
    auto* k = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, dimensions, keys, 1, 1);
    auto* v = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, dimensions, keys, 1, 1);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dimensions));
    auto* precise = ggml_flash_attn_ext(context.get(), q, k, v, nullptr, scale, 0, 0);
    if (!ggml_prec_set_acc(precise, GGML_PREC_F32)) {
        throw std::runtime_error("Attention precision request was rejected");
    }
    auto* default_precision = ggml_flash_attn_ext(context.get(), q, k, v, nullptr, scale, 0, 0);
    if (!ggml_backend_supports_op(backend, precise)) {
        throw std::runtime_error("Selected backend does not support the attention precision probe");
    }
    const bool default_supported = ggml_backend_supports_op(backend, default_precision);
    auto* graph = ggml_new_graph_custom(context.get(), 64, false);
    ggml_build_forward_expand(graph, precise);
    if (default_supported) ggml_build_forward_expand(graph, default_precision);
    sam::internal::BufferPtr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) throw std::runtime_error("Could not allocate attention precision buffers");
    const std::vector<float> q_values(dimensions * queries, 1.0003f);
    std::vector<float> k_values(dimensions * keys), v_values(dimensions * keys);
    for (int key = 0; key < keys; ++key) {
        for (int dimension = 0; dimension < dimensions; ++dimension) {
            k_values[key * dimensions + dimension] = key < keys / 2 ? 0.0625f : -0.0625f;
            v_values[key * dimensions + dimension] = key < keys / 2 ? 1.0f : -1.0f;
        }
    }
    ggml_backend_tensor_set(q, q_values.data(), 0, q_values.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_values.data(), 0, k_values.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_values.data(), 0, v_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Attention precision execution failed");
    }
    std::vector<float> precise_values(dimensions * queries), default_values(dimensions * queries);
    ggml_backend_tensor_get(precise, precise_values.data(), 0, precise_values.size() * sizeof(float));
    if (default_supported)
        ggml_backend_tensor_get(default_precision, default_values.data(), 0, default_values.size() * sizeof(float));
    const double expected = std::tanh(dimensions * 0.0625 * scale * static_cast<double>(q_values[0]));
    double precise_error = 0, default_error = 0;
    for (std::size_t i = 0; i < precise_values.size(); ++i) {
        if (!std::isfinite(precise_values[i]) || (default_supported && !std::isfinite(default_values[i]))) {
            throw std::runtime_error("Attention precision produced a non-finite value");
        }
        precise_error = std::max(precise_error, std::abs(precise_values[i] - expected));
        if (default_supported) default_error = std::max(default_error, std::abs(default_values[i] - expected));
    }
    // Allow FP32 exp and key reduction order differences. This 1e-5 bound
    // still rejects the measured 1.18e-4 half-Q loss; model gates are unchanged.
    if (precise_error > 1e-5) throw std::runtime_error("Attention lost FP32 query precision");
    std::cout << ggml_backend_name(backend) << " Q" << queries << " D" << dimensions << " K" << keys
              << " attention: precise max error=" << precise_error
              << ", default=" << (default_supported ? std::to_string(default_error) : "unsupported") << '\n';
}

void check_precision(sam::Backend selected) {
    sam::internal::GgmlRuntime runtime({selected, 1}, false);
    if (runtime.backend() != selected) throw std::runtime_error("Precision probe used a different backend");
    auto* backend = runtime.weights_backend();
    check_matrix(backend, GGML_TYPE_F32, 256, 256, 32);
    check_matrix(backend, GGML_TYPE_F32, 257, 65, 33);
    // CPU weights are promoted to F32; its native F16 dot path rounds the RHS.
    if (selected == sam::Backend::Metal || selected == sam::Backend::Cuda)
        check_matrix(backend, GGML_TYPE_F16, 257, 65, 33);
    check_attention(backend, 64, 1, 128);
    check_attention(backend, 32, 1, 130);
    check_attention(backend, 64, 33, 130);
    if (selected == sam::Backend::Cuda) {
        check_attention(backend, 32, 129, 5184);
        check_attention(backend, 64, 257, 576);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (sam::test::cuda_requested(argc, argv)) {
            if (!sam::test::cuda_available()) return 77;
            check_precision(sam::Backend::Cuda);
            return 0;
        }
        check_precision(sam::Backend::Cpu);
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto* device = ggml_backend_dev_get(i);
            if (std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) == "MTL") {
                check_precision(sam::Backend::Metal);
                break;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "precision check: " << error.what() << '\n';
        return 1;
    }
}
