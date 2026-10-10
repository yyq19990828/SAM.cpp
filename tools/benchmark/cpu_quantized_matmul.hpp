#ifndef SAM_CPP_TOOLS_BENCHMARK_CPU_QUANTIZED_MATMUL_HPP
#define SAM_CPP_TOOLS_BENCHMARK_CPU_QUANTIZED_MATMUL_HPP

#include <runtime/ggml.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam_cpu_probe {

struct Fixture {
    ggml_type type;
    std::int64_t width, rows, columns;
    bool qkv = false, strided_rhs = false;
    std::string scenario;
    std::vector<float> source, rhs, decoded_weights, decoded_rhs;
    std::vector<char> packed, packed_rhs;

    Fixture(ggml_type format, std::int64_t k, std::int64_t m, std::int64_t n,
            std::string pattern = "ordinary", bool views = false, bool strided = false)
        : type(format), width(k), rows(m), columns(n), qkv(views), strided_rhs(strided), scenario(std::move(pattern)) {
        if (k < 32 || k > 8192 || m < 1 || m > 4096 || n < 1 || n > 128 ||
            k * m > 8 * 1024 * 1024 || (views && m % 3) || k % ggml_blck_size(type))
            throw std::invalid_argument("CPU fixture shape exceeds its bounded scope");
        sam::internal::cpu_quantized_rhs_type(type);
        source.resize(static_cast<std::size_t>(k * m));
        rhs.resize(static_cast<std::size_t>(k * n));
        for (std::int64_t row = 0; row < m; ++row)
            for (std::int64_t i = 0; i < k; ++i)
                source[row * k + i] = row % 31 == 0 ? 0.f : static_cast<float>((row * 17 + i * 13 + 7) % 101 - 50) * .007f;
        for (std::int64_t column = 0; column < n; ++column)
            for (std::int64_t i = 0; i < k; ++i)
                rhs[column * k + i] = scenario == "zero" || (n > 1 && column == 1) ? 0.f :
                    static_cast<float>((column * 19 + i * 7 + 3) % 79 - 39) * .009f;
        if (scenario == "outlier") {
            for (std::int64_t column = 0; column < n; ++column) {
                rhs[column * k + 3] = 17.1f;
                rhs[column * k + 41] = -.0013f;
            }
        }
        const auto* traits = ggml_get_type_traits(type);
        packed.resize(static_cast<std::size_t>(m) * ggml_row_size(type, k));
        traits->from_float_ref(source.data(), packed.data(), k * m);
        decoded_weights.resize(source.size());
        traits->to_float(packed.data(), decoded_weights.data(), k * m);
        const auto rhs_type = sam::internal::cpu_quantized_rhs_type(type);
        packed_rhs.resize(static_cast<std::size_t>(n) * ggml_row_size(rhs_type, k));
        pack_rhs(packed_rhs);
        decoded_rhs.resize(rhs.size());
        decode_rhs();
    }

    void decode_rhs() {
        const auto rhs_type = sam::internal::cpu_quantized_rhs_type(type);
        if (rhs_type == GGML_TYPE_Q8_0) {
            const auto decode = ggml_get_type_traits(rhs_type)->to_float;
            if (!decode) throw std::runtime_error("CPU Q8_0 RHS decoder unavailable");
            decode(packed_rhs.data(), decoded_rhs.data(), width * columns);
            return;
        }
        // Q8_K is kernel scratch only: GGML's public type traits deliberately
        // have no to_float callback. Decode its pinned wire layout independently:
        // F32 d, 256 signed codes, 16 int16 block sums (not needed for values).
        constexpr std::size_t block_elements = 256;
        constexpr std::size_t block_bytes = sizeof(float) + block_elements + 16 * sizeof(std::int16_t);
        if (ggml_blck_size(rhs_type) != block_elements || ggml_type_size(rhs_type) != block_bytes)
            throw std::runtime_error("CPU Q8_K scratch layout differs from the pinned reference contract");
        for (std::size_t block = 0; block < decoded_rhs.size() / block_elements; ++block) {
            const auto* bytes = packed_rhs.data() + block * block_bytes;
            float scale;
            std::memcpy(&scale, bytes, sizeof(scale));
            for (std::size_t i = 0; i < block_elements; ++i) {
                std::int8_t code;
                std::memcpy(&code, bytes + sizeof(scale) + i, sizeof(code));
                decoded_rhs[block * block_elements + i] = scale * code;
            }
        }
    }

    void pack_rhs(std::vector<char>& destination) const {
        const auto type = sam::internal::cpu_quantized_rhs_type(this->type);
        const auto row_bytes = ggml_row_size(type, width);
        if (destination.size() != row_bytes * static_cast<std::size_t>(columns))
            throw std::invalid_argument("CPU RHS packing destination size differs");
        const auto pack = ggml_get_type_traits_cpu(type)->from_float;
        for (std::int64_t row = 0; row < columns; ++row)
            pack(rhs.data() + row * width, destination.data() + row * row_bytes, width);
    }
};

inline sam::BackendOptions options(sam::CpuComputeMode mode, int threads) {
    sam::BackendOptions result{sam::Backend::Cpu, threads};
    result.cpu_compute = mode;
    return result;
}

class Matmul {
public:
    Matmul(const Fixture& fixture, sam::CpuComputeMode mode, int threads,
           std::shared_ptr<sam::internal::GraphObserver> observer = {})
        : fixture_(fixture), native_(mode == sam::CpuComputeMode::NativeQuantized),
          runtime_(options(mode, threads), false, true), context_(sam::internal::make_context(4)) {
        runtime_.set_graph_observer(std::move(observer));
        weight_ = ggml_new_tensor_2d(context_.get(), fixture.type, fixture.width, fixture.rows);
        ggml_set_name(weight_, "synthetic_packed_weight");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(), runtime_.weights_backend()));
        if (!buffer_) throw std::runtime_error("CPU matmul fixture allocation failed");
        ggml_backend_tensor_set(weight_, fixture.packed.data(), 0, fixture.packed.size());
        execution_ = std::make_unique<sam::internal::GraphExecution>(runtime_, 32, stats_);
        auto* ctx = execution_->context();
        input_ = sam::internal::input_tensor(ctx, "synthetic_f32_activation", fixture.width,
                                           fixture.columns * (fixture.strided_rhs ? 2 : 1));
        auto* rhs = fixture.strided_rhs ? ggml_view_2d(ctx, input_, fixture.width, fixture.columns, 2 * input_->nb[1], 0) : input_;
        const auto parts = fixture.qkv ? 3 : 1;
        const auto rows = fixture.rows / parts;
        for (int part = 0; part < parts; ++part) {
            auto* weight = fixture.qkv ? ggml_view_2d(ctx, weight_, fixture.width, rows, weight_->nb[1], part * rows * weight_->nb[1]) : weight_;
            auto* output = ggml_mul_mat(ctx, weight, rhs);
            ggml_set_name(output, ("synthetic_matmul_" + std::to_string(part)).c_str());
            outputs_.push_back(output);
            execution_->output(output);
        }
        execution_->allocate();
        upload_rhs(fixture.rhs);
        for (auto* output : outputs_) {
            if (output->src[0]->type != (native_ ? fixture.type : GGML_TYPE_F32) || output->src[1]->type != GGML_TYPE_F32)
                throw std::runtime_error("CPU matmul did not retain its requested graph operand types");
        }
    }

    void compute() { execution_->compute(); }
    void upload_rhs(const std::vector<float>& values) {
        if (values.size() != fixture_.rhs.size()) throw std::invalid_argument("CPU fixture RHS size differs");
        std::vector<float> uploaded(static_cast<std::size_t>(ggml_nelements(input_)), 91.f);
        for (std::int64_t column = 0; column < fixture_.columns; ++column)
            std::copy_n(values.data() + column * fixture_.width, fixture_.width,
                        uploaded.data() + column * fixture_.width * (fixture_.strided_rhs ? 2 : 1));
        sam::internal::upload(input_, uploaded, stats_, &runtime_);
    }
    void check_resident_weights() const {
        std::vector<char> actual(fixture_.packed.size());
        ggml_backend_tensor_get(weight_, actual.data(), 0, actual.size());
        if (actual != fixture_.packed) throw std::runtime_error("CPU compute changed resident packed weights");
    }
    const sam::RuntimeStats& stats() const { return stats_; }
    const sam::internal::GraphDiagnostics& diagnostics() const { return execution_->diagnostics(); }
    const char* profile() const { return runtime_.arithmetic_profile(); }
    ggml_type left_type() const { return outputs_.front()->src[0]->type; }
    void set_observer(std::shared_ptr<sam::internal::GraphObserver> observer) { runtime_.set_graph_observer(std::move(observer)); }

    std::vector<float> read() {
        std::vector<float> values(static_cast<std::size_t>(fixture_.rows * fixture_.columns));
        const auto rows = fixture_.rows / static_cast<std::int64_t>(outputs_.size());
        for (std::size_t part = 0; part < outputs_.size(); ++part) {
            const auto data = sam::internal::download(outputs_[part], stats_, &runtime_);
            for (std::int64_t column = 0; column < fixture_.columns; ++column)
                std::copy_n(data.data() + column * rows, rows,
                            values.data() + column * fixture_.rows + part * rows);
        }
        if (stats_.cuda_nodes || stats_.metal_nodes || !stats_.cpu_nodes)
            throw std::runtime_error("CPU fixture escaped CPU execution");
        return values;
    }

private:
    const Fixture& fixture_;
    bool native_;
    sam::internal::GgmlRuntime runtime_;
    sam::RuntimeStats stats_;
    sam::internal::ContextPtr context_;
    sam::internal::BufferPtr buffer_;
    ggml_tensor* weight_ = nullptr;
    ggml_tensor* input_ = nullptr;
    std::vector<ggml_tensor*> outputs_;
    std::unique_ptr<sam::internal::GraphExecution> execution_;
};

struct Accuracy {
    std::size_t checked = 0;
    double max_effective_error = 0, max_f32_activation_difference = 0, squared_difference = 0;
};

inline Accuracy validate(const Fixture& fixture, const std::vector<float>& actual, bool native, bool full = true) {
    if (actual.size() != static_cast<std::size_t>(fixture.rows * fixture.columns))
        throw std::runtime_error("CPU matmul output shape differs");
    Accuracy result;
    for (std::int64_t column = 0; column < fixture.columns; ++column) {
        if (!full && column != 0 && column != 1 && column != fixture.columns / 2 && column != fixture.columns - 1) continue;
        for (std::int64_t row = 0; row < fixture.rows; ++row) {
            if (!full && row != 0 && row != 1 && row != fixture.rows / 2 && row != fixture.rows - 1) continue;
            double effective = 0, original_rhs = 0;
            for (std::int64_t k = 0; k < fixture.width; ++k) {
                const double weight = fixture.decoded_weights[row * fixture.width + k];
                const auto index = column * fixture.width + k;
                effective += weight * (native ? fixture.decoded_rhs[index] : fixture.rhs[index]);
                original_rhs += weight * fixture.rhs[index];
            }
            const auto value = actual[column * fixture.rows + row];
            const auto error = std::abs(value - effective);
            if (!std::isfinite(value) || error > 5e-4 + 5e-5 * std::abs(effective))
                throw std::runtime_error("CPU matmul differs from independent decoded scalar reference: " + std::to_string(error));
            const auto difference = std::abs(value - original_rhs);
            result.max_effective_error = std::max(result.max_effective_error, error);
            result.max_f32_activation_difference = std::max(result.max_f32_activation_difference, difference);
            result.squared_difference += difference * difference;
            ++result.checked;
        }
    }
    return result;
}

} // namespace sam_cpu_probe

#endif // SAM_CPP_TOOLS_BENCHMARK_CPU_QUANTIZED_MATMUL_HPP
