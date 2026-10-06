#ifndef SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_RUNTIME_HPP
#define SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_RUNTIME_HPP

#include "backend.hpp"
#include "backends/cpu.hpp"
#include "backends/metal.hpp"
#include "backends/cuda.hpp"
#include "sam/internal/input_validation.hpp"
#include "sam/types.hpp"
#include "ggml-backend.h"
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sam::internal {

class GgmlRuntime {
public:
    GgmlRuntime(BackendOptions options, bool fp32, bool quantized_profile = false) {
        validate_backend_options(options);
        drivers_.push_back(make_cpu_backend(options.threads));
        if (options.backend == Backend::Cuda) {
            selected_ = drivers_.size();
            drivers_.push_back(make_cuda_backend(options.cuda_device));
        }
        if (options.backend == Backend::Metal ||
            (!fp32 && !quantized_profile && options.backend == Backend::Auto)) {
            auto metal = make_metal_backend();
            if (!metal.handle && options.backend == Backend::Metal)
                throw std::runtime_error("requested GGML Metal backend is unavailable or failed initialization");
            if (metal.handle) {
                selected_ = drivers_.size();
                drivers_.push_back(std::move(metal));
            }
        }
        if (drivers_[selected_].kind == Backend::Cpu) {
            auto blas = make_cpu_blas_backend(options.threads);
            if (blas.handle) {
                drivers_.push_back(std::move(blas));
                backends_.push_back(drivers_.back().handle.get());
            }
        }
        backends_.push_back(drivers_[selected_].handle.get());
        for (std::size_t i = 0; i < drivers_.size(); ++i)
            if (i != selected_ && drivers_[i].node_counter != &RuntimeStats::blas_nodes)
                backends_.push_back(drivers_[i].handle.get());
        quantized_profile_ = quantized_profile;
        quantized_native_metal_only_ = quantized_profile && options.backend == Backend::Metal;
    }
    Backend backend() const { return drivers_[selected_].kind; }
    ggml_backend_t weights_backend() const { return drivers_[selected_].handle.get(); }
    const std::vector<ggml_backend_t>& backends() const { return backends_; }
    bool promote_f16_weights() const { return drivers_[selected_].promote_f16_weights; }
    bool quantized_cpu_f32_weights() const { return quantized_profile_ && backend() == Backend::Cpu; }
    bool quantized_native_metal_only() const { return quantized_native_metal_only_; }
    bool requires_primary_compute() const { return backend() == Backend::Cuda || quantized_native_metal_only_; }
    const std::string& device_name() const { return drivers_[selected_].device_name; }
    int cuda_device() const { return drivers_[selected_].cuda_device; }
    const char* arithmetic_profile() const {
        if (!quantized_profile_) return "";
        if (backend() == Backend::Cuda) return "ggml-quantized-cuda-native-v1";
        return quantized_cpu_f32_weights() ? "ggml-quantized-weights-f32-v1" : "ggml-quantized-native-v1";
    }

    void record_node(ggml_backend_t backend, RuntimeStats& stats) const {
        for (const auto& driver : drivers_) {
            if (driver.handle.get() == backend && driver.node_counter) {
                ++(stats.*driver.node_counter);
                if (driver.node_counter == &RuntimeStats::blas_nodes) ++stats.cpu_nodes;
                return;
            }
        }
        throw std::runtime_error("GGML scheduler assigned a node to an unknown backend");
    }
private:
    // GGML requires CPU last in the backend list; the selected device is first,
    // and CPU remains available for input copies and legacy fallback. Explicit
    // CUDA and quantized Metal require every compute node on the primary device.
    std::vector<BackendDriver> drivers_;
    std::vector<ggml_backend_t> backends_;
    std::size_t selected_ = 0;
    bool quantized_profile_ = false;
    bool quantized_native_metal_only_ = false;
};

} // namespace sam::internal

#endif // SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_RUNTIME_HPP
