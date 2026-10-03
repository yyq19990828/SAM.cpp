#ifndef SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_RUNTIME_HPP
#define SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_RUNTIME_HPP

#include "backend.hpp"
#include "backends/cpu.hpp"
#include "backends/metal.hpp"
#include "sam/types.hpp"
#include "ggml-backend.h"
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sam::internal {

class GgmlRuntime {
public:
    GgmlRuntime(BackendOptions options, bool fp32) {
        if (options.threads <= 0) throw std::invalid_argument("threads must be positive");
        if (options.backend != Backend::Auto && options.backend != Backend::Cpu && options.backend != Backend::Metal)
            throw std::invalid_argument("invalid backend selection");
        drivers_.push_back(make_cpu_backend(options.threads));
        if (options.backend == Backend::Metal || (!fp32 && options.backend == Backend::Auto)) {
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
    }
    Backend backend() const { return drivers_[selected_].kind; }
    ggml_backend_t weights_backend() const { return drivers_[selected_].handle.get(); }
    const std::vector<ggml_backend_t>& backends() const { return backends_; }
    bool promote_f16_weights() const { return drivers_[selected_].promote_f16_weights; }

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
    // Devices are initialized CPU first. The scheduler receives the selected
    // device first, followed by the other supported devices for operator fallback.
    std::vector<BackendDriver> drivers_;
    std::vector<ggml_backend_t> backends_;
    std::size_t selected_ = 0;
};

} // namespace sam::internal

#endif // SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_RUNTIME_HPP
