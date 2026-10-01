#ifndef SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKENDS_CPU_HPP
#define SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKENDS_CPU_HPP

#include "../backend.hpp"
#include "ggml-backend.h"
#include <stdexcept>

namespace sam::internal {

inline BackendDriver make_cpu_backend(int threads) {
    BackendDriver driver{Backend::Cpu, BackendPtr{}, true, &RuntimeStats::cpu_nodes};
    auto* cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_device) throw std::runtime_error("GGML CPU backend is unavailable");
    driver.handle.reset(ggml_backend_dev_init(cpu_device, nullptr));
    if (!driver.handle) throw std::runtime_error("GGML CPU backend initialization failed");
    auto* registration = ggml_backend_dev_backend_reg(cpu_device);
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(registration, "ggml_backend_set_n_threads"));
    if (!set_threads) throw std::runtime_error("GGML CPU backend lacks thread configuration");
    set_threads(driver.handle.get(), threads);
    return driver;
}

} // namespace sam::internal

#endif // SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKENDS_CPU_HPP
