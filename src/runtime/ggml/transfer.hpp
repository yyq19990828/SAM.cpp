#ifndef SAM_CPP_SRC_RUNTIME_GGML_TRANSFER_HPP
#define SAM_CPP_SRC_RUNTIME_GGML_TRANSFER_HPP

#include "runtime.hpp"
#include <chrono>
#include <cstddef>

namespace sam::internal {

// Timing and callbacks are opt-in. Observe the synchronous tensor transfer API,
// without including host-vector allocation or validation in its duration.
template<class Operation>
void observe_transfer(GgmlRuntime* runtime, ggml_tensor* tensor, const char* direction,
                      std::size_t bytes, Operation&& operation) {
    const auto observer = runtime ? runtime->graph_observer() : nullptr;
    if (!observer) {
        operation();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    operation();
    const auto milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    observer->transferred(tensor, direction, bytes, milliseconds);
}

} // namespace sam::internal

#endif // SAM_CPP_SRC_RUNTIME_GGML_TRANSFER_HPP
