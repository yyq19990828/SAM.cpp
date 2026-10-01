#ifndef SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKEND_HPP
#define SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKEND_HPP

#include "resources.hpp"
#include "sam/types.hpp"
#include <cstdint>

namespace sam::internal {

// Each implemented device supplies ownership and initialization-time policies.
// The counter points to its existing public statistics field; scheduling never
// guesses a device kind from an unrecognized GGML handle.
struct BackendDriver {
    Backend kind = Backend::Auto;
    BackendPtr handle;
    bool promote_f16_weights = false;
    std::uint64_t RuntimeStats::* node_counter = nullptr;
};

} // namespace sam::internal

#endif // SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_BACKEND_HPP
