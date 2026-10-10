#ifndef SAM_CPP_SRC_RUNTIME_GGML_OBSERVER_HPP
#define SAM_CPP_SRC_RUNTIME_GGML_OBSERVER_HPP

#include "ggml.h"
#include "ggml-backend.h"
#include <cstddef>

namespace sam::internal {

// Internal, opt-in diagnostics. Callbacks borrow graph metadata only for the
// duration of the call and must not mutate it or reenter this runtime. An
// observer may synchronize/read tensors or install a scheduler evaluation
// callback, so observed runs are not benchmarks.
class GraphObserver {
public:
    virtual ~GraphObserver() = default;
    virtual void allocated(ggml_context*, ggml_cgraph*, ggml_backend_sched_t) = 0;
    virtual void computing(ggml_context*, ggml_cgraph*, ggml_backend_sched_t) {}
    virtual void computed(ggml_context*, ggml_cgraph*, double) {}
    virtual void bound(double, std::size_t, bool) {}
    virtual void transferred(ggml_tensor*, const char*, std::size_t, double) {}
};

} // namespace sam::internal

#endif // SAM_CPP_SRC_RUNTIME_GGML_OBSERVER_HPP
