#ifndef SAM_CPP_SRC_RUNTIME_GGML_OBSERVER_HPP
#define SAM_CPP_SRC_RUNTIME_GGML_OBSERVER_HPP

#include "ggml.h"
#include "ggml-backend.h"

namespace sam::internal {

// Internal, opt-in diagnostics. Callbacks borrow graph metadata only for the
// duration of the call and must not mutate it or reenter this runtime. An
// observer may synchronize/read tensors, so observed runs are not benchmarks.
class GraphObserver {
public:
    virtual ~GraphObserver() = default;
    virtual void allocated(ggml_context*, ggml_cgraph*, ggml_backend_sched_t) = 0;
    virtual void computed(ggml_context*, ggml_cgraph*, double) {}
};

} // namespace sam::internal

#endif // SAM_CPP_SRC_RUNTIME_GGML_OBSERVER_HPP
