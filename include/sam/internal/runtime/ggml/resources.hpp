#ifndef SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_RESOURCES_HPP
#define SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_RESOURCES_HPP

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace sam::internal {

struct ContextDeleter { void operator()(ggml_context* p) const { if (p) ggml_free(p); } };
struct BufferDeleter { void operator()(ggml_backend_buffer* p) const { if (p) ggml_backend_buffer_free(p); } };
struct BackendDeleter { void operator()(ggml_backend* p) const { if (p) ggml_backend_free(p); } };
struct SchedulerDeleter { void operator()(ggml_backend_sched* p) const { if (p) ggml_backend_sched_free(p); } };
using ContextPtr = std::unique_ptr<ggml_context, ContextDeleter>;
using BufferPtr = std::unique_ptr<ggml_backend_buffer, BufferDeleter>;
using BackendPtr = std::unique_ptr<ggml_backend, BackendDeleter>;
using SchedulerPtr = std::unique_ptr<ggml_backend_sched, SchedulerDeleter>;

inline ContextPtr make_context(std::size_t tensor_count, std::size_t graph_size = 0) {
    const std::size_t size = ggml_tensor_overhead() * tensor_count +
        (graph_size ? ggml_graph_overhead_custom(graph_size, false) : 0) + 1024;
    ContextPtr context(ggml_init({size, nullptr, true}));
    if (!context) throw std::runtime_error("failed to create GGML metadata context");
    return context;
}

} // namespace sam::internal

#endif // SAM_CPP_SAM_INTERNAL_RUNTIME_GGML_RESOURCES_HPP
