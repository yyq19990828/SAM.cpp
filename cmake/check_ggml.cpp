#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <gguf.h>

#include <type_traits>

// Check critical scheduler signatures and the required operator symbol surface.
// Runtime graph execution is verified separately; these declarations do not
// prove backend support or numerical compatibility.
using SchedulerConstructor = ggml_backend_sched_t (*)(ggml_backend_t*, ggml_backend_buffer_type_t*,
                                                       int, size_t, bool, bool);
using SchedulerCompute = ggml_status (*)(ggml_backend_sched_t, ggml_cgraph*);
using TensorBackend = ggml_backend_t (*)(ggml_backend_sched_t, ggml_tensor*);
static_assert(std::is_same_v<decltype(&ggml_backend_sched_new), SchedulerConstructor>,
              "SAM requires the pinned six-argument GGML scheduler constructor");
static_assert(std::is_same_v<decltype(&ggml_backend_sched_graph_compute), SchedulerCompute>);
static_assert(std::is_same_v<decltype(&ggml_backend_sched_get_tensor_backend), TensorBackend>);
static_assert(std::is_pointer_v<decltype(&ggml_im2col)>);
static_assert(std::is_pointer_v<decltype(&ggml_flash_attn_ext)>);
using PrecisionSetter = bool (*)(ggml_tensor*, ggml_prec);
static_assert(std::is_same_v<decltype(&ggml_prec_set_acc), PrecisionSetter>);
static_assert(std::is_pointer_v<decltype(&ggml_flash_attn_ext_get_prec)>);
static_assert(std::is_pointer_v<decltype(&ggml_upscale)>);
static_assert(std::is_pointer_v<decltype(&ggml_backend_load_all)>);
static_assert(std::is_pointer_v<decltype(&ggml_backend_dev_by_type)>);
static_assert(std::is_pointer_v<decltype(&ggml_backend_dev_init)>);
static_assert(std::is_pointer_v<decltype(&ggml_backend_cpu_set_n_threads)>);

// The reader relies on bounded callback parsing, no-allocation metadata and
// exact typed getters. Its canonical metadata check also uses GGUF copy/writer
// primitives; require their actual signatures from caller-owned GGML headers.
using GgufReaderCallback = size_t (*)(void*, void*, uint64_t, size_t);
using GgufCallbackInitializer = gguf_context* (*)(gguf_reader_callback_t, void*, size_t,
                                                uint64_t, gguf_init_params);
using GgufSizeGetter = size_t (*)(const gguf_context*);
using GgufTypeGetter = gguf_type (*)(const gguf_context*, int64_t);
using GgufStringGetter = const char* (*)(const gguf_context*, int64_t);
using GgufTensorSizeGetter = size_t (*)(const gguf_context*, int64_t);
static_assert(std::is_same_v<gguf_reader_callback_t, GgufReaderCallback> &&
              std::is_same_v<decltype(&gguf_init_from_callback), GgufCallbackInitializer>,
              "SAM requires the pinned GGML 0.26.0 bounded GGUF callback reader API");
static_assert(std::is_same_v<decltype(gguf_init_params::no_alloc), bool> &&
              std::is_same_v<decltype(gguf_init_params::ctx), ggml_context**>,
              "SAM requires no-allocation GGUF metadata initialization");
static_assert(std::is_same_v<decltype(&gguf_free), void (*)(gguf_context*)>);
static_assert(std::is_same_v<decltype(&gguf_get_alignment), GgufSizeGetter> &&
              std::is_same_v<decltype(&gguf_get_data_offset), GgufSizeGetter>);
static_assert(std::is_same_v<decltype(&gguf_get_n_kv), int64_t (*)(const gguf_context*)>);
static_assert(std::is_same_v<decltype(&gguf_find_key), int64_t (*)(const gguf_context*, const char*)>);
static_assert(std::is_same_v<decltype(&gguf_get_key), GgufStringGetter> &&
              std::is_same_v<decltype(&gguf_get_val_str), GgufStringGetter> &&
              std::is_same_v<decltype(&gguf_get_tensor_name), GgufStringGetter>);
static_assert(std::is_same_v<decltype(&gguf_get_kv_type), GgufTypeGetter> &&
              std::is_same_v<decltype(&gguf_get_arr_type), GgufTypeGetter>);
static_assert(std::is_same_v<decltype(&gguf_get_val_u32), uint32_t (*)(const gguf_context*, int64_t)>);
static_assert(std::is_same_v<decltype(&gguf_get_arr_n), GgufTensorSizeGetter> &&
              std::is_same_v<decltype(&gguf_get_tensor_offset), GgufTensorSizeGetter> &&
              std::is_same_v<decltype(&gguf_get_tensor_size), GgufTensorSizeGetter>);
static_assert(std::is_same_v<decltype(&gguf_get_arr_data), const void* (*)(const gguf_context*, int64_t)>);
static_assert(std::is_same_v<decltype(&gguf_get_arr_str), const char* (*)(const gguf_context*, int64_t, size_t)>);
static_assert(std::is_same_v<decltype(&gguf_get_tensor_ne), const int64_t* (*)(const gguf_context*, int64_t)>);
static_assert(std::is_same_v<decltype(&gguf_get_tensor_type), ggml_type (*)(const gguf_context*, int64_t)>);
static_assert(std::is_same_v<decltype(&gguf_init_empty), gguf_context* (*)()> &&
              std::is_same_v<decltype(&gguf_set_kv), void (*)(gguf_context*, const gguf_context*)> &&
              std::is_same_v<decltype(&gguf_add_tensor), void (*)(gguf_context*, const ggml_tensor*)> &&
              std::is_same_v<decltype(&gguf_get_meta_size), GgufSizeGetter> &&
              std::is_same_v<decltype(&gguf_get_meta_data), void (*)(const gguf_context*, void*)>,
              "SAM requires GGUF canonical metadata copy/writer APIs");
