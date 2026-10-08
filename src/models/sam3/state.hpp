#ifndef SAM_CPP_SRC_MODELS_SAM3_STATE_HPP
#define SAM_CPP_SRC_MODELS_SAM3_STATE_HPP

#include "sam/types.hpp"
#include "execution.hpp"
#include "tokenizer.hpp"

#include <memory>
#include <mutex>

namespace sam::internal::sam3 {

struct ModelState {
    std::unique_ptr<GgmlRuntime> runtime;
    ContextPtr context;
    BufferPtr buffer;
    ModelDefinition definition;
    TokenizerData tokenizer;
    ModelInfo model_info;
    // ponytail: one execution lock per model; independent backend execution
    // contexts are the upgrade path when concurrent sessions need throughput.
    std::mutex execution_mutex;
};

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_STATE_HPP
