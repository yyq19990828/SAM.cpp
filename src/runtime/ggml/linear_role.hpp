#ifndef SAM_CPP_SRC_RUNTIME_GGML_LINEAR_ROLE_HPP
#define SAM_CPP_SRC_RUNTIME_GGML_LINEAR_ROLE_HPP

#include "ggml.h"

namespace sam::internal {

// Models describe intent; runtime backends own storage and kernel selection.
inline void mark_vision_linear(ggml_tensor* node) { node->op_params[6] = 0x53414c01; }
inline bool is_vision_linear(const ggml_tensor* node) {
    return node->op == GGML_OP_MUL_MAT && node->op_params[6] == 0x53414c01;
}

} // namespace sam::internal

#endif // SAM_CPP_SRC_RUNTIME_GGML_LINEAR_ROLE_HPP
