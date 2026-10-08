#ifndef SAM_CPP_TESTS_BACKEND_TEST_SUPPORT_HPP
#define SAM_CPP_TESTS_BACKEND_TEST_SUPPORT_HPP

#include <runtime/ggml.hpp>
#include <stdexcept>
#include <string>

namespace sam::test {

inline bool cuda_available() {
    ggml_backend_load_all();
    auto* registration = ggml_backend_reg_by_name("CUDA");
    return registration && ggml_backend_reg_dev_count(registration) > 0;
}

inline bool cuda_requested(int argc, char** argv) {
    if (argc == 1) return false;
    if (argc == 2 && std::string(argv[1]) == "cuda") return true;
    throw std::invalid_argument("expected no arguments or cuda");
}

} // namespace sam::test

#endif // SAM_CPP_TESTS_BACKEND_TEST_SUPPORT_HPP
