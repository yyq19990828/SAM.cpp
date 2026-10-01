# A host-provided target takes priority. Never load a second GGML runtime.
if(NOT TARGET ggml)
    include(FetchContent)
    if(APPLE)
        set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed runtime-compiled Metal source")
    endif()
    set(GGML_BUILD_TESTS OFF CACHE BOOL "Build GGML tests")
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Build GGML examples")
    FetchContent_Declare(ggml
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG 353b63b439f27ab2cc19dac97ab1681ba6d2d084
        GIT_SHALLOW FALSE
        SOURCE_SUBDIR sam_source_only)
    FetchContent_MakeAvailable(ggml)
    if(NOT TARGET ggml)
        include(${CMAKE_CURRENT_LIST_DIR}/prepare_ggml.cmake)
        sam_prepare_ggml("${ggml_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/sam-ggml-src" patched_ggml_source)
        sam_add_ggml("${patched_ggml_source}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/ggml-build")
    endif()
endif()
if(TARGET ggml AND NOT patched_ggml_source)
    message(STATUS "Using caller-owned GGML target; Metal must honor GGML_PREC_F32 (SAM patch or equivalent)")
endif()

# Check the actual supplied headers at build time, including host targets whose
# generated include paths cannot be resolved during CMake configuration.
add_library(sam_ggml_api_check OBJECT ${CMAKE_CURRENT_LIST_DIR}/check_ggml.cpp)
target_compile_features(sam_ggml_api_check PRIVATE cxx_std_17)
target_link_libraries(sam_ggml_api_check PRIVATE ggml)
