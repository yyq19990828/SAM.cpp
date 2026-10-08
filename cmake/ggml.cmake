# A host-provided target takes priority. Never load a second GGML runtime.
if(NOT TARGET ggml AND NOT TARGET ggml::ggml)
    include(FetchContent)
    if(APPLE)
        set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed runtime-compiled Metal source")
    endif()
    set(GGML_BUILD_TESTS OFF CACHE BOOL "Build GGML tests")
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Build GGML examples")
    set(GGML_CUDA_GRAPHS OFF CACHE BOOL "CUDA graph capture requires separate SAM validation")
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
    message(STATUS "Using caller-owned GGML target; Metal/CUDA must honor SAM operator and F32 precision contracts")
endif()

# The public dependency is expressed through the namespaced GGML target. An
# imported ggml::ggml comes from a caller-owned package and stays resolvable in
# the installed package through find_dependency(). SAM-prepared GGML is linked
# as the build-tree ggml target, which the SAM package exports itself.
if(patched_ggml_source)
    set(SAM_GGML_LINK_TARGET ggml)
    set(SAM_GGML_DEPENDENCY_MODE prepared)
elseif(TARGET ggml::ggml)
    set(SAM_GGML_LINK_TARGET ggml::ggml)
    get_target_property(SAM_GGML_IMPORTED ggml::ggml IMPORTED)
    if(SAM_GGML_IMPORTED)
        set(SAM_GGML_DEPENDENCY_MODE imported)
    else()
        set(SAM_GGML_DEPENDENCY_MODE buildtree)
    endif()
elseif(TARGET ggml)
    set(SAM_GGML_LINK_TARGET ggml)
    set(SAM_GGML_DEPENDENCY_MODE buildtree)
else()
    message(FATAL_ERROR "SAM requires either the ggml target or the namespaced ggml::ggml target")
endif()

# Check the actual supplied headers at build time, including host targets whose
# generated include paths cannot be resolved during CMake configuration.
add_library(sam_ggml_api_check OBJECT ${CMAKE_CURRENT_LIST_DIR}/check_ggml.cpp)
target_compile_features(sam_ggml_api_check PRIVATE cxx_std_17)
target_link_libraries(sam_ggml_api_check PRIVATE ${SAM_GGML_LINK_TARGET})
