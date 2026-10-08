cmake_minimum_required(VERSION 3.20)
project(sam_cpp VERSION 0.1.0 LANGUAGES C CXX)

set(SAM_STANDALONE OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SAM_STANDALONE ON)
endif()
option(SAM_BUILD_TESTS "Build SAM contract and integration checks" ${SAM_STANDALONE})
option(SAM_BUILD_EXAMPLES "Build the sam_image and sam_video command-line applications" ${SAM_STANDALONE})
option(SAM_BUILD_TOOLS "Build the standalone conversion, validation and profiling tools" ${SAM_STANDALONE})
option(SAM_BUILD_CUDA_PROBES "Build experimental CUDA quantization probes" OFF)
option(SAM_ENABLE_INSTALL "Generate install rules for the sam package" ${SAM_STANDALONE})
set(SAM_REFERENCE_DIR "" CACHE PATH "Optional official reference bundle directory")

# SAM is static by default and follows an explicit parent BUILD_SHARED_LIBS.
# Capture the value before GGML's own option() creates its cache entry in a
# fresh build directory; keep the choice stable across reconfigures. When a
# caller-owned GGML already fixed BUILD_SHARED_LIBS, set SAM_BUILD_SHARED_LIBS
# directly to pin the SAM library type. Change library type with a clean build
# directory.
if(NOT DEFINED SAM_BUILD_SHARED_LIBS)
    if(DEFINED BUILD_SHARED_LIBS)
        set(SAM_BUILD_SHARED_DEFAULT "${BUILD_SHARED_LIBS}")
    else()
        set(SAM_BUILD_SHARED_DEFAULT OFF)
    endif()
    set(SAM_BUILD_SHARED_LIBS "${SAM_BUILD_SHARED_DEFAULT}" CACHE BOOL "Build the SAM library as shared")
    unset(SAM_BUILD_SHARED_DEFAULT)
endif()

include(cmake/ggml.cmake)

if(SAM_BUILD_CUDA_PROBES)
    enable_language(CUDA)
endif()

add_subdirectory(src)

# Decoding support is only needed by the CLI applications, the probes and the
# internal tests. The SDK build does not compile STB or any host image code.
if(SAM_BUILD_EXAMPLES OR SAM_BUILD_TESTS OR SAM_BUILD_TOOLS OR SAM_BUILD_CUDA_PROBES)
    add_subdirectory(support/image_io)
endif()
if(SAM_BUILD_EXAMPLES)
    add_subdirectory(apps)
endif()
if(SAM_BUILD_TOOLS OR SAM_BUILD_CUDA_PROBES)
    add_subdirectory(tools)
endif()
if(SAM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SAM_ENABLE_INSTALL)
    include(cmake/install.cmake)
endif()
