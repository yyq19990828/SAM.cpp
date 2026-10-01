cmake_minimum_required(VERSION 3.20)
project(sam_cpp VERSION 0.1.0 LANGUAGES C CXX)

set(SAM_STANDALONE OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SAM_STANDALONE ON)
endif()
option(SAM_BUILD_TESTS "Build SAM contract and integration checks" ${SAM_STANDALONE})
option(SAM_BUILD_EXAMPLES "Build the SAM image command-line example" ${SAM_STANDALONE})
set(SAM_REFERENCE_DIR "" CACHE PATH "Optional official reference bundle directory")

include(cmake/ggml.cmake)

add_library(sam INTERFACE)
add_library(sam::sam ALIAS sam)
target_compile_features(sam INTERFACE cxx_std_17)
target_include_directories(sam INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(sam INTERFACE ggml)
add_dependencies(sam sam_ggml_api_check)

if(SAM_BUILD_EXAMPLES OR SAM_BUILD_TESTS)
    add_subdirectory(examples)
endif()
if(SAM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
