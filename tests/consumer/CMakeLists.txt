cmake_minimum_required(VERSION 3.20)
project(sam_consumer LANGUAGES C CXX)

set(SAM_SOURCE_DIR "../.." CACHE STRING "SAM source checkout, relative to this consumer's source directory")
get_filename_component(SAM_SOURCE_DIR "${SAM_SOURCE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
# Capture the requested library type once, before SAM/GGML option() entries can
# create or change BUILD_SHARED_LIBS; keep it stable across reconfigures.
if(NOT DEFINED SAM_CONSUMER_SHARED)
    if(DEFINED BUILD_SHARED_LIBS)
        set(SAM_CONSUMER_SHARED "${BUILD_SHARED_LIBS}" CACHE INTERNAL "Consumer library type captured at configure entry")
    else()
        set(SAM_CONSUMER_SHARED OFF CACHE INTERNAL "Consumer library type captured at configure entry")
    endif()
endif()
# An existing source override also exercises reuse of a caller-owned target.
if(FETCHCONTENT_SOURCE_DIR_GGML)
    include("${SAM_SOURCE_DIR}/cmake/prepare_ggml.cmake")
    sam_prepare_ggml("${FETCHCONTENT_SOURCE_DIR_GGML}" "${CMAKE_CURRENT_BINARY_DIR}/caller-ggml-src" caller_ggml_source)
    sam_add_ggml("${caller_ggml_source}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")
endif()
# Forward the consumer's explicit library type to SAM: a caller-owned GGML
# target may already have created a BUILD_SHARED_LIBS cache entry.
set(BUILD_SHARED_LIBS "${SAM_CONSUMER_SHARED}")
add_subdirectory("${SAM_SOURCE_DIR}" sam)

# The public target is a compiled library; private model/runtime sources are
# not part of the consumer's include or link interface.
get_target_property(sam_type sam TYPE)
if(NOT sam_type OR sam_type STREQUAL "INTERFACE_LIBRARY")
    message(FATAL_ERROR "sam::sam must be a compiled library")
endif()
if(SAM_CONSUMER_SHARED AND NOT sam_type STREQUAL "SHARED_LIBRARY")
    message(FATAL_ERROR "BUILD_SHARED_LIBS=ON must produce a shared sam library")
endif()
if(NOT SAM_CONSUMER_SHARED AND NOT sam_type STREQUAL "STATIC_LIBRARY")
    message(FATAL_ERROR "The default source consumer must link the static sam library")
endif()

add_executable(sam_consumer main.cpp)
target_link_libraries(sam_consumer PRIVATE sam::sam)

# One public API from two translation units against the real library.
add_executable(sam_consumer_two_tu ../api/test_two_tu_main.cpp ../api/two_tu_other.cpp)
target_link_libraries(sam_consumer_two_tu PRIVATE sam::sam)

enable_testing()
add_test(NAME sam_consumer COMMAND sam_consumer)
add_test(NAME sam_consumer_two_tu COMMAND sam_consumer_two_tu)
