add_executable(test_contracts api/test_contracts.cpp)
target_link_libraries(test_contracts PRIVATE sam_private sam_image_io)
add_test(NAME sam_contracts COMMAND test_contracts)

add_executable(test_quantization runtime/ggml/test_quantization.cpp)
target_link_libraries(test_quantization PRIVATE sam_private)
add_test(NAME sam_quantization COMMAND test_quantization)

# One public API used from two translation units must link against the real
# library and fail identically in both.
add_executable(test_two_tu api/test_two_tu_main.cpp api/two_tu_other.cpp)
target_link_libraries(test_two_tu PRIVATE sam::sam)
add_test(NAME sam_two_tu COMMAND test_two_tu)

# Public headers are checked with only the public sam::sam include set: they
# must not need GGML or private implementation headers. Private headers from
# src/, tools/ and tests/ are checked separately through the non-installed
# sam_private support target. The support decoding headers are checked through
# their own target; vendored third_party headers keep their upstream guards.
file(GLOB sam_public_headers CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/sam/*.hpp")
file(GLOB_RECURSE sam_internal_headers CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.hpp"
    "${PROJECT_SOURCE_DIR}/tools/*.hpp"
    "${PROJECT_SOURCE_DIR}/tests/*.hpp")
file(GLOB sam_support_headers CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/support/*/*.hpp")
set(sam_checked_headers ${sam_public_headers} ${sam_internal_headers} ${sam_support_headers})
set(public_header_sources)
set(internal_header_sources)
set(support_header_sources)
set(header_guards)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/header_checks")
foreach(header IN LISTS sam_checked_headers)
    file(READ "${header}" header_text)
    if(header_text MATCHES "#pragma[ \t]+once")
        message(FATAL_ERROR "Project header must use an include guard: ${header}")
    endif()
    string(REGEX MATCH "#ifndef[ \t]+(SAM_CPP_[A-Z0-9_]+)" guard_match "${header_text}")
    set(guard "${CMAKE_MATCH_1}")
    if(NOT guard_match OR NOT header_text MATCHES "#define[ \t]+${guard}([ \t\r\n]|$)")
        message(FATAL_ERROR "Missing SAM_CPP include guard: ${header}")
    endif()
    if(guard IN_LIST header_guards)
        message(FATAL_ERROR "Duplicate include guard ${guard}: ${header}")
    endif()
    list(APPEND header_guards "${guard}")
    file(RELATIVE_PATH relative_header "${PROJECT_SOURCE_DIR}" "${header}")
    string(REPLACE "/" "_" source_name "${relative_header}")
    if(header IN_LIST sam_public_headers)
        set(category public)
    elseif(header IN_LIST sam_support_headers)
        set(category support)
    else()
        set(category internal)
    endif()
    set(source "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${category}/${source_name}.cpp")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${category}")
    file(CONFIGURE OUTPUT "${source}" CONTENT "#include \"${header}\"\n#include \"${header}\"\n")
    if(category STREQUAL "public")
        list(APPEND public_header_sources "${source}")
    elseif(category STREQUAL "support")
        list(APPEND support_header_sources "${source}")
    else()
        list(APPEND internal_header_sources "${source}")
    endif()
endforeach()
add_library(sam_public_header_checks OBJECT ${public_header_sources})
target_link_libraries(sam_public_header_checks PRIVATE sam::sam)
add_library(sam_internal_header_checks OBJECT ${internal_header_sources})
target_link_libraries(sam_internal_header_checks PRIVATE sam_private)
add_library(sam_support_header_checks OBJECT ${support_header_sources})
target_link_libraries(sam_support_header_checks PRIVATE sam_image_io sam::sam)

# No GGML target or include directory: shared value types and the task
# contract must stand alone without the compiled execution stack.
add_executable(test_task_contract api/test_task_contract.cpp)
target_compile_features(test_task_contract PRIVATE cxx_std_17)
target_include_directories(test_task_contract PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
add_test(NAME sam_task_contract COMMAND test_task_contract)

add_executable(test_graph runtime/ggml/test_graph.cpp)
target_link_libraries(test_graph PRIVATE sam_private)
add_test(NAME sam_graph COMMAND test_graph)

add_executable(test_rope runtime/ggml/test_rope.cpp)
target_link_libraries(test_rope PRIVATE sam_private)
add_test(NAME sam_rope COMMAND test_rope cpu)
if(GGML_METAL)
    add_test(NAME sam_rope_metal COMMAND test_rope metal)
endif()

add_executable(test_graph_workspace runtime/ggml/test_graph_workspace.cpp)
target_link_libraries(test_graph_workspace PRIVATE sam_private)
add_test(NAME sam_graph_workspace COMMAND test_graph_workspace)

add_executable(test_graph_profile runtime/ggml/test_graph_profile.cpp)
target_link_libraries(test_graph_profile PRIVATE sam_private)
add_test(NAME sam_graph_profile COMMAND test_graph_profile)

add_executable(test_host_tensor runtime/ggml/test_host_tensor.cpp)
target_link_libraries(test_host_tensor PRIVATE sam_private)
add_test(NAME sam_host_tensor COMMAND test_host_tensor)

add_executable(test_precision_output runtime/ggml/test_precision_output.cpp)
target_link_libraries(test_precision_output PRIVATE sam_private)
add_test(NAME sam_precision_output COMMAND test_precision_output)

add_executable(test_backend runtime/ggml/test_backend.cpp)
target_link_libraries(test_backend PRIVATE sam_private)
add_test(NAME sam_backend COMMAND test_backend)

add_executable(test_precision runtime/ggml/test_precision.cpp)
target_link_libraries(test_precision PRIVATE sam_private)
add_test(NAME sam_precision COMMAND test_precision)

add_executable(test_window runtime/ggml/test_window.cpp)
target_link_libraries(test_window PRIVATE sam_private)
add_test(NAME sam_window COMMAND test_window)

option(SAM_REQUIRE_CUDA_TESTS "Fail CUDA checks instead of skipping when no GPU is visible" OFF)
if(SAM_REQUIRE_CUDA_TESTS AND NOT SAM_GGML_CUDA_TARGET)
    message(FATAL_ERROR "SAM_REQUIRE_CUDA_TESTS requires a CUDA-capable ggml target")
endif()
if(SAM_GGML_CUDA_TARGET)
    add_executable(test_cuda_copy runtime/ggml/test_cuda_copy.cpp)
    target_link_libraries(test_cuda_copy PRIVATE sam_private)
    add_test(NAME sam_cuda_copy COMMAND test_cuda_copy)
    add_executable(test_cuda_concat runtime/ggml/test_cuda_concat.cpp)
    target_link_libraries(test_cuda_concat PRIVATE sam_private)
    add_test(NAME sam_cuda_concat COMMAND test_cuda_concat)
    add_executable(test_cuda_numerics runtime/ggml/test_cuda_numerics.cpp)
    target_link_libraries(test_cuda_numerics PRIVATE sam_private)
    add_test(NAME sam_cuda_numerics COMMAND test_cuda_numerics)
    if(patched_ggml_source)
        find_package(CUDAToolkit REQUIRED)
        add_executable(test_cuda_q8_mmq_scale runtime/ggml/test_cuda_q8_mmq_scale.cpp)
        target_include_directories(test_cuda_q8_mmq_scale PRIVATE ${patched_ggml_source}/include)
        target_link_libraries(test_cuda_q8_mmq_scale PRIVATE ${SAM_GGML_CUDA_TARGET} CUDA::cudart)
        add_test(NAME sam_cuda_q8_mmq_scale COMMAND test_cuda_q8_mmq_scale)
    endif()
    if(NOT SAM_REQUIRE_CUDA_TESTS)
        set_tests_properties(sam_cuda_copy PROPERTIES SKIP_RETURN_CODE 77)
        set_tests_properties(sam_cuda_concat PROPERTIES SKIP_RETURN_CODE 77)
        set_tests_properties(sam_cuda_numerics PROPERTIES SKIP_RETURN_CODE 77)
        if(patched_ggml_source)
            set_tests_properties(sam_cuda_q8_mmq_scale PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
    foreach(check IN ITEMS backend precision window rope graph_workspace graph_profile host_tensor quantization)
        add_test(NAME sam_${check}_cuda COMMAND test_${check} cuda)
        if(NOT SAM_REQUIRE_CUDA_TESTS)
            set_tests_properties(sam_${check}_cuda PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endforeach()
    add_test(NAME sam_precision_cuda_env COMMAND ${CMAKE_COMMAND} -E env
        GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 $<TARGET_FILE:test_precision> cuda)
    if(NOT SAM_REQUIRE_CUDA_TESTS)
        set_tests_properties(sam_precision_cuda_env PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

add_executable(test_tracking models/test_tracking.cpp)
target_link_libraries(test_tracking PRIVATE sam_private)
target_compile_definitions(test_tracking PRIVATE SAM_VIDEO_INVENTORY="${CMAKE_CURRENT_SOURCE_DIR}/data/sam3-video-tensor-inventory.txt")
target_compile_definitions(test_tracking PRIVATE SAM_MEMORY_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/data/video_memory_selection_goldens.txt")
add_test(NAME sam_tracking_math COMMAND test_tracking)
if(SAM_GGML_CUDA_TARGET)
    add_test(NAME sam_tracking_math_cuda COMMAND test_tracking cuda)
    if(NOT SAM_REQUIRE_CUDA_TESTS)
        set_tests_properties(sam_tracking_math_cuda PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

add_executable(test_video models/test_video.cpp)
target_link_libraries(test_video PRIVATE sam_private)
add_test(NAME sam_video_behavior COMMAND test_video)

if(patched_ggml_source)
    get_filename_component(sam_ggml_test_build_dir "${CMAKE_CURRENT_BINARY_DIR}" REALPATH)
    add_test(NAME sam_ggml_preparation
        COMMAND ${CMAKE_COMMAND}
            -DSAM_GGML_SOURCE_DIR=${ggml_SOURCE_DIR}
            -DSAM_TEST_BUILD_DIR=${sam_ggml_test_build_dir}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/integration/test_prepare_ggml.cmake)
endif()

add_executable(test_image_cli models/test_image_cli.cpp)
target_link_libraries(test_image_cli PRIVATE sam::sam sam_image_io)
add_test(NAME sam_image_cli COMMAND test_image_cli)

add_executable(test_image models/test_image.cpp)
target_link_libraries(test_image PRIVATE sam::sam sam_image_io)
set_target_properties(test_image PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Real-checkpoint session checks are manually invoked; ordinary CTest never
# downloads or loads external weights.
add_executable(test_session models/test_session.cpp)
target_link_libraries(test_session PRIVATE sam::sam sam_image_io)
set_target_properties(test_session PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_video_session models/test_video_session.cpp)
target_link_libraries(test_video_session PRIVATE sam::sam sam_image_io)
set_target_properties(test_video_session PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Optional real-weight 64+64 interleaving; default CTest remains weight-free.
add_executable(test_video_long_session models/test_video_long_session.cpp)
target_link_libraries(test_video_long_session PRIVATE sam::sam sam_image_io)
set_target_properties(test_video_long_session PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(SAM_VIDEO_LONG_SESSION_ARGS "" CACHE STRING "Optional MODEL;ENTRY_64_DIR;NEGATIVE_IMAGE;BACKEND;THREADS;NEW_DIR")
if(SAM_VIDEO_LONG_SESSION_ARGS)
    add_test(NAME sam_video_long_session COMMAND test_video_long_session ${SAM_VIDEO_LONG_SESSION_ARGS})
endif()

set(SAM_REFERENCE_MODEL "" CACHE FILEPATH "Converted checkpoint for the official reference suite")
set(SAM_REFERENCE_BACKEND "cpu" CACHE STRING "Backend for the official reference suite")
set_property(CACHE SAM_REFERENCE_BACKEND PROPERTY STRINGS cpu metal cuda)
if(SAM_REFERENCE_DIR)
    if(NOT IS_DIRECTORY "${SAM_REFERENCE_DIR}" OR NOT EXISTS "${SAM_REFERENCE_MODEL}")
        message(FATAL_ERROR "SAM_REFERENCE_DIR requires an existing bundle directory and SAM_REFERENCE_MODEL")
    endif()
    if(NOT SAM_REFERENCE_BACKEND MATCHES "^(cpu|metal|cuda)$")
        message(FATAL_ERROR "SAM_REFERENCE_BACKEND must be cpu, metal or cuda")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME sam_official_reference
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/validation/validate_image.py
            --build-dir ${PROJECT_BINARY_DIR}
            --model ${SAM_REFERENCE_MODEL}
            --reference ${SAM_REFERENCE_DIR}
            --backend ${SAM_REFERENCE_BACKEND})
endif()
