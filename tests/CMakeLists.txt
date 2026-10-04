add_executable(test_contracts test_contracts.cpp)
target_link_libraries(test_contracts PRIVATE sam::sam)
add_test(NAME sam_contracts COMMAND test_contracts)

add_executable(test_quantization test_quantization.cpp)
target_link_libraries(test_quantization PRIVATE sam::sam)
add_test(NAME sam_quantization COMMAND test_quantization)

add_executable(test_header_only test_header_only_main.cpp header_only_other.cpp)
target_link_libraries(test_header_only PRIVATE sam::sam)
add_test(NAME sam_header_only COMMAND test_header_only)

# Each header starts its own translation unit and is included twice. Linking
# sam::sam supplies only its declared dependencies, not an umbrella preinclude.
file(GLOB_RECURSE sam_headers CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/sam/*.hpp")
file(GLOB example_headers CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/examples/*.hpp")
set(header_sources)
set(header_guards)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/header_checks")
foreach(header IN LISTS sam_headers example_headers)
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
    set(source "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${source_name}.cpp")
    file(CONFIGURE OUTPUT "${source}" CONTENT "#include \"${header}\"\n#include \"${header}\"\n")
    list(APPEND header_sources "${source}")
endforeach()
add_library(sam_header_checks OBJECT ${header_sources})
target_link_libraries(sam_header_checks PRIVATE sam::sam)

# No GGML target or include directory: shared value contracts must stand alone.
add_executable(test_task_contract test_task_contract.cpp)
target_compile_features(test_task_contract PRIVATE cxx_std_17)
target_include_directories(test_task_contract PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME sam_task_contract COMMAND test_task_contract)

add_executable(test_graph test_graph.cpp)
target_link_libraries(test_graph PRIVATE sam::sam)
add_test(NAME sam_graph COMMAND test_graph)

add_executable(test_graph_workspace test_graph_workspace.cpp)
target_link_libraries(test_graph_workspace PRIVATE sam::sam)
add_test(NAME sam_graph_workspace COMMAND test_graph_workspace)

add_executable(test_backend test_backend.cpp)
target_link_libraries(test_backend PRIVATE sam::sam)
add_test(NAME sam_backend COMMAND test_backend)

add_executable(test_precision test_precision.cpp)
target_link_libraries(test_precision PRIVATE sam::sam)
add_test(NAME sam_precision COMMAND test_precision)

add_executable(test_window test_window.cpp)
target_link_libraries(test_window PRIVATE sam::sam)
add_test(NAME sam_window COMMAND test_window)

add_executable(test_tracking test_tracking.cpp)
target_link_libraries(test_tracking PRIVATE sam::sam)
target_compile_definitions(test_tracking PRIVATE SAM_VIDEO_INVENTORY="${CMAKE_CURRENT_SOURCE_DIR}/data/sam3-video-tensor-inventory.txt")
target_compile_definitions(test_tracking PRIVATE SAM_MEMORY_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/data/video_memory_selection_goldens.txt")
add_test(NAME sam_tracking_math COMMAND test_tracking)

add_executable(test_video test_video.cpp)
target_link_libraries(test_video PRIVATE sam::sam)
add_test(NAME sam_video_behavior COMMAND test_video)

if(patched_ggml_source)
    add_test(NAME sam_ggml_preparation
        COMMAND ${CMAKE_COMMAND}
            -DSAM_GGML_SOURCE_DIR=${ggml_SOURCE_DIR}
            -DSAM_TEST_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test_prepare_ggml.cmake)
endif()

add_executable(test_image_cli test_image_cli.cpp)
target_link_libraries(test_image_cli PRIVATE sam::sam sam_image_io)
add_test(NAME sam_image_cli COMMAND test_image_cli)

add_executable(test_image test_image.cpp)
target_link_libraries(test_image PRIVATE sam::sam sam_image_io)
set_target_properties(test_image PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Real-checkpoint session checks are manually invoked; ordinary CTest never
# downloads or loads external weights.
add_executable(test_session test_session.cpp)
target_link_libraries(test_session PRIVATE sam::sam sam_image_io)
set_target_properties(test_session PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_video_session test_video_session.cpp)
target_link_libraries(test_video_session PRIVATE sam::sam sam_image_io)
set_target_properties(test_video_session PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Optional real-weight 64+64 interleaving; default CTest remains weight-free.
add_executable(test_video_long_session test_video_long_session.cpp)
target_link_libraries(test_video_long_session PRIVATE sam::sam sam_image_io)
set_target_properties(test_video_long_session PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(SAM_VIDEO_LONG_SESSION_ARGS "" CACHE STRING "Optional MODEL;ENTRY_64_DIR;NEGATIVE_IMAGE;BACKEND;THREADS;NEW_DIR")
if(SAM_VIDEO_LONG_SESSION_ARGS)
    add_test(NAME sam_video_long_session COMMAND test_video_long_session ${SAM_VIDEO_LONG_SESSION_ARGS})
endif()

set(SAM_REFERENCE_MODEL "" CACHE FILEPATH "Converted checkpoint for the official reference suite")
set(SAM_REFERENCE_BACKEND "cpu" CACHE STRING "Backend for the official reference suite")
set_property(CACHE SAM_REFERENCE_BACKEND PROPERTY STRINGS cpu metal)
if(SAM_REFERENCE_DIR)
    if(NOT IS_DIRECTORY "${SAM_REFERENCE_DIR}" OR NOT EXISTS "${SAM_REFERENCE_MODEL}")
        message(FATAL_ERROR "SAM_REFERENCE_DIR requires an existing bundle directory and SAM_REFERENCE_MODEL")
    endif()
    if(NOT SAM_REFERENCE_BACKEND MATCHES "^(cpu|metal)$")
        message(FATAL_ERROR "SAM_REFERENCE_BACKEND must be cpu or metal")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME sam_official_reference
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/validate_image.py
            --build-dir ${PROJECT_BINARY_DIR}
            --model ${SAM_REFERENCE_MODEL}
            --reference ${SAM_REFERENCE_DIR}
            --backend ${SAM_REFERENCE_BACKEND})
endif()
