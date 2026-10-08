# Public SAM library: the public headers in include/sam/ expose the task API,
# and this target compiles the wrapper implementations, the model factory, the
# private model implementation under src/models/ and the shared runtime under
# src/runtime/.
include(GenerateExportHeader)

set(SAM_LIBRARY_SOURCES
    api/image_session.cpp
    api/model.cpp
    api/video_session.cpp
    model_factory.cpp)

if(SAM_BUILD_SHARED_LIBS)
    add_library(sam SHARED ${SAM_LIBRARY_SOURCES})
else()
    add_library(sam STATIC ${SAM_LIBRARY_SOURCES})
endif()
add_library(sam::sam ALIAS sam)
target_compile_features(sam PUBLIC cxx_std_17)

generate_export_header(sam
    BASE_NAME sam
    EXPORT_MACRO_NAME SAM_API
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/generated/sam/export.hpp")

# Public includes only: standard library, public value types and the generated
# export header. GGML include directories and implementation macros stay
# PRIVATE so consumers do not compile against the private stack.
target_include_directories(sam
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(sam PRIVATE ggml)
add_dependencies(sam sam_ggml_api_check)

if(SAM_BUILD_SHARED_LIBS)
    # Only the declarations marked with SAM_API are exported from a shared
    # library; model and runtime internals keep hidden visibility.
    set_target_properties(sam PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
endif()

# Not installed. Internal tests, private header checks and the existing
# experimental probes obtain the private include paths, GGML compile
# dependencies and link libraries here. Applications use sam::sam only.
add_library(sam_private INTERFACE)
target_include_directories(sam_private INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_compile_features(sam_private INTERFACE cxx_std_17)
target_link_libraries(sam_private INTERFACE sam::sam ggml)
