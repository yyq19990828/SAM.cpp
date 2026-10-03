add_library(sam_image_io STATIC image_io.cpp)
target_compile_features(sam_image_io PRIVATE cxx_std_17)
target_include_directories(sam_image_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(SAM_BUILD_EXAMPLES)
    add_executable(sam_image image.cpp)
    target_link_libraries(sam_image PRIVATE sam::sam sam_image_io)
    set_target_properties(sam_image PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_executable(sam_video video.cpp)
    target_link_libraries(sam_video PRIVATE sam::sam sam_image_io)
    set_target_properties(sam_video PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_executable(sam_quantize_rows ../tools/quantize_rows.cpp)
    target_compile_features(sam_quantize_rows PRIVATE cxx_std_17)
    target_compile_definitions(sam_quantize_rows PRIVATE
        SAM_GGML_REVISION=\"353b63b439f27ab2cc19dac97ab1681ba6d2d084\"
        SAM_GGML_LIBRARY_PATH=\"$<TARGET_FILE:ggml>\")
    target_link_libraries(sam_quantize_rows PRIVATE ggml ${CMAKE_DL_LIBS})
    set_target_properties(sam_quantize_rows PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
