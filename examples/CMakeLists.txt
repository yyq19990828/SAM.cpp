add_library(sam_image_io STATIC image_io.cpp)
target_compile_features(sam_image_io PRIVATE cxx_std_17)
target_include_directories(sam_image_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(SAM_BUILD_EXAMPLES)
    add_executable(sam_image image.cpp)
    target_link_libraries(sam_image PRIVATE sam::sam sam_image_io)
    set_target_properties(sam_image PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
