# Command-line applications controlled by SAM_BUILD_EXAMPLES. RUNTIME output
# stays in the historical ${build}/examples directory so existing scripts and
# documentation keep resolving the sam_image and sam_video binaries.
add_executable(sam_image image/image.cpp)
target_link_libraries(sam_image PRIVATE sam::sam sam_image_io)
set_target_properties(sam_image PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)

add_executable(sam_video video/video.cpp)
target_link_libraries(sam_video PRIVATE sam::sam sam_image_io)
set_target_properties(sam_video PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
