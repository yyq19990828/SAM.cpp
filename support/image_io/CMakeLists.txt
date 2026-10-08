# Decoding support shared by the command-line applications, probes and tests.
# This is not part of the SDK: it is only built when one of those consumers is
# enabled, and it is never installed.
add_library(sam_image_io STATIC image_io.cpp)
target_compile_features(sam_image_io PRIVATE cxx_std_17)
target_include_directories(sam_image_io
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
