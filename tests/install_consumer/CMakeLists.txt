cmake_minimum_required(VERSION 3.20)

# Installed-package consumer: no add_subdirectory, no source-tree path. Copy
# this directory outside the SAM checkout, point CMAKE_PREFIX_PATH at the
# installed prefix and configure. The default consumer enables only CXX: the
# installed final-link dependencies must resolve without a C compiler. Pass
# -DSAM_INSTALL_CONSUMER_LANGUAGES="C;CXX" to also cover a C-enabled consumer.
set(SAM_INSTALL_CONSUMER_LANGUAGES "CXX" CACHE STRING
    "Languages enabled by the installed-package consumer")
project(sam_install_consumer LANGUAGES ${SAM_INSTALL_CONSUMER_LANGUAGES})

find_package(sam CONFIG REQUIRED)

get_target_property(sam_type sam::sam TYPE)
if(NOT sam_type OR sam_type STREQUAL "INTERFACE_LIBRARY")
    message(FATAL_ERROR "The installed sam::sam must be a compiled library")
endif()
if(DEFINED SAM_INSTALL_CONSUMER_SHARED AND SAM_INSTALL_CONSUMER_SHARED
        AND NOT sam_type STREQUAL "SHARED_LIBRARY")
    message(FATAL_ERROR "SAM_INSTALL_CONSUMER_SHARED=ON requires an installed shared library")
endif()
if(DEFINED SAM_INSTALL_CONSUMER_SHARED AND NOT SAM_INSTALL_CONSUMER_SHARED
        AND NOT sam_type STREQUAL "STATIC_LIBRARY")
    message(FATAL_ERROR "SAM_INSTALL_CONSUMER_SHARED=OFF requires an installed static library")
endif()

add_executable(sam_install_consumer main.cpp)
target_link_libraries(sam_install_consumer PRIVATE sam::sam)

enable_testing()
add_test(NAME sam_install_consumer COMMAND sam_install_consumer)
