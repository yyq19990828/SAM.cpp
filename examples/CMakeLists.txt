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

    add_executable(sam_profile_graph ../tools/profile_graph.cpp)
    target_link_libraries(sam_profile_graph PRIVATE sam::sam sam_image_io)
    set_target_properties(sam_profile_graph PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_executable(sam_precision_image_probe ../tools/precision_image_probe.cpp)
    target_link_libraries(sam_precision_image_probe PRIVATE sam::sam sam_image_io)
    set_target_properties(sam_precision_image_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_executable(sam_precision_benchmark_probe ../tools/precision_benchmark_probe.cpp)
    target_link_libraries(sam_precision_benchmark_probe PRIVATE sam::sam sam_image_io)
    set_target_properties(sam_precision_benchmark_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_executable(sam_quantize_rows ../tools/quantize_rows.cpp)
    target_compile_features(sam_quantize_rows PRIVATE cxx_std_17)
    target_compile_definitions(sam_quantize_rows PRIVATE
        SAM_GGML_REVISION=\"353b63b439f27ab2cc19dac97ab1681ba6d2d084\"
        SAM_GGML_LIBRARY_PATH=\"$<TARGET_FILE:ggml>\")
    target_link_libraries(sam_quantize_rows PRIVATE ggml ${CMAKE_DL_LIBS})
    set_target_properties(sam_quantize_rows PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(SAM_BUILD_CUDA_PROBES)
    if(NOT TARGET ggml-cuda)
        message(FATAL_ERROR "SAM_BUILD_CUDA_PROBES requires GGML_CUDA")
    endif()
    find_package(CUDAToolkit REQUIRED)
    add_executable(sam_ggml_linear_probe ../tools/ggml_linear_probe.cpp)
    target_link_libraries(sam_ggml_linear_probe PRIVATE sam::sam)
    add_executable(sam_cache_codec_probe ../tools/cache_codec_probe.cpp)
    target_link_libraries(sam_cache_codec_probe PRIVATE sam::sam)
    add_executable(sam_cache_image_probe ../tools/cache_image_probe.cpp)
    target_link_libraries(sam_cache_image_probe PRIVATE sam::sam sam_image_io)
    # The numerical cache oracle needs the exact private CUDA encoder. A
    # caller-owned GGML target can expose only public headers, so it need not
    # provide this diagnostic bridge; the cache probe still builds there.
    if(patched_ggml_source)
        add_library(sam_cuda_cache_bridge SHARED ../tools/ggml_cuda_cache_bridge.cu)
        target_include_directories(sam_cuda_cache_bridge PRIVATE
            ${patched_ggml_source}/include ${patched_ggml_source}/src)
        target_link_libraries(sam_cuda_cache_bridge PRIVATE CUDA::cudart)
        target_compile_options(sam_cuda_cache_bridge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
        set_target_properties(sam_cuda_cache_bridge PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    endif()
    add_executable(sam_cuda_linear_probe ../tools/cuda_linear_probe.cu)
    target_link_libraries(sam_cuda_linear_probe PRIVATE CUDA::cublasLt CUDA::cublas CUDA::cudart)
    set_target_properties(sam_cuda_linear_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()
