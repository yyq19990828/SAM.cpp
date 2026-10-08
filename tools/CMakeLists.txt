# Experimental tools controlled by SAM_BUILD_TOOLS; the CUDA-only probes stay
# under SAM_BUILD_CUDA_PROBES. Both groups keep the historical ${build}/examples
# runtime output paths so existing scripts keep working.

if(SAM_BUILD_TOOLS)
    # Tools inspect private runtime/model headers; they link the non-installed
    # sam_private support target instead of the public library.
    add_executable(sam_profile_graph benchmark/profile_graph.cpp)
    target_link_libraries(sam_profile_graph PRIVATE sam_private sam_image_io)
    set_target_properties(sam_profile_graph PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)

    add_executable(sam_precision_image_probe validation/precision_image_probe.cpp)
    target_link_libraries(sam_precision_image_probe PRIVATE sam_private sam_image_io)
    set_target_properties(sam_precision_image_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)

    add_executable(sam_precision_benchmark_probe benchmark/precision_benchmark_probe.cpp)
    target_link_libraries(sam_precision_benchmark_probe PRIVATE sam_private sam_image_io)
    set_target_properties(sam_precision_benchmark_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)

    add_executable(sam_quantize_rows quantize/quantize_rows.cpp)
    target_compile_features(sam_quantize_rows PRIVATE cxx_std_17)
    target_compile_definitions(sam_quantize_rows PRIVATE
        SAM_GGML_REVISION=\"353b63b439f27ab2cc19dac97ab1681ba6d2d084\"
        SAM_GGML_LIBRARY_PATH=\"$<TARGET_FILE:${SAM_GGML_LINK_TARGET}>\")
    target_link_libraries(sam_quantize_rows PRIVATE ${SAM_GGML_LINK_TARGET} ${CMAKE_DL_LIBS})
    set_target_properties(sam_quantize_rows PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
endif()

if(SAM_BUILD_CUDA_PROBES)
    if(NOT SAM_GGML_CUDA_TARGET)
        message(FATAL_ERROR "SAM_BUILD_CUDA_PROBES requires GGML_CUDA")
    endif()
    find_package(CUDAToolkit REQUIRED)
    add_executable(sam_ggml_linear_probe benchmark/ggml_linear_probe.cpp)
    target_link_libraries(sam_ggml_linear_probe PRIVATE sam_private)
    add_executable(sam_cache_codec_probe quantize/cache_codec_probe.cpp)
    target_link_libraries(sam_cache_codec_probe PRIVATE sam_private)
    add_executable(sam_cache_image_probe quantize/cache_image_probe.cpp)
    target_link_libraries(sam_cache_image_probe PRIVATE sam_private sam_image_io)
    # The numerical cache oracle needs the exact private CUDA encoder. A
    # caller-owned GGML target can expose only public headers, so it need not
    # provide this diagnostic bridge; the cache probe still builds there.
    if(patched_ggml_source)
        add_library(sam_cuda_cache_bridge SHARED quantize/ggml_cuda_cache_bridge.cu)
        target_include_directories(sam_cuda_cache_bridge PRIVATE
            ${patched_ggml_source}/include ${patched_ggml_source}/src)
        target_link_libraries(sam_cuda_cache_bridge PRIVATE CUDA::cudart)
        target_compile_options(sam_cuda_cache_bridge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
        set_target_properties(sam_cuda_cache_bridge PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    endif()
    add_executable(sam_cuda_linear_probe benchmark/cuda_linear_probe.cu)
    target_link_libraries(sam_cuda_linear_probe PRIVATE CUDA::cublasLt CUDA::cublas CUDA::cudart)
    set_target_properties(sam_cuda_linear_probe PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
    foreach(probe IN ITEMS sam_ggml_linear_probe sam_cache_codec_probe sam_cache_image_probe)
        set_target_properties(${probe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
    endforeach()
endif()
