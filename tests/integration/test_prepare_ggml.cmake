cmake_minimum_required(VERSION 3.20)
if(NOT SAM_GGML_SOURCE_DIR OR NOT SAM_TEST_BUILD_DIR)
    message(FATAL_ERROR "This check requires explicit source and test build directories")
endif()
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/prepare_ggml.cmake)
find_package(Git REQUIRED)
set(fixture "${SAM_TEST_BUILD_DIR}/ggml-preparation-fixture")
file(REMOVE_RECURSE "${fixture}")
file(MAKE_DIRECTORY "${fixture}")
execute_process(COMMAND ${GIT_EXECUTABLE} init --quiet "${fixture}" RESULT_VARIABLE init_result)
if(NOT init_result EQUAL 0)
    message(FATAL_ERROR "Could not create the enclosing test repository")
endif()
sam_ggml_tree_hash("${SAM_GGML_SOURCE_DIR}" before_hash)

# Destination is inside another Git repository, the formerly silent-skip case.
sam_prepare_ggml("${SAM_GGML_SOURCE_DIR}" "${fixture}/copy" prepared)
# Touch an existing source file without changing content. Recopying would
# restore the original timestamp; a cache hit must leave it alone.
file(TIMESTAMP "${SAM_GGML_SOURCE_DIR}/CMakeLists.txt" original_time "%s")
file(TOUCH "${prepared}/CMakeLists.txt")
file(TIMESTAMP "${prepared}/CMakeLists.txt" touched_time "%s")
if(touched_time STREQUAL original_time)
    execute_process(COMMAND ${CMAKE_COMMAND} -E sleep 1)
    file(TOUCH "${prepared}/CMakeLists.txt")
    file(TIMESTAMP "${prepared}/CMakeLists.txt" touched_time "%s")
endif()
sam_prepare_ggml("${SAM_GGML_SOURCE_DIR}" "${fixture}/copy" prepared)
file(TIMESTAMP "${prepared}/CMakeLists.txt" retained_time "%s")
if(NOT retained_time STREQUAL touched_time)
    message(FATAL_ERROR "Unchanged input unnecessarily recopied the dependency")
endif()
# Drift outside patched paths and unexpected extra files must also be repaired.
file(APPEND "${prepared}/include/ggml.h" "\n// generated copy drift\n")
file(WRITE "${prepared}/unexpected.cpp" "unverified generated source\n")
sam_prepare_ggml("${SAM_GGML_SOURCE_DIR}" "${fixture}/copy" prepared)
if(EXISTS "${prepared}/unexpected.cpp")
    message(FATAL_ERROR "Drifted generated source was not repaired from the original checkout")
endif()

# A superseded fingerprint already requires replacement. Do not try to read
# obsolete files (for example, a stalled filesystem placeholder) first.
if(UNIX)
    file(WRITE "${prepared}/.sam-source-fingerprint" "superseded source identity")
    file(CREATE_LINK "${fixture}/intentionally-missing" "${prepared}/unreadable-stale-file" SYMBOLIC)
    sam_prepare_ggml("${SAM_GGML_SOURCE_DIR}" "${fixture}/copy" prepared)
    if(EXISTS "${prepared}/unreadable-stale-file" OR IS_SYMLINK "${prepared}/unreadable-stale-file")
        message(FATAL_ERROR "Superseded dependency copy was not replaced")
    endif()
endif()

# Already-patched archives are valid inputs even without Git metadata.
sam_prepare_ggml("${prepared}" "${fixture}/prepatched-copy" prepatched)
sam_ggml_tree_hash("${prepared}" prepared_hash)
sam_ggml_tree_hash("${prepatched}" prepatched_hash)
if(NOT prepatched_hash STREQUAL prepared_hash)
    message(FATAL_ERROR "Already-patched source changed during preparation")
endif()

# The prior combined archive acquires only the new short-dot patch.
file(COPY "${prepared}/" DESTINATION "${fixture}/legacy-combined")
execute_process(COMMAND ${CMAKE_COMMAND} -E env --unset=GIT_DIR --unset=GIT_WORK_TREE
    "GIT_CEILING_DIRECTORIES=${fixture}" ${GIT_EXECUTABLE} apply --reverse
    "${CMAKE_CURRENT_LIST_DIR}/../../cmake/patches/ggml-short-dot-cuda.patch"
    WORKING_DIRECTORY "${fixture}/legacy-combined" RESULT_VARIABLE reverse_short_dot_result
    ERROR_VARIABLE reverse_short_dot_error)
if(NOT reverse_short_dot_result EQUAL 0)
    message(FATAL_ERROR "Could not prepare prior combined archive fixture: ${reverse_short_dot_error}")
endif()
sam_prepare_ggml("${fixture}/legacy-combined" "${fixture}/short-dot-upgraded" short_dot_upgraded)
sam_ggml_tree_hash("${short_dot_upgraded}" short_dot_upgraded_hash)
if(NOT short_dot_upgraded_hash STREQUAL prepared_hash)
    message(FATAL_ERROR "Prior combined archive did not acquire the short-dot patch")
endif()

# The combined archive accepted before either numerical correction must also
# upgrade. Pin its historical identity independently of the current preparer.
file(COPY "${fixture}/legacy-combined/" DESTINATION "${fixture}/historical-combined")
execute_process(COMMAND ${CMAKE_COMMAND} -E env --unset=GIT_DIR --unset=GIT_WORK_TREE
    "GIT_CEILING_DIRECTORIES=${fixture}" ${GIT_EXECUTABLE} apply --reverse
    --include=src/ggml-cuda/quantize.cu
    "${CMAKE_CURRENT_LIST_DIR}/../../cmake/patches/ggml-precise-cuda.patch"
    WORKING_DIRECTORY "${fixture}/historical-combined" RESULT_VARIABLE reverse_scale_result
    ERROR_VARIABLE reverse_scale_error)
if(NOT reverse_scale_result EQUAL 0)
    message(FATAL_ERROR "Could not prepare historical combined archive: ${reverse_scale_error}")
endif()
sam_ggml_tree_hash("${fixture}/historical-combined" historical_hash)
if(NOT historical_hash STREQUAL "37f787b8a432f2399e9ee50270f8d025378c605ae23f5efaae6a6ae9148fd5cb")
    message(FATAL_ERROR "Historical combined archive fixture differs from the previously accepted tree")
endif()
sam_prepare_ggml("${fixture}/historical-combined" "${fixture}/historical-upgraded" historical_upgraded)
sam_prepare_ggml("${fixture}/historical-combined" "${fixture}/historical-upgraded" historical_upgraded)
sam_ggml_tree_hash("${historical_upgraded}" historical_upgraded_hash)
sam_ggml_tree_hash("${fixture}/historical-combined" historical_after_hash)
if(NOT historical_upgraded_hash STREQUAL prepared_hash OR NOT historical_after_hash STREQUAL historical_hash)
    message(FATAL_ERROR "Historical archive upgrade changed its input or did not produce the verified combined tree")
endif()

# Previously prepared Metal-only archives acquire both CUDA patches exactly once.
file(COPY "${fixture}/legacy-combined/" DESTINATION "${fixture}/metal-only")
execute_process(COMMAND ${CMAKE_COMMAND} -E env --unset=GIT_DIR --unset=GIT_WORK_TREE
    "GIT_CEILING_DIRECTORIES=${fixture}" ${GIT_EXECUTABLE} apply --reverse
    "${CMAKE_CURRENT_LIST_DIR}/../../cmake/patches/ggml-precise-cuda.patch"
    WORKING_DIRECTORY "${fixture}/metal-only" RESULT_VARIABLE reverse_cuda_result
    ERROR_VARIABLE reverse_cuda_error)
if(NOT reverse_cuda_result EQUAL 0)
    message(FATAL_ERROR "Could not prepare legacy Metal-only archive fixture: ${reverse_cuda_error}")
endif()
sam_prepare_ggml("${fixture}/metal-only" "${fixture}/upgraded-copy" upgraded)
sam_ggml_tree_hash("${upgraded}" upgraded_hash)
if(NOT upgraded_hash STREQUAL prepared_hash)
    message(FATAL_ERROR "Metal-only archive did not acquire the verified combined patches")
endif()

# A dirty archive must fail provenance validation before patch application.
file(APPEND "${prepatched}/include/ggml.h" "\n// unverified source override\n")
file(WRITE "${fixture}/reject-source.cmake"
    "cmake_minimum_required(VERSION 3.20)\ninclude(\"${CMAKE_CURRENT_LIST_DIR}/../../cmake/prepare_ggml.cmake\")\nsam_prepare_ggml(\"${prepatched}\" \"${fixture}/rejected-copy\" rejected)\n")
execute_process(COMMAND ${CMAKE_COMMAND} -P "${fixture}/reject-source.cmake"
    RESULT_VARIABLE reject_result ERROR_VARIABLE reject_error OUTPUT_VARIABLE reject_output)
if(reject_result EQUAL 0 OR NOT reject_error MATCHES "GGML source content differs")
    message(FATAL_ERROR "Unverified source override was not clearly rejected: ${reject_output} ${reject_error}")
endif()
if(EXISTS "${fixture}/rejected-copy")
    message(FATAL_ERROR "Rejected source left a generated dependency copy")
endif()
# An ancestor destination would otherwise delete the caller's source checkout.
file(WRITE "${fixture}/reject-overlap.cmake"
    "cmake_minimum_required(VERSION 3.20)\ninclude(\"${CMAKE_CURRENT_LIST_DIR}/../../cmake/prepare_ggml.cmake\")\nsam_prepare_ggml(\"${prepared}\" \"${fixture}\" rejected)\n")
execute_process(COMMAND ${CMAKE_COMMAND} -P "${fixture}/reject-overlap.cmake"
    RESULT_VARIABLE overlap_result ERROR_VARIABLE overlap_error OUTPUT_VARIABLE overlap_output)
if(overlap_result EQUAL 0 OR NOT overlap_error MATCHES "must not overlap" OR NOT EXISTS "${prepared}/include/ggml.h")
    message(FATAL_ERROR "Overlapping destination was not rejected without deleting its source: ${overlap_output} ${overlap_error}")
endif()
# The sibling temporary name is also removed during preparation.
sam_prepare_ggml("${prepared}" "${fixture}/temporary.tmp" temporary_source)
file(WRITE "${fixture}/reject-temporary.cmake"
    "cmake_minimum_required(VERSION 3.20)\ninclude(\"${CMAKE_CURRENT_LIST_DIR}/../../cmake/prepare_ggml.cmake\")\nsam_prepare_ggml(\"${temporary_source}\" \"${fixture}/temporary\" rejected)\n")
execute_process(COMMAND ${CMAKE_COMMAND} -P "${fixture}/reject-temporary.cmake"
    RESULT_VARIABLE temporary_result ERROR_VARIABLE temporary_error OUTPUT_VARIABLE temporary_output)
if(temporary_result EQUAL 0 OR NOT temporary_error MATCHES "must not overlap" OR NOT EXISTS "${temporary_source}/include/ggml.h")
    message(FATAL_ERROR "Temporary path overlap was not rejected without deleting its source: ${temporary_output} ${temporary_error}")
endif()
# All symlink targets are disposable fixture directories, never user paths.
if(UNIX)
    set(output_link_failures)
    sam_ggml_tree_hash("${prepared}" link_source_before)
    foreach(kind IN ITEMS destination temporary)
        set(output "${fixture}/${kind}-output")
        set(target "${fixture}/${kind}-protected-target")
        set(link "${output}")
        if(kind STREQUAL "temporary")
            set(link "${output}.tmp")
        endif()
        file(MAKE_DIRECTORY "${target}")
        file(WRITE "${target}/sentinel" "preserve this unrelated target")
        file(CREATE_LINK "${target}" "${link}" SYMBOLIC)
        file(WRITE "${fixture}/reject-${kind}-symlink.cmake"
            "cmake_minimum_required(VERSION 3.20)\ninclude(\"${CMAKE_CURRENT_LIST_DIR}/../../cmake/prepare_ggml.cmake\")\nsam_prepare_ggml(\"${prepared}\" \"${output}\" rejected)\n")
        execute_process(COMMAND ${CMAKE_COMMAND} -P "${fixture}/reject-${kind}-symlink.cmake"
            RESULT_VARIABLE link_result ERROR_VARIABLE link_error OUTPUT_VARIABLE link_output)
        if(NOT IS_SYMLINK "${link}")
            list(APPEND output_link_failures "${kind}: original link was removed")
        else()
            file(READ_SYMLINK "${link}" retained_target)
            if(NOT retained_target STREQUAL target)
                list(APPEND output_link_failures "${kind}: original link target changed")
            endif()
        endif()
        if(NOT EXISTS "${target}/sentinel")
            list(APPEND output_link_failures "${kind}: unrelated target sentinel was deleted")
        else()
            file(READ "${target}/sentinel" sentinel)
            if(NOT sentinel STREQUAL "preserve this unrelated target")
                list(APPEND output_link_failures "${kind}: unrelated target sentinel changed")
            endif()
        endif()
        if(link_result EQUAL 0 OR NOT link_error MATCHES "must not be symbolic links")
            list(APPEND output_link_failures "${kind}: symbolic-link output was not clearly rejected")
        endif()
    endforeach()
    sam_ggml_tree_hash("${prepared}" link_source_after)
    if(NOT link_source_after STREQUAL link_source_before)
        list(APPEND output_link_failures "supplied source was modified")
    endif()
    if(output_link_failures)
        message(FATAL_ERROR "Output symlink preparation did not preserve fixtures: ${output_link_failures}")
    endif()
endif()
sam_ggml_tree_hash("${SAM_GGML_SOURCE_DIR}" after_hash)
if(NOT after_hash STREQUAL before_hash)
    message(FATAL_ERROR "Dependency preparation modified the supplied checkout")
endif()
file(REMOVE_RECURSE "${fixture}")
