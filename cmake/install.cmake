# Standard CMake install/export for the sam package. Included from the root
# directory only when SAM_ENABLE_INSTALL is on (standalone default). The rules
# install the public headers, the compiled library and a relocatable CMake
# package; private implementation, tests, tools and support targets stay out.
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(SAM_GGML_PACKAGE_NAME "ggml" CACHE STRING "CMake package that provides the caller-owned ggml::ggml target")

if(APPLE)
    set(SAM_INSTALL_RPATH "@loader_path")
else()
    set(SAM_INSTALL_RPATH "$ORIGIN")
endif()
# Installed shared libraries must locate their siblings in the same directory
# without environment variables: GGML backends are transitive dependencies of
# libggml, and a shared SAM library depends on libggml in the same prefix.
if(SAM_BUILD_SHARED_LIBS)
    set_target_properties(sam PROPERTIES INSTALL_RPATH "${SAM_INSTALL_RPATH}")
endif()

if(SAM_GGML_DEPENDENCY_MODE STREQUAL "prepared")
    # SAM-prepared GGML is delivered with this package. GGML's own install
    # rules cover shared libraries, public headers and its package config;
    # upstream does not install static ARCHIVE artifacts, so add them for
    # static GGML builds.
    set(SAM_GGML_INSTALL_TARGETS ggml ggml-base)
    foreach(backend IN ITEMS ggml-cpu ggml-blas ggml-metal ggml-cuda ggml-rpc ggml-vulkan
                               ggml-opencl ggml-hip ggml-sycl)
        if(TARGET ${backend})
            list(APPEND SAM_GGML_INSTALL_TARGETS ${backend})
        endif()
    endforeach()
    # GGML links the language-specific OpenMP targets privately; static
    # archives keep them in the exported final-link interface. A consumer that
    # enables only CXX creates only OpenMP::OpenMP_CXX from
    # find_dependency(OpenMP), so each OpenMP target is referenced through
    # $<TARGET_NAME_IF_EXISTS:...> and resolves exactly when the consuming
    # project defines it.
    function(sam_resolve_openmp_link_targets target)
        get_target_property(link_libraries ${target} INTERFACE_LINK_LIBRARIES)
        if(NOT link_libraries)
            return()
        endif()
        set(resolved)
        foreach(item IN LISTS link_libraries)
            foreach(openmp_target IN ITEMS OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
                if(item STREQUAL "$<LINK_ONLY:${openmp_target}>")
                    set(item "$<LINK_ONLY:$<TARGET_NAME_IF_EXISTS:${openmp_target}>>")
                elseif(item STREQUAL "${openmp_target}")
                    set(item "$<TARGET_NAME_IF_EXISTS:${openmp_target}>")
                endif()
            endforeach()
            list(APPEND resolved "${item}")
        endforeach()
        set_target_properties(${target} PROPERTIES INTERFACE_LINK_LIBRARIES "${resolved}")
    endfunction()
    if(GGML_OPENMP_ENABLED)
        foreach(ggml_target IN LISTS SAM_GGML_INSTALL_TARGETS)
            sam_resolve_openmp_link_targets(${ggml_target})
        endforeach()
    endif()

    # PUBLIC_HEADER entries are relative to the directory that created them.
    # Resolve them against the prepared GGML source before this additional
    # install command re-reads the property, and keep GGML's own public-header
    # rule intact.
    get_target_property(SAM_GGML_PUBLIC_HEADERS ggml PUBLIC_HEADER)
    if(SAM_GGML_PUBLIC_HEADERS)
        set(SAM_GGML_ABSOLUTE_HEADERS)
        foreach(header IN LISTS SAM_GGML_PUBLIC_HEADERS)
            if(IS_ABSOLUTE "${header}")
                list(APPEND SAM_GGML_ABSOLUTE_HEADERS "${header}")
            else()
                get_filename_component(absolute "${patched_ggml_source}/${header}" ABSOLUTE)
                list(APPEND SAM_GGML_ABSOLUTE_HEADERS "${absolute}")
            endif()
        endforeach()
        set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${SAM_GGML_ABSOLUTE_HEADERS}")
    endif()
    install(TARGETS ${SAM_GGML_INSTALL_TARGETS}
        EXPORT ggmlTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    set_target_properties(${SAM_GGML_INSTALL_TARGETS} PROPERTIES INSTALL_RPATH "${SAM_INSTALL_RPATH}")
    install(EXPORT ggmlTargets
        FILE samDependencies.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sam)

    set(SAM_GGML_PACKAGE_FIND "")
    set(SAM_GGML_DEPENDENCY_FINDS "find_dependency(Threads)")
    set(SAM_GGML_OPENMP_TARGET_CHECK "")
    if(GGML_OPENMP_ENABLED)
        string(APPEND SAM_GGML_DEPENDENCY_FINDS "\nfind_dependency(OpenMP)")
        # The exported prepared-GGML link interface resolves each
        # language-specific OpenMP target only when the consuming project
        # defines it, so at least one target must exist for the static final
        # link to carry the OpenMP runtime. find_dependency() already fails
        # when OpenMP is unavailable; this guard rejects a finder that reports
        # success without defining any language target.
        set(SAM_GGML_OPENMP_TARGET_CHECK [=[
# At least one OpenMP language target must exist in the consuming project:
# otherwise the final link would silently omit the OpenMP runtime.
if(NOT TARGET OpenMP::OpenMP_C AND NOT TARGET OpenMP::OpenMP_CXX)
    message(FATAL_ERROR "SAM: find_dependency(OpenMP) defined neither OpenMP::OpenMP_C nor OpenMP::OpenMP_CXX; the static final link would omit the OpenMP runtime")
endif()]=])
    endif()
    if(GGML_CUDA)
        string(APPEND SAM_GGML_DEPENDENCY_FINDS "\nfind_dependency(CUDAToolkit)")
    endif()
    if(GGML_BLAS)
        string(APPEND SAM_GGML_DEPENDENCY_FINDS "\nfind_dependency(BLAS)")
    endif()
    if(GGML_VULKAN)
        string(APPEND SAM_GGML_DEPENDENCY_FINDS "\nfind_dependency(Vulkan)")
    endif()
    if(GGML_OPENCL)
        string(APPEND SAM_GGML_DEPENDENCY_FINDS "\nfind_dependency(OpenCL)")
    endif()
elseif(SAM_GGML_DEPENDENCY_MODE STREQUAL "imported")
    # The caller-owned GGML package must be resolvable again from the installed
    # package; its package defines ggml::ggml for the exported SAM target.
    set(SAM_GGML_PACKAGE_FIND "find_dependency(${SAM_GGML_PACKAGE_NAME} CONFIG)")
    set(SAM_GGML_DEPENDENCY_FINDS "")
    set(SAM_GGML_OPENMP_TARGET_CHECK "")
else()
    message(FATAL_ERROR
        "SAM_ENABLE_INSTALL cannot export a caller-owned build-tree GGML target. "
        "Let SAM prepare its pinned GGML, consume SAM with add_subdirectory and "
        "SAM_ENABLE_INSTALL=OFF, or provide GGML through an importable "
        "find_package(${SAM_GGML_PACKAGE_NAME}) package.")
endif()

install(TARGETS sam
    EXPORT samTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/sam
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")
install(FILES ${SAM_GENERATED_INCLUDE_DIR}/sam/export.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sam)

install(EXPORT samTargets
    NAMESPACE sam::
    FILE samTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sam)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/sam-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sam-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sam)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sam-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sam-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/sam-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sam)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/licenses/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/sam)
install(FILES ${PROJECT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/sam)
