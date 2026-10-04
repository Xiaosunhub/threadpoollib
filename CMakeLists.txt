cmake_minimum_required(VERSION 3.16)

project(
    threadpool
    VERSION 1.0.0
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(Threads REQUIRED)

# ------------------------------------------------------------
# Static library
# ------------------------------------------------------------

add_library(
    threadpool
    STATIC
    src/threadpool.cpp
)

add_library(
    threadpool::threadpool
    ALIAS threadpool
)

target_compile_features(
    threadpool
    PUBLIC
        cxx_std_17
)

target_include_directories(
    threadpool
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
    threadpool
    PUBLIC
        Threads::Threads
)

# ------------------------------------------------------------
# Example
# ------------------------------------------------------------

option(
    THREADPOOL_BUILD_EXAMPLE
    "Build threadpool example"
    OFF
)

if(THREADPOOL_BUILD_EXAMPLE)

    add_executable(
        threadpooluse
        template/threadpooluse.cpp
    )

    target_link_libraries(
        threadpooluse
        PRIVATE
            threadpool::threadpool
    )

endif()

# ------------------------------------------------------------
# Install
# ------------------------------------------------------------

install(
    TARGETS threadpool
    EXPORT threadpoolTargets

    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}

    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT threadpoolTargets

    FILE threadpoolTargets.cmake

    NAMESPACE threadpool::

    DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake/threadpool
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/threadpoolConfig.cmake.in

    ${CMAKE_CURRENT_BINARY_DIR}/threadpoolConfig.cmake

    INSTALL_DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake/threadpool
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/threadpoolConfigVersion.cmake

    VERSION ${PROJECT_VERSION}

    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/threadpoolConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/threadpoolConfigVersion.cmake

    DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake/threadpool
)