vcpkg_check_linkage(
    ONLY_STATIC_LIBRARY
)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Xiaosunhub/threadpoollib
    REF v2.1.4
    SHA512  ac3b1bb69f7ff713ceaf5b8f81a1b116f20e7502ad400f08c402a1474f22add5fae64e5716b878c3b42f91f7da3e1a69d1cfb24009c9e3a5b19ca3dafa37478a
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"

    OPTIONS
        -DTHREADPOOL_BUILD_EXAMPLE=OFF
)

vcpkg_cmake_install()

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/threadpool)

file(
    REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
)

vcpkg_install_copyright(
    FILE_LIST
        "${SOURCE_PATH}/LICENSE"
)