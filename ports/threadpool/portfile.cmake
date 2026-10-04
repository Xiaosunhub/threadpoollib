vcpkg_check_linkage(
    ONLY_STATIC_LIBRARY
)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Xiaosunhub/threadpoollib
    REF v1.0.1
    SHA512  77c02bf0dcaccf7a3163457f5ddc4170fd84f083dc1a98fe0f88f3120ed04386dd5025756ba2e592ed31a028834ce2e8688039bde37546caba6af99f4a9c1282
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