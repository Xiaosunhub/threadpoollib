vcpkg_check_linkage(
    ONLY_STATIC_LIBRARY
)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Xiaosunhub/threadpoollib
    REF v3.0.1
    SHA512  fd1e6c6cd8753c3d6af2a836ebc163a3a1f41ff177b53ca62012a76920e32af131ed1d1d5b6146e303eefce8b32443d25f04aeb807449181f6933464740b08ac
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