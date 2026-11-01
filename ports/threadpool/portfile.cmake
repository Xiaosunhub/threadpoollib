vcpkg_check_linkage(
    ONLY_STATIC_LIBRARY
)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Xiaosunhub/threadpoollib
    REF v3.0.0
    SHA512  06f2ad008fffef07e0e4de3827d2b294b7e0c9322d2bf37660160cd8dfc35d7262af18b04759a2b0486a5cddd290964213c259c699eac9b478f25c88bcfd0e49
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