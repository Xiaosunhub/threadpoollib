vcpkg_check_linkage(
    ONLY_STATIC_LIBRARY
)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Xiaosunhub/threadpoollib
    REF v2.1.1
    SHA512  ff2718f73c4726eff92a3587bc33a857d0456e99fc5856155c30eaa71b70630bbd86463f7b4d3a72575fd795813fd417c8aaf51a6af3caf731c7a8565bfa9d11
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