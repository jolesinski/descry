cmake_minimum_required(VERSION 3.5)
project(nonius_builder CXX)
include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
    nonius
    PREFIX ${CMAKE_BINARY_DIR}/nonius
    GIT_REPOSITORY https://github.com/libnonius/nonius.git
    GIT_TAG devel
    TIMEOUT 10
    UPDATE_DISCONNECTED 1
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

# Expose required variable (NONIUS_INCLUDE_DIR) to parent scope
ExternalProject_Get_Property(nonius source_dir)
set(NONIUS_INCLUDE_DIR ${source_dir}/include CACHE INTERNAL "Path to include folder for nonius")