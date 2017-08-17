cmake_minimum_required(VERSION 3.5)
project(spdlog_fetcher CXX)
include(ExternalProject)

ExternalProject_Add(
    spdlog
    PREFIX ${CMAKE_BINARY_DIR}/spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    TIMEOUT 10
    UPDATE_DISCONNECTED 1
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

# Expose required variable (SPDLOG_INCLUDE_DIR) to parent scope
ExternalProject_Get_Property(spdlog source_dir)
set(SPDLOG_INCLUDE_DIR ${source_dir}/include CACHE INTERNAL "Path to include folder for spdlog")
