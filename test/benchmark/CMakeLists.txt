cmake_minimum_required(VERSION 3.5)
project(benchmark_builder CXX)
include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
    benchmark
    PREFIX ${CMAKE_BINARY_DIR}/benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    TIMEOUT 10
    UPDATE_DISCONNECTED 1
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_LTO=true
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
)

# Expose required variable (BENCHMARK_INCLUDE_DIR) to parent scope
ExternalProject_Get_Property(benchmark source_dir)
set(BENCHMARK_INCLUDE_DIR ${source_dir}/include CACHE INTERNAL "Path to include folder for Benchmark")