set(DESCRY_TEST_DATA_PATH "${CMAKE_CURRENT_LIST_DIR}/data" CACHE PATH "Path to test data")
configure_file(data/data_paths.h.in generated/descry/test/data_paths.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)

enable_testing()

add_executable(descry_ut
        ut/catch.cpp
        ut/test_alignment.cpp
        ut/test_clusters.cpp
        ut/test_describer.cpp
        ut/test_image.cpp
        ut/test_keypoints.cpp
        ut/test_model.cpp
        ut/test_matcher.cpp
        ut/test_normals.cpp
        ut/test_recognizer.cpp
        ut/test_ref_frames.cpp
        ut/test_willow.cpp)

target_include_directories(descry_ut
        PRIVATE include
        ${CATCH_INCLUDE_DIR})

target_link_libraries(descry_ut descry cupcl)

set_target_properties(descry_ut
        PROPERTIES CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)

add_dependencies(descry_ut catch)

add_test(descry::image descry_ut [image])
add_test(descry::model descry_ut [model])
add_test(descry::recognizer descry_ut [recognizer])

add_executable(descry_perf
        perf/perf.cpp
        perf/keypoints.cpp
        perf/normals.cpp)

target_include_directories(descry_perf
        PRIVATE include
        ${BENCHMARK_INCLUDE_DIR})

target_link_libraries(descry_perf descry gbenchmark pthread)

set_target_properties(descry_perf
        PROPERTIES CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)

add_dependencies(descry_perf benchmark)

add_subdirectory(eval)