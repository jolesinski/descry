set(DESCRY_TEST_DATA_PATH "${CMAKE_CURRENT_LIST_DIR}/data" CACHE PATH "Path to test data")
configure_file(data/data.h.in include/data.h)

add_subdirectory(catch EXCLUDE_FROM_ALL)

enable_testing()

add_executable(descry_ut
        ut/catch.cpp
        ut/test_model.cpp
        ut/test_recognizer.cpp)

target_include_directories(descry_ut
        PRIVATE include
        ${CATCH_INCLUDE_DIR})

target_link_libraries(descry_ut descry)

set_target_properties(descry_ut
        PROPERTIES CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)

add_dependencies(descry_ut catch)

add_test(descry::model descry_ut [model])
add_test(descry::recognizer descry_ut [recognizer])