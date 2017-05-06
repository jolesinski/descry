cmake_minimum_required(VERSION 3.5)
project(descry-eval CXX)

add_executable(descry_key_eval
        src/keypoints.cpp)

set_target_properties(descry_key_eval
        PROPERTIES CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)

target_include_directories(descry_key_eval
        PRIVATE ../include)

target_link_libraries(descry_key_eval descry)

add_executable(descry_sandbox
        src/sandbox.cpp)

set_target_properties(descry_sandbox
        PROPERTIES CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)

target_include_directories(descry_sandbox
        PRIVATE ../include)

target_link_libraries(descry_sandbox descry)