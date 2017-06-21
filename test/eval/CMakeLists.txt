cmake_minimum_required(VERSION 3.5)
project(descry-eval CXX)

add_executable(descry_key_eval src/keypoints.cpp)
target_include_directories(descry_key_eval PRIVATE ../include)
target_link_libraries(descry_key_eval descry)

add_executable(descry_sandbox src/sandbox.cpp)
target_include_directories(descry_sandbox PRIVATE ../include)
target_link_libraries(descry_sandbox descry)

add_executable(descry_recognition_roc src/recognition_roc.cpp)
target_include_directories(descry_recognition_roc PRIVATE ../include)
target_link_libraries(descry_recognition_roc descry)

set_target_properties(descry_key_eval descry_sandbox descry_recognition_roc
        PROPERTIES CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)