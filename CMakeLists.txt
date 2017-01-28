cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(descry VERSION 0.1 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.1 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(yaml-cpp 0.5.3 REQUIRED)

include_directories(
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR})

add_subdirectory(descry)
add_subdirectory(test)