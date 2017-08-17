cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(descry VERSION 0.1 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.1 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(yaml-cpp 0.5.3 REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=gold")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")

include_directories(SYSTEM
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR})

add_subdirectory(extern)
add_subdirectory(descry)
add_subdirectory(cupcl)
add_subdirectory(test)
