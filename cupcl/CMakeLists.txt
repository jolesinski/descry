cmake_minimum_required(VERSION 3.5)
project(cupcl CXX)


if(USE_CUDA)
    # Remove malformed vtk flags
    get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
    set(vtk_flags)
    foreach(it ${dir_defs})
        if(it MATCHES "vtk*")
            list(APPEND vtk_flags ${it})
        endif()
    endforeach()
    foreach(d ${vtk_flags})
        remove_definitions(-D${d})
    endforeach()

    find_package(CUDA 6.5 QUIET REQUIRED)
    set(CUDA_SEPARABLE_COMPILATION ON)
    # Adjust maxregcount for specific device
    LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
    #LIST(APPEND CUDA_NVCC_FLAGS -Xptxas=-v -maxregcount=20)
    #set(CUDA_NVCC_FLAGS "-lineinfo -maxrregcount=20")
    # Suppress PCL warnings
    LIST(APPEND CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets)

    if(${CUDA_VERSION_STRING} VERSION_GREATER "6.5")
        LIST(APPEND CUDA_NVCC_FLAGS --expt-relaxed-constexpr --expt-extended-lambda)
    endif()

    # CUDA cmake does not support target_include_directories
    include_directories(include ../descry/include )

    cuda_add_library(cupcl SHARED
            src/conversion.cu
            src/eigen.cu
            src/iss.cu
            src/memory.cu
            src/normals.cu
            src/support.cu
            OPTIONS -shared -Xcompiler -fPIC --std=c++11)

    # Add it anyway for dependent targets
    target_include_directories(cupcl
            PUBLIC include)

    set_target_properties(cupcl
            PROPERTIES CXX_STANDARD 11
            CXX_STANDARD_REQUIRED ON)

    target_link_libraries(cupcl
            ${PCL_LIBRARIES})
else()
    add_library(cupcl INTERFACE)
    target_include_directories(cupcl
            INTERFACE include)
endif()