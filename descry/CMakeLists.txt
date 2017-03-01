add_library(descry SHARED
        src/descriptors.cpp
        src/image.cpp
        src/keypoints.cpp
        src/model.cpp
        src/normals.cpp
        src/recognizer.cpp
        src/ref_frames.cpp)

target_include_directories(descry
        PUBLIC include
        PRIVATE src)

set_target_properties(descry
        PROPERTIES CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)

target_link_libraries(descry
        cupcl
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${YAML_CPP_LIBRARIES})