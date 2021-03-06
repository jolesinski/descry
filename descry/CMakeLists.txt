add_library(descry SHARED
        src/alignment.cpp
        src/clusters.cpp
        src/descriptors.cpp
        src/image.cpp
        src/keypoints.cpp
        src/matching.cpp
        src/matcher.cpp
        src/model.cpp
        src/normals.cpp
        src/preprocess.cpp
        src/projector.cpp
        src/recognizer.cpp
        src/ref_frames.cpp
        src/refinement.cpp
        src/segmentation.cpp
        src/willow.cpp
        src/verification.cpp
        src/viewer.cpp)

target_include_directories(descry
        PUBLIC include ${SPDLOG_INCLUDE_DIR}
        PRIVATE src)

set_target_properties(descry
        PROPERTIES CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)

target_link_libraries(descry
        cupcl
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${YAML_CPP_LIBRARIES})

target_compile_options(descry PRIVATE "-Wall" "-Wextra" "-Werror" "--pedantic")

add_dependencies(descry spdlog)