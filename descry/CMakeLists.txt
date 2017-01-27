add_library(descry
        src/config.cpp
        src/model.cpp
        src/recognizer.cpp)

target_include_directories(descry
        PUBLIC include
        PRIVATE src)

set_target_properties(descry
        PROPERTIES CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)

target_link_libraries(descry
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES})