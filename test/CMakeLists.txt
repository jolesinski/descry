add_subdirectory(catch EXCLUDE_FROM_ALL)

enable_testing()

add_executable(descry_ut
        ut/test_model.cpp)

add_test(descry_model descry_ut [model])