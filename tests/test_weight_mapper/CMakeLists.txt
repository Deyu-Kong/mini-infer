add_executable(test_weight_mapper main.cpp)
target_include_directories(test_weight_mapper PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_weight_mapper PRIVATE mini_infer_model)
add_test(NAME weight_mapper_test COMMAND test_weight_mapper
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})