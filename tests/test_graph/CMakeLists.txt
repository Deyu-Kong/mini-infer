add_executable(test_graph main.cpp)
target_include_directories(test_graph PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_graph PRIVATE mini_infer_core)
add_test(NAME graph_test COMMAND test_graph
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})