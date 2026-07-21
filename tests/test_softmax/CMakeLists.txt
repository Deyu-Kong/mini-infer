add_executable(test_softmax main.cpp)
target_include_directories(test_softmax PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_softmax PRIVATE mini_infer_kernels mini_infer_core)
add_test(NAME softmax_test COMMAND test_softmax
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)