add_executable(test_gelu main.cpp)
target_include_directories(test_gelu PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_gelu PRIVATE mini_infer_kernels)
add_test(NAME gelu_test COMMAND test_gelu
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)