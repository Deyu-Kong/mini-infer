add_executable(test_tensor main.cpp)
target_link_libraries(test_tensor
    PRIVATE mini_infer_core
)
# main.cpp launches a CUDA kernel — must be compiled by nvcc.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_test(NAME tensor_tests COMMAND test_tensor)