add_executable(test_allocator main.cpp)
target_link_libraries(test_allocator
    PRIVATE mini_infer_core
)
# main.cpp launches a CUDA kernel — must be compiled by nvcc.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_test(NAME allocator_tests COMMAND test_allocator)