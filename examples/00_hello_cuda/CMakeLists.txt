add_executable(00_hello_cuda main.cpp)
target_link_libraries(00_hello_cuda
    PRIVATE mini_infer_core
)
# main.cpp contains CUDA kernel launch syntax — must be compiled by nvcc.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)