add_executable(mini_infer main.cc)
target_link_libraries(mini_infer PRIVATE mini_infer_model)
# main.cc includes CUDA runtime headers; force nvcc to compile it.
set_source_files_properties(main.cc PROPERTIES LANGUAGE CUDA)