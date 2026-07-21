add_executable(test_swiglu main.cpp)
target_include_directories(test_swiglu PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_swiglu PRIVATE mini_infer_kernels mini_infer_core)
add_test(NAME swiglu_test COMMAND test_swiglu
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)