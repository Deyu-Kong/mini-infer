add_executable(test_rmsnorm main.cpp)
target_include_directories(test_rmsnorm PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_rmsnorm PRIVATE mini_infer_layers)
add_test(NAME rmsnorm_test COMMAND test_rmsnorm
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)