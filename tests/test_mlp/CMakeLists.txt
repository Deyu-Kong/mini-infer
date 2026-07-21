add_executable(test_mlp main.cpp)
target_include_directories(test_mlp PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_mlp PRIVATE mini_infer_layers)
add_test(NAME mlp_test COMMAND test_mlp
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)