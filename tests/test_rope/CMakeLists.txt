add_executable(test_rope main.cpp)
target_include_directories(test_rope PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_rope PRIVATE mini_infer_layers)
add_test(NAME rope_test COMMAND test_rope
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)