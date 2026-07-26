add_executable(test_paged_attention main.cpp)
target_include_directories(test_paged_attention PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_paged_attention PRIVATE mini_infer_engine)
add_test(NAME paged_attention_test COMMAND test_paged_attention
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)