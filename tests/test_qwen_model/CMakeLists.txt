add_executable(test_qwen_model main.cpp)
target_include_directories(test_qwen_model PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_qwen_model PRIVATE mini_infer_model)
add_test(NAME qwen_model_test COMMAND test_qwen_model
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)