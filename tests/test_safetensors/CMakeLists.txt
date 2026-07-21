add_executable(test_safetensors main.cpp)
target_include_directories(test_safetensors PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_safetensors PRIVATE mini_infer_model)
add_test(NAME safetensors_test COMMAND test_safetensors
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})