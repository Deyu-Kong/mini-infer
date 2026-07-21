add_executable(test_model_config main.cpp)
target_include_directories(test_model_config PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_model_config PRIVATE mini_infer_core)
add_test(NAME model_config_test COMMAND test_model_config
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})