cmake_minimum_required(VERSION 3.20)
project(test_request CXX)

# Unit test for the Request state machine and metrics.
add_executable(test_request main.cpp)
target_include_directories(test_request PRIVATE ${CMAKE_SOURCE_DIR}/../../src)
target_link_libraries(test_request PRIVATE mini_infer_core)
add_test(NAME request_test COMMAND test_request)