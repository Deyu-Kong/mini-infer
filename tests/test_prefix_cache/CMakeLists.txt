cmake_minimum_required(VERSION 3.20)
project(test_prefix_cache CXX)

# Unit test for the Week 7/8 PrefixCache placeholder (no-op in Week 6).
add_executable(test_prefix_cache main.cpp)
target_include_directories(test_prefix_cache PRIVATE ${CMAKE_SOURCE_DIR}/../../src)
target_link_libraries(test_prefix_cache PRIVATE mini_infer_core)
add_test(NAME prefix_cache_test COMMAND test_prefix_cache)