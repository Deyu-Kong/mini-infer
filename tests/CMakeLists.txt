# Aggregate CMakeLists for tests/. Each test lives in its own
# subdirectory and registers a CTest entry.

add_subdirectory(test_tensor)
add_subdirectory(test_allocator)