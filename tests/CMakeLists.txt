# Aggregate CMakeLists for tests/. Each test lives in its own
# subdirectory and registers a CTest entry.

add_subdirectory(test_tensor)
add_subdirectory(test_allocator)
add_subdirectory(test_rmsnorm)
add_subdirectory(test_rope)
add_subdirectory(test_softmax)
add_subdirectory(test_swiglu)
add_subdirectory(test_gelu)
add_subdirectory(test_mlp)
add_subdirectory(test_model_config)
add_subdirectory(test_graph)
add_subdirectory(test_safetensors)
add_subdirectory(test_qwen_model)
add_subdirectory(test_weight_mapper)
add_subdirectory(test_paged_attention)
add_subdirectory(test_request)
add_subdirectory(test_prefix_cache)

# Ensure the python verifier is discoverable from tests.
set(MINI_INFER_PYTHON "python3" CACHE STRING "Python interpreter used for verification")
set(MINI_INFER_VERIFY_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/verify/verify.py")

# Make test_helpers.h visible as <test_helpers.h> to test sources.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})