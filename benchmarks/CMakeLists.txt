# Week 1-2 kernel latency microbenchmark.
add_executable(bench_kernels bench_kernels.cpp)
target_include_directories(bench_kernels PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(bench_kernels PRIVATE mini_infer_kernels mini_infer_core)
set_source_files_properties(bench_kernels.cpp PROPERTIES LANGUAGE CUDA)

# ---------------------------------------------------------------------------
# Week 6 standardized benchmark binaries.
# ---------------------------------------------------------------------------

# Helper library: dataset loading + metrics aggregation used by both
# bench_continuous and bench_static.
add_library(bench_common STATIC
    common/dataset.cpp
    common/metrics.cpp
)
target_include_directories(bench_common
    PUBLIC ${CMAKE_SOURCE_DIR}
)
target_link_libraries(bench_common
    PUBLIC mini_infer_core
)
# nlohmann::json is vendored under third_party/nlohmann/.
target_include_directories(bench_common
    PUBLIC ${CMAKE_SOURCE_DIR}/third_party/nlohmann
)

# Continuous batching benchmark (uses Scheduler).
add_executable(bench_continuous bench_continuous.cpp)
target_include_directories(bench_continuous
    PRIVATE ${CMAKE_SOURCE_DIR}
)
target_link_libraries(bench_continuous
    PRIVATE bench_common
            mini_infer_engine
            mini_infer_model
            mini_infer_layers
            mini_infer_kernels
            mini_infer_core
)
set_source_files_properties(bench_continuous.cpp PROPERTIES LANGUAGE CUDA)

# Static batching baseline (uses Engine::generate_batched_paged).
add_executable(bench_static bench_static.cpp)
target_include_directories(bench_static
    PRIVATE ${CMAKE_SOURCE_DIR}
)
target_link_libraries(bench_static
    PRIVATE bench_common
            mini_infer_engine
            mini_infer_model
            mini_infer_layers
            mini_infer_kernels
            mini_infer_core
)
set_source_files_properties(bench_static.cpp PROPERTIES LANGUAGE CUDA)