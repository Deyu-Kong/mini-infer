cmake_minimum_required(VERSION 3.20)
project(mini-infer
    VERSION 0.1.0
    DESCRIPTION "Lightweight C++/CUDA LLM inference engine"
    LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Default to Ninja if available (faster), else Unix Makefiles
if(NOT CMAKE_GENERATOR AND EXISTS "/data1/tyh/miniconda3/bin/ninja")
    set(CMAKE_GENERATOR "Ninja" CACHE STRING "CMake generator" FORCE)
endif()

# -----------------------------------------------------------------------------
# CUDA Toolkit discovery (use locally installed 12.1; driver supports 12.2)
# -----------------------------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH /usr/local/cuda-12.1)
set(CUDAToolkit_ROOT /usr/local/cuda-12.1 CACHE PATH "CUDA toolkit root")

# Target GPU architectures — set BEFORE find_package so CUDAToolkit does
# not stomp it with a stale default. RTX A6000 = sm_86 (Ampere).
set(MINI_INFER_CUDA_ARCH "86" CACHE STRING "CUDA SM architectures to target")
set(CMAKE_CUDA_ARCHITECTURES ${MINI_INFER_CUDA_ARCH}
    CACHE STRING "CUDA architectures" FORCE)

find_package(CUDAToolkit 12.0 REQUIRED)

# Locate cuBLAS / cuBLASLt for the Week 2 SwiGLU path.
find_library(CUBLAS_LIB  NAMES cublas   PATHS ${CUDAToolkit_ROOT}/lib64 REQUIRED)
find_library(CUBLASLT_LIB NAMES cublasLt PATHS ${CUDAToolkit_ROOT}/lib64)

message(STATUS "===========================================================")
message(STATUS "mini-infer build configuration")
message(STATUS "  Build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard   : ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA std       : ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA arch      : ${MINI_INFER_CUDA_ARCH}")
message(STATUS "  CUDA toolkit   : ${CUDAToolkit_VERSION} at ${CUDAToolkit_ROOT}")
message(STATUS "  Generator      : ${CMAKE_GENERATOR}")
message(STATUS "===========================================================")

# -----------------------------------------------------------------------------
# mini_infer_core library: Tensor, Allocator, basic CUDA helpers, Graph
# -----------------------------------------------------------------------------
add_library(mini_infer_core STATIC
    src/core/tensor.cpp
    src/core/allocator.cpp
    src/core/graph.cpp
    src/model/model_config.cpp
    src/scheduler/paged_kv_cache.cpp
    src/scheduler/prefix_cache.cpp
    src/scheduler/request.cpp
)
target_include_directories(mini_infer_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann
            ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(mini_infer_core
    PUBLIC  CUDA::cudart
            CUDA::cuda_driver
            ${CUBLAS_LIB}
            m
)
# nvcc-friendly flags
target_compile_options(mini_infer_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall>
)

# Files that contain CUDA kernels but live in .cpp (we keep them .cpp so
# CMake can treat them as the project default language). Force nvcc on them.
function(mi_force_cuda target)
    foreach(src ${ARGN})
        set_source_files_properties(${src} PROPERTIES LANGUAGE CUDA)
    endforeach()
endfunction()

mi_force_cuda(mini_infer_core
    src/core/tensor.cpp          # uses cuda_runtime.h + cudaMalloc
    src/core/allocator.cpp       # uses cuda_runtime.h + cudaMalloc
)

# -----------------------------------------------------------------------------
# mini_infer_kernels library: hand-written CUDA kernels (Week 2+)
# -----------------------------------------------------------------------------
add_library(mini_infer_kernels STATIC
    src/kernels/rmsnorm_kernel.cu
    src/kernels/rope_kernel.cu
    src/kernels/softmax_kernel.cu
    src/kernels/swiglu_kernel.cu
    src/kernels/naive_attn_kernel.cu
    src/kernels/sampling_kernel.cu
    src/kernels/model_utils_kernel.cu
    src/kernels/paged_attn_kernel.cu
)
target_include_directories(mini_infer_kernels
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(mini_infer_kernels
    PUBLIC  mini_infer_core
)
target_compile_options(mini_infer_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall>
)

# -----------------------------------------------------------------------------
# mini_infer_layers library: MLP / RMSNorm / RoPE wrappers (Week 2+)
# -----------------------------------------------------------------------------
add_library(mini_infer_layers STATIC
    src/layers/mlp.cpp
    src/layers/rmsnorm.cpp
    src/layers/rope.cpp
    src/layers/attention.cpp
)
target_include_directories(mini_infer_layers
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann
)
target_link_libraries(mini_infer_layers
    PUBLIC  mini_infer_core
            mini_infer_kernels
            ${CUBLAS_LIB}
            m
)
mi_force_cuda(mini_infer_layers src/layers/mlp.cpp)

# -----------------------------------------------------------------------------
# mini_infer_model library: safetensors loader, QwenModel
# -----------------------------------------------------------------------------
add_library(mini_infer_model STATIC
    src/model/safetensors_loader.cpp
    src/model/qwen_model.cpp
    src/model/weight_mapper.cpp
)
target_include_directories(mini_infer_model
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann
)
target_link_libraries(mini_infer_model
    PUBLIC  mini_infer_core
            mini_infer_layers
            ${CUBLAS_LIB}
            m
)

# -----------------------------------------------------------------------------
# mini_infer_engine library: Engine, Tokenizer, KVCache
# -----------------------------------------------------------------------------
add_library(mini_infer_engine STATIC
    src/core/engine.cpp
    src/core/tokenizer.cpp
    src/scheduler/kv_cache.cpp
    src/scheduler/scheduler.cpp
)
target_include_directories(mini_infer_engine
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(mini_infer_engine
    PUBLIC  mini_infer_model
            mini_infer_kernels
)
mi_force_cuda(mini_infer_engine src/core/engine.cpp src/scheduler/scheduler.cpp)

# -----------------------------------------------------------------------------
# mini_infer_speculative library: DraftEngine, SpecDecoder (Week 7)
# -----------------------------------------------------------------------------
add_library(mini_infer_speculative STATIC
    src/speculative/draft_engine.cpp
    src/speculative/spec_decoder.cpp
)
target_include_directories(mini_infer_speculative
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(mini_infer_speculative
    PUBLIC  mini_infer_engine
            mini_infer_kernels
            curand
)
mi_force_cuda(mini_infer_speculative
    src/speculative/draft_engine.cpp
    src/speculative/spec_decoder.cpp
)

# -----------------------------------------------------------------------------
# mini_infer executable: CLI entry point
# -----------------------------------------------------------------------------
add_executable(mini_infer src/core/main.cc)
target_link_libraries(mini_infer PRIVATE mini_infer_engine mini_infer_speculative)
set_source_files_properties(src/core/main.cc PROPERTIES LANGUAGE CUDA)

# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------
add_subdirectory(examples)

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------
add_subdirectory(benchmarks)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)