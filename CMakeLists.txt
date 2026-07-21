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
# mini_infer_core library: Tensor, Allocator, basic CUDA helpers
# -----------------------------------------------------------------------------
add_library(mini_infer_core STATIC
    src/core/tensor.cpp
    src/core/allocator.cpp
)
target_include_directories(mini_infer_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(mini_infer_core
    PUBLIC  CUDA::cudart
            CUDA::cuda_driver
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
# Examples
# -----------------------------------------------------------------------------
add_subdirectory(examples)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)