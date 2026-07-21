#!/usr/bin/env bash
# Configure + build mini-infer from scratch with the right env.
set -euo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
export PATH=/data1/kdy/anaconda3/envs/vllm/bin:/usr/local/cuda-12.1/bin:/data1/tyh/miniconda3/bin:$PATH
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=/usr/bin/g++-9
cd "$HERE"
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja "$@"
cmake --build build -j