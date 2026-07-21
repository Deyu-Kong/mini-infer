#!/usr/bin/env bash
# Run the full ctest suite with the right env.
# Usage: scripts/run_tests.sh [ctest args...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
export PATH=/data1/kdy/anaconda3/envs/vllm/bin:/usr/local/cuda-12.1/bin:/data1/tyh/miniconda3/bin:$PATH
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=/usr/bin/g++-9
cd "$HERE/build"
ctest "$@"