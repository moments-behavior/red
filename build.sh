#!/usr/bin/env bash
# Build red into ./release. Extra arguments are passed to the configure step:
#     ./build.sh                          auto-detect CUDA (see CMakeLists)
#     ./build.sh -DRED_ENABLE_CUDA=OFF    force the software-decode path
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
cmake -S . -B release "$@"
cmake --build release -j
