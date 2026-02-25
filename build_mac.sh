#!/bin/bash
set -e
cmake -S . -B release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/opt/homebrew"
cmake --build release -j $(sysctl -n hw.logicalcpu)
