#!/bin/bash
# Resolve the script's own directory so run.sh works from any cwd.
RED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# libtorch/lib: provides libcudnn.so.9 that ONNX Runtime's CUDA provider
#   dlopens at runtime. Must be reachable even though the main binary's
#   RPATH includes it — dlopen'd .so files go through LD_LIBRARY_PATH first.
# onnxruntime/lib: provides libonnxruntime.so.* and the CUDA/TensorRT
#   provider .so files that get dlopen'd when the EPs are registered.
export LD_LIBRARY_PATH="$RED_DIR/lib/libtorch/lib:$RED_DIR/lib/onnxruntime/lib:${LD_LIBRARY_PATH}"

exec "$RED_DIR/release/red" "$@"
