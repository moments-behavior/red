#!/usr/bin/env bash
#
# Compile JARVIS ONNX models to TensorRT engines for fast inference in red.
#
# Usage:
#   scripts/compile_tensorrt_engines.sh <model_dir>
#
# Example:
#   scripts/compile_tensorrt_engines.sh /data0/quanshare/mouse_merge_24kp_aug/onnx
#
# The script invokes nvidia's trtexec on each .onnx file in the given
# directory and writes the result alongside as <stem>.engine. red detects
# .engine files at JARVIS Predict panel load time and prefers them over
# the .onnx files (TRT runtime is much faster and has bounded memory).
#
# Engines are GPU-architecture-specific (Ada / Ampere / Hopper / ...).
# Run this script on every rig that needs to do inference. Engines from
# different architectures are NOT portable.
#
# Engines are also TRT-version-specific. red is built against TRT 8.6.1.6;
# engines compiled with a different TRT release won't load.
#
# Environment overrides:
#   TRT_DIR           — path to TensorRT install root
#                       (default: $HOME/nvidia/TensorRT-8.6.1.6)
#   WORKSPACE_MIB     — trtexec workspace cap during compile
#                       (default: 4096, i.e. 4 GiB)
#   FP16              — set FP16=1 to enable --fp16 (smaller, faster, may
#                       lose accuracy; verify against the smoke test)
#   FORCE             — set FORCE=1 to recompile even if engine is newer
#                       than its ONNX

set -euo pipefail

MODEL_DIR="${1:-}"
if [[ -z "$MODEL_DIR" ]] || [[ ! -d "$MODEL_DIR" ]]; then
    echo "Usage: $0 <model_dir>" >&2
    echo "  e.g.  $0 /data0/quanshare/mouse_merge_24kp_aug/onnx" >&2
    exit 1
fi

TRT_DIR="${TRT_DIR:-$HOME/nvidia/TensorRT-8.6.1.6}"
TRTEXEC="$TRT_DIR/bin/trtexec"

if [[ ! -x "$TRTEXEC" ]]; then
    echo "ERROR: trtexec not executable at $TRTEXEC" >&2
    echo "Set TRT_DIR env var to your TensorRT install root if it lives elsewhere." >&2
    exit 1
fi

export LD_LIBRARY_PATH="$TRT_DIR/lib:${LD_LIBRARY_PATH:-}"

if pgrep -f release/red >/dev/null; then
    echo "WARNING: red appears to be running. trtexec may run out of GPU memory." >&2
    echo "  Recommend: pkill -f release/red, then re-run this script." >&2
    echo
fi

# Identify which GPU trtexec will target. CUDA defaults to the device with
# highest compute capability, NOT nvidia-smi index 0 — so picking by sorted
# compute_cap matches what trtexec actually selects.
GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null \
           | awk -F, '{ gsub(/^ +| +$/,"",$1); gsub(/^ +| +$/,"",$2); printf "%s %s\n", $2, $1 }' \
           | sort -k1,1 -r | head -1 || echo "? unknown")
COMPUTE_CAP=$(echo "$GPU_INFO" | awk '{print $1}')
GPU_NAME=$(echo "$GPU_INFO" | cut -d' ' -f2-)

echo "TRT compile: GPU=$GPU_NAME (compute cap $COMPUTE_CAP), model dir=$MODEL_DIR"
WORKSPACE_MIB="${WORKSPACE_MIB:-4096}"
FP16_FLAG=""
if [[ "${FP16:-0}" == "1" ]]; then
    FP16_FLAG="--fp16"
    echo "FP16 enabled (smaller engines, faster inference, verify accuracy)"
fi
echo

# Models red's HybridNet pipeline uses (hybridnet_efftrack is HN-fine-tuned 2D).
# keypoint_detect is the standalone 2D model — only used for the older 2-stage
# fallback path, but compile if present so the user can switch models.
STEMS=( center_detect hybridnet_efftrack hybrid3d keypoint_detect )

for stem in "${STEMS[@]}"; do
    onnx="$MODEL_DIR/$stem.onnx"
    engine="$MODEL_DIR/$stem.engine"
    log="$MODEL_DIR/$stem.trt.log"

    if [[ ! -f "$onnx" ]]; then
        echo "SKIP $stem: $onnx not found"
        continue
    fi
    if [[ -f "$engine" ]] && [[ "$engine" -nt "$onnx" ]] && [[ "${FORCE:-0}" != "1" ]]; then
        size_mib=$(($(stat -c%s "$engine") / 1024 / 1024))
        echo "SKIP $stem: $engine already up-to-date (${size_mib} MiB). FORCE=1 to recompile."
        continue
    fi

    echo "=== Compiling $stem ==="
    if "$TRTEXEC" \
        --onnx="$onnx" \
        --saveEngine="$engine" \
        --workspace="$WORKSPACE_MIB" \
        $FP16_FLAG \
        > "$log" 2>&1
    then
        if [[ -f "$engine" ]]; then
            size_mib=$(($(stat -c%s "$engine") / 1024 / 1024))
            echo "  OK: $engine (${size_mib} MiB)  — log: $log"
        else
            echo "  FAILED: trtexec returned 0 but no engine was written. See $log." >&2
            exit 1
        fi
    else
        echo "  FAILED: trtexec returned non-zero. Tail of $log:" >&2
        tail -10 "$log" >&2
        exit 1
    fi
    echo
done

echo "All requested engines compiled."
echo
echo "Engine files in $MODEL_DIR:"
ls -lh "$MODEL_DIR"/*.engine 2>/dev/null || true
