#!/bin/bash
# progressive_refinement.sh — Run calibration refinement in progressive stages
#
# Stage 1: Tight prior (rot=10, trans=100) — anchor to init calibration
# Stage 2: Medium prior (rot=3, trans=50) — allow more rotation with better init
# Stage 3: Loose prior (rot=1, trans=20) — final refinement
#
# Each stage uses the refined calibration from the previous stage as input,
# and re-runs the Python feature matching + triangulation against the updated
# calibration before running BA.
#
# Usage:
#   ./scripts/progressive_refinement.sh <landmarks_json> <init_calib_dir> <output_base>
#   ./scripts/progressive_refinement.sh \
#       /path/to/correspondences/landmarks.json \
#       /path/to/init_calibration \
#       /path/to/progressive_output

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <landmarks_json> <init_calib_dir> <output_base>"
    echo ""
    echo "  landmarks_json:  from calibration_refinement.py (landmarks.json)"
    echo "  init_calib_dir:  folder with initial Cam*.yaml files"
    echo "  output_base:     base directory for progressive output"
    exit 1
fi

LANDMARKS="$1"
INIT_CALIB="$2"
OUTPUT_BASE="$3"
POINTS_3D="$(dirname "$LANDMARKS")/points_3d.json"

# Derive image dirs from landmarks (find the corresponding image sets)
# For now, just use the landmarks as-is for all stages

RED_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEST_BIN="$RED_DIR/release/test_feature_refinement"

if [ ! -x "$TEST_BIN" ]; then
    echo "ERROR: test_feature_refinement not built. Run:"
    echo "  cmake --build release --target test_feature_refinement"
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

echo "============================================================"
echo "Progressive Calibration Refinement"
echo "============================================================"
echo "  Landmarks:  $LANDMARKS"
echo "  Points 3D:  $POINTS_3D"
echo "  Init calib: $INIT_CALIB"
echo "  Output:     $OUTPUT_BASE"
echo ""

# ---- Stage 1: Tight prior ----
STAGE1_OUT="$OUTPUT_BASE/stage1_tight"
echo "=== Stage 1: Tight prior (rot=10, trans=100) ==="
"$TEST_BIN" "$LANDMARKS" "$INIT_CALIB" "$STAGE1_OUT" 10.0 100.0 2>&1 | tail -25
echo ""

# ---- Re-triangulate with Stage 1 calibration ----
echo "=== Re-triangulating with Stage 1 calibration ==="
RETRI_DIR="$OUTPUT_BASE/retri_stage1"
mkdir -p "$RETRI_DIR"

# Find the image dirs used to create the landmarks
# For now, re-run calibration_refinement.py with the stage1 calibration
# to get better triangulation. This is the key: better calibration → better triangulation → better BA.
RETRI_LANDMARKS="$RETRI_DIR/landmarks.json"
RETRI_POINTS="$RETRI_DIR/points_3d.json"

# Extract image dirs from match_stats if available
MATCH_STATS="$(dirname "$LANDMARKS")/match_stats.json"
if [ -f "$MATCH_STATS" ]; then
    IMAGE_DIR_ARG=$(python3 -c "
import json
with open('$MATCH_STATS') as f:
    s = json.load(f)
print(s.get('config', {}).get('image_dir', ''))
" 2>/dev/null)
fi

if [ -n "$IMAGE_DIR_ARG" ] && [ "$IMAGE_DIR_ARG" != "[]" ] && [ "$IMAGE_DIR_ARG" != "" ]; then
    echo "  Re-running feature matching against Stage 1 calibration..."
    python3 "$RED_DIR/data_exporter/calibration_refinement.py" \
        --image_dir $IMAGE_DIR_ARG \
        --calib_dir "$STAGE1_OUT" \
        --output_dir "$RETRI_DIR" \
        --reproj_thresh 15.0 \
        --min_matches 5 \
        --workers 4 \
        --device cpu 2>&1 | grep -E "Processing|tracks|Done|Merging|Per-camera"
    echo ""
else
    echo "  No image dirs in match_stats.json — reusing original landmarks with stage1 points"
    cp "$LANDMARKS" "$RETRI_LANDMARKS"
    cp "$POINTS_3D" "$RETRI_POINTS" 2>/dev/null || true
fi

# ---- Stage 2: Medium prior ----
STAGE2_OUT="$OUTPUT_BASE/stage2_medium"
echo "=== Stage 2: Medium prior (rot=3, trans=50) ==="
if [ -f "$RETRI_LANDMARKS" ]; then
    "$TEST_BIN" "$RETRI_LANDMARKS" "$STAGE1_OUT" "$STAGE2_OUT" 3.0 50.0 2>&1 | tail -25
else
    "$TEST_BIN" "$LANDMARKS" "$STAGE1_OUT" "$STAGE2_OUT" 3.0 50.0 2>&1 | tail -25
fi
echo ""

# ---- Stage 3: Loose prior ----
STAGE3_OUT="$OUTPUT_BASE/stage3_loose"
echo "=== Stage 3: Loose prior (rot=1, trans=20) ==="
if [ -f "$RETRI_LANDMARKS" ]; then
    "$TEST_BIN" "$RETRI_LANDMARKS" "$STAGE2_OUT" "$STAGE3_OUT" 1.0 20.0 2>&1 | tail -25
else
    "$TEST_BIN" "$LANDMARKS" "$STAGE2_OUT" "$STAGE3_OUT" 1.0 20.0 2>&1 | tail -25
fi
echo ""

# ---- Summary ----
echo "============================================================"
echo "Progressive Refinement Complete"
echo "============================================================"
echo "  Stage 1 (tight):  $STAGE1_OUT"
echo "  Stage 2 (medium): $STAGE2_OUT"
echo "  Stage 3 (loose):  $STAGE3_OUT"
echo ""
echo "Compare calibrations:"
echo "  python3 -c \""
echo "import numpy as np"
echo "import sys; sys.path.insert(0, '$RED_DIR/data_exporter')"
echo "from calibration_refinement import load_calibration"
echo "init = load_calibration('$INIT_CALIB')"
echo "s3 = load_calibration('$STAGE3_OUT')"
echo "for s in sorted(init):"
echo "    if s not in s3: continue"
echo "    dR = init[s]['R'].T @ s3[s]['R']"
echo "    d_rot = np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2,-1,1)))"
echo "    d_t = np.linalg.norm(s3[s]['t'] - init[s]['t'])"
echo "    print(f'Cam{s}: dRot={d_rot:.4f} deg, dTrans={d_t:.4f} mm')"
echo "\""
