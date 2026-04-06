#!/bin/bash
# Submit fly body model fitting to Janelia cluster (A100 GPU).
#
# Prerequisites:
#   1. Data prepared: /Users/johnsonr/datasets/fly_April5/fly_adjustabodies/
#   2. adjustabodies installed in mjx conda env on cluster
#   3. Fly species config in adjustabodies/species/fly.py
#
# Usage:
#   # From login node:
#   bash scripts/cluster_fly_fit.sh
#
# Or interactive (for debugging):
#   bsub -W 2:00 -n 12 -gpu "num=1" -q gpu_a100 -P johnson -Is /bin/bash

set -e

# Paths (adjust for cluster mount points)
DATA_DIR="/groups/johnson/johnsonlab/fly_April5/fly_adjustabodies"
SCRIPT_DIR="/groups/johnson/johnsonlab/red/scripts"
ADJUSTABODIES="/groups/johnson/johnsonlab/adjustabodies"

# Stage data to cluster (run from local machine)
echo "=== Staging data to cluster ==="
echo "Run these commands from your local machine first:"
echo ""
echo "  # Copy data"
echo "  rsync -avz /Users/johnsonr/datasets/fly_April5/fly_adjustabodies/ login1.int.janelia.org:${DATA_DIR}/"
echo ""
echo "  # Copy scripts"
echo "  rsync -avz /Users/johnsonr/src/red/scripts/fit_fly_body.py login1.int.janelia.org:${SCRIPT_DIR}/"
echo ""
echo "  # Copy adjustabodies (with fly species config)"
echo "  rsync -avz /Users/johnsonr/src/adjustabodies/ login1.int.janelia.org:${ADJUSTABODIES}/"
echo ""
echo "  # Copy fly model"
echo "  rsync -avz /Users/johnsonr/src/red/models/fruitfly/ login1.int.janelia.org:/groups/johnson/johnsonlab/fly_April5/fruitfly_model/"
echo ""

# Submit job
cat << 'SUBMIT_SCRIPT'
# SSH to cluster, then run:

# 1. Full adjustabodies fit (GPU, ~30 min)
bsub -W 2:00 -n 12 -gpu "num=1" -q gpu_a100 -P johnson \
  -o /groups/johnson/johnsonlab/fly_April5/logs/fit_%J.out \
  -e /groups/johnson/johnsonlab/fly_April5/logs/fit_%J.err \
  bash -c '
    source ~/miniconda3/bin/activate
    conda activate mjx
    pip install -e /groups/johnson/johnsonlab/adjustabodies 2>/dev/null
    cd /groups/johnson/johnsonlab/red/scripts
    python3 fit_fly_body.py \
      --mode full \
      --max-frames 500 \
      --initial-scale 1.17
  '

# 2. Batch IK on all bout frames (CPU, ~1 hr for 2800 frames)
bsub -W 4:00 -n 24 -P johnson \
  -o /groups/johnson/johnsonlab/fly_April5/logs/ik_%J.out \
  -e /groups/johnson/johnsonlab/fly_April5/logs/ik_%J.err \
  bash -c '
    source ~/miniconda3/bin/activate
    conda activate mjx
    cd /groups/johnson/johnsonlab/red/scripts
    python3 fit_fly_body.py \
      --mode scale \
      --max-frames 2800 \
      --n-scales 1 \
      --scale-range 1.17 1.17
  '

SUBMIT_SCRIPT
