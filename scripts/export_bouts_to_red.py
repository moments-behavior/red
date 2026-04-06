#!/usr/bin/env python3
"""Export walking bout predictions to RED annotation format.

Reads JARVIS data3D.csv predictions and walking_bouts_summary.csv,
exports 3D keypoints in RED v2 CSV format so they can be loaded
and inspected per-bout in RED.

Also creates a bouts.json index file for bout navigation in RED.

Usage:
    python export_bouts_to_red.py
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────
DATA3D = Path("/Users/johnsonr/datasets/fly_April5/new_models_and_bout_prediction/predictions_S6_male/data3D.csv")
BOUTS_CSV = Path("/Users/johnsonr/datasets/fly_April5/fly_juan/bout_segmentation_v2/walking_bouts_summary.csv")
PROJECT_DIR = Path("/Users/johnsonr/datasets/fly_April5/fly_juan")
SKELETON = "Fly50"
SCALE = 10.0  # JARVIS raw / 10 = mm
CONF_THRESHOLD = 0.5  # include keypoints above this confidence

FLY50_NAMES = [
    "Antenna_Base", "EyeL", "EyeR", "Scutellum", "Abd_A4", "Abd_tip",
    "WingL_base", "WingL_V12", "WingL_V13",
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",
    "T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",
    "WingR_base", "WingR_V12", "WingR_V13",
    "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",
    "T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",
    "T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",
]
N_KP = len(FLY50_NAMES)


def main():
    print("=== Export bouts to RED ===\n")

    # Load bouts
    bouts_df = pd.read_csv(BOUTS_CSV)
    print(f"Bouts: {len(bouts_df)}")

    # Collect all frame indices we need
    all_frames = set()
    for _, row in bouts_df.iterrows():
        s, e = int(row['start_frame']), int(row['end_frame'])
        all_frames.update(range(s, e + 1))
    print(f"Total frames across all bouts: {len(all_frames):,}")

    # Load data3D.csv header to get column indices
    df_header = pd.read_csv(DATA3D, nrows=0, low_memory=False)
    cols = df_header.columns.tolist()
    kp_col_map = {}
    for kp_name in FLY50_NAMES:
        for i, c in enumerate(cols):
            if c.split('.')[0] == kp_name:
                kp_col_map[kp_name] = i
                break
    print(f"Mapped {len(kp_col_map)}/50 keypoints")

    # Read data3D.csv and extract needed frames
    print(f"Loading {DATA3D}...")
    max_frame = max(all_frames)
    frame_data = {}  # frame → (kp_mm[50,3], conf[50])

    t0 = time.time()
    reader = pd.read_csv(DATA3D, skiprows=[1], low_memory=False,
                          chunksize=100000, header=0)
    row_offset = 0
    for chunk in reader:
        for local_idx in range(len(chunk)):
            global_idx = row_offset + local_idx
            if global_idx in all_frames:
                kp_mm = np.full((N_KP, 3), np.nan, dtype=np.float32)
                conf = np.zeros(N_KP, dtype=np.float32)
                for k, kp_name in enumerate(FLY50_NAMES):
                    ci = kp_col_map.get(kp_name)
                    if ci is None:
                        continue
                    try:
                        x = float(chunk.iloc[local_idx, ci]) / SCALE
                        y = float(chunk.iloc[local_idx, ci + 1]) / SCALE
                        z = float(chunk.iloc[local_idx, ci + 2]) / SCALE
                        c = float(chunk.iloc[local_idx, ci + 3])
                        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                            kp_mm[k] = [x, y, z]
                            conf[k] = c
                    except (ValueError, IndexError):
                        pass
                frame_data[global_idx] = (kp_mm, conf)
        row_offset += len(chunk)
        if row_offset > max_frame:
            break
        print(f"  {row_offset:,} rows, {len(frame_data):,} frames cached...", end='\r')

    print(f"\n  Cached {len(frame_data):,} frames in {time.time()-t0:.0f}s")

    # Create labeled_data folder with timestamp
    ts = time.strftime("%Y%m%d_%H%M%S")
    label_dir = PROJECT_DIR / "labeled_data" / f"jarvis_bouts_{ts}"
    label_dir.mkdir(parents=True, exist_ok=True)

    # Write keypoints3d.csv (RED v2 format)
    csv_3d = label_dir / "keypoints3d.csv"
    with open(csv_3d, 'w') as f:
        f.write("#red_csv v2\n")
        f.write(f"#skeleton {SKELETON}\n")
        # Column header
        header = "frame"
        for k in range(N_KP):
            header += f",x{k},y{k},z{k},c{k}"
        f.write(header + "\n")

        # Write all bout frames
        for frame_idx in sorted(frame_data.keys()):
            kp_mm, conf = frame_data[frame_idx]
            parts = [str(frame_idx)]
            for k in range(N_KP):
                if not np.isnan(kp_mm[k, 0]) and conf[k] >= CONF_THRESHOLD:
                    parts.append(f"{kp_mm[k,0]:.4f}")
                    parts.append(f"{kp_mm[k,1]:.4f}")
                    parts.append(f"{kp_mm[k,2]:.4f}")
                    parts.append(f"{conf[k]:.4f}")
                else:
                    parts.extend(["", "", "", ""])
            f.write(",".join(parts) + "\n")

    print(f"  Wrote {csv_3d} ({len(frame_data)} frames)")

    # Write bouts.json index for RED bout navigation
    bouts_index = []
    for _, row in bouts_df.iterrows():
        bouts_index.append({
            'bout_idx': int(row['bout_idx']),
            'start_frame': int(row['start_frame']),
            'end_frame': int(row['end_frame']),
            'n_frames': int(row['n_frames']),
            'duration_s': float(row['duration_s']),
            'mean_speed_mm_s': float(row['mean_speed_mm_s']),
        })

    bouts_json = label_dir / "bouts.json"
    with open(bouts_json, 'w') as f:
        json.dump({
            'skeleton': SKELETON,
            'source': 'JARVIS predictions + bout_segmentation_v2',
            'n_bouts': len(bouts_index),
            'bouts': bouts_index,
        }, f, indent=2)
    print(f"  Wrote {bouts_json}")

    # Also save in project root for easy access
    proj_bouts = PROJECT_DIR / "bouts.json"
    with open(proj_bouts, 'w') as f:
        json.dump({
            'skeleton': SKELETON,
            'source': 'JARVIS predictions + bout_segmentation_v2',
            'n_bouts': len(bouts_index),
            'label_folder': str(label_dir),
            'bouts': bouts_index,
        }, f, indent=2)
    print(f"  Wrote {proj_bouts}")

    # Summary
    print(f"\n{'='*50}")
    print(f"  {len(bouts_df)} bouts exported")
    print(f"  {len(frame_data):,} frames with 3D predictions")
    print(f"  Label folder: {label_dir}")
    print(f"\n  To load in RED:")
    print(f"    1. Open fly_juan.redproj")
    print(f"    2. Predictions auto-load from {label_dir.name}")
    print(f"    3. Use bouts.json for bout navigation")


if __name__ == "__main__":
    main()
