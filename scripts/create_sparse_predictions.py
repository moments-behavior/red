#!/usr/bin/env python3
"""Create sparse prediction binary for RED's bout inspector.

Format v3 (sparse): only stores frames with valid predictions.
Index array enables O(log n) binary search by frame number.

Much smaller than dense: 454 MB vs 1.37 GB for 33% density.
"""

import math
import os
import struct
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA3D = Path("/Users/johnsonr/datasets/fly_April5/new_models_and_bout_prediction/predictions_S6_male/data3D.csv")
BOUTS_CSV = Path("/Users/johnsonr/datasets/fly_April5/fly_juan/bout_segmentation_v2/walking_bouts_summary.csv")
IK_CSV = Path("/Users/johnsonr/datasets/fly_April5/fly_juan/bout_segmentation/bout_ik_summary.csv")
OUTPUT_DIR = Path("/Users/johnsonr/datasets/fly_April5/fly_juan/bout_db")

SCALE = 10.0
N_KP = 50
EPF = N_KP * 4  # elements per frame: 50 kp × (x,y,z,conf) = 200
FPS = 800
PAGE_SIZE = 4096

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


def main():
    print("=== Create Sparse Prediction Binary ===\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Column map
    df_header = pd.read_csv(DATA3D, nrows=0, low_memory=False)
    cols = df_header.columns.tolist()
    kp_col_map = {}
    for kp_name in FLY50_NAMES:
        for i, c in enumerate(cols):
            if c.split('.')[0] == kp_name:
                kp_col_map[kp_name] = i
                break

    # Pass 1: read all frames, collect valid ones
    print("Reading data3D.csv (collecting valid frames)...")
    stored_frames = []  # (frame_number, float32[EPF])
    total_frames = 0

    reader = pd.read_csv(DATA3D, skiprows=[1], low_memory=False,
                          chunksize=100000, header=0)
    row_offset = 0
    for chunk in reader:
        for local_idx in range(len(chunk)):
            global_idx = row_offset + local_idx
            total_frames = global_idx + 1

            # Check if first keypoint is valid
            ci0 = kp_col_map[FLY50_NAMES[0]]
            try:
                x0 = float(chunk.iloc[local_idx, ci0])
            except (ValueError, IndexError):
                continue
            if math.isnan(x0) or x0 == 0:
                continue

            # Valid frame — extract all keypoints
            arr = np.zeros(EPF, dtype=np.float32)
            for k, kp_name in enumerate(FLY50_NAMES):
                ci = kp_col_map.get(kp_name)
                if ci is None:
                    continue
                try:
                    x = float(chunk.iloc[local_idx, ci]) / SCALE
                    y = float(chunk.iloc[local_idx, ci + 1]) / SCALE
                    z = float(chunk.iloc[local_idx, ci + 2]) / SCALE
                    conf = float(chunk.iloc[local_idx, ci + 3])
                    if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                        arr[k*4:k*4+4] = [x, y, z, conf]
                except (ValueError, IndexError):
                    pass
            stored_frames.append((global_idx, arr))

        row_offset += len(chunk)
        print(f"  {row_offset:,} rows, {len(stored_frames):,} valid...", end='\r')

    n_stored = len(stored_frames)
    print(f"\n  Total: {total_frames:,} frames, {n_stored:,} valid ({100*n_stored/total_frames:.1f}%)")

    # Write sparse binary (v3 format)
    bin_path = OUTPUT_DIR / "fly_predictions.bin"
    print(f"\nWriting {bin_path}...")

    # Header: 32 bytes
    #   magic(4) version(4) total_video_frames(4) n_stored(4) header_size(4) fps(4) num_kp(2) epf(2)
    index_bytes = n_stored * 8  # (frame_num:u32, data_idx:u32) pairs
    header_end = 32 + index_bytes
    data_start = (header_end + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)

    # Header: 32 bytes (6×uint32 + 2×uint16 + 4 bytes padding = 32)
    header = struct.pack('<IIIIIIHHI',
        0x024E5247,  # magic
        3,           # version (sparse)
        total_frames,
        n_stored,
        data_start,
        FPS,
        N_KP,
        EPF,
        0)           # reserved (padding to 32 bytes)

    with open(bin_path, 'wb') as f:
        f.write(header)

        # Frame index: sorted (frame_num, data_idx) pairs
        for data_idx, (frame_num, _) in enumerate(stored_frames):
            f.write(struct.pack('<II', frame_num, data_idx))

        # Padding to data_start
        pos = 32 + index_bytes
        if pos < data_start:
            f.write(b'\x00' * (data_start - pos))

        # Data section: contiguous float32 arrays
        for _, arr in stored_frames:
            f.write(arr.tobytes())

    file_size = os.path.getsize(bin_path)
    print(f"  Size: {file_size / 1e6:.0f} MB (vs {total_frames * EPF * 4 / 1e6:.0f} MB dense)")

    # Create/update DuckDB
    db_path = OUTPUT_DIR / "fly_bouts.duckdb"
    print(f"\nCreating {db_path}...")
    if os.path.exists(db_path):
        os.remove(db_path)

    bouts_df = pd.read_csv(BOUTS_CSV)
    ik_df = pd.read_csv(IK_CSV) if IK_CSV.exists() else None

    db = duckdb.connect(str(db_path))
    db.execute("""
        CREATE TABLE bouts (
            id INTEGER PRIMARY KEY,
            start_frame INTEGER NOT NULL,
            end_frame INTEGER NOT NULL,
            n_frames INTEGER NOT NULL,
            duration_s REAL NOT NULL,
            mean_speed REAL,
            max_speed REAL,
            mean_confidence REAL,
            scut_z_mean REAL,
            ik_mean_mm REAL,
            status INTEGER DEFAULT 0,
            notes TEXT DEFAULT ''
        )
    """)

    # Compute mean confidence per bout from the stored frames
    # Build a quick lookup: frame_num → index
    frame_lookup = {fn: idx for idx, (fn, _) in enumerate(stored_frames)}

    for i, row in bouts_df.iterrows():
        s, e = int(row['start_frame']), int(row['end_frame'])

        # Mean confidence across bout frames
        confs = []
        for fi in range(s, e + 1):
            if fi in frame_lookup:
                arr = stored_frames[frame_lookup[fi]][1]
                # Average confidence of all keypoints with data
                kp_confs = [arr[k*4+3] for k in range(N_KP) if arr[k*4+3] > 0]
                if kp_confs:
                    confs.append(np.mean(kp_confs))
        mean_conf = float(np.mean(confs)) if confs else 0.0

        # IK residual
        ik_val = None
        if ik_df is not None:
            ik_row = ik_df[ik_df['bout_idx'] == row['bout_idx']]
            if len(ik_row) > 0:
                ik_val = float(ik_row.iloc[0]['mean_residual_mm'])

        db.execute("INSERT INTO bouts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            i, s, e, int(row['n_frames']), float(row['duration_s']),
            float(row.get('mean_speed_mm_s', 0)),
            float(row.get('max_speed_mm_s', 0)),
            mean_conf,
            float(row.get('scut_z_mean', 0)),
            ik_val,
            0, ''
        ])

    db.execute("CREATE INDEX idx_bouts_start ON bouts(start_frame)")
    db.execute("CREATE INDEX idx_bouts_status ON bouts(status)")
    n = db.execute("SELECT COUNT(*) FROM bouts").fetchone()[0]
    db.close()
    print(f"  {n} bouts")

    # Metadata
    import json
    meta = {
        'predictions_path': str(bin_path),
        'db_path': str(db_path),
        'format_version': 3,
        'total_frames': total_frames,
        'stored_frames': n_stored,
        'n_keypoints': N_KP,
        'elements_per_frame': EPF,
        'fps': FPS,
        'keypoint_names': FLY50_NAMES,
        'skeleton': 'Fly50',
        'media_folder': '/Users/johnsonr/datasets/fly/videos/2026_03_02_13_11_00',
        'calibration_folder': '/Users/johnsonr/red_demos/fly_calib1/dlt_linear',
        'camera_names': ['Cam2012630', 'Cam2012631', 'Cam2012853',
                         'Cam2012855', 'Cam2012857', 'Cam2012861', 'Cam2012862'],
    }
    meta_path = OUTPUT_DIR / "fly_bouts_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  fly_predictions.bin  — {file_size/1e6:.0f} MB sparse ({n_stored:,} frames)")
    print(f"  fly_bouts.duckdb     — {n} bouts")
    print(f"  fly_bouts_meta.json  — config")


if __name__ == "__main__":
    main()
