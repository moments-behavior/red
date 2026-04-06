#!/usr/bin/env python3
"""Create DuckDB database + binary trajectory file for fly bout viewer.

Inspired by Green app's three-tier architecture:
  1. DuckDB — bout metadata (fast queries, filtering, sorting)
  2. Binary trajectory — 50-keypoint 3D pose per frame (mmap, zero-copy)
  3. ImGui viewer — trial table + video + skeleton (built separately in C++)

This script creates the first two tiers from our bout segmentation output
and JARVIS data3D.csv predictions.

Usage:
    python create_fly_bout_db.py

Output:
    fly_bouts.duckdb     — bout metadata database
    fly_bouts_traj3d.bin — memory-mappable 3D pose binary (Green v2 format)
"""

import math
import os
import struct
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────
DATA3D = Path("/Users/johnsonr/datasets/fly_April5/new_models_and_bout_prediction/predictions_S6_male/data3D.csv")
BOUTS_CSV = Path("/Users/johnsonr/datasets/fly_April5/fly_juan/bout_segmentation_v2/walking_bouts_summary.csv")
IK_CSV = Path("/Users/johnsonr/datasets/fly_April5/fly_juan/bout_segmentation/bout_ik_summary.csv")
OUTPUT_DIR = Path("/Users/johnsonr/datasets/fly_April5/fly_juan/bout_db")

SCALE = 10.0  # JARVIS raw / 10 = mm
N_KP = 50
FPS = 800

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


def create_duckdb(bouts_df, ik_df, db_path):
    """Create DuckDB with bout metadata."""
    print(f"Creating DuckDB: {db_path}")

    if os.path.exists(db_path):
        os.remove(db_path)

    db = duckdb.connect(str(db_path))

    # Merge IK residuals if available
    if ik_df is not None and len(ik_df) > 0:
        merged = bouts_df.merge(
            ik_df[['bout_idx', 'mean_residual_mm', 'median_residual_mm', 'pct_below_0.5mm']],
            on='bout_idx', how='left')
    else:
        merged = bouts_df.copy()
        merged['mean_residual_mm'] = np.nan
        merged['median_residual_mm'] = np.nan
        merged['pct_below_0.5mm'] = np.nan

    # Create table
    db.execute("""
        CREATE TABLE bouts (
            id              INTEGER PRIMARY KEY,
            fly_id          TEXT,
            bout_idx        INTEGER,
            start_frame     INTEGER NOT NULL,
            end_frame       INTEGER NOT NULL,
            n_frames        INTEGER NOT NULL,
            duration_s      REAL NOT NULL,
            min_cycles      INTEGER,
            total_distance_mm   REAL,
            net_displacement_mm REAL,
            mean_speed_mm_s     REAL,
            max_speed_mm_s      REAL,
            scut_z_mean         REAL,
            scut_z_std          REAL,
            ik_mean_mm          REAL,
            ik_median_mm        REAL,
            ik_pct_good         REAL
        )
    """)

    for i, row in merged.iterrows():
        db.execute("""
            INSERT INTO bouts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            i,
            row.get('fly_id', ''),
            int(row['bout_idx']),
            int(row['start_frame']),
            int(row['end_frame']),
            int(row['n_frames']),
            float(row['duration_s']),
            int(row['min_cycles']) if 'min_cycles' in row and pd.notna(row.get('min_cycles')) else None,
            float(row['total_distance_mm']) if 'total_distance_mm' in row else None,
            float(row['net_displacement_mm']) if 'net_displacement_mm' in row else None,
            float(row['mean_speed_mm_s']) if 'mean_speed_mm_s' in row else None,
            float(row['max_speed_mm_s']) if 'max_speed_mm_s' in row else None,
            float(row['scut_z_mean']) if 'scut_z_mean' in row else None,
            float(row['scut_z_std']) if 'scut_z_std' in row else None,
            float(row['mean_residual_mm']) if pd.notna(row.get('mean_residual_mm')) else None,
            float(row['median_residual_mm']) if pd.notna(row.get('median_residual_mm')) else None,
            float(row['pct_below_0.5mm']) if pd.notna(row.get('pct_below_0.5mm')) else None,
        ])

    # Indices
    db.execute("CREATE INDEX idx_bouts_start ON bouts(start_frame)")
    db.execute("CREATE INDEX idx_bouts_duration ON bouts(duration_s)")
    db.execute("CREATE INDEX idx_bouts_speed ON bouts(mean_speed_mm_s)")
    db.execute("CREATE INDEX idx_bouts_ik ON bouts(ik_mean_mm)")

    # Summary view
    db.execute("""
        CREATE VIEW bout_summary AS
        SELECT
            COUNT(*) as n_bouts,
            ROUND(SUM(duration_s), 1) as total_duration_s,
            ROUND(AVG(duration_s), 3) as avg_duration_s,
            ROUND(AVG(mean_speed_mm_s), 1) as avg_speed,
            ROUND(AVG(ik_mean_mm), 3) as avg_ik_mm,
            ROUND(MIN(start_frame)) as first_frame,
            ROUND(MAX(end_frame)) as last_frame
        FROM bouts
    """)

    # Verify
    result = db.execute("SELECT * FROM bout_summary").fetchone()
    print(f"  {result[0]} bouts, {result[1]}s total, avg {result[2]}s, avg speed {result[3]} mm/s")

    db.close()
    print(f"  Saved: {db_path}")


def create_trajectory_binary(bouts_df, data3d_path, bin_path):
    """Create Green v2-style memory-mappable binary with 3D pose per bout.

    Format (matching Green's traj_reader.h):
      Header (32 bytes): magic, version, num_trials, num_fields, header_size, fps, num_kp
      Field descriptors (44 bytes each): name, elements_per_frame, element_size, dtype
      Index table (12 bytes per trial): data_offset (u64), num_frames (u32)
      Data section: contiguous float32 arrays, page-aligned per trial
    """
    print(f"Creating trajectory binary: {bin_path}")
    n_bouts = len(bouts_df)

    # Load data3D column map
    df_header = pd.read_csv(data3d_path, nrows=0, low_memory=False)
    cols = df_header.columns.tolist()
    kp_col_map = {}
    for kp_name in FLY50_NAMES:
        for i, c in enumerate(cols):
            if c.split('.')[0] == kp_name:
                kp_col_map[kp_name] = i
                break

    # Collect all frames per bout
    print(f"  Loading 3D data for {n_bouts} bouts...")
    all_needed = set()
    bout_frames = []  # [(start, end), ...]
    for _, row in bouts_df.iterrows():
        s, e = int(row['start_frame']), int(row['end_frame'])
        bout_frames.append((s, e))
        all_needed.update(range(s, e + 1))

    max_frame = max(all_needed)

    # Read from CSV
    frame_data = {}  # frame → float32[N_KP*4] (x,y,z,conf per kp, in mm)
    reader = pd.read_csv(data3d_path, skiprows=[1], low_memory=False,
                          chunksize=100000, header=0)
    row_offset = 0
    for chunk in reader:
        for local_idx in range(len(chunk)):
            global_idx = row_offset + local_idx
            if global_idx in all_needed:
                arr = np.zeros(N_KP * 4, dtype=np.float32)
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
                frame_data[global_idx] = arr
        row_offset += len(chunk)
        if row_offset > max_frame:
            break
        print(f"    {row_offset:,} rows, {len(frame_data):,} frames...", end='\r')
    print(f"    Cached {len(frame_data):,} frames")

    # Build binary file (Green v2 format)
    MAGIC = 0x024E5247   # "GRN\x02"
    VERSION = 2
    NUM_FIELDS = 1       # just traj3d
    ELEMENTS_PER_FRAME = N_KP * 4  # 50 kp × (x,y,z,conf)
    ELEMENT_SIZE = 4     # float32
    PAGE_SIZE = 4096

    # Header: 32 bytes
    header = struct.pack('<IIIIIIHH',
        MAGIC, VERSION, n_bouts, NUM_FIELDS,
        0,  # header_size (filled later)
        FPS, N_KP, 0)

    # Field descriptor: 44 bytes
    field_name = b'traj3d\x00' + b'\x00' * 25  # 32 bytes padded
    field_desc = field_name + struct.pack('<III', ELEMENTS_PER_FRAME, ELEMENT_SIZE, 0)

    desc_end = 32 + NUM_FIELDS * 44
    index_start = (desc_end + 7) & ~7  # 8-byte aligned
    index_size = n_bouts * 12
    header_total = index_start + index_size
    # Page-align the data section start
    data_start = (header_total + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)

    # Update header_size
    header = struct.pack('<IIIIIIHH',
        MAGIC, VERSION, n_bouts, NUM_FIELDS,
        data_start, FPS, N_KP, 0)

    # Build index and data
    index_entries = []
    data_chunks = []
    current_offset = data_start

    for bout_idx, (s, e) in enumerate(bout_frames):
        n_frames = e - s + 1
        # Collect frames for this bout
        bout_data = np.zeros((n_frames, ELEMENTS_PER_FRAME), dtype=np.float32)
        for fi in range(n_frames):
            frame_idx = s + fi
            if frame_idx in frame_data:
                bout_data[fi] = frame_data[frame_idx]

        raw = bout_data.tobytes()
        index_entries.append(struct.pack('<QI', current_offset, n_frames))
        data_chunks.append(raw)

        # Page-align next bout
        chunk_end = current_offset + len(raw)
        current_offset = (chunk_end + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)

    # Write file
    with open(bin_path, 'wb') as f:
        f.write(header)
        f.write(field_desc)

        # Padding to index start
        f.write(b'\x00' * (index_start - desc_end))

        # Index table
        for entry in index_entries:
            f.write(entry)

        # Padding to data start
        pos = index_start + index_size
        f.write(b'\x00' * (data_start - pos))

        # Data section
        for ci, chunk in enumerate(data_chunks):
            f.write(chunk)
            # Pad to page boundary
            written = len(chunk)
            pad = (PAGE_SIZE - (written % PAGE_SIZE)) % PAGE_SIZE
            if pad > 0:
                f.write(b'\x00' * pad)

    file_size = os.path.getsize(bin_path)
    print(f"  Wrote {bin_path} ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  {n_bouts} bouts, {sum(e-s+1 for s,e in bout_frames):,} total frames")


def main():
    print("=== Create Fly Bout Database ===\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load bout data
    bouts_df = pd.read_csv(BOUTS_CSV)
    print(f"Bouts: {len(bouts_df)}")

    # Load IK residuals if available
    ik_df = None
    if IK_CSV.exists():
        ik_df = pd.read_csv(IK_CSV)
        print(f"IK summary: {len(ik_df)} bouts")

    t0 = time.time()

    # 1. Create DuckDB
    db_path = OUTPUT_DIR / "fly_bouts.duckdb"
    create_duckdb(bouts_df, ik_df, db_path)

    # 2. Create binary trajectory
    bin_path = OUTPUT_DIR / "fly_bouts_traj3d.bin"
    create_trajectory_binary(bouts_df, DATA3D, bin_path)

    # 3. Write metadata JSON (for the viewer to find everything)
    import json
    meta = {
        'project': 'fly_juan',
        'source': 'JARVIS predictions + bout_segmentation_v2',
        'n_bouts': len(bouts_df),
        'n_keypoints': N_KP,
        'keypoint_names': FLY50_NAMES,
        'fps': FPS,
        'skeleton': 'Fly50',
        'db_path': str(db_path),
        'traj_path': str(bin_path),
        'media_folder': '/Users/johnsonr/datasets/fly/videos/2026_03_02_13_11_00',
        'calibration_folder': '/Users/johnsonr/red_demos/fly_calib1/dlt_linear',
        'camera_names': ['Cam2012630', 'Cam2012631', 'Cam2012853',
                         'Cam2012855', 'Cam2012857', 'Cam2012861', 'Cam2012862'],
        'coordinate_units': 'mm',
    }
    meta_path = OUTPUT_DIR / "fly_bouts_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {meta_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  fly_bouts.duckdb       — bout metadata ({len(bouts_df)} bouts)")
    print(f"  fly_bouts_traj3d.bin   — 3D pose binary (mmap-ready)")
    print(f"  fly_bouts_meta.json    — viewer config")

    # Quick query test
    db = duckdb.connect(str(db_path), read_only=True)
    print(f"\nSample queries:")
    print(f"  Fastest bouts:")
    for row in db.execute("SELECT bout_idx, duration_s, mean_speed_mm_s FROM bouts ORDER BY mean_speed_mm_s DESC LIMIT 5").fetchall():
        print(f"    Bout {row[0]}: {row[1]:.2f}s, {row[2]:.1f} mm/s")
    print(f"  Longest bouts:")
    for row in db.execute("SELECT bout_idx, duration_s, mean_speed_mm_s FROM bouts ORDER BY duration_s DESC LIMIT 5").fetchall():
        print(f"    Bout {row[0]}: {row[1]:.2f}s, {row[2]:.1f} mm/s")
    db.close()


if __name__ == "__main__":
    main()
