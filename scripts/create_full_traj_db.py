#!/usr/bin/env python3
"""Create full-trajectory binary file and DuckDB bout database for Green TrajReader v2.

Reads the JARVIS data3D.csv (1.7M frames, 50 keypoints), writes a memory-mappable
binary in TrajReader v2 format, and creates a DuckDB with 132 walking bouts.

Usage:
    python scripts/create_full_traj_db.py
"""

import json
import math
import os
import struct
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA3D_CSV = "/Users/johnsonr/datasets/fly_April5/new_models_and_bout_prediction/predictions_S6_male/data3D.csv"
BOUTS_CSV = "/Users/johnsonr/datasets/fly_April5/fly_juan/bout_segmentation_v2/walking_bouts_summary.csv"
IK_CSV = "/Users/johnsonr/datasets/fly_April5/fly_juan/bout_segmentation/bout_ik_summary.csv"
OUT_DIR = Path("/Users/johnsonr/datasets/fly_April5/fly_juan/bout_db")

OUT_BIN = OUT_DIR / "fly_predictions.bin"
OUT_DB = OUT_DIR / "fly_bouts.duckdb"
OUT_META = OUT_DIR / "fly_bouts_meta.json"

SCALE = 10.0  # JARVIS raw values / SCALE = mm
FPS = 800
NUM_KEYPOINTS = 50
VALUES_PER_KP = 4  # x, y, z, conf
ELEMENTS_PER_FRAME = NUM_KEYPOINTS * VALUES_PER_KP  # 200
CHUNK_SIZE = 100_000  # rows per chunk when reading CSV
PAGE_SIZE = 4096  # for alignment

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

# TrajReader v2 constants
MAGIC = 0x024E5247
VERSION = 2
FIELD_NAME = b"traj3d"

# ---------------------------------------------------------------------------
# Binary file creation
# ---------------------------------------------------------------------------

def page_align(offset: int) -> int:
    """Round up to next PAGE_SIZE boundary."""
    return (offset + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE


def write_binary(csv_path: str, out_path: Path) -> int:
    """Write TrajReader v2 binary. Returns total number of frames written."""
    print(f"Reading CSV: {csv_path}")
    t0 = time.time()

    # Count data rows (skip 2 header rows)
    # We already know it's ~1.7M but let's be precise
    total_lines = 0
    with open(csv_path, "r") as f:
        for _ in f:
            total_lines += 1
    num_frames = total_lines - 2  # two header rows
    print(f"  Total frames: {num_frames:,}")

    # Layout:
    #   Header:           32 bytes
    #   Field descriptor:  44 bytes
    #   Index table:       12 bytes (1 trial)
    #   Padding to page boundary
    #   Data:             num_frames * ELEMENTS_PER_FRAME * 4 bytes
    header_raw_size = 32 + 44 + 12
    header_size = page_align(header_raw_size)  # page-aligned
    data_offset = header_size
    data_size = num_frames * ELEMENTS_PER_FRAME * 4

    print(f"  Header size (page-aligned): {header_size}")
    print(f"  Data size: {data_size / 1e9:.2f} GB")
    print(f"  Total file size: {(header_size + data_size) / 1e9:.2f} GB")

    with open(out_path, "wb") as f:
        # --- Header (32 bytes) ---
        # magic(4) + version(4) + num_trials(4) + num_fields(4) + header_size(4) + fps(4) + num_keypoints(4) + reserved(4)
        f.write(struct.pack("<IIIIIIII",
            MAGIC,
            VERSION,
            1,                  # num_trials
            1,                  # num_fields
            header_size,        # header_size (page-aligned)
            FPS,                # fps
            NUM_KEYPOINTS,      # num_keypoints
            0,                  # reserved
        ))

        # --- Field descriptor (44 bytes) ---
        # name (32 bytes, null-padded) + elements_per_frame(4) + element_size(4) + dtype(4)
        name_padded = FIELD_NAME.ljust(32, b"\x00")
        f.write(name_padded)
        f.write(struct.pack("<III",
            ELEMENTS_PER_FRAME,  # elements_per_frame = 200
            4,                   # element_size = sizeof(float32)
            0,                   # dtype = 0 (float32)
        ))

        # --- Index table: 1 entry (12 bytes) ---
        # offset(4, in frames from data start) + num_frames(4) + reserved(4)
        f.write(struct.pack("<III",
            0,                  # offset in frames (starts at 0)
            num_frames,         # num_frames
            0,                  # reserved
        ))

        # --- Pad to page boundary ---
        current = f.tell()
        assert current == header_raw_size
        padding = header_size - current
        if padding > 0:
            f.write(b"\x00" * padding)

        assert f.tell() == header_size

        # --- Data section: read CSV in chunks ---
        frames_written = 0
        reader = pd.read_csv(csv_path, header=[0, 1], chunksize=CHUNK_SIZE,
                             dtype=np.float32, na_values=["NaN"])

        for chunk_idx, chunk in enumerate(reader):
            # chunk is DataFrame with MultiIndex columns, shape (N, 200)
            arr = chunk.values.astype(np.float32)  # (N, 200)

            # Replace NaN with 0
            np.nan_to_num(arr, copy=False, nan=0.0)

            # Scale xyz (every 4th value starting at 0,1,2 is x,y,z; index 3 is conf)
            for i in range(NUM_KEYPOINTS):
                base = i * VALUES_PER_KP
                arr[:, base:base + 3] /= SCALE  # x, y, z

            f.write(arr.tobytes())
            frames_written += len(arr)

            if (chunk_idx + 1) % 5 == 0 or frames_written == num_frames:
                elapsed = time.time() - t0
                print(f"  Written {frames_written:,} / {num_frames:,} frames "
                      f"({100 * frames_written / num_frames:.1f}%) [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    file_size = os.path.getsize(out_path)
    print(f"  Binary written: {out_path} ({file_size / 1e9:.2f} GB, {elapsed:.1f}s)")
    return num_frames


# ---------------------------------------------------------------------------
# DuckDB creation
# ---------------------------------------------------------------------------

def compute_bout_confidence(bin_path: Path, num_frames_total: int,
                            bouts_df: pd.DataFrame) -> list[float]:
    """Compute mean confidence per bout by mmap-ing the binary file."""
    header_size = page_align(32 + 44 + 12)
    frame_bytes = ELEMENTS_PER_FRAME * 4  # 800 bytes per frame

    # Memory-map the data section
    data = np.memmap(bin_path, dtype=np.float32, mode="r",
                     offset=header_size,
                     shape=(num_frames_total, ELEMENTS_PER_FRAME))

    confidences = []
    for _, row in bouts_df.iterrows():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        # Confidence columns: indices 3, 7, 11, ..., 199
        conf_cols = list(range(3, ELEMENTS_PER_FRAME, VALUES_PER_KP))
        bout_data = data[start:end + 1, conf_cols]  # (n_frames, 50)
        # Only count non-zero (valid) frames
        valid_mask = bout_data > 0
        if valid_mask.sum() > 0:
            mean_conf = float(bout_data[valid_mask].mean())
        else:
            mean_conf = 0.0
        confidences.append(mean_conf)

    return confidences


def create_duckdb(bouts_csv: str, ik_csv: str, bin_path: Path,
                  num_frames_total: int, out_db: Path):
    """Create DuckDB with bout table."""
    print(f"\nCreating DuckDB: {out_db}")

    bouts_df = pd.read_csv(bouts_csv)
    print(f"  Loaded {len(bouts_df)} bouts from {bouts_csv}")

    # Load IK data if available
    ik_map = {}
    if os.path.exists(ik_csv):
        ik_df = pd.read_csv(ik_csv)
        for _, row in ik_df.iterrows():
            ik_map[int(row["bout_idx"])] = float(row["mean_residual_mm"])
        print(f"  Loaded {len(ik_map)} IK entries from {ik_csv}")

    # Compute mean confidence per bout from binary
    print("  Computing mean confidence per bout from binary...")
    confidences = compute_bout_confidence(bin_path, num_frames_total, bouts_df)

    # Remove existing DB file
    if out_db.exists():
        out_db.unlink()

    con = duckdb.connect(str(out_db))

    con.execute("""
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

    for i, row in bouts_df.iterrows():
        bout_idx = int(row["bout_idx"])
        ik_val = ik_map.get(bout_idx, None)
        con.execute("""
            INSERT INTO bouts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '')
        """, [
            bout_idx,
            int(row["start_frame"]),
            int(row["end_frame"]),
            int(row["n_frames"]),
            float(row["duration_s"]),
            float(row["mean_speed_mm_s"]) if "mean_speed_mm_s" in row else None,
            float(row["max_speed_mm_s"]) if "max_speed_mm_s" in row else None,
            confidences[i],
            float(row["scut_z_mean"]) if "scut_z_mean" in row else None,
            ik_val,
        ])

    con.execute("CREATE INDEX idx_bouts_start ON bouts(start_frame)")
    con.execute("CREATE INDEX idx_bouts_status ON bouts(status)")

    # Verify
    result = con.execute("SELECT COUNT(*), MIN(id), MAX(id) FROM bouts").fetchone()
    print(f"  Inserted {result[0]} bouts (id {result[1]}..{result[2]})")

    result = con.execute("SELECT AVG(mean_confidence), MIN(mean_confidence), MAX(mean_confidence) FROM bouts").fetchone()
    print(f"  Confidence: avg={result[0]:.4f}, min={result[1]:.4f}, max={result[2]:.4f}")

    result = con.execute("SELECT COUNT(*) FROM bouts WHERE ik_mean_mm IS NOT NULL").fetchone()
    print(f"  Bouts with IK data: {result[0]}")

    con.close()
    print(f"  DuckDB written: {out_db} ({os.path.getsize(out_db) / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Metadata JSON
# ---------------------------------------------------------------------------

def write_meta(out_meta: Path, num_frames: int, bouts_csv: str, ik_csv: str):
    meta = {
        "format": "green_trajreader_v2",
        "binary": str(OUT_BIN),
        "duckdb": str(OUT_DB),
        "source_csv": DATA3D_CSV,
        "bouts_csv": bouts_csv,
        "ik_csv": ik_csv if os.path.exists(ik_csv) else None,
        "num_frames": num_frames,
        "num_keypoints": NUM_KEYPOINTS,
        "fps": FPS,
        "scale_factor": SCALE,
        "keypoint_names": FLY50_NAMES,
        "units": "mm",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata written: {out_meta}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Create Full Trajectory DB (TrajReader v2)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Write binary
    num_frames = write_binary(DATA3D_CSV, OUT_BIN)

    # Step 2: Create DuckDB
    create_duckdb(BOUTS_CSV, IK_CSV, OUT_BIN, num_frames, OUT_DB)

    # Step 3: Write metadata
    write_meta(OUT_META, num_frames, BOUTS_CSV, IK_CSV)

    print("\nDone!")


if __name__ == "__main__":
    main()
