#!/usr/bin/env python3
"""Prepare fly bout data for adjustabodies fitting.

Extracts frames from JARVIS data3D.csv for the 64 bouts that overlap
with Juan's validated walking bouts. Outputs keypoints3d.csv in RED v2
format (frame, [x,y,z,conf] × 50 keypoints) and a mujoco_session.json
with the arena transform for the fly telecentric DLT rig.

Usage:
    python prepare_fly_bout_data.py

Output structure:
    /Users/johnsonr/datasets/fly_April5/fly_adjustabodies/
    ├── mujoco_session.json
    └── labeled_data/
        └── bout_frames/
            └── keypoints3d.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
DATA3D = Path("/Users/johnsonr/datasets/fly_April5/new_models_and_bout_prediction/predictions_S6_male/data3D.csv")
OUR_BOUTS = Path("/Users/johnsonr/datasets/fly_April5/fly_juan/bout_segmentation/walking_bouts_summary.csv")
JUAN_BOUTS = Path("/Users/johnsonr/datasets/fly_April5/new_models_and_bout_prediction/walking_bouts_summary.csv")
OUTPUT_DIR = Path("/Users/johnsonr/datasets/fly_April5/fly_adjustabodies")

# ── Config ────────────────────────────────────────────────────────────
SCALE = 10.0  # JARVIS raw → mm
# Fly arena: telecentric DLT, coordinates already in mm after /SCALE
# For MuJoCo: mm → meters (0.001), identity rotation (DLT frame ≈ MuJoCo frame)
# The arena transform maps from calibration mm → MuJoCo meters.
# For telecentric fly rigs, the coordinate system is already well-aligned.
# We use a 180° Z rotation to match the fly model's anterior direction.

# Maximum frames to sample per bout (for fitting, we don't need every frame)
MAX_FRAMES_PER_BOUT = 50  # ~3200 total for 64 bouts (plenty for adjustabodies)

# Fly50 keypoint names in order (must match FLY50_SITES in species/fly.py)
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


def find_juan_overlap_bouts():
    """Find our bouts that overlap with Juan's 65 validated bouts."""
    ours = pd.read_csv(OUR_BOUTS)
    juans = pd.read_csv(JUAN_BOUTS)

    overlap_indices = []
    for _, j in juans.iterrows():
        js, je = j.start_frame, j.end_frame
        matches = ours[(ours.start_frame <= je) & (ours.end_frame >= js)]
        if len(matches) > 0:
            overlap_indices.append(matches.index[0])

    overlap_bouts = ours.loc[sorted(set(overlap_indices))]
    print(f"Found {len(overlap_bouts)} bouts overlapping with Juan's {len(juans)}")
    return overlap_bouts


def load_data3d_columns():
    """Load data3D.csv header to find column indices for each keypoint."""
    df_header = pd.read_csv(DATA3D, nrows=0)
    cols = df_header.columns.tolist()
    # Map keypoint name → column index of its x column
    col_map = {}
    seen = set()
    idx = 0
    for col in cols:
        base = col.split('.')[0]
        if base not in seen:
            seen.add(base)
            if base in FLY50_NAMES:
                col_map[base] = idx
            idx += 1
        # Count raw column index
    # Actually, use column positions directly
    col_map = {}
    for kp_name in FLY50_NAMES:
        # Find the first column with this name
        for i, c in enumerate(cols):
            base = c.split('.')[0]
            if base == kp_name:
                col_map[kp_name] = i
                break
    return col_map


def extract_bout_frames(bouts_df, max_per_bout=MAX_FRAMES_PER_BOUT):
    """Extract frame indices from bouts, subsampling if needed."""
    all_frames = []
    for _, row in bouts_df.iterrows():
        s, e = int(row['start_frame']), int(row['end_frame'])
        n = e - s + 1
        if n <= max_per_bout:
            frames = list(range(s, e + 1))
        else:
            # Evenly spaced subsample
            frames = np.linspace(s, e, max_per_bout, dtype=int).tolist()
        all_frames.extend(frames)
    return sorted(set(all_frames))


def main():
    print("=== Preparing fly bout data for adjustabodies ===\n")

    # 1. Find overlapping bouts
    bouts = find_juan_overlap_bouts()

    # 2. Get frame indices
    frame_indices = extract_bout_frames(bouts)
    print(f"Total frames to extract: {len(frame_indices)}")

    # 3. Load data3D.csv (only the rows we need)
    print(f"Loading {DATA3D} ...")
    # Read header
    df_header = pd.read_csv(DATA3D, skiprows=[1], nrows=0, low_memory=False)
    cols = df_header.columns.tolist()

    # Build column index map for fly50 keypoints
    kp_col_indices = {}
    for kp_name in FLY50_NAMES:
        for i, c in enumerate(cols):
            base = c.split('.')[0]
            if base == kp_name:
                kp_col_indices[kp_name] = i  # x column
                break

    print(f"  Mapped {len(kp_col_indices)}/50 keypoints to columns")

    # Read full CSV (skip subheader row)
    # For 1.7M rows this takes ~30s, but we only need specific rows
    # Use chunked reading to extract specific frame indices
    frame_set = set(frame_indices)
    max_frame = max(frame_indices)

    # Read in chunks to save memory
    chunk_size = 100000
    kp_data = {}  # frame_idx → (50, 4) array [x, y, z, conf] in mm

    reader = pd.read_csv(DATA3D, skiprows=[1], low_memory=False,
                          chunksize=chunk_size, header=0)
    row_offset = 0
    for chunk in reader:
        for local_idx in range(len(chunk)):
            global_idx = row_offset + local_idx
            if global_idx in frame_set:
                vals = np.zeros((50, 4), dtype=np.float32)
                for k, kp_name in enumerate(FLY50_NAMES):
                    ci = kp_col_indices.get(kp_name)
                    if ci is None:
                        continue
                    try:
                        x = float(chunk.iloc[local_idx, ci]) / SCALE
                        y = float(chunk.iloc[local_idx, ci + 1]) / SCALE
                        z = float(chunk.iloc[local_idx, ci + 2]) / SCALE
                        conf = float(chunk.iloc[local_idx, ci + 3])
                        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                            vals[k] = [x, y, z, conf]
                    except (ValueError, IndexError):
                        pass
                kp_data[global_idx] = vals
            if global_idx > max_frame:
                break
        row_offset += len(chunk)
        if row_offset > max_frame:
            break
        print(f"  Read {row_offset:,} rows, extracted {len(kp_data)} frames ...", end='\r')

    print(f"\n  Extracted {len(kp_data)} frames")

    # 4. Write keypoints3d.csv
    output_dir = OUTPUT_DIR / "labeled_data" / "bout_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "keypoints3d.csv"

    # Header: frame, [kp_x, kp_y, kp_z, kp_conf] × 50
    header_parts = ["frame"]
    for kp_name in FLY50_NAMES:
        header_parts.extend([f"{kp_name}_x", f"{kp_name}_y", f"{kp_name}_z", f"{kp_name}_conf"])

    with open(csv_path, 'w') as f:
        f.write(','.join(header_parts) + '\n')
        for frame_idx in sorted(kp_data.keys()):
            vals = kp_data[frame_idx]
            parts = [str(frame_idx)]
            for k in range(50):
                x, y, z, conf = vals[k]
                if x == 0 and y == 0 and z == 0 and conf == 0:
                    parts.extend(['nan', 'nan', 'nan', '0'])
                else:
                    parts.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{conf:.4f}"])
            f.write(','.join(parts) + '\n')

    print(f"  Wrote {csv_path}")

    # 5. Write mujoco_session.json
    # Fly telecentric DLT: coordinates are in mm.
    # MuJoCo model is in meters. Arena transform: mm → m with identity rotation.
    # The fly model (fruitfly.xml) has anterior = +Y in body frame.
    # DLT predictions: X = long axis of arena, Y = width, Z = height.
    # For the fly model: X_mujoco ≈ Y_dlt, Y_mujoco ≈ X_dlt, Z_mujoco = Z_dlt
    # But this depends on the specific rig. For now use identity + scale;
    # adjustabodies will optimize position/orientation via IK.
    session = {
        "arena": {
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [0, 0, 0],
            "scale": 0.001  # mm → meters
        },
        "species": "fly",
        "model_xml": "fruitfly.xml",
        "notes": "Fly telecentric DLT, arena 23.5 x 5.5 mm, data from JARVIS predictions"
    }

    session_path = OUTPUT_DIR / "mujoco_session.json"
    with open(session_path, 'w') as f:
        json.dump(session, f, indent=2)
    print(f"  Wrote {session_path}")

    # 6. Summary
    print(f"\n=== Summary ===")
    print(f"  Bouts: {len(bouts)} (Juan-overlap)")
    print(f"  Frames: {len(kp_data)}")
    print(f"  Keypoints: 50 (Fly50)")
    print(f"  Output: {OUTPUT_DIR}")

    # Quick data quality check
    valid_counts = []
    for frame_idx in sorted(kp_data.keys()):
        vals = kp_data[frame_idx]
        n_valid = np.sum(np.any(vals[:, :3] != 0, axis=1))
        valid_counts.append(n_valid)
    vc = np.array(valid_counts)
    print(f"  Valid keypoints per frame: {vc.mean():.1f} ± {vc.std():.1f} (min={vc.min()}, max={vc.max()})")
    print(f"\nReady for: adjustabodies-fit --data-dir {OUTPUT_DIR} --model-xml fruitfly.xml")


if __name__ == "__main__":
    main()
