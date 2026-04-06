#!/usr/bin/env python3
"""Compute IK residuals for fly walking bouts.

Runs Mason-style IK on each bout frame and outputs per-frame and per-bout
residuals. These can be used as a quality filter for bout selection.

Supports two model modes:
  --model default   : base fruitfly model at 0.82x global scale (fast, good enough)
  --model fitted    : adjustabodies-fitted .mjb (slightly better proportions)

Usage:
    python fly_bout_ik.py                          # default model, all bouts
    python fly_bout_ik.py --model fitted           # fitted model
    python fly_bout_ik.py --max-bouts 10           # first 10 bouts only

Output:
    bout_ik_residuals.csv — per-frame: bout_idx, frame, residual_mm, n_valid_sites
    bout_ik_summary.csv   — per-bout:  bout_idx, mean_res, median_res, max_res, ...
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/Users/johnsonr/src/adjustabodies')

from adjustabodies.species.fly import FLY50_SITE_DEFS, FLY50_SITES, FLY_CONFIG

# ── Paths ─────────────────────────────────────────────────────────────
DATA3D = "/Users/johnsonr/datasets/fly_April5/new_models_and_bout_prediction/predictions_S6_male/data3D.csv"
BOUTS_CSV = "/Users/johnsonr/datasets/fly_April5/fly_juan/bout_segmentation/walking_bouts_summary.csv"
JUAN_BOUTS_CSV = "/Users/johnsonr/datasets/fly_April5/new_models_and_bout_prediction/walking_bouts_summary.csv"
MJX_MODEL = "/Users/johnsonr/src/fruitfly/mjx/fruitfly_v2/assets/fruitfly.xml"
FITTED_MJB = "/Users/johnsonr/datasets/fly_April5/fly_adjustabodies/results/fruitfly_fitted.mjb"
OUTPUT_DIR = "/Users/johnsonr/datasets/fly_April5/fly_juan/bout_segmentation"

SCALE = 10.0       # JARVIS raw / 10 = mm
MODEL_SCALE = 0.82 # mm * 0.1 * 0.82 = model units (cm)


def load_model_default():
    """Load base fruitfly model with fly50 sites."""
    import mujoco
    spec = mujoco.MjSpec.from_file(MJX_MODEL)
    for name, body_name, pos in FLY50_SITE_DEFS:
        body = spec.body(body_name)
        if body:
            site = body.add_site()
            site.name = name
            site.pos = pos
    m = spec.compile()
    return m


def load_model_fitted():
    """Load adjustabodies-fitted model."""
    import mujoco
    m = mujoco.MjModel.from_binary_path(FITTED_MJB)
    return m


def build_site_map(m):
    """Map fly50 site names → model site IDs."""
    site_ids = {}
    for i in range(m.nsite):
        name = m.site(i).name
        if name in FLY50_SITES:
            site_ids[name] = i
    return site_ids


def solve_frame_mason(m, d, site_ids, kp_model, valid,
                       max_iter=200, lr_trans=0.001, lr_joints=1.0, beta=0.9):
    """Mason-style IK. Returns (per_site_residual_mm[50], mean_residual_mm)."""
    import mujoco

    nv = m.nv
    active = []
    for k in range(50):
        if not valid[k]:
            continue
        name = FLY50_SITES[k]
        if name not in site_ids:
            continue
        active.append((k, site_ids[name], kp_model[k]))

    if len(active) < 5:
        return np.full(50, np.nan), np.nan, 0

    # Per-DOF LR
    is_trans = np.zeros(nv, dtype=bool)
    for j in range(m.njnt):
        if m.jnt_type[j] == 0:
            da = int(m.jnt_dofadr[j])
            is_trans[da:da+3] = True
    lr_vec = np.full(nv, lr_joints)
    lr_vec[is_trans] = lr_trans

    # Reset + centroid align
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    tc = np.mean([t for _, _, t in active], axis=0)
    mc = np.mean([d.site_xpos[sid] for _, sid, _ in active], axis=0)
    for j in range(m.njnt):
        if m.jnt_type[j] == 0:
            qa = int(m.jnt_qposadr[j])
            d.qpos[qa:qa+3] += tc - mc
            break
    mujoco.mj_forward(m, d)

    N = len(active)
    vel = np.zeros(nv)
    J = np.zeros((3 * N, nv))
    jacp = np.zeros((3, nv))

    for it in range(max_iter):
        mujoco.mj_forward(m, d)
        residual = np.zeros(3 * N)
        for j, (_, sid, tgt) in enumerate(active):
            residual[3*j:3*j+3] = tgt - d.site_xpos[sid]
            jacp[:] = 0
            mujoco.mj_jacSite(m, d, jacp, None, sid)
            J[3*j:3*j+3] = jacp

        grad = J.T @ residual
        decay = 0.5 * (1 + math.cos(math.pi * it / max_iter))
        vel = beta * vel + lr_vec * decay * grad
        mujoco.mj_integratePos(m, d.qpos, vel, 1.0)

        for j in range(m.njnt):
            if m.jnt_type[j] == 3 and m.jnt_limited[j]:
                qa = int(m.jnt_qposadr[j])
                lo, hi = m.jnt_range[j]
                d.qpos[qa] = np.clip(d.qpos[qa], lo, hi)

    mujoco.mj_forward(m, d)

    per_site = np.full(50, np.nan)
    for k, sid, tgt in active:
        per_site[k] = np.linalg.norm(d.site_xpos[sid] - tgt)

    # Convert to mm
    if hasattr(m, '_is_fitted'):
        per_site_mm = per_site * 10  # fitted model already at correct scale
    else:
        per_site_mm = per_site / MODEL_SCALE * 10

    mean_mm = np.nanmean(per_site_mm)
    return per_site_mm, mean_mm, len(active)


def main():
    parser = argparse.ArgumentParser(description="IK residuals for fly walking bouts")
    parser.add_argument("--model", choices=["default", "fitted"], default="default")
    parser.add_argument("--max-bouts", type=int, default=0, help="0 = all bouts")
    parser.add_argument("--frames-per-bout", type=int, default=20,
                        help="Subsample frames per bout (0 = all)")
    parser.add_argument("--ik-iters", type=int, default=200)
    parser.add_argument("--juan-only", action="store_true",
                        help="Only process bouts overlapping with Juan's")
    parser.add_argument("--conf-threshold", type=float, default=0.8,
                        help="Min keypoint confidence to include in IK (default: 0.8)")
    args = parser.parse_args()

    import mujoco

    # Load model
    print(f"Loading model ({args.model})...")
    if args.model == "fitted":
        m = load_model_fitted()
        m._is_fitted = True
    else:
        m = load_model_default()
    d = mujoco.MjData(m)
    site_ids = build_site_map(m)
    print(f"  Sites mapped: {len(site_ids)}/50")

    # Load bouts
    bouts_df = pd.read_csv(BOUTS_CSV)
    if args.juan_only:
        juan_df = pd.read_csv(JUAN_BOUTS_CSV)
        overlap_idx = []
        for _, j in juan_df.iterrows():
            matches = bouts_df[(bouts_df.start_frame <= j.end_frame) &
                               (bouts_df.end_frame >= j.start_frame)]
            if len(matches) > 0:
                overlap_idx.append(matches.index[0])
        bouts_df = bouts_df.loc[sorted(set(overlap_idx))].reset_index(drop=True)
        print(f"  {len(bouts_df)} Juan-overlap bouts")
    if args.max_bouts > 0:
        bouts_df = bouts_df.head(args.max_bouts)
    print(f"  Processing {len(bouts_df)} bouts")

    # Load data3D column mapping
    df_header = pd.read_csv(DATA3D, nrows=0, low_memory=False)
    cols = df_header.columns.tolist()
    kp_col_map = {}
    for kp_name in FLY50_SITES:
        for i, c in enumerate(cols):
            if c.split('.')[0] == kp_name:
                kp_col_map[kp_name] = i
                break

    # Process bouts
    print(f"Loading data3D.csv and running IK ({args.ik_iters} iters)...")
    frame_results = []  # per-frame
    bout_results = []   # per-bout summary

    t0 = time.time()
    reader = pd.read_csv(DATA3D, skiprows=[1], low_memory=False, chunksize=100000, header=0)

    # Build set of all frames we need
    all_needed = {}
    for bi, row in bouts_df.iterrows():
        s, e = int(row['start_frame']), int(row['end_frame'])
        n = e - s + 1
        if args.frames_per_bout > 0 and n > args.frames_per_bout:
            frame_idx = np.linspace(s, e, args.frames_per_bout, dtype=int).tolist()
        else:
            frame_idx = list(range(s, e + 1))
        for fi in frame_idx:
            if fi not in all_needed:
                all_needed[fi] = []
            all_needed[fi].append(bi)

    max_frame = max(all_needed.keys())
    frame_data = {}  # frame_idx → (kp_mm[50,3], valid[50])

    row_offset = 0
    for chunk in reader:
        for local_idx in range(len(chunk)):
            global_idx = row_offset + local_idx
            if global_idx in all_needed:
                kp_mm = np.zeros((50, 3), dtype=np.float32)
                valid = np.zeros(50, dtype=bool)
                for k, kp_name in enumerate(FLY50_SITES):
                    ci = kp_col_map.get(kp_name)
                    if ci is None:
                        continue
                    try:
                        x = float(chunk.iloc[local_idx, ci]) / SCALE
                        y = float(chunk.iloc[local_idx, ci + 1]) / SCALE
                        z = float(chunk.iloc[local_idx, ci + 2]) / SCALE
                        conf = float(chunk.iloc[local_idx, ci + 3])
                        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                            kp_mm[k] = [x, y, z]
                            valid[k] = (conf >= args.conf_threshold)
                    except (ValueError, IndexError):
                        pass
                frame_data[global_idx] = (kp_mm, valid)
        row_offset += len(chunk)
        if row_offset > max_frame:
            break
        print(f"  Read {row_offset:,} rows, cached {len(frame_data)} frames...", end='\r')

    print(f"\n  Cached {len(frame_data)} frames in {time.time()-t0:.0f}s")

    # Run IK per bout
    for bi, row in bouts_df.iterrows():
        s, e = int(row['start_frame']), int(row['end_frame'])
        bout_idx = int(row['bout_idx'])
        n = e - s + 1
        if args.frames_per_bout > 0 and n > args.frames_per_bout:
            frame_idx = np.linspace(s, e, args.frames_per_bout, dtype=int).tolist()
        else:
            frame_idx = list(range(s, e + 1))

        bout_residuals = []
        for fi in frame_idx:
            if fi not in frame_data:
                continue
            kp_mm, valid = frame_data[fi]

            # Transform to model units
            if args.model == "fitted":
                kp_model = kp_mm * 0.082  # 0.82 baked into arena scale
            else:
                kp_model = kp_mm * 0.1 * MODEL_SCALE

            per_site, mean_mm, n_active = solve_frame_mason(
                m, d, site_ids, kp_model, valid, max_iter=args.ik_iters)

            frame_results.append({
                'bout_idx': bout_idx,
                'frame': fi,
                'residual_mm': mean_mm,
                'n_valid_sites': n_active,
            })
            if not math.isnan(mean_mm):
                bout_residuals.append(mean_mm)

        if bout_residuals:
            arr = np.array(bout_residuals)
            bout_results.append({
                'bout_idx': bout_idx,
                'start_frame': s,
                'end_frame': e,
                'n_frames': len(bout_residuals),
                'mean_residual_mm': np.mean(arr),
                'median_residual_mm': np.median(arr),
                'std_residual_mm': np.std(arr),
                'max_residual_mm': np.max(arr),
                'min_residual_mm': np.min(arr),
                'pct_below_0.5mm': 100 * np.mean(arr < 0.5),
                'duration_s': row['duration_s'],
                'mean_speed_mm_s': row['mean_speed_mm_s'],
            })

        if (bi + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Bout {bi+1}/{len(bouts_df)} — {elapsed:.0f}s elapsed")

    # Save
    frame_df = pd.DataFrame(frame_results)
    bout_df = pd.DataFrame(bout_results)

    frame_path = os.path.join(OUTPUT_DIR, "bout_ik_residuals.csv")
    bout_path = os.path.join(OUTPUT_DIR, "bout_ik_summary.csv")
    frame_df.to_csv(frame_path, index=False)
    bout_df.to_csv(bout_path, index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Per-frame: {frame_path} ({len(frame_df)} rows)")
    print(f"  Per-bout:  {bout_path} ({len(bout_df)} rows)")

    if len(bout_df) > 0:
        print(f"\n  IK residual summary:")
        print(f"    Mean:   {bout_df['mean_residual_mm'].mean():.3f} mm")
        print(f"    Median: {bout_df['median_residual_mm'].median():.3f} mm")
        print(f"    Best:   {bout_df['mean_residual_mm'].min():.3f} mm")
        print(f"    Worst:  {bout_df['mean_residual_mm'].max():.3f} mm")
        print(f"    <0.5mm: {bout_df['pct_below_0.5mm'].mean():.0f}% of frames")


if __name__ == "__main__":
    main()
