#!/usr/bin/env python3
"""Walking bout segmentation v2 — with IK body model validation.

Finds contiguous walking sequences from JARVIS 3D predictions using:
  1. Prediction confidence (all keypoints > threshold)
  2. Upright posture (scutellum above leg tips = on ground surface)
  3. Arena boundaries (not on walls)
  4. IK body model fit (per-frame residual < threshold)
  5. Minimum duration

The fly spends most of its time NOT walking on the ground. This pipeline
finds the rare contiguous walking periods using Juan's filter approach
(Sandbox_Strict.ipynb) plus physics-based IK validation.

Usage:
    python bout_segmentation_v2.py /path/to/predictions_folder
    python bout_segmentation_v2.py /path/to/predictions_folder --ik  # enable IK filter
    python bout_segmentation_v2.py /path/to/predictions_folder --min-duration 0.2
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

sys.path.insert(0, '/Users/johnsonr/src/adjustabodies')

# ── Fly50 keypoint names ─────────────────────────────────────────────

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

LEG_TIPS = ["T1L_TaTip", "T1R_TaTip", "T2L_TaTip", "T2R_TaTip", "T3L_TaTip", "T3R_TaTip"]
X_WALL_TIPS = ["T1L_TaTip", "T1R_TaTip", "T3L_TaTip", "T3R_TaTip"]


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Data
    scale: float = 10.0     # JARVIS raw / scale = mm
    fps: int = 800

    # Arena (rectangular chamber, mm)
    arena_x_mm: float = 23.5
    arena_y_mm: float = 5.5

    # Frame-level filters
    conf_threshold: float = 0.80   # all keypoints must exceed
    conf_gap_bridge: int = 15      # bridge short confidence dips
    upright: bool = True           # scutellum Z > all leg tips
    floor_z_mm: float = 0.40      # min leg Z (reject if below)
    y_wall_max: float = 5.4
    y_wall_min: float = 0.0
    x_wall_margin: float = 0.0

    # IK filter (optional, slower but physics-based)
    ik_enabled: bool = False
    ik_max_residual_mm: float = 0.8   # reject frames above this
    ik_iters: int = 200
    ik_model_scale: float = 0.82      # global body model scale
    ik_subsample: int = 4             # only run IK every Nth frame, interpolate

    # Immobility splitting
    immobility_max_frames: int = 25
    immobility_speed_mm_s: float = 5.0

    # Bout detection
    min_duration_s: float = 0.125     # minimum bout length (100 frames at 800Hz)
    max_gap_bridge: int = 100         # bridge gaps in valid mask

    # Walking validation
    min_walking_cycles: int = 2
    min_distance_mm: float = 5.0
    swing_max_duration: int = 35
    swing_prominence: float = 0.05


# ── Data loading ──────────────────────────────────────────────────────

def load_predictions(csv_path, scale):
    """Load data3D.csv → dict of {kp_name: {x, y, z, conf}} arrays in mm."""
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    if df.iloc[-1].isna().all():
        df = df.iloc[:-1].reset_index(drop=True)

    cols = df.columns.tolist()
    kp_data = {}
    for kp_name in FLY50_NAMES:
        for i, c in enumerate(cols):
            if c.split('.')[0] == kp_name:
                kp_data[kp_name] = {
                    'x': df.iloc[:, i].to_numpy(dtype=float) / scale,
                    'y': df.iloc[:, i+1].to_numpy(dtype=float) / scale,
                    'z': df.iloc[:, i+2].to_numpy(dtype=float) / scale,
                    'conf': df.iloc[:, i+3].to_numpy(dtype=float),
                }
                break
    return df, kp_data


# ── Frame-level filters ──────────────────────────────────────────────

def bridge_gaps(mask, max_gap):
    if max_gap <= 0:
        return mask
    out = mask.copy()
    in_gap, gap_start = False, 0
    for i in range(len(out)):
        if not out[i]:
            if not in_gap: gap_start = i; in_gap = True
        elif in_gap:
            if (i - gap_start) <= max_gap and gap_start > 0:
                out[gap_start:i] = True
            in_gap = False
    return out


def compute_valid_mask(kp_data, n_frames, cfg):
    """Per-frame boolean mask: True = candidate walking frame."""
    # 1. Confidence: all keypoints >= threshold
    conf_mask = np.ones(n_frames, dtype=bool)
    for kp_name, d in kp_data.items():
        conf_mask &= (d['conf'] >= cfg.conf_threshold)
    conf_mask = bridge_gaps(conf_mask, cfg.conf_gap_bridge)

    # 2. Upright posture: scutellum Z > all leg tip Z
    upright_mask = np.ones(n_frames, dtype=bool)
    if cfg.upright:
        scut_z = kp_data['Scutellum']['z']
        for tip in LEG_TIPS:
            upright_mask &= (scut_z > kp_data[tip]['z'])

    # 3. Floor Z: any leg below threshold → reject
    floor_mask = np.ones(n_frames, dtype=bool)
    for tip in LEG_TIPS:
        floor_mask &= (kp_data[tip]['z'] >= cfg.floor_z_mm)

    # 4. Arena boundaries
    wall_mask = np.ones(n_frames, dtype=bool)
    for tip in LEG_TIPS:
        wall_mask &= (kp_data[tip]['y'] < cfg.y_wall_max)
        wall_mask &= (kp_data[tip]['y'] > cfg.y_wall_min)
    for tip in X_WALL_TIPS:
        wall_mask &= (kp_data[tip]['x'] < cfg.arena_x_mm - cfg.x_wall_margin)
        wall_mask &= (kp_data[tip]['x'] > cfg.x_wall_margin)

    valid = conf_mask & upright_mask & floor_mask & wall_mask

    stats = {
        'n_frames': n_frames,
        'confidence': int(conf_mask.sum()),
        'upright': int(upright_mask.sum()),
        'floor': int(floor_mask.sum()),
        'walls': int(wall_mask.sum()),
        'combined': int(valid.sum()),
    }
    return valid, stats


# ── IK filter ─────────────────────────────────────────────────────────

def compute_ik_mask(kp_data, valid_mask, n_frames, cfg):
    """Run IK on valid frames and reject those with high residual.

    Only runs on every cfg.ik_subsample-th valid frame, then interpolates
    the pass/fail decision to neighboring frames for speed.
    """
    import mujoco
    from adjustabodies.species.fly import FLY50_SITE_DEFS

    # Load model
    spec = mujoco.MjSpec.from_file('/Users/johnsonr/src/fruitfly/mjx/fruitfly_v2/assets/fruitfly.xml')
    for name, body_name, pos in FLY50_SITE_DEFS:
        body = spec.body(body_name)
        if body:
            site = body.add_site(); site.name = name; site.pos = pos
    m = spec.compile()
    d = mujoco.MjData(m)

    site_ids = {}
    for i in range(m.nsite):
        if m.site(i).name in FLY50_NAMES:
            site_ids[m.site(i).name] = i

    nv = m.nv
    is_trans = np.zeros(nv, dtype=bool)
    for j in range(m.njnt):
        if m.jnt_type[j] == 0:
            da = int(m.jnt_dofadr[j])
            is_trans[da:da+3] = True
    lr_vec = np.full(nv, 1.0)
    lr_vec[is_trans] = 0.001

    # Get frames to evaluate
    valid_indices = np.where(valid_mask)[0]
    eval_indices = valid_indices[::cfg.ik_subsample]

    ik_residual = np.full(n_frames, np.nan)
    n_eval = len(eval_indices)

    for idx_i, fi in enumerate(eval_indices):
        # Build targets: only high-confidence keypoints
        kp_model = np.zeros((50, 3))
        kp_valid = np.zeros(50, dtype=bool)
        for k, kp_name in enumerate(FLY50_NAMES):
            dd = kp_data[kp_name]
            if dd['conf'][fi] >= cfg.conf_threshold:
                kp_model[k] = [dd['x'][fi], dd['y'][fi], dd['z'][fi]]
                kp_model[k] *= 0.1 * cfg.ik_model_scale  # mm → model units
                kp_valid[k] = True

        targets = [(site_ids[FLY50_NAMES[k]], kp_model[k])
                    for k in range(50) if kp_valid[k] and FLY50_NAMES[k] in site_ids]
        if len(targets) < 10:
            continue

        # Mason-style IK
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        tc = np.mean([t for _, t in targets], axis=0)
        mc = np.mean([d.site_xpos[sid] for sid, _ in targets], axis=0)
        for j in range(m.njnt):
            if m.jnt_type[j] == 0:
                qa = int(m.jnt_qposadr[j])
                d.qpos[qa:qa+3] += tc - mc
                break

        vel = np.zeros(nv)
        N = len(targets)
        J = np.zeros((3*N, nv))
        jacp = np.zeros((3, nv))

        for it in range(cfg.ik_iters):
            mujoco.mj_forward(m, d)
            residual = np.zeros(3*N)
            for j, (sid, tgt) in enumerate(targets):
                residual[3*j:3*j+3] = tgt - d.site_xpos[sid]
                jacp[:] = 0
                mujoco.mj_jacSite(m, d, jacp, None, sid)
                J[3*j:3*j+3] = jacp
            grad = J.T @ residual
            decay = 0.5 * (1 + math.cos(math.pi * it / cfg.ik_iters))
            vel = 0.9 * vel + lr_vec * decay * grad
            mujoco.mj_integratePos(m, d.qpos, vel, 1.0)
            for j in range(m.njnt):
                if m.jnt_type[j] == 3 and m.jnt_limited[j]:
                    qa = int(m.jnt_qposadr[j])
                    lo, hi = m.jnt_range[j]
                    d.qpos[qa] = np.clip(d.qpos[qa], lo, hi)

        mujoco.mj_forward(m, d)
        res = [np.linalg.norm(d.site_xpos[sid] - tgt) for sid, tgt in targets]
        ik_residual[fi] = np.mean(res) / cfg.ik_model_scale * 10  # → mm

        if (idx_i + 1) % 500 == 0:
            print(f"    IK: {idx_i+1}/{n_eval} frames evaluated")

    # Interpolate: for non-evaluated valid frames, use nearest evaluated neighbor
    evaluated = np.where(np.isfinite(ik_residual))[0]
    if len(evaluated) > 0:
        for fi in valid_indices:
            if np.isnan(ik_residual[fi]):
                nearest = evaluated[np.argmin(np.abs(evaluated - fi))]
                if abs(nearest - fi) <= cfg.ik_subsample * 2:
                    ik_residual[fi] = ik_residual[nearest]

    ik_pass = np.isfinite(ik_residual) & (ik_residual < cfg.ik_max_residual_mm)
    # For frames we couldn't evaluate, keep them if they passed other filters
    ik_pass |= (valid_mask & np.isnan(ik_residual))

    n_evaluated = np.isfinite(ik_residual).sum()
    n_pass = (np.isfinite(ik_residual) & (ik_residual < cfg.ik_max_residual_mm)).sum()
    print(f"    IK: {n_evaluated} evaluated, {n_pass} pass (<{cfg.ik_max_residual_mm}mm)")
    if n_evaluated > 0:
        print(f"    IK residual: median={np.nanmedian(ik_residual[np.isfinite(ik_residual)]):.3f}mm")

    return ik_pass, ik_residual


# ── Bout detection ────────────────────────────────────────────────────

def find_bouts(valid_mask, min_frames, max_gap):
    bridged = bridge_gaps(valid_mask, max_gap)
    bouts = []
    in_bout, start = False, 0
    for i, v in enumerate(bridged):
        if v and not in_bout:
            start = i; in_bout = True
        elif not v and in_bout:
            if i - start >= min_frames: bouts.append((start, i - 1))
            in_bout = False
    if in_bout and len(bridged) - start >= min_frames:
        bouts.append((start, len(bridged) - 1))
    return bouts


def split_at_immobility(bouts, kp_data, cfg):
    result = []
    for start, end in bouts:
        x = kp_data['Scutellum']['x'][start:end+1]
        y = kp_data['Scutellum']['y'][start:end+1]
        spd = np.sqrt(np.diff(x)**2 + np.diff(y)**2) * cfg.fps
        spd = np.concatenate([[spd[0] if len(spd) > 0 else 0], spd])
        is_stat = spd < cfg.immobility_speed_mm_s

        viol = np.zeros(len(spd), dtype=bool)
        run_len, run_start = 0, 0
        for i, s in enumerate(is_stat):
            if s:
                if run_len == 0: run_start = i
                run_len += 1
            else:
                if run_len > cfg.immobility_max_frames:
                    viol[run_start:run_start+run_len] = True
                run_len = 0
        if run_len > cfg.immobility_max_frames:
            viol[run_start:run_start+run_len] = True

        min_frames = int(cfg.min_duration_s * cfg.fps)
        in_valid, sub_start = False, 0
        for i, v in enumerate(~viol):
            if v and not in_valid: sub_start = i; in_valid = True
            elif not v and in_valid:
                if i - sub_start >= min_frames:
                    result.append((start + sub_start, start + i - 1))
                in_valid = False
        if in_valid and (len(viol) - sub_start) >= min_frames:
            result.append((start + sub_start, end))
    return result


def count_swings(z, cfg):
    z_clean = z.copy()
    nan_mask = np.isnan(z_clean)
    if nan_mask.all(): return 0
    if nan_mask.any():
        vi = np.where(~nan_mask)[0]
        z_clean[nan_mask] = np.interp(np.where(nan_mask)[0], vi, z_clean[vi])
    z_smooth = uniform_filter1d(z_clean, size=5)
    peaks, _ = find_peaks(z_smooth, prominence=cfg.swing_prominence,
                           distance=8, width=(1, cfg.swing_max_duration))
    return len(peaks)


def validate_bouts(bouts, kp_data, cfg):
    valid, rejected = [], []
    for start, end in bouts:
        n = end - start + 1

        min_cycles = min(
            count_swings(kp_data[tip]['z'][start:end+1], cfg)
            for tip in LEG_TIPS)
        if min_cycles < cfg.min_walking_cycles:
            rejected.append({'start': start, 'end': end, 'reason': f'cycles({min_cycles})'})
            continue

        sx = kp_data['Scutellum']['x'][start:end+1]
        sy = kp_data['Scutellum']['y'][start:end+1]
        ok = np.isfinite(sx) & np.isfinite(sy)
        if ok.sum() < 2:
            rejected.append({'start': start, 'end': end, 'reason': 'no_data'})
            continue
        xv, yv = sx[ok], sy[ok]
        total_dist = np.sum(np.sqrt(np.diff(xv)**2 + np.diff(yv)**2))
        net_disp = np.sqrt((xv[-1]-xv[0])**2 + (yv[-1]-yv[0])**2)
        if total_dist < cfg.min_distance_mm:
            rejected.append({'start': start, 'end': end, 'reason': f'dist({total_dist:.1f})'})
            continue

        spd = np.sqrt(np.diff(sx)**2 + np.diff(sy)**2) * cfg.fps
        spd = np.concatenate([[spd[0] if len(spd) > 0 else 0], spd])
        scut_z = kp_data['Scutellum']['z'][start:end+1]

        valid.append({
            'start_frame': start, 'end_frame': end, 'n_frames': n,
            'duration_s': n / cfg.fps,
            'min_cycles': min_cycles,
            'total_distance_mm': float(total_dist),
            'net_displacement_mm': float(net_disp),
            'mean_speed_mm_s': float(np.nanmean(spd)),
            'max_speed_mm_s': float(np.nanmax(spd)),
            'scut_z_mean': float(np.nanmean(scut_z)),
            'scut_z_std': float(np.nanstd(scut_z)),
        })
    return valid, rejected


# ── Main pipeline ─────────────────────────────────────────────────────

def run(data_dir, cfg, fly_id="", output_dir=None):
    csv_path = Path(data_dir) / "data3D.csv"
    output_dir = Path(output_dir or data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Loading {csv_path}...")
    df, kp_data = load_predictions(csv_path, cfg.scale)
    n_frames = len(df)
    print(f"  {n_frames:,} frames ({n_frames/cfg.fps:.1f}s at {cfg.fps}Hz)")

    # Frame-level filters
    print("Frame-level filters...")
    valid, stats = compute_valid_mask(kp_data, n_frames, cfg)
    for k, v in stats.items():
        if k != 'n_frames':
            print(f"  {k:12s}: {v:>10,} ({100*v/n_frames:.1f}%)")

    # Optional IK filter
    if cfg.ik_enabled:
        print("IK body model filter...")
        ik_pass, ik_residual = compute_ik_mask(kp_data, valid, n_frames, cfg)
        n_before = valid.sum()
        valid &= ik_pass
        print(f"  IK filter: {n_before:,} → {valid.sum():,} frames")

    # Find contiguous bouts
    min_frames = int(cfg.min_duration_s * cfg.fps)
    print(f"Finding bouts (min {cfg.min_duration_s}s = {min_frames} frames)...")
    candidates = find_bouts(valid, min_frames, cfg.max_gap_bridge)
    print(f"  {len(candidates)} candidates")

    # Split at immobility
    candidates = split_at_immobility(candidates, kp_data, cfg)
    print(f"  {len(candidates)} after immobility split")

    # Validate walking
    print("Validating walking...")
    valid_bouts, rejected = validate_bouts(candidates, kp_data, cfg)
    print(f"  {len(valid_bouts)} valid, {len(rejected)} rejected")

    # Build DataFrame
    rows = [{'fly_id': fly_id, 'bout_idx': i+1, **b} for i, b in enumerate(valid_bouts)]
    bouts_df = pd.DataFrame(rows)

    # Save
    csv_out = output_dir / "walking_bouts_summary.csv"
    bouts_df.to_csv(csv_out, index=False)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"  Bouts: {len(bouts_df)}")
    if len(bouts_df) > 0:
        print(f"  Duration: {bouts_df['duration_s'].min():.2f} – {bouts_df['duration_s'].max():.2f}s")
        print(f"  Speed:    {bouts_df['mean_speed_mm_s'].min():.1f} – {bouts_df['mean_speed_mm_s'].max():.1f} mm/s")
    if rejected:
        reasons = {}
        for r in rejected:
            k = r['reason'].split('(')[0]
            reasons[k] = reasons.get(k, 0) + 1
        print(f"  Rejected: {dict(reasons)}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Saved: {csv_out}")

    return bouts_df, rejected


def main():
    parser = argparse.ArgumentParser(description="Walking bout segmentation v2")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--fly-id", default="")
    parser.add_argument("--fps", type=int, default=800)
    parser.add_argument("--min-duration", type=float, default=0.125,
                        help="Minimum bout duration in seconds (default: 0.125 = 100 frames)")
    parser.add_argument("--conf", type=float, default=0.80)
    parser.add_argument("--ik", action="store_true", help="Enable IK body model filter")
    parser.add_argument("--ik-threshold", type=float, default=0.8,
                        help="Max IK residual in mm (default: 0.8)")
    parser.add_argument("--ik-subsample", type=int, default=4,
                        help="Run IK every Nth valid frame (default: 4)")
    parser.add_argument("--min-cycles", type=int, default=2)
    parser.add_argument("--min-distance", type=float, default=5.0)
    args = parser.parse_args()

    cfg = Config(
        fps=args.fps,
        conf_threshold=args.conf,
        min_duration_s=args.min_duration,
        min_walking_cycles=args.min_cycles,
        min_distance_mm=args.min_distance,
        ik_enabled=args.ik,
        ik_max_residual_mm=args.ik_threshold,
        ik_subsample=args.ik_subsample,
    )

    run(args.data_dir, cfg, fly_id=args.fly_id, output_dir=args.output)


if __name__ == "__main__":
    main()
