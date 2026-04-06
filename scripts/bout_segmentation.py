#!/usr/bin/env python3
"""Walking bout segmentation for fly 3D pose predictions.

Detects walking bouts from JARVIS 3D prediction data (data3D.csv) using:
  1. Frame-level quality filters (confidence, posture, arena boundaries)
  2. Contiguous-region detection with gap bridging
  3. Walking validation (swing cycles, distance, speed)

Outputs a walking_bouts_summary.csv compatible with Juan's curation GUI.

Usage:
    python bout_segmentation.py /path/to/predictions_folder [--output /path/to/output]
    python bout_segmentation.py /path/to/predictions_folder --visualize
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class BoutConfig:
    """All tunable parameters in one place."""
    # Data
    scale: float = 10.0        # raw coords / scale = mm
    fps: int = 800

    # Arena (rectangular fly chamber, mm)
    arena_x_mm: float = 23.5
    arena_y_mm: float = 5.5

    # Confidence filter
    confidence_enabled: bool = True
    confidence_threshold: float = 0.80
    confidence_gap_bridge: int = 15  # bridge gaps <= this many frames

    # Upright posture: scutellum Z > all leg tip Z
    upright_enabled: bool = True

    # Floor Z: reject frames where any leg Z < threshold
    floor_z_enabled: bool = True
    floor_z_threshold: float = 0.40  # mm

    # Arena boundary filters (split bouts at wall contacts)
    y_wall_max: float = 5.4
    y_wall_min: float = 0.0
    x_wall_margin: float = 0.0

    # Immobility: split bouts at prolonged stops
    immobility_max_frames: int = 25
    immobility_speed_threshold: float = 5.0  # mm/s

    # Bout detection
    min_bout_frames: int = 100
    max_gap_bridge: int = 100  # bridge gaps in valid mask

    # Walking validation
    min_walking_cycles: int = 2
    min_distance_mm: float = 5.0
    max_swing_duration: int = 35
    swing_prominence: float = 0.05  # mm

    # Keypoints
    scutellum: str = "Scutellum"
    leg_tips: List[str] = field(default_factory=lambda: [
        "T1L_TaTip", "T1R_TaTip", "T2L_TaTip", "T2R_TaTip",
        "T3L_TaTip", "T3R_TaTip",
    ])
    x_wall_tips: List[str] = field(default_factory=lambda: [
        "T1L_TaTip", "T1R_TaTip", "T3L_TaTip", "T3R_TaTip",
    ])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data3d(csv_path: Path, scale: float) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """Load data3D.csv and extract leg tip + scutellum arrays.

    Returns (df, leg_tip_data, scutellum_data, all_conf)
    where leg_tip_data[tip] = {'x','y','z','conf'} as numpy arrays in mm.
    """
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    # Drop trailing summary row if present
    if df.iloc[-1].isna().all():
        df = df.iloc[:-1].reset_index(drop=True)

    cols = df.columns.tolist()

    def extract(kp_name):
        idx = cols.index(kp_name)
        x = df.iloc[:, idx].to_numpy(dtype=float) / scale
        y = df.iloc[:, idx + 1].to_numpy(dtype=float) / scale
        z = df.iloc[:, idx + 2].to_numpy(dtype=float) / scale
        conf = df.iloc[:, idx + 3].to_numpy(dtype=float)
        return {'x': x, 'y': y, 'z': z, 'conf': conf}

    cfg = BoutConfig()  # just for keypoint names
    leg_tip_data = {tip: extract(tip) for tip in cfg.leg_tips}
    scutellum_data = extract(cfg.scutellum)

    # All keypoint confidences (for the global confidence filter)
    seen = set()
    all_conf = {}
    for col in cols:
        base = col.split('.')[0]
        if base not in seen:
            seen.add(base)
            try:
                all_conf[base] = extract(base)['conf']
            except (ValueError, IndexError):
                pass

    return df, leg_tip_data, scutellum_data, all_conf


# ── Frame-level filters ──────────────────────────────────────────────────────

def bridge_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Bridge short False-runs bounded by True on both sides."""
    if max_gap <= 0:
        return mask
    out = mask.copy()
    in_gap = False
    gap_start = 0
    for i in range(len(out)):
        if not out[i]:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                if (i - gap_start) <= max_gap and gap_start > 0:
                    out[gap_start:i] = True
                in_gap = False
    return out


def compute_filter_masks(
    n_frames: int,
    leg_tip_data: Dict,
    scutellum_data: Dict,
    all_conf: Dict,
    cfg: BoutConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute per-frame valid mask and individual filter masks."""

    # Confidence: ALL keypoints >= threshold
    conf_mask = np.ones(n_frames, dtype=bool)
    if cfg.confidence_enabled:
        for conf_arr in all_conf.values():
            conf_mask &= (conf_arr >= cfg.confidence_threshold)
        conf_mask = bridge_gaps(conf_mask, cfg.confidence_gap_bridge)

    # Upright posture
    upright_mask = np.ones(n_frames, dtype=bool)
    if cfg.upright_enabled:
        for tip in cfg.leg_tips:
            upright_mask &= (scutellum_data['z'] > leg_tip_data[tip]['z'])

    # Floor Z violations
    floor_viol = np.zeros(n_frames, dtype=bool)
    if cfg.floor_z_enabled:
        for tip in cfg.leg_tips:
            floor_viol |= (leg_tip_data[tip]['z'] < cfg.floor_z_threshold)

    # Y-wall violations
    y_wall_viol = np.zeros(n_frames, dtype=bool)
    for tip in cfg.leg_tips:
        y = leg_tip_data[tip]['y']
        y_wall_viol |= (y >= cfg.y_wall_max)
        y_wall_viol |= (y <= cfg.y_wall_min)

    # X-wall violations (front + back legs at arena edges)
    x_wall_viol = np.zeros(n_frames, dtype=bool)
    for tip in cfg.x_wall_tips:
        x = leg_tip_data[tip]['x']
        x_wall_viol |= (x >= cfg.arena_x_mm - cfg.x_wall_margin)
        x_wall_viol |= (x <= cfg.x_wall_margin)

    valid = conf_mask & upright_mask & ~floor_viol & ~y_wall_viol & ~x_wall_viol

    masks = {
        'confidence': conf_mask,
        'upright': upright_mask,
        'floor_violation': floor_viol,
        'y_wall_violation': y_wall_viol,
        'x_wall_violation': x_wall_viol,
    }

    return valid, masks


# ── Bout detection ────────────────────────────────────────────────────────────

def find_bouts(valid_mask: np.ndarray, min_frames: int, max_gap: int) -> List[Tuple[int, int]]:
    """Find contiguous valid regions, bridging small gaps."""
    bridged = bridge_gaps(valid_mask, max_gap)
    bouts = []
    in_bout = False
    start = 0
    for i, v in enumerate(bridged):
        if v and not in_bout:
            start = i
            in_bout = True
        elif not v and in_bout:
            if i - start >= min_frames:
                bouts.append((start, i - 1))
            in_bout = False
    if in_bout and len(bridged) - start >= min_frames:
        bouts.append((start, len(bridged) - 1))
    return bouts


def compute_speed(scutellum_data: Dict, start: int, end: int, fps: int) -> np.ndarray:
    """Instantaneous speed (mm/s) for frames [start, end]."""
    x = scutellum_data['x'][start:end + 1]
    y = scutellum_data['y'][start:end + 1]
    dx, dy = np.diff(x), np.diff(y)
    spd = np.sqrt(dx**2 + dy**2) * fps
    return np.concatenate([[spd[0] if len(spd) > 0 else 0], spd])


def split_at_immobility(
    bouts: List[Tuple[int, int]],
    scutellum_data: Dict,
    cfg: BoutConfig,
) -> List[Tuple[int, int]]:
    """Split bouts where the fly is stationary for too long."""
    result = []
    for start, end in bouts:
        spd = compute_speed(scutellum_data, start, end, cfg.fps)
        is_stat = spd < cfg.immobility_speed_threshold
        # Find stationary runs > threshold and mark as violations
        viol = np.zeros(len(spd), dtype=bool)
        run_len = 0
        run_start = 0
        for i, s in enumerate(is_stat):
            if s:
                if run_len == 0:
                    run_start = i
                run_len += 1
            else:
                if run_len > cfg.immobility_max_frames:
                    viol[run_start:run_start + run_len] = True
                run_len = 0
        if run_len > cfg.immobility_max_frames:
            viol[run_start:run_start + run_len] = True

        # Split at violations
        in_valid = False
        sub_start = 0
        for i, v in enumerate(~viol):
            if v and not in_valid:
                sub_start = i
                in_valid = True
            elif not v and in_valid:
                if i - sub_start >= cfg.min_bout_frames:
                    result.append((start + sub_start, start + i - 1))
                in_valid = False
        if in_valid and (len(viol) - sub_start) >= cfg.min_bout_frames:
            result.append((start + sub_start, end))
    return result


# ── Walking validation ────────────────────────────────────────────────────────

def count_swing_phases(z_signal: np.ndarray, cfg: BoutConfig) -> int:
    """Count swing (leg-lift) peaks in a Z trace."""
    z = z_signal.copy()
    nan_mask = np.isnan(z)
    if nan_mask.all():
        return 0
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        z[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, z[valid_idx])
    z_smooth = uniform_filter1d(z, size=5)
    peaks, _ = find_peaks(
        z_smooth,
        prominence=cfg.swing_prominence,
        distance=8,
        width=(1, cfg.max_swing_duration),
    )
    return len(peaks)


def validate_bouts(
    bouts: List[Tuple[int, int]],
    leg_tip_data: Dict,
    scutellum_data: Dict,
    cfg: BoutConfig,
) -> Tuple[List[dict], List[dict]]:
    """Validate candidate bouts. Returns (valid, rejected)."""
    valid, rejected = [], []

    for start, end in bouts:
        n = end - start + 1

        # Swing cycles per leg
        min_cycles = min(
            count_swing_phases(leg_tip_data[tip]['z'][start:end + 1], cfg)
            for tip in cfg.leg_tips
        )
        if min_cycles < cfg.min_walking_cycles:
            rejected.append({'start': start, 'end': end, 'reason': f'too few cycles ({min_cycles})'})
            continue

        # Distance
        x = scutellum_data['x'][start:end + 1]
        y = scutellum_data['y'][start:end + 1]
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 2:
            rejected.append({'start': start, 'end': end, 'reason': 'insufficient valid frames'})
            continue
        xv, yv = x[ok], y[ok]
        total_dist = np.sum(np.sqrt(np.diff(xv)**2 + np.diff(yv)**2))
        net_disp = np.sqrt((xv[-1] - xv[0])**2 + (yv[-1] - yv[0])**2)
        if total_dist < cfg.min_distance_mm:
            rejected.append({'start': start, 'end': end, 'reason': f'too short ({total_dist:.1f}mm)'})
            continue

        spd = compute_speed(scutellum_data, start, end, cfg.fps)
        scut_z = scutellum_data['z'][start:end + 1]

        # Compute height-speed correlation
        from scipy.stats import pearsonr, spearmanr
        finite = np.isfinite(scut_z) & np.isfinite(spd)
        if finite.sum() > 10:
            pr, pp = pearsonr(scut_z[finite], spd[finite])
            sr, sp = spearmanr(scut_z[finite], spd[finite])
        else:
            pr, pp, sr, sp = 0, 1, 0, 1

        valid.append({
            'start_frame': start,
            'end_frame': end,
            'n_frames': n,
            'duration_s': n / cfg.fps,
            'min_cycles': min_cycles,
            'total_distance_mm': total_dist,
            'net_displacement_mm': net_disp,
            'mean_speed_mm_s': float(np.nanmean(spd)),
            'max_speed_mm_s': float(np.nanmax(spd)),
            'scut_z_mean': float(np.nanmean(scut_z)),
            'scut_z_std': float(np.nanstd(scut_z)),
            'scut_z_min': float(np.nanmin(scut_z)),
            'scut_z_max': float(np.nanmax(scut_z)),
            'height_speed_pearson_r': pr,
            'height_speed_pearson_p': pp,
            'height_speed_spearman_r': sr,
            'height_speed_spearman_p': sp,
        })

    return valid, rejected


# ── Main pipeline ─────────────────────────────────────────────────────────────

def detect_walking_bouts(
    data_dir: Path,
    cfg: BoutConfig = BoutConfig(),
    fly_id: str = "",
) -> Tuple[pd.DataFrame, List[dict], dict]:
    """Full pipeline: load data → filter → detect → validate → summary CSV.

    Returns (bouts_df, rejected, diagnostics).
    """
    csv_path = data_dir / "data3D.csv"
    print(f"Loading {csv_path} ...")
    df, leg_tip_data, scutellum_data, all_conf = load_data3d(csv_path, cfg.scale)
    n_frames = len(df)
    print(f"  {n_frames:,} frames, {n_frames / cfg.fps:.1f}s at {cfg.fps} Hz")

    # 1. Frame-level filters
    print("Applying frame-level filters ...")
    valid_mask, filter_masks = compute_filter_masks(
        n_frames, leg_tip_data, scutellum_data, all_conf, cfg)
    pct = 100 * valid_mask.sum() / n_frames
    print(f"  {valid_mask.sum():,} / {n_frames:,} frames pass all filters ({pct:.1f}%)")
    print(f"    confidence: {100 * filter_masks['confidence'].sum() / n_frames:.1f}%")
    print(f"    upright:    {100 * filter_masks['upright'].sum() / n_frames:.1f}%")
    print(f"    floor viol: {100 * filter_masks['floor_violation'].sum() / n_frames:.1f}%")
    print(f"    y-wall:     {100 * filter_masks['y_wall_violation'].sum() / n_frames:.1f}%")
    print(f"    x-wall:     {100 * filter_masks['x_wall_violation'].sum() / n_frames:.1f}%")

    # 2. Find candidate bouts
    print("Finding candidate bouts ...")
    candidates = find_bouts(valid_mask, cfg.min_bout_frames, cfg.max_gap_bridge)
    print(f"  {len(candidates)} candidates after gap bridging")

    # 3. Split at immobility
    candidates = split_at_immobility(candidates, scutellum_data, cfg)
    print(f"  {len(candidates)} after immobility splitting")

    # 4. Validate walking
    print("Validating walking criteria ...")
    valid_bouts, rejected = validate_bouts(candidates, leg_tip_data, scutellum_data, cfg)
    print(f"  {len(valid_bouts)} valid bouts, {len(rejected)} rejected")

    # 5. Build summary DataFrame
    rows = []
    for i, b in enumerate(valid_bouts):
        row = {'fly_id': fly_id, 'bout_idx': i + 1}
        row.update(b)
        rows.append(row)
    bouts_df = pd.DataFrame(rows)

    diagnostics = {
        'n_frames': n_frames,
        'n_valid_frames': int(valid_mask.sum()),
        'n_candidates': len(candidates),
        'n_valid_bouts': len(valid_bouts),
        'n_rejected': len(rejected),
    }

    return bouts_df, rejected, diagnostics


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_overview(data_dir: Path, bouts_df: pd.DataFrame, cfg: BoutConfig):
    """Plot timeline overview + sample bout traces."""
    import matplotlib.pyplot as plt

    csv_path = data_dir / "data3D.csv"
    _, leg_tip_data, scutellum_data, _ = load_data3d(csv_path, cfg.scale)
    n_frames = max(len(v['x']) for v in leg_tip_data.values())

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=False)

    # Filter to finite data for background trajectory
    sx, sy = scutellum_data['x'], scutellum_data['y']
    finite = np.isfinite(sx) & np.isfinite(sy)
    # Further filter to arena bounds (reject outliers from bad predictions)
    in_arena = finite & (sx >= -1) & (sx <= cfg.arena_x_mm + 1) & (sy >= -1) & (sy <= cfg.arena_y_mm + 1)

    # Panel 1: Scutellum XY trajectory with bouts highlighted
    ax = axes[0]
    # Subsample background for speed
    bg_step = max(1, int(in_arena.sum()) // 20000)
    bg_idx = np.where(in_arena)[0][::bg_step]
    ax.plot(sx[bg_idx], sy[bg_idx], '.', ms=0.1, color='#ccc', alpha=0.3)
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(bouts_df))))
    for i, row in bouts_df.iterrows():
        s, e = int(row['start_frame']), int(row['end_frame'])
        c = colors[i % len(colors)]
        ax.plot(sx[s:e], sy[s:e], '-', lw=0.8, color=c)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(f'Scutellum trajectory — {len(bouts_df)} walking bouts highlighted')
    ax.set_xlim(-0.5, cfg.arena_x_mm + 0.5)
    ax.set_ylim(-0.5, cfg.arena_y_mm + 0.5)
    ax.set_aspect('equal')

    # Panel 2: Speed over time with bout regions
    ax = axes[1]
    # Compute speed only where data is finite; fill NaN elsewhere
    dx = np.diff(sx)
    dy = np.diff(sy)
    spd_raw = np.sqrt(dx**2 + dy**2) * cfg.fps
    spd_raw = np.concatenate([[0], spd_raw])
    spd_raw[~finite] = np.nan
    # Clip extreme outliers for display
    spd_clipped = np.clip(spd_raw, 0, 100)
    # Smooth for display (nanmean via uniform_filter on non-nan)
    spd_display = np.where(np.isnan(spd_clipped), 0, spd_clipped)
    spd_smooth = uniform_filter1d(spd_display, size=min(801, n_frames))
    step = max(1, n_frames // 5000)
    frames = np.arange(0, n_frames, step)
    ax.plot(frames, spd_smooth[::step], '-', lw=0.5, color='#333')
    for _, row in bouts_df.iterrows():
        ax.axvspan(row['start_frame'], row['end_frame'], alpha=0.3, color='#2ca02c')
    ax.set_ylabel('Speed (mm/s)')
    ax.set_title('Body speed (smoothed) — green = walking bouts')
    ax.set_xlim(0, n_frames)

    # Panel 3: Bout duration histogram
    ax = axes[2]
    if len(bouts_df) > 0:
        ax.hist(bouts_df['duration_s'], bins=20, color='steelblue', edgecolor='white')
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Count')
    ax.set_title('Bout duration distribution')

    # Panel 4: Bout speed histogram
    ax = axes[3]
    if len(bouts_df) > 0:
        ax.hist(bouts_df['mean_speed_mm_s'], bins=20, color='coral', edgecolor='white')
    ax.set_xlabel('Mean speed (mm/s)')
    ax.set_ylabel('Count')
    ax.set_title('Bout mean speed distribution')

    plt.tight_layout()
    return fig


def plot_bout_detail(data_dir: Path, bouts_df: pd.DataFrame, bout_idx: int, cfg: BoutConfig):
    """Plot leg-tip traces for a single bout (Y, Z, confidence)."""
    import matplotlib.pyplot as plt

    csv_path = data_dir / "data3D.csv"
    _, leg_tip_data, scutellum_data, _ = load_data3d(csv_path, cfg.scale)

    row = bouts_df.iloc[bout_idx]
    s, e = int(row['start_frame']), int(row['end_frame'])
    ctx = 250  # context frames
    ps = max(0, s - ctx)
    pe = min(len(scutellum_data['x']) - 1, e + ctx)
    frames = np.arange(ps, pe + 1)

    leg_colors = {
        "T1L_TaTip": "#74c476", "T1R_TaTip": "#1b7837",
        "T2L_TaTip": "#762a83", "T2R_TaTip": "#e08214",
        "T3L_TaTip": "#2166ac", "T3R_TaTip": "#d73027",
    }

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    for tip in cfg.leg_tips:
        c = leg_colors.get(tip, '#888')
        axes[0].plot(frames, leg_tip_data[tip]['y'][ps:pe + 1], '-', lw=0.8, color=c, label=tip)
        axes[1].plot(frames, leg_tip_data[tip]['z'][ps:pe + 1], '-', lw=0.8, color=c)

    # Mean confidence
    conf_arrays = [leg_tip_data[tip]['conf'][ps:pe + 1] for tip in cfg.leg_tips]
    mean_conf = np.nanmean(conf_arrays, axis=0)
    axes[2].plot(frames, mean_conf, '-', lw=0.8, color='#555')
    axes[2].fill_between(frames, 0, mean_conf, alpha=0.1, color='#555')
    axes[2].axhline(cfg.confidence_threshold, color='red', ls=':', lw=1, alpha=0.7)

    for ax in axes:
        ax.axvspan(s, e, alpha=0.1, color='green')
        ax.axvline(s, color='green', ls='--', lw=1.5)
        ax.axvline(e, color='red', ls='--', lw=1.5)

    axes[0].set_ylabel('Y (mm)')
    axes[0].set_title(f'Bout {int(row["bout_idx"])} — frames {s}–{e} ({row["duration_s"]:.2f}s, {row["mean_speed_mm_s"]:.1f} mm/s)')
    axes[0].legend(loc='upper right', fontsize=7, ncol=3)
    axes[1].set_ylabel('Z (mm)')
    axes[2].set_ylabel('Confidence')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walking bout segmentation for fly 3D predictions")
    parser.add_argument("data_dir", type=Path, help="Folder containing data3D.csv")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output folder (default: data_dir)")
    parser.add_argument("--fly-id", default="", help="Fly/session identifier for CSV")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Generate overview + sample bout plots")
    parser.add_argument("--fps", type=int, default=800)
    parser.add_argument("--min-bout-frames", type=int, default=100)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--min-distance", type=float, default=5.0)
    parser.add_argument("--min-cycles", type=int, default=2)
    args = parser.parse_args()

    cfg = BoutConfig(
        fps=args.fps,
        min_bout_frames=args.min_bout_frames,
        confidence_threshold=args.confidence_threshold,
        min_distance_mm=args.min_distance,
        min_walking_cycles=args.min_cycles,
    )

    output_dir = args.output or args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bouts_df, rejected, diagnostics = detect_walking_bouts(args.data_dir, cfg, args.fly_id)

    # Save CSV
    csv_out = output_dir / "walking_bouts_summary.csv"
    bouts_df.to_csv(csv_out, index=False)
    print(f"\nSaved {len(bouts_df)} bouts → {csv_out}")

    # Print summary
    if len(bouts_df) > 0:
        print(f"\nSummary:")
        print(f"  Total bouts:    {len(bouts_df)}")
        print(f"  Duration range: {bouts_df['duration_s'].min():.2f} – {bouts_df['duration_s'].max():.2f} s")
        print(f"  Speed range:    {bouts_df['mean_speed_mm_s'].min():.1f} – {bouts_df['mean_speed_mm_s'].max():.1f} mm/s")
        print(f"  Median frames:  {bouts_df['n_frames'].median():.0f}")

    if rejected:
        print(f"\nRejected ({len(rejected)}):")
        reasons = {}
        for r in rejected:
            key = r['reason'].split('(')[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    if args.visualize:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\nGenerating plots ...")
        fig = plot_overview(args.data_dir, bouts_df, cfg)
        overview_path = output_dir / "bout_overview.png"
        fig.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {overview_path}")

        # Plot first 5 bouts as samples
        n_sample = min(5, len(bouts_df))
        for i in range(n_sample):
            fig = plot_bout_detail(args.data_dir, bouts_df, i, cfg)
            bout_path = output_dir / f"bout_{i + 1}_detail.png"
            fig.savefig(bout_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {bout_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
