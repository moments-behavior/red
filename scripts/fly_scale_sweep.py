#!/usr/bin/env python3
"""Parameter sweep to find optimal fly body model scale factor.

Uses segment-length matching between the MuJoCo fruitfly model and
labeled 3D keypoints from flypred5. The model is in cm, keypoints in mm,
so we convert keypoints to cm (multiply by 0.1).

For each candidate scale factor, we:
1. Scale all body positions in the model by that factor
2. Run forward kinematics (mj_forward)
3. Compute model segment lengths (distances between connected site pairs)
4. Compare to the median segment lengths from labeled data
5. The scale minimizing mean absolute segment length error wins.
"""

import os
import glob
import csv
import numpy as np
import mujoco

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "/Users/johnsonr/src/red/models/fruitfly/fruitfly_fly50.mjb"
DATA_DIR = "/Users/johnsonr/red_demos/flypred5/labeled_data"

SCALES = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88,
          0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]

# Fly50 keypoint names (order matches CSV columns x0..x49)
FLY50_NAMES = [
    "Antenna_Base", "EyeL",       "EyeR",       "Scutellum",
    "Abd_A4",       "Abd_tip",    "WingL_base", "WingL_V12",
    "WingL_V13",    "T1L_ThxCx",  "T1L_Tro",    "T1L_FeTi",
    "T1L_TiTa",     "T1L_TaT1",   "T1L_TaT3",   "T1L_TaTip",
    "T2L_Tro",      "T2L_FeTi",   "T2L_TiTa",   "T2L_TaT1",
    "T2L_TaT3",     "T2L_TaTip",  "T3L_Tro",    "T3L_FeTi",
    "T3L_TiTa",     "T3L_TaT1",   "T3L_TaT3",   "T3L_TaTip",
    "WingR_base",   "WingR_V12",  "WingR_V13",  "T1R_ThxCx",
    "T1R_Tro",      "T1R_FeTi",   "T1R_TiTa",   "T1R_TaT1",
    "T1R_TaT3",     "T1R_TaTip",  "T2R_Tro",    "T2R_FeTi",
    "T2R_TiTa",     "T2R_TaT1",   "T2R_TaT3",   "T2R_TaTip",
    "T3R_Tro",      "T3R_FeTi",   "T3R_TiTa",   "T3R_TaT1",
    "T3R_TaT3",     "T3R_TaTip",
]

# Skeleton edges (index pairs into FLY50_NAMES) — segments we measure
EDGES = [
    (0, 1),   (0, 2),   (1, 3),   (2, 3),   (3, 4),   (4, 5),
    (6, 7),   (7, 8),   (8, 6),   (9, 10),  (10, 11), (11, 12),
    (12, 13), (13, 14), (14, 15), (16, 17), (17, 18), (18, 19),
    (19, 20), (20, 21), (22, 23), (23, 24), (24, 25), (25, 26),
    (26, 27), (28, 29), (29, 30), (30, 28), (31, 32), (32, 33),
    (33, 34), (34, 35), (35, 36), (36, 37), (38, 39), (39, 40),
    (40, 41), (41, 42), (42, 43), (44, 45), (45, 46), (46, 47),
    (47, 48), (48, 49),
]


def load_all_keypoints3d():
    """Load all labeled 3D keypoints from all sessions. Returns Nx50x3 array in cm."""
    sessions = sorted(glob.glob(os.path.join(DATA_DIR, "*", "keypoints3d.csv")))
    all_frames = []
    for path in sessions:
        with open(path) as f:
            lines = f.readlines()
        # Skip comment lines (#red_csv, #skeleton, header)
        data_lines = [l for l in lines if not l.startswith('#') and not l.startswith('frame,')]
        for line in data_lines:
            parts = line.strip().split(',')
            # Need at least frame + 50*(x,y,z) = 151 values; c columns may be empty
            if len(parts) < 1 + 50 * 3:
                continue
            coords = np.zeros((50, 3))
            valid = np.ones(50, dtype=bool)
            for k in range(50):
                base = 1 + k * 4  # skip frame col; each kp has x,y,z,c
                try:
                    x = float(parts[base])
                    y = float(parts[base + 1])
                    z = float(parts[base + 2])
                    coords[k] = [x, y, z]
                except (ValueError, IndexError):
                    valid[k] = False
            # Only keep frames where all keypoints are labeled
            if valid.all():
                all_frames.append(coords)
    arr = np.array(all_frames)  # N x 50 x 3, in mm
    arr *= 0.1  # convert to cm (model units)
    return arr


def compute_segment_lengths(positions):
    """Compute segment lengths from Nx50x3 positions. Returns Nx44 array."""
    N = positions.shape[0]
    lengths = np.zeros((N, len(EDGES)))
    for ei, (a, b) in enumerate(EDGES):
        diff = positions[:, a, :] - positions[:, b, :]
        lengths[:, ei] = np.linalg.norm(diff, axis=1)
    return lengths


def get_model_site_indices(model):
    """Map FLY50_NAMES to site indices in the MuJoCo model."""
    site_name_to_id = {}
    for i in range(model.nsite):
        site_name_to_id[model.site(i).name] = i
    indices = []
    for name in FLY50_NAMES:
        if name in site_name_to_id:
            indices.append(site_name_to_id[name])
        else:
            print(f"  WARNING: site '{name}' not found in model")
            indices.append(-1)
    return indices


def get_model_segment_lengths_at_scale(scale):
    """Load model, scale body positions, compute segment lengths."""
    m = mujoco.MjModel.from_binary_path(MODEL_PATH)
    d = mujoco.MjData(m)

    # Scale all body positions (body_pos is nbody x 3)
    # Also scale site_pos (local offsets within bodies)
    # Also scale geom_pos and geom_size for completeness
    m.body_pos[:] *= scale
    m.site_pos[:] *= scale
    m.geom_pos[:] *= scale
    m.geom_size[:] *= scale
    # Scale joint anchor positions
    m.jnt_pos[:] *= scale

    mujoco.mj_forward(m, d)

    site_ids = get_model_site_indices(m)
    positions = np.zeros((50, 3))
    for k, sid in enumerate(site_ids):
        if sid >= 0:
            positions[k] = d.site_xpos[sid]

    # Compute segment lengths (single pose, default qpos)
    lengths = np.zeros(len(EDGES))
    for ei, (a, b) in enumerate(EDGES):
        lengths[ei] = np.linalg.norm(positions[a] - positions[b])

    return lengths, positions


def main():
    print("Loading labeled 3D keypoints from flypred5...")
    data = load_all_keypoints3d()
    print(f"  Loaded {data.shape[0]} fully-labeled frames")

    if data.shape[0] == 0:
        print("ERROR: No fully-labeled frames found")
        # Try with partial frames
        print("Trying with frames that have at least leg keypoints labeled...")
        return

    # Compute data segment lengths
    data_seg = compute_segment_lengths(data)  # N x 44
    median_seg = np.median(data_seg, axis=0)  # 44
    mean_seg = np.mean(data_seg, axis=0)

    print(f"\nData segment length statistics (cm):")
    print(f"  Median total body length (segments summed): {median_seg.sum():.4f} cm")
    print(f"  Per-segment median range: [{median_seg.min():.4f}, {median_seg.max():.4f}] cm")

    # Also compute per-frame standard deviation of segment lengths
    seg_std = np.std(data_seg, axis=0)
    print(f"  Mean per-segment std: {seg_std.mean():.5f} cm")

    print(f"\n{'Scale':>6s}  {'Mean Abs Err':>12s}  {'Mean Rel Err':>12s}  {'Max Rel Err':>11s}")
    print("-" * 55)

    results = []
    for scale in SCALES:
        model_seg, _ = get_model_segment_lengths_at_scale(scale)

        # Mean absolute error per segment
        abs_err = np.abs(model_seg - median_seg)
        mean_abs = np.mean(abs_err)

        # Mean relative error
        rel_err = abs_err / (median_seg + 1e-10)
        mean_rel = np.mean(rel_err)
        max_rel = np.max(rel_err)

        results.append((scale, mean_abs, mean_rel, max_rel))
        print(f"{scale:6.2f}  {mean_abs:12.5f} cm  {mean_rel:11.3%}  {max_rel:10.3%}")

    # Find best scale
    best_idx = np.argmin([r[1] for r in results])
    best_scale, best_abs, best_rel, best_max_rel = results[best_idx]

    print(f"\n{'='*55}")
    print(f"OPTIMAL SCALE: {best_scale:.2f}")
    print(f"  Mean absolute segment error: {best_abs:.5f} cm ({best_abs*10:.4f} mm)")
    print(f"  Mean relative segment error: {best_rel:.3%}")
    print(f"  Max  relative segment error: {best_max_rel:.3%}")

    # Confidence interval via bootstrap on data segment lengths
    print(f"\nBootstrap confidence interval (1000 resamples)...")
    n_boot = 1000
    boot_scales = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.choice(data.shape[0], size=data.shape[0], replace=True)
        boot_median = np.median(data_seg[idx], axis=0)
        # Find best scale for this bootstrap sample
        best_boot_abs = np.inf
        best_boot_scale = SCALES[0]
        for scale in SCALES:
            model_seg, _ = get_model_segment_lengths_at_scale(scale)
            mae = np.mean(np.abs(model_seg - boot_median))
            if mae < best_boot_abs:
                best_boot_abs = mae
                best_boot_scale = scale
        boot_scales.append(best_boot_scale)

    boot_scales = np.array(boot_scales)
    ci_lo, ci_hi = np.percentile(boot_scales, [2.5, 97.5])
    print(f"  Bootstrap optimal scale: {np.mean(boot_scales):.3f} +/- {np.std(boot_scales):.3f}")
    print(f"  95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")

    # Detailed per-segment analysis at best scale
    print(f"\nPer-segment analysis at scale={best_scale:.2f}:")
    model_seg, model_pos = get_model_segment_lengths_at_scale(best_scale)
    print(f"{'Edge':>30s}  {'Model':>8s}  {'Data':>8s}  {'Err':>8s}  {'Rel':>7s}")
    print("-" * 72)
    for ei, (a, b) in enumerate(EDGES):
        name = f"{FLY50_NAMES[a]}-{FLY50_NAMES[b]}"
        err = model_seg[ei] - median_seg[ei]
        rel = abs(err) / (median_seg[ei] + 1e-10)
        print(f"{name:>30s}  {model_seg[ei]:8.4f}  {median_seg[ei]:8.4f}  {err:+8.4f}  {rel:6.1%}")

    # Also try finer-grained search around best
    fine_scales = np.arange(best_scale - 0.05, best_scale + 0.06, 0.01)
    print(f"\nFine-grained search around {best_scale:.2f}:")
    print(f"{'Scale':>6s}  {'Mean Abs Err':>12s}")
    print("-" * 25)
    fine_results = []
    for scale in fine_scales:
        model_seg, _ = get_model_segment_lengths_at_scale(scale)
        mae = np.mean(np.abs(model_seg - median_seg))
        fine_results.append((scale, mae))
        print(f"{scale:6.2f}  {mae:12.5f} cm")

    fine_best_idx = np.argmin([r[1] for r in fine_results])
    fine_best = fine_results[fine_best_idx]
    print(f"\nFINAL OPTIMAL SCALE: {fine_best[0]:.2f} (MAE = {fine_best[1]:.5f} cm = {fine_best[1]*10:.4f} mm)")


if __name__ == "__main__":
    main()
