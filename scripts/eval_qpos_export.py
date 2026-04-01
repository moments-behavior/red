#!/usr/bin/env python3
"""Evaluate quality of a RED qpos_export.csv + keypoints3d.csv pair.

Checks:
  1. IK residuals from the export
  2. Forward kinematics validation (qpos → site_xpos vs GT keypoints)
  3. Per-site error breakdown
  4. Camera visibility / arena filtering
  5. Data quality for neural network training

Usage:
    python3 scripts/eval_qpos_export.py [--project-dir /path/to/project]
"""

import os
import sys
import math
import argparse
import numpy as np

PROJECT_DIR = "/Users/johnsonr/datasets/rat/tiny_project"
MODEL_XML = "/Users/johnsonr/rat_modeling/IK_resources/rodent_no_collision.xml"

RAT24_SITES = [
    "nose_0_kpsite", "ear_L_1_kpsite", "ear_R_2_kpsite", "neck_3_kpsite",
    "spineL_4_kpsite", "tailbase_5_kpsite", "shoulder_L_6_kpsite",
    "elbow_L_7_kpsite", "wrist_L_8_kpsite", "hand_L_9_kpsite",
    "shoulder_R_10_kpsite", "elbow_R_11_kpsite", "wrist_R_12_kpsite",
    "hand_R_13_kpsite", "knee_L_14_kpsite", "ankle_L_15_kpsite",
    "foot_L_16_kpsite", "knee_R_17_kpsite", "ankle_R_18_kpsite",
    "foot_R_19_kpsite", "tailtip_20_kpsite", "tailmid_21_kpsite",
    "tail1Q_22_kpsite", "tail3Q_23_kpsite",
]

RAT24_NAMES = [
    "Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase",
    "ShoulderL", "ElbowL", "WristL", "HandL",
    "ShoulderR", "ElbowR", "WristR", "HandR",
    "KneeL", "AnkleL", "FootL", "KneeR", "AnkleR", "FootR",
    "TailTip", "TailMid", "Tail1Q", "Tail3Q",
]


def load_qpos_export(csv_path):
    """Load RED qpos export CSV, return metadata + frame data."""
    metadata = {}
    frames = []
    nq = None
    stac_offsets = {}

    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# ') and ':' in line:
                key, _, val = line[2:].partition(':')
                metadata[key.strip()] = val.strip()
                if key.strip() == 'nq':
                    nq = int(val.strip())
                # Parse STAC site offsets
                if key.strip().startswith(' ') or ',' in key:
                    # This is a STAC offset line like "#   site_name, body=N, dx, dy, dz"
                    pass
                continue
            if line.startswith('#'):
                # Parse STAC offset lines
                stripped = line[1:].strip()
                if '_kpsite' in stripped and ',' in stripped:
                    parts = stripped.split(',')
                    if len(parts) >= 5:
                        site_name = parts[0].strip()
                        dx = float(parts[2].strip())
                        dy = float(parts[3].strip())
                        dz = float(parts[4].strip())
                        stac_offsets[site_name] = (dx, dy, dz)
                continue
            if line.startswith('frame,') or line.startswith('frame'):
                if 'qpos_0' in line:
                    cols = line.split(',')
                    qpos_cols = [c for c in cols if c.startswith('qpos_')]
                    nq = len(qpos_cols)
                continue
            if not line:
                continue

            parts = line.split(',')
            frame_id = int(parts[0])
            if nq is None:
                nq = len(parts) - 4  # frame, qpos..., residual, iterations, converged
            qpos = np.array([float(x) for x in parts[1:1+nq]], dtype=np.float64)
            residual_mm = float(parts[1+nq])
            iterations = int(parts[2+nq])
            converged = bool(int(parts[3+nq]))

            frames.append({
                'frame': frame_id,
                'qpos': qpos,
                'residual_mm': residual_mm,
                'iterations': iterations,
                'converged': converged,
            })

    return metadata, frames, nq, stac_offsets


def load_keypoints3d(csv_path):
    """Load RED v2 keypoints3d.csv."""
    frames = {}
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('frame,'):
                continue
            parts = line.split(',')
            frame_id = int(parts[0])
            kp3d = np.full((24, 3), np.nan)
            valid = np.zeros(24, dtype=bool)
            n_cams = np.zeros(24, dtype=int)  # number of cameras (from confidence)

            for kp in range(24):
                base = 1 + kp * 4
                if base + 2 < len(parts):
                    try:
                        x = float(parts[base])
                        y = float(parts[base + 1])
                        z = float(parts[base + 2])
                        # Confidence field (column c) — may contain camera count info
                        conf_str = parts[base + 3].strip() if base + 3 < len(parts) else ''
                        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                            kp3d[kp] = [x, y, z]
                            valid[kp] = True
                    except (ValueError, IndexError):
                        pass

            frames[frame_id] = {'kp3d': kp3d, 'valid': valid}
    return frames


def load_model_with_stac(model_xml, stac_offsets):
    """Load MuJoCo model, add free joint, apply STAC offsets."""
    import mujoco

    spec = mujoco.MjSpec.from_file(model_xml)

    # Add free joint to torso (matching RED)
    torso = spec.body("torso")
    has_free = False
    if torso:
        j = torso.first_joint()
        while j is not None:
            if j.type == mujoco.mjtJoint.mjJNT_FREE:
                has_free = True
                break
            j = j.next(torso)
        if not has_free:
            torso.add_freejoint()

    model = spec.compile()
    data = mujoco.MjData(model)

    # Apply STAC site offsets (site_pos is [nsite, 3])
    n_applied = 0
    for i in range(model.nsite):
        name = model.site(i).name
        if name in stac_offsets:
            dx, dy, dz = stac_offsets[name]
            model.site_pos[i, 0] += dx
            model.site_pos[i, 1] += dy
            model.site_pos[i, 2] += dz
            n_applied += 1

    # Map site names to indices
    site_ids = []
    for name in RAT24_SITES:
        found = -1
        for i in range(model.nsite):
            if model.site(i).name == name:
                found = i
                break
        site_ids.append(found)

    return model, data, site_ids, n_applied


def run_fk(model, data, qpos, site_ids):
    """Run forward kinematics, return site positions [24, 3]."""
    import mujoco
    data.qpos[:] = qpos
    mujoco.mj_fwdPosition(model, data)
    sites = np.zeros((24, 3))
    for i, sid in enumerate(site_ids):
        if sid >= 0:
            sites[i] = data.site_xpos[sid].copy()
    return sites


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', default=PROJECT_DIR)
    parser.add_argument('--model-xml', default=MODEL_XML)
    args = parser.parse_args()

    proj = args.project_dir
    qpos_csv = os.path.join(proj, 'qpos_export.csv')
    # Find keypoints3d.csv
    kp3d_csv = None
    for session in sorted(os.listdir(os.path.join(proj, 'labeled_data'))):
        candidate = os.path.join(proj, 'labeled_data', session, 'keypoints3d.csv')
        if os.path.exists(candidate):
            kp3d_csv = candidate
    if kp3d_csv is None:
        print("ERROR: No keypoints3d.csv found")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────
    print(f"Project: {proj}")
    print(f"qpos:    {qpos_csv}")
    print(f"kp3d:    {kp3d_csv}")

    meta, qpos_frames, nq, stac_offsets = load_qpos_export(qpos_csv)
    kp3d_data = load_keypoints3d(kp3d_csv)

    print(f"\n{'='*65}")
    print("EXPORT METADATA")
    print(f"{'='*65}")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"  STAC offsets: {len(stac_offsets)} sites")
    print(f"  nq: {nq}")
    print(f"  Frames exported: {len(qpos_frames)}")
    print(f"  Frames with 3D keypoints: {len(kp3d_data)}")

    scale_factor_meta = float(meta.get('scale_factor', '0'))

    # Auto-detect actual scale: compare qpos root translation to keypoint centroids.
    # If keypoints are in mm (~1400) and qpos root is in meters (~1.4),
    # the IK used scale=0.001 regardless of what metadata says.
    first_qpos = qpos_frames[0]['qpos']
    first_fid = qpos_frames[0]['frame']
    root_pos = first_qpos[:3]  # meters (model units)
    if first_fid in kp3d_data:
        kp = kp3d_data[first_fid]
        valid_kp = kp['kp3d'][kp['valid']]
        kp_centroid = valid_kp.mean(axis=0)
        root_mag = np.linalg.norm(root_pos)
        kp_mag = np.linalg.norm(kp_centroid)
        if kp_mag > 0 and root_mag > 0:
            ratio = kp_mag / root_mag
            if ratio > 500:  # keypoints are ~1000x root → scale is 0.001
                scale_factor = 0.001
            elif ratio > 5:
                scale_factor = 0.01
            else:
                scale_factor = 1.0
            print(f"\n  Scale auto-detect: kp_centroid_mag={kp_mag:.1f}, "
                  f"root_mag={root_mag:.3f}, ratio={ratio:.0f}")
            print(f"  → Using scale_factor={scale_factor} "
                  f"(metadata said {scale_factor_meta})")
        else:
            scale_factor = scale_factor_meta if scale_factor_meta > 0 else 0.001
    else:
        scale_factor = scale_factor_meta if scale_factor_meta > 0 else 0.001

    # ── Reported residuals ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("1. REPORTED IK RESIDUALS (from export)")
    print(f"{'='*65}")
    residuals = np.array([f['residual_mm'] for f in qpos_frames])
    converged = np.array([f['converged'] for f in qpos_frames])
    iterations = np.array([f['iterations'] for f in qpos_frames])

    print(f"  Frames: {len(residuals)}")
    print(f"  Converged: {converged.sum()}/{len(converged)} "
          f"({100*converged.mean():.1f}%)")
    print(f"  Residual (mm):")
    print(f"    Mean:   {residuals.mean():.4f}")
    print(f"    Median: {np.median(residuals):.4f}")
    print(f"    Min:    {residuals.min():.4f}")
    print(f"    Max:    {residuals.max():.4f}")
    print(f"    Std:    {residuals.std():.4f}")
    print(f"    < 1mm:  {(residuals < 1).sum()}/{len(residuals)} ({100*(residuals < 1).mean():.1f}%)")
    print(f"    < 2mm:  {(residuals < 2).sum()}/{len(residuals)} ({100*(residuals < 2).mean():.1f}%)")
    print(f"    < 5mm:  {(residuals < 5).sum()}/{len(residuals)} ({100*(residuals < 5).mean():.1f}%)")
    print(f"  Iterations:")
    print(f"    Mean: {iterations.mean():.0f}, Max: {iterations.max()}")

    # Distribution of residuals
    bins = [0, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, float('inf')]
    print(f"\n  Residual distribution:")
    for i in range(len(bins) - 1):
        count = ((residuals >= bins[i]) & (residuals < bins[i+1])).sum()
        pct = 100 * count / len(residuals)
        bar = '#' * int(pct / 2)
        label = f"{bins[i]:.0f}-{bins[i+1]:.0f}" if bins[i+1] < float('inf') else f">{bins[i]:.0f}"
        print(f"    {label:>8s} mm: {count:5d} ({pct:5.1f}%) {bar}")

    # ── FK validation ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("2. FORWARD KINEMATICS VALIDATION")
    print(f"{'='*65}")

    model, data, site_ids, n_stac = load_model_with_stac(args.model_xml, stac_offsets)
    print(f"  Model: nq={model.nq}, nv={model.nv}, {model.nsite} sites")
    print(f"  STAC offsets applied: {n_stac}")
    print(f"  Scale factor: {scale_factor}")

    # Match frames between qpos and kp3d
    common_frames = []
    for qf in qpos_frames:
        fid = qf['frame']
        if fid in kp3d_data:
            common_frames.append(qf)

    print(f"  Common frames (qpos + kp3d): {len(common_frames)}")

    # Run FK on all common frames, compare to GT keypoints
    per_site_errors = np.full((len(common_frames), 24), np.nan)
    n_valid_per_frame = np.zeros(len(common_frames), dtype=int)
    frame_mean_errors = np.zeros(len(common_frames))
    frame_ids = np.zeros(len(common_frames), dtype=int)

    for i, qf in enumerate(common_frames):
        fid = qf['frame']
        frame_ids[i] = fid
        kp = kp3d_data[fid]
        kp3d = kp['kp3d']
        valid = kp['valid']
        n_valid_per_frame[i] = valid.sum()

        # Run FK
        sites = run_fk(model, data, qf['qpos'], site_ids)

        # Compare to GT keypoints (apply scale factor)
        for j in range(24):
            if valid[j] and site_ids[j] >= 0:
                gt = kp3d[j] * scale_factor  # convert to model units if needed
                err_m = np.linalg.norm(sites[j] - gt)
                per_site_errors[i, j] = err_m * 1000.0  # meters → mm

        valid_errors = per_site_errors[i, ~np.isnan(per_site_errors[i])]
        frame_mean_errors[i] = np.mean(valid_errors) if len(valid_errors) > 0 else np.nan

    valid_frame_errors = frame_mean_errors[~np.isnan(frame_mean_errors)]
    print(f"\n  FK site error (mm) — comparing FK(qpos) to GT keypoints:")
    print(f"    Mean:   {valid_frame_errors.mean():.4f}")
    print(f"    Median: {np.median(valid_frame_errors):.4f}")
    print(f"    Min:    {valid_frame_errors.min():.4f}")
    print(f"    Max:    {valid_frame_errors.max():.4f}")
    print(f"    Std:    {valid_frame_errors.std():.4f}")

    # Per-site breakdown
    print(f"\n  Per-site FK error (mm):")
    print(f"    {'Site':<12s} {'Mean':>8s} {'Median':>8s} {'Max':>8s} {'N':>6s}")
    print(f"    {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    site_stats = []
    for j in range(24):
        col = per_site_errors[:, j]
        vals = col[~np.isnan(col)]
        if len(vals) > 0:
            site_stats.append((j, RAT24_NAMES[j], np.mean(vals),
                               np.median(vals), np.max(vals), len(vals)))
    site_stats.sort(key=lambda x: x[2])
    for j, name, mean, med, mx, n in site_stats:
        flag = " ⚠" if mean > 5.0 else ""
        print(f"    {name:<12s} {mean:>8.3f} {med:>8.3f} {mx:>8.3f} {n:>6d}{flag}")

    # ── Keypoint validity / camera coverage ─────────────────────────────
    print(f"\n{'='*65}")
    print("3. KEYPOINT VALIDITY / CAMERA COVERAGE")
    print(f"{'='*65}")

    print(f"\n  Keypoints per frame:")
    print(f"    Min: {n_valid_per_frame.min()}, Max: {n_valid_per_frame.max()}, "
          f"Mean: {n_valid_per_frame.mean():.1f}")
    for thresh in [24, 22, 20, 18, 15, 10]:
        count = (n_valid_per_frame >= thresh).sum()
        pct = 100 * count / len(n_valid_per_frame)
        print(f"    >= {thresh} keypoints: {count}/{len(n_valid_per_frame)} ({pct:.1f}%)")

    # Spatial analysis: detect ramp vs arena frames
    # Arena frames should have the rat at a more central location with more cameras
    print(f"\n  Spatial distribution of rat centroid (model units):")
    centroids = np.zeros((len(common_frames), 3))
    for i, qf in enumerate(common_frames):
        fid = qf['frame']
        kp = kp3d_data[fid]
        valid_pts = kp['kp3d'][kp['valid']]
        centroids[i] = valid_pts.mean(axis=0) * scale_factor

    for axis, name in enumerate(['x', 'y', 'z']):
        print(f"    {name}: [{centroids[:, axis].min():.4f}, "
              f"{centroids[:, axis].max():.4f}] "
              f"(range {centroids[:, axis].ptp():.4f})")

    # Identify potential ramp frames by z-height or position outliers
    z_vals = centroids[:, 2]
    z_median = np.median(z_vals)
    z_std = np.std(z_vals)
    high_z = z_vals > z_median + 2 * z_std  # rat on elevated ramp?
    low_kps = n_valid_per_frame < 20

    print(f"\n  Potential ramp/edge frames:")
    print(f"    High z (>{z_median + 2*z_std:.4f}): {high_z.sum()}")
    print(f"    Low keypoints (<20): {low_kps.sum()}")
    print(f"    Either: {(high_z | low_kps).sum()}")

    # ── Quality filtering recommendations ──────────────────────────────
    print(f"\n{'='*65}")
    print("4. TRAINING DATA QUALITY ASSESSMENT")
    print(f"{'='*65}")

    # Filter: converged + residual < 5mm + >= 20 keypoints
    good_mask = np.zeros(len(common_frames), dtype=bool)
    for i, qf in enumerate(common_frames):
        good_mask[i] = (qf['converged'] and
                        qf['residual_mm'] < 5.0 and
                        n_valid_per_frame[i] >= 20)

    n_good = good_mask.sum()
    print(f"\n  Quality filter (converged + res<5mm + >=20 kps):")
    print(f"    Pass: {n_good}/{len(common_frames)} ({100*n_good/len(common_frames):.1f}%)")

    if n_good > 0:
        good_res = np.array([common_frames[i]['residual_mm']
                             for i in range(len(common_frames)) if good_mask[i]])
        good_fk = frame_mean_errors[good_mask]
        good_fk = good_fk[~np.isnan(good_fk)]
        print(f"    Residual: mean={good_res.mean():.3f}mm, "
              f"median={np.median(good_res):.3f}mm")
        if len(good_fk) > 0:
            print(f"    FK error: mean={good_fk.mean():.3f}mm, "
                  f"median={np.median(good_fk):.3f}mm")

    # Stricter filter for high-quality training
    strict_mask = np.zeros(len(common_frames), dtype=bool)
    for i, qf in enumerate(common_frames):
        strict_mask[i] = (qf['converged'] and
                          qf['residual_mm'] < 2.0 and
                          n_valid_per_frame[i] >= 22)

    n_strict = strict_mask.sum()
    print(f"\n  Strict filter (converged + res<2mm + >=22 kps):")
    print(f"    Pass: {n_strict}/{len(common_frames)} ({100*n_strict/len(common_frames):.1f}%)")

    if n_strict > 0:
        strict_res = np.array([common_frames[i]['residual_mm']
                               for i in range(len(common_frames)) if strict_mask[i]])
        strict_fk = frame_mean_errors[strict_mask]
        strict_fk = strict_fk[~np.isnan(strict_fk)]
        print(f"    Residual: mean={strict_res.mean():.3f}mm, "
              f"median={np.median(strict_res):.3f}mm")
        if len(strict_fk) > 0:
            print(f"    FK error: mean={strict_fk.mean():.3f}mm, "
                  f"median={np.median(strict_fk):.3f}mm")

    # ── Worst frames ───────────────────────────────────────────────────
    print(f"\n  10 worst frames by FK error:")
    worst_idx = np.argsort(frame_mean_errors)[::-1][:10]
    for idx in worst_idx:
        if np.isnan(frame_mean_errors[idx]):
            continue
        qf = common_frames[idx]
        print(f"    frame={qf['frame']:6d}  fk_err={frame_mean_errors[idx]:.2f}mm  "
              f"reported_res={qf['residual_mm']:.2f}mm  "
              f"kps={n_valid_per_frame[idx]}  conv={'Y' if qf['converged'] else 'N'}")

    # ── qpos statistics ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("5. QPOS STATISTICS")
    print(f"{'='*65}")
    all_qpos = np.array([f['qpos'] for f in common_frames])
    print(f"  Shape: {all_qpos.shape}")
    print(f"\n  Free joint translation (qpos[0:3]):")
    for j, name in enumerate(['x', 'y', 'z']):
        print(f"    {name}: [{all_qpos[:, j].min():.4f}, {all_qpos[:, j].max():.4f}] "
              f"(range {all_qpos[:, j].ptp():.4f})")
    print(f"\n  Free joint quaternion (qpos[3:7]):")
    for j, name in enumerate(['w', 'x', 'y', 'z']):
        print(f"    {name}: [{all_qpos[:, 3+j].min():.4f}, {all_qpos[:, 3+j].max():.4f}]")

    # Check quaternion norm
    quat_norms = np.linalg.norm(all_qpos[:, 3:7], axis=1)
    print(f"    norm: [{quat_norms.min():.6f}, {quat_norms.max():.6f}] "
          f"(should be ~1.0)")

    print(f"\n  Hinge joint angles (qpos[7:68], radians):")
    hinge = all_qpos[:, 7:]
    print(f"    Overall range: [{hinge.min():.3f}, {hinge.max():.3f}] rad")
    print(f"    Mean abs: {np.abs(hinge).mean():.3f} rad ({np.abs(hinge).mean()*180/math.pi:.1f}°)")

    # Per-joint range
    print(f"\n  Per-joint angle range (top 10 most variable):")
    joint_ranges = hinge.ptp(axis=0)
    top_joints = np.argsort(joint_ranges)[::-1][:10]
    for j in top_joints:
        print(f"    qpos[{j+7}]: range={joint_ranges[j]:.3f} rad "
              f"({joint_ranges[j]*180/math.pi:.1f}°), "
              f"mean={hinge[:, j].mean():.3f}")

    # Joint saturation analysis — joints pinned at limits
    print(f"\n  Joint limit saturation (joints at their min/max limit):")
    import mujoco as _mj
    _spec = _mj.MjSpec.from_file(args.model_xml)
    _torso = _spec.body("torso")
    if _torso:
        _j = _torso.first_joint()
        _has_free = False
        while _j is not None:
            if _j.type == _mj.mjtJoint.mjJNT_FREE:
                _has_free = True
                break
            _j = _j.next(_torso)
        if not _has_free:
            _torso.add_freejoint()
    _model = _spec.compile()

    saturated_joints = []
    for j in range(_model.njnt):
        if not _model.jnt_limited[j]:
            continue
        jt = _model.jnt_type[j]
        if jt not in (2, 3):  # SLIDE or HINGE
            continue
        qa = int(_model.jnt_qposadr[j])
        lo = float(_model.jnt_range[j, 0])
        hi = float(_model.jnt_range[j, 1])
        if qa >= 7 and qa < nq:  # hinge joints only
            vals = all_qpos[:, qa]
            at_lo = (np.abs(vals - lo) < 1e-4).sum()
            at_hi = (np.abs(vals - hi) < 1e-4).sum()
            pct_sat = 100 * (at_lo + at_hi) / len(vals)
            if pct_sat > 5:
                jname = _model.joint(j).name or f"joint_{j}"
                saturated_joints.append((jname, qa, lo, hi, at_lo, at_hi, pct_sat))

    if saturated_joints:
        saturated_joints.sort(key=lambda x: -x[6])
        print(f"    {'Joint':<25s} {'qpos':>5s} {'Lo':>7s} {'Hi':>7s} "
              f"{'@Lo':>5s} {'@Hi':>5s} {'Sat%':>6s}")
        print(f"    {'─'*25} {'─'*5} {'─'*7} {'─'*7} {'─'*5} {'─'*5} {'─'*6}")
        for jname, qa, lo, hi, at_lo, at_hi, pct in saturated_joints[:20]:
            print(f"    {jname:<25s} [{qa:3d}] {lo:>7.3f} {hi:>7.3f} "
                  f"{at_lo:>5d} {at_hi:>5d} {pct:>5.1f}%")
    else:
        print(f"    No joints saturated >5% of the time")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  Total frames: {len(qpos_frames)}")
    print(f"  With 3D keypoints: {len(common_frames)}")
    print(f"  Converged: {converged.sum()} ({100*converged.mean():.1f}%)")
    print(f"  Mean reported residual: {residuals.mean():.3f} mm")
    print(f"  Mean FK error: {valid_frame_errors.mean():.3f} mm")
    print(f"  Good for training (res<5mm, >=20kps): {n_good}")
    print(f"  High quality (res<2mm, >=22kps): {n_strict}")

    if valid_frame_errors.mean() < 3.0 and n_good > 100:
        print(f"\n  ✓ Data quality looks GOOD for training")
        print(f"    Recommended: use {n_good} frames with quality filter")
    elif valid_frame_errors.mean() < 5.0:
        print(f"\n  ~ Data quality is MARGINAL")
        print(f"    Consider filtering to strict set ({n_strict} frames)")
    else:
        print(f"\n  ✗ Data quality needs improvement")
        print(f"    Investigate high-residual frames")


if __name__ == "__main__":
    main()
