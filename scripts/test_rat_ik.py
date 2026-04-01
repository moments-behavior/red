#!/usr/bin/env python3
"""Pre-flight test: validate IK solver + model geometry on ~1000 rat frames.

Replicates RED's mujoco_ik.h algorithm exactly:
  - Loads rodent_no_collision.xml + adds free joint to torso (matching RED)
  - Maps 24 Rat24Target keypoint sites
  - Runs gradient-descent IK with momentum, warm-starting, joint clamping
  - Reports per-site residuals, convergence stats, and sanity checks

This is a TEST script — not for production qpos generation.
Production export uses RED's built-in "Solve All & Export" (body_model_window.h).

Usage:
    python3 scripts/test_rat_ik.py [--n-frames 1000] [--max-iter 5000]
"""

import os
import sys
import time
import math
import argparse
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_XML = "/Users/johnsonr/rat_modeling/IK_resources/rodent_no_collision.xml"
KP3D_CSV  = "/Users/johnsonr/datasets/rat/red_mj_dev/labeled_data/2025_11_04_06_17_13_v2/keypoints3d.csv"

# 24 keypoint sites in Rat24Target skeleton order
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


def load_model():
    """Load rodent model + add free joint (matching RED's mujoco_context.h)."""
    import mujoco

    print(f"Loading model: {MODEL_XML}")
    spec = mujoco.MjSpec.from_file(MODEL_XML)

    # Add free joint to torso (matching RED)
    torso = spec.body("torso")
    if torso is None:
        print("ERROR: 'torso' body not found in model")
        sys.exit(1)

    # Check if free joint already exists
    has_free = False
    j = torso.first_joint()
    while j is not None:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            has_free = True
            break
        j = j.next(torso)
    if not has_free:
        torso.add_freejoint()
        print("  Added free joint to 'torso' body")

    model = spec.compile()
    data = mujoco.MjData(model)

    print(f"  Model: {model.nbody} bodies, {model.njnt} joints, "
          f"{model.nsite} sites, nv={model.nv}, nq={model.nq}")

    # Map site names to indices
    site_name_to_id = {}
    for i in range(model.nsite):
        site_name_to_id[model.site(i).name] = i

    site_ids = []
    for name in RAT24_SITES:
        sid = site_name_to_id.get(name, -1)
        site_ids.append(sid)
        if sid < 0:
            print(f"  WARNING: site '{name}' not found")

    n_mapped = sum(1 for s in site_ids if s >= 0)
    print(f"  Site mapping: {n_mapped}/24 keypoint sites found")

    return model, data, site_ids


def precompute_model_info(model):
    """Pre-compute joint metadata (matching test_fly_ik.py pattern)."""
    nv = model.nv
    has_free = False
    free_qa = -1
    free_dof = -1

    for j in range(model.njnt):
        if model.jnt_type[j] == 0:  # mjJNT_FREE
            has_free = True
            free_qa = int(model.jnt_qposadr[j])
            free_dof = int(model.jnt_dofadr[j])
            break

    hinge_dofs = []
    hinge_qa = []
    for j in range(model.njnt):
        if model.jnt_type[j] == 3:  # mjJNT_HINGE
            hinge_dofs.append(int(model.jnt_dofadr[j]))
            hinge_qa.append(int(model.jnt_qposadr[j]))

    is_trans = np.zeros(nv, dtype=bool)
    if has_free:
        is_trans[free_dof:free_dof + 3] = True

    limited = []
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            jt = model.jnt_type[j]
            if jt in (2, 3):  # SLIDE or HINGE
                qa = int(model.jnt_qposadr[j])
                lo = float(model.jnt_range[j, 0])
                hi = float(model.jnt_range[j, 1])
                limited.append((qa, lo, hi))

    return {
        'has_free': has_free, 'free_qa': free_qa, 'free_dof': free_dof,
        'hinge_dofs': np.array(hinge_dofs, dtype=np.intp),
        'hinge_qa': np.array(hinge_qa, dtype=np.intp),
        'is_trans': is_trans, 'limited': limited,
    }


def load_keypoints3d(csv_path, max_frames=None):
    """Load RED v2 keypoints3d.csv."""
    frames = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('frame,'):
                continue
            parts = line.split(',')
            frame_id = int(parts[0])
            kp3d = np.full((24, 3), np.nan)
            valid = np.zeros(24, dtype=bool)

            for kp in range(24):
                base = 1 + kp * 4  # x, y, z, confidence
                if base + 2 < len(parts):
                    try:
                        x, y, z = float(parts[base]), float(parts[base+1]), float(parts[base+2])
                        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                            kp3d[kp] = [x, y, z]
                            valid[kp] = True
                    except (ValueError, IndexError):
                        pass

            if valid.sum() >= 4:  # RED requires >= 4 active sites
                frames.append((frame_id, kp3d, valid))

            if max_frames and len(frames) >= max_frames:
                break

    return frames


def solve_ik(model, data, minfo, site_ids, kp3d_mm, valid,
             lr=0.01, beta=0.99, max_iter=5000, reg_strength=1e-4,
             progress_thresh=0.01, check_every=100, warm_qpos=None):
    """IK solver matching RED's mujoco_ik.h exactly.

    Scale factor: RED auto-detects 0.001 (mm → meters) when keypoint
    centroid magnitude > 10. See mujoco_ik.h lines 88-105.

    Final residual: sqrt(sum_of_squared_errors / N_active_sites) in model
    units, then * 1000 for mm. Matches body_model_window.h line 477.
    """
    import mujoco

    nv = model.nv
    nq = model.nq
    SF = 0.001  # mm → meters (auto-detected by RED for rat data)

    # Build active targets (matching mujoco_ik.h lines 108-113)
    active_kp = []
    active_sid = []
    active_target = []
    for kp_idx in range(24):
        if not valid[kp_idx]:
            continue
        sid = site_ids[kp_idx]
        if sid < 0:
            continue
        active_kp.append(kp_idx)
        active_sid.append(sid)
        active_target.append(kp3d_mm[kp_idx] * SF)

    N = len(active_kp)
    if N == 0:
        return None, 0.0, 0, False, np.full(24, np.nan)

    active_sid_arr = np.array(active_sid, dtype=np.intp)
    target_arr = np.array(active_target)  # [N, 3] in meters

    # Initialize qpos
    cold_start = warm_qpos is None
    if warm_qpos is not None:
        data.qpos[:] = warm_qpos
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Cold-start root alignment (matching mujoco_ik.h lines 127-145)
    if cold_start and minfo['has_free']:
        tc = target_arr.mean(axis=0)
        mc = data.site_xpos[active_sid_arr].mean(axis=0)
        qa = minfo['free_qa']
        data.qpos[qa:qa+3] += tc - mc
        mujoco.mj_forward(model, data)

    # Pre-allocate
    jacp_k = np.zeros((3, nv))
    update = np.zeros(nv)
    hinge_dofs = minfo['hinge_dofs']
    hinge_qa = minfo['hinge_qa']

    converged = False
    iterations_used = 0

    for it in range(max_iter):
        iterations_used = it + 1

        # Stacked Jacobian
        jacp = np.zeros((3 * N, nv))
        for k in range(N):
            jacp_k[:] = 0.0
            mujoco.mj_jacSite(model, data, jacp_k, None, active_sid_arr[k])
            jacp[3*k:3*k+3, :] = jacp_k

        # Residual: site_xpos - target
        site_pos = data.site_xpos[active_sid_arr]  # [N, 3]
        diff = (site_pos - target_arr).ravel()  # [3N]
        err_sq = float(np.dot(diff, diff))

        # Gradient: 2 * J^T * diff
        grad = 2.0 * (jacp.T @ diff)

        # Regularization on hinge joints
        if reg_strength > 0 and len(hinge_dofs) > 0:
            grad[hinge_dofs] += 2.0 * reg_strength * data.qpos[hinge_qa]

        # Momentum
        update = beta * update + grad

        # Step (single LR — matches default when lr_joint == 0)
        step = np.zeros(nv)
        step[:] = -lr * update
        mujoco.mj_integratePos(model, data.qpos, step, 1.0)

        # Clamp joints
        for qa, lo, hi in minfo['limited']:
            data.qpos[qa] = max(lo, min(hi, data.qpos[qa]))

        # FK
        mujoco.mj_fwdPosition(model, data)

        # Convergence check (matching mujoco_ik.h lines 283-300)
        if check_every > 0 and it > 0 and it % check_every == 0:
            update_norm = lr * np.linalg.norm(update)
            obj = err_sq + reg_strength * float(np.sum(data.qpos[hinge_qa]**2))
            if obj > 1e-12 and update_norm / obj < progress_thresh:
                converged = True
                break

    # Final residual: sqrt(err_sq / N) in model units, * 1000 for mm
    # (matching mujoco_ik.h lines 316-327 and body_model_window.h line 477)
    site_pos = data.site_xpos[active_sid_arr]
    err_sq = 0.0
    for k in range(N):
        for c in range(3):
            d = site_pos[k, c] - target_arr[k, c]
            err_sq += d * d
    final_residual_m = math.sqrt(err_sq / N)
    residual_mm = final_residual_m * 1000.0

    # Per-site residual in mm
    per_site_mm = np.full(24, np.nan)
    for i, kp_idx in enumerate(active_kp):
        d = np.linalg.norm(site_pos[i] - target_arr[i])
        per_site_mm[kp_idx] = d * 1000.0  # meters → mm

    return data.qpos.copy(), residual_mm, iterations_used, converged, per_site_mm


def main():
    parser = argparse.ArgumentParser(description="Test IK on rat keypoints")
    parser.add_argument('--n-frames', type=int, default=1000,
                        help="Number of frames to test (default 1000)")
    parser.add_argument('--max-iter', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.99)
    parser.add_argument('--reg', type=float, default=1e-4)
    args = parser.parse_args()

    model, data, site_ids = load_model()
    minfo = precompute_model_info(model)

    print(f"\nLoading keypoints from: {KP3D_CSV}")
    frames = load_keypoints3d(KP3D_CSV, max_frames=args.n_frames)
    print(f"Loaded {len(frames)} frames with >= 4 valid keypoints")

    # ── Sanity check 1: model geometry ──────────────────────────────────
    print("\n" + "="*60)
    print("SANITY CHECK 1: Model geometry (default pose)")
    print("="*60)
    import mujoco
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print(f"  nq={model.nq}, nv={model.nv}")
    print(f"  Free joint: {'yes' if minfo['has_free'] else 'no'} "
          f"(qpos addr={minfo['free_qa']})")
    print(f"  Hinge joints: {len(minfo['hinge_dofs'])}")
    print(f"  Limited joints: {len(minfo['limited'])}")

    print(f"\n  Default pose site positions (meters):")
    for i, (name, sid) in enumerate(zip(RAT24_NAMES, site_ids)):
        if sid >= 0:
            pos = data.site_xpos[sid]
            print(f"    [{i:2d}] {name:<12s} ({RAT24_SITES[i]}): "
                  f"[{pos[0]:+.5f}, {pos[1]:+.5f}, {pos[2]:+.5f}]")
        else:
            print(f"    [{i:2d}] {name:<12s}: NOT FOUND")

    # Check default pose span — should be ~0.3m (30cm) for a rat
    all_pos = []
    for sid in site_ids:
        if sid >= 0:
            all_pos.append(data.site_xpos[sid].copy())
    all_pos = np.array(all_pos)
    span = all_pos.max(axis=0) - all_pos.min(axis=0)
    print(f"\n  Default pose span: [{span[0]:.4f}, {span[1]:.4f}, {span[2]:.4f}] m")
    print(f"  Total extent: {np.linalg.norm(span):.4f} m "
          f"({np.linalg.norm(span)*1000:.1f} mm)")

    # ── Sanity check 2: data statistics ─────────────────────────────────
    print("\n" + "="*60)
    print("SANITY CHECK 2: Keypoint data statistics")
    print("="*60)
    n_valid_per_frame = [v.sum() for _, _, v in frames]
    print(f"  Frames: {len(frames)}")
    print(f"  Keypoints/frame: min={min(n_valid_per_frame)}, "
          f"max={max(n_valid_per_frame)}, mean={np.mean(n_valid_per_frame):.1f}")

    # Data range
    all_kp = np.concatenate([kp[v] for _, kp, v in frames], axis=0)
    print(f"  Keypoint range (mm):")
    print(f"    x: [{all_kp[:,0].min():.1f}, {all_kp[:,0].max():.1f}]")
    print(f"    y: [{all_kp[:,1].min():.1f}, {all_kp[:,1].max():.1f}]")
    print(f"    z: [{all_kp[:,2].min():.1f}, {all_kp[:,2].max():.1f}]")
    centroid = all_kp.mean(axis=0)
    print(f"    centroid: [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]")
    mag = np.linalg.norm(centroid)
    print(f"    centroid magnitude: {mag:.1f} mm")
    print(f"    → scale_factor: {'0.001 (mm→m)' if mag > 10 else '1.0'} "
          f"(RED auto-detects > 10)")

    kp_span = all_kp.max(axis=0) - all_kp.min(axis=0)
    print(f"  Data span: [{kp_span[0]:.1f}, {kp_span[1]:.1f}, {kp_span[2]:.1f}] mm")

    # Per-frame span
    frame_spans = []
    for _, kp, v in frames:
        pts = kp[v]
        if len(pts) >= 2:
            frame_spans.append(np.linalg.norm(pts.max(0) - pts.min(0)))
    print(f"  Per-frame body span: mean={np.mean(frame_spans):.1f} mm, "
          f"std={np.std(frame_spans):.1f} mm")

    # ── Run IK ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"IK TEST: {len(frames)} frames, max_iter={args.max_iter}, "
          f"lr={args.lr}, beta={args.beta}, reg={args.reg}")
    print("="*60)

    residuals = []
    per_site_all = np.full((len(frames), 24), np.nan)
    converged_count = 0
    prev_qpos = None
    prev_frame = -999

    t0 = time.time()
    for i, (frame_id, kp3d, valid) in enumerate(frames):
        # Warm-start from previous frame if within 5 frames
        warm = prev_qpos if abs(frame_id - prev_frame) <= 5 else None

        qpos, res_mm, iters, conv, per_site = solve_ik(
            model, data, minfo, site_ids, kp3d, valid,
            lr=args.lr, beta=args.beta, max_iter=args.max_iter,
            reg_strength=args.reg, warm_qpos=warm,
        )

        if qpos is not None:
            residuals.append(res_mm)
            per_site_all[i] = per_site
            if conv:
                converged_count += 1
            prev_qpos = qpos.copy()
            prev_frame = frame_id

        if (i + 1) % 200 == 0 or i == 0 or i == len(frames) - 1:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            eta = (len(frames) - i - 1) / fps if fps > 0 else 0
            n_valid = int(valid.sum())
            status = "CONV" if conv else "FAIL"
            print(f"  [{i+1:5d}/{len(frames)}] frame={frame_id:6d} "
                  f"kps={n_valid:2d} res={res_mm:.3f}mm "
                  f"iter={iters:4d} {status}  "
                  f"[{fps:.1f} fps, ETA {eta:.0f}s]")

    total_time = time.time() - t0

    # ── Results ─────────────────────────────────────────────────────────
    res = np.array(residuals)
    print(f"\n{'='*60}")
    print(f"RESULTS ({len(residuals)} frames in {total_time:.1f}s, "
          f"{len(residuals)/total_time:.1f} fps)")
    print(f"{'='*60}")

    print(f"\n  Convergence: {converged_count}/{len(residuals)} "
          f"({100*converged_count/len(residuals):.1f}%)")

    print(f"\n  Mean residual (mm): {res.mean():.4f}")
    print(f"  Median:             {np.median(res):.4f}")
    print(f"  Min:                {res.min():.4f}")
    print(f"  Max:                {res.max():.4f}")
    print(f"  Std:                {res.std():.4f}")
    print(f"  < 1.0 mm: {(res < 1.0).sum()}/{len(res)} ({100*(res < 1.0).mean():.1f}%)")
    print(f"  < 0.5 mm: {(res < 0.5).sum()}/{len(res)} ({100*(res < 0.5).mean():.1f}%)")
    print(f"  < 2.0 mm: {(res < 2.0).sum()}/{len(res)} ({100*(res < 2.0).mean():.1f}%)")

    # Per-site breakdown
    print(f"\n  Per-site mean residual (mm):")
    print(f"    {'Site':<12s} {'MJ Site':<28s} {'Mean':>8s} {'Median':>8s} {'Max':>8s} {'N':>6s}")
    print(f"    {'─'*12:<12s} {'─'*28:<28s} {'─'*8:>8s} {'─'*8:>8s} {'─'*8:>8s} {'─'*6:>6s}")
    site_stats = []
    for kp_idx in range(24):
        col = per_site_all[:, kp_idx]
        vals = col[~np.isnan(col)]
        if len(vals) > 0:
            site_stats.append((kp_idx, RAT24_NAMES[kp_idx], RAT24_SITES[kp_idx],
                               np.mean(vals), np.median(vals), np.max(vals), len(vals)))
    site_stats.sort(key=lambda x: x[3])  # sort by mean residual

    for kp_idx, name, mj_name, mean, med, mx, n in site_stats:
        flag = " ⚠" if mean > 2.0 else ""
        print(f"    {name:<12s} {mj_name:<28s} {mean:>8.3f} {med:>8.3f} {mx:>8.3f} {n:>6d}{flag}")

    # ── Geometry validation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GEOMETRY VALIDATION")
    print(f"{'='*60}")

    # Pick a well-converged frame and check body proportions
    best_idx = np.argmin(res)
    best_fid, best_kp, best_valid = frames[best_idx]
    print(f"\n  Best frame: {best_fid} (residual {res[best_idx]:.4f} mm)")

    # Check bilateral symmetry: L vs R distances should be similar
    lr_pairs = [
        ("ShoulderL-ElbowL", 6, 7, "ShoulderR-ElbowR", 10, 11),
        ("ElbowL-WristL", 7, 8, "ElbowR-WristR", 11, 12),
        ("KneeL-AnkleL", 14, 15, "KneeR-AnkleR", 17, 18),
        ("AnkleL-FootL", 15, 16, "AnkleR-FootR", 18, 19),
    ]

    print(f"\n  Bilateral symmetry check (mm):")
    for name_l, i_l1, i_l2, name_r, i_r1, i_r2 in lr_pairs:
        if best_valid[i_l1] and best_valid[i_l2] and best_valid[i_r1] and best_valid[i_r2]:
            d_l = np.linalg.norm(best_kp[i_l1] - best_kp[i_l2])
            d_r = np.linalg.norm(best_kp[i_r1] - best_kp[i_r2])
            ratio = d_l / d_r if d_r > 0 else float('inf')
            flag = " ⚠" if abs(ratio - 1.0) > 0.3 else " ✓"
            print(f"    {name_l}: {d_l:.1f}mm  vs  {name_r}: {d_r:.1f}mm  "
                  f"(ratio {ratio:.2f}){flag}")

    # Check body length (nose to tailbase) — should be ~150-250mm for a rat
    if best_valid[0] and best_valid[5]:
        body_len = np.linalg.norm(best_kp[0] - best_kp[5])
        flag = " ✓" if 100 < body_len < 300 else " ⚠ (unexpected for rat)"
        print(f"\n  Nose-to-TailBase: {body_len:.1f} mm{flag}")

    # Tail length
    tail_pts = [5, 22, 21, 23, 20]  # tailbase → tail1Q → tailmid → tail3Q → tailtip
    tail_len = 0.0
    for j in range(len(tail_pts) - 1):
        if best_valid[tail_pts[j]] and best_valid[tail_pts[j+1]]:
            tail_len += np.linalg.norm(best_kp[tail_pts[j]] - best_kp[tail_pts[j+1]])
    if tail_len > 0:
        print(f"  Tail length: {tail_len:.1f} mm")

    # ── Summary verdict ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    issues = []
    if converged_count / len(residuals) < 0.5:
        issues.append(f"Low convergence: {100*converged_count/len(residuals):.0f}%")
    if res.mean() > 5.0:
        issues.append(f"High mean residual: {res.mean():.2f} mm")
    if res.max() > 20.0:
        issues.append(f"Very high max residual: {res.max():.2f} mm")

    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        print("  → Investigate before running full export")
    else:
        print(f"  ALL CHECKS PASSED")
        print(f"  Convergence: {100*converged_count/len(residuals):.0f}%")
        print(f"  Mean residual: {res.mean():.3f} mm")
        print(f"  → Safe to run RED's 'Solve All & Export' on full dataset")


if __name__ == "__main__":
    main()
