#!/usr/bin/env python3
"""Thorough IK test on fly model with Mason-style settings.

Loads the fruitfly MuJoCo model, adds 50 keypoint sites, scales by 0.82,
reads all labeled 3D keypoints from all sessions, and runs gradient-descent
IK with analytical Jacobians (mj_jacSite). Compares:
  A) Cosine annealing + separate LR (lr_trans=0.001, lr_joints=1.0)
  B) Baseline: single lr=0.01, no annealing

Matches the C++ mujoco_ik.h solver as closely as possible.
All hot loops are fully vectorized with numpy for performance.

Usage:
    pip install mujoco numpy
    python3 scripts/test_fly_ik.py
"""

import os
import sys
import glob
import time
import math
import numpy as np

# ---------- Fly50 site definitions (from build_fly_model.py) ----------

FLY50_SITES = [
    ("Antenna_Base", "head",              [0.0, 0.038, 0.012]),
    ("EyeL",         "head",              [-0.0245, 0.0135, 0.0285]),
    ("EyeR",         "head",              [0.0245, 0.0135, 0.0285]),
    ("Scutellum",    "thorax",            [-0.049, 0.0, 0.04]),
    ("Abd_A4",       "abdomen_3",         [0.0, 0.0335, 0.021]),
    ("Abd_tip",      "abdomen_7",         [0.0, 0.0395, -0.001]),
    ("WingL_base",   "thorax",            [-0.0095, 0.045, 0.0175]),
    ("WingL_V12",    "wing_left",         [-0.0072, -0.2125, 0.0075]),
    ("WingL_V13",    "wing_left",         [0.0221, -0.2562, -0.0253]),
    ("T1L_ThxCx",    "coxa_T1_left",      [0, 0, 0]),
    ("T1L_Tro",      "femur_T1_left",     [0, 0, 0]),
    ("T1L_FeTi",     "tibia_T1_left",     [0, 0, 0]),
    ("T1L_TiTa",     "tarsus1_T1_left",   [0, 0, 0]),
    ("T1L_TaT1",     "tarsus2_T1_left",   [0, 0, 0]),
    ("T1L_TaT3",     "tarsus4_T1_left",   [0, 0, 0]),
    ("T1L_TaTip",    "tarsal_claw_T1_left", [0, 0.0105, 0.0006]),
    ("T2L_Tro",      "femur_T2_left",     [0, 0, 0]),
    ("T2L_FeTi",     "tibia_T2_left",     [0, 0, 0]),
    ("T2L_TiTa",     "tarsus1_T2_left",   [0, 0, 0]),
    ("T2L_TaT1",     "tarsus2_T2_left",   [0, 0, 0]),
    ("T2L_TaT3",     "tarsus4_T2_left",   [0, 0, 0]),
    ("T2L_TaTip",    "tarsal_claw_T2_left", [0, 0.0122, 0.0006]),
    ("T3L_Tro",      "femur_T3_left",     [0, 0, 0]),
    ("T3L_FeTi",     "tibia_T3_left",     [0, 0, 0]),
    ("T3L_TiTa",     "tarsus1_T3_left",   [0, 0, 0]),
    ("T3L_TaT1",     "tarsus2_T3_left",   [0, 0, 0]),
    ("T3L_TaT3",     "tarsus4_T3_left",   [0, 0, 0]),
    ("T3L_TaTip",    "tarsal_claw_T3_left", [0, 0.0111, 0.0008]),
    ("WingR_base",   "thorax",            [-0.0095, -0.045, 0.0175]),
    ("WingR_V12",    "wing_right",        [0.0072, 0.2125, -0.0075]),
    ("WingR_V13",    "wing_right",        [-0.0221, 0.2562, 0.0253]),
    ("T1R_ThxCx",    "coxa_T1_right",     [0, 0, 0]),
    ("T1R_Tro",      "femur_T1_right",    [0, 0, 0]),
    ("T1R_FeTi",     "tibia_T1_right",    [0, 0, 0]),
    ("T1R_TiTa",     "tarsus1_T1_right",  [0, 0, 0]),
    ("T1R_TaT1",     "tarsus2_T1_right",  [0, 0, 0]),
    ("T1R_TaT3",     "tarsus4_T1_right",  [0, 0, 0]),
    ("T1R_TaTip",    "tarsal_claw_T1_right", [0, -0.0101, -0.0006]),
    ("T2R_Tro",      "femur_T2_right",    [0, 0, 0]),
    ("T2R_FeTi",     "tibia_T2_right",    [0, 0, 0]),
    ("T2R_TiTa",     "tarsus1_T2_right",  [0, 0, 0]),
    ("T2R_TaT1",     "tarsus2_T2_right",  [0, 0, 0]),
    ("T2R_TaT3",     "tarsus4_T2_right",  [0, 0, 0]),
    ("T2R_TaTip",    "tarsal_claw_T2_right", [0, -0.0118, -0.0006]),
    ("T3R_Tro",      "femur_T3_right",    [0, 0, 0]),
    ("T3R_FeTi",     "tibia_T3_right",    [0, 0, 0]),
    ("T3R_TiTa",     "tarsus1_T3_right",  [0, 0, 0]),
    ("T3R_TaT1",     "tarsus2_T3_right",  [0, 0, 0]),
    ("T3R_TaT3",     "tarsus4_T3_right",  [0, 0, 0]),
    ("T3R_TaTip",    "tarsal_claw_T3_right", [0, -0.0109, -0.0008]),
]

FLY50_NAMES = [s[0] for s in FLY50_SITES]


def load_model(scale=0.82):
    """Load fruitfly XML, add 50 sites, compile, return (model, data, site_ids, scale)."""
    import mujoco

    script_dir = os.path.dirname(os.path.abspath(__file__))
    red_root = os.path.dirname(script_dir)

    # Try repo clone first, then models dir
    xml_path = os.path.join(red_root, "lib", "fruitfly", "fruitfly_v2", "assets", "fruitfly.xml")
    if not os.path.exists(xml_path):
        xml_path = os.path.join(red_root, "models", "fruitfly", "fruitfly.xml")
    if not os.path.exists(xml_path):
        print(f"ERROR: fruitfly.xml not found")
        sys.exit(1)

    print(f"Loading model: {xml_path}")
    spec = mujoco.MjSpec.from_file(xml_path)

    # Add fly50 sites
    added = 0
    for name, body_name, pos in FLY50_SITES:
        body = spec.body(body_name)
        if body is None:
            print(f"  WARNING: body '{body_name}' not found for site '{name}'")
            continue
        site = body.add_site()
        site.name = name
        site.pos = pos
        added += 1
    print(f"Added {added}/50 keypoint sites")

    model = spec.compile()
    data = mujoco.MjData(model)
    print(f"Model: {model.nbody} bodies, {model.njnt} joints, {model.nsite} sites, nv={model.nv}, nq={model.nq}")

    # Map site names to indices
    site_ids = {}
    for i in range(model.nsite):
        site_ids[model.site(i).name] = i

    missing = [n for n in FLY50_NAMES if n not in site_ids]
    if missing:
        print(f"WARNING: {len(missing)} sites not found: {missing[:5]}...")

    return model, data, site_ids, scale


def precompute_model_info(model):
    """Pre-compute joint info that doesn't change between frames."""
    nv = model.nv

    # Find free joint
    has_free_joint = False
    free_jnt_qa = -1
    free_jnt_dofadr = -1
    for j in range(model.njnt):
        if model.jnt_type[j] == 0:  # mjJNT_FREE
            has_free_joint = True
            free_jnt_qa = int(model.jnt_qposadr[j])
            free_jnt_dofadr = int(model.jnt_dofadr[j])
            break

    # Hinge DOFs for regularization
    hinge_dofs = []
    hinge_qa = []  # corresponding qpos addresses
    for j in range(model.njnt):
        if model.jnt_type[j] == 3:  # mjJNT_HINGE
            dof = int(model.jnt_dofadr[j])
            qa = int(model.jnt_qposadr[j])
            hinge_dofs.append(dof)
            hinge_qa.append(qa)
    hinge_dofs = np.array(hinge_dofs, dtype=np.intp)
    hinge_qa = np.array(hinge_qa, dtype=np.intp)

    # Per-DOF LR mask: True for translation DOFs of free joint
    is_trans = np.zeros(nv, dtype=bool)
    if has_free_joint:
        is_trans[free_jnt_dofadr:free_jnt_dofadr + 3] = True

    # Joint limits for clamping
    limited_joints = []  # (qa, lo, hi)
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        jt = model.jnt_type[j]
        if jt == 3 or jt == 2:  # HINGE or SLIDE
            qa = int(model.jnt_qposadr[j])
            lo = float(model.jnt_range[j, 0])
            hi = float(model.jnt_range[j, 1])
            limited_joints.append((qa, lo, hi))

    return {
        'has_free_joint': has_free_joint,
        'free_jnt_qa': free_jnt_qa,
        'hinge_dofs': hinge_dofs,
        'hinge_qa': hinge_qa,
        'is_trans': is_trans,
        'limited_joints': limited_joints,
    }


def load_all_keypoints(data_dir):
    """Load all keypoints3d.csv files. Returns list of (session, frame, pos[50,3], valid[50])."""
    sessions = sorted(glob.glob(os.path.join(data_dir, "*", "keypoints3d.csv")))
    print(f"Found {len(sessions)} sessions with keypoints3d.csv")

    frames = []
    for csv_path in sessions:
        session = os.path.basename(os.path.dirname(csv_path))
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('frame,'):
                    continue
                parts = line.split(',')
                frame_num = int(parts[0])
                positions = np.full((50, 3), np.nan)
                valid = np.zeros(50, dtype=bool)

                for kp_idx in range(50):
                    base = 1 + kp_idx * 4
                    if base + 2 < len(parts):
                        try:
                            x, y, z = float(parts[base]), float(parts[base+1]), float(parts[base+2])
                            if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                                positions[kp_idx] = [x, y, z]
                                valid[kp_idx] = True
                        except (ValueError, IndexError):
                            pass

                if valid.sum() >= 3:
                    frames.append((session, frame_num, positions, valid))

    print(f"Loaded {len(frames)} frames with >= 3 valid keypoints")
    return frames


def solve_ik(model, data, minfo, site_ids, positions_mm, valid, scale,
             lr_trans, lr_joints, beta, max_iter, reg_strength, cosine_annealing):
    """Vectorized IK solver matching C++ mujoco_ik.h.

    Returns: per_site_residual_mm[50] (NaN for invalid), iterations, converged
    """
    import mujoco

    nv = model.nv

    # Build active targets
    active_kp = []
    active_sid = []
    active_target = []
    for kp_idx in range(50):
        if not valid[kp_idx]:
            continue
        sname = FLY50_NAMES[kp_idx]
        if sname not in site_ids:
            continue
        active_kp.append(kp_idx)
        active_sid.append(site_ids[sname])
        # mm -> model units: * 0.1 (mm->cm) * scale
        active_target.append(positions_mm[kp_idx] * 0.1 * scale)

    N = len(active_kp)
    if N == 0:
        return np.full(50, np.nan), 0, False

    active_sid_arr = np.array(active_sid, dtype=np.intp)
    target_arr = np.array(active_target)  # [N, 3]

    # Reset to default pose
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Root alignment
    if minfo['has_free_joint']:
        tc = target_arr.mean(axis=0)
        mc = data.site_xpos[active_sid_arr].mean(axis=0)
        qa = minfo['free_jnt_qa']
        data.qpos[qa:qa+3] += tc - mc
        mujoco.mj_forward(model, data)

    # Pre-allocate
    jacp = np.zeros((3 * N, nv))
    jacp_k = np.zeros((3, nv))
    update = np.zeros(nv)
    step = np.zeros(nv)

    # Pre-compute per-DOF LR vector
    is_trans = minfo['is_trans']
    hinge_dofs = minfo['hinge_dofs']
    hinge_qa = minfo['hinge_qa']

    # Cosine annealing schedule (precompute all decay factors)
    if cosine_annealing and max_iter > 0:
        iters = np.arange(max_iter, dtype=np.float64)
        decay_schedule = 0.5 * (1.0 + np.cos(np.pi * iters / max_iter))
    else:
        decay_schedule = None

    for it in range(max_iter):
        # Stacked Jacobian [3N x nv]
        for k in range(N):
            jacp_k[:] = 0.0
            mujoco.mj_jacSite(model, data, jacp_k, None, active_sid_arr[k])
            jacp[3*k:3*k+3, :] = jacp_k

        # Residual: site_xpos - target  [3N]
        site_pos = data.site_xpos[active_sid_arr]  # [N, 3]
        diff = (site_pos - target_arr).ravel()  # [3N]

        # Gradient: 2 * J^T * diff  [nv]
        grad = 2.0 * (jacp.T @ diff)

        # Regularization on hinge joints
        if reg_strength > 0 and len(hinge_dofs) > 0:
            grad[hinge_dofs] += 2.0 * reg_strength * data.qpos[hinge_qa]

        # Momentum
        update[:] = beta * update + grad

        # Learning rate
        if decay_schedule is not None:
            decay = decay_schedule[it]
            lr_t = lr_trans * decay
            lr_j = lr_joints * decay
        else:
            lr_t = lr_trans
            lr_j = lr_joints

        # Build step with separate LR (vectorized)
        if abs(lr_t - lr_j) > 1e-12 and minfo['has_free_joint']:
            # is_trans mask selects translation DOFs -> lr_t, rest -> lr_j
            lr_vec = np.where(is_trans, lr_t, lr_j)
            step[:] = -lr_vec * update
        else:
            step[:] = -lr_t * update

        # Integrate qpos
        mujoco.mj_integratePos(model, data.qpos, step, 1.0)

        # Clamp joints
        for qa, lo, hi in minfo['limited_joints']:
            data.qpos[qa] = max(lo, min(hi, data.qpos[qa]))

        # Forward kinematics
        mujoco.mj_fwdPosition(model, data)

    # Final per-site residual in mm
    per_site_mm = np.full(50, np.nan)
    site_pos = data.site_xpos[active_sid_arr]  # [N, 3]
    residuals = np.linalg.norm(site_pos - target_arr, axis=1)
    for i, kp_idx in enumerate(active_kp):
        per_site_mm[kp_idx] = residuals[i] / (0.1 * scale)

    return per_site_mm, max_iter, True


def run_experiment(model, data, minfo, site_ids, frames, scale, config_name,
                   lr_trans, lr_joints, beta, max_iter, reg_strength,
                   cosine_annealing):
    """Run IK on all frames, report statistics."""
    print(f"\n{'='*70}")
    print(f"Config: {config_name}")
    print(f"  lr_trans={lr_trans}, lr_joints={lr_joints}, beta={beta}")
    print(f"  max_iter={max_iter}, reg={reg_strength}, cosine={cosine_annealing}")
    print(f"  scale={scale}, frames={len(frames)}")
    print(f"{'='*70}")

    all_mean_residuals = []
    all_per_site = np.full((len(frames), 50), np.nan)
    total_t0 = time.time()

    for fi, (session, frame_num, positions, valid) in enumerate(frames):
        per_site_mm, iters, conv = solve_ik(
            model, data, minfo, site_ids, positions, valid, scale,
            lr_trans=lr_trans, lr_joints=lr_joints, beta=beta,
            max_iter=max_iter, reg_strength=reg_strength,
            cosine_annealing=cosine_annealing,
        )

        valid_residuals = per_site_mm[~np.isnan(per_site_mm)]
        mean_res = np.mean(valid_residuals) if len(valid_residuals) > 0 else np.nan
        all_mean_residuals.append(mean_res)
        all_per_site[fi] = per_site_mm

        if (fi + 1) % 50 == 0 or fi == 0 or fi == len(frames) - 1:
            elapsed = time.time() - total_t0
            n_valid = valid.sum()
            eta = elapsed / (fi + 1) * (len(frames) - fi - 1) if fi > 0 else 0
            print(f"  [{fi+1:4d}/{len(frames)}] session={session} frame={frame_num:5d} "
                  f"kps={n_valid:2d} mean_res={mean_res:.3f}mm "
                  f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    total_time = time.time() - total_t0
    mean_res_arr = np.array(all_mean_residuals)
    valid_mean = mean_res_arr[~np.isnan(mean_res_arr)]

    print(f"\n--- Results: {config_name} ---")
    print(f"  Total time: {total_time:.1f}s ({total_time/len(frames):.2f}s/frame)")
    print(f"  Per-frame mean residual (mm):")
    print(f"    Mean:   {np.mean(valid_mean):.4f}")
    print(f"    Median: {np.median(valid_mean):.4f}")
    print(f"    Min:    {np.min(valid_mean):.4f}")
    print(f"    Max:    {np.max(valid_mean):.4f}")
    print(f"    Std:    {np.std(valid_mean):.4f}")

    n_total = len(valid_mean)
    n_below_1 = int(np.sum(valid_mean < 1.0))
    n_below_05 = int(np.sum(valid_mean < 0.5))
    n_below_03 = int(np.sum(valid_mean < 0.3))
    print(f"  Convergence:")
    print(f"    < 1.0 mm: {n_below_1}/{n_total} ({100*n_below_1/n_total:.1f}%)")
    print(f"    < 0.5 mm: {n_below_05}/{n_total} ({100*n_below_05/n_total:.1f}%)")
    print(f"    < 0.3 mm: {n_below_03}/{n_total} ({100*n_below_03/n_total:.1f}%)")

    # Per-site statistics
    print(f"\n  Per-site mean residual (mm):")
    site_means = []
    for kp_idx in range(50):
        col = all_per_site[:, kp_idx]
        vals = col[~np.isnan(col)]
        if len(vals) > 0:
            site_means.append((FLY50_NAMES[kp_idx], np.mean(vals), len(vals)))
    site_means.sort(key=lambda x: x[1])

    print(f"    {'Site':<20s} {'Mean(mm)':>10s} {'N frames':>10s}")
    print(f"    {'----':<20s} {'--------':>10s} {'--------':>10s}")
    print(f"    --- Best 10 ---")
    for name, mean, n in site_means[:10]:
        print(f"    {name:<20s} {mean:>10.4f} {n:>10d}")
    print(f"    --- Worst 10 ---")
    for name, mean, n in site_means[-10:]:
        print(f"    {name:<20s} {mean:>10.4f} {n:>10d}")

    return {
        'config': config_name,
        'mean': np.mean(valid_mean),
        'median': np.median(valid_mean),
        'min': np.min(valid_mean),
        'max': np.max(valid_mean),
        'std': np.std(valid_mean),
        'n_below_1mm': n_below_1,
        'n_below_05mm': n_below_05,
        'n_below_03mm': n_below_03,
        'n_total': n_total,
        'total_time': total_time,
    }


def main():
    data_dir = "/Users/johnsonr/red_demos/flypred5/labeled_data"
    scale = 0.82

    model, data, site_ids, scale = load_model(scale=scale)
    minfo = precompute_model_info(model)

    frames = load_all_keypoints(data_dir)
    if not frames:
        print("ERROR: No frames loaded")
        sys.exit(1)

    n_kps = [f[3].sum() for f in frames]
    print(f"Keypoints per frame: min={min(n_kps)}, max={max(n_kps)}, mean={np.mean(n_kps):.1f}")

    results = []

    # Note on iteration count: convergence profiling on 5 frames shows:
    #   Mason-style: converged by ~200 iters (0.3472mm at 200 == 0.3472 at 5000)
    #   Baseline:    converged by ~2000 iters (0.3474mm at 2000 == 0.3472 at 5000)
    # Using 2000 iters for thorough testing — well past convergence for both.

    # --- Config A: Mason-style (cosine annealing + separate LR) ---
    r = run_experiment(
        model, data, minfo, site_ids, frames, scale,
        config_name="Mason-style (cosine + separate LR)",
        lr_trans=0.001, lr_joints=1.0, beta=0.9,
        max_iter=2000, reg_strength=0.01,
        cosine_annealing=True,
    )
    results.append(r)

    # --- Config B: Baseline (single LR, no annealing) ---
    r = run_experiment(
        model, data, minfo, site_ids, frames, scale,
        config_name="Baseline (single lr=0.01, no annealing)",
        lr_trans=0.01, lr_joints=0.01, beta=0.9,
        max_iter=2000, reg_strength=0.01,
        cosine_annealing=False,
    )
    results.append(r)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Config':<45s} {'Mean':>8s} {'Median':>8s} {'<1mm':>8s} {'<0.5mm':>8s} {'Time':>8s}")
    print(f"{'-'*45:<45s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s}")
    for r in results:
        pct1 = 100 * r['n_below_1mm'] / r['n_total'] if r['n_total'] > 0 else 0
        pct05 = 100 * r['n_below_05mm'] / r['n_total'] if r['n_total'] > 0 else 0
        print(f"{r['config']:<45s} {r['mean']:>7.3f}m {r['median']:>7.3f}m {pct1:>6.1f}% {pct05:>6.1f}% {r['total_time']:>6.1f}s")


if __name__ == "__main__":
    main()
