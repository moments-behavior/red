#!/usr/bin/env python3
"""Fit fly body model using adjustabodies pipeline.

Two modes:
  1. Global scale sweep (--mode scale) — find optimal uniform scale
  2. Full adjustabodies fit (--mode full) — per-segment + STAC

Uses the fly species config and 50-keypoint data from prepare_fly_bout_data.py.

Usage:
    # Quick scale sweep (runs locally in <5 min)
    python fit_fly_body.py --mode scale

    # Full adjustabodies pipeline (needs GPU, run on cluster)
    python fit_fly_body.py --mode full

    # Both
    python fit_fly_body.py --mode both
"""

import os
import sys
import json
import time
import math
import argparse
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/johnsonr/datasets/fly_April5/fly_adjustabodies")
# Use MJX-compatible version (stripped actuators/tendons for differentiable FK)
MODEL_XML = Path("/Users/johnsonr/src/fruitfly/mjx/fruitfly_v2/assets/fruitfly.xml")
OUTPUT_DIR = DATA_DIR / "results"

# ── Fly50 site definitions (from species/fly.py) ─────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Also need adjustabodies on path
sys.path.insert(0, "/Users/johnsonr/src/adjustabodies")

from adjustabodies.species.fly import FLY50_SITE_DEFS, FLY50_SITES, FLY_CONFIG
from adjustabodies.arena import ArenaTransform


def load_fly_model(xml_path, scale=1.0, add_sites=True, add_free_joint=True):
    """Load fruitfly XML, add 50 sites, optionally apply global scale."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml_path))

    if add_free_joint:
        body = spec.body("thorax")
        if body is not None:
            # Only add if not already present
            has_free = any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in body.joints)
            if not has_free:
                body.add_freejoint()

    if add_sites:
        for name, body_name, pos in FLY50_SITE_DEFS:
            body = spec.body(body_name)
            if body is None:
                continue
            site = body.add_site()
            site.name = name
            site.pos = pos

    m = spec.compile()

    if abs(scale - 1.0) > 1e-6:
        # Uniform scale: all positions, sizes, etc.
        for i in range(m.nbody):
            m.body_pos[i] *= scale
            m.body_ipos[i] *= scale
        for i in range(m.ngeom):
            m.geom_pos[i] *= scale
            m.geom_size[i] *= scale
        for i in range(m.nsite):
            m.site_pos[i] *= scale
        for i in range(m.njnt):
            m.jnt_pos[i] *= scale
        import mujoco
        mujoco.mj_setConst(m, mujoco.MjData(m))

    return m


def load_fly_keypoints(csv_path, n_kp=50, arena_tf=None, max_frames=None):
    """Load keypoints3d.csv for fly (50 keypoints).

    Returns list of (kp[n_kp, 3], valid[n_kp]) tuples.
    """
    frames = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('frame,'):
                continue
            parts = line.strip().split(',')
            kp = np.zeros((n_kp, 3), dtype=np.float32)
            valid = np.zeros(n_kp, dtype=np.float32)
            for k in range(n_kp):
                base = 1 + k * 4
                if base + 2 < len(parts):
                    try:
                        x, y, z = float(parts[base]), float(parts[base+1]), float(parts[base+2])
                        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                            kp[k] = [x, y, z]
                            valid[k] = 1.0
                    except (ValueError, IndexError):
                        pass
            if arena_tf is not None:
                kp = arena_tf(kp).astype(np.float32)
            frames.append((kp, valid))
            if max_frames and len(frames) >= max_frames:
                break
    return frames


def run_ik_single(m, d, site_ids, target, valid, max_iter=500,
                   lr_trans=0.001, lr_joints=1.0, beta=0.9):
    """Mason-style IK solver for one frame. Returns residual in meters."""
    import mujoco

    nv = m.nv
    active = [i for i, (sid, v) in enumerate(zip(site_ids, valid)) if sid >= 0 and v > 0.5]
    if not active:
        return float('inf')

    n_active = len(active)
    act_sids = [site_ids[i] for i in active]
    act_tgt = target[active]  # [n_active, 3] in meters

    vel = np.zeros(nv)
    mujoco.mj_forward(m, d)

    # Pre-compute cosine decay schedule
    decay = np.array([0.5 * (1 + math.cos(math.pi * t / max_iter)) for t in range(max_iter)])

    for it in range(max_iter):
        mujoco.mj_forward(m, d)

        # Residual
        pos = np.array([d.site_xpos[sid] for sid in act_sids])
        residual = act_tgt - pos  # [n_active, 3]

        # Jacobian (stacked)
        J_full = np.zeros((3 * n_active, nv))
        jacp = np.zeros((3, nv))
        for j, sid in enumerate(act_sids):
            mujoco.mj_jacSite(m, d, jacp, None, sid)
            J_full[3*j:3*j+3] = jacp

        # Gradient
        grad = J_full.T @ residual.ravel()

        # Per-DOF learning rates with cosine decay
        lr = np.full(nv, lr_joints)
        lr[:6] = lr_trans  # free joint (3 trans + 3 rot)
        lr *= decay[it]

        # Momentum update
        vel = beta * vel + lr * grad
        mujoco.mj_integratePos(m, d.qpos, vel, 1.0)

        # Clamp joint limits
        for j in range(m.njnt):
            if m.jnt_type[j] == 3:  # hinge
                qi = m.jnt_qposadr[j]
                lo, hi = m.jnt_range[j]
                if lo < hi:
                    d.qpos[qi] = np.clip(d.qpos[qi], lo, hi)

    mujoco.mj_forward(m, d)
    pos = np.array([d.site_xpos[sid] for sid in act_sids])
    res = np.linalg.norm(act_tgt - pos, axis=1)
    return float(np.mean(res))


def run_scale_sweep(frames, n_scales=21, scale_range=(0.70, 1.10)):
    """Sweep global scale, find optimum via IK residuals."""
    import mujoco

    scales = np.linspace(scale_range[0], scale_range[1], n_scales)
    # Use a subset of frames for speed
    n_test = min(100, len(frames))
    test_indices = np.linspace(0, len(frames)-1, n_test, dtype=int)
    test_frames = [frames[i] for i in test_indices]

    print(f"\nGlobal scale sweep: {n_scales} scales in [{scale_range[0]}, {scale_range[1]}]")
    print(f"Testing on {n_test} frames, 200 IK iters each\n")

    # Load model ONCE at 1.0 scale (we scale the data instead, like test_fly_ik.py)
    m = load_fly_model(MODEL_XML, scale=1.0)
    d = mujoco.MjData(m)

    # Map site names to IDs
    site_ids = []
    for name in FLY50_SITES:
        sid = -1
        for i in range(m.nsite):
            if m.site(i).name == name:
                sid = i
                break
        site_ids.append(sid)

    results = []
    for scale in scales:
        residuals = []
        for kp_cm, valid in test_frames:
            # Scale data: arena_tf gives cm, multiply by model scale
            kp_scaled = kp_cm * scale

            # Reset to default pose
            mujoco.mj_resetData(m, d)
            centroid = kp_scaled[valid > 0.5].mean(axis=0) if (valid > 0.5).any() else np.zeros(3)
            d.qpos[:3] = centroid
            mujoco.mj_forward(m, d)

            res = run_ik_single(m, d, site_ids, kp_scaled, valid, max_iter=200)
            if res < float('inf'):
                residuals.append(res)

        # Residual is in model units. Data was scaled by `scale`, so
        # res_real_cm = res / scale → res_mm = res / scale * 10
        mean_res = np.mean(residuals) / scale * 10 if residuals else float('inf')
        results.append((scale, mean_res))
        print(f"  scale={scale:.3f}  residual={mean_res:.3f} mm  ({len(residuals)}/{n_test} converged)")

    # Find optimum
    best_scale, best_res = min(results, key=lambda x: x[1])
    print(f"\n  BEST: scale={best_scale:.3f}  residual={best_res:.3f} mm")

    return results, best_scale


def run_full_fit(frames, global_scale):
    """Run full adjustabodies pipeline (resize + STAC) with fly species config."""
    import mujoco

    try:
        from adjustabodies.model import build_segment_indices, build_site_indices, save_originals
        from adjustabodies.model import apply_segment_scales
    except ImportError:
        print("ERROR: adjustabodies not installed. Run: pip install -e /Users/johnsonr/src/adjustabodies")
        sys.exit(1)

    # Check for JAX/MJX
    try:
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
        from adjustabodies.resize import run_resize_phase, build_mjx_scale_fn
        from adjustabodies.stac import run_stac_phase
        print(f"JAX devices: {jax.devices()}")
    except ImportError as e:
        print(f"ERROR: JAX/MJX not available ({e}). Run on cluster with GPU.")
        print("For CPU-only global scale sweep, use: --mode scale")
        sys.exit(1)

    # Load model with fly sites
    m = load_fly_model(MODEL_XML, scale=global_scale, add_free_joint=True)

    # Disable collision (we only need FK for fitting)
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT |
                            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    # Fix geoms for MJX (ellipsoid not supported)
    for i in range(m.ngeom):
        if m.geom_type[i] == 8:  # ELLIPSOID → SPHERE
            m.geom_type[i] = 2

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"Model: nq={m.nq}, nbody={m.nbody}, nsite={m.nsite}, nv={m.nv}")

    # Build indices using fly species config
    segments = build_segment_indices(m, FLY_CONFIG.segments)
    site_ids = build_site_indices(m, FLY_CONFIG.sites)
    orig = save_originals(m)
    n_seg = len(segments)

    print(f"Segments: {n_seg} ({', '.join(n for n, _ in segments)})")
    print(f"Sites: {sum(1 for s in site_ids if s >= 0)}/{len(site_ids)} mapped")

    # MJX model
    mx_base = mjx.put_model(m)
    apply_scales_fn = build_mjx_scale_fn(m, segments, orig)

    # Init scales at 1.0 (global scale already applied)
    init_rel_scales = np.array(
        [FLY_CONFIG.segment_length_init.get(name, 1.0) for name, _ in segments],
        dtype=np.float32)

    # ════════════════════════════════════════════════════════════════
    # PHASE 1: Body segment scaling
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 1: Body segment scaling")
    print(f"{'='*60}")

    params, pre_res, post_res_1 = run_resize_phase(
        m, mx_base, segments, site_ids, orig, frames, apply_scales_fn,
        init_global=1.0, init_rel_scales=init_rel_scales,
        n_rounds=6, m_iters=300, ik_iters=500,
        lr_scale=0.003, reg_scale=0.001, verbose=True)

    gs = float(params['global_scale'])
    rs = np.array(params['rel_scales'])
    print(f"\nPhase 1 scales (absolute = global_scale × rel_scale × {global_scale:.3f}):")
    for g, (name, _) in enumerate(segments):
        print(f"  {name:<14s} {global_scale * gs * rs[g]:.4f} (rel={rs[g]:.3f})")

    # ════════════════════════════════════════════════════════════════
    # PHASE 2: STAC site offsets with L/R symmetry
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: STAC site offsets with L/R symmetry")
    print(f"{'='*60}")

    # Bake Phase 1 scales
    scales = {name: gs * rs[g] for g, (name, _) in enumerate(segments)}
    apply_segment_scales(m, segments, scales, orig)
    orig_scaled = save_originals(m)
    mx_base_scaled = mjx.put_model(m)
    apply_scales_fn_2 = build_mjx_scale_fn(m, segments, orig_scaled)

    params['rel_scales'] = jnp.ones(n_seg)
    params['global_scale'] = jnp.array(1.0)

    params, pre_res_2, post_res_2 = run_stac_phase(
        m, mx_base_scaled, segments, site_ids, orig_scaled, frames,
        apply_scales_fn_2, params,
        sym_config=(FLY_CONFIG.lr_site_pairs, FLY_CONFIG.midline_sites),
        n_rounds=6, m_iters=300, ik_iters=500,
        lr_scale=0.0001, lr_offset=0.001,
        reg_scale=10.0, reg_offset=0.01, verbose=True)

    # ════════════════════════════════════════════════════════════════
    # Save results
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    gs_final = float(params['global_scale'])
    rs_final = np.array(params['rel_scales'])
    offs_final = np.array(params['site_offsets'])
    final_scales = {name: gs_final * rs_final[g] for g, (name, _) in enumerate(segments)}
    apply_segment_scales(m, segments, final_scales, orig_scaled)
    m.site_pos[:] += offs_final
    mujoco.mj_setConst(m, mujoco.MjData(m))

    # Absolute scales (Phase 1 × Phase 2 × initial global)
    abs_scales = {}
    for g, (name, _) in enumerate(segments):
        abs_scales[name] = global_scale * scales[name] * final_scales[name]

    output_path = str(OUTPUT_DIR / "fruitfly_fitted.mjb")
    disps = np.linalg.norm(offs_final, axis=1) * 1000
    metadata = {
        'adjustabodies_version': '0.1.0-fly',
        'species': 'fly',
        'base_model': str(MODEL_XML),
        'data_source': str(DATA_DIR),
        'initial_global_scale': global_scale,
        'n_frames': len(frames),
        'phase1_residual_mm': post_res_1,
        'phase2_residual_mm': post_res_2,
        'segment_scales': abs_scales,
        'top_site_offsets': {m.site(i).name: float(disps[i])
                             for i in np.argsort(disps)[::-1][:10] if disps[i] > 0.1},
    }

    from adjustabodies.io import save_fitted_model
    save_fitted_model(m, output_path, metadata)

    print(f"\nFinal segment scales (absolute):")
    for name, s in abs_scales.items():
        print(f"  {name:<14s} {s:.4f}")
    print(f"\nResidual: {pre_res:.3f} → {post_res_1:.3f} → {post_res_2:.3f} mm")
    print(f"Saved: {output_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Fit fly body model")
    parser.add_argument("--mode", choices=["scale", "full", "both"], default="scale",
                        help="scale: global scale sweep, full: adjustabodies fit, both: sweep then fit")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Max frames for fitting (default: 500)")
    parser.add_argument("--scale-range", type=float, nargs=2, default=[0.70, 1.10],
                        help="Scale sweep range")
    parser.add_argument("--n-scales", type=int, default=21)
    parser.add_argument("--initial-scale", type=float, default=1.0,
                        help="Initial global scale for full fit (default: 1.0, since 0.82 is baked into arena transform)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    session_path = DATA_DIR / "mujoco_session.json"
    arena_tf = ArenaTransform.from_session(str(session_path))
    kp3d_csv = DATA_DIR / "labeled_data" / "bout_frames" / "keypoints3d.csv"
    assert kp3d_csv.exists(), f"Not found: {kp3d_csv}"

    print(f"Loading keypoints from {kp3d_csv} ...")
    frames = load_fly_keypoints(str(kp3d_csv), n_kp=50, arena_tf=arena_tf,
                                 max_frames=args.max_frames)
    print(f"Loaded {len(frames)} frames")

    # Check data range
    all_kp = np.array([kp for kp, _ in frames])
    valid_kp = all_kp[all_kp != 0].reshape(-1, 3)
    print(f"Keypoint range (meters): x=[{valid_kp[:,0].min():.4f}, {valid_kp[:,0].max():.4f}], "
          f"y=[{valid_kp[:,1].min():.4f}, {valid_kp[:,1].max():.4f}], "
          f"z=[{valid_kp[:,2].min():.4f}, {valid_kp[:,2].max():.4f}]")

    global_scale = args.initial_scale

    if args.mode in ("scale", "both"):
        results, best_scale = run_scale_sweep(frames, n_scales=args.n_scales,
                                               scale_range=tuple(args.scale_range))
        # Save scale sweep results
        scale_path = OUTPUT_DIR / "scale_sweep.json"
        with open(scale_path, 'w') as f:
            json.dump({'results': [(float(s), float(r)) for s, r in results],
                       'best_scale': float(best_scale)}, f, indent=2)
        print(f"Saved scale sweep: {scale_path}")
        global_scale = best_scale

    if args.mode in ("full", "both"):
        metadata = run_full_fit(frames, global_scale)
        # Save metadata
        meta_path = OUTPUT_DIR / "fit_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
