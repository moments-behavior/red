#!/usr/bin/env python3
"""Differentiable body model fitting via MuJoCo MJX on GPU.

Alternating optimization (like STAC but with gradient-based body scaling):
  Q-phase: IK solve per frame (non-differentiable, uses MuJoCo C API)
  M-phase: Gradient descent on body scales + site offsets via MJX FK (differentiable)

This avoids the huge compilation cost of differentiating through the IK loop.
The M-phase only differentiates through FK: scales → body_pos → FK → site_xpos → loss

Usage (on cluster):
    bsub -W 1:00 -n 12 -gpu "num=1" -q gpu_a100 -P johnson -Is /bin/bash
    source ~/miniconda3/bin/activate && conda activate mjx
    python3 mjx_body_resize.py --data-dir /path/to/project
"""

import os
import sys
import json
import math
import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp
import optax

import mujoco
from mujoco import mjx

# ── Segment definitions ──────────────────────────────────────────────────
SEGMENT_DEFS = [
    ('head',      ['skull', 'jaw']),
    ('neck',      ['vertebra_cervical_5', 'vertebra_cervical_4', 'vertebra_cervical_3',
                   'vertebra_cervical_2', 'vertebra_cervical_1', 'vertebra_axis',
                   'vertebra_atlant']),
    ('spine',     [f'vertebra_{i}' for i in range(1, 7)]),
    ('pelvis',    ['pelvis']),
    ('tail',      [f'vertebra_C{i}' for i in range(1, 31)]),
    ('scapula',   ['scapula_L', 'scapula_R']),
    ('upper_arm', ['upper_arm_L', 'upper_arm_R']),
    ('lower_arm', ['lower_arm_L', 'lower_arm_R']),
    ('hand',      ['hand_L', 'hand_R', 'finger_L', 'finger_R']),
    ('upper_leg', ['upper_leg_L', 'upper_leg_R']),
    ('lower_leg', ['lower_leg_L', 'lower_leg_R']),
    ('foot',      ['foot_L', 'foot_R', 'toe_L', 'toe_R']),
]

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


def load_model(xml_path):
    """Load rodent model, add free joint, fix geoms for MJX."""
    spec = mujoco.MjSpec.from_file(xml_path)
    spec.body("torso").add_freejoint()
    m = spec.compile()
    # Fix geoms for MJX
    for i in range(m.ngeom):
        if m.geom_type[i] == 8:  # ELLIPSOID → SPHERE
            m.geom_type[i] = 2
        if m.geom_type[i] == 6:  # BOX
            m.geom_contype[i] = 0
            m.geom_conaffinity[i] = 0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    return m


def load_keypoints(data_dir, max_frames=None):
    """Load keypoints3d.csv with arena transform."""
    with open(os.path.join(data_dir, 'mujoco_session.json')) as f:
        sess = json.load(f)
    arena = sess['arena']
    R = np.array(arena['R']).reshape(3, 3)
    scale = arena['scale']
    t = np.array(arena.get('t', [0, 0, 0]))

    labeled_dir = os.path.join(data_dir, 'labeled_data')
    kp3d_csv = None
    for session in sorted(os.listdir(labeled_dir), reverse=True):
        candidate = os.path.join(labeled_dir, session, 'keypoints3d.csv')
        if os.path.exists(candidate):
            kp3d_csv = candidate
            break
    assert kp3d_csv, f"No keypoints3d.csv in {labeled_dir}"

    frames = []
    with open(kp3d_csv) as f:
        for line in f:
            if line.startswith('#') or line.startswith('frame,'):
                continue
            parts = line.strip().split(',')
            kp = np.zeros((24, 3), dtype=np.float32)
            valid = np.zeros(24, dtype=np.float32)
            for k in range(24):
                base = 1 + k * 4
                x, y, z = float(parts[base]), float(parts[base+1]), float(parts[base+2])
                if not math.isnan(x):
                    kp[k] = [x, y, z]
                    valid[k] = 1.0
            kp_mj = (scale * (kp @ R.T) + t).astype(np.float32)
            frames.append((kp_mj, valid))
            if max_frames and len(frames) >= max_frames:
                break
    print(f"Loaded {len(frames)} frames")
    return frames


def q_phase_cpu(m, frames, site_ids, max_iters=1000):
    """Q-phase: solve IK for all frames using MuJoCo C API (CPU, non-differentiable).
    Returns qpos array [N, nq]."""
    d = mujoco.MjData(m)
    nq = m.nq
    N = len(frames)
    all_qpos = np.zeros((N, nq), dtype=np.float64)

    nv = m.nv
    jacp = np.zeros((3, nv))
    skeleton_to_site = []
    for k in range(24):
        skeleton_to_site.append(site_ids[k])

    for i, (kp_mj, valid) in enumerate(frames):
        # Reset + root alignment
        mujoco.mj_resetData(m, d)
        valid_kp = kp_mj[valid > 0.5]
        if len(valid_kp) == 0:
            continue
        centroid = valid_kp.mean(axis=0)
        for j in range(m.njnt):
            if m.jnt_type[j] == 0:  # FREE
                qa = int(m.jnt_qposadr[j])
                d.qpos[qa:qa+3] = centroid
                d.qpos[qa+3] = 1.0  # quaternion w
                break
        mujoco.mj_forward(m, d)

        # Gradient descent IK
        update = np.zeros(nv)
        targets = [(site_ids[k], kp_mj[k]) for k in range(24)
                    if valid[k] > 0.5 and site_ids[k] >= 0]

        for it in range(max_iters):
            grad = np.zeros(nv)
            for sid, tgt in targets:
                jacp[:] = 0
                mujoco.mj_jacSite(m, d, jacp, None, sid)
                grad += 2.0 * jacp.T @ (d.site_xpos[sid] - tgt)
            update = 0.99 * update + grad
            step = -0.001 * update
            mujoco.mj_integratePos(m, d.qpos, step, 1.0)
            mujoco.mj_fwdPosition(m, d)

        all_qpos[i] = d.qpos.copy()

    return all_qpos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-xml', default=None)
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--ik-iters', type=int, default=500)
    parser.add_argument('--n-rounds', type=int, default=6)
    parser.add_argument('--m-iters', type=int, default=200)
    parser.add_argument('--m-lr', type=float, default=0.001)
    parser.add_argument('--reg-scale', type=float, default=0.1)
    parser.add_argument('--reg-offset', type=float, default=0.1)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    if args.model_xml is None:
        args.model_xml = "/groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_data_driven_limits.xml"

    m = load_model(args.model_xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    nq = m.nq
    print(f"Model: nq={nq}, nbody={m.nbody}, nsite={m.nsite}")

    # Build segment indices
    segments = []
    for name, body_names in SEGMENT_DEFS:
        bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
                for bn in body_names]
        bids = [b for b in bids if b >= 0]
        if bids:
            segments.append((name, bids))
    n_seg = len(segments)
    print(f"Segments: {n_seg}")

    # Site indices
    site_ids = []
    for name in RAT24_SITES:
        sid = -1
        for i in range(m.nsite):
            if m.site(i).name == name:
                sid = i
                break
        site_ids.append(sid)

    # Load data
    frames = load_keypoints(args.data_dir, max_frames=args.max_frames)
    N = len(frames)
    kp3d_all = jnp.array(np.stack([f[0] for f in frames]))   # [N, 24, 3]
    valid_all = jnp.array(np.stack([f[1] for f in frames]))   # [N, 24]

    # Precompute mappings for JAX
    body_to_seg = np.full(m.nbody, -1, dtype=np.int32)
    for g, (name, bids) in enumerate(segments):
        for bid in bids:
            body_to_seg[bid] = g
    body_to_seg_j = jnp.array(body_to_seg)
    geom_bodyid_j = jnp.array(m.geom_bodyid)
    site_bodyid_j = jnp.array(m.site_bodyid)
    jnt_bodyid_j = jnp.array(m.jnt_bodyid)
    site_ids_j = jnp.array(site_ids)

    # Save originals
    orig_body_pos = jnp.array(m.body_pos.copy())
    orig_body_ipos = jnp.array(m.body_ipos.copy())
    orig_geom_pos = jnp.array(m.geom_pos.copy())
    orig_geom_size = jnp.array(m.geom_size.copy())
    orig_site_pos = jnp.array(m.site_pos.copy())
    orig_jnt_pos = jnp.array(m.jnt_pos.copy())

    # ── M-phase: differentiable FK loss ──────────────────────────────
    # Given fixed qpos (from Q-phase), compute site positions via MJX FK
    # and compare to keypoint targets. Differentiate w.r.t. scales + offsets.

    mx_base = mjx.put_model(m)

    def apply_scales_to_mx(mx, global_scale, rel_scales, site_offsets):
        """Apply segment scales and site offsets to MJX model."""
        seg_scales = global_scale * rel_scales  # [n_seg]
        body_scale = jnp.where(
            body_to_seg_j >= 0,
            seg_scales[jnp.clip(body_to_seg_j, 0, n_seg - 1)],
            1.0
        )
        new_body_pos = orig_body_pos * body_scale[:, None]
        new_body_ipos = orig_body_ipos * body_scale[:, None]
        geom_scale = body_scale[geom_bodyid_j]
        new_geom_pos = orig_geom_pos * geom_scale[:, None]
        new_geom_size = orig_geom_size * geom_scale[:, None]
        site_scale = body_scale[site_bodyid_j]
        new_site_pos = orig_site_pos * site_scale[:, None] + site_offsets
        jnt_scale = body_scale[jnt_bodyid_j]
        new_jnt_pos = orig_jnt_pos * jnt_scale[:, None]
        return mx.replace(
            body_pos=new_body_pos, body_ipos=new_body_ipos,
            geom_pos=new_geom_pos, geom_size=new_geom_size,
            site_pos=new_site_pos, jnt_pos=new_jnt_pos,
        )

    def fk_one_frame(mx_scaled, qpos):
        """Run FK for one frame, return site positions [nsite, 3]."""
        dx = mjx.make_data(mx_scaled)
        dx = dx.replace(qpos=qpos)
        dx = mjx.kinematics(mx_scaled, dx)
        dx = mjx.com_pos(mx_scaled, dx)
        return dx.site_xpos

    def m_phase_loss(params, all_qpos_j, kp3d, valid):
        """M-phase loss: FK residual + regularization. Differentiable w.r.t params."""
        mx_scaled = apply_scales_to_mx(
            mx_base, params['global_scale'], params['rel_scales'], params['site_offsets'])

        # Batch FK over all frames
        all_sites = jax.vmap(lambda q: fk_one_frame(mx_scaled, q))(all_qpos_j)  # [N, nsite, 3]

        # Extract our 24 keypoint sites
        kp_sites = all_sites[:, site_ids_j, :]  # [N, 24, 3]

        # Residual (weighted by validity)
        residual = (kp_sites - kp3d) * valid[:, :, None]  # [N, 24, 3]
        ik_loss = jnp.mean(jnp.sum(residual ** 2, axis=-1))  # mean over frames and sites

        # Regularization
        reg_s = args.reg_scale * jnp.sum((params['rel_scales'] - 1.0) ** 2)
        reg_o = args.reg_offset * jnp.sum(params['site_offsets'] ** 2)

        return ik_loss + reg_s + reg_o, {'ik_loss': ik_loss, 'reg_s': reg_s, 'reg_o': reg_o}

    # JIT compile M-phase
    print("\nCompiling M-phase...")
    t0 = time.time()

    optimizer = optax.adam(args.m_lr)
    params = {
        'global_scale': jnp.array(1.1),
        'rel_scales': jnp.ones(n_seg),
        'site_offsets': jnp.zeros((m.nsite, 3)),
    }
    opt_state = optimizer.init(params)

    @jax.jit
    def m_step(params, opt_state, all_qpos_j, kp3d, valid):
        (loss, metrics), grads = jax.value_and_grad(m_phase_loss, has_aux=True)(
            params, all_qpos_j, kp3d, valid)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params['global_scale'] = jnp.clip(new_params['global_scale'], 0.8, 1.5)
        new_params['rel_scales'] = jnp.clip(new_params['rel_scales'], 0.8, 1.2)
        return new_params, new_opt_state, loss, metrics

    # Warmup compile with dummy data
    dummy_qpos = jnp.zeros((N, nq))
    _, _, loss0, _ = m_step(params, opt_state, dummy_qpos, kp3d_all, valid_all)
    print(f"Compiled in {time.time() - t0:.1f}s")

    # ── Alternating optimization ─────────────────────────────────────
    def apply_scales_to_cpu_model(m, params):
        """Apply current scales to CPU model for Q-phase IK."""
        gs = float(params['global_scale'])
        rs = np.array(params['rel_scales'])
        offsets = np.array(params['site_offsets'])

        m.body_pos[:] = np.array(orig_body_pos)
        m.body_ipos[:] = np.array(orig_body_ipos)
        m.geom_pos[:] = np.array(orig_geom_pos)
        m.geom_size[:] = np.array(orig_geom_size)
        m.site_pos[:] = np.array(orig_site_pos)
        m.jnt_pos[:] = np.array(orig_jnt_pos)

        for g, (name, bids) in enumerate(segments):
            s = gs * rs[g]
            for bid in bids:
                m.body_pos[bid] *= s
                m.body_ipos[bid] *= s
                for gi in range(m.ngeom):
                    if m.geom_bodyid[gi] == bid:
                        m.geom_pos[gi] *= s
                        m.geom_size[gi] *= s
                for si in range(m.nsite):
                    if m.site_bodyid[si] == bid:
                        m.site_pos[si] *= s
                for ji in range(m.njnt):
                    if m.jnt_bodyid[ji] == bid:
                        m.jnt_pos[ji] *= s

        # Add site offsets
        m.site_pos[:] += offsets
        mujoco.mj_setConst(m, mujoco.MjData(m))

    print(f"\n{'='*60}")
    print(f"Alternating optimization: {args.n_rounds} rounds")
    print(f"  Q-phase: {N} frames × {args.ik_iters} IK iters (CPU)")
    print(f"  M-phase: {args.m_iters} Adam steps (GPU)")
    print(f"{'='*60}\n")

    for round_idx in range(args.n_rounds):
        t_round = time.time()

        # === Q-phase: IK on CPU ===
        print(f"Round {round_idx+1}/{args.n_rounds} Q-phase: solving IK...", end=" ", flush=True)
        t_q = time.time()
        apply_scales_to_cpu_model(m, params)
        all_qpos = q_phase_cpu(m, frames, site_ids, max_iters=args.ik_iters)
        all_qpos_j = jnp.array(all_qpos)
        print(f"{time.time()-t_q:.1f}s")

        # Compute pre-M residual
        _, pre_metrics = m_phase_loss(params, all_qpos_j, kp3d_all, valid_all)
        pre_ik_mm = float(jnp.sqrt(pre_metrics['ik_loss'] / 24) * 1000)

        # === M-phase: gradient descent on scales + offsets (GPU) ===
        print(f"Round {round_idx+1}/{args.n_rounds} M-phase: optimizing scales...", end=" ", flush=True)
        t_m = time.time()

        # Reset optimizer momentum for each round
        opt_state = optimizer.init(params)

        for mi in range(args.m_iters):
            params, opt_state, loss, metrics = m_step(
                params, opt_state, all_qpos_j, kp3d_all, valid_all)

        post_ik_mm = float(jnp.sqrt(metrics['ik_loss'] / 24) * 1000)
        print(f"{time.time()-t_m:.1f}s")

        dt = time.time() - t_round
        gs = float(params['global_scale'])
        print(f"  Residual: {pre_ik_mm:.2f} → {post_ik_mm:.2f} mm | "
              f"global={gs:.3f} | {dt:.1f}s total")

        # Print segment scales
        rs = np.array(params['rel_scales'])
        changed = [(SEGMENT_DEFS[g][0], gs * rs[g]) for g in range(n_seg) if abs(rs[g] - 1.0) > 0.01]
        if changed:
            print(f"  Scales: " + ", ".join(f"{n}={s:.3f}" for n, s in changed))

    # ── Final results ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    gs = float(params['global_scale'])
    print(f"\nGlobal scale: {gs:.4f}")
    print(f"\nSegment scales:")
    rs = np.array(params['rel_scales'])
    for g, (name, _) in enumerate(segments):
        abs_s = gs * rs[g]
        print(f"  {name:<12s} {abs_s:.3f} (rel={rs[g]:.3f})")

    offsets = np.array(params['site_offsets'])
    disps = np.linalg.norm(offsets, axis=1) * 1000
    print(f"\nSite offsets (top 10):")
    for i in np.argsort(disps)[::-1][:10]:
        if disps[i] > 0.1:
            print(f"  {m.site(i).name:<30s} {disps[i]:.1f} mm")

    # Save
    output = {
        'global_scale': gs,
        'segment_scales': {name: float(gs * rs[g]) for g, (name, _) in enumerate(segments)},
        'rel_scales': {name: float(rs[g]) for g, (name, _) in enumerate(segments)},
    }
    out_path = os.path.join(args.data_dir, 'mjx_body_resize.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
