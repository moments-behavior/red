#!/usr/bin/env python3
"""End-to-end differentiable body resize: gradients flow through IK.

Unlike v3 (alternating Q/M), this differentiates through a SHORT IK loop
so gradients propagate from site residuals → through joint angles → to body scales.

Key tricks to keep compilation fast:
  - Small batch (8 frames)
  - Short inner IK (10-20 steps, not 500)
  - jax.checkpoint on scan body to reduce memory
  - Outer loop runs many iterations to compensate for short inner IK

Usage (on cluster):
    python3 mjx_body_resize_e2e.py --data-dir /path/to/project
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
    spec = mujoco.MjSpec.from_file(xml_path)
    spec.body("torso").add_freejoint()
    m = spec.compile()
    for i in range(m.ngeom):
        if m.geom_type[i] == 8: m.geom_type[i] = 2
        if m.geom_type[i] == 6:
            m.geom_contype[i] = 0; m.geom_conaffinity[i] = 0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    return m


def load_keypoints(data_dir, max_frames=None):
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
            kp3d_csv = candidate; break
    assert kp3d_csv

    frames = []
    with open(kp3d_csv) as f:
        for line in f:
            if line.startswith('#') or line.startswith('frame,'): continue
            parts = line.strip().split(',')
            kp = np.zeros((24, 3), dtype=np.float32)
            valid = np.zeros(24, dtype=np.float32)
            for k in range(24):
                base = 1 + k * 4
                x, y, z = float(parts[base]), float(parts[base+1]), float(parts[base+2])
                if not math.isnan(x):
                    kp[k] = [x, y, z]; valid[k] = 1.0
            frames.append(((scale * (kp @ R.T) + t).astype(np.float32), valid))
            if max_frames and len(frames) >= max_frames: break
    print(f"Loaded {len(frames)} frames")
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-xml', default=None)
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--ik-steps', type=int, default=15,
                        help="Inner IK steps (keep small for fast compile)")
    parser.add_argument('--outer-iters', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--reg-scale', type=float, default=0.01)
    parser.add_argument('--reg-offset', type=float, default=1.0)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    if args.model_xml is None:
        args.model_xml = "/groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_data_driven_limits.xml"

    m = load_model(args.model_xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    nq = m.nq
    print(f"Model: nq={nq}, nbody={m.nbody}, nsite={m.nsite}")

    segments = []
    for name, body_names in SEGMENT_DEFS:
        bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn) for bn in body_names]
        bids = [b for b in bids if b >= 0]
        if bids: segments.append((name, bids))
    n_seg = len(segments)

    site_ids = []
    for name in RAT24_SITES:
        sid = -1
        for i in range(m.nsite):
            if m.site(i).name == name: sid = i; break
        site_ids.append(sid)

    frames = load_keypoints(args.data_dir, max_frames=args.max_frames)
    N = len(frames)
    kp3d_all = jnp.array(np.stack([f[0] for f in frames]))
    valid_all = jnp.array(np.stack([f[1] for f in frames]))

    # Precompute mappings
    body_to_seg = np.full(m.nbody, -1, dtype=np.int32)
    for g, (name, bids) in enumerate(segments):
        for bid in bids: body_to_seg[bid] = g
    body_to_seg_j = jnp.array(body_to_seg)
    geom_bodyid_j = jnp.array(m.geom_bodyid)
    site_bodyid_j = jnp.array(m.site_bodyid)
    jnt_bodyid_j = jnp.array(m.jnt_bodyid)
    site_ids_j = jnp.array(site_ids)

    orig_body_pos = jnp.array(m.body_pos.copy())
    orig_body_ipos = jnp.array(m.body_ipos.copy())
    orig_geom_pos = jnp.array(m.geom_pos.copy())
    orig_geom_size = jnp.array(m.geom_size.copy())
    orig_site_pos = jnp.array(m.site_pos.copy())
    orig_jnt_pos = jnp.array(m.jnt_pos.copy())

    mx_base = mjx.put_model(m)

    def apply_scales(mx, global_scale, rel_scales, site_offsets):
        seg_scales = global_scale * rel_scales
        body_scale = jnp.where(
            body_to_seg_j >= 0,
            seg_scales[jnp.clip(body_to_seg_j, 0, n_seg - 1)], 1.0)
        return mx.replace(
            body_pos=orig_body_pos * body_scale[:, None],
            body_ipos=orig_body_ipos * body_scale[:, None],
            geom_pos=orig_geom_pos * body_scale[geom_bodyid_j][:, None],
            geom_size=orig_geom_size * body_scale[geom_bodyid_j][:, None],
            site_pos=orig_site_pos * body_scale[site_bodyid_j][:, None] + site_offsets,
            jnt_pos=orig_jnt_pos * body_scale[jnt_bodyid_j][:, None],
        )

    def fk(mx, qpos):
        """FK: qpos → site positions."""
        dx = mjx.make_data(mx)
        dx = dx.replace(qpos=qpos)
        dx = mjx.kinematics(mx, dx)
        dx = mjx.com_pos(mx, dx)
        return dx.site_xpos

    def ik_and_loss(params, kp3d, valid):
        """Differentiable IK + loss for ONE frame.
        Short IK loop with gradient flow through body scales."""
        mx_scaled = apply_scales(mx_base, params['global_scale'],
                                  params['rel_scales'], params['site_offsets'])

        # Initialize qpos at centroid
        n_valid = jnp.maximum(valid.sum(), 1.0)
        centroid = (kp3d * valid[:, None]).sum(0) / n_valid
        qpos = jnp.zeros(nq).at[0].set(centroid[0]).at[1].set(centroid[1]).at[2].set(centroid[2]).at[3].set(1.0)

        # Short IK loop with checkpointed scan
        def ik_step(qpos, _):
            sites = fk(mx_scaled, qpos)
            kp_sites = sites[site_ids_j]
            residual = (kp_sites - kp3d) * valid[:, None]
            loss = jnp.sum(residual ** 2)
            g = jax.grad(lambda q: jnp.sum((fk(mx_scaled, q)[site_ids_j] - kp3d) ** 2 * valid[:, None]))(qpos)
            qpos = qpos - 0.005 * g
            return qpos, None

        # Use checkpoint to save memory
        ik_step_ckpt = jax.checkpoint(ik_step)
        qpos_final, _ = jax.lax.scan(ik_step_ckpt, qpos, None, length=args.ik_steps)

        # Final loss
        sites = fk(mx_scaled, qpos_final)
        kp_sites = sites[site_ids_j]
        residual = (kp_sites - kp3d) * valid[:, None]
        return jnp.sum(residual ** 2)

    def batch_loss(params, kp3d_batch, valid_batch):
        """Loss over a batch of frames + regularization."""
        frame_losses = jax.vmap(lambda kp, v: ik_and_loss(params, kp, v))(
            kp3d_batch, valid_batch)
        ik_loss = jnp.mean(frame_losses)
        reg_s = args.reg_scale * jnp.sum((params['rel_scales'] - 1.0) ** 2)
        reg_o = args.reg_offset * jnp.sum(params['site_offsets'] ** 2)
        return ik_loss + reg_s + reg_o, {'ik': ik_loss, 'rs': reg_s, 'ro': reg_o}

    # Initialize
    params = {
        'global_scale': jnp.array(1.1),
        'rel_scales': jnp.ones(n_seg),
        'site_offsets': jnp.zeros((m.nsite, 3)),
    }

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, kp3d_batch, valid_batch):
        (loss, metrics), grads = jax.value_and_grad(batch_loss, has_aux=True)(
            params, kp3d_batch, valid_batch)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params['global_scale'] = jnp.clip(new_params['global_scale'], 0.8, 1.5)
        new_params['rel_scales'] = jnp.clip(new_params['rel_scales'], 0.7, 1.3)
        return new_params, new_opt, loss, metrics

    # Compile
    print(f"\nCompiling (batch={args.batch_size}, ik_steps={args.ik_steps})...")
    t0 = time.time()
    warmup_kp = kp3d_all[:args.batch_size]
    warmup_v = valid_all[:args.batch_size]
    p_, o_, l_, m_ = train_step(params, opt_state, warmup_kp, warmup_v)
    print(f"Compiled in {time.time()-t0:.1f}s")
    print(f"Initial loss: {float(l_):.4f}")

    # Train
    print(f"\nTraining: {args.outer_iters} iters, batch={args.batch_size}")
    print(f"{'Iter':>5s} {'Loss':>10s} {'IK(mm)':>8s} {'Global':>7s} {'Scales':>40s} {'Time':>6s}")
    print("-" * 80)

    key = jax.random.PRNGKey(42)
    for i in range(args.outer_iters):
        t_i = time.time()
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, N, shape=(args.batch_size,), replace=False)
        params, opt_state, loss, metrics = train_step(
            params, opt_state, kp3d_all[idx], valid_all[idx])
        dt = time.time() - t_i

        if (i+1) % 50 == 0 or i == 0:
            ik_mm = float(jnp.sqrt(metrics['ik'] / 24) * 1000)
            gs = float(params['global_scale'])
            rs = np.array(params['rel_scales'])
            # Show segments that deviate > 1% from 1.0
            changed = [(SEGMENT_DEFS[g][0], rs[g]) for g in range(n_seg) if abs(rs[g]-1.0) > 0.01]
            scale_str = ", ".join(f"{n}={r:.3f}" for n, r in changed) if changed else "(all ~1.0)"
            print(f"{i+1:5d} {float(loss):10.4f} {ik_mm:8.2f} {gs:7.3f} {scale_str:>40s} {dt:5.2f}s")

    # Results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    gs = float(params['global_scale'])
    rs = np.array(params['rel_scales'])
    print(f"Global scale: {gs:.4f}")
    print(f"\nSegment scales:")
    for g, (name, _) in enumerate(segments):
        print(f"  {name:<12s} {gs*rs[g]:.3f} (rel={rs[g]:.3f})")

    offsets = np.array(params['site_offsets'])
    disps = np.linalg.norm(offsets, axis=1) * 1000
    print(f"\nSite offsets (top 5):")
    for i in np.argsort(disps)[::-1][:5]:
        if disps[i] > 0.1:
            print(f"  {m.site(i).name:<30s} {disps[i]:.1f} mm")

    out = {
        'global_scale': gs,
        'segment_scales': {name: float(gs*rs[g]) for g, (name, _) in enumerate(segments)},
        'rel_scales': {name: float(rs[g]) for g, (name, _) in enumerate(segments)},
        'method': 'mjx_e2e_differentiable',
        'ik_steps': args.ik_steps,
        'outer_iters': args.outer_iters,
    }
    out_path = os.path.join(args.data_dir, 'mjx_body_resize_e2e.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
