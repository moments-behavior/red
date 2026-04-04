#!/usr/bin/env python3
"""Body resize experiments: test multiple hypotheses about why segments don't move.

Experiments:
  1. scales-only (no offsets) — are proportions actually wrong?
  2. gradient diagnostics — per-group gradient norms
  3. initialized from segment length analysis — does better init help?
  4. scales-only initialized from segment lengths — definitive test

Usage:
    python3 mjx_body_resize_experiments.py --data-dir /path/to/project --experiment 1
"""

import os, sys, json, math, time, argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
import mujoco
from mujoco import mjx

SEGMENT_DEFS = [
    ('head',      ['skull', 'jaw']),
    ('neck',      ['vertebra_cervical_5', 'vertebra_cervical_4', 'vertebra_cervical_3',
                   'vertebra_cervical_2', 'vertebra_cervical_1', 'vertebra_axis', 'vertebra_atlant']),
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

# From segment length analysis (model vs data ratios)
SEGMENT_LENGTH_INIT = {
    'head': 0.92, 'neck': 1.0, 'spine': 1.26, 'pelvis': 1.0,
    'tail': 1.0, 'scapula': 1.0, 'upper_arm': 0.69, 'lower_arm': 0.86,
    'hand': 0.9, 'upper_leg': 0.9, 'lower_leg': 1.19, 'foot': 1.0,
}

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
        if m.geom_type[i] == 6: m.geom_contype[i] = 0; m.geom_conaffinity[i] = 0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    return m


def load_keypoints(data_dir, max_frames=None):
    with open(os.path.join(data_dir, 'mujoco_session.json')) as f:
        sess = json.load(f)
    arena = sess['arena']
    R = np.array(arena['R']).reshape(3, 3); scale = arena['scale']
    t = np.array(arena.get('t', [0, 0, 0]))
    labeled_dir = os.path.join(data_dir, 'labeled_data')
    kp3d_csv = None
    for session in sorted(os.listdir(labeled_dir), reverse=True):
        c = os.path.join(labeled_dir, session, 'keypoints3d.csv')
        if os.path.exists(c): kp3d_csv = c; break
    assert kp3d_csv
    frames = []
    with open(kp3d_csv) as f:
        for line in f:
            if line.startswith('#') or line.startswith('frame,'): continue
            parts = line.strip().split(',')
            kp = np.zeros((24, 3), dtype=np.float32); valid = np.zeros(24, dtype=np.float32)
            for k in range(24):
                base = 1 + k * 4
                x, y, z = float(parts[base]), float(parts[base+1]), float(parts[base+2])
                if not math.isnan(x): kp[k] = [x, y, z]; valid[k] = 1.0
            frames.append(((scale * (kp @ R.T) + t).astype(np.float32), valid))
            if max_frames and len(frames) >= max_frames: break
    return frames


def q_phase_cpu(m, frames, site_ids, max_iters=1000):
    """IK solve on CPU. Returns [N, nq]."""
    d = mujoco.MjData(m); nq = m.nq; nv = m.nv; N = len(frames)
    all_qpos = np.zeros((N, nq))
    jacp = np.zeros((3, nv))
    for i, (kp_mj, valid) in enumerate(frames):
        mujoco.mj_resetData(m, d)
        vk = kp_mj[valid > 0.5]
        if len(vk) == 0: continue
        c = vk.mean(0)
        for j in range(m.njnt):
            if m.jnt_type[j] == 0:
                qa = int(m.jnt_qposadr[j])
                d.qpos[qa:qa+3] = c; d.qpos[qa+3] = 1.0; break
        mujoco.mj_forward(m, d)
        update = np.zeros(nv)
        targets = [(site_ids[k], kp_mj[k]) for k in range(24) if valid[k] > 0.5 and site_ids[k] >= 0]
        for _ in range(max_iters):
            g = np.zeros(nv)
            for sid, tgt in targets:
                jacp[:] = 0; mujoco.mj_jacSite(m, d, jacp, None, sid)
                g += 2.0 * jacp.T @ (d.site_xpos[sid] - tgt)
            update = 0.99 * update + g
            mujoco.mj_integratePos(m, d.qpos, -0.001 * update, 1.0)
            mujoco.mj_fwdPosition(m, d)
        all_qpos[i] = d.qpos.copy()
    return all_qpos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-xml', default=None)
    parser.add_argument('--experiment', type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--ik-iters', type=int, default=1000)
    parser.add_argument('--n-rounds', type=int, default=8)
    parser.add_argument('--m-iters', type=int, default=300)
    args = parser.parse_args()

    print(f"=== Experiment {args.experiment} ===")
    print(f"JAX devices: {jax.devices()}")

    if args.model_xml is None:
        args.model_xml = "/groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_data_driven_limits.xml"

    m = load_model(args.model_xml)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d); nq = m.nq
    print(f"Model: nq={nq}, nbody={m.nbody}, nsite={m.nsite}")

    segments = []
    for name, bnames in SEGMENT_DEFS:
        bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bnames]
        bids = [b for b in bids if b >= 0]
        if bids: segments.append((name, bids))
    n_seg = len(segments)
    seg_names = [s[0] for s in segments]

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
    for g, (_, bids) in enumerate(segments):
        for bid in bids: body_to_seg[bid] = g
    bts_j = jnp.array(body_to_seg)
    gb_j = jnp.array(m.geom_bodyid)
    sb_j = jnp.array(m.site_bodyid)
    jb_j = jnp.array(m.jnt_bodyid)
    si_j = jnp.array(site_ids)

    orig = {
        'body_pos': jnp.array(m.body_pos.copy()),
        'body_ipos': jnp.array(m.body_ipos.copy()),
        'geom_pos': jnp.array(m.geom_pos.copy()),
        'geom_size': jnp.array(m.geom_size.copy()),
        'site_pos': jnp.array(m.site_pos.copy()),
        'jnt_pos': jnp.array(m.jnt_pos.copy()),
    }

    mx_base = mjx.put_model(m)

    def apply_scales(mx, gs, rs, offsets):
        ss = gs * rs
        bs = jnp.where(bts_j >= 0, ss[jnp.clip(bts_j, 0, n_seg-1)], 1.0)
        return mx.replace(
            body_pos=orig['body_pos'] * bs[:, None],
            body_ipos=orig['body_ipos'] * bs[:, None],
            geom_pos=orig['geom_pos'] * bs[gb_j][:, None],
            geom_size=orig['geom_size'] * bs[gb_j][:, None],
            site_pos=orig['site_pos'] * bs[sb_j][:, None] + offsets,
            jnt_pos=orig['jnt_pos'] * bs[jb_j][:, None],
        )

    def fk_sites(mx, qpos):
        dx = mjx.make_data(mx).replace(qpos=qpos)
        dx = mjx.kinematics(mx, dx)
        dx = mjx.com_pos(mx, dx)
        return dx.site_xpos

    # ── Experiment-specific setup ──────────────────────────────────
    if args.experiment == 1:
        # Scales only, no offsets
        desc = "Scales-only (no site offsets)"
        reg_scale, reg_offset = 0.01, 0.0
        lr_scale, lr_offset = 0.003, 0.0
        init_gs = 1.1
        init_rs = np.ones(n_seg)
        freeze_offsets = True

    elif args.experiment == 2:
        # Per-group optimizers + gradient diagnostics
        desc = "Per-group LR + gradient diagnostics"
        reg_scale, reg_offset = 0.01, 0.1
        lr_scale, lr_offset = 0.01, 0.0003
        init_gs = 1.1
        init_rs = np.ones(n_seg)
        freeze_offsets = False

    elif args.experiment == 3:
        # Initialized from segment lengths, with offsets
        desc = "Segment-length initialization + offsets"
        reg_scale, reg_offset = 0.01, 0.1
        lr_scale, lr_offset = 0.003, 0.001
        init_gs = 1.0  # segment lengths already account for global
        init_rs = np.array([SEGMENT_LENGTH_INIT.get(name, 1.0) for name, _ in segments])
        freeze_offsets = False

    elif args.experiment == 4:
        # Initialized from segment lengths, NO offsets
        desc = "Segment-length initialization, scales-only"
        reg_scale, reg_offset = 0.001, 0.0
        lr_scale, lr_offset = 0.003, 0.0
        init_gs = 1.0
        init_rs = np.array([SEGMENT_LENGTH_INIT.get(name, 1.0) for name, _ in segments])
        freeze_offsets = True

    print(f"Config: {desc}")
    print(f"  init_global={init_gs}, freeze_offsets={freeze_offsets}")
    print(f"  lr_scale={lr_scale}, lr_offset={lr_offset}")
    print(f"  reg_scale={reg_scale}, reg_offset={reg_offset}")
    if not np.allclose(init_rs, 1.0):
        print(f"  init_rel_scales: " + ", ".join(f"{seg_names[g]}={init_rs[g]:.2f}" for g in range(n_seg) if abs(init_rs[g]-1.0) > 0.01))

    params = {
        'global_scale': jnp.array(float(init_gs)),
        'rel_scales': jnp.array(init_rs.astype(np.float32)),
        'site_offsets': jnp.zeros((m.nsite, 3)),
    }

    # Optimizer: separate LR for scales vs offsets
    if freeze_offsets:
        optimizer = optax.adam(lr_scale)
    else:
        optimizer = optax.multi_transform(
            {'scales': optax.adam(lr_scale), 'offsets': optax.adam(lr_offset)},
            param_labels={'global_scale': 'scales', 'rel_scales': 'scales',
                          'site_offsets': 'offsets'}
        )
    opt_state = optimizer.init(params)

    def m_loss(params, qpos_j, kp3d, valid):
        offsets = params['site_offsets'] if not freeze_offsets else jnp.zeros_like(params['site_offsets'])
        mx_s = apply_scales(mx_base, params['global_scale'], params['rel_scales'], offsets)
        all_sites = jax.vmap(lambda q: fk_sites(mx_s, q))(qpos_j)
        kp_sites = all_sites[:, si_j, :]
        res = (kp_sites - kp3d) * valid[:, :, None]
        ik_loss = jnp.mean(jnp.sum(res ** 2, axis=-1))
        reg_s = reg_scale * jnp.sum((params['rel_scales'] - 1.0) ** 2)
        reg_o = reg_offset * jnp.sum(offsets ** 2) if not freeze_offsets else 0.0
        return ik_loss + reg_s + reg_o, {'ik': ik_loss, 'rs': reg_s}

    @jax.jit
    def m_step(params, opt_state, qpos_j, kp3d, valid):
        (loss, metrics), grads = jax.value_and_grad(m_loss, has_aux=True)(
            params, qpos_j, kp3d, valid)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params['global_scale'] = jnp.clip(new_params['global_scale'], 0.7, 1.5)
        new_params['rel_scales'] = jnp.clip(new_params['rel_scales'], 0.5, 1.5)
        # Gradient norms for diagnostics
        g_gs = jnp.linalg.norm(grads['global_scale'])
        g_rs = jnp.linalg.norm(grads['rel_scales'])
        g_so = jnp.linalg.norm(grads['site_offsets'])
        return new_params, new_opt, loss, metrics, g_gs, g_rs, g_so

    # Compile
    print("\nCompiling...", end=" ", flush=True)
    t0 = time.time()
    dummy_qpos = jnp.zeros((N, nq))
    _, _, _, _, _, _, _ = m_step(params, opt_state, dummy_qpos, kp3d_all, valid_all)
    print(f"{time.time()-t0:.1f}s")

    # Apply scales to CPU model helper
    def apply_to_cpu(m, params):
        gs = float(params['global_scale']); rs = np.array(params['rel_scales'])
        offs = np.array(params['site_offsets']) if not freeze_offsets else np.zeros_like(np.array(params['site_offsets']))
        for k, v in orig.items(): getattr(m, k)[:] = np.array(v)
        for g, (_, bids) in enumerate(segments):
            s = gs * rs[g]
            for bid in bids:
                m.body_pos[bid] *= s; m.body_ipos[bid] *= s
                for gi in range(m.ngeom):
                    if m.geom_bodyid[gi] == bid: m.geom_pos[gi] *= s; m.geom_size[gi] *= s
                for si in range(m.nsite):
                    if m.site_bodyid[si] == bid: m.site_pos[si] *= s
                for ji in range(m.njnt):
                    if m.jnt_bodyid[ji] == bid: m.jnt_pos[ji] *= s
        m.site_pos[:] += offs
        mujoco.mj_setConst(m, mujoco.MjData(m))

    # ── Run alternating optimization ───────────────────────────────
    print(f"\n{'Round':>5s} {'Q(s)':>5s} {'M(s)':>5s} {'IK(mm)':>8s} {'Global':>7s} "
          f"{'|∇gs|':>8s} {'|∇rs|':>8s} {'|∇off|':>8s} {'Changed segments':>30s}")
    print("-" * 100)

    for rnd in range(args.n_rounds):
        # Q-phase
        t_q = time.time()
        apply_to_cpu(m, params)
        all_qpos = q_phase_cpu(m, frames, site_ids, max_iters=args.ik_iters)
        all_qpos_j = jnp.array(all_qpos)
        dt_q = time.time() - t_q

        # M-phase
        t_m = time.time()
        opt_state = optimizer.init(params)  # reset momentum each round
        last_g_gs = last_g_rs = last_g_so = 0.0
        for mi in range(args.m_iters):
            params, opt_state, loss, metrics, g_gs, g_rs, g_so = m_step(
                params, opt_state, all_qpos_j, kp3d_all, valid_all)
            if mi == 0:
                last_g_gs = float(g_gs); last_g_rs = float(g_rs); last_g_so = float(g_so)
        dt_m = time.time() - t_m

        ik_mm = float(jnp.sqrt(metrics['ik'] / 24) * 1000)
        gs = float(params['global_scale'])
        rs = np.array(params['rel_scales'])
        changed = [(seg_names[g], rs[g]) for g in range(n_seg) if abs(rs[g] - 1.0) > 0.01]
        ch_str = ", ".join(f"{n}={r:.3f}" for n, r in changed) if changed else "(all ~1.0)"

        print(f"{rnd+1:5d} {dt_q:5.1f} {dt_m:5.1f} {ik_mm:8.2f} {gs:7.3f} "
              f"{last_g_gs:8.4f} {last_g_rs:8.4f} {last_g_so:8.4f} {ch_str:>30s}")

    # Results
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {args.experiment}: {desc}")
    print(f"{'='*60}")
    gs = float(params['global_scale'])
    rs = np.array(params['rel_scales'])
    print(f"Global: {gs:.4f}")
    for g, (name, _) in enumerate(segments):
        print(f"  {name:<12s} {gs*rs[g]:.3f} (rel={rs[g]:.3f})")

    if not freeze_offsets:
        offs = np.array(params['site_offsets'])
        disps = np.linalg.norm(offs, axis=1) * 1000
        print(f"\nTop offsets:")
        for i in np.argsort(disps)[::-1][:5]:
            if disps[i] > 0.1: print(f"  {m.site(i).name:<30s} {disps[i]:.1f} mm")

    out = {
        'experiment': args.experiment, 'description': desc,
        'global_scale': gs,
        'segment_scales': {name: float(gs*rs[g]) for g, (name, _) in enumerate(segments)},
        'rel_scales': {name: float(rs[g]) for g, (name, _) in enumerate(segments)},
    }
    out_path = os.path.join(args.data_dir, f'mjx_experiment_{args.experiment}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
