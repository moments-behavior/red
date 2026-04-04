#!/usr/bin/env python3
"""MJX Body Model Fitting: complete pipeline on GPU.

Phase 1: Optimize body segment scales (no site offsets)
Phase 2: Fine-tune site offsets on the resized body (STAC-like)
Output:  Fitted .mjb file ready to load in RED

Usage (on cluster):
    bsub -W 1:00 -n 12 -gpu "num=1" -q gpu_a100 -P johnson -Is /bin/bash
    source ~/miniconda3/bin/activate && conda activate mjx
    python3 mjx_fit_body_model.py --data-dir /path/to/project --output fitted.mjb
"""

import os, sys, json, math, time, argparse
import numpy as np
import jax, jax.numpy as jnp, optax
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

# Initialization from segment length analysis (model vs data ratios)
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

# L/R symmetry: site name pairs and midline sites (matching RED's STAC)
LR_SITE_PAIRS = [
    ("ear_L_1_kpsite",      "ear_R_2_kpsite"),
    ("shoulder_L_6_kpsite", "shoulder_R_10_kpsite"),
    ("elbow_L_7_kpsite",    "elbow_R_11_kpsite"),
    ("wrist_L_8_kpsite",    "wrist_R_12_kpsite"),
    ("hand_L_9_kpsite",     "hand_R_13_kpsite"),
    ("knee_L_14_kpsite",    "knee_R_17_kpsite"),
    ("ankle_L_15_kpsite",   "ankle_R_18_kpsite"),
    ("foot_L_16_kpsite",    "foot_R_19_kpsite"),
]
MIDLINE_SITES = [
    "nose_0_kpsite", "neck_3_kpsite", "spineL_4_kpsite", "tailbase_5_kpsite",
    "tailtip_20_kpsite", "tailmid_21_kpsite", "tail1Q_22_kpsite", "tail3Q_23_kpsite",
]


def build_symmetry_indices(m):
    """Build JAX-compatible symmetry index arrays for enforce_symmetry."""
    site_name_to_id = {}
    for i in range(m.nsite):
        site_name_to_id[m.site(i).name] = i

    midline_ids = [site_name_to_id[n] for n in MIDLINE_SITES if n in site_name_to_id]
    pair_L_ids = []
    pair_R_ids = []
    for ln, rn in LR_SITE_PAIRS:
        li = site_name_to_id.get(ln, -1)
        ri = site_name_to_id.get(rn, -1)
        if li >= 0 and ri >= 0:
            pair_L_ids.append(li)
            pair_R_ids.append(ri)

    return (jnp.array(midline_ids, dtype=jnp.int32),
            jnp.array(pair_L_ids, dtype=jnp.int32),
            jnp.array(pair_R_ids, dtype=jnp.int32))


def enforce_symmetry(offsets, midline_ids, pair_L_ids, pair_R_ids):
    """Enforce bilateral symmetry on site offsets (JAX-compatible).
    Midline: Y offset = 0. L/R pairs: average X/Z, mirror Y."""
    # Midline: zero Y
    offsets = offsets.at[midline_ids, 1].set(0.0)

    # L/R pairs: average X and Z, mirror Y
    avg_x = (offsets[pair_L_ids, 0] + offsets[pair_R_ids, 0]) * 0.5
    avg_z = (offsets[pair_L_ids, 2] + offsets[pair_R_ids, 2]) * 0.5
    avg_y = (offsets[pair_L_ids, 1] - offsets[pair_R_ids, 1]) * 0.5

    offsets = offsets.at[pair_L_ids, 0].set(avg_x)
    offsets = offsets.at[pair_L_ids, 1].set(avg_y)
    offsets = offsets.at[pair_L_ids, 2].set(avg_z)
    offsets = offsets.at[pair_R_ids, 0].set(avg_x)
    offsets = offsets.at[pair_R_ids, 1].set(-avg_y)
    offsets = offsets.at[pair_R_ids, 2].set(avg_z)

    return offsets


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
    assert kp3d_csv, f"No keypoints3d.csv in {labeled_dir}"
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
    d = mujoco.MjData(m); nq = m.nq; nv = m.nv
    all_qpos = np.zeros((len(frames), nq))
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
    parser.add_argument('--output', default=None, help="Output .mjb path")
    parser.add_argument('--max-frames', type=int, default=500)
    parser.add_argument('--ik-iters', type=int, default=1000)
    parser.add_argument('--n-rounds', type=int, default=6)
    parser.add_argument('--m-iters', type=int, default=300)
    args = parser.parse_args()

    if args.model_xml is None:
        args.model_xml = "/groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_data_driven_limits.xml"
    if args.output is None:
        args.output = os.path.join(args.data_dir, "rodent_mjx_fitted.mjb")

    print(f"JAX devices: {jax.devices()}")
    m = load_model(args.model_xml)
    nq = m.nq; nsite = m.nsite
    print(f"Model: nq={nq}, nbody={m.nbody}, nsite={nsite}")

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
        for i in range(nsite):
            if m.site(i).name == name: sid = i; break
        site_ids.append(sid)

    frames = load_keypoints(args.data_dir, max_frames=args.max_frames)
    N = len(frames)
    kp3d_all = jnp.array(np.stack([f[0] for f in frames]))
    valid_all = jnp.array(np.stack([f[1] for f in frames]))
    print(f"Frames: {N}")

    # Precompute JAX mappings
    body_to_seg = np.full(m.nbody, -1, dtype=np.int32)
    for g, (_, bids) in enumerate(segments):
        for bid in bids: body_to_seg[bid] = g
    bts_j = jnp.array(body_to_seg)
    gb_j = jnp.array(m.geom_bodyid)
    sb_j = jnp.array(m.site_bodyid)
    jb_j = jnp.array(m.jnt_bodyid)
    si_j = jnp.array(site_ids)

    orig = {k: jnp.array(getattr(m, k).copy()) for k in
            ['body_pos', 'body_ipos', 'geom_pos', 'geom_size', 'site_pos', 'jnt_pos']}

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
        dx = mjx.kinematics(mx, dx); dx = mjx.com_pos(mx, dx)
        return dx.site_xpos

    def apply_to_cpu(m, gs, rs, offsets_np):
        for k in orig: getattr(m, k)[:] = np.array(orig[k])
        for g, (_, bids) in enumerate(segments):
            s = float(gs * rs[g]); s3, s5 = s**3, s**5
            for bid in bids:
                m.body_pos[bid] *= s; m.body_ipos[bid] *= s
                m.body_mass[bid] *= s3; m.body_inertia[bid] *= s5
                for gi in range(m.ngeom):
                    if m.geom_bodyid[gi] == bid: m.geom_pos[gi] *= s; m.geom_size[gi] *= s
                for si in range(m.nsite):
                    if m.site_bodyid[si] == bid: m.site_pos[si] *= s
                for ji in range(m.njnt):
                    if m.jnt_bodyid[ji] == bid: m.jnt_pos[ji] *= s
        m.site_pos[:] += offsets_np
        mujoco.mj_setConst(m, mujoco.MjData(m))

    # Build symmetry indices for L/R enforcement
    sym_mid, sym_L, sym_R = build_symmetry_indices(m)
    print(f"Symmetry: {len(sym_mid)} midline, {len(sym_L)} L/R pairs")

    def run_phase(phase_name, params, freeze_offsets, lr_s, lr_o, reg_s, reg_o, n_rounds, m_iters):
        """Run alternating Q/M optimization."""
        if freeze_offsets:
            optimizer = optax.adam(lr_s)
        else:
            optimizer = optax.multi_transform(
                {'scales': optax.adam(lr_s), 'offsets': optax.adam(lr_o)},
                param_labels={'global_scale': 'scales', 'rel_scales': 'scales',
                              'site_offsets': 'offsets'})
        opt_state = optimizer.init(params)

        def m_loss(params, qpos_j, kp3d, valid):
            offs = params['site_offsets'] if not freeze_offsets else jnp.zeros((nsite, 3))
            mx_s = apply_scales(mx_base, params['global_scale'], params['rel_scales'], offs)
            all_sites = jax.vmap(lambda q: fk_sites(mx_s, q))(qpos_j)
            kp_sites = all_sites[:, si_j, :]
            res = (kp_sites - kp3d) * valid[:, :, None]
            ik_loss = jnp.mean(jnp.sum(res ** 2, axis=-1))
            r_s = reg_s * jnp.sum((params['rel_scales'] - 1.0) ** 2)
            r_o = reg_o * jnp.sum(offs ** 2) if not freeze_offsets else 0.0
            return ik_loss + r_s + r_o, {'ik': ik_loss}

        @jax.jit
        def m_step(params, opt_state, qpos_j, kp3d, valid):
            (loss, metrics), grads = jax.value_and_grad(m_loss, has_aux=True)(
                params, qpos_j, kp3d, valid)
            updates, new_opt = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            new_params['global_scale'] = jnp.clip(new_params['global_scale'], 0.7, 1.5)
            new_params['rel_scales'] = jnp.clip(new_params['rel_scales'], 0.5, 1.5)
            # Enforce bilateral symmetry on site offsets
            if not freeze_offsets:
                new_params['site_offsets'] = enforce_symmetry(
                    new_params['site_offsets'], sym_mid, sym_L, sym_R)
            return new_params, new_opt, loss, metrics

        # Compile
        print(f"\n{'='*60}")
        print(f"{phase_name}")
        print(f"{'='*60}")
        print("Compiling...", end=" ", flush=True)
        t0 = time.time()
        dummy = jnp.zeros((N, nq))
        m_step(params, opt_state, dummy, kp3d_all, valid_all)
        print(f"{time.time()-t0:.1f}s")

        for rnd in range(n_rounds):
            t_q = time.time()
            offs_np = np.array(params['site_offsets']) if not freeze_offsets else np.zeros((nsite, 3))
            apply_to_cpu(m, float(params['global_scale']), np.array(params['rel_scales']), offs_np)
            all_qpos_j = jnp.array(q_phase_cpu(m, frames, site_ids, max_iters=args.ik_iters))
            dt_q = time.time() - t_q

            t_m = time.time()
            opt_state = optimizer.init(params)
            for _ in range(m_iters):
                params, opt_state, loss, metrics = m_step(params, opt_state, all_qpos_j, kp3d_all, valid_all)
            dt_m = time.time() - t_m

            ik_mm = float(jnp.sqrt(metrics['ik'] / 24) * 1000)
            gs = float(params['global_scale'])
            rs = np.array(params['rel_scales'])
            changed = [(seg_names[g], rs[g]) for g in range(n_seg) if abs(rs[g]-1.0) > 0.01]
            ch_str = ", ".join(f"{n}={r:.3f}" for n, r in changed[:6]) if changed else "(~1.0)"
            print(f"  Round {rnd+1}/{n_rounds}: IK={ik_mm:.2f}mm Q={dt_q:.0f}s M={dt_m:.1f}s global={gs:.3f} {ch_str}")

        return params

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Scales only (no site offsets)
    # ══════════════════════════════════════════════════════════════════
    init_rs = np.array([SEGMENT_LENGTH_INIT.get(name, 1.0) for name, _ in segments], dtype=np.float32)
    params = {
        'global_scale': jnp.array(1.0),
        'rel_scales': jnp.array(init_rs),
        'site_offsets': jnp.zeros((nsite, 3)),
    }

    params = run_phase("PHASE 1: Body segment scaling (no site offsets)",
                        params, freeze_offsets=True,
                        lr_s=0.003, lr_o=0.0, reg_s=0.001, reg_o=0.0,
                        n_rounds=args.n_rounds, m_iters=args.m_iters)

    print(f"\nPhase 1 scales:")
    gs1 = float(params['global_scale']); rs1 = np.array(params['rel_scales'])
    for g, (name, _) in enumerate(segments):
        print(f"  {name:<12s} {gs1*rs1[g]:.3f} (rel={rs1[g]:.3f})")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Site offsets (STAC) on the resized body
    # Freeze scales, only optimize offsets
    # ══════════════════════════════════════════════════════════════════
    # Freeze scales by using high reg
    params = run_phase("PHASE 2: Site offset calibration (STAC) on resized body",
                        params, freeze_offsets=False,
                        lr_s=0.0001, lr_o=0.001, reg_s=10.0, reg_o=0.01,
                        n_rounds=args.n_rounds, m_iters=args.m_iters)

    # ══════════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    gs = float(params['global_scale']); rs = np.array(params['rel_scales'])
    offs = np.array(params['site_offsets'])
    disps = np.linalg.norm(offs, axis=1) * 1000

    print(f"\nFinal segment scales:")
    for g, (name, _) in enumerate(segments):
        print(f"  {name:<12s} {gs*rs[g]:.3f}")

    print(f"\nTop site offsets:")
    for i in np.argsort(disps)[::-1][:10]:
        if disps[i] > 0.1:
            print(f"  {m.site(i).name:<30s} {disps[i]:.1f} mm")

    # Apply to CPU model and save MJB
    apply_to_cpu(m, gs, rs, offs)
    mujoco.mj_saveModel(m, args.output)
    sz = os.path.getsize(args.output)
    print(f"\nSaved: {args.output} ({sz} bytes)")

    # Verify
    m2 = mujoco.MjModel.from_binary_path(args.output)
    print(f"Verified: nq={m2.nq}, nbody={m2.nbody}, nsite={m2.nsite}")

    # Also save JSON for reference
    json_path = args.output.replace('.mjb', '.json')
    result = {
        'global_scale': gs,
        'segment_scales': {name: float(gs*rs[g]) for g, (name, _) in enumerate(segments)},
        'rel_scales': {name: float(rs[g]) for g, (name, _) in enumerate(segments)},
        'site_offsets': {m.site(i).name: offs[i].tolist() for i in range(nsite) if disps[i] > 0.01},
        'model_xml': args.model_xml,
        'n_frames': N,
    }
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {json_path}")
    print("\nDone! Load the .mjb in RED → Body Model")


if __name__ == "__main__":
    main()
