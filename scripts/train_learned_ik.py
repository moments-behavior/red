#!/usr/bin/env python3
"""Train a neural network to replace iterative IK with a single forward pass.

Maps 3D keypoints (24 × 3) → MuJoCo qpos (68 dims for rodent with free joint).
Validates via MuJoCo forward kinematics: predicted qpos → site positions → compare to GT.

The arena alignment (from mujoco_session.json) transforms keypoints from calibration
frame (mm) to MuJoCo frame (meters): p_mj = scale * R @ p_calib + t

Usage:
    # 1. Export qpos from RED: Body Model → Solve All & Export
    # 2. Train:
    python3 scripts/train_learned_ik.py --project-dir /path/to/project

Dependencies:
    pip install torch mujoco coremltools onnx
"""

import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RED_ROOT   = os.path.dirname(SCRIPT_DIR)
MODEL_XML  = "/Users/johnsonr/rat_modeling/IK_resources/rodent_no_collision.xml"
OUTPUT_DIR = os.path.join(RED_ROOT, "models", "learned_ik")

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


# ═══════════════════════════════════════════════════════════════════════════
# Arena transform (calibration mm → MuJoCo meters)
# ═══════════════════════════════════════════════════════════════════════════

class ArenaTransform:
    """Rigid transform from calibration frame (mm) to MuJoCo frame (meters).
    p_mj = scale * R @ p_calib + t
    """
    def __init__(self, R=None, t=None, scale=0.001):
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros(3)
        self.scale = scale

    def __call__(self, p_mm):
        """Transform points: [N, 3] or [3] in mm → meters."""
        return self.scale * (p_mm @ self.R.T) + self.t

    @classmethod
    def from_session(cls, session_path):
        """Load from mujoco_session.json."""
        with open(session_path) as f:
            session = json.load(f)
        arena = session.get('arena', {})
        if not arena.get('valid', False):
            print("  WARNING: arena alignment not valid, using identity + scale=0.001")
            return cls()
        R_flat = arena.get('R', [1,0,0, 0,1,0, 0,0,1])
        R = np.array(R_flat, dtype=np.float64).reshape(3, 3)
        t = np.array(arena.get('t', [0,0,0]), dtype=np.float64)
        scale = arena.get('scale', 0.001)
        return cls(R=R, t=t, scale=scale)

    def to_torch(self, device='cpu'):
        """Return R, t, scale as torch tensors for batched transform."""
        return (torch.tensor(self.R, dtype=torch.float32, device=device),
                torch.tensor(self.t, dtype=torch.float32, device=device),
                self.scale)


def load_stac_offsets(session_path):
    """Load STAC site offsets from mujoco_session.json."""
    with open(session_path) as f:
        session = json.load(f)
    history = session.get('calibration_history', [])
    if len(history) < 2:
        return {}, []
    # Last entry is the STAC calibration
    stac = history[-1]
    names = stac.get('site_names', [])
    offsets = stac.get('site_offsets', [])
    result = {}
    for i, name in enumerate(names):
        dx = offsets[i*3] if i*3 < len(offsets) else 0
        dy = offsets[i*3+1] if i*3+1 < len(offsets) else 0
        dz = offsets[i*3+2] if i*3+2 < len(offsets) else 0
        if abs(dx) > 1e-10 or abs(dy) > 1e-10 or abs(dz) > 1e-10:
            result[name] = (dx, dy, dz)
    return result, names


# ═══════════════════════════════════════════════════════════════════════════
# Rotation representation utilities (Zhou et al., CVPR 2019)
# ═══════════════════════════════════════════════════════════════════════════

def rotation_6d_to_matrix(r6d):
    """[B, 6] → [B, 3, 3]"""
    a1 = F.normalize(r6d[:, :3], dim=-1)
    a2 = r6d[:, 3:6]
    a2 = a2 - (a1 * a2).sum(-1, keepdim=True) * a1
    a2 = F.normalize(a2, dim=-1)
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack([a1, a2, a3], dim=-1)


def matrix_to_quaternion(R):
    """[B, 3, 3] → [B, 4] (w, x, y, z).

    Branchless implementation using torch.where — fully traceable for
    CoreML/ONNX export (no dynamic reshape, no masked indexing).
    """
    R00, R01, R02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    R10, R11, R12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    R20, R21, R22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    trace = R00 + R11 + R22

    # Case 1: trace > 0
    s1 = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2.0
    q1 = torch.stack([0.25 * s1,
                       (R21 - R12) / s1,
                       (R02 - R20) / s1,
                       (R10 - R01) / s1], dim=-1)

    # Case 2: R00 largest diagonal
    s2 = torch.sqrt(torch.clamp(1.0 + R00 - R11 - R22, min=1e-10)) * 2.0
    q2 = torch.stack([(R21 - R12) / s2,
                       0.25 * s2,
                       (R01 + R10) / s2,
                       (R02 + R20) / s2], dim=-1)

    # Case 3: R11 largest diagonal
    s3 = torch.sqrt(torch.clamp(1.0 + R11 - R00 - R22, min=1e-10)) * 2.0
    q3 = torch.stack([(R02 - R20) / s3,
                       (R01 + R10) / s3,
                       0.25 * s3,
                       (R12 + R21) / s3], dim=-1)

    # Case 4: R22 largest diagonal
    s4 = torch.sqrt(torch.clamp(1.0 + R22 - R00 - R11, min=1e-10)) * 2.0
    q4 = torch.stack([(R10 - R01) / s4,
                       (R02 + R20) / s4,
                       (R12 + R21) / s4,
                       0.25 * s4], dim=-1)

    # Select branchlessly
    quat = torch.where((trace > 0).unsqueeze(-1), q1,
           torch.where(((R00 > R11) & (R00 > R22)).unsqueeze(-1), q2,
           torch.where((R11 > R22).unsqueeze(-1), q3, q4)))

    return F.normalize(quat, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Model architecture
# ═══════════════════════════════════════════════════════════════════════════

class LearnedIK(nn.Module):
    """Predict MuJoCo qpos from 3D keypoint positions.

    Position-invariant design:
      1. Compute centroid of valid keypoints
      2. Center-subtract keypoints (network sees relative geometry only)
      3. Network predicts: root offset (3) + rotation (6D) + hinge angles (61)
      4. Root position = centroid + predicted offset

    This generalizes across arena positions — a rat in the same pose at
    different locations produces the same joint angles.
    """

    def __init__(self, n_keypoints=24, n_hinge=61, hidden=256, n_layers=4,
                 dropout=0.0):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.n_hinge = n_hinge
        # Input: centered keypoints (24×3=72) + validity mask (24) = 96
        input_dim = n_keypoints * 3 + n_keypoints

        layers = []
        in_dim = input_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        # Output: 3 (root offset from centroid) + 6 (rot6d) + n_hinge (angles)
        layers.append(nn.Linear(in_dim, 3 + 6 + n_hinge))
        self.net = nn.Sequential(*layers)

    def forward(self, kp3d, valid_mask):
        """
        Args:
            kp3d: [B, 24, 3] — keypoints in MuJoCo frame (meters)
            valid_mask: [B, 24] — 1.0 if valid
        Returns:
            qpos: [B, 68] — root position (absolute) + quaternion + hinge angles
        """
        # Compute centroid of valid keypoints
        kp3d_masked = kp3d * valid_mask.unsqueeze(-1)
        n_valid = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
        centroid = kp3d_masked.sum(dim=1) / n_valid  # [B, 3]

        # Center-subtract: network sees relative geometry only
        kp_centered = (kp3d - centroid.unsqueeze(1)) * valid_mask.unsqueeze(-1)

        x = torch.cat([kp_centered.reshape(-1, self.n_keypoints * 3),
                        valid_mask], dim=-1)
        out = self.net(x)

        # Decompose output
        root_offset = out[:, :3]     # offset from centroid
        rot6d = out[:, 3:9]          # 6D rotation
        hinge = out[:, 9:]           # hinge joint angles

        # Root position = centroid + learned offset
        root_pos = centroid + root_offset

        # Convert rotation
        rot_mat = rotation_6d_to_matrix(rot6d)
        quat = matrix_to_quaternion(rot_mat)

        return torch.cat([root_pos, quat, hinge], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# MuJoCo FK wrapper
# ═══════════════════════════════════════════════════════════════════════════

class MuJoCoFK:
    """Batch forward kinematics using MuJoCo."""

    def __init__(self, model_xml, site_names, stac_offsets=None):
        import mujoco
        spec = mujoco.MjSpec.from_file(model_xml)
        torso = spec.body("torso")
        if torso is not None:
            has_free = False
            j = torso.first_joint()
            while j is not None:
                if j.type == mujoco.mjtJoint.mjJNT_FREE:
                    has_free = True
                    break
                j = j.next(torso)
            if not has_free:
                torso.add_freejoint()
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq

        # Apply STAC offsets
        if stac_offsets:
            for i in range(self.model.nsite):
                name = self.model.site(i).name
                if name in stac_offsets:
                    dx, dy, dz = stac_offsets[name]
                    self.model.site_pos[i, 0] += dx
                    self.model.site_pos[i, 1] += dy
                    self.model.site_pos[i, 2] += dz

        self.site_ids = []
        for name in site_names:
            found = -1
            for i in range(self.model.nsite):
                if self.model.site(i).name == name:
                    found = i
                    break
            self.site_ids.append(found)

    def forward(self, qpos_batch):
        """[B, nq] → [B, N_sites, 3] in MuJoCo meters."""
        import mujoco
        B = qpos_batch.shape[0]
        N = len(self.site_ids)
        sites = np.zeros((B, N, 3))
        for b in range(B):
            self.data.qpos[:] = qpos_batch[b]
            mujoco.mj_fwdPosition(self.model, self.data)
            for i, sid in enumerate(self.site_ids):
                if sid >= 0:
                    sites[b, i] = self.data.site_xpos[sid]
        return sites


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

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
            valid = np.zeros(24, dtype=np.float32)
            for kp in range(24):
                base = 1 + kp * 4
                if base + 2 < len(parts):
                    try:
                        x, y, z = float(parts[base]), float(parts[base+1]), float(parts[base+2])
                        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                            kp3d[kp] = [x, y, z]
                            valid[kp] = 1.0
                    except (ValueError, IndexError):
                        pass
            frames[frame_id] = {'kp3d': kp3d.astype(np.float32), 'valid': valid}
    return frames


def load_qpos(csv_path, max_residual_mm=None, require_converged=False):
    """Load RED qpos export CSV."""
    frames = {}
    nq = None
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# nq:'):
                nq = int(line.split(':')[1].strip())
                continue
            if not line or line.startswith('#'):
                continue
            if line.startswith('frame,'):
                cols = line.split(',')
                qpos_cols = [c for c in cols if c.startswith('qpos_')]
                if qpos_cols:
                    nq = len(qpos_cols)
                continue
            parts = line.split(',')
            frame_id = int(parts[0])
            if nq is None:
                nq = len(parts) - 4
            qpos = np.array([float(x) for x in parts[1:1+nq]], dtype=np.float32)
            residual = float(parts[1+nq])
            iterations = int(parts[2+nq])
            converged = bool(int(parts[3+nq]))
            if require_converged and not converged:
                continue
            if max_residual_mm is not None and residual > max_residual_mm:
                continue
            frames[frame_id] = {
                'qpos': qpos, 'residual_mm': residual, 'converged': converged,
            }
    print(f"Loaded {len(frames)} qpos frames (nq={nq})")
    return frames, nq


class IKDataset(Dataset):
    """Dataset pairing arena-transformed 3D keypoints with IK-solved qpos."""

    def __init__(self, kp3d_data, qpos_data, frame_ids, arena_tf):
        self.frame_ids = frame_ids
        self.arena_R = arena_tf.R.astype(np.float32)
        self.arena_t = arena_tf.t.astype(np.float32)
        self.arena_scale = arena_tf.scale

        N = len(frame_ids)
        nq = qpos_data[frame_ids[0]]['qpos'].shape[0]
        self.kp3d_mm = np.zeros((N, 24, 3), dtype=np.float32)  # raw mm
        self.valid = np.zeros((N, 24), dtype=np.float32)
        self.qpos = np.zeros((N, nq), dtype=np.float32)

        for i, fid in enumerate(frame_ids):
            kp = kp3d_data[fid]
            self.kp3d_mm[i] = np.nan_to_num(kp['kp3d'], nan=0.0)
            self.valid[i] = kp['valid']
            self.qpos[i] = qpos_data[fid]['qpos']

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        kp_mm = self.kp3d_mm[idx].copy()
        valid = self.valid[idx]
        qpos = self.qpos[idx]

        # Apply arena transform: p_mj = scale * R @ p_mm + t
        kp_mj = self.arena_scale * (kp_mm @ self.arena_R.T) + self.arena_t
        kp_mj = kp_mj.astype(np.float32)

        return (torch.from_numpy(kp_mj),
                torch.from_numpy(valid),
                torch.from_numpy(qpos))


# ═══════════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════════

def qpos_loss(pred_qpos, gt_qpos):
    """Direct qpos supervision with representation-aware losses."""
    loss_trans = F.mse_loss(pred_qpos[:, :3], gt_qpos[:, :3])
    q_pred = F.normalize(pred_qpos[:, 3:7], dim=-1)
    q_gt = F.normalize(gt_qpos[:, 3:7], dim=-1)
    dot = (q_pred * q_gt).sum(dim=-1).abs()
    loss_quat = (1.0 - dot ** 2).mean()
    loss_hinge = F.mse_loss(pred_qpos[:, 7:], gt_qpos[:, 7:])
    return loss_trans, loss_quat, loss_hinge


def fk_loss(pred_qpos, gt_kp_mj, valid_mask, fk_engine):
    """FK loss: compare FK(predicted qpos) to GT keypoints in MuJoCo frame.
    gt_kp_mj: [B, 24, 3] already in MuJoCo meters (arena-transformed).
    """
    pred_np = pred_qpos.detach().cpu().numpy()
    sites = fk_engine.forward(pred_np)  # [B, 24, 3] in MuJoCo meters
    sites_t = torch.from_numpy(sites).float().to(pred_qpos.device)

    diff = (sites_t - gt_kp_mj) * valid_mask.unsqueeze(-1)
    loss = (diff ** 2).sum(-1).mean()

    with torch.no_grad():
        per_site_err = torch.sqrt((diff ** 2).sum(-1))
        valid_err = per_site_err[valid_mask > 0]
        mean_err_mm = valid_err.mean().item() * 1000 if len(valid_err) > 0 else 0.0
    return loss, mean_err_mm


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, fk_engine, fk_weight=0.5,
                    device='cpu'):
    model.train()
    totals = {'loss': 0, 'trans': 0, 'quat': 0, 'hinge': 0, 'fk': 0}
    n = 0
    for kp3d, valid, gt_qpos in loader:
        kp3d, valid, gt_qpos = kp3d.to(device), valid.to(device), gt_qpos.to(device)
        pred_qpos = model(kp3d, valid)
        l_trans, l_quat, l_hinge = qpos_loss(pred_qpos, gt_qpos)
        l_direct = l_trans + l_quat + l_hinge
        l_fk = torch.tensor(0.0)
        if fk_engine is not None and fk_weight > 0:
            l_fk, _ = fk_loss(pred_qpos, kp3d, valid, fk_engine)
        loss = l_direct + fk_weight * l_fk
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        totals['loss'] += loss.item()
        totals['trans'] += l_trans.item()
        totals['quat'] += l_quat.item()
        totals['hinge'] += l_hinge.item()
        totals['fk'] += l_fk.item()
        n += 1
    return {k: v/n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, fk_engine, fk_weight=0.5, device='cpu'):
    model.eval()
    total_loss = 0.0
    total_fk_err_mm = 0.0
    all_angle_errors = []
    n = 0
    for kp3d, valid, gt_qpos in loader:
        kp3d, valid, gt_qpos = kp3d.to(device), valid.to(device), gt_qpos.to(device)
        pred_qpos = model(kp3d, valid)
        l_trans, l_quat, l_hinge = qpos_loss(pred_qpos, gt_qpos)
        l_fk, fk_err_mm = (torch.tensor(0.0), 0.0)
        if fk_engine is not None:
            l_fk, fk_err_mm = fk_loss(pred_qpos, kp3d, valid, fk_engine)
        total_loss += (l_trans + l_quat + l_hinge + fk_weight * l_fk).item()
        total_fk_err_mm += fk_err_mm
        all_angle_errors.append((pred_qpos[:, 7:] - gt_qpos[:, 7:]).abs().cpu())
        n += 1
    angle_errors = torch.cat(all_angle_errors, dim=0)
    return {
        'loss': total_loss / n,
        'fk_err_mm': total_fk_err_mm / n,
        'mean_angle_err_deg': float(angle_errors.mean()) * 180 / math.pi,
        'max_angle_err_deg': float(angle_errors.max()) * 180 / math.pi,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

def export_onnx(model, output_path):
    model.eval()
    kp3d = torch.randn(1, 24, 3)
    valid = torch.ones(1, 24)
    try:
        torch.onnx.export(model, (kp3d, valid), output_path,
                          input_names=['kp3d', 'valid_mask'],
                          output_names=['qpos'],
                          dynamic_axes={'kp3d': {0: 'batch'},
                                        'valid_mask': {0: 'batch'},
                                        'qpos': {0: 'batch'}},
                          opset_version=18)
        print(f"Exported ONNX: {output_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("  Install onnxscript: pip install onnxscript")


def export_coreml(model, output_path):
    try:
        import coremltools as ct
    except ImportError:
        print("coremltools not installed — skipping CoreML export")
        return
    model.eval()
    traced = torch.jit.trace(model, (torch.randn(1,24,3), torch.ones(1,24)))
    mlmodel = ct.convert(traced,
                         inputs=[ct.TensorType(name="kp3d", shape=(1,24,3)),
                                 ct.TensorType(name="valid_mask", shape=(1,24))],
                         outputs=[ct.TensorType(name="qpos")],
                         minimum_deployment_target=ct.target.macOS13)
    mlmodel.save(output_path)
    print(f"Exported CoreML: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Learned IK model")
    parser.add_argument('--project-dir', type=str,
                        default="/Users/johnsonr/datasets/rat/tiny_project",
                        help="RED project directory with qpos_export.csv")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--fk-weight', type=float, default=0.5)
    parser.add_argument('--max-residual', type=float, default=10.0,
                        help="Max IK residual (mm) to include in training")
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--no-fk', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    proj = args.project_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Find project files ─────────────────────────────────────────────
    qpos_csv = os.path.join(proj, 'qpos_export.csv')
    session_json = os.path.join(proj, 'mujoco_session.json')

    # Find keypoints3d.csv (use latest session)
    kp3d_csv = None
    labeled_dir = os.path.join(proj, 'labeled_data')
    if os.path.isdir(labeled_dir):
        for session in sorted(os.listdir(labeled_dir), reverse=True):
            candidate = os.path.join(labeled_dir, session, 'keypoints3d.csv')
            if os.path.exists(candidate):
                kp3d_csv = candidate
                break

    if not os.path.exists(qpos_csv):
        print(f"ERROR: {qpos_csv} not found. Export qpos from RED first.")
        sys.exit(1)
    if kp3d_csv is None:
        print(f"ERROR: No keypoints3d.csv found in {labeled_dir}")
        sys.exit(1)

    print(f"Project:  {proj}")
    print(f"qpos:     {qpos_csv}")
    print(f"kp3d:     {kp3d_csv}")
    print(f"session:  {session_json}")

    # ── Load arena transform ───────────────────────────────────────────
    if os.path.exists(session_json):
        arena_tf = ArenaTransform.from_session(session_json)
        stac_offsets, _ = load_stac_offsets(session_json)
        print(f"Arena:    R={'180° z-rot' if arena_tf.R[0,0] < 0 else 'identity'}, "
              f"scale={arena_tf.scale}, t={arena_tf.t}")
        print(f"STAC:     {len(stac_offsets)} site offsets")
    else:
        arena_tf = ArenaTransform()
        stac_offsets = {}
        print("WARNING: No mujoco_session.json — using default transform")

    # ── Load data ──────────────────────────────────────────────────────
    print("\nLoading data...")
    kp3d_data = load_keypoints3d(kp3d_csv)
    print(f"  keypoints3d: {len(kp3d_data)} frames")

    qpos_data, nq = load_qpos(qpos_csv, max_residual_mm=args.max_residual,
                               require_converged=False)
    print(f"  qpos: {len(qpos_data)} frames (nq={nq}, max_res<{args.max_residual}mm)")

    common = sorted(set(kp3d_data.keys()) & set(qpos_data.keys()))
    print(f"  common: {len(common)} frames")
    if len(common) < 50:
        print("ERROR: Too few common frames.")
        sys.exit(1)

    # ── Temporal split ─────────────────────────────────────────────────
    n = len(common)
    n_val = max(1, int(n * args.val_split))
    train_frames = common[:n - n_val]
    val_frames = common[n - n_val:]
    print(f"  train: {len(train_frames)}, val: {len(val_frames)}")

    # ── Datasets ───────────────────────────────────────────────────────
    train_ds = IKDataset(kp3d_data, qpos_data, train_frames, arena_tf)
    val_ds = IKDataset(kp3d_data, qpos_data, val_frames, arena_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Sanity check: print first sample
    kp_sample, v_sample, q_sample = train_ds[0]
    print(f"\n  Sample kp3d (MJ frame): [{kp_sample[0,0]:.4f}, {kp_sample[0,1]:.4f}, {kp_sample[0,2]:.4f}]")
    print(f"  Sample qpos root:      [{q_sample[0]:.4f}, {q_sample[1]:.4f}, {q_sample[2]:.4f}]")

    # ── FK engine ──────────────────────────────────────────────────────
    fk_engine = None
    if not args.no_fk:
        print("Loading MuJoCo FK engine (with STAC offsets)...")
        fk_engine = MuJoCoFK(MODEL_XML, RAT24_SITES, stac_offsets=stac_offsets)
        print(f"  FK model nq={fk_engine.nq}")

    # ── Model ──────────────────────────────────────────────────────────
    n_hinge = nq - 7
    device = args.device
    model = LearnedIK(n_keypoints=24, n_hinge=n_hinge, hidden=args.hidden,
                      n_layers=args.layers, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters "
          f"(in=96, hidden={args.hidden}×{args.layers-1}, out={3+6+n_hinge})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ───────────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Ep':>4} {'Train':>10} {'Val':>10} {'FK(mm)':>8} "
          f"{'Ang(°)':>7} {'LR':>9}")
    print("-" * 55)

    for epoch in range(args.epochs):
        t0 = time.time()
        tm = train_one_epoch(model, train_loader, optimizer, fk_engine,
                             fk_weight=args.fk_weight if not args.no_fk else 0.0,
                             device=device)
        vm = validate(model, val_loader, fk_engine,
                      fk_weight=args.fk_weight if not args.no_fk else 0.0,
                      device=device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"{epoch+1:4d} {tm['loss']:10.6f} {vm['loss']:10.6f} "
                  f"{vm['fk_err_mm']:8.2f} {vm['mean_angle_err_deg']:7.2f} "
                  f"{lr:9.1e} ({dt:.1f}s)")

        if vm['loss'] < best_val_loss:
            best_val_loss = vm['loss']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': vm['loss'],
                'val_fk_err_mm': vm['fk_err_mm'],
                'val_angle_err_deg': vm['mean_angle_err_deg'],
                'n_keypoints': 24, 'n_hinge': n_hinge, 'nq': nq,
                'hidden': args.hidden, 'n_layers': args.layers,
                'arena_R': arena_tf.R.tolist(),
                'arena_t': arena_tf.t.tolist(),
                'arena_scale': arena_tf.scale,
                'model_xml': MODEL_XML,
            }, os.path.join(args.output_dir, 'learned_ik.pt'))

    print(f"\nBest epoch: {best_epoch} (val_loss={best_val_loss:.6f})")

    # ── Final validation ───────────────────────────────────────────────
    ckpt = torch.load(os.path.join(args.output_dir, 'learned_ik.pt'),
                      weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    fv = validate(model, val_loader, fk_engine, device=device)

    print(f"\nFinal validation:")
    print(f"  FK site error:    {fv['fk_err_mm']:.2f} mm")
    print(f"  Mean angle error: {fv['mean_angle_err_deg']:.2f}°")
    print(f"  Max angle error:  {fv['max_angle_err_deg']:.2f}°")

    # ── Speed benchmark ────────────────────────────────────────────────
    model.eval()
    dummy = (torch.randn(1, 24, 3, device=device), torch.ones(1, 24, device=device))
    for _ in range(10):
        model(*dummy)
    t0 = time.time()
    for _ in range(1000):
        model(*dummy)
    ms = (time.time() - t0)
    print(f"  Inference: {ms:.1f}ms / 1000 frames = {ms/1000*1000:.3f} ms/frame")

    # ── Export ─────────────────────────────────────────────────────────
    print("\nExporting...")
    model = model.to('cpu')
    export_onnx(model, os.path.join(args.output_dir, 'learned_ik.onnx'))
    export_coreml(model, os.path.join(args.output_dir, 'learned_ik.mlpackage'))
    print(f"\nDone. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
