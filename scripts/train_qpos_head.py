#!/usr/bin/env python3
"""Train QposHead on cached HybridNet features.

Two modes:
  B: Fix coordinate frame — arena-transform centroid, predict full qpos
  D: Residual refinement — learned IK gives initial qpos, V2VNet features refine it

Usage:
    python3 train_qpos_head.py --mode B ...  # or --mode D
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# Arena Transform
# ═══════════════════════════════════════════════════════════════════════════

class ArenaTransform:
    """Rigid transform: calibration mm → MuJoCo meters.
    p_mj = scale * (p_mm @ R.T) + t
    """
    def __init__(self, R=None, t=None, scale=0.001):
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros(3)
        self.scale = scale

    @classmethod
    def from_session(cls, path):
        with open(path) as f:
            session = json.load(f)
        arena = session.get('arena', {})
        if not arena.get('valid', False):
            print("  WARNING: arena not valid, using identity + scale=0.001")
            return cls()
        R = np.array(arena['R'], dtype=np.float64).reshape(3, 3)
        t = np.array(arena.get('t', [0, 0, 0]), dtype=np.float64)
        scale = arena.get('scale', 0.001)
        return cls(R=R, t=t, scale=scale)

    def to_torch(self, device='cpu'):
        return (torch.tensor(self.R, dtype=torch.float32, device=device),
                torch.tensor(self.t, dtype=torch.float32, device=device),
                self.scale)


def arena_transform_batch(kp3d_mm, R, t, scale):
    """Transform [B, N, 3] keypoints from calibration mm → MuJoCo meters."""
    return scale * (kp3d_mm @ R.T) + t


# ═══════════════════════════════════════════════════════════════════════════
# Rotation utilities
# ═══════════════════════════════════════════════════════════════════════════

def rotation_6d_to_matrix(r6d):
    a1 = F.normalize(r6d[:, :3], dim=-1)
    a2 = r6d[:, 3:6]
    a2 = a2 - (a1 * a2).sum(-1, keepdim=True) * a1
    a2 = F.normalize(a2, dim=-1)
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack([a1, a2, a3], dim=-1)


def matrix_to_quaternion(R):
    R00, R01, R02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    R10, R11, R12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    R20, R21, R22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    trace = R00 + R11 + R22
    s1 = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2.0
    q1 = torch.stack([0.25*s1, (R21-R12)/s1, (R02-R20)/s1, (R10-R01)/s1], dim=-1)
    s2 = torch.sqrt(torch.clamp(1.0+R00-R11-R22, min=1e-10)) * 2.0
    q2 = torch.stack([(R21-R12)/s2, 0.25*s2, (R01+R10)/s2, (R02+R20)/s2], dim=-1)
    s3 = torch.sqrt(torch.clamp(1.0+R11-R00-R22, min=1e-10)) * 2.0
    q3 = torch.stack([(R02-R20)/s3, (R01+R10)/s3, 0.25*s3, (R12+R21)/s3], dim=-1)
    s4 = torch.sqrt(torch.clamp(1.0+R22-R00-R11, min=1e-10)) * 2.0
    q4 = torch.stack([(R10-R01)/s4, (R02+R20)/s4, (R12+R21)/s4, 0.25*s4], dim=-1)
    quat = torch.where((trace > 0).unsqueeze(-1), q1,
           torch.where(((R00 > R11) & (R00 > R22)).unsqueeze(-1), q2,
           torch.where((R11 > R22).unsqueeze(-1), q3, q4)))
    return F.normalize(quat, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Mode B: Full QposHead with corrected coordinate frame
# ═══════════════════════════════════════════════════════════════════════════

class QposHeadB(nn.Module):
    """Predict full qpos from V2VNet features + arena-transformed centroid."""

    def __init__(self, n_joints=24, n_hinge=61, hidden=256,
                 joint_limits_lo=None, joint_limits_hi=None):
        super().__init__()
        self.n_hinge = n_hinge
        self.encoder = nn.Sequential(
            nn.Conv3d(n_joints, 64, 3, padding=1),
            nn.InstanceNorm3d(64), nn.SiLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm3d(128), nn.SiLU(),
            nn.Conv3d(128, hidden, 3, stride=2, padding=1),
            nn.InstanceNorm3d(hidden), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        # Root offset: small displacement from centroid (in MuJoCo meters)
        self.root_offset_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 3))
        # Root rotation: 6D continuous → quaternion
        self.root_rot_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 6))
        # Hinge angles
        self.hinge_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, n_hinge))

        if joint_limits_lo is not None:
            self.register_buffer('jnt_lo', torch.tensor(joint_limits_lo, dtype=torch.float32))
            self.register_buffer('jnt_hi', torch.tensor(joint_limits_hi, dtype=torch.float32))
        else:
            self.register_buffer('jnt_lo', torch.full((n_hinge,), -1.5))
            self.register_buffer('jnt_hi', torch.full((n_hinge,), 1.5))

    def forward(self, volume_features, centroid_mj):
        """
        Args:
            volume_features: [B, n_joints, G, G, G]
            centroid_mj: [B, 3] — keypoint centroid in MuJoCo meters
        """
        B = volume_features.shape[0]
        feat = self.encoder(volume_features).view(B, -1)
        root_offset = self.root_offset_head(feat)
        root_pos = centroid_mj + root_offset  # both in MuJoCo meters now
        rot6d = self.root_rot_head(feat)
        quat = matrix_to_quaternion(rotation_6d_to_matrix(rot6d))
        hinge_raw = self.hinge_head(feat)
        hinge = self.jnt_lo + torch.sigmoid(hinge_raw) * (self.jnt_hi - self.jnt_lo)
        qpos = torch.cat([root_pos, quat, hinge], dim=-1)
        return qpos


# ═══════════════════════════════════════════════════════════════════════════
# Mode D: Residual refinement — learned IK initial estimate + V2VNet refine
# ═══════════════════════════════════════════════════════════════════════════

class LearnedIKSimple(nn.Module):
    """Simplified learned IK: 3D keypoints (MuJoCo meters) → qpos.
    Position-invariant: center-subtract, predict root as centroid + offset.
    """
    def __init__(self, n_keypoints=24, n_hinge=61, hidden=256):
        super().__init__()
        input_dim = n_keypoints * 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.root_offset = nn.Linear(hidden, 3)
        self.root_rot = nn.Linear(hidden, 6)
        self.hinge_out = nn.Linear(hidden, n_hinge)

    def forward(self, kp3d_mj, jnt_lo, jnt_hi):
        """kp3d_mj: [B, 24, 3] in MuJoCo meters."""
        centroid = kp3d_mj.mean(dim=1)  # [B, 3]
        kp_centered = (kp3d_mj - centroid.unsqueeze(1)).reshape(kp3d_mj.shape[0], -1)
        feat = self.net(kp_centered)
        root_pos = centroid + self.root_offset(feat)
        quat = matrix_to_quaternion(rotation_6d_to_matrix(self.root_rot(feat)))
        hinge_raw = self.hinge_out(feat)
        hinge = jnt_lo + torch.sigmoid(hinge_raw) * (jnt_hi - jnt_lo)
        return torch.cat([root_pos, quat, hinge], dim=-1)


class QposHeadD(nn.Module):
    """Residual refinement: initial qpos (from learned IK) + V2VNet features → refined qpos."""

    def __init__(self, n_joints=24, n_hinge=61, hidden=256,
                 joint_limits_lo=None, joint_limits_hi=None):
        super().__init__()
        self.n_hinge = n_hinge

        # Learned IK for initial estimate
        self.learned_ik = LearnedIKSimple(n_keypoints=n_joints, n_hinge=n_hinge)

        # V2VNet feature encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(n_joints, 64, 3, padding=1),
            nn.InstanceNorm3d(64), nn.SiLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm3d(128), nn.SiLU(),
            nn.Conv3d(128, hidden, 3, stride=2, padding=1),
            nn.InstanceNorm3d(hidden), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        # Refinement: takes initial qpos encoding + V2VNet features → residuals
        qpos_dim = 3 + 6 + n_hinge  # root_pos + rot6d + hinge = 70
        self.refine = nn.Sequential(
            nn.Linear(hidden + qpos_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.delta_pos = nn.Linear(hidden, 3)
        self.delta_rot = nn.Linear(hidden, 6)
        self.delta_hinge = nn.Linear(hidden, n_hinge)

        if joint_limits_lo is not None:
            self.register_buffer('jnt_lo', torch.tensor(joint_limits_lo, dtype=torch.float32))
            self.register_buffer('jnt_hi', torch.tensor(joint_limits_hi, dtype=torch.float32))
        else:
            self.register_buffer('jnt_lo', torch.full((n_hinge,), -1.5))
            self.register_buffer('jnt_hi', torch.full((n_hinge,), 1.5))

    def forward(self, volume_features, kp3d_mj):
        """
        Args:
            volume_features: [B, n_joints, G, G, G]
            kp3d_mj: [B, 24, 3] — keypoints in MuJoCo meters
        """
        B = volume_features.shape[0]

        # Initial estimate from learned IK (differentiable)
        init_qpos = self.learned_ik(kp3d_mj, self.jnt_lo, self.jnt_hi)
        init_pos = init_qpos[:, :3]
        init_quat = init_qpos[:, 3:7]
        init_hinge = init_qpos[:, 7:]

        # Encode V2VNet features
        vol_feat = self.encoder(volume_features).view(B, -1)

        # Encode initial qpos (use 6D rot for continuity, not quaternion)
        init_rot6d = self.learned_ik.root_rot(
            self.learned_ik.net(
                (kp3d_mj - kp3d_mj.mean(dim=1, keepdim=True)).reshape(B, -1)))
        init_encoding = torch.cat([init_pos, init_rot6d, init_hinge], dim=-1)

        # Predict residuals
        combined = torch.cat([vol_feat, init_encoding], dim=-1)
        refine_feat = self.refine(combined)

        delta_pos = self.delta_pos(refine_feat) * 0.01  # small corrections (meters)
        delta_rot6d = self.delta_rot(refine_feat) * 0.1  # small rotation corrections
        delta_hinge = self.delta_hinge(refine_feat) * 0.1  # small angle corrections

        # Apply residuals
        refined_pos = init_pos + delta_pos
        refined_rot6d = init_rot6d + delta_rot6d
        refined_quat = matrix_to_quaternion(rotation_6d_to_matrix(refined_rot6d))
        refined_hinge = torch.clamp(init_hinge + delta_hinge, self.jnt_lo, self.jnt_hi)

        qpos = torch.cat([refined_pos, refined_quat, refined_hinge], dim=-1)
        return qpos


# ═══════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════

class QposLoss(nn.Module):
    def __init__(self, w_trans=10.0, w_rot=1.0, w_hinge=1.0, w_reg=0.01):
        super().__init__()
        self.w_trans = w_trans
        self.w_rot = w_rot
        self.w_hinge = w_hinge
        self.w_reg = w_reg

    def forward(self, pred_qpos, gt_qpos):
        l_trans = F.mse_loss(pred_qpos[:, :3], gt_qpos[:, :3])
        q_pred = F.normalize(pred_qpos[:, 3:7], dim=-1)
        q_gt = F.normalize(gt_qpos[:, 3:7], dim=-1)
        dot = (q_pred * q_gt).sum(dim=-1).abs()
        l_rot = (1.0 - dot ** 2).mean()
        l_hinge = F.mse_loss(pred_qpos[:, 7:], gt_qpos[:, 7:])
        l_reg = (pred_qpos[:, 7:] ** 2).mean()
        total = (self.w_trans * l_trans + self.w_rot * l_rot +
                 self.w_hinge * l_hinge + self.w_reg * l_reg)
        return total, {
            'trans': l_trans.item(), 'rot': l_rot.item(),
            'hinge': l_hinge.item(), 'reg': l_reg.item(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_qpos_csv(csv_path, max_residual_mm=20.0):
    qpos_dict = {}
    with open(csv_path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('frame,') or not line.strip():
                continue
            parts = line.strip().split(',')
            frame = int(parts[0])
            qpos = np.array([float(x) for x in parts[1:69]], dtype=np.float32)
            residual = float(parts[69])
            if residual <= max_residual_mm:
                qpos_dict[frame] = qpos
    return qpos_dict


def get_frame_number_from_dataset(dataset, idx):
    img_id = dataset.image_ids[idx]
    file_name = dataset.imgs[img_id]["file_name"]
    frame_str = file_name.split("Frame_")[-1].split(".")[0]
    return int(frame_str)


def get_joint_limits(model_path):
    try:
        import mujoco
        m = mujoco.MjModel.from_binary_path(model_path)
        lo = m.jnt_range[1:, 0].copy().astype(np.float32)
        hi = m.jnt_range[1:, 1].copy().astype(np.float32)
        return lo, hi
    except Exception as e:
        print(f"  WARNING: Could not load MuJoCo model: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Feature caching (shared between modes)
# ═══════════════════════════════════════════════════════════════════════════

def cache_features(args):
    """Run pretrained HybridNet, save V2VNet features + 3D keypoints."""
    from jarvis.config.project_manager import ProjectManager
    from jarvis.hybridnet.hybridnet import HybridNet
    from jarvis.dataset.dataset3D import Dataset3D

    cache_dir = os.path.join(args.output_dir, "cached_features")
    if os.path.exists(os.path.join(cache_dir, "train_features.pt")):
        print("Cached features exist, skipping.")
        return
    os.makedirs(cache_dir, exist_ok=True)

    pm = ProjectManager()
    pm.load(args.project)
    cfg = pm.get_cfg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading pretrained HybridNet...")
    hybridnet = HybridNet("inference", cfg,
                          weights=args.pretrained_hybridnet,
                          efficienttrack_weights=args.pretrained_keypoint)
    model = hybridnet.model.to(device).eval()

    print(f"Loading qpos labels...")
    qpos_dict = load_qpos_csv(args.qpos_csv, max_residual_mm=args.max_residual)
    print(f"  {len(qpos_dict)} frames with residual <= {args.max_residual}mm")

    for split in ["train", "val"]:
        print(f"\nCaching {split}...")
        ds = Dataset3D(cfg, set=split)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                           num_workers=cfg.DATALOADER_NUM_WORKERS, pin_memory=True)

        all_features, all_points3d, all_qpos, all_frames = [], [], [], []
        skipped = 0

        with torch.no_grad():
            for i, sample in enumerate(loader):
                imgs = sample[0].permute(0, 1, 4, 2, 3).float().to(device)
                centerHM = sample[2].to(device)
                center3D = sample[3].float().to(device)
                cameraMatrices = sample[5].to(device)
                intrinsicMatrices = sample[6].to(device)
                distCoeffs = sample[7].to(device)
                img_size = torch.tensor([imgs.shape[3], imgs.shape[4]],
                                       dtype=torch.float32).to(device)

                frame_num = get_frame_number_from_dataset(ds, i)
                if frame_num not in qpos_dict:
                    skipped += 1
                    continue

                heatmap_final, _, points3D, _ = model(
                    imgs, img_size, centerHM, center3D,
                    cameraMatrices, intrinsicMatrices, distCoeffs)

                all_features.append(heatmap_final.cpu())
                all_points3d.append(points3D.cpu())
                all_qpos.append(torch.tensor(qpos_dict[frame_num]).unsqueeze(0))
                all_frames.append(frame_num)

                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(ds)} ({len(all_features)} kept, {skipped} skipped)")

        if not all_features:
            print(f"  WARNING: No features for {split}!")
            continue

        torch.save({
            'features': torch.cat(all_features, dim=0),
            'points3d': torch.cat(all_points3d, dim=0),
            'qpos': torch.cat(all_qpos, dim=0),
            'frames': all_frames,
        }, os.path.join(cache_dir, f"{split}_features.pt"))
        print(f"  Saved: {len(all_features)} frames")


# ═══════════════════════════════════════════════════════════════════════════
# Cached dataset with arena transform
# ═══════════════════════════════════════════════════════════════════════════

class CachedDataset(Dataset):
    def __init__(self, cache_path, arena_R, arena_t, arena_scale):
        data = torch.load(cache_path, weights_only=False)
        self.features = data['features']
        self.points3d = data['points3d']  # [N, 24, 3] in calibration mm
        self.qpos = data['qpos']
        self.arena_R = arena_R
        self.arena_t = arena_t
        self.arena_scale = arena_scale

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Arena-transform keypoints: mm → MuJoCo meters
        kp_mm = self.points3d[idx]  # [24, 3]
        kp_mj = self.arena_scale * (kp_mm @ self.arena_R.T) + self.arena_t
        centroid_mj = kp_mj.mean(dim=0)  # [3]
        return self.features[idx], kp_mj, centroid_mj, self.qpos[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Training loop (shared, mode-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    cache_dir = os.path.join(args.output_dir, "cached_features")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # Load arena transform
    arena = ArenaTransform.from_session(args.session_json)
    arena_R, arena_t, arena_scale = arena.to_torch()
    print(f"Arena: scale={arena_scale}, R_diag={np.diag(arena.R)}, t={arena.t}")

    # Load datasets
    train_ds = CachedDataset(os.path.join(cache_dir, "train_features.pt"),
                             arena_R, arena_t, arena_scale)
    val_ds = CachedDataset(os.path.join(cache_dir, "val_features.pt"),
                           arena_R, arena_t, arena_scale)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Joint limits
    jnt_lo, jnt_hi = None, None
    if args.mujoco_model:
        jnt_lo, jnt_hi = get_joint_limits(args.mujoco_model)

    # Create model
    n_joints = train_ds.features.shape[1]
    if args.mode == 'B':
        model = QposHeadB(n_joints=n_joints, n_hinge=61,
                          joint_limits_lo=jnt_lo, joint_limits_hi=jnt_hi).to(device)
    elif args.mode == 'D':
        model = QposHeadD(n_joints=n_joints, n_hinge=61,
                          joint_limits_lo=jnt_lo, joint_limits_hi=jnt_hi).to(device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}, {n_params:,} params")

    criterion = QposLoss(w_trans=10.0, w_rot=1.0, w_hinge=1.0, w_reg=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs)

    print(f"Training for {args.epochs} epochs...\n")

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0
        train_comp = {}
        for features, kp_mj, centroid_mj, gt_qpos in train_loader:
            features = features.to(device)
            kp_mj = kp_mj.to(device)
            centroid_mj = centroid_mj.to(device)
            gt_qpos = gt_qpos.to(device)

            if args.mode == 'B':
                pred_qpos = model(features, centroid_mj)
            else:  # D
                pred_qpos = model(features, kp_mj)

            loss, comp = criterion(pred_qpos, gt_qpos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            for k, v in comp.items():
                train_comp[k] = train_comp.get(k, 0) + v

        train_loss /= len(train_loader)
        for k in train_comp:
            train_comp[k] /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_comp = {}
        with torch.no_grad():
            for features, kp_mj, centroid_mj, gt_qpos in val_loader:
                features = features.to(device)
                kp_mj = kp_mj.to(device)
                centroid_mj = centroid_mj.to(device)
                gt_qpos = gt_qpos.to(device)

                if args.mode == 'B':
                    pred_qpos = model(features, centroid_mj)
                else:
                    pred_qpos = model(features, kp_mj)

                loss, comp = criterion(pred_qpos, gt_qpos)
                val_loss += loss.item()
                for k, v in comp.items():
                    val_comp[k] = val_comp.get(k, 0) + v

        val_loss /= len(val_loader)
        for k in val_comp:
            val_comp[k] /= len(val_loader)

        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch, 'mode': args.mode,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, f"qpos_head_{args.mode}_best.pth"))
            improved = " *"

        print(f"  {epoch+1:3d}/{args.epochs} | "
              f"train={train_loss:.4f} val={val_loss:.4f}{improved} | "
              f"t={train_comp.get('trans',0):.6f} "
              f"r={train_comp.get('rot',0):.4f} "
              f"h={train_comp.get('hinge',0):.4f} | "
              f"lr={lr:.6f} | {dt:.1f}s")

    print(f"\nDone. Best val: {best_val_loss:.4f}")
    print(f"Saved: {args.output_dir}/qpos_head_{args.mode}_best.pth")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=['B', 'D'])
    p.add_argument("--project", required=True)
    p.add_argument("--pretrained-hybridnet", required=True)
    p.add_argument("--pretrained-keypoint", required=True)
    p.add_argument("--qpos-csv", required=True)
    p.add_argument("--session-json", required=True, help="mujoco_session.json for arena transform")
    p.add_argument("--mujoco-model", default=None)
    p.add_argument("--output-dir", default="qpos_output")
    p.add_argument("--max-residual", type=float, default=20.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--train-only", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.train_only:
        cache_features(args)
    if not args.cache_only:
        train(args)


if __name__ == "__main__":
    main()
