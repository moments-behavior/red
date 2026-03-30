#!/usr/bin/env python3
"""
Trace JARVIS CenterDetect + KeypointDetect to TorchScript .pt files
for loading in RED's JarvisInfer C++ class.

Usage (from anywhere):
    python3 /home/user/src/red/tools/trace_jarvis_torchscript.py

Output (written automatically):
    /home/user/src/JARVIS-HybridNet/projects/mouseJan30/trt-models/predict2D/
        centerDetect.pt
        keypointDetect.pt
"""

import sys
import os

JARVIS_ROOT = "/home/user/src/JARVIS-HybridNet"
PROJECT_NAME = "mouseJan30"

sys.path.insert(0, JARVIS_ROOT)

import torch
from jarvis.config.project_manager import ProjectManager

# ── Load project config ──────────────────────────────────────────────────────
pm = ProjectManager()
pm.load(PROJECT_NAME)
cfg = pm.get_cfg()

center_size = cfg.CENTERDETECT.IMAGE_SIZE          # 320
kp_size     = cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE # 704
n_joints    = cfg.KEYPOINTDETECT.NUM_JOINTS         # 24

print(f"[INFO] Project: {PROJECT_NAME}")
print(f"[INFO] CenterDetect input: {center_size}×{center_size}")
print(f"[INFO] KeypointDetect input: {kp_size}×{kp_size},  joints: {n_joints}")

from jarvis.efficienttrack.efficienttrack import EfficientTrack

device = torch.device("cpu")  # RED's libtorch is CPU-only; trace on CPU
print(f"[INFO] Device: {device}")

# ── Load CenterDetect ────────────────────────────────────────────────────────
print("\n[INFO] Loading CenterDetect …")
cd = EfficientTrack("CenterDetectInference", cfg, weights="latest")
cd_model = cd.model.eval().to(device)

ex_center = torch.randn(1, 3, center_size, center_size, device=device)
with torch.no_grad():
    out_c = cd_model(ex_center)
out_c0 = out_c[0] if isinstance(out_c, (tuple, list)) else out_c
print(f"[INFO] CenterDetect output shape: {list(out_c0.shape)}")

print("[INFO] Tracing CenterDetect …")
with torch.no_grad():
    traced_cd = torch.jit.trace(cd_model, ex_center)

# ── Load KeypointDetect ──────────────────────────────────────────────────────
print("\n[INFO] Loading KeypointDetect …")
kd = EfficientTrack("KeypointDetectInference", cfg, weights="latest")
kd_model = kd.model.eval().to(device)

ex_kp = torch.randn(1, 3, kp_size, kp_size, device=device)
with torch.no_grad():
    out_kp = kd_model(ex_kp)
out_kp0 = out_kp[0] if isinstance(out_kp, (tuple, list)) else out_kp
print(f"[INFO] KeypointDetect output shape: {list(out_kp0.shape)}")

print("[INFO] Tracing KeypointDetect …")
with torch.no_grad():
    traced_kd = torch.jit.trace(kd_model, ex_kp)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = os.path.join(JARVIS_ROOT, "projects", PROJECT_NAME,
                       "trt-models", "predict2D")
os.makedirs(out_dir, exist_ok=True)

center_out = os.path.join(out_dir, "centerDetect.pt")
kp_out     = os.path.join(out_dir, "keypointDetect.pt")

torch.jit.save(traced_cd, center_out)
torch.jit.save(traced_kd, kp_out)

print(f"\n[DONE] Saved:")
print(f"  {center_out}")
print(f"  {kp_out}")
print(f"\nLoad these in RED → Labeling Tool → JARVIS 2D Prediction → Browse buttons.")
