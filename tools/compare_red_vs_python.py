#!/usr/bin/env python3
"""
Compare JARVIS inference output (Python/ONNX) vs what RED's C++ should produce.

Run in jarvis conda env:
    conda run -n jarvis python3 compare_red_vs_python.py

Extracts frame 41760 from each camera video, runs CenterDetect + KeypointDetect,
prints kp[0], kp[1], kp[12] in PIXEL space (Y=0 at top).

Then run RED at the same frame, click "Run JARVIS — This Frame", and compare the
console output from RED (which also logs pixel-space positions) with this script's output.

If numbers match  → inference is correct, issue is in RED's display/coordinate mapping.
If numbers differ → inference bug (preprocessing, channel order, etc).
"""

import numpy as np
import onnxruntime as ort
import cv2
import os
import subprocess
import tempfile

SESSION_DIR = "/mnt/mouse2/rory/2026_02_07_16_01_45"
PT_DIR      = "/home/user/src/JARVIS-HybridNet/projects/mouseJan30/trt-models/predict2D"
TARGET_FRAME = 41760

CENTER_ONNX = os.path.join(PT_DIR, "centerDetect.onnx")
KP_ONNX     = os.path.join(PT_DIR, "keypointDetect.onnx")

CENTER_SIZE = 320
KP_SIZE     = 704
NUM_KP      = 24

BGR_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)  # B, G, R order
BGR_STD  = np.array([0.225, 0.224, 0.229], dtype=np.float32)


def extract_frame(mp4_path, frame_idx):
    """Extract a single frame from mp4 as BGR numpy array using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp = f.name
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", mp4_path,
        "-vf", f"select=eq(n\\,{frame_idx})",
        "-vframes", "1",
        tmp
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"[WARN] ffmpeg failed for {mp4_path}: {r.stderr.decode()[:200]}")
        return None
    img = cv2.imread(tmp)
    os.unlink(tmp)
    return img


def preprocess_bgr(img_bgr, out_size):
    """Resize to out_size×out_size, normalize BGR channels."""
    img = cv2.resize(img_bgr, (out_size, out_size)).astype(np.float32) / 255.0
    img = (img - BGR_MEAN) / BGR_STD          # apply per-channel BGR stats
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)


def crop_around_center(img_bgr, cx_i, cy_i, crop_size):
    """Crop crop_size×crop_size window centered at (cx_i, cy_i) with replicate padding."""
    half = crop_size // 2
    padded = cv2.copyMakeBorder(img_bgr, half, half, half, half, cv2.BORDER_REPLICATE)
    x0 = cx_i  # in padded space, original (cx_i, cy_i) → (cx_i+half, cy_i+half)
    y0 = cy_i  #   top-left of desired crop: (cx_i, cy_i)
    return padded[y0:y0 + crop_size, x0:x0 + crop_size]


def argmax2d(hmap):
    """Return (row, col, value) of argmax of 2D array."""
    idx = np.argmax(hmap)
    h, w = hmap.shape
    return idx // w, idx % w, float(hmap.flat[idx])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def run_jarvis(img_bgr, sess_c, sess_kp):
    H, W = img_bgr.shape[:2]

    # --- CenterDetect ---
    inp_c = preprocess_bgr(img_bgr, CENTER_SIZE)
    out_c = sess_c.run(None, {"input": inp_c})[0]   # (1,1,80,80)
    hmap_c = out_c[0, 0]                             # (80,80)
    center_out_size = hmap_c.shape[0]
    r_c, c_c, v_c = argmax2d(hmap_c)
    center_conf = sigmoid(v_c)
    cx_px = c_c / center_out_size * W
    cy_px = r_c / center_out_size * H

    # --- KeypointDetect ---
    cx_i = int(round(cx_px))
    cy_i = int(round(cy_px))
    crop = crop_around_center(img_bgr, cx_i, cy_i, KP_SIZE)
    inp_kp = preprocess_bgr(crop, KP_SIZE)
    out_kp = sess_kp.run(None, {"input": inp_kp})[0]  # (1,24,176,176)
    kp_out_size = out_kp.shape[2]
    scale = KP_SIZE / kp_out_size

    # Top-left of crop in original image
    crop_x0 = cx_px - KP_SIZE / 2
    crop_y0 = cy_px - KP_SIZE / 2

    kps = []
    for k in range(NUM_KP):
        r_k, c_k, v_k = argmax2d(out_kp[0, k])
        kp_x = crop_x0 + c_k * scale
        kp_y = crop_y0 + r_k * scale
        kps.append((kp_x, kp_y, sigmoid(v_k)))

    return cx_px, cy_px, center_conf, kps


def main():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_c  = ort.InferenceSession(CENTER_ONNX,  providers=providers)
    sess_kp = ort.InferenceSession(KP_ONNX,      providers=providers)
    print(f"Providers: {ort.get_available_providers()}")
    print(f"Frame: {TARGET_FRAME}\n")

    cameras = sorted([
        f.replace(".mp4", "")
        for f in os.listdir(SESSION_DIR)
        if f.endswith(".mp4")
    ])

    for cam in cameras:
        mp4 = os.path.join(SESSION_DIR, f"{cam}.mp4")
        img = extract_frame(mp4, TARGET_FRAME)
        if img is None:
            print(f"[{cam}] frame extract failed")
            continue
        H, W = img.shape[:2]

        cx, cy, cconf, kps = run_jarvis(img, sess_c, sess_kp)
        print(f"[{cam}]  image={W}x{H}  center=({cx:.1f},{cy:.1f})  conf={cconf:.3f}")
        print(f"         kp[0]=({kps[0][0]:.1f},{kps[0][1]:.1f})  "
              f"kp[1]=({kps[1][0]:.1f},{kps[1][1]:.1f})  "
              f"kp[12]=({kps[12][0]:.1f},{kps[12][1]:.1f})")
        print()


if __name__ == "__main__":
    main()
