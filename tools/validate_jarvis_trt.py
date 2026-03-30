#!/usr/bin/env python3
"""
Validate JARVIS conversion: .pt → ONNX → TRT
Compares .pt vs ONNX output on a real camera frame.
ONNX is the critical conversion step; TRT just optimizes ONNX.

Run in jarvis conda env:
    conda run -n jarvis python3 validate_jarvis_trt.py
"""

import numpy as np
import torch
import onnxruntime as ort
import cv2
import os

PT_DIR   = "/home/user/src/JARVIS-HybridNet/projects/mouseJan30/trt-models/predict2D"
TEST_IMG = "/home/user/testRTEngine/videos/jarvis_dataset/train/2025_09_15_13_34_07/Cam2012858/Frame_4324.jpg"

CENTER_PT   = os.path.join(PT_DIR, "centerDetect.pt")
KP_PT       = os.path.join(PT_DIR, "keypointDetect.pt")
CENTER_ONNX = os.path.join(PT_DIR, "centerDetect.onnx")
KP_ONNX     = os.path.join(PT_DIR, "keypointDetect.onnx")

CENTER_SIZE = 320
KP_SIZE     = 704
NUM_KP      = 24

BGR_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)  # B, G, R
BGR_STD  = np.array([0.225, 0.224, 0.229], dtype=np.float32)


def preprocess(img_bgr, out_size):
    img = cv2.resize(img_bgr, (out_size, out_size)).astype(np.float32) / 255.0
    img = (img - BGR_MEAN) / BGR_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)


def crop_around_center(img_bgr, cx, cy, crop_size):
    half = crop_size // 2
    padded = cv2.copyMakeBorder(img_bgr, half, half, half, half, cv2.BORDER_REPLICATE)
    x0 = max(0, cx)
    y0 = max(0, cy)
    return padded[y0:y0+crop_size, x0:x0+crop_size]


def run_pt(model, input_np, device):
    t = torch.from_numpy(input_np).to(device)
    with torch.no_grad():
        out = model(t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.cpu().numpy()


def argmax2d(hmap):
    idx = np.argmax(hmap)
    h, w = hmap.shape
    return idx // w, idx % w, float(hmap.flat[idx])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {device}")
    print(f"OnnxRuntime providers: {ort.get_available_providers()}")

    img = cv2.imread(TEST_IMG)
    if img is None:
        print(f"[ERROR] Cannot read: {TEST_IMG}")
        return
    H, W = img.shape[:2]
    print(f"Test image: {os.path.basename(TEST_IMG)}  ({W}x{H})\n")

    # ── CenterDetect ──────────────────────────────────────────────────────
    print("=" * 60)
    print("CenterDetect")
    print("=" * 60)
    inp_c = preprocess(img, CENTER_SIZE)

    # .pt
    m_c = torch.jit.load(CENTER_PT, map_location=device)
    m_c.eval()
    out_pt_c = run_pt(m_c, inp_c, device)
    hmap_pt = out_pt_c[0, 0]
    r_pt, c_pt, v_pt = argmax2d(hmap_pt)
    cx_pt = c_pt / hmap_pt.shape[1] * W
    cy_pt = r_pt / hmap_pt.shape[0] * H
    print(f"  .pt   heatmap {hmap_pt.shape}  argmax ({r_pt},{c_pt})  conf={sigmoid(v_pt):.3f}  → ({cx_pt:.1f}, {cy_pt:.1f}) px")

    # ONNX
    sess_c = ort.InferenceSession(CENTER_ONNX, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out_onnx_c = sess_c.run(None, {"input": inp_c})[0]
    hmap_onnx = out_onnx_c[0, 0]
    r_on, c_on, v_on = argmax2d(hmap_onnx)
    cx_on = c_on / hmap_onnx.shape[1] * W
    cy_on = r_on / hmap_onnx.shape[0] * H
    print(f"  ONNX  heatmap {hmap_onnx.shape}  argmax ({r_on},{c_on})  conf={sigmoid(v_on):.3f}  → ({cx_on:.1f}, {cy_on:.1f}) px")

    diff_c = np.abs(hmap_pt - hmap_onnx).mean()
    match_c = (r_pt == r_on and c_pt == c_on)
    print(f"  Mean heatmap diff: {diff_c:.5f}   Argmax match: {'✓' if match_c else '✗'}")

    # Use pt center for KP crop
    cx_i = int(round(cx_pt))
    cy_i = int(round(cy_pt))

    # ── KeypointDetect ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("KeypointDetect")
    print("=" * 60)
    half = KP_SIZE // 2
    crop = crop_around_center(img, cx_i - half, cy_i - half, KP_SIZE)
    inp_kp = preprocess(crop, KP_SIZE)
    crop_x0 = cx_pt - half
    crop_y0 = cy_pt - half

    m_kp = torch.jit.load(KP_PT, map_location=device)
    m_kp.eval()
    out_pt_kp = run_pt(m_kp, inp_kp, device)

    sess_kp = ort.InferenceSession(KP_ONNX, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out_onnx_kp = sess_kp.run(None, {"input": inp_kp})[0]

    print(f"  .pt   output {out_pt_kp.shape}   ONNX output {out_onnx_kp.shape}")
    scale_pt   = KP_SIZE / out_pt_kp.shape[2]
    scale_onnx = KP_SIZE / out_onnx_kp.shape[2]

    print(f"\n  {'KP':>3}  {'pt_px':>14}  {'onnx_px':>14}  {'conf_pt':>8}  {'match':>5}")
    matched = 0
    for k in range(NUM_KP):
        r_p, c_p, v_p = argmax2d(out_pt_kp[0, k])
        r_o, c_o, v_o = argmax2d(out_onnx_kp[0, k])
        px_pt   = (crop_x0 + c_p * scale_pt,   crop_y0 + r_p * scale_pt)
        px_onnx = (crop_x0 + c_o * scale_onnx, crop_y0 + r_o * scale_onnx)
        dist = ((px_pt[0]-px_onnx[0])**2 + (px_pt[1]-px_onnx[1])**2)**0.5
        ok = "✓" if dist < 5 else "✗"
        if dist < 5:
            matched += 1
        print(f"  {k:>3}  ({px_pt[0]:>6.1f},{px_pt[1]:>6.1f})  ({px_onnx[0]:>6.1f},{px_onnx[1]:>6.1f})"
              f"  {sigmoid(v_p):>8.3f}  {ok:>5}")

    diff_kp = np.abs(out_pt_kp - out_onnx_kp).mean()
    print(f"\n  Mean heatmap diff: {diff_kp:.5f}")
    print(f"  Keypoints within 5px: {matched}/{NUM_KP}")

    print()
    if matched == NUM_KP and match_c:
        print("[PASS] ONNX conversion is correct. TRT engines should match.")
    elif matched >= NUM_KP * 0.8:
        print(f"[WARN] {matched}/{NUM_KP} kpts match — FP32 vs FP32 should be exact. Check low-conf KPs.")
    else:
        print(f"[FAIL] Only {matched}/{NUM_KP} match — ONNX conversion has a bug.")
        print("       Check which tuple element was selected as ONNX output.")


if __name__ == "__main__":
    main()
