#!/usr/bin/env python3
"""
Validate CoTracker3 propagation independently of RED.

Steps:
  1. Run JARVIS on TARGET_FRAME for selected cameras → get 24-keypoint 2D detections
  2. Feed those detections as anchors into CoTracker3
  3. Track forward N_PROP_FRAMES frames from TARGET_FRAME
  4. Save an annotated video per camera showing the propagated tracks

Run in jarvis conda env:
    conda run -n base python3 validate_cotracker.py

Adjust SESSION_DIR, CAMERAS, TARGET_FRAME, and N_PROP_FRAMES below.
"""

import numpy as np
import onnxruntime as ort
import torch
import cv2
import os

# ── Config ───────────────────────────────────────────────────────────────────
SESSION_DIR   = "/mnt/ssd2/mickey/2025_12_05_15_59_05"
PT_DIR        = "/home/user/src/JARVIS-HybridNet/projects/mouseJan30/trt-models/predict2D"
CT_MODEL      = "/home/user/src/red/models/cotracker3_offline.pt"

# Only run on a handful of cameras with known good detection
CAMERAS       = ["Cam2002486", "Cam2002487", "Cam2005325", "Cam2006050"]
TARGET_FRAME  = 4600           # frame to run JARVIS on (anchor for CoTracker)
N_PROP_FRAMES = 50             # how many frames to propagate forward
CT_SCALE      = 0.25           # must match what RED uses in cotracker_infer.cpp

CENTER_ONNX   = os.path.join(PT_DIR, "centerDetect.onnx")
KP_ONNX       = os.path.join(PT_DIR, "keypointDetect.onnx")

CENTER_SIZE   = 320
KP_SIZE       = 704
NUM_KP        = 24

BGR_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
BGR_STD  = np.array([0.225, 0.224, 0.229], dtype=np.float32)

OUT_DIR = "/tmp/cotracker_validate"
# ─────────────────────────────────────────────────────────────────────────────


def extract_frames(mp4_path, start, count):
    """Extract `count` consecutive frames starting at `start` using OpenCV seek."""
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def extract_frame(mp4_path, frame_idx):
    frames = extract_frames(mp4_path, frame_idx, 1)
    return frames[0] if frames else None


def preprocess_bgr(img_bgr, out_size):
    img = cv2.resize(img_bgr, (out_size, out_size)).astype(np.float32) / 255.0
    img = (img - BGR_MEAN) / BGR_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)


def crop_around_center(img_bgr, cx_i, cy_i, crop_size):
    half = crop_size // 2
    padded = cv2.copyMakeBorder(img_bgr, half, half, half, half, cv2.BORDER_REPLICATE)
    x0 = cx_i   # in padded space
    y0 = cy_i
    return padded[y0:y0 + crop_size, x0:x0 + crop_size]


def argmax2d(hmap):
    idx = np.argmax(hmap)
    h, w = hmap.shape
    return idx // w, idx % w, float(hmap.flat[idx])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def run_jarvis(img_bgr, sess_c, sess_kp):
    H, W = img_bgr.shape[:2]
    inp_c = preprocess_bgr(img_bgr, CENTER_SIZE)
    out_c = sess_c.run(None, {"input": inp_c})[0]
    hmap_c = out_c[0, 0]
    center_out_size = hmap_c.shape[0]
    r_c, c_c, v_c = argmax2d(hmap_c)
    cx_px = c_c / center_out_size * W
    cy_px = r_c / center_out_size * H

    cx_i = int(round(cx_px))
    cy_i = int(round(cy_px))
    crop = crop_around_center(img_bgr, cx_i, cy_i, KP_SIZE)
    inp_kp = preprocess_bgr(crop, KP_SIZE)
    out_kp = sess_kp.run(None, {"input": inp_kp})[0]
    kp_out_size = out_kp.shape[2]
    scale = KP_SIZE / kp_out_size

    crop_x0 = cx_px - KP_SIZE / 2
    crop_y0 = cy_px - KP_SIZE / 2

    kps = []
    for k in range(NUM_KP):
        r_k, c_k, v_k = argmax2d(out_kp[0, k])
        kp_x = crop_x0 + c_k * scale
        kp_y = crop_y0 + r_k * scale
        kps.append((kp_x, kp_y, sigmoid(v_k)))
    return cx_px, cy_px, sigmoid(v_c), kps


def bgr_to_rgb_tensor(img_bgr, H_ct, W_ct, device):
    """Convert BGR uint8 frame to float RGB tensor (3, H_ct, W_ct) in [0,255]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1)          # (3, H, W)
    t = torch.nn.functional.interpolate(
        t.unsqueeze(0),
        size=(H_ct, W_ct),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return t.to(device)


def run_cotracker(ct_model, frames_bgr, anchor_kps, device):
    """
    frames_bgr : list of BGR images (anchor frame first)
    anchor_kps : list of (x_px, y_px) in pixel space
    Returns: tracks (T, N, 2) and vis (T, N) in pixel space
    """
    T_real = len(frames_bgr)
    N_real = len(anchor_kps)
    T_TRACE = 100
    N_TRACE = 24

    H, W = frames_bgr[0].shape[:2]
    H_ct = max(8, ((int(H * CT_SCALE) + 7) // 8) * 8)
    W_ct = max(8, ((int(W * CT_SCALE) + 7) // 8) * 8)
    print(f"  CoTracker spatial: {W}x{H} → {W_ct}x{H_ct}")

    # Build video tensor (1, T_TRACE, 3, H_ct, W_ct)
    zero_frame = torch.zeros(3, H_ct, W_ct, device=device)
    frame_tensors = []
    for i in range(T_TRACE):
        if i < T_real and frames_bgr[i] is not None:
            frame_tensors.append(bgr_to_rgb_tensor(frames_bgr[i], H_ct, W_ct, device))
        else:
            frame_tensors.append(zero_frame)
    video = torch.stack(frame_tensors, 0).unsqueeze(0)  # (1, T_TRACE, 3, H_ct, W_ct)

    # Build queries (1, N_TRACE, 3): [frame_idx, x_ct, y_ct]
    queries = torch.zeros(1, N_TRACE, 3, device=device)
    for n in range(min(N_real, N_TRACE)):
        x_px, y_px = anchor_kps[n]
        queries[0, n, 0] = 0                         # anchor is frame 0
        queries[0, n, 1] = x_px * CT_SCALE
        queries[0, n, 2] = y_px * CT_SCALE

    print(f"  Running CoTracker T={T_TRACE} N={N_TRACE} anchor query[0]=({queries[0,0,1]:.1f},{queries[0,0,2]:.1f})")
    with torch.no_grad():
        tracks, vis = ct_model(video, queries)

    tracks = tracks.cpu()                    # (1, T_TRACE, N_TRACE, 2)
    vis    = vis.sigmoid().cpu()             # (1, T_TRACE, N_TRACE)

    inv = 1.0 / CT_SCALE
    tracks_out = tracks[0, :T_real, :N_real].numpy() * inv   # (T_real, N_real, 2)
    vis_out    = vis[0,    :T_real, :N_real].numpy()          # (T_real, N_real)
    return tracks_out, vis_out


def draw_kps(img, kps_xy, vis=None, radius=6, color=(0, 255, 0)):
    out = img.copy()
    for n, (x, y) in enumerate(kps_xy):
        v = 1.0 if vis is None else vis[n]
        if v < 0.3:
            continue
        c = color if v > 0.5 else (0, 165, 255)
        cv2.circle(out, (int(round(x)), int(round(y))), radius, c, -1)
        cv2.putText(out, str(n), (int(round(x)) + 4, int(round(y)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading ONNX sessions…")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_c  = ort.InferenceSession(CENTER_ONNX,  providers=providers)
    sess_kp = ort.InferenceSession(KP_ONNX,      providers=providers)

    print("Loading CoTracker TorchScript model…")
    ct_model = torch.jit.load(CT_MODEL, map_location=device)
    ct_model.eval()

    # Find available cameras
    available = sorted([
        f.replace(".mp4", "")
        for f in os.listdir(SESSION_DIR) if f.endswith(".mp4")
    ])
    cams = [c for c in CAMERAS if c in available]
    if not cams:
        print(f"[ERROR] None of {CAMERAS} found in {SESSION_DIR}")
        print(f"Available cameras: {available[:8]}")
        return

    for cam in cams:
        mp4 = os.path.join(SESSION_DIR, f"{cam}.mp4")
        print(f"\n── Camera {cam} ──")

        # Step 1: JARVIS on anchor frame
        anchor_img = extract_frame(mp4, TARGET_FRAME)
        if anchor_img is None:
            print("  [SKIP] could not extract anchor frame")
            continue
        H, W = anchor_img.shape[:2]
        cx, cy, cconf, kps = run_jarvis(anchor_img, sess_c, sess_kp)
        print(f"  JARVIS center=({cx:.1f},{cy:.1f}) conf={cconf:.3f}")
        print(f"  kp[0]=({kps[0][0]:.1f},{kps[0][1]:.1f}) conf={kps[0][2]:.3f}")
        if cconf < 0.3:
            print("  [SKIP] low center confidence")
            continue

        # Step 2: Extract N_PROP_FRAMES frames starting at anchor
        print(f"  Extracting {N_PROP_FRAMES + 1} frames from {TARGET_FRAME}…")
        frames = extract_frames(mp4, TARGET_FRAME, N_PROP_FRAMES + 1)
        frames = [f for f in frames if f is not None]
        print(f"  Got {len(frames)} frames")

        # Anchor keypoints in pixel space
        anchor_kps = [(kps[k][0], kps[k][1]) for k in range(NUM_KP)]

        # Step 3: Run CoTracker
        tracks, vis = run_cotracker(ct_model, frames, anchor_kps, device)
        print(f"  tracks shape={tracks.shape}  vis range=[{vis.min():.3f},{vis.max():.3f}]")
        print(f"  kp[0] anchor=({anchor_kps[0][0]:.1f},{anchor_kps[0][1]:.1f})")
        for t in [0, 1, 5, 10, 25, len(frames)-1]:
            if t < len(frames):
                x, y = tracks[t, 0]
                print(f"    t={t:3d} kp[0]=({x:.1f},{y:.1f}) vis={vis[t,0]:.3f}")

        # Step 4: Save annotated video
        out_path = os.path.join(OUT_DIR, f"{cam}_cotracker.mp4")
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            25.0,
            (w, h),
        )
        for t, frame in enumerate(frames):
            annotated = draw_kps(frame, tracks[t], vis[t])
            # Label the frame number
            fn = TARGET_FRAME + t
            cv2.putText(annotated, f"frame {fn}  t={t}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            writer.write(annotated)
        writer.release()
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
