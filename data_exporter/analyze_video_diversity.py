#!/usr/bin/env python3
"""
analyze_video_diversity.py — Find maximally diverse frames in a video.

Scans a top-down camera video at high temporal resolution, scores frames by
visual diversity (feature count, scene content change, object presence), and
selects N frames that maximize spatial diversity of scene content.

Uses a two-pass approach:
  Pass 1: Fast scan with pixel-level metrics (every 2s)
  Pass 2: SuperPoint feature extraction on top candidates

Designed to find frames with the rat, ball, ramp, robots in different positions.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import torch
    from lightglue import SuperPoint
    from lightglue.utils import load_image
except ImportError:
    print("ERROR: torch and lightglue required.")
    sys.exit(1)


FPS = 180.0


def extract_frames_fast(video_path, output_dir, interval=2.0):
    """Extract frames at fixed interval using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps=1/{interval}",
        "-qscale:v", "3",  # slightly lower quality for speed
        pattern,
    ]
    print(f"  Extracting frames every {interval}s...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:300]}")
    frames = sorted(output_dir.glob("frame_*.jpg"))
    print(f"  Extracted {len(frames)} frames")
    return frames


def compute_frame_metrics(frame_paths, interval):
    """Compute per-frame metrics using fast pixel-level analysis.

    For each frame, compute:
    - mean_brightness: average pixel value
    - content_score: standard deviation of pixels (high = complex scene)
    - diff_from_prev: pixel difference from previous frame (high = something moved)
    - diff_from_first: pixel difference from first frame (high = scene changed)
    """
    print(f"  Computing pixel metrics for {len(frame_paths)} frames...")
    metrics = []
    prev_gray = None
    first_gray = None

    for i, fpath in enumerate(frame_paths):
        # Load as grayscale using torch (no cv2)
        img = load_image(str(fpath))  # (3, H, W) float [0,1]
        gray = img.mean(dim=0).numpy()  # (H, W)

        # Downsample for speed (4x)
        gray_small = gray[::4, ::4]

        if first_gray is None:
            first_gray = gray_small

        content_score = float(gray_small.std())
        mean_bright = float(gray_small.mean())
        diff_first = float(np.abs(gray_small - first_gray).mean())
        diff_prev = float(np.abs(gray_small - prev_gray).mean()) if prev_gray is not None else 0.0

        metrics.append({
            "idx": i,
            "timestamp": i * interval,
            "frame_num": int(round(i * interval * FPS)),
            "content_score": content_score,
            "mean_brightness": mean_bright,
            "diff_from_first": diff_first,
            "diff_from_prev": diff_prev,
        })

        prev_gray = gray_small

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(frame_paths)}]")

    return metrics



def select_diverse_frames(metrics, frame_paths, num_select, min_separation, device):
    """Select maximally diverse frames using a multi-criteria approach.

    1. Score each frame by combined diversity metric
    2. Cluster frames into temporal segments
    3. Pick the best frame from each segment
    4. Fill remaining slots with highest-scoring uncovered frames
    """
    n = len(metrics)

    # Normalize metrics to [0,1]
    content = np.array([m["content_score"] for m in metrics])
    diff_first = np.array([m["diff_from_first"] for m in metrics])
    diff_prev = np.array([m["diff_from_prev"] for m in metrics])

    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-8 else np.zeros_like(x)

    content_n = norm(content)
    diff_first_n = norm(diff_first)
    diff_prev_n = norm(diff_prev)

    # Combined score: content complexity + scene change + difference from start
    # Higher weight on diff_from_first (captures rat/ball/ramp presence)
    for i, m in enumerate(metrics):
        m["combined_score"] = float(
            0.2 * content_n[i] +
            0.5 * diff_first_n[i] +
            0.3 * diff_prev_n[i]
        )

    # Now run SuperPoint on the top 150 candidates (by combined_score)
    # to get precise feature counts
    sorted_by_score = sorted(range(n), key=lambda i: metrics[i]["combined_score"], reverse=True)
    top_candidates = sorted_by_score[:min(150, n)]

    print(f"\n  Running SuperPoint on top {len(top_candidates)} candidates...")
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    for rank, idx in enumerate(top_candidates):
        img = load_image(str(frame_paths[idx])).to(device)
        with torch.no_grad():
            feats = extractor.extract(img, resize=1024)
        nkp = feats["keypoints"].shape[1]
        metrics[idx]["num_features"] = nkp
        if (rank + 1) % 30 == 0:
            print(f"    [{rank+1}/{len(top_candidates)}]")

    # Final score: combined_score * 0.6 + feature_count_norm * 0.4
    feat_counts = np.array([metrics[i].get("num_features", 0) for i in top_candidates])
    feat_n = norm(feat_counts) if len(feat_counts) > 1 else np.ones(len(feat_counts))

    for rank, idx in enumerate(top_candidates):
        metrics[idx]["final_score"] = float(
            0.6 * metrics[idx]["combined_score"] +
            0.4 * feat_n[rank]
        )

    # Greedy selection with temporal separation
    scored_candidates = [(idx, metrics[idx].get("final_score", metrics[idx]["combined_score"]))
                         for idx in top_candidates]
    scored_candidates.sort(key=lambda x: -x[1])

    selected = []
    selected_times = set()

    for idx, score in scored_candidates:
        if len(selected) >= num_select:
            break
        t = metrics[idx]["timestamp"]
        # Check minimum separation
        too_close = any(abs(t - st) < min_separation for st in selected_times)
        if too_close:
            continue
        selected.append(idx)
        selected_times.add(t)

    # Sort by time
    selected.sort(key=lambda i: metrics[i]["timestamp"])

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Find maximally diverse frames in a top-down camera video"
    )
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_frames", type=int, default=50, help="Number of frames to select")
    parser.add_argument("--scan_interval", type=float, default=2.0,
                        help="Scan interval in seconds (default: 2.0)")
    parser.add_argument("--min_separation", type=float, default=5.0,
                        help="Minimum seconds between selected frames (default: 5.0)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
    elif args.device != "cpu":
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Analyzing: {args.video}")

    # Extract candidate frames
    print(f"\n=== Pass 1: Extract frames every {args.scan_interval}s ===")
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="red_diversity_") as tmpdir:
        tmp_path = Path(tmpdir)
        frame_paths = extract_frames_fast(Path(args.video), tmp_path, args.scan_interval)

        if not frame_paths:
            print("ERROR: no frames extracted")
            sys.exit(1)

        # Compute pixel metrics
        print(f"\n=== Pass 2: Pixel-level analysis ===")
        metrics = compute_frame_metrics(frame_paths, args.scan_interval)
        elapsed1 = time.time() - t0
        print(f"  Done in {elapsed1:.1f}s")

        # Select diverse frames
        print(f"\n=== Pass 3: Select {args.num_frames} diverse frames ===")
        t0 = time.time()
        selected = select_diverse_frames(
            metrics, frame_paths, args.num_frames, args.min_separation, device
        )
        elapsed2 = time.time() - t0
        print(f"  Selected {len(selected)} frames in {elapsed2:.1f}s")

    # Print selected frames
    print(f"\n=== Selected Frames ===")
    print(f"  {'#':>3s}  {'Time':>8s}  {'Frame':>8s}  {'Score':>7s}  {'Features':>8s}  "
          f"{'DiffFirst':>10s}  {'Content':>8s}")
    for rank, idx in enumerate(selected):
        m = metrics[idx]
        nf = m.get("num_features", "?")
        fs = m.get("final_score", m["combined_score"])
        print(f"  {rank+1:3d}  {m['timestamp']:8.1f}  {m['frame_num']:8d}  {fs:7.3f}  "
              f"{nf:>8}  {m['diff_from_first']:10.4f}  {m['content_score']:8.4f}")

    # Save selection
    selection = {
        "video": str(args.video),
        "scan_interval": args.scan_interval,
        "min_separation": args.min_separation,
        "fps": FPS,
        "num_candidates": len(metrics),
        "num_candidates": len(metrics),
        "frames": [
            {
                "rank": rank + 1,
                "frame_num": metrics[idx]["frame_num"],
                "timestamp_sec": round(metrics[idx]["timestamp"], 3),
                "score": round(metrics[idx].get("final_score", metrics[idx]["combined_score"]), 4),
                "num_features": metrics[idx].get("num_features", 0),
                "diff_from_first": round(metrics[idx]["diff_from_first"], 6),
                "content_score": round(metrics[idx]["content_score"], 6),
            }
            for rank, idx in enumerate(selected)
        ],
    }

    sel_path = output_dir / "diverse_frame_selection.json"
    with open(sel_path, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"\nSelection saved to {sel_path}")
    print(f"\nTo extract synced frame sets from all cameras:")
    print(f"  Use the frame_num values from {sel_path}")


if __name__ == "__main__":
    main()
