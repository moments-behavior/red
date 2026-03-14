#!/usr/bin/env python3
"""
extract_synced_frames.py — Extract diverse synchronized frame sets from multi-camera videos.

Given a directory of synced multi-camera videos (CamXXXXXXX.mp4), selects temporally
diverse frames with interesting scene content and extracts the corresponding frame
from every camera. Useful for calibration refinement where you want frames with
objects in different positions across the arena.

Dependencies: torch, numpy (NO OpenCV — uses ffmpeg subprocess calls)
  pip install torch numpy

Usage:
  python extract_synced_frames.py \\
      --video_dir /Volumes/johnsonlab/jinyao_share/2025_09_03_15_18_21 \\
      --output_dir /path/to/output \\
      --num_sets 30

Frame extraction uses ffmpeg via subprocess (fast seeking with -ss before -i).
Diversity scoring uses SuperPoint feature count + pixel difference from frame 0.
"""

import argparse
import json
import os
import re
import shutil
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
    print("ERROR: torch and lightglue are required.")
    print("  pip install torch")
    print("  pip install git+https://github.com/cvg/LightGlue.git")
    sys.exit(1)


FPS = 180.0


def find_camera_videos(video_dir: Path) -> dict[str, Path]:
    """Find all CamXXXXXXX.mp4 files and return {serial: path} dict."""
    videos = {}
    for f in sorted(video_dir.glob("Cam*.mp4")):
        m = re.match(r"Cam(\d+)\.mp4", f.name)
        if m:
            videos[m.group(1)] = f
    return videos


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {result.stderr}")
    return float(result.stdout.strip())


def extract_candidate_frames(video_path: Path, output_dir: Path,
                             sample_interval: float) -> list[Path]:
    """Extract one frame every sample_interval seconds from a video.

    Returns list of output jpg paths, ordered by time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "frame_%05d.jpg")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps=1/{sample_interval}",
        "-qscale:v", "2",
        pattern,
    ]
    print(f"  Extracting candidates: ffmpeg fps=1/{sample_interval} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg candidate extraction failed: {result.stderr[:500]}")

    frames = sorted(output_dir.glob("frame_*.jpg"))
    print(f"  Extracted {len(frames)} candidate frames")
    return frames


def extract_single_frame(video_path: Path, frame_num: int, output_path: Path) -> bool:
    """Extract a single frame by number using fast seeking.

    Uses -ss before -i for fast seeking, then grabs one frame.
    """
    timestamp = frame_num / FPS
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp:.6f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-qscale:v", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def score_candidates(frame_paths: list[Path], device: torch.device,
                     sample_interval: float) -> list[dict]:
    """Score each candidate frame for diversity/interest.

    Scoring combines:
      - SuperPoint feature count (normalized) — more features = more texture/objects
      - Pixel difference from first frame (normalized) — more difference = more activity

    Returns list of dicts with frame_idx, timestamp, score, num_features, pixel_diff.
    """
    print(f"  Loading SuperPoint on {device} ...")
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    # Load first frame as baseline for pixel difference
    first_image = load_image(str(frame_paths[0])).to(device)
    # Convert to grayscale mean for pixel diff (image is [C,H,W] float 0-1)
    first_gray = first_image.mean(dim=0)

    results = []
    feature_counts = []
    pixel_diffs = []

    print(f"  Scoring {len(frame_paths)} candidates ...")
    for i, fpath in enumerate(frame_paths):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    [{i+1}/{len(frame_paths)}]")

        image = load_image(str(fpath)).to(device)

        # SuperPoint features
        with torch.no_grad():
            feats = extractor.extract(image)
        num_kp = feats["keypoints"].shape[1]
        feature_counts.append(num_kp)

        # Pixel difference from first frame
        gray = image.mean(dim=0)
        # Resize if shapes don't match (shouldn't happen but be safe)
        if gray.shape != first_gray.shape:
            pixel_diff = 0.0
        else:
            pixel_diff = (gray - first_gray).abs().mean().item()
        pixel_diffs.append(pixel_diff)

        results.append({
            "frame_idx": i,
            "timestamp": i * sample_interval,
            "frame_num": int(round(i * sample_interval * FPS)),
            "num_features": num_kp,
            "pixel_diff": pixel_diff,
        })

    # Normalize both scores to [0, 1]
    fc = np.array(feature_counts, dtype=np.float64)
    pd = np.array(pixel_diffs, dtype=np.float64)

    fc_min, fc_max = fc.min(), fc.max()
    pd_min, pd_max = pd.min(), pd.max()

    for i, r in enumerate(results):
        fc_norm = (fc[i] - fc_min) / (fc_max - fc_min + 1e-8)
        pd_norm = (pd[i] - pd_min) / (pd_max - pd_min + 1e-8)
        r["score"] = float(fc_norm * 0.5 + pd_norm * 0.5)

    return results


def greedy_select(candidates: list[dict], num_sets: int,
                  min_separation: float) -> list[dict]:
    """Greedy selection of diverse, high-scoring frames.

    1. Sort by score descending
    2. Pick highest-scoring frame
    3. Iteratively add next-highest that is >= min_separation from all selected
    4. Stop at num_sets
    """
    # Filter out very low-scoring frames (bottom 10%)
    scores = [c["score"] for c in candidates]
    threshold = np.percentile(scores, 10)

    eligible = [c for c in candidates if c["score"] >= threshold]
    eligible.sort(key=lambda c: c["score"], reverse=True)

    selected = []
    for c in eligible:
        if len(selected) >= num_sets:
            break

        t = c["timestamp"]
        too_close = any(abs(t - s["timestamp"]) < min_separation for s in selected)
        if too_close:
            continue

        selected.append(c)

    # Sort selected by timestamp for nice ordering
    selected.sort(key=lambda c: c["timestamp"])
    return selected


def extract_frame_sets(selected: list[dict], videos: dict[str, Path],
                       output_dir: Path) -> None:
    """Extract the selected frame from every camera — parallel per frame."""
    from concurrent.futures import ThreadPoolExecutor

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(selected)

    for set_idx, info in enumerate(selected, start=1):
        frame_num = info["frame_num"]
        timestamp = info["timestamp"]
        set_dir = output_dir / f"set_{set_idx:03d}"
        set_dir.mkdir(parents=True, exist_ok=True)

        # Extract all cameras in parallel for this frame
        def extract_cam(args):
            serial, vpath = args
            out_path = set_dir / f"Cam{serial}.jpg"
            return extract_single_frame(vpath, frame_num, out_path)

        with ThreadPoolExecutor(max_workers=len(videos)) as executor:
            list(executor.map(extract_cam, sorted(videos.items())))

        print(f"  Set {set_idx:3d}/{total} — frame {frame_num} "
              f"(t={timestamp:.1f}s, score={info['score']:.3f})")


def choose_device(device_str: str) -> torch.device:
    """Select torch device: 'auto' picks MPS > CUDA > CPU."""
    if device_str != "auto":
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Extract diverse synchronized frame sets from multi-camera videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to folder with CamXXXXXXX.mp4 videos")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for frame sets")
    parser.add_argument("--ref_camera", type=str, default="710038",
                        help="Reference camera serial for diversity scoring")
    parser.add_argument("--num_sets", type=int, default=30,
                        help="Number of frame sets to extract")
    parser.add_argument("--sample_interval", type=float, default=5.0,
                        help="Sample candidate frames every N seconds")
    parser.add_argument("--min_separation", type=float, default=10.0,
                        help="Minimum seconds between selected frames")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device for SuperPoint (auto, cpu, cuda, mps)")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    # --- Find cameras ---
    print(f"Scanning {video_dir} for camera videos ...")
    videos = find_camera_videos(video_dir)
    if not videos:
        print(f"ERROR: No CamXXXXXXX.mp4 files found in {video_dir}")
        sys.exit(1)
    print(f"Found {len(videos)} cameras: {', '.join(sorted(videos.keys()))}")

    # --- Validate reference camera ---
    if args.ref_camera not in videos:
        print(f"ERROR: Reference camera Cam{args.ref_camera} not found.")
        print(f"  Available: {', '.join(sorted(videos.keys()))}")
        sys.exit(1)
    ref_video = videos[args.ref_camera]

    # --- Get video info ---
    duration = get_video_duration(ref_video)
    total_frames = int(duration * FPS)
    print(f"Reference video: {ref_video.name}")
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min), ~{total_frames} frames at {FPS} fps")

    # --- Step 1: Extract candidate frames from reference camera ---
    print(f"\n=== Step 1: Extract candidate frames (every {args.sample_interval}s) ===")
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="red_candidates_") as tmpdir:
        tmp_path = Path(tmpdir)
        candidate_frames = extract_candidate_frames(ref_video, tmp_path,
                                                    args.sample_interval)
        if not candidate_frames:
            print("ERROR: No candidate frames extracted.")
            sys.exit(1)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        # --- Step 2: Score candidates ---
        print(f"\n=== Step 2: Score frame diversity ===")
        t0 = time.time()
        device = choose_device(args.device)
        candidates = score_candidates(candidate_frames, device, args.sample_interval)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

    # --- Step 3: Greedy selection ---
    print(f"\n=== Step 3: Select {args.num_sets} diverse frames "
          f"(min separation {args.min_separation}s) ===")
    selected = greedy_select(candidates, args.num_sets, args.min_separation)
    print(f"  Selected {len(selected)} frame sets")
    if len(selected) < args.num_sets:
        print(f"  WARNING: Only {len(selected)} frames met criteria "
              f"(requested {args.num_sets})")

    for i, s in enumerate(selected, 1):
        print(f"    {i:3d}. t={s['timestamp']:7.1f}s  frame={s['frame_num']:7d}  "
              f"score={s['score']:.3f}  features={s['num_features']}  "
              f"pixel_diff={s['pixel_diff']:.4f}")

    # --- Step 4: Extract synced frame sets from all cameras ---
    print(f"\n=== Step 4: Extract synced frames from {len(videos)} cameras ===")
    t0 = time.time()
    extract_frame_sets(selected, videos, output_dir)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # --- Save metadata ---
    metadata = {
        "video_dir": str(video_dir),
        "ref_camera": args.ref_camera,
        "fps": FPS,
        "sample_interval": args.sample_interval,
        "min_separation": args.min_separation,
        "num_cameras": len(videos),
        "cameras": sorted(videos.keys()),
        "frames": [
            {
                "set": i + 1,
                "frame_num": s["frame_num"],
                "timestamp_sec": round(s["timestamp"], 3),
                "score": round(s["score"], 4),
                "num_features": s["num_features"],
                "pixel_diff": round(s["pixel_diff"], 6),
            }
            for i, s in enumerate(selected)
        ],
    }
    meta_path = output_dir / "frame_selection.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")
    print(f"Extracted {len(selected)} frame sets to {output_dir}")


if __name__ == "__main__":
    main()
