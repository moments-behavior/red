#!/usr/bin/env python3
"""Export RED calibration + video frames to nerfstudio transforms.json format.

Reads calibration from ba_poses.json (extrinsics) and intrinsics.json (image dims),
reads camera list from a .redproj file, and produces a transforms.json compatible
with nerfstudio's OPENCV camera model. Optionally extracts video frames via ffmpeg.

Dependencies: numpy, json, subprocess (no OpenCV, no scipy).
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BA_POSES = (
    "/Users/johnsonr/red_projects/hurdles_calib2/aruco_video_experimental/"
    "2026_03_11_00_01_54/summary_data/bundle_adjustment/ba_poses.json"
)
DEFAULT_PROJECT = "/Users/johnsonr/red_demos/mouse_active1/mouse_active1.redproj"
DEFAULT_OUTPUT_DIR = "/tmp/red_nerfstudio_test"
DEFAULT_FRAMES = "39348"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_fps(video_path: str) -> float:
    """Get video FPS via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    for stream in info["streams"]:
        if stream["codec_type"] == "video":
            # r_frame_rate is typically "num/den"
            num, den = stream["r_frame_rate"].split("/")
            return float(num) / float(den)
    raise RuntimeError(f"No video stream found in {video_path}")


def extract_frame(video_path: str, frame_number: int, fps: float,
                  output_path: str) -> str:
    """Extract a single frame from a video using ffmpeg."""
    timestamp = frame_number / fps
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp:.6f}",
        "-i", video_path,
        "-frames:v", "1",
        "-qscale:v", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"FAIL {output_path}: {result.stderr.strip()}"
    return f"OK   {output_path}"


def opencv_to_opengl_c2w(R_w2c: np.ndarray, t_cam: np.ndarray) -> np.ndarray:
    """Convert OpenCV world-to-camera (R, t) to nerfstudio OpenGL c2w 4x4."""
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_cam
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w
    # Negate Y and Z columns: OpenCV (+X right, +Y down, +Z forward)
    # -> OpenGL (+X right, +Y up, +Z backward)
    c2w[:3, 1:3] *= -1
    return c2w


def validate_transforms(frames: list[dict]) -> None:
    """Validate c2w matrices and print camera position stats."""
    positions = []
    max_orth_err = 0.0

    for f in frames:
        mat = np.array(f["transform_matrix"])
        R = mat[:3, :3]
        pos = mat[:3, 3]
        positions.append(pos)

        # Orthonormality check
        orth_err = np.max(np.abs(R.T @ R - np.eye(3)))
        max_orth_err = max(max_orth_err, orth_err)

    positions = np.array(positions)
    print("\n--- Validation ---")
    print(f"  Orthonormality max error: {max_orth_err:.2e}  "
          f"({'PASS' if max_orth_err < 1e-6 else 'FAIL'})")

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    spread = maxs - mins
    print(f"  Camera positions (X): min={mins[0]:.1f}  max={maxs[0]:.1f}  spread={spread[0]:.1f}")
    print(f"  Camera positions (Y): min={mins[1]:.1f}  max={maxs[1]:.1f}  spread={spread[1]:.1f}")
    print(f"  Camera positions (Z): min={mins[2]:.1f}  max={maxs[2]:.1f}  spread={spread[2]:.1f}")

    total_spread = np.linalg.norm(spread)
    all_at_origin = total_spread < 1.0
    print(f"  Total spread: {total_spread:.1f}  "
          f"({'FAIL - cameras clustered at origin!' if all_at_origin else 'PASS'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export RED calibration to nerfstudio transforms.json"
    )
    parser.add_argument(
        "--ba-poses", default=DEFAULT_BA_POSES,
        help="Path to ba_poses.json",
    )
    parser.add_argument(
        "--project", default=DEFAULT_PROJECT,
        help="Path to .redproj file",
    )
    parser.add_argument(
        "--frames", default=DEFAULT_FRAMES,
        help="Comma-separated frame numbers to extract (default: 39348)",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: /tmp/red_nerfstudio_test)",
    )
    parser.add_argument(
        "--skip-frames", action="store_true",
        help="Only generate transforms.json, skip ffmpeg frame extraction",
    )
    args = parser.parse_args()

    frame_numbers = [int(x.strip()) for x in args.frames.split(",")]

    # ---- Load calibration ----
    ba_poses = load_json(args.ba_poses)

    # Load intrinsics.json for image dimensions (sibling of bundle_adjustment/)
    intrinsics_path = os.path.join(
        os.path.dirname(os.path.dirname(args.ba_poses)), "intrinsics.json"
    )
    intrinsics = load_json(intrinsics_path)

    # ---- Load project ----
    project = load_json(args.project)
    camera_names = project["camera_names"]  # e.g. ["Cam2002486", ...]
    media_folder = project["media_folder"]

    # Map camera name -> cam_id string (strip "Cam" prefix)
    cam_ids = [name.replace("Cam", "") for name in camera_names]

    # Filter to cameras that exist in ba_poses
    available = [(name, cid) for name, cid in zip(camera_names, cam_ids)
                 if cid in ba_poses]
    if not available:
        print("ERROR: No cameras from project found in ba_poses.json")
        sys.exit(1)

    missing = set(cam_ids) - {cid for _, cid in available}
    if missing:
        print(f"WARNING: Cameras not in ba_poses: {sorted(missing)}")

    print(f"Cameras: {len(available)}  Frames: {frame_numbers}")

    # ---- Build transforms.json ----
    frames_list = []

    for cam_name, cam_id in available:
        pose = ba_poses[cam_id]
        intr = intrinsics.get(cam_id, {})

        R = np.array(pose["R"])
        t = np.array(pose["t"])
        K = np.array(pose["K"])
        dist = pose["dist"]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        w = intr.get("image_width", 3208)
        h = intr.get("image_height", 2200)

        c2w = opencv_to_opengl_c2w(R, t)

        for frame_num in frame_numbers:
            rel_path = f"images/{cam_name}_{frame_num}.jpg"
            entry = {
                "file_path": rel_path,
                "transform_matrix": c2w.tolist(),
                "fl_x": fx,
                "fl_y": fy,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "k1": dist[0],
                "k2": dist[1],
                "p1": dist[2],
                "p2": dist[3],
            }
            frames_list.append(entry)

    transforms = {
        "camera_model": "OPENCV",
        "frames": frames_list,
    }

    # ---- Write transforms.json ----
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "transforms.json")
    with open(out_path, "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"Wrote {out_path}  ({len(frames_list)} frame entries)")

    # ---- Validate ----
    validate_transforms(frames_list)

    # ---- Extract frames ----
    if args.skip_frames:
        print("\n--skip-frames set, skipping ffmpeg extraction.")
        return

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Get FPS from first available video
    first_video = os.path.join(media_folder, f"{available[0][0]}.mp4")
    fps = get_fps(first_video)
    print(f"\nVideo FPS: {fps:.2f}")
    print(f"Extracting {len(available) * len(frame_numbers)} frames...")

    tasks = []
    for cam_name, cam_id in available:
        video_path = os.path.join(media_folder, f"{cam_name}.mp4")
        if not os.path.isfile(video_path):
            print(f"WARNING: Video not found: {video_path}")
            continue
        for frame_num in frame_numbers:
            out_jpg = os.path.join(images_dir, f"{cam_name}_{frame_num}.jpg")
            tasks.append((video_path, frame_num, fps, out_jpg))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(extract_frame, *t) for t in tasks]
        for fut in as_completed(futures):
            print(f"  {fut.result()}")

    print(f"\nDone. Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
