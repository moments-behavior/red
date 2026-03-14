#!/usr/bin/env python3
"""
red_to_colmap.py — Convert RED calibration YAMLs to COLMAP sparse model format.

Reads Cam*.yaml files (OpenCV format) and writes COLMAP's cameras.txt,
images.txt, and points3D.txt for use with 3DGS tools.

Usage:
  python red_to_colmap.py \
      --calib_dir /path/to/calibration \
      --image_dir /path/to/images \
      --output_dir /path/to/colmap_output

Dependencies: numpy (NO OpenCV, NO scipy — uses pure numpy quaternion conversion)
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np


def read_yaml_matrix(path, key):
    """Parse an opencv-matrix entry from a YAML file."""
    with open(path, "r") as f:
        content = f.read()
    pattern = (
        rf"{key}:\s*!!opencv-matrix\s*"
        rf"rows:\s*(\d+)\s*cols:\s*(\d+)\s*dt:\s*d\s*data:\s*\[([\s\S]*?)\]"
    )
    match = re.search(pattern, content)
    if not match:
        return None
    rows, cols = int(match.group(1)), int(match.group(2))
    data_str = match.group(3)
    values = [float(x.strip()) for x in data_str.split(",") if x.strip()]
    return np.array(values).reshape(rows, cols)


def read_yaml_scalar(path, key):
    """Parse a simple key: value scalar from a YAML file."""
    with open(path, "r") as f:
        for line in f:
            m = re.match(rf"^\s*{key}\s*:\s*(.+)$", line)
            if m:
                return m.group(1).strip()
    return None


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz].

    Uses Shepperd's method for numerical stability.
    Handles proper rotations only (det(R) = +1).
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    # Normalize
    q = np.array([qw, qx, qy, qz])
    q /= np.linalg.norm(q)
    # Convention: qw > 0
    if q[0] < 0:
        q = -q
    return q


def main():
    parser = argparse.ArgumentParser(
        description="Convert RED calibration YAMLs to COLMAP sparse model format"
    )
    parser.add_argument("--calib_dir", type=str, required=True,
                        help="Path to folder with Cam*.yaml calibration files")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to folder with Cam*.jpg images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for COLMAP model")
    parser.add_argument("--points_3d", type=str, default=None,
                        help="Optional points_3d.json for initial 3D points")
    args = parser.parse_args()

    calib_path = Path(args.calib_dir)
    image_path = Path(args.image_dir)
    output_path = Path(args.output_dir) / "sparse" / "0"
    output_path.mkdir(parents=True, exist_ok=True)

    # Also symlink images into the output structure
    images_link = Path(args.output_dir) / "images"
    if not images_link.exists():
        images_link.symlink_to(image_path.resolve())

    # Load cameras
    yaml_files = sorted(calib_path.glob("Cam*.yaml"))
    if not yaml_files:
        print(f"ERROR: no Cam*.yaml files in {calib_path}")
        sys.exit(1)

    cameras = []
    for yf in yaml_files:
        serial = re.search(r"Cam(\d+)", yf.stem)
        if not serial:
            continue
        serial = serial.group(1)

        K = read_yaml_matrix(str(yf), "camera_matrix")
        dist = read_yaml_matrix(str(yf), "distortion_coefficients")
        R = read_yaml_matrix(str(yf), "rc_ext")
        t = read_yaml_matrix(str(yf), "tc_ext")
        w = read_yaml_scalar(str(yf), "image_width")
        h = read_yaml_scalar(str(yf), "image_height")

        if K is None or R is None or t is None:
            print(f"  WARNING: skipping {yf.name}")
            continue

        # Find matching image
        img_name = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = image_path / f"Cam{serial}{ext}"
            if candidate.exists():
                img_name = candidate.name
                break
        if img_name is None:
            print(f"  WARNING: no image for Cam{serial}, skipping")
            continue

        cameras.append({
            "serial": serial,
            "K": K,
            "dist": dist.flatten()[:5] if dist is not None else np.zeros(5),
            "R": R,
            "t": t.flatten()[:3],
            "width": int(w) if w else 0,
            "height": int(h) if h else 0,
            "image_name": img_name,
        })

    print(f"Loaded {len(cameras)} cameras")

    # Write cameras.txt
    cam_file = output_path / "cameras.txt"
    with open(cam_file, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, cam in enumerate(cameras):
            cam_id = i + 1
            K = cam["K"]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            d = cam["dist"]
            k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
            # RED and COLMAP both use pixel-center convention (0,0 = center of top-left pixel)
            cx_c = cx
            cy_c = cy
            f.write(f"{cam_id} OPENCV {cam['width']} {cam['height']} "
                    f"{fx} {fy} {cx_c} {cy_c} {k1} {k2} {p1} {p2}\n")
    print(f"  Written {cam_file}")

    # Write images.txt
    img_file = output_path / "images.txt"
    with open(img_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, cam in enumerate(cameras):
            cam_id = i + 1
            R = cam["R"].copy()
            t = cam["t"].copy()

            # Handle improper rotation (det(R) = -1)
            # Negate third column to get proper rotation for quaternion
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1

            quat = rotation_matrix_to_quaternion(R)  # [qw, qx, qy, qz]
            f.write(f"{cam_id} {quat[0]} {quat[1]} {quat[2]} {quat[3]} "
                    f"{t[0]} {t[1]} {t[2]} {cam_id} {cam['image_name']}\n")
            f.write("\n")  # empty 2D points line
    print(f"  Written {img_file}")

    # Write points3D.txt
    pts_file = output_path / "points3D.txt"
    with open(pts_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        if args.points_3d and os.path.exists(args.points_3d):
            import json
            with open(args.points_3d) as pf:
                pts = json.load(pf)
            for pid_str, coords in pts.items():
                pid = int(pid_str) + 1  # COLMAP IDs are 1-based
                f.write(f"{pid} {coords[0]} {coords[1]} {coords[2]} "
                        f"128 128 128 0.5\n")
            print(f"  Written {len(pts)} 3D points")
        else:
            print(f"  Written empty points3D.txt (no initial points)")
    print(f"  Written {pts_file}")

    print(f"\nCOLMAP model written to {output_path}")
    print(f"  Images symlinked from {image_path}")
    print(f"\nTo use with 3DGS:")
    print(f"  python train.py -s {args.output_dir}")


if __name__ == "__main__":
    main()
