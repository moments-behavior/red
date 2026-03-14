#!/usr/bin/env python3
"""
calibration_refinement.py — Feature matching & epipolar diagnostic for multi-camera calibration.

Given multi-view images and existing calibration YAML files, produces multi-view
feature correspondences (landmarks.json) for C++ bundle adjustment, along with
epipolar diagnostics showing which cameras may have drifted or moved.

Dependencies: torch, numpy, lightglue (NO OpenCV)
  pip install git+https://github.com/cvg/LightGlue.git

Usage:
  python calibration_refinement.py \\
      --image_dir /path/to/images \\
      --calib_dir /path/to/calibration \\
      --output_dir /path/to/output

  Images should be named CamXXXXXXX.jpg (serial number in filename).
  Calibration files should be CamXXXXXXX.yaml (OpenCV YAML format).
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
except ImportError:
    print("ERROR: lightglue is required but not installed.")
    print("  pip install git+https://github.com/cvg/LightGlue.git")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Camera loading (uses shared yaml_utils — reads each file once)
# ---------------------------------------------------------------------------

from yaml_utils import load_camera_yaml


def load_calibration(calib_dir):
    """Load K, dist, R, t from Cam*.yaml files.

    Returns dict: {serial: {'K': 3x3, 'dist': 5-array, 'R': 3x3, 't': 3-array,
                             'image_width': int, 'image_height': int}}
    """
    cameras = {}
    calib_path = Path(calib_dir)
    for yaml_file in sorted(calib_path.glob("Cam*.yaml")):
        serial = re.search(r"Cam(\d+)", yaml_file.stem)
        if not serial:
            continue
        serial = serial.group(1)

        cam = load_camera_yaml(str(yaml_file))
        if cam is None:
            print(f"  WARNING: skipping {yaml_file.name} — missing calibration data")
            continue

        cameras[serial] = cam
    return cameras


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def select_viable_pairs(cameras, max_angle_deg=120.0):
    """Select camera pairs with optical axis angle < max_angle_deg.

    Optical axis in world coords: R^T @ [0,0,1].
    Returns list of (serial_a, serial_b) tuples, sorted.
    """
    serials = sorted(cameras.keys())
    z_axis = np.array([0.0, 0.0, 1.0])
    optical_axes = {}
    for s in serials:
        axis = cameras[s]["R"].T @ z_axis
        optical_axes[s] = axis / np.linalg.norm(axis)

    pairs = []
    cos_limit = np.cos(np.radians(max_angle_deg))
    for i in range(len(serials)):
        for j in range(i + 1, len(serials)):
            sa, sb = serials[i], serials[j]
            cos_angle = np.dot(optical_axes[sa], optical_axes[sb])
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            if cos_angle > cos_limit:
                pairs.append((sa, sb))
    return pairs


# ---------------------------------------------------------------------------
# Feature extraction and matching
# ---------------------------------------------------------------------------

def extract_and_match(image_dir, cameras, viable_pairs, device,
                      max_keypoints, resize, match_threshold):
    """Extract SuperPoint features and run LightGlue matching.

    Returns:
    - all_matches: {(sa, sb): {'pts_a': Nx2, 'pts_b': Nx2,
                                'indices_a': N, 'indices_b': N, 'scores': N}}
    - features: {serial: {'keypoints': Kx2, 'num_keypoints': int}}
    """
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(
        features="superpoint", flash=False, filter_threshold=match_threshold
    ).eval().to(device)

    img_path = Path(image_dir)

    # Determine which cameras we need features for
    needed_serials = set()
    for sa, sb in viable_pairs:
        needed_serials.add(sa)
        needed_serials.add(sb)

    # Extract features once per camera
    raw_feats = {}  # with batch dim, for matcher
    features = {}   # without batch dim, for track building
    for serial in sorted(needed_serials):
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            candidate = img_path / f"Cam{serial}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        if img_file is None:
            print(f"  WARNING: no image found for Cam{serial}, skipping")
            continue

        image = load_image(str(img_file)).to(device)
        with torch.no_grad():
            feats = extractor.extract(image, resize=resize)
        raw_feats[serial] = feats

        # Store CPU numpy copy without batch dim for track building
        feats_no_batch = rbd(feats)
        kpts = feats_no_batch["keypoints"].cpu().numpy()
        features[serial] = {"keypoints": kpts, "num_keypoints": len(kpts)}
        print(f"    Cam{serial}: {len(kpts)} keypoints")

    # Cache CPU numpy keypoints (avoid repeated rbd + cpu transfer in matching loop)
    kpts_numpy = {}
    for serial in raw_feats:
        kpts_numpy[serial] = features[serial]["keypoints"]

    # Match all viable pairs
    all_matches = {}
    matched_count = 0
    for sa, sb in viable_pairs:
        if sa not in raw_feats or sb not in raw_feats:
            continue

        with torch.no_grad():
            result = matcher({"image0": raw_feats[sa], "image1": raw_feats[sb]})

        result = rbd(result)

        matches = result["matches"].cpu().numpy()   # (N, 2)
        scores = result["scores"].cpu().numpy()      # (N,)

        if len(matches) == 0:
            continue

        idx_a = matches[:, 0]
        idx_b = matches[:, 1]
        pts_a = kpts_numpy[sa][idx_a]  # (N, 2)
        pts_b = kpts_numpy[sb][idx_b]  # (N, 2)

        all_matches[(sa, sb)] = {
            "pts_a": pts_a,
            "pts_b": pts_b,
            "indices_a": idx_a,
            "indices_b": idx_b,
            "scores": scores,
        }
        matched_count += 1
        print(f"    Cam{sa}-Cam{sb}: {len(matches)} matches")

    print(f"  {matched_count}/{len(viable_pairs)} pairs matched")
    return all_matches, features


# ---------------------------------------------------------------------------
# Undistortion (pure numpy, matches red_math.h)
# ---------------------------------------------------------------------------

def undistort_points(pts, K, dist):
    """Undistort pixel points to normalized camera coordinates.

    pts: (N, 2) pixel coordinates
    K: 3x3 camera matrix
    dist: 5-element array [k1, k2, p1, p2, k3]
    Returns: (N, 2) normalized undistorted coordinates
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]

    x0 = (pts[:, 0] - cx) / fx
    y0 = (pts[:, 1] - cy) / fy
    x, y = x0.copy(), y0.copy()

    for _ in range(10):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (x0 - dx) / radial
        y = (y0 - dy) / radial

    return np.stack([x, y], axis=1)


# ---------------------------------------------------------------------------
# Epipolar geometry
# ---------------------------------------------------------------------------

def skew_symmetric(v):
    """3x3 skew-symmetric matrix from a 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def project_point(pt3d, R, t, K, dist):
    """Project a 3D point to 2D pixel coords with distortion (matches red_math.h)."""
    cam = R @ pt3d + t
    if cam[2] <= 0:
        return None  # behind camera
    xp = cam[0] / cam[2]
    yp = cam[1] / cam[2]

    k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]
    r2 = xp * xp + yp * yp
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    xpp = xp * radial + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp)
    ypp = yp * radial + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([xpp * fx + cx, ypp * fy + cy])


def triangulate_dlt_batch(pts_a_undist, pts_b_undist, P_a, P_b):
    """Batch DLT triangulation for N point pairs.
    pts_a_undist, pts_b_undist: (N, 2) undistorted pixel coords
    P_a, P_b: (3, 4) projection matrices
    Returns: (N, 3) 3D points
    """
    N = len(pts_a_undist)
    # Build A matrix for each point: (N, 4, 4)
    A = np.zeros((N, 4, 4))
    A[:, 0, :] = pts_a_undist[:, 0:1] * P_a[2:3, :] - P_a[0:1, :]
    A[:, 1, :] = pts_a_undist[:, 1:2] * P_a[2:3, :] - P_a[1:2, :]
    A[:, 2, :] = pts_b_undist[:, 0:1] * P_b[2:3, :] - P_b[0:1, :]
    A[:, 3, :] = pts_b_undist[:, 1:2] * P_b[2:3, :] - P_b[1:2, :]
    # Batched SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1, :]  # (N, 4) last right singular vector
    # Dehomogenize
    pts3d = X[:, :3] / X[:, 3:4]
    return pts3d


def project_points_batch(pts3d, R, t, K, dist):
    """Project N 3D points to 2D with distortion. All numpy arrays.
    pts3d: (N, 3)
    Returns: (N, 2) pixel coordinates, (N,) boolean mask for valid (in front of camera)
    """
    cam = (R @ pts3d.T).T + t  # (N, 3)
    valid = cam[:, 2] > 0

    # Avoid division by zero
    z = np.where(valid, cam[:, 2], 1.0)
    xp = cam[:, 0] / z
    yp = cam[:, 1] / z

    k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]
    r2 = xp*xp + yp*yp
    r4 = r2*r2
    r6 = r4*r2
    radial = 1.0 + k1*r2 + k2*r4 + k3*r6
    xpp = xp*radial + 2.0*p1*xp*yp + p2*(r2 + 2.0*xp*xp)
    ypp = yp*radial + p1*(r2 + 2.0*yp*yp) + 2.0*p2*xp*yp

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    px = xpp * fx + cx
    py = ypp * fy + cy

    return np.stack([px, py], axis=1), valid


def triangulate_dlt(pts_undist, proj_matrices):
    """DLT triangulation from N undistorted 2D points and their 3x4 projection matrices.

    pts_undist: list of (2,) arrays in undistorted pixel coords
    proj_matrices: list of (3, 4) projection matrices (K @ [R|t])
    Returns: (3,) 3D point
    """
    n = len(pts_undist)
    A = np.zeros((2 * n, 4))
    for i in range(n):
        x, y = pts_undist[i][0], pts_undist[i][1]
        P = proj_matrices[i]
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def compute_reproj_errors(pts_a_px, pts_b_px, cam_a, cam_b):
    """Vectorized: triangulate all matches and compute max reprojection error.

    This is more robust than epipolar distance, especially for nearly co-located cameras.
    Returns: (N,) array of max reprojection errors (max of error in cam_a and cam_b) in pixels
    """
    N = len(pts_a_px)

    K_a, d_a, R_a, t_a = cam_a["K"], cam_a["dist"], cam_a["R"], cam_a["t"]
    K_b, d_b, R_b, t_b = cam_b["K"], cam_b["dist"], cam_b["R"], cam_b["t"]

    # Undistort (already vectorized)
    pts_a_undist_n = undistort_points(pts_a_px, K_a, d_a)
    pts_b_undist_n = undistort_points(pts_b_px, K_b, d_b)

    # Convert to undistorted pixel coords for DLT
    pts_a_undist = pts_a_undist_n * np.array([[K_a[0,0], K_a[1,1]]]) + np.array([[K_a[0,2], K_a[1,2]]])
    pts_b_undist = pts_b_undist_n * np.array([[K_b[0,0], K_b[1,1]]]) + np.array([[K_b[0,2], K_b[1,2]]])

    P_a = K_a @ np.hstack([R_a, t_a.reshape(3,1)])
    P_b = K_b @ np.hstack([R_b, t_b.reshape(3,1)])

    # Batch triangulate
    pts3d = triangulate_dlt_batch(pts_a_undist, pts_b_undist, P_a, P_b)

    # Batch reproject into both cameras
    proj_a, valid_a = project_points_batch(pts3d, R_a, t_a, K_a, d_a)
    proj_b, valid_b = project_points_batch(pts3d, R_b, t_b, K_b, d_b)

    # Max reprojection error per point
    err_a = np.linalg.norm(proj_a - pts_a_px, axis=1)
    err_b = np.linalg.norm(proj_b - pts_b_px, axis=1)
    errors = np.maximum(err_a, err_b)

    # Invalid points (behind camera) get max error
    invalid = ~(valid_a & valid_b)
    errors[invalid] = 1e6

    return errors


# ---------------------------------------------------------------------------
# Triangulation-based match filtering & diagnostic
# ---------------------------------------------------------------------------

def run_reproj_diagnostic(cameras, all_matches, reproj_thresh):
    """For each pair, triangulate matches and filter by reprojection error.

    Uses the init_calibration to triangulate each matched point pair, then
    checks if the reprojection error is below threshold. This is more robust
    than epipolar filtering, especially for nearly co-located cameras (overhead cluster).

    Returns:
    - filtered_matches: same format as all_matches but with outliers removed
    - diagnostic: {'pairs': {...}, 'per_camera': {...}}
    """
    filtered_matches = {}
    pair_stats = {}
    cam_errors = defaultdict(list)

    for (sa, sb), mdata in all_matches.items():
        errors = compute_reproj_errors(
            mdata["pts_a"], mdata["pts_b"], cameras[sa], cameras[sb]
        )
        mask = errors < reproj_thresh
        num_raw = len(errors)
        num_kept = int(np.sum(mask))
        valid = errors < 1e5  # exclude failed triangulations

        pair_stats[(sa, sb)] = {
            "median_err": float(np.median(errors[valid])) if np.any(valid) else 999.0,
            "p90_err": float(np.percentile(errors[valid], 90)) if np.any(valid) else 999.0,
            "num_raw": num_raw,
            "num_filtered": num_kept,
        }

        if np.any(valid):
            cam_errors[sa].extend(errors[valid].tolist())
            cam_errors[sb].extend(errors[valid].tolist())

        if num_kept > 0:
            filtered_matches[(sa, sb)] = {
                "pts_a": mdata["pts_a"][mask],
                "pts_b": mdata["pts_b"][mask],
                "indices_a": mdata["indices_a"][mask],
                "indices_b": mdata["indices_b"][mask],
                "scores": mdata["scores"][mask],
            }

    # Per-camera summary
    per_camera = {}
    for serial in sorted(cam_errors.keys()):
        errs = np.array(cam_errors[serial])
        med = float(np.median(errs))
        p90 = float(np.percentile(errs, 90))
        num_pairs = sum(
            1 for (sa, sb) in pair_stats if sa == serial or sb == serial
        )
        if med < 2.0:
            status = "OK"
        elif med < 5.0:
            status = "DRIFT"
        else:
            status = "MOVED"
        per_camera[serial] = {
            "median_err": med,
            "p90_err": p90,
            "num_pairs": num_pairs,
            "status": status,
        }

    # Print formatted table
    print("\n=== Reprojection Diagnostic (triangulate + reproject via init_calibration) ===")
    print(f"  {'Camera':>9s}  {'Median(px)':>10s}  {'P90(px)':>9s}  {'Pairs':>5s}  Status")
    for serial in sorted(per_camera, key=lambda s: -per_camera[s]["median_err"]):
        info = per_camera[serial]
        flag = ""
        if info["status"] == "MOVED":
            flag = "  ***"
        elif info["status"] == "DRIFT":
            flag = "   *"
        print(
            f"  {serial:>9s}  {info['median_err']:>10.2f}  {info['p90_err']:>9.2f}"
            f"  {info['num_pairs']:>5d}  {info['status']:<5s}{flag}"
        )

    # Print per-pair summary sorted by error
    print(f"\n  Per-pair summary (sorted by median reproj error):")
    print(f"  {'Pair':>20s}  {'Median':>8s}  {'P90':>8s}  {'Raw':>5s}  {'Kept':>5s}")
    for (sa, sb) in sorted(pair_stats, key=lambda k: -pair_stats[k]["median_err"]):
        ps = pair_stats[(sa, sb)]
        print(
            f"  {sa + '-' + sb:>20s}  {ps['median_err']:>8.2f}"
            f"  {ps['p90_err']:>8.2f}  {ps['num_raw']:>5d}  {ps['num_filtered']:>5d}"
        )

    diagnostic = {
        "pairs": {f"{sa}-{sb}": v for (sa, sb), v in pair_stats.items()},
        "per_camera": per_camera,
    }
    return filtered_matches, diagnostic


# ---------------------------------------------------------------------------
# Track assembly (union-find)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def build_tracks(cameras, filtered_matches, features):
    """Merge pairwise matches into multi-view tracks via union-find.

    Each keypoint gets a global ID: cam_offset[serial] + local_keypoint_index.
    For each pairwise match, union the two global IDs.
    Each connected component = one 3D point track.
    Keep tracks with >= 2 cameras.

    Returns: ({serial: {'ids': [...], 'landmarks': [[x, y], ...]}}, track_lengths)
    """
    serials = sorted(features.keys())

    # Assign global ID offsets
    cam_offset = {}
    total = 0
    for s in serials:
        cam_offset[s] = total
        total += features[s]["num_keypoints"]

    uf = UnionFind(total)

    # Union matched keypoints
    for (sa, sb), mdata in filtered_matches.items():
        for ia, ib in zip(mdata["indices_a"], mdata["indices_b"]):
            ga = cam_offset[sa] + int(ia)
            gb = cam_offset[sb] + int(ib)
            uf.union(ga, gb)

    # Collect active keypoints (only those participating in matches)
    active_keypoints = set()
    for (sa, sb), mdata in filtered_matches.items():
        for ia in mdata["indices_a"]:
            active_keypoints.add((sa, int(ia)))
        for ib in mdata["indices_b"]:
            active_keypoints.add((sb, int(ib)))

    # Group by union-find root
    components = defaultdict(list)
    for serial, local_idx in active_keypoints:
        gid = cam_offset[serial] + local_idx
        root = uf.find(gid)
        components[root].append((serial, local_idx))

    # Build tracks: keep those with >= 2 distinct cameras
    track_id = 0
    tracks = defaultdict(lambda: {"ids": [], "landmarks": []})
    track_lengths = []

    for root, members in components.items():
        cam_set = set(s for s, _ in members)
        if len(cam_set) < 2:
            continue

        # Deduplicate: one observation per camera (pick first)
        seen_cams = set()
        for serial, local_idx in members:
            if serial in seen_cams:
                continue
            seen_cams.add(serial)
            kpt = features[serial]["keypoints"][local_idx]
            tracks[serial]["ids"].append(track_id)
            tracks[serial]["landmarks"].append([float(kpt[0]), float(kpt[1])])

        track_lengths.append(len(seen_cams))
        track_id += 1

    # Convert to plain dict (only cameras with tracks)
    result = {}
    for serial in serials:
        if serial in tracks:
            result[serial] = {
                "ids": tracks[serial]["ids"],
                "landmarks": tracks[serial]["landmarks"],
            }

    return result, track_lengths


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def triangulate_and_filter_tracks(cameras, tracks, pair_quality, reproj_thresh=20.0):
    """Triangulate each track from the best camera pair and validate all observations.

    For each track:
    1. Find the best pair of cameras (lowest median reproj from diagnostic)
    2. Triangulate using that pair
    3. Reproject into ALL cameras that see this point
    4. Keep only observations with reproj error < threshold

    This produces high-quality 3D seed points for the C++ BA.

    Returns: (filtered_tracks, points_3d, stats)
    """
    # Build inverse map: track_id -> {serial: [x, y]}
    track_obs = defaultdict(dict)
    for serial, data in tracks.items():
        for tid, lm in zip(data["ids"], data["landmarks"]):
            track_obs[tid][serial] = np.array(lm)

    # Rank camera pairs by quality (from diagnostic)
    # pair_quality: {(sa, sb): median_reproj_error}

    filtered = defaultdict(lambda: {"ids": [], "landmarks": []})
    points_3d = {}
    stats = {"total": 0, "triangulated": 0, "kept": 0}

    for tid, obs in track_obs.items():
        stats["total"] += 1
        if len(obs) < 2:
            continue

        # Find best pair among cameras that see this point
        serials = list(obs.keys())
        best_pair = None
        best_err = 1e9
        for i in range(len(serials)):
            for j in range(i + 1, len(serials)):
                sa, sb = serials[i], serials[j]
                key = (min(sa, sb), max(sa, sb))
                if key in pair_quality and pair_quality[key] < best_err:
                    best_err = pair_quality[key]
                    best_pair = (sa, sb)

        if best_pair is None or best_err > reproj_thresh * 2:
            # No good pair — try any pair
            best_pair = (serials[0], serials[1])

        sa, sb = best_pair
        ca, cb = cameras[sa], cameras[sb]
        pa, pb = obs[sa], obs[sb]

        # Triangulate from best pair
        K_a, d_a, R_a, t_a = ca["K"], ca["dist"], ca["R"], ca["t"]
        K_b, d_b, R_b, t_b = cb["K"], cb["dist"], cb["R"], cb["t"]

        pa_n = undistort_points(pa.reshape(1, 2), K_a, d_a)[0]
        pb_n = undistort_points(pb.reshape(1, 2), K_b, d_b)[0]
        pa_px = np.array([pa_n[0] * K_a[0, 0] + K_a[0, 2], pa_n[1] * K_a[1, 1] + K_a[1, 2]])
        pb_px = np.array([pb_n[0] * K_b[0, 0] + K_b[0, 2], pb_n[1] * K_b[1, 1] + K_b[1, 2]])

        P_a = K_a @ np.hstack([R_a, t_a.reshape(3, 1)])
        P_b = K_b @ np.hstack([R_b, t_b.reshape(3, 1)])
        pt3d = triangulate_dlt([pa_px, pb_px], [P_a, P_b])

        # Check 3D point is in front of both seed cameras
        if (R_a @ pt3d + t_a)[2] <= 0 or (R_b @ pt3d + t_b)[2] <= 0:
            continue

        stats["triangulated"] += 1

        # Validate against ALL cameras that see this point (batch projection)
        obs_serials = list(obs.keys())
        obs_px = np.array([obs[s] for s in obs_serials])  # (M, 2)
        pt3d_batch = pt3d.reshape(1, 3).repeat(len(obs_serials), axis=0)  # (M, 3)

        good_obs = {}
        # Project into each camera (batch per-camera since each has different params)
        for idx_c, serial in enumerate(obs_serials):
            cam = cameras[serial]
            proj, valid = project_points_batch(
                pt3d.reshape(1, 3), cam["R"], cam["t"], cam["K"], cam["dist"]
            )
            if valid[0]:
                err = np.linalg.norm(proj[0] - obs_px[idx_c])
                if err < reproj_thresh:
                    good_obs[serial] = obs[serial]

        if len(good_obs) >= 2:
            stats["kept"] += 1
            points_3d[str(tid)] = [float(pt3d[0]), float(pt3d[1]), float(pt3d[2])]
            for serial, px in good_obs.items():
                filtered[serial]["ids"].append(tid)
                filtered[serial]["landmarks"].append([float(px[0]), float(px[1])])

    # Convert to plain dict
    result = {s: dict(v) for s, v in filtered.items() if v["ids"]}
    return result, points_3d, stats


def save_landmarks(tracks, output_dir, points_3d=None):
    """Save landmarks.json and optionally points_3d.json for the C++ pipeline."""
    output_path = Path(output_dir) / "landmarks.json"
    with open(output_path, "w") as f:
        json.dump(tracks, f, indent=2)

    if points_3d:
        pts_path = Path(output_dir) / "points_3d.json"
        with open(pts_path, "w") as f:
            json.dump(points_3d, f, indent=2)

    return str(output_path)


def save_match_stats(args, cameras, viable_pairs, all_matches, filtered_matches,
                     tracks, track_lengths, diagnostic, output_dir):
    """Save match_stats.json with diagnostic metadata."""
    hist = defaultdict(int)
    for length in track_lengths:
        hist[str(length)] += 1

    total_raw = sum(len(m["pts_a"]) for m in all_matches.values())
    total_filtered = sum(len(m["pts_a"]) for m in filtered_matches.values())
    per_cam_counts = {s: len(tracks[s]["ids"]) for s in sorted(tracks.keys())}

    stats = {
        "config": {
            "image_dir": [str(d) for d in args.image_dir] if isinstance(args.image_dir, list) else [str(args.image_dir)],
            "calib_dir": str(args.calib_dir),
            "max_keypoints": args.max_keypoints,
            "resize": args.resize,
            "match_threshold": args.match_threshold,
            "reproj_thresh": args.reproj_thresh,
            "min_matches": args.min_matches,
            "max_angle": args.max_angle,
        },
        "num_cameras": len(cameras),
        "num_viable_pairs": len(viable_pairs),
        "num_matched_pairs": len(all_matches),
        "num_filtered_pairs": len(filtered_matches),
        "total_raw_matches": total_raw,
        "total_filtered_matches": total_filtered,
        "num_tracks": len(track_lengths),
        "track_length_histogram": dict(sorted(hist.items())),
        "per_camera_landmarks": per_cam_counts,
        "diagnostic": diagnostic,
    }

    output_path = Path(output_dir) / "match_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    return str(output_path)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(requested):
    """Auto-detect best available torch device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-camera feature matching & epipolar diagnostic for calibration refinement"
    )
    parser.add_argument("--image_dir", type=str, required=True, nargs='+',
                        help="Path(s) to folder(s) with CamXXXXXXX.jpg images (multiple for multi-set)")
    parser.add_argument("--calib_dir", type=str, required=True,
                        help="Path to folder with CamXXXXXXX.yaml calibration files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <image_dir>/../correspondences)")
    parser.add_argument("--max_keypoints", type=int, default=4096,
                        help="SuperPoint max keypoints (default: 4096)")
    parser.add_argument("--resize", type=int, default=1600,
                        help="SuperPoint resize longest edge (default: 1600)")
    parser.add_argument("--match_threshold", type=float, default=0.2,
                        help="LightGlue filter threshold (default: 0.2)")
    parser.add_argument("--reproj_thresh", type=float, default=10.0,
                        help="Reprojection error threshold in pixels for match filtering (default: 10.0)")
    parser.add_argument("--min_matches", type=int, default=15,
                        help="Minimum matches to keep a pair (default: 15)")
    parser.add_argument("--max_angle", type=float, default=120.0,
                        help="Max angle between optical axes for pair viability (default: 120.0)")
    parser.add_argument("--device", type=str, default="auto",
                        help='Torch device: "mps", "cpu", or "cuda" (default: auto-detect)')
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for processing image sets (default: 1)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_single_image_set(image_dir, cameras, viable_pairs, device, args, track_id_offset=0):
    """Process one image directory: extract, match, filter, build tracks, triangulate.

    Returns: (tracks, points_3d, tri_stats, diagnostic, track_id_offset_after)
    """
    t0 = time.time()
    all_matches, features = extract_and_match(
        image_dir, cameras, viable_pairs, device,
        args.max_keypoints, args.resize, args.match_threshold,
    )
    t_match = time.time() - t0

    all_matches = {
        k: v for k, v in all_matches.items() if len(v["pts_a"]) >= args.min_matches
    }
    if len(all_matches) == 0:
        return {}, {}, {"total": 0, "triangulated": 0, "kept": 0}, {}, track_id_offset

    filtered_matches, diagnostic = run_reproj_diagnostic(
        cameras, all_matches, args.reproj_thresh
    )
    filtered_matches = {
        k: v for k, v in filtered_matches.items() if len(v["pts_a"]) >= args.min_matches
    }
    if len(filtered_matches) == 0:
        return {}, {}, {"total": 0, "triangulated": 0, "kept": 0}, diagnostic, track_id_offset

    tracks, track_lengths = build_tracks(cameras, filtered_matches, features)

    # Triangulate from best pairs
    pair_quality = {}
    for pair_key, pinfo in diagnostic.get("pairs", {}).items():
        parts = pair_key.split("-")
        if len(parts) == 2:
            sa, sb = min(parts[0], parts[1]), max(parts[0], parts[1])
            pair_quality[(sa, sb)] = pinfo["median_err"]
    tracks, points_3d, tri_stats = triangulate_and_filter_tracks(
        cameras, tracks, pair_quality, reproj_thresh=args.reproj_thresh
    )

    # Offset track IDs to avoid collisions between image sets
    if track_id_offset > 0:
        offset_tracks = {}
        offset_points = {}
        for serial, data in tracks.items():
            offset_tracks[serial] = {
                "ids": [tid + track_id_offset for tid in data["ids"]],
                "landmarks": data["landmarks"],
            }
        for tid_str, pt in points_3d.items():
            offset_points[str(int(tid_str) + track_id_offset)] = pt
        tracks = offset_tracks
        points_3d = offset_points

    next_offset = track_id_offset + tri_stats["total"]
    return tracks, points_3d, tri_stats, diagnostic, next_offset


def _worker_init(cameras_dict, viable_pairs_list, device_str, args_dict):
    """Initialize worker process with shared data and per-worker models."""
    global _w_cameras, _w_pairs, _w_device, _w_args
    _w_cameras = cameras_dict
    _w_pairs = viable_pairs_list
    _w_device = torch.device(device_str)
    # Reconstruct a simple namespace for args
    _w_args = argparse.Namespace(**args_dict)


def _worker_process(task):
    """Process a single image set in a worker. Returns (idx, tracks, points_3d, tri_stats)."""
    idx, image_dir, track_id_offset = task
    try:
        tracks, points_3d, tri_stats, _, _ = process_single_image_set(
            image_dir, _w_cameras, _w_pairs, _w_device, _w_args, track_id_offset
        )
        kept = tri_stats.get("kept", 0)
        print(f"  [{idx+1}] {Path(image_dir).name}: {kept} tracks", flush=True)
        return (idx, tracks, points_3d, tri_stats)
    except Exception as e:
        print(f"  [{idx+1}] {Path(image_dir).name}: ERROR {e}", flush=True)
        return (idx, {}, {}, {"total": 0, "triangulated": 0, "kept": 0})


def merge_tracks(all_tracks_list, all_points_list):
    """Merge tracks and points from multiple image sets."""
    merged_tracks = defaultdict(lambda: {"ids": [], "landmarks": []})
    merged_points = {}

    for tracks in all_tracks_list:
        for serial, data in tracks.items():
            merged_tracks[serial]["ids"].extend(data["ids"])
            merged_tracks[serial]["landmarks"].extend(data["landmarks"])

    for points in all_points_list:
        merged_points.update(points)

    return {s: dict(v) for s, v in merged_tracks.items()}, merged_points


def main():
    args = parse_args()

    image_dirs = args.image_dir  # now a list
    if args.output_dir is None:
        args.output_dir = str(Path(image_dirs[0]).parent / "correspondences")
    os.makedirs(args.output_dir, exist_ok=True)

    device = detect_device(args.device)
    print(f"Using device: {device}")
    print(f"Processing {len(image_dirs)} image set(s)")

    # Load calibration
    print("Loading calibration...")
    cameras = load_calibration(args.calib_dir)
    print(f"  Loaded {len(cameras)} cameras")
    if len(cameras) < 2:
        print("ERROR: need at least 2 cameras")
        sys.exit(1)

    # Select viable pairs (same for all image sets)
    print("Selecting viable pairs...")
    viable_pairs = select_viable_pairs(cameras, args.max_angle)
    total_pairs = len(cameras) * (len(cameras) - 1) // 2
    print(f"  {len(viable_pairs)} viable pairs from {len(cameras)}C2={total_pairs} total")

    # Process image sets (parallel or sequential)
    all_tracks_list = []
    all_points_list = []
    total_kept = 0

    # Pre-assign track ID offsets (10000 per set to avoid collisions)
    OFFSET_STRIDE = 100000
    tasks = [(i, img_dir, i * OFFSET_STRIDE) for i, img_dir in enumerate(image_dirs)]

    num_workers = min(args.workers, len(image_dirs))

    if num_workers > 1:
        import multiprocessing as mp
        print(f"\nProcessing {len(image_dirs)} image sets with {num_workers} workers...")
        t0 = time.time()

        # Serialize args for workers (Namespace isn't picklable by default)
        args_dict = vars(args)
        device_str = str(device)

        with mp.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(cameras, viable_pairs, device_str, args_dict),
        ) as pool:
            results = pool.map(_worker_process, tasks)

        for idx, tracks, points_3d, tri_stats in sorted(results, key=lambda r: r[0]):
            if tracks:
                all_tracks_list.append(tracks)
                all_points_list.append(points_3d)
                total_kept += tri_stats.get("kept", 0)

        elapsed = time.time() - t0
        print(f"  Parallel processing done in {elapsed:.1f}s ({elapsed/len(image_dirs):.1f}s per set)")
    else:
        for i, img_dir in enumerate(image_dirs):
            print(f"\n{'='*60}")
            print(f"Image set {i+1}/{len(image_dirs)}: {img_dir}")
            print(f"{'='*60}")
            tracks, points_3d, tri_stats, diagnostic, _ = process_single_image_set(
                img_dir, cameras, viable_pairs, device, args, i * OFFSET_STRIDE
            )
            if tracks:
                all_tracks_list.append(tracks)
                all_points_list.append(points_3d)
                total_kept += tri_stats.get("kept", 0)
            print(f"  {tri_stats.get('kept', 0)} tracks kept")

    if total_kept == 0:
        print("\nERROR: no tracks survived across all image sets.")
        sys.exit(1)

    # Merge all tracks
    print(f"\n{'='*60}")
    print(f"Merging {len(all_tracks_list)} image sets: {total_kept} total tracks")
    print(f"{'='*60}")
    tracks, points_3d = merge_tracks(all_tracks_list, all_points_list)
    num_tracks = total_kept

    # Check camera coverage
    connected_cams = set(tracks.keys())
    all_cams = set(cameras.keys())
    disconnected = all_cams - connected_cams
    if disconnected:
        print(f"  WARNING: {len(disconnected)} cameras have no tracks: {sorted(disconnected)}")

    per_cam_kept = {s: len(tracks[s]["ids"]) for s in sorted(tracks.keys())}
    print(f"  Per-camera: {per_cam_kept}")

    # Save results
    print("\nSaving results...")
    lm_path = save_landmarks(tracks, args.output_dir, points_3d)

    diagnostic = {}
    track_lengths = []  # not easily reconstructed after merge
    all_matches = {}  # not available after merge
    filtered_matches = {}
    stats_path = save_match_stats(
        args, cameras, viable_pairs, all_matches, filtered_matches,
        tracks, track_lengths, diagnostic, args.output_dir,
    )

    print(f"\nDone! Landmarks saved to {lm_path}")
    print(f"  Match stats saved to {stats_path}")
    print(f"  {num_tracks} tracks, {len(points_3d)} 3D points across {len(tracks)} cameras")
    print(f"  Ready for C++ bundle adjustment")


if __name__ == "__main__":
    main()
