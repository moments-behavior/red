#!/usr/bin/env python3
"""Generate a publication-quality 3D SVG diagram of a calibration result.

Reads camera YAML files and ba_points.json from a calibration output folder,
plots camera frustums and 3D landmark points, and exports as SVG + PDF.

Usage:
    python plot_calibration_3d.py /path/to/calibration_folder [--output diagram.svg]

Requires: numpy, matplotlib, pyyaml (or opencv-python for YAML parsing)
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re


def read_yaml_matrix(path, key):
    """Read a matrix from an OpenCV-style YAML file (simple parser)."""
    with open(path, 'r') as f:
        content = f.read()

    # Find the key
    pattern = rf'{key}:\s*!!opencv-matrix\s*rows:\s*(\d+)\s*cols:\s*(\d+)\s*dt:\s*d\s*data:\s*\[([\s\S]*?)\]'
    match = re.search(pattern, content)
    if not match:
        return None

    rows, cols = int(match.group(1)), int(match.group(2))
    data_str = match.group(3)
    values = [float(x.strip()) for x in data_str.split(',') if x.strip()]
    return np.array(values).reshape(rows, cols)


def load_cameras(folder):
    """Load all Cam*.yaml files from a calibration folder."""
    cameras = {}
    yaml_files = sorted(glob.glob(os.path.join(folder, 'Cam*.yaml')))
    for yf in yaml_files:
        name = os.path.basename(yf).replace('Cam', '').replace('.yaml', '')
        R = read_yaml_matrix(yf, 'rc_ext')
        t = read_yaml_matrix(yf, 'tc_ext')
        K = read_yaml_matrix(yf, 'camera_matrix')
        if R is not None and t is not None and K is not None:
            cameras[name] = {'R': R, 't': t.flatten(), 'K': K}
    return cameras


def load_points(folder):
    """Load 3D points from ba_points.json."""
    pts_path = os.path.join(folder, 'summary_data', 'bundle_adjustment', 'ba_points.json')
    if not os.path.exists(pts_path):
        return None
    with open(pts_path) as f:
        data = json.load(f)
    pts = np.array([data[k] for k in sorted(data.keys(), key=int)])
    return pts


def load_metrics(folder):
    """Load per-camera metrics from calibration_data.json."""
    db_path = os.path.join(folder, 'calibration_data.json')
    if not os.path.exists(db_path):
        return {}, {}
    with open(db_path) as f:
        j = json.load(f)

    # Per-camera metrics (intrinsic reproj, detection count)
    saved = j.get('per_camera_metrics', {})

    # Count observations from board_poses
    board_poses = j.get('board_poses', {})

    # Load landmarks to count observations
    lm_path = os.path.join(folder, 'summary_data', 'landmarks.json')
    obs_counts = {}
    if os.path.exists(lm_path):
        with open(lm_path) as f:
            lm = json.load(f)
        for cam, data in lm.items():
            obs_counts[cam] = len(data.get('ids', []))

    metrics = {}
    for cam_name, m in saved.items():
        metrics[cam_name] = {
            'intrinsic_reproj': m.get('intrinsic_reproj', 0),
            'detection_count': m.get('detection_count', 0),
            'observations': obs_counts.get(cam_name, 0),
        }

    # Also compute BA reproj from landmarks + points if available
    # (already computed in the pipeline, stored in board_poses reproj)
    for cam_name in board_poses:
        if cam_name not in metrics:
            metrics[cam_name] = {
                'intrinsic_reproj': 0,
                'detection_count': len(board_poses[cam_name]),
                'observations': obs_counts.get(cam_name, 0),
            }

    return metrics


def camera_center(R, t):
    """Compute camera center in world coordinates: C = -R^T * t"""
    return -R.T @ t


def frustum_corners(R, t, K, w=3208, h=2200, depth=150):
    """Compute frustum corner points in world coordinates."""
    C = camera_center(R, t)
    Rt = R.T
    Kinv = np.linalg.inv(K)
    corners_img = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    corners_3d = []
    for cx, cy in corners_img:
        ray = Kinv @ np.array([cx, cy, 1.0])
        ray = ray / np.linalg.norm(ray) * depth
        pt = C + Rt @ ray
        corners_3d.append(pt)
    return C, np.array(corners_3d)


def reproj_color(err, vmin=0.2, vmax=1.0):
    """Map reprojection error to green→yellow→red color."""
    t = np.clip((err - vmin) / (vmax - vmin), 0, 1)
    r = min(1.0, 2 * t)
    g = min(1.0, 2 * (1 - t))
    return (r, g, 0.2, 1.0)


def compute_ba_reproj(folder, cam_names_sorted):
    """Compute per-camera mean BA reproj from residuals in calibration_data.json."""
    db_path = os.path.join(folder, 'calibration_data.json')
    if not os.path.exists(db_path):
        return {}
    with open(db_path) as f:
        j = json.load(f)
    res = j.get('residuals', {})
    if not res or 'camera_idx' not in res:
        return {}
    cam_idxs = res['camera_idx']
    errors = res['error']
    per_cam = {}
    for ci, e in zip(cam_idxs, errors):
        if ci not in per_cam:
            per_cam[ci] = []
        per_cam[ci].append(e)
    result = {}
    for ci, errs in per_cam.items():
        if ci < len(cam_names_sorted):
            result[cam_names_sorted[ci]] = {
                'mean': np.mean(errs),
                'median': np.median(errs),
                'count': len(errs),
            }
    return result


def plot_calibration(cameras, points, output_path, title=None,
                     frustum_depth=150, elev=25, azim=-60,
                     figsize=(16, 12), dpi=150, calib_folder=None):
    """Create a 3D plot with camera info boxes around the periphery."""

    bg_color = '#1a1a2e'
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d', facecolor=bg_color)

    cam_names_sorted = sorted(cameras.keys())
    n_cams = len(cam_names_sorted)

    # Load metrics
    metrics = {}
    ba_reproj = {}
    if calib_folder:
        metrics = load_metrics(calib_folder)
        ba_reproj = compute_ba_reproj(calib_folder, cam_names_sorted)

    # Compute camera centers for projection
    centers_3d = {}
    for name in cam_names_sorted:
        cam = cameras[name]
        centers_3d[name] = camera_center(cam['R'], cam['t'])

    # Plot camera frustums colored by BA reproj error
    for i, name in enumerate(cam_names_sorted):
        cam = cameras[name]
        C, corners = frustum_corners(cam['R'], cam['t'], cam['K'],
                                      depth=frustum_depth)

        ba = ba_reproj.get(name, {})
        err = ba.get('mean', 0.3)
        color = reproj_color(err)

        # Frustum edges (center to each corner)
        for corner in corners:
            ax.plot3D(*zip(C, corner), color=color, linewidth=1.4, alpha=0.85)

        # Image plane edges
        for j in range(4):
            c1, c2 = corners[j], corners[(j + 1) % 4]
            ax.plot3D(*zip(c1, c2), color=color, linewidth=1.8, alpha=0.95)

        # Frustum face
        verts = [list(zip(corners[:, 0], corners[:, 1], corners[:, 2]))]
        face = Poly3DCollection(verts, alpha=0.1, facecolor=color,
                                 edgecolor='none')
        ax.add_collection3d(face)

        # Camera center marker + name label
        ax.scatter(*C, color=color, s=60, zorder=5, edgecolors='white',
                   linewidth=0.4)
        ax.text(C[0], C[1], C[2], f'  {name}', fontsize=7,
                color='white', fontweight='bold', ha='left', va='bottom')

    # Plot 3D points
    if points is not None and len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=3.5, c='#6699ff', alpha=0.55, edgecolors='none',
                   rasterized=True)

    # View angle and limits
    ax.view_init(elev=elev, azim=azim)

    all_pts = list(centers_3d.values())
    if points is not None:
        all_pts.extend(points.tolist())
    all_pts = np.array(all_pts)
    mid = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.15
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', color='white', y=0.97)

    # Finalize layout BEFORE computing 2D projections.
    # The 3D axes occupy the center; info boxes go in the margins.
    plt.subplots_adjust(left=0.18, right=0.82, top=0.88, bottom=0.12)
    fig.canvas.draw()

    # Project 3D camera centers to 2D figure coordinates
    from mpl_toolkits.mplot3d import proj3d
    projected = {}
    for name in cam_names_sorted:
        C = centers_3d[name]
        x2d, y2d, _ = proj3d.proj_transform(C[0], C[1], C[2], ax.get_proj())
        x_disp, y_disp = ax.transData.transform((x2d, y2d))
        x_fig, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))
        projected[name] = (x_fig, y_fig)

    # Box dimensions in figure coords
    box_w, box_h = 0.135, 0.065
    border = 0.02
    top_limit = 0.92

    # Strategy: start each box at its camera's projected position,
    # then push outward just enough to avoid the central scene area.
    # This minimizes line length — each box stays near its camera.
    cx_fig = np.mean([p[0] for p in projected.values()])
    cy_fig = np.mean([p[1] for p in projected.values()])

    label_positions = {}
    for name in cam_names_sorted:
        px, py = projected[name]
        # Push outward from center by a fixed amount
        dx, dy = px - cx_fig, py - cy_fig
        dist = max(np.sqrt(dx*dx + dy*dy), 0.01)
        push = 0.20  # push distance in figure coords
        bx = px + dx / dist * push - box_w / 2
        by = py + dy / dist * push - box_h / 2
        # Clamp to figure bounds
        bx = np.clip(bx, border, 1.0 - border - box_w)
        by = np.clip(by, border, top_limit - box_h)
        label_positions[name] = [bx, by]

    # Resolve overlaps: push boxes apart, preferring to move
    # in the direction away from center (preserving proximity)
    names_list = list(label_positions.keys())
    for _ in range(500):
        moved = False
        for i in range(len(names_list)):
            ni = names_list[i]
            xi, yi = label_positions[ni]
            for j in range(i + 1, len(names_list)):
                nj = names_list[j]
                xj, yj = label_positions[nj]
                ox = box_w + 0.005 - abs(xi - xj)  # small gap
                oy = box_h + 0.005 - abs(yi - yj)
                if ox > 0 and oy > 0:
                    # Push apart along the shorter overlap axis
                    if ox < oy:
                        nudge = ox / 2 + 0.002
                        if xi < xj:
                            label_positions[ni][0] -= nudge
                            label_positions[nj][0] += nudge
                        else:
                            label_positions[ni][0] += nudge
                            label_positions[nj][0] -= nudge
                    else:
                        nudge = oy / 2 + 0.002
                        if yi < yj:
                            label_positions[ni][1] -= nudge
                            label_positions[nj][1] += nudge
                        else:
                            label_positions[ni][1] += nudge
                            label_positions[nj][1] -= nudge
                    moved = True
            # Clamp
            label_positions[ni][0] = np.clip(
                label_positions[ni][0], border, 1.0 - border - box_w)
            label_positions[ni][1] = np.clip(
                label_positions[ni][1], border, top_limit - box_h)
        if not moved:
            break

    # Draw info boxes and connection lines
    for name in cam_names_sorted:
        bx, by = label_positions[name]

        ba = ba_reproj.get(name, {})
        m = metrics.get(name, {})
        err = ba.get('mean', 0)
        color = reproj_color(err) if err > 0 else (0.3, 0.9, 0.3, 1.0)

        # Info text
        lines = [f'{name}']
        if ba:
            lines.append(f'BA: {ba["mean"]:.3f} px ({ba["count"]} obs)')
        if m.get('intrinsic_reproj', 0) > 0:
            lines.append(f'Intr: {m["intrinsic_reproj"]:.3f} px')
        if m.get('detection_count', 0) > 0:
            lines.append(f'Dets: {m["detection_count"]}')
        text = '\n'.join(lines)

        # Text box (anchored at top-left corner)
        fig.text(bx, by + box_h, text, fontsize=10, color='white',
                 fontfamily='monospace', fontweight='bold',
                 ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor=(*color[:3], 0.25),
                           edgecolor=(*color[:3], 0.7),
                           linewidth=1.2),
                 transform=fig.transFigure, zorder=10)

    # Save
    svg_path = output_path
    pdf_path = output_path.replace('.svg', '.pdf')
    png_path = output_path.replace('.svg', '.png')

    fig.savefig(svg_path, format='svg', bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=200,
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

    print(f"Saved: {svg_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"  {len(cameras)} cameras, {len(points) if points is not None else 0} points")


def find_calibration_folder(base_folder):
    """Find the actual data folder (may be in a timestamped subfolder)."""
    if os.path.exists(os.path.join(base_folder, 'calibration_data.json')):
        return base_folder
    # Check timestamped subfolders
    subdirs = sorted([d for d in os.listdir(base_folder)
                      if os.path.isdir(os.path.join(base_folder, d))
                      and len(d) >= 10 and d[4] == '_' and d[7] == '_'],
                     reverse=True)
    if subdirs:
        return os.path.join(base_folder, subdirs[0])
    return base_folder


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D SVG diagram of calibration result')
    parser.add_argument('folder', help='Calibration output folder')
    parser.add_argument('--output', '-o', default=None,
                        help='Output SVG path (default: <folder>/calibration_3d.svg)')
    parser.add_argument('--title', '-t', default=None,
                        help='Plot title')
    parser.add_argument('--depth', type=float, default=150,
                        help='Frustum depth in mm (default: 150)')
    parser.add_argument('--elev', type=float, default=25,
                        help='Elevation angle (default: 25)')
    parser.add_argument('--azim', type=float, default=-60,
                        help='Azimuth angle (default: -60)')
    args = parser.parse_args()

    folder = find_calibration_folder(args.folder)
    print(f"Loading from: {folder}")

    cameras = load_cameras(folder)
    if not cameras:
        print(f"Error: No Cam*.yaml files found in {folder}")
        sys.exit(1)

    points = load_points(folder)

    output = args.output or os.path.join(folder, 'calibration_3d.svg')
    title = args.title or f'Calibration Result ({len(cameras)} cameras, {len(points) if points is not None else 0} points)'

    plot_calibration(cameras, points, output, title=title,
                     frustum_depth=args.depth, elev=args.elev, azim=args.azim,
                     calib_folder=folder)


if __name__ == '__main__':
    main()
