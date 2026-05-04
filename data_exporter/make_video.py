import argparse
import json
import os

import cv2 as cv
import numpy as np

from utils import csv_reader_red2d, get_all_cams_in_labeled_folder
from keypoints import (
    load_skeleton_json_format_for_jarvis,
    skeleton_selector,
)


parser = argparse.ArgumentParser(
    description="Render an mp4 with the red 2D skeleton drawn on each frame."
)
parser.add_argument("-p", "--project_dir", type=str, required=True,
                    help="Red project dir holding project.redproj and labeled_data/.")
parser.add_argument("-c", "--cams", nargs="+", default=None,
                    help="Camera names to render. Default: all cams in latest snapshot.")
parser.add_argument("-o", "--output_dir", type=str, default=None,
                    help="Output dir. Default <project_dir>/overlay_videos/.")
parser.add_argument("-s", "--frame_start", type=int, default=0)
parser.add_argument("-e", "--frame_end", type=int, default=-1,
                    help="Inclusive end frame. -1 = last frame of each video.")
parser.add_argument("--edges", nargs="+", type=int, default=None,
                    help="Pairs of indices, e.g. --edges 0 1 1 2. "
                    "Overrides skeleton from project.redproj.")
parser.add_argument("--snapshot_dir", type=str, required=True,
                    help="Path to the timestamp label folder, relative to "
                    "--project_dir (e.g. labeled_data/2025_08_25_19_06_56). "
                    "Absolute paths also work.")
parser.add_argument("--video_folder", type=str, default=None,
                    help="Override the media_folder from project.redproj.")
parser.add_argument("--point_radius", type=int, default=4)
parser.add_argument("--line_thickness", type=int, default=2)
parser.add_argument("--alpha", type=float, default=1.0,
                    help="Contrast multiplier (1.0 = unchanged, 1.5 = brighter+punchier).")
parser.add_argument("--beta", type=float, default=0.0,
                    help="Brightness offset added to every pixel "
                    "(0 = unchanged, ~20 for slight lift).")
args = parser.parse_args()


with open(os.path.join(args.project_dir, "project.redproj")) as f:
    project = json.load(f)
video_folder = args.video_folder or project["media_folder"]

selected = (args.snapshot_dir
            if os.path.isabs(args.snapshot_dir)
            else os.path.join(args.project_dir, args.snapshot_dir))
if not os.path.isdir(selected):
    raise SystemExit(f"--snapshot_dir not a directory: {selected}")
print(f"Loading keypoints from: {selected}")

cams = args.cams or get_all_cams_in_labeled_folder(selected)

if args.edges:
    edges = list(zip(args.edges[::2], args.edges[1::2]))
else:
    if project.get("load_skeleton_from_json"):
        keypoint_names, jarvis_skeleton, _ = (
            load_skeleton_json_format_for_jarvis(project["skeleton_file"])
        )
    else:
        keypoint_names, jarvis_skeleton, _ = (
            skeleton_selector[project["skeleton_name"]]()
        )
    name_to_idx = {n: i for i, n in enumerate(keypoint_names)}
    edges = [
        (name_to_idx[j["keypointA"]], name_to_idx[j["keypointB"]])
        for j in jarvis_skeleton
        if j["keypointA"] in name_to_idx and j["keypointB"] in name_to_idx
    ]
    print(f"Loaded {len(edges)} skeleton edges from project.redproj.")

out_dir = args.output_dir or os.path.join(args.project_dir, "overlay_videos")
os.makedirs(out_dir, exist_ok=True)


def hsv_colors(n):
    cols = []
    for i in range(n):
        hsv = np.array([[[int(i / max(n, 1) * 179), 255, 255]]], dtype=np.uint8)
        b, g, r = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)[0, 0]
        cols.append((int(b), int(g), int(r)))
    return cols


def to_xy(pt):
    x, y = float(pt[0]), float(pt[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return int(round(x)), int(round(y))


for cam in cams:
    csv_path = os.path.join(selected, f"{cam}.csv")
    video_path = os.path.join(video_folder, f"{cam}.mp4")
    if not (os.path.isfile(csv_path) and os.path.isfile(video_path)):
        print(f"[{cam}] missing csv or mp4; skipping.")
        continue

    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    labels = csv_reader_red2d(csv_path, img_height=h)
    n_kps = max((v.shape[0] for v in labels.values()), default=1)
    colors = hsv_colors(n_kps)

    end = total - 1 if args.frame_end < 0 else args.frame_end
    start = max(0, args.frame_start)
    out_path = os.path.join(out_dir, f"{cam}_f{start}-{end}_overlay.mp4")
    writer = cv.VideoWriter(
        out_path, cv.VideoWriter_fourcc(*"avc1"), fps, (w, h)
    )

    cap.set(cv.CAP_PROP_POS_FRAMES, start)
    n_written = 0
    for frame_number in range(start, end + 1):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if args.alpha != 1.0 or args.beta != 0.0:
            frame = cv.convertScaleAbs(frame, alpha=args.alpha, beta=args.beta)
        kps = labels.get(frame_number)
        if kps is not None:
            for i, pt in enumerate(kps):
                xy = to_xy(pt)
                if xy is not None:
                    cv.circle(frame, xy, args.point_radius,
                              colors[i % len(colors)], -1, lineType=cv.LINE_AA)
            for a, b in edges:
                if a < len(kps) and b < len(kps):
                    a_xy, b_xy = to_xy(kps[a]), to_xy(kps[b])
                    if a_xy and b_xy:
                        cv.line(frame, a_xy, b_xy, colors[b % len(colors)],
                                args.line_thickness, lineType=cv.LINE_AA)
        writer.write(frame)
        n_written += 1

    writer.release()
    cap.release()
    print(f"[{cam}] wrote {n_written} frames -> {out_path}")
