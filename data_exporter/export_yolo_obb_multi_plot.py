#!/usr/bin/env python3
"""
Export YOLO OBB (Oriented Bounding Box) Dataset — Multi-CSV & Multi-Folder with Dated Videos

Features
- Accept multiple label roots: repeat -i (e.g., -i cylinder -i objectSmallTable)
- Accept multiple VIDEO roots: repeat -v (e.g., -v /data/2025_09_20 -v /data/2025_09_21)
- Each label CSV is matched to a video by (camera_id, date_tag), where date_tag is taken from
  a folder in its path named like YYYY_MM_DD_hh_mm_ss. The same logic is applied to videos.
  This lets you have duplicate camera filenames living under different date folders.
- Fallback rules if an exact (camera_id, date_tag) video is not found:
  1) use the closest-by-date video for that camera_id across all -v roots
  2) otherwise skip with a warning.

Also supports:
- Explicit CSVs via --csv_files (repeat or comma-separated)
- Explicit videos via --video_files (repeat or comma-separated). These are matched by date_tag.

Output
- Ultralytics-style dataset structure: {train,val,test}/{images,labels}
- labels: YOLO OBB lines:  class x1 y1 x2 y2 x3 y3 x4 y4  (normalized [0,1])
- data.yaml with class names.
"""

import argparse
import cv2
import os
import random
import re
import yaml
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ----------------------------
# Date tag helpers
# ----------------------------

_TS_RE = re.compile(r'^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$')

def find_timestamp_in_path(path: str) -> Optional[str]:
    """
    Walk up the directory tree of 'path' to find the first folder that looks like
    YYYY_MM_DD_hh_mm_ss. Returns that folder name (the tag) or None.
    """
    cur = os.path.abspath(path)
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    while True:
        base = os.path.basename(cur)
        if _TS_RE.match(base):
            return base
        nxt = os.path.dirname(cur)
        if nxt == cur:
            return None
        cur = nxt

def parse_timestamp_sort_key(tag: Optional[str]) -> Tuple:
    """
    Convert 'YYYY_MM_DD_hh_mm_ss' to a tuple for sorting. None sorts last.
    """
    if not tag or not _TS_RE.match(tag):
        return (9999, 99, 99, 99, 99, 99)
    y, m, d, H, M, S = (int(x) for x in tag.split('_'))
    return (y, m, d, H, M, S)

# ----------------------------
# Image resize / letterbox
# ----------------------------

def resize_image_for_yolo(image, target_size=640):
    h, w = image.shape[:2]
    if max(h, w) <= 0:
        scale = 1.0
        new_w, new_h = w, h
        resized = image.copy()
    else:
        scale = target_size / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas, scale, x_offset, y_offset

def adjust_points_for_resize(points_xy, scale, x_offset, y_offset):
    out = []
    for (x, y) in points_xy:
        x_new = x * scale + x_offset
        y_new = y * scale + y_offset
        out.append((float(x_new), float(y_new)))
    return out

def normalize_points(points_xy, img_w, img_h, flip_x=False, flip_y=True):
    out = []
    for (x, y) in points_xy:
        if flip_x:
            x = img_w - x
        if flip_y:
            y = img_h - y
        out.append((x / img_w, y / img_h))
    return out

def order_corners_consistently(pts4):
    """Order 4 points clockwise starting from top-left (approx)."""
    P = np.array(pts4, dtype=np.float32)
    if P.shape != (4, 2):
        return pts4
    c = P.mean(axis=0)
    ang = np.arctan2(P[:,1] - c[1], P[:,0] - c[0])  # [-pi, pi]
    idx = np.argsort(ang)  # CCW
    P = P[idx]
    # rotate so first ~ top-left (min y then min x)
    start = np.lexsort((P[:,0], P[:,1]))[0]
    P = np.roll(P, -start, axis=0)
    return P.tolist()

# ----------------------------
# CSV parsing / merging
# ----------------------------

def parse_obb_csv(csv_path: str) -> Dict[str, List[dict]]:
    """Return dict: frame_id -> list of {'class_id': int, 'points': [(x1,y1),...,(x4,y4)]}"""
    labels = {}
    if not os.path.exists(csv_path):
        return labels
    with open(csv_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return labels
    # Drop a possible title line if header doesn't start with 'frame'
    if not lines[0].lower().startswith('frame'):
        lines = lines[1:]
        if not lines:
            return labels
    # header = lines[0].split(',')  # not used directly
    for row in lines[1:]:
        parts = row.split(',')
        if len(parts) < 11:
            continue
        frame_id = parts[0].strip()
        # class id in col[2] by convention
        try:
            class_id = int(float(parts[2]))
        except:
            try:
                class_id = int(parts[2])
            except:
                class_id = 0
        try:
            x1, y1 = float(parts[3]), float(parts[4])
            x2, y2 = float(parts[5]), float(parts[6])
            x3, y3 = float(parts[7]), float(parts[8])
            x4, y4 = float(parts[9]), float(parts[10])
        except:
            continue
        labels.setdefault(frame_id, []).append(
            {'class_id': class_id, 'points': [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]}
        )
    return labels

_CAM_RE = re.compile(r'^(.*?)(?:_obb(?:_.+)?)\.csv$', re.IGNORECASE)

def camera_id_from_csv(filename: str) -> str:
    m = _CAM_RE.match(filename)
    if m:
        return m.group(1)
    return os.path.splitext(filename)[0]

def collect_csvs_with_dates(label_roots: List[str], explicit_csvs: List[str]) -> List[Tuple[str, Optional[str], str]]:
    """
    Return list of tuples: (csv_path, date_tag, camera_id)
    - Scans each label_root for *_obb*.csv (in latest timestamped child if present; otherwise the root itself).
    - Adds any explicit csvs provided.
    """
    csvs = []

    def _add_csv(path):
        if not (os.path.isfile(path) and path.lower().endswith('.csv') and '_obb' in os.path.basename(path).lower()):
            return
        cam = camera_id_from_csv(os.path.basename(path))
        date_tag = find_timestamp_in_path(path)
        csvs.append((path, date_tag, cam))

    # from roots
    for root in label_roots:
        if not os.path.isdir(root):
            continue
        children = [os.path.join(root, d) for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d)) and _TS_RE.match(d)]
        search_dirs = [children[-1]] if children else [root]
        for d in search_dirs:
            for fname in os.listdir(d):
                if fname.lower().endswith('.csv') and '_obb' in fname.lower():
                    _add_csv(os.path.join(d, fname))

    # explicit
    for p in explicit_csvs:
        for part in [pp.strip() for pp in p.split(',') if pp.strip()]:
            _add_csv(part)

    # de-dup
    seen = set()
    out = []
    for row in csvs:
        if row[0] in seen:
            continue
        seen.add(row[0])
        out.append(row)
    return out

# ----------------------------
# Video discovery & matching
# ----------------------------

def parse_video_files(video_roots: List[str], explicit_video_files: List[str]) -> Dict[Tuple[str, Optional[str]], str]:
    """
    Discover videos under given roots (recursively) and explicit files.
    Returns a mapping: (camera_id, date_tag) -> video_path
    Where date_tag is the nearest YYYY_MM_DD_hh_mm_ss in the path (or None).
    """
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    mapping = {}

    def _add_video(path):
        if not os.path.isfile(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in video_exts:
            return
        cam = os.path.splitext(os.path.basename(path))[0]
        tag = find_timestamp_in_path(path)
        mapping[(cam, tag)] = path

    # walk roots
    for root in video_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                _add_video(os.path.join(dirpath, fname))

    # explicit
    for v in explicit_video_files:
        for part in [pp.strip() for pp in v.split(',') if pp.strip()]:
            _add_video(part)

    return mapping

def best_video_for(cam_id: str, desired_tag: Optional[str], video_map: Dict[Tuple[str, Optional[str]], str]) -> Optional[str]:
    """
    Choose the best-matching video path for (cam_id, desired_tag).
    Priority: exact tag -> nearest by date -> latest for cam.
    """
    # exact match
    if (cam_id, desired_tag) in video_map:
        return video_map[(cam_id, desired_tag)]

    # collect all (cam_id, tag) for this cam
    candidates = [(tag, path) for (c, tag), path in video_map.items() if c == cam_id]
    if not candidates:
        return None

    # if we have a desired tag, choose the closest by absolute time distance
    if desired_tag and _TS_RE.match(desired_tag):
        desired_key = parse_timestamp_sort_key(desired_tag)
        candidates_sorted = sorted(
            candidates,
            key=lambda tp: tuple(abs(a - b) for a, b in zip(parse_timestamp_sort_key(tp[0]), desired_key))
        )
        return candidates_sorted[0][1]

    # otherwise return the latest-dated video
    candidates_sorted = sorted(candidates, key=lambda tp: parse_timestamp_sort_key(tp[0]))
    return candidates_sorted[-1][1]

# ----------------------------
# Dataset helpers
# ----------------------------

def create_dataset_structure(output_dir):
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    return splits

def split_data(keys: List[str], train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    rng = random.Random(seed)
    arr = list(keys)
    rng.shuffle(arr)
    n = len(arr)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = arr[:n_train]
    val = arr[n_train:n_train+n_val]
    test = arr[n_train+n_val:]
    return {'train': train, 'val': val, 'test': test}

def load_class_names(class_file):
    if class_file and os.path.exists(class_file):
        with open(class_file, 'r') as f:
            return [ln.strip() for ln in f if ln.strip()]
    return ['object']

def create_data_yaml(output_dir, class_names):
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path

# ----------------------------
# Frame extraction
# ----------------------------

def extract_frame_opencv(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_number < 0 or (total and frame_number >= total):
        print(f"Warning: frame {frame_number} out of 0..{max(total-1,0)} for {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number, 0))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ----------------------------
# Visualization (test split)
# ----------------------------

def visualize_test_set(output_dir: str, image_size: int, line_thickness: int = 2):
    """
    For every label in test/labels, draw the oriented bbox on top of test/images
    and save to test/vis with the same filename.
    """
    labels_dir = os.path.join(output_dir, 'test', 'labels')
    images_dir = os.path.join(output_dir, 'test', 'images')
    vis_dir    = os.path.join(output_dir, 'test', 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    if not os.path.isdir(labels_dir):
        print(f"[viz] No labels found at {labels_dir}; skipping test visualization.")
        return

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    if not label_files:
        print(f"[viz] No label files in {labels_dir}; skipping.")
        return

    total = 0
    for lbl_name in label_files:
        base = os.path.splitext(lbl_name)[0]
        img_path = os.path.join(images_dir, base + '.jpg')
        lbl_path = os.path.join(labels_dir, lbl_name)
        if not os.path.isfile(img_path):
            print(f"[viz] Missing image for {lbl_name}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[viz] Failed to read {img_path}")
            continue

        # Parse YOLO-OBB label lines: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
        with open(lbl_path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 1 + 8:
                # some exporters may include more than 1 object per line; handle common case (one per line)
                try:
                    cls = int(float(parts[0]))
                    coords = [float(x) for x in parts[1:9]]
                except:
                    continue
            else:
                cls = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]

            pts = []
            for i in range(0, 8, 2):
                x = int(round(coords[i]   * image_size))
                y = int(round(coords[i+1] * image_size))
                pts.append([x, y])
            pts = np.array(pts, dtype=np.int32)

            # Draw polygon
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=line_thickness)
            # Put class id near the first corner
            cv2.putText(img, str(cls), (pts[0][0], pts[0][1]-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        out_path = os.path.join(vis_dir, base + '.jpg')
        cv2.imwrite(out_path, img)
        total += 1

    print(f"[viz] Wrote {total} visualization(s) to {vis_dir}")

# ----------------------------
# Main export
# ----------------------------

def export_yolo_obb_dataset(csv_tuples: List[Tuple[str, Optional[str], str]], video_map, output_dir: str,
                            class_file=None, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                            image_size=640, seed=42, flip_x=False, flip_y=True, dedup=True):
    # class names
    class_names = load_class_names(class_file)
    print(f"Using class names: {class_names}")

    # Merge labels per (camera_id, date_tag)
    merged_by_cam_tag: Dict[Tuple[str, Optional[str]], Dict[str, List[dict]]] = {}
    for (csv_path, date_tag, cam_id) in csv_tuples:
        labels = parse_obb_csv(csv_path)
        key = (cam_id, date_tag)
        if key not in merged_by_cam_tag:
            merged_by_cam_tag[key] = {}
        for frame_id, items in labels.items():
            merged_by_cam_tag[key].setdefault(frame_id, []).extend(items)

    # Optional dedup
    if dedup:
        for key, frames in merged_by_cam_tag.items():
            for frame_id, items in list(frames.items()):
                seen = set()
                out = []
                for it in items:
                    cls = it['class_id']
                    pts = it['points']
                    k = (cls,) + tuple(round(v / 1e-4) for xy in pts for v in xy)
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(it)
                frames[frame_id] = out

    # Build master list of labeled frames with matched videos
    all_labeled = {}
    skipped = 0
    for (cam_id, tag), frames in merged_by_cam_tag.items():
        vpath = best_video_for(cam_id, tag, video_map)
        if not vpath:
            print(f"Warning: No video for camera {cam_id} (desired tag {tag}); skipping {len(frames)} frames.")
            skipped += len(frames)
            continue
        for frame_id, items in frames.items():
            key = f"{cam_id}_{tag or 'NA'}_{frame_id}"
            all_labeled[key] = {'camera': cam_id, 'date_tag': tag, 'frame_id': frame_id,
                                'video_path': vpath, 'labels': items}

    if not all_labeled:
        raise ValueError("No labeled frames with matching videos were found.")

    print(f"Matched {len(all_labeled)} labeled frames to videos. Skipped {skipped}.")

    # prepare dirs
    _ = create_dataset_structure(output_dir)

    # split
    splits = split_data(list(all_labeled.keys()), train_ratio, val_ratio, test_ratio, seed=seed)
    print(f"Split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # export
    for split_name, keys in splits.items():
        print(f"\nExporting {split_name}...")
        images_dir = os.path.join(output_dir, split_name, 'images')
        labels_dir = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for i, frame_key in enumerate(keys):
            meta = all_labeled[frame_key]
            frame_id = meta['frame_id']
            vpath = meta['video_path']
            labels = meta['labels']

            try:
                frame_number = int(frame_id)
            except:
                frame_number = i

            frame = extract_frame_opencv(vpath, frame_number)
            if frame is None:
                print(f"Failed to extract frame {frame_key}")
                continue

            letterboxed, scale, xoff, yoff = resize_image_for_yolo(frame, image_size)

            img_filename = f"{frame_key}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            cv2.imwrite(img_path, letterboxed)

            label_filename = f"{frame_key}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            with open(label_path, 'w') as f:
                for lab in labels:
                    cls = lab['class_id']
                    pts = order_corners_consistently(lab['points'])
                    pts_resized = adjust_points_for_resize(pts, scale, xoff, yoff)
                    pts_norm = normalize_points(pts_resized, image_size, image_size,
                                                flip_x=flip_x, flip_y=flip_y)
                    flat = [v for xy in pts_norm for v in xy]
                    f.write(f"{cls} " + " ".join(f"{v:.6f}" for v in flat) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(keys)} frames")

    create_data_yaml(output_dir, class_names)

    readme = os.path.join(output_dir, 'README.md')
    with open(readme, 'w') as f:
        f.write("# YOLO OBB Dataset\n\n")
        f.write("Generated by export_yolo_obb_multi.py\n\n")
        f.write("## Label format\n")
        f.write("Each line: `class x1 y1 x2 y2 x3 y3 x4 y4` (normalized to [0,1]).\n\n")
        f.write("## Axis flips\n")
        f.write("- `--flip_y` (default true) flips the vertical axis (useful if origin is bottom-left).\n")
        f.write("- `--flip_x` (default false) flips the horizontal axis.\n\n")
        f.write("## Multiple sources & dated videos\n")
        f.write("- Pass `-i` multiple times for label roots; script detects the nearest YYYY_MM_DD_hh_mm_ss tag in each CSV path.\n")
        f.write("- Pass `-v` multiple times for video roots; videos are matched by camera_id and the same date tag.\n")
        f.write("- If no exact match: chooses closest-by-date video for that camera; otherwise latest.\n")

    print("\nDone!")
    print(f"Output: {output_dir}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower().strip()
    if s in ('true','1','yes','y','on'):
        return True
    if s in ('false','0','no','n','off'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")

def parse_list_args(values: List[str]) -> List[str]:
    out = []
    for v in values or []:
        out.extend([p.strip() for p in v.split(',') if p.strip()])
    return out

def main():
    ap = argparse.ArgumentParser(description="Export YOLO OBB Dataset (Multi-CSV & Multi-Folder with Dated Videos)")
    ap.add_argument('-i','--label_dir', action='append', default=[],
                    help='Label root directory (repeatable). Script finds *_obb*.csv (latest timestamped child or the dir itself).')
    ap.add_argument('--csv_files', action='append', default=[],
                    help='Explicit *_obb*.csv files (repeatable or comma-separated).')
    ap.add_argument('-v','--video_dir', action='append', default=[],
                    help='Video root directory (repeatable). Script recursively indexes videos and matches by camera_id + date tag.')
    ap.add_argument('--video_files', action='append', default=[],
                    help='Explicit video file paths (repeatable or comma-separated).')
    ap.add_argument('-o','--output_dir', required=True, type=str, help='Output directory for dataset')
    ap.add_argument('-c','--class_file', type=str, default=None, help='Text file with class names (one per line)')
    ap.add_argument('--image_size', type=int, default=640, help='Output square size (default=640)')
    ap.add_argument('--train_ratio', type=float, default=0.7)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--flip_x', type=str2bool, default=False)
    ap.add_argument('--flip_y', type=str2bool, default=True)
    ap.add_argument('--dedup', type=str2bool, default=True, help='Deduplicate near-identical polygons')
    ap.add_argument('--viz_test', type=str2bool, default=True, help='Render OBB overlays for the test split')
    ap.add_argument('--viz_line_thickness', type=int, default=2, help='Line thickness for OBB visualization')
    args = ap.parse_args()

    csv_list = collect_csvs_with_dates(parse_list_args(args.label_dir), parse_list_args(args.csv_files))
    if not csv_list:
        raise SystemExit("No *_obb*.csv files found. Check your -i roots or --csv_files.")

    video_map = parse_video_files(parse_list_args(args.video_dir), parse_list_args(args.video_files))

    export_yolo_obb_dataset(
        csv_tuples=csv_list,
        video_map=video_map,
        output_dir=args.output_dir,
        class_file=args.class_file,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        image_size=args.image_size, seed=args.seed,
        flip_x=args.flip_x, flip_y=args.flip_y, dedup=args.dedup
    )

    # Optional: visualize test split
    if args.viz_test:
        try:
            visualize_test_set(args.output_dir, args.image_size, line_thickness=args.viz_line_thickness)
        except Exception as e:
            print(f"[viz] Failed: {e}")

if __name__ == '__main__':
    main()
