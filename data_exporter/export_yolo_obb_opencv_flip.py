#!/usr/bin/env python3
"""
Export YOLO OBB (Oriented Bounding Box) Dataset — OpenCV-only (with axis flip options)

Adds support for flipping axes before normalization to handle datasets whose source
coordinates use a different origin (e.g., bottom-left for Y).

Usage:
  python export_yolo_obb_opencv_flip.py \
    -i /path/to/labels_root \
    -v /path/to/videos \
    -o /path/to/output_dataset \
    -c /path/to/class_names.txt \
    --image_size 640 \
    --flip_y true \
    --flip_x false

Notes:
- If your previous axis-aligned exporter did `yc = 1 - yc`, your annotations likely
  have Y origin at the bottom. In that case, set `--flip_y true` (default).
- `--flip_x` is available if your X origin is right-aligned (rare). Default false.
- Label format: one line per object -> `class x1 y1 x2 y2 x3 y3 x4 y4` (normalized).
"""

import argparse
import cv2
import os
import random
import re
import yaml
import numpy as np

# ----------------------------
# Image resize / letterbox
# ----------------------------

def resize_image_for_yolo(image, target_size=640):
    h, w = image.shape[:2]
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
    """
    Normalize points to [0,1] after optionally flipping axes in pixel space.
    flip_y=True matches prior pipelines that used bottom-origin Y (i.e., did 1 - y_norm).
    """
    out = []
    for (x, y) in points_xy:
        if flip_x:
            x = img_w - x
        if flip_y:
            y = img_h - y
        out.append((x / img_w, y / img_h))
    return out
def order_corners_consistently(pts4):
    """
    pts4: list[(x,y)] in pixel coords
    returns: list[(x,y)] ordered clockwise starting at top-left
    """
    P = np.array(pts4, dtype=np.float32)
    c = P.mean(axis=0)
    ang = np.arctan2(P[:,1] - c[1], P[:,0] - c[0])        # angle around centroid
    idx = np.argsort(ang)                                 # CCW
    P = P[idx]                                            # reorder CCW
    # rotate so first is top-left (min y, then min x)
    start = np.lexsort((P[:,0], P[:,1]))[0]
    P = np.roll(P, -start, axis=0)
    # flip to clockwise if needed (swap order except first)
    # (ultralytics doesn’t mandate CW vs CCW, but pick one and stay consistent)
    return P.tolist()
# ----------------------------
# CSV parsing
# ----------------------------

def parse_obb_csv(csv_path):
    labels = {}
    if not os.path.exists(csv_path):
        return labels

    with open(csv_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # Drop an optional title line like "OrientedBoundingBox"
    if lines and not lines[0].startswith('frame'):
        lines = lines[1:]
    if not lines:
        return labels
    # header = [h.strip() for h in lines[0].split(',')]
    for row in lines[1:]:
        parts = row.split(',')
        if len(parts) < 11:
            continue
        frame_id = parts[0].strip()
        class_id = int(float(parts[2]))
        x1, y1 = float(parts[3]), float(parts[4])
        x2, y2 = float(parts[5]), float(parts[6])
        x3, y3 = float(parts[7]), float(parts[8])
        x4, y4 = float(parts[9]), float(parts[10])
        item = {
            'class_id': class_id,
            'points': [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        }
        labels.setdefault(frame_id, []).append(item)
    return labels

# ----------------------------
# Dataset structure helpers
# ----------------------------

def create_dataset_structure(output_dir):
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    return splits

def split_data(labeled_frames, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    rng = random.Random(seed)
    all_frames = list(labeled_frames.keys())
    rng.shuffle(all_frames)
    n_total = len(all_frames)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_frames = all_frames[:n_train]
    val_frames = all_frames[n_train:n_train + n_val]
    test_frames = all_frames[n_train + n_val:]
    return {'train': train_frames, 'val': val_frames, 'test': test_frames}

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
# Frame extraction (OpenCV)
# ----------------------------

def extract_frame_opencv(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_number < 0 or (total and frame_number >= total):
        print(f"Warning: frame {frame_number} out of range 0..{max(total-1,0)} for {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number, 0))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# ----------------------------
# Main export
# ----------------------------

def export_yolo_obb_dataset(label_dir, video_dir, output_dir, class_file=None,
                            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                            image_size=640, seed=42, flip_x=False, flip_y=True):
    # Class names
    class_names = load_class_names(class_file)
    print(f"Using class names: {class_names}")

    # Pick most recent timestamped label folder
    datetime_pattern = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")
    matching_folders = [
        name for name in os.listdir(label_dir)
        if os.path.isdir(os.path.join(label_dir, name)) and datetime_pattern.match(name)
    ]
    if not matching_folders:
        raise ValueError(f"No labeled data folders found in {label_dir}")
    matching_folders.sort()
    label_folder = os.path.join(label_dir, matching_folders[-1])
    print(f"Using most recent labels: {matching_folders[-1]}")

    # Create dataset dirs
    _ = create_dataset_structure(output_dir)

    # Find all OBB csvs
    obb_files = [f for f in os.listdir(label_folder) if f.endswith('_obb.csv')]
    if not obb_files:
        raise ValueError(f"No *_obb.csv files found in {label_folder}")

    all_labeled_frames = {}
    video_files = {}

    # Map each camera file to its video
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    for obb_file in obb_files:
        camera_name = obb_file.replace('_obb.csv', '')
        # Find video
        vpath = None
        for ext in video_extensions:
            cand = os.path.join(video_dir, f"{camera_name}{ext}")
            if os.path.exists(cand):
                vpath = cand
                break
        if vpath is None:
            print(f"Warning: no video found for camera {camera_name}")
            continue
        video_files[camera_name] = vpath

        # Parse labels
        csv_path = os.path.join(label_folder, obb_file)
        camera_labels = parse_obb_csv(csv_path)

        # Namespacing frames
        for frame_id, labels in camera_labels.items():
            full_frame_id = f"{camera_name}_{frame_id}"
            all_labeled_frames[full_frame_id] = {
                'camera': camera_name,
                'frame_id': frame_id,
                'video_path': vpath,
                'labels': labels
            }

    if not all_labeled_frames:
        raise ValueError("No labeled frames found. Check *_obb.csv and videos.")

    print(f"Found {len(all_labeled_frames)} labeled frames across {len(video_files)} cameras")

    # Split
    data_splits = split_data(all_labeled_frames, train_ratio, val_ratio, test_ratio, seed=seed)
    print(f"Split: Train={len(data_splits['train'])}, Val={len(data_splits['val'])}, Test={len(data_splits['test'])}")

    # Export each split
    for split_name, frame_ids in data_splits.items():
        print(f"\nExporting {split_name}...")
        images_dir = os.path.join(output_dir, split_name, 'images')
        labels_dir = os.path.join(output_dir, split_name, 'labels')

        for i, frame_key in enumerate(frame_ids):
            meta = all_labeled_frames[frame_key]
            cam = meta['camera']
            frame_id = meta['frame_id']
            vpath = meta['video_path']
            labels = meta['labels']

            # Frame index
            try:
                frame_number = int(frame_id)
            except Exception:
                frame_number = i

            # Extract frame
            frame = extract_frame_opencv(vpath, frame_number)
            if frame is None:
                print(f"Failed to extract frame {frame_key}")
                continue

            # Letterbox
            letterboxed, scale, xoff, yoff = resize_image_for_yolo(frame, image_size)

            # Save image
            img_filename = f"{frame_key}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            cv2.imwrite(img_path, letterboxed)

            # Write label (YOLO OBB)
            label_filename = f"{frame_key}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            with open(label_path, 'w') as f:
                for lab in labels:
                    cls = lab['class_id']
                    pts = lab['points']  # [(x1,y1)...(x4,y4)]
                    #pts = lab['points']                            # [(x1,y1),...,(x4,y4)] in pixels
                    pts = order_corners_consistently(pts)
                    pts_resized = adjust_points_for_resize(pts, scale, xoff, yoff)
                    pts_norm    = normalize_points(pts_resized, image_size, image_size, flip_y=True)  # if your source uses bottom-origin Y

                   # pts_resized = adjust_points_for_resize(pts, scale, xoff, yoff)
                   # pts_norm = normalize_points(pts_resized, image_size, image_size,
                   #                             flip_x=flip_x, flip_y=flip_y)
                    flat = [coord for xy in pts_norm for coord in xy]
                    f.write(f"{cls} " + " ".join(f"{v:.6f}" for v in flat) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(frame_ids)} frames")

    # data.yaml
    create_data_yaml(output_dir, class_names)

    # README
    readme = os.path.join(output_dir, 'README.md')
    with open(readme, 'w') as f:
        f.write("# YOLO OBB Dataset\n\n")
        f.write("Generated by export_yolo_obb_opencv_flip.py\n\n")
        f.write("## Label format\n")
        f.write("Each label line: `class x1 y1 x2 y2 x3 y3 x4 y4` (normalized to [0,1]).\n\n")
        f.write("## Axis flips\n")
        f.write("- `--flip_y` (default true) flips the vertical axis before normalization.\n")
        f.write("- `--flip_x` (default false) flips the horizontal axis before normalization.\n")
        f.write("- Use these to match your source annotation coordinate origin.\n")

    print("\nDone!")
    print(f"Output: {output_dir}")

def str2bool(v):
    if isinstance(v, bool): return v
    s = str(v).lower().strip()
    if s in ('true','1','yes','y','on'): return True
    if s in ('false','0','no','n','off'): return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")

def main():
    ap = argparse.ArgumentParser(description="Export YOLO OBB Dataset (OpenCV-only, with axis flips)")
    ap.add_argument('-i','--label_dir', required=True, type=str,
                    help='Directory containing timestamped label folders')
    ap.add_argument('-v','--video_dir', required=True, type=str,
                    help='Directory containing camera videos (CamName.mp4, etc.)')
    ap.add_argument('-o','--output_dir', required=True, type=str,
                    help='Output directory for dataset')
    ap.add_argument('-c','--class_file', type=str, default=None,
                    help='Optional file with class names, one per line')
    ap.add_argument('--train_ratio', type=float, default=0.7)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--image_size', type=int, default=640,
                    help='Square size after letterbox (default 640)')
    ap.add_argument('--flip_x', type=str2bool, default=False,
                    help='Flip horizontal axis before normalization (default: false)')
    ap.add_argument('--flip_y', type=str2bool, default=True,
                    help='Flip vertical axis before normalization (default: true)')
    args = ap.parse_args()

    if not os.path.exists(args.label_dir):
        print(f"Error: label dir {args.label_dir} not found"); return
    if not os.path.exists(args.video_dir):
        print(f"Error: video dir {args.video_dir} not found"); return
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("Error: splits must sum to 1.0"); return

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        export_yolo_obb_dataset(
            label_dir=args.label_dir,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            class_file=args.class_file,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            image_size=args.image_size,
            seed=args.seed,
            flip_x=args.flip_x,
            flip_y=args.flip_y
        )
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
