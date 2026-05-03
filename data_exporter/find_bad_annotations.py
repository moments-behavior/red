"""Scan a parent folder of JARVIS-exported datasets and report
annotations whose 2D keypoints are all (0, 0) — these triangulate to
(0, 0, 0) in 3D and crash JARVIS's get_dataset_config.

Usage:
    python find_bad_annotations.py /path/to/parent_folder
"""

import json
import os
import sys
from collections import defaultdict


def all_keypoints_zero(kps_flat):
    """JARVIS keypoints are flat: [x0, y0, v0, x1, y1, v1, ...].
    Return True if every (x, y) is (0, 0)."""
    for i in range(0, len(kps_flat), 3):
        if kps_flat[i] != 0 or kps_flat[i + 1] != 0:
            return False
    return True


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    parent = sys.argv[1]

    sources = sorted(d for d in os.listdir(parent)
                     if os.path.isdir(os.path.join(parent, d)))

    for src in sources:
        ann_dir = os.path.join(parent, src, "annotations")
        if not os.path.isdir(ann_dir):
            continue

        for split in ["train", "val", "test"]:
            path = os.path.join(ann_dir, f"instances_{split}.json")
            if not os.path.isfile(path):
                continue

            with open(path) as f:
                data = json.load(f)

            # Build image_id -> file_name lookup so we can name bad framesets.
            img_lookup = {im["id"]: im["file_name"] for im in data.get("images", [])}

            # For each frame (logical group: file path before /Cam/), count
            # how many annotations are "all-zero" vs total.
            per_frame_total = defaultdict(int)
            per_frame_zero = defaultdict(int)
            for ann in data.get("annotations", []):
                fn = img_lookup.get(ann["image_id"], f"id={ann['image_id']}")
                # file name: <trial>/<cam>/Frame_<N>.jpg
                parts = fn.split("/")
                if len(parts) >= 3:
                    frame_key = (parts[0], parts[-1])  # (trial, Frame_N.jpg)
                else:
                    frame_key = (fn,)
                per_frame_total[frame_key] += 1
                if all_keypoints_zero(ann["keypoints"]):
                    per_frame_zero[frame_key] += 1

            bad = [k for k, n in per_frame_zero.items()
                   if n == per_frame_total[k]]
            if bad:
                print(f"\n[{src}/{split}] {len(bad)} fully-zero framesets "
                      f"(every camera all-zero):")
                for k in bad[:10]:
                    print(f"    {k}")
                if len(bad) > 10:
                    print(f"    ... and {len(bad) - 10} more")


if __name__ == "__main__":
    main()
