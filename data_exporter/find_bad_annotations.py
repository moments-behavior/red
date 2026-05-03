"""Scan a parent folder of JARVIS-exported datasets and report
framesets that would crash JARVIS's get_dataset_config — those where
every keypoint has non-zero 2D in fewer than 2 cameras (so JARVIS can't
triangulate any keypoint, and self.keypoints3D ends up all-zero).

Usage:
    python find_bad_annotations.py /path/to/parent_folder
"""

import json
import os
import sys
from collections import defaultdict


def per_keypoint_visible_cams(kps_flat):
    """JARVIS keypoints are flat: [x0, y0, v0, x1, y1, v1, ...].
    Return a list of bools, one per keypoint, True if (x, y) != (0, 0)."""
    out = []
    for i in range(0, len(kps_flat), 3):
        out.append(kps_flat[i] != 0 or kps_flat[i + 1] != 0)
    return out


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

            img_lookup = {im["id"]: im["file_name"]
                          for im in data.get("images", [])}

            # For each frame (trial + Frame_N.jpg), tally per-keypoint
            # how many cameras saw it (non-zero 2D).
            per_frame_kp_views = defaultdict(lambda: None)
            for ann in data.get("annotations", []):
                fn = img_lookup.get(ann["image_id"], f"id={ann['image_id']}")
                parts = fn.split("/")
                frame_key = (parts[0], parts[-1]) if len(parts) >= 3 else (fn,)
                visible = per_keypoint_visible_cams(ann["keypoints"])
                if per_frame_kp_views[frame_key] is None:
                    per_frame_kp_views[frame_key] = [0] * len(visible)
                for i, v in enumerate(visible):
                    if v:
                        per_frame_kp_views[frame_key][i] += 1

            # A frameset is "bad" if no keypoint has >= 2 cameras seeing it.
            bad = [k for k, counts in per_frame_kp_views.items()
                   if max(counts) < 2]
            if bad:
                print(f"\n[{src}/{split}] {len(bad)} untriangulatable "
                      f"frameset(s) (no keypoint has >= 2 cameras):")
                for k in bad[:10]:
                    print(f"    {k}  cam-views per kp: "
                          f"{per_frame_kp_views[k]}")
                if len(bad) > 10:
                    print(f"    ... and {len(bad) - 10} more")


if __name__ == "__main__":
    main()
