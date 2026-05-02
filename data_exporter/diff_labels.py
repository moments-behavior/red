"""Compare keypoints3d.csv between two label snapshot folders, and
(optionally) write a new folder containing only the added/modified frames
so red3d2jarvis.py can export just the diff.

Typical use after an active-learning round:

    python diff_labels.py \\
        /path/to/labeled_data/2025_07_01_13_08_28 \\
        /path/to/labeled_data/2025_08_28_14_55_31 \\
        -o /path/to/labeled_data/2099_01_01_00_00_00

The -o folder will contain:
- keypoints3d.csv with only the changed frames
- per-camera Cam<serial>.csv with the same subset

If you place -o under the project's labeled_data/ folder, its name must
match red3d2jarvis's strict YYYY_MM_DD_HH_MM_SS pattern and sort after
every existing snapshot so red3d2jarvis picks it as the "most recent".
A far-future date like 2099_01_01_00_00_00 is the simplest way.

Run red3d2jarvis.py against the project afterwards to export a JARVIS
dataset of just the diff, then merge with your prior export via
merge_jarvis_datasets.py.
"""

import argparse
import csv
import os
import re
import shutil

import numpy as np

from utils import csv_reader_red3d


TOLERANCE_MM = 1e-3


def _frame_changed(a, b):
    """True if two (n_kp, 3) arrays differ in NaN pattern or in value
    beyond TOLERANCE_MM."""
    if a.shape != b.shape:
        return True
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return True
    valid = ~nan_a
    if not np.any(valid):
        return False
    return bool(np.max(np.abs(a[valid] - b[valid])) > TOLERANCE_MM)


def _filter_csv(src_csv, dst_csv, frame_ids):
    """Copy rows from src_csv to dst_csv whose first column (frame id)
    is in frame_ids. Keeps the header row."""
    keep = {str(f) for f in frame_ids}
    with open(src_csv, newline="") as fin, open(dst_csv, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for i, row in enumerate(reader):
            if i == 0 or (row and row[0] in keep):
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_dir",
                        help="Older label snapshot folder "
                             "(contains keypoints3d.csv).")
    parser.add_argument("new_dir",
                        help="Newer label snapshot folder.")
    parser.add_argument("-o", "--output_dir", default=None,
                        help="If set, write a new folder at this path "
                             "containing only the changed frames.")
    args = parser.parse_args()

    for d, label in [(args.old_dir, "old"), (args.new_dir, "new")]:
        if not os.path.isfile(os.path.join(d, "keypoints3d.csv")):
            raise SystemExit(
                f"{label} folder missing keypoints3d.csv: {d}")

    print(f"Comparing\n  old: {args.old_dir}\n  new: {args.new_dir}")

    old_labels = csv_reader_red3d(os.path.join(args.old_dir, "keypoints3d.csv"))
    new_labels = csv_reader_red3d(os.path.join(args.new_dir, "keypoints3d.csv"))

    old_frames = set(old_labels)
    new_frames = set(new_labels)
    added = sorted(new_frames - old_frames)
    removed = sorted(old_frames - new_frames)
    common = old_frames & new_frames

    modified = []
    for fid in sorted(common):
        if _frame_changed(old_labels[fid], new_labels[fid]):
            modified.append(fid)

    print(f"\n  added:    {len(added):4d} frames")
    print(f"  modified: {len(modified):4d} frames")
    print(f"  removed:  {len(removed):4d} frames")
    print(f"  unchanged:{len(common) - len(modified):4d} frames")

    if added:
        print(f"\nAdded frame ids (first 20): {added[:20]}")
    if modified:
        print(f"Modified frame ids (first 20): {modified[:20]}")
    if removed:
        print(f"Removed frame ids (first 20): {removed[:20]}")

    if args.output_dir:
        diff_frames = sorted(set(added) | set(modified))
        if not diff_frames:
            print("\nNo diff frames to write; skipping output folder.")
            return

        out_basename = os.path.basename(os.path.normpath(args.output_dir))
        if not re.fullmatch(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}",
                            out_basename):
            print(f"\nWARN: output folder name '{out_basename}' does not "
                  "match YYYY_MM_DD_HH_MM_SS — red3d2jarvis.py will not "
                  "pick it up automatically.")

        os.makedirs(args.output_dir, exist_ok=True)

        # keypoints3d.csv
        _filter_csv(os.path.join(args.new_dir, "keypoints3d.csv"),
                    os.path.join(args.output_dir, "keypoints3d.csv"),
                    diff_frames)

        # per-camera Cam<serial>.csv
        for fname in os.listdir(args.new_dir):
            if fname == "keypoints3d.csv" or not fname.endswith(".csv"):
                continue
            _filter_csv(os.path.join(args.new_dir, fname),
                        os.path.join(args.output_dir, fname),
                        diff_frames)

        # copy any non-CSV files (skeleton refs, etc.) untouched
        for fname in os.listdir(args.new_dir):
            src = os.path.join(args.new_dir, fname)
            if os.path.isfile(src) and not fname.endswith(".csv"):
                shutil.copy2(src, os.path.join(args.output_dir, fname))

        print(f"\nWrote diff snapshot ({len(diff_frames)} frames) "
              f"to {args.output_dir}")


if __name__ == "__main__":
    main()
