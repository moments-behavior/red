"""Compare keypoints3d.csv between two timestamped label snapshots in a red
project, and (optionally) write a new timestamped subset folder containing
only the added/modified frames so red3d2jarvis.py can export just the diff.

Typical use after an active-learning round:

    python diff_labels.py -p /path/to/red_project \
        --old 2025_07_01_13_08_28 --new 2025_08_28_14_55_31 \
        -o 2099_01_01_00_00_00

The -o folder lands under <project>/labeled_data/ and contains:
- keypoints3d.csv with only the changed frames
- per-camera Cam<serial>.csv with the same subset

The -o name must match red3d2jarvis's strict YYYY_MM_DD_HH_MM_SS pattern
and sort after every existing snapshot (so red3d2jarvis picks it as the
"most recent"). Using a far-future date like 2099_01_01_00_00_00 is the
simplest way to guarantee both.

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


DATETIME_PATTERN = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}")
TOLERANCE_MM = 1e-3


def _list_timestamped(label_folder):
    folders = [
        n for n in os.listdir(label_folder)
        if os.path.isdir(os.path.join(label_folder, n))
        and DATETIME_PATTERN.match(n)
    ]
    folders.sort()
    return folders


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
    parser.add_argument("-p", "--project_dir", required=True,
                        help="Red project dir containing labeled_data/.")
    parser.add_argument("--old", default=None,
                        help="Older timestamp folder name. "
                             "Default: second-most-recent.")
    parser.add_argument("--new", default=None,
                        help="Newer timestamp folder name. "
                             "Default: most-recent.")
    parser.add_argument("-o", "--output_name", default=None,
                        help="If set, write a new folder of this name under "
                             "labeled_data/ with only the changed frames.")
    args = parser.parse_args()

    label_folder = os.path.join(args.project_dir, "labeled_data")
    snapshots = _list_timestamped(label_folder)
    if len(snapshots) < 2:
        raise SystemExit(
            f"Need at least two timestamped snapshots under {label_folder}; "
            f"found {len(snapshots)}.")

    new_ts = args.new or snapshots[-1]
    old_ts = args.old or snapshots[-2]
    if new_ts not in snapshots or old_ts not in snapshots:
        raise SystemExit(
            f"Snapshots not found. Available: {snapshots}")

    old_dir = os.path.join(label_folder, old_ts)
    new_dir = os.path.join(label_folder, new_ts)
    print(f"Comparing\n  old: {old_ts}\n  new: {new_ts}")

    old_labels = csv_reader_red3d(os.path.join(old_dir, "keypoints3d.csv"))
    new_labels = csv_reader_red3d(os.path.join(new_dir, "keypoints3d.csv"))

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

    if args.output_name:
        diff_frames = sorted(set(added) | set(modified))
        if not diff_frames:
            print("\nNo diff frames to write; skipping output folder.")
            return

        if not re.fullmatch(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}",
                            args.output_name):
            print(f"\nWARN: output name '{args.output_name}' does not match "
                  "YYYY_MM_DD_HH_MM_SS — red3d2jarvis.py will not pick it up "
                  "automatically.")

        out_dir = os.path.join(label_folder, args.output_name)
        os.makedirs(out_dir, exist_ok=True)

        # keypoints3d.csv
        _filter_csv(os.path.join(new_dir, "keypoints3d.csv"),
                    os.path.join(out_dir, "keypoints3d.csv"),
                    diff_frames)

        # per-camera Cam<serial>.csv
        for fname in os.listdir(new_dir):
            if fname == "keypoints3d.csv" or not fname.endswith(".csv"):
                continue
            _filter_csv(os.path.join(new_dir, fname),
                        os.path.join(out_dir, fname),
                        diff_frames)

        # copy any non-CSV files (skeleton refs, etc.) untouched
        for fname in os.listdir(new_dir):
            src = os.path.join(new_dir, fname)
            if os.path.isfile(src) and not fname.endswith(".csv"):
                shutil.copy2(src, os.path.join(out_dir, fname))

        print(f"\nWrote diff snapshot ({len(diff_frames)} frames) to {out_dir}")
        print("Run red3d2jarvis.py on the project to export just the diff.")


if __name__ == "__main__":
    main()
