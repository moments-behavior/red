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


def _frame_max_diff(a, b):
    """Returns (max_abs_diff_mm, nan_pattern_changed). max diff is np.inf
    when shapes differ; 0.0 when both fully NaN."""
    if a.shape != b.shape:
        return float("inf"), True
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    nan_changed = not np.array_equal(nan_a, nan_b)
    valid = ~(nan_a | nan_b)
    if not np.any(valid):
        return 0.0, nan_changed
    return float(np.max(np.abs(a[valid] - b[valid]))), nan_changed


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
    parser.add_argument("--tolerance_mm", type=float, default=1.0,
                        help="A frame counts as 'modified' only if some "
                             "keypoint moved by more than this many mm or "
                             "its NaN-status changed. Default 1.0 mm "
                             "(below this is usually FP noise from red "
                             "re-saving unchanged frames).")
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

    diffs = []  # (fid, max_diff_mm, nan_changed)
    # per-keypoint accumulator: list of euclidean shifts (mm) per kp,
    # only over (frame, kp) pairs where both old & new are non-NaN.
    per_kp_shifts = None
    for fid in sorted(common):
        old = old_labels[fid]
        new = new_labels[fid]
        if per_kp_shifts is None and old.shape == new.shape:
            per_kp_shifts = [[] for _ in range(old.shape[0])]
        d, nc = _frame_max_diff(old, new)
        diffs.append((fid, d, nc))
        if old.shape == new.shape and per_kp_shifts is not None:
            valid = ~(np.isnan(old).any(axis=1) | np.isnan(new).any(axis=1))
            shift = np.linalg.norm(new - old, axis=1)
            for k in np.where(valid)[0]:
                per_kp_shifts[k].append(float(shift[k]))

    modified = [fid for fid, d, nc in diffs
                if nc or d > args.tolerance_mm]

    finite_diffs = np.array([d for _, d, _ in diffs if np.isfinite(d)])
    if finite_diffs.size:
        print("\nMax keypoint shift per common frame (mm) — distribution:")
        for q, label in [(50, "p50"), (90, "p90"), (99, "p99"),
                         (100, "max")]:
            print(f"  {label}: {np.percentile(finite_diffs, q):8.4f} mm")
        print(f"  frames with shift > {args.tolerance_mm} mm: "
              f"{int((finite_diffs > args.tolerance_mm).sum())}")

    print(f"\n  added:    {len(added):4d} frames")
    print(f"  modified: {len(modified):4d} frames "
          f"(threshold > {args.tolerance_mm} mm)")
    print(f"  removed:  {len(removed):4d} frames")
    print(f"  unchanged:{len(common) - len(modified):4d} frames")

    if per_kp_shifts:
        # mean shift per keypoint, restricted to frames where the
        # keypoint actually moved by more than the tolerance (so noise
        # doesn't dilute the average).
        rows = []
        for k, shifts in enumerate(per_kp_shifts):
            real_shifts = [s for s in shifts if s > args.tolerance_mm]
            if real_shifts:
                rows.append((k, len(real_shifts),
                             float(np.mean(real_shifts)),
                             float(np.max(real_shifts))))
        rows.sort(key=lambda r: r[2], reverse=True)
        print("\nPer-keypoint change (frames above threshold only):")
        print(f"  {'kp':>4}  {'n':>6}  {'mean (mm)':>10}  {'max (mm)':>10}")
        for k, n, mean_mm, max_mm in rows:
            print(f"  {k:>4d}  {n:>6d}  {mean_mm:>10.2f}  {max_mm:>10.2f}")

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
