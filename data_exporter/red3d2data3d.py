#!/usr/bin/env python3
"""red3d2data3d.py — convert a RED `#red_csv v2` keypoints3d.csv into the
JARVIS `data3D.csv` format consumed by the RED "Import JARVIS predictions"
window (src/jarvis_import.h) and by data_exporter/jarvis2red3d.py.

RED keypoints3d.csv (input):
    #red_csv v2
    #skeleton <name>
    frame,x0,y0,z0,c0,x1,y1,z1,c1,...
    0,<x0>,<y0>,<z0>,<c0>,...

JARVIS data3D.csv (output):
    2 header rows (ignored by the reader),
    then per-frame rows of groups of 4: x,y,z,conf,x,y,z,conf,...
    frame_id is implicit = row index. Missing frames are written as NaN rows
    so the reader's whole-row NaN skip keeps frame_id aligned to the row index.
"""
import argparse
import csv
import os


def convert(input_path, output_path):
    with open(input_path) as f:
        rows = list(csv.reader(f))

    # Strip leading comment/header lines (#red_csv, #skeleton, column header).
    data_rows = [r for r in rows if r and not r[0].startswith("#") and r[0] != "frame"]

    # Map frame_id -> list of 4*num_kp value tokens (frame column dropped).
    by_frame = {}
    num_cols = 0
    for r in data_rows:
        if not r[0].strip():
            continue
        frame_id = int(float(r[0]))
        vals = r[1:]
        by_frame[frame_id] = vals
        num_cols = max(num_cols, len(vals))

    if num_cols % 4 != 0:
        raise ValueError(
            f"expected value count divisible by 4 (x,y,z,c); got {num_cols}"
        )

    max_frame = max(by_frame) if by_frame else -1
    nan_row = ["NaN"] * num_cols

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        # 2 header rows — content is ignored by the reader, kept human-readable.
        w.writerow(["scorer"] + ["red3d"] * num_cols)
        w.writerow(["coords"] + (["x", "y", "z", "conf"] * (num_cols // 4)))
        for frame_id in range(max_frame + 1):
            row = by_frame.get(frame_id, nan_row)
            # Normalise empty cells to NaN so the reader skips them as a group.
            row = [v if v.strip() else "NaN" for v in row]
            w.writerow(row)

    return len(by_frame), max_frame + 1, num_cols // 4


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True,
                    help="path to RED keypoints3d.csv (#red_csv v2)")
    ap.add_argument("-o", "--output", default=None,
                    help="output data3D.csv path (default: data3D.csv beside input)")
    args = ap.parse_args()

    out = args.output or os.path.join(os.path.dirname(args.input), "data3D.csv")
    n, total, kp = convert(args.input, out)
    print(f"Wrote {out}")
    print(f"  frames written: {total} ({n} populated), keypoints: {kp}")


if __name__ == "__main__":
    main()
