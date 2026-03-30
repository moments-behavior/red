#!/usr/bin/env python3
"""
Export CoTracker3 (offline) to a TorchScript .pt file for use in RED.

Usage:
    python3 export_cotracker_torchscript.py \
        --output /path/to/cotracker3_offline.pt

The resulting .pt file is loaded by CoTrackerInfer::load() in RED.

Model inputs (after tracing):
    video   : (1, T, 3, H, W)  float32, values in [0, 255]
    queries : (1, N, 3)        float32, each row [frame_idx, x, y]

Model outputs:
    tracks  : (1, T, N, 2)    float32  (x, y) per frame per query
    vis     : (1, T, N)       float32  raw logit; apply sigmoid for [0,1]
"""

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Export CoTracker3 offline model to TorchScript")
    parser.add_argument("--output", required=True,
                        help="Output path for cotracker3_offline.pt")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device (default: cuda:0)")
    # Tracing dimensions — use small values for speed; the model is flexible
    parser.add_argument("--T", type=int, default=8,
                        help="Number of frames for tracing (default: 8)")
    parser.add_argument("--H", type=int, default=256,
                        help="Frame height for tracing (default: 256)")
    parser.add_argument("--W", type=int, default=256,
                        help="Frame width for tracing (default: 256)")
    parser.add_argument("--N", type=int, default=4,
                        help="Number of query points for tracing (default: 4)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    device = torch.device(args.device)

    print("[INFO] Loading CoTracker3 offline via torch.hub …")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model.to(device).eval()

    # Example inputs for tracing
    T, H, W, N = args.T, args.H, args.W, args.N
    video   = torch.randn(1, T, 3, H, W, device=device)
    # queries: [frame_idx, x, y] — use middle frame
    queries = torch.zeros(1, N, 3, device=device)
    queries[0, :, 0] = T // 2      # anchor frame
    queries[0, :, 1] = W / 2.0    # x
    queries[0, :, 2] = H / 2.0    # y

    print(f"[INFO] Tracing with video={list(video.shape)}, queries={list(queries.shape)} …")
    with torch.no_grad():
        traced = torch.jit.trace(model, (video, queries), strict=False)

    # Verify the trace produces sensible output
    with torch.no_grad():
        tracks, vis = traced(video, queries)
    print(f"[INFO] Trace OK — tracks {list(tracks.shape)}, vis {list(vis.shape)}")

    traced.save(args.output)
    print(f"[DONE] Saved TorchScript model → {args.output}")
    print()
    print("Load in RED via 'Load CoTracker Model' and selecting this .pt file.")


if __name__ == "__main__":
    main()
