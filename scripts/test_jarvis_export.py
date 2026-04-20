#!/usr/bin/env python3
"""Test JARVIS export JPEG extraction pipeline.

Extracts a few frames from one camera using the same approach as RED's
jarvis_export.h: ffmpeg seek → decode → RGB24 → JPEG via Pillow.
Also tests reading back the saved JPEGs to verify they're valid.

Usage:
    python3 scripts/test_jarvis_export.py
"""

import os
import sys
import subprocess
import struct
import numpy as np

# Paths
VIDEO_DIR = "/Users/johnsonr/datasets/rat/sessions/2025_09_03_15_18_21"
OUTPUT_DIR = "/tmp/red_jarvis_export_test"
TEST_CAM = "Cam2002479"
TEST_FRAMES = [10360, 10361, 10362, 10400, 10500]  # first few labeled frames


def detect_negative_pts_offset(video_path, fps):
    """Match RED's detect_negative_pts_offset: check if first packet has negative PTS."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_packets", "-show_entries", "packet=pts_time",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10
        )
        first_line = result.stdout.strip().split('\n')[0]
        first_pts = float(first_line)
        if first_pts < -0.001:
            return int(round(abs(first_pts) * fps))
    except Exception as e:
        print(f"  PTS detection error: {e}")
    return 0


def get_video_info(video_path):
    """Get video width, height, fps via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    parts = result.stdout.strip().split(',')
    w, h = int(parts[0]), int(parts[1])
    fps_parts = parts[2].split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    return w, h, fps


def extract_frame_ffmpeg(video_path, frame_num, fps, width, height):
    """Extract a single frame as RGB24 numpy array using ffmpeg."""
    timestamp = frame_num / fps
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-ss", f"{timestamp:.6f}",
        "-i", video_path,
        "-frames:v", "1",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-"
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0 or len(result.stdout) != width * height * 3:
        return None
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def save_jpeg_pillow(rgb_array, output_path, quality=95):
    """Save RGB numpy array as JPEG via Pillow."""
    from PIL import Image
    img = Image.fromarray(rgb_array, 'RGB')
    img.save(output_path, 'JPEG', quality=quality)


def save_jpeg_turbojpeg(rgb_array, output_path, quality=95):
    """Save RGB numpy array as JPEG via turbojpeg (matches RED's macOS path)."""
    try:
        import turbojpeg
        tj = turbojpeg.TurboJPEG()
        # turbojpeg expects BGR by default, but we can specify pixel format
        jpeg_buf = tj.encode(rgb_array, quality=quality, pixel_format=turbojpeg.TJPF_RGB)
        with open(output_path, 'wb') as f:
            f.write(jpeg_buf)
        return True
    except ImportError:
        return False


def verify_jpeg(path):
    """Check if a JPEG file is valid by reading it back."""
    from PIL import Image
    try:
        img = Image.open(path)
        img.load()  # force full decode
        return img.size, img.mode
    except Exception as e:
        return None, str(e)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_path = os.path.join(VIDEO_DIR, f"{TEST_CAM}.mp4")
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Get video info
    w, h, fps = get_video_info(video_path)
    print(f"Video: {TEST_CAM}.mp4 — {w}x{h} @ {fps:.1f} fps")

    # Detect PTS offset (matches RED's logic)
    pts_offset = detect_negative_pts_offset(video_path, fps)
    print(f"PTS offset: {pts_offset} frames")

    # Test each frame
    print(f"\nExtracting {len(TEST_FRAMES)} test frames...")
    results = []

    for frame_num in TEST_FRAMES:
        seek_frame = frame_num - pts_offset

        # Extract frame via ffmpeg (same approach as RED's FrameReader)
        rgb = extract_frame_ffmpeg(video_path, seek_frame, fps, w, h)
        if rgb is None:
            print(f"  Frame {frame_num}: FAILED to decode")
            results.append((frame_num, False, "decode failed"))
            continue

        # Check for obviously bad data
        mean_val = rgb.mean()
        is_black = mean_val < 5
        is_white = mean_val > 250
        if is_black:
            print(f"  Frame {frame_num}: WARNING — all black (mean={mean_val:.1f})")
        elif is_white:
            print(f"  Frame {frame_num}: WARNING — all white (mean={mean_val:.1f})")

        # Save via Pillow
        pillow_path = os.path.join(OUTPUT_DIR, f"Frame_{frame_num}_pillow.jpg")
        save_jpeg_pillow(rgb, pillow_path)

        # Save via turbojpeg if available
        tj_path = os.path.join(OUTPUT_DIR, f"Frame_{frame_num}_turbojpeg.jpg")
        tj_ok = save_jpeg_turbojpeg(rgb, tj_path)

        # Verify saved JPEGs
        size_p, mode_p = verify_jpeg(pillow_path)
        file_size_p = os.path.getsize(pillow_path)

        status = "OK"
        if size_p is None:
            status = f"CORRUPT: {mode_p}"
        elif size_p != (w, h):
            status = f"SIZE MISMATCH: expected {w}x{h}, got {size_p}"

        print(f"  Frame {frame_num}: {status} — "
              f"mean={mean_val:.1f}, jpeg={file_size_p//1024}KB"
              f"{', turbojpeg=OK' if tj_ok else ', turbojpeg=N/A'}")
        results.append((frame_num, size_p is not None, status))

    # Summary
    print(f"\n{'='*60}")
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"Results: {n_ok}/{len(results)} frames extracted successfully")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"\nInspect the JPEGs visually:")
    print(f"  open {OUTPUT_DIR}/Frame_{TEST_FRAMES[0]}_pillow.jpg")

    if n_ok == len(results):
        print("\nPython extraction pipeline works. If RED's export produces "
              "different results, the bug is in the C++ code.")
    else:
        print("\nSome frames failed — check video file and frame numbering.")


if __name__ == "__main__":
    main()
