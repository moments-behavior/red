"""
yaml_utils.py — Shared OpenCV YAML parsing utilities.

Parses OpenCV-style YAML files without requiring PyYAML or OpenCV.
Used by calibration_refinement.py, extract_synced_frames.py,
analyze_video_diversity.py, and red_to_colmap.py.
"""

import re
import numpy as np


def read_yaml_file(path):
    """Read an entire YAML file and return its content string."""
    with open(path, "r") as f:
        return f.read()


def parse_yaml_matrix(content, key):
    """Parse an opencv-matrix entry from YAML content string.

    Args:
        content: Full file content (from read_yaml_file)
        key: Matrix field name (e.g., 'camera_matrix', 'rc_ext')

    Returns: numpy array or None if not found
    """
    pattern = (
        rf"{key}:\s*!!opencv-matrix\s*"
        rf"rows:\s*(\d+)\s*cols:\s*(\d+)\s*dt:\s*d\s*data:\s*\[([\s\S]*?)\]"
    )
    match = re.search(pattern, content)
    if not match:
        return None
    rows, cols = int(match.group(1)), int(match.group(2))
    data_str = match.group(3)
    values = [float(x.strip()) for x in data_str.split(",") if x.strip()]
    return np.array(values).reshape(rows, cols)


def parse_yaml_scalar(content, key):
    """Parse a simple key: value scalar from YAML content string.

    Args:
        content: Full file content (from read_yaml_file)
        key: Scalar field name (e.g., 'image_width')

    Returns: string value or None if not found
    """
    m = re.search(rf"^\s*{key}\s*:\s*(.+)$", content, re.MULTILINE)
    return m.group(1).strip() if m else None


def load_camera_yaml(path):
    """Load all calibration fields from a single Cam*.yaml file.

    Reads the file ONCE and extracts all fields.

    Returns: dict with keys 'K', 'dist', 'R', 't', 'image_width', 'image_height'
             or None if required fields are missing.
    """
    content = read_yaml_file(path)

    K = parse_yaml_matrix(content, "camera_matrix")
    dist = parse_yaml_matrix(content, "distortion_coefficients")
    R = parse_yaml_matrix(content, "rc_ext")
    t = parse_yaml_matrix(content, "tc_ext")

    if K is None or R is None or t is None:
        return None

    w = parse_yaml_scalar(content, "image_width")
    h = parse_yaml_scalar(content, "image_height")

    return {
        "K": K,
        "dist": dist.flatten()[:5] if dist is not None else np.zeros(5),
        "R": R,
        "t": t.flatten()[:3],
        "image_width": int(w) if w else None,
        "image_height": int(h) if h else None,
    }
