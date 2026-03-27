#!/usr/bin/env python3
"""
Visual hull reconstruction from multi-camera silhouettes.

Performs voxel carving using calibrated camera poses (ba_poses.json) and
foreground masks extracted from video frames. Outputs a PLY point cloud
and optionally an OBJ mesh via marching cubes.

Usage:
    python visual_hull.py --frame 39348 --images_dir /tmp/red_nerfstudio_test/images/
    python visual_hull.py --frame 39348 --images_dir /tmp/frames/ --bg_color 40,40,40
    python visual_hull.py --frame 39348 --images_dir /tmp/frames/ --masks_dir /tmp/masks/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image


def load_calibration(ba_poses_path):
    """Load camera calibration from ba_poses.json.

    Returns dict of cam_id -> {K: 3x3, R: 3x3, t: 3x1, dist: 5x1, P: 3x4}.
    R is world-to-camera, t is translation in camera frame.
    P = K @ [R | t].
    """
    with open(ba_poses_path, "r") as f:
        poses = json.load(f)

    cameras = {}
    for cam_id, data in poses.items():
        K = np.array(data["K"], dtype=np.float64)
        R = np.array(data["R"], dtype=np.float64)
        t = np.array(data["t"], dtype=np.float64).reshape(3, 1)
        dist = np.array(data["dist"], dtype=np.float64) if "dist" in data else None

        # P = K @ [R | t]
        Rt = np.hstack([R, t])
        P = K @ Rt

        cameras[cam_id] = {"K": K, "R": R, "t": t, "dist": dist, "P": P}

    return cameras


def get_camera_positions(cameras):
    """Compute world-space camera positions: C = -R^T @ t."""
    positions = []
    for cam in cameras.values():
        C = -cam["R"].T @ cam["t"]
        positions.append(C.flatten())
    return np.array(positions)


def compute_default_bbox(cameras, padding=100.0, height=300.0):
    """Compute a default bounding box from camera positions.

    Estimates the arena center from camera positions projected onto XY plane,
    then creates a box centered there with reasonable dimensions.
    """
    positions = get_camera_positions(cameras)
    center_xy = positions[:, :2].mean(axis=0)

    # Use the spread of cameras to estimate arena size, but cap it
    spread_xy = positions[:, :2].max(axis=0) - positions[:, :2].min(axis=0)
    arena_size = min(spread_xy.max() * 0.5, 800.0)
    arena_size = max(arena_size, 300.0)

    # Z: cameras look down, arena floor is likely near the lowest camera Z
    # or significantly below camera mean Z
    z_min_cam = positions[:, 2].min()
    z_mean = positions[:, 2].mean()

    # Arena floor is typically well below cameras
    z_floor = z_min_cam - 200.0
    z_ceil = z_floor + height

    half = arena_size / 2.0 + padding
    bbox = [
        center_xy[0] - half,
        center_xy[1] - half,
        z_floor,
        center_xy[0] + half,
        center_xy[1] + half,
        z_ceil,
    ]
    return bbox


def load_images_for_frame(images_dir, frame_num, camera_ids):
    """Load images for a specific frame number from the images directory.

    Expects filenames like Cam2002486_39348.jpg (or .png).
    Returns dict of cam_id -> PIL.Image.
    """
    images = {}
    for cam_id in camera_ids:
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            # Try both "Cam{cam_id}_{frame}" and just "{cam_id}_{frame}"
            for prefix in [f"Cam{cam_id}", cam_id]:
                path = os.path.join(images_dir, f"{prefix}_{frame_num}{ext}")
                if os.path.exists(path):
                    images[cam_id] = Image.open(path).convert("RGB")
                    break
            if cam_id in images:
                break

    return images


def generate_mask_brightness(image, threshold=60):
    """Generate foreground mask based on overall brightness.

    Pixels brighter than threshold are considered foreground.
    """
    arr = np.array(image, dtype=np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    mask = gray > threshold
    return mask


def generate_mask_bg_color(image, bg_color, threshold=40):
    """Generate foreground mask based on deviation from a background color."""
    arr = np.array(image, dtype=np.float32)
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
    diff = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    mask = diff > threshold
    return mask


def generate_mask_green_ratio(image, threshold=1.1):
    """Generate foreground mask where green channel is NOT dominant.

    In a dark arena, the background tends to have relatively higher green.
    The animal (mouse/rat) will have more uniform or reddish tones.
    This inverts the logic: foreground = where green is NOT dominant.
    """
    arr = np.array(image, dtype=np.float32) + 1.0  # avoid division by zero
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mean_rb = (r + b) / 2.0
    brightness = (r + g + b) / 3.0

    # Foreground: bright enough AND green not dominant
    mask = (brightness > 30) & (g / (mean_rb + 1.0) < threshold)
    return mask


def load_masks(masks_dir, frame_num, camera_ids):
    """Load pre-computed binary masks from a directory.

    Expects filenames like Cam2002486_39348.png (white = foreground).
    """
    masks = {}
    for cam_id in camera_ids:
        for ext in [".png", ".jpg", ".bmp"]:
            for prefix in [f"Cam{cam_id}", cam_id]:
                path = os.path.join(masks_dir, f"{prefix}_{frame_num}{ext}")
                if os.path.exists(path):
                    img = Image.open(path).convert("L")
                    masks[cam_id] = np.array(img) > 127
                    break
            if cam_id in masks:
                break
    return masks


def project_points(P, points_3d):
    """Project Nx3 world points using 3x4 projection matrix P.

    Returns Nx2 pixel coordinates.
    """
    N = points_3d.shape[0]
    # Homogeneous: Nx4
    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.hstack([points_3d, ones])

    # Project: (3x4) @ (4xN) -> 3xN
    projected = P @ pts_h.T  # 3 x N

    # Normalize
    z = projected[2, :]
    valid = np.abs(z) > 1e-8
    u = np.full(N, -1.0)
    v = np.full(N, -1.0)
    u[valid] = projected[0, valid] / z[valid]
    v[valid] = projected[1, valid] / z[valid]

    return u, v, z > 0  # return "in front of camera" flag


def voxel_carving(cameras, masks, bbox, voxel_size, min_cameras):
    """Perform voxel carving.

    Args:
        cameras: dict cam_id -> {P: 3x4, ...}
        masks: dict cam_id -> 2D bool array (H x W)
        bbox: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: size of each voxel in mm
        min_cameras: minimum number of cameras that must see the voxel as foreground

    Returns:
        surviving_points: Nx3 array of voxel centers that survived carving
        occupancy_grid: 3D bool array
        grid_origin: (x_min, y_min, z_min)
        grid_shape: (nx, ny, nz)
    """
    x_min, y_min, z_min, x_max, y_max, z_max = bbox

    # Create voxel grid
    xs = np.arange(x_min + voxel_size / 2, x_max, voxel_size)
    ys = np.arange(y_min + voxel_size / 2, y_max, voxel_size)
    zs = np.arange(z_min + voxel_size / 2, z_max, voxel_size)

    nx, ny, nz = len(xs), len(ys), len(zs)
    total_voxels = nx * ny * nz

    print(f"  Grid: {nx} x {ny} x {nz} = {total_voxels:,} voxels")
    print(f"  Voxel size: {voxel_size} mm")

    if total_voxels > 50_000_000:
        print(f"  WARNING: Very large grid ({total_voxels:,} voxels). Consider larger --voxel_size.")

    # Create all voxel centers as Nx3 array
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    voxel_centers = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    # Vote array: count how many cameras see each voxel as foreground
    votes = np.zeros(total_voxels, dtype=np.int32)

    cam_ids = sorted(set(cameras.keys()) & set(masks.keys()))
    print(f"  Carving with {len(cam_ids)} cameras...")

    for cam_id in cam_ids:
        t0 = time.time()
        P = cameras[cam_id]["P"]
        mask = masks[cam_id]
        H, W = mask.shape

        u, v, in_front = project_points(P, voxel_centers)

        # Check which projections land inside the mask
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)

        valid = in_front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

        # Look up mask values for valid projections
        inside = np.zeros(total_voxels, dtype=bool)
        valid_idx = np.where(valid)[0]
        inside[valid_idx] = mask[vi[valid_idx], ui[valid_idx]]

        votes += inside.astype(np.int32)
        dt = time.time() - t0
        print(f"    Cam {cam_id}: {inside.sum():,} voxels inside silhouette ({dt:.2f}s)")

    # Survive if seen in enough cameras
    survived = votes >= min_cameras
    surviving_points = voxel_centers[survived]

    # Build 3D occupancy grid
    occupancy = survived.reshape((nx, ny, nz))

    return surviving_points, occupancy, (x_min, y_min, z_min), (nx, ny, nz), (xs, ys, zs)


def save_ply(path, points):
    """Save point cloud as PLY file."""
    N = points.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{points[i, 0]:.3f} {points[i, 1]:.3f} {points[i, 2]:.3f}\n")


def save_obj(path, vertices, faces):
    """Save mesh as OBJ file."""
    with open(path, "w") as f:
        f.write("# Visual hull mesh\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def laplacian_smooth(vertices, faces, iterations=3, factor=0.3):
    """Simple Laplacian smoothing of mesh vertices."""
    from collections import defaultdict

    # Build adjacency
    adjacency = defaultdict(set)
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[f[i]].add(f[j])

    verts = vertices.copy()
    for _ in range(iterations):
        new_verts = verts.copy()
        for i in range(len(verts)):
            if adjacency[i]:
                neighbors = list(adjacency[i])
                avg = verts[neighbors].mean(axis=0)
                new_verts[i] = verts[i] + factor * (avg - verts[i])
        verts = new_verts
    return verts


def extract_mesh(occupancy, grid_origin, voxel_size, smooth_iterations=3):
    """Extract mesh from occupancy grid using marching cubes."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("  scikit-image not available, skipping mesh extraction.")
        print("  Install with: pip install scikit-image")
        return None, None

    # Pad occupancy grid with zeros so marching cubes can close the surface
    padded = np.pad(occupancy, 1, mode="constant", constant_values=0)

    # marching_cubes expects a scalar field; our bool grid works (True=1, False=0)
    volume = padded.astype(np.float32)

    try:
        verts, faces, normals, values = marching_cubes(volume, level=0.5)
    except Exception as e:
        print(f"  Marching cubes failed: {e}")
        return None, None

    # Convert from grid indices to world coordinates
    # Account for padding offset (-1 voxel) and half-voxel center offset
    x_min, y_min, z_min = grid_origin
    verts_world = np.zeros_like(verts)
    verts_world[:, 0] = (verts[:, 0] - 1) * voxel_size + x_min + voxel_size / 2
    verts_world[:, 1] = (verts[:, 1] - 1) * voxel_size + y_min + voxel_size / 2
    verts_world[:, 2] = (verts[:, 2] - 1) * voxel_size + z_min + voxel_size / 2

    if smooth_iterations > 0:
        verts_world = laplacian_smooth(verts_world, faces, iterations=smooth_iterations)

    return verts_world, faces


def main():
    parser = argparse.ArgumentParser(
        description="Visual hull reconstruction from multi-camera silhouettes."
    )
    parser.add_argument(
        "--frame", type=int, required=True, help="Frame number to reconstruct"
    )
    parser.add_argument(
        "--images_dir", type=str, required=True, help="Directory with extracted frames"
    )
    parser.add_argument(
        "--ba_poses",
        type=str,
        default=None,
        help="Path to ba_poses.json. If not given, reads from --project file.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Path to .redproj file (used to locate ba_poses.json)",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default=None,
        help="Directory with pre-computed binary masks (white=foreground)",
    )
    parser.add_argument(
        "--bg_color",
        type=str,
        default=None,
        help="Background color as R,G,B (e.g., '40,40,40'). Used for mask generation.",
    )
    parser.add_argument(
        "--mask_method",
        type=str,
        choices=["brightness", "bg_color", "green_ratio"],
        default="brightness",
        help="Mask generation method (default: brightness)",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=None,
        help="Threshold for mask generation (meaning depends on method)",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Bounding box as x_min,y_min,z_min,x_max,y_max,z_max (mm)",
    )
    parser.add_argument(
        "--voxel_size", type=float, default=5.0, help="Voxel size in mm (default: 5.0)"
    )
    parser.add_argument(
        "--min_cameras",
        type=int,
        default=None,
        help="Min cameras that must see voxel as foreground (default: all)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=3,
        help="Laplacian smoothing iterations for mesh (default: 3, 0 to disable)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as images_dir)",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="Save generated masks as PNG files for inspection",
    )

    args = parser.parse_args()

    # --- Locate ba_poses.json ---
    ba_poses_path = args.ba_poses
    if ba_poses_path is None:
        project_path = args.project
        if project_path is None:
            # Try default project
            default_proj = "/Users/johnsonr/red_demos/mouse_active1/mouse_active1.redproj"
            if os.path.exists(default_proj):
                project_path = default_proj
            else:
                print("ERROR: Must specify --ba_poses or --project")
                sys.exit(1)

        print(f"Reading project: {project_path}")
        with open(project_path, "r") as f:
            proj = json.load(f)

        calib_folder = proj["calibration_folder"]
        ba_poses_path = os.path.join(
            calib_folder, "summary_data", "bundle_adjustment", "ba_poses.json"
        )

    if not os.path.exists(ba_poses_path):
        print(f"ERROR: ba_poses.json not found at {ba_poses_path}")
        sys.exit(1)

    print(f"Loading calibration: {ba_poses_path}")
    cameras = load_calibration(ba_poses_path)
    print(f"  Loaded {len(cameras)} cameras")

    # Show camera positions
    positions = get_camera_positions(cameras)
    print(f"  Camera center of mass: [{positions.mean(axis=0)[0]:.1f}, "
          f"{positions.mean(axis=0)[1]:.1f}, {positions.mean(axis=0)[2]:.1f}]")

    # --- Load images ---
    camera_ids = sorted(cameras.keys())
    images = load_images_for_frame(args.images_dir, args.frame, camera_ids)
    print(f"  Loaded {len(images)} images for frame {args.frame}")

    if len(images) == 0:
        print("ERROR: No images found. Check --images_dir and --frame.")
        print(f"  Expected files like Cam<id>_{args.frame}.jpg in {args.images_dir}")
        sys.exit(1)

    # --- Generate or load masks ---
    if args.masks_dir:
        print(f"Loading masks from: {args.masks_dir}")
        masks = load_masks(args.masks_dir, args.frame, camera_ids)
        print(f"  Loaded {len(masks)} masks")
    else:
        print("Generating foreground masks...")
        masks = {}

        if args.bg_color:
            bg = [float(x) for x in args.bg_color.split(",")]
            thresh = args.mask_threshold if args.mask_threshold is not None else 40
            method_name = f"bg_color({args.bg_color}), threshold={thresh}"
            for cam_id, img in images.items():
                masks[cam_id] = generate_mask_bg_color(img, bg, threshold=thresh)
        elif args.mask_method == "green_ratio":
            thresh = args.mask_threshold if args.mask_threshold is not None else 1.1
            method_name = f"green_ratio, threshold={thresh}"
            for cam_id, img in images.items():
                masks[cam_id] = generate_mask_green_ratio(img, threshold=thresh)
        else:
            thresh = args.mask_threshold if args.mask_threshold is not None else 60
            method_name = f"brightness, threshold={thresh}"
            for cam_id, img in images.items():
                masks[cam_id] = generate_mask_brightness(img, threshold=thresh)

        print(f"  Method: {method_name}")
        for cam_id, mask in masks.items():
            fg_pct = mask.sum() / mask.size * 100
            print(f"    Cam {cam_id}: {fg_pct:.1f}% foreground")

    # Save masks if requested
    output_dir = args.output_dir or args.images_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.save_masks:
        masks_out = os.path.join(output_dir, "masks")
        os.makedirs(masks_out, exist_ok=True)
        for cam_id, mask in masks.items():
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(os.path.join(masks_out, f"Cam{cam_id}_{args.frame}.png"))
        print(f"  Saved masks to {masks_out}")

    # --- Bounding box ---
    if args.bbox:
        bbox = [float(x) for x in args.bbox.split(",")]
        assert len(bbox) == 6, "bbox must have 6 values: x_min,y_min,z_min,x_max,y_max,z_max"
    else:
        bbox = compute_default_bbox(cameras)

    print(f"  Bounding box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}] to "
          f"[{bbox[3]:.1f}, {bbox[4]:.1f}, {bbox[5]:.1f}]")
    dims = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]
    print(f"  Dimensions: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")

    # --- Min cameras ---
    active_cams = sorted(set(cameras.keys()) & set(masks.keys()))
    min_cameras = args.min_cameras if args.min_cameras is not None else len(active_cams)
    print(f"  Min cameras for survival: {min_cameras} / {len(active_cams)}")

    # --- Voxel carving ---
    print("\nVoxel carving...")
    t0 = time.time()
    points, occupancy, grid_origin, grid_shape, grid_axes = voxel_carving(
        cameras, masks, bbox, args.voxel_size, min_cameras
    )
    dt = time.time() - t0
    print(f"\n  Carving complete in {dt:.2f}s")
    print(f"  Surviving voxels: {len(points):,}")

    if len(points) == 0:
        print("\nWARNING: No voxels survived. Try:")
        print("  - Lower --min_cameras (e.g., half the camera count)")
        print("  - Adjust mask threshold (--mask_threshold)")
        print("  - Check --bbox covers the animal")
        print("  - Use --save_masks to inspect masks")
        # Still save empty PLY for debugging
        ply_path = os.path.join(output_dir, f"visual_hull_{args.frame}.ply")
        save_ply(ply_path, points)
        print(f"\n  Saved empty PLY: {ply_path}")
        return

    # Stats
    hull_min = points.min(axis=0)
    hull_max = points.max(axis=0)
    hull_size = hull_max - hull_min
    print(f"  Hull bounds: [{hull_min[0]:.1f}, {hull_min[1]:.1f}, {hull_min[2]:.1f}] to "
          f"[{hull_max[0]:.1f}, {hull_max[1]:.1f}, {hull_max[2]:.1f}]")
    print(f"  Hull size: {hull_size[0]:.1f} x {hull_size[1]:.1f} x {hull_size[2]:.1f} mm")
    vol_mm3 = len(points) * (args.voxel_size ** 3)
    print(f"  Volume: {vol_mm3:.0f} mm^3 ({vol_mm3 / 1e3:.1f} cm^3)")

    # --- Save PLY ---
    ply_path = os.path.join(output_dir, f"visual_hull_{args.frame}.ply")
    save_ply(ply_path, points)
    print(f"\n  Saved PLY: {ply_path}")

    # --- Mesh extraction ---
    print("\nExtracting mesh (marching cubes)...")
    verts, faces = extract_mesh(
        occupancy, grid_origin, args.voxel_size, smooth_iterations=args.smooth
    )

    if verts is not None and faces is not None and len(faces) > 0:
        obj_path = os.path.join(output_dir, f"visual_hull_{args.frame}.obj")
        save_obj(obj_path, verts, faces)
        print(f"  Mesh: {len(verts):,} vertices, {len(faces):,} faces")
        print(f"  Saved OBJ: {obj_path}")
    else:
        print("  No mesh generated.")

    print("\nDone.")


if __name__ == "__main__":
    main()
