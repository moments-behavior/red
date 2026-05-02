from keypoints import *
import re
import argparse
import numpy as np
import json
from utils import *
import glob
import cv2 as cv
from multiprocessing import Pool
import platform
from multiprocessing import get_context
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--project_dir",
    type=str,
    required=True,
    help="labeled_data and project.redproj need to be under this directory.",
)
parser.add_argument("-o", "--output_folder", type=str, required=True)
parser.add_argument(
    "-s",
    "--select_indices",
    nargs="+",
    type=int,
    help="List of numbers, for instance: -s 0 1 2 3",
    default=[],
)
parser.add_argument(
    "-m",
    "--margin",
    type=float,
    help="Margin in pixel to add when deciding bounding box.",
    required=True,
)
parser.add_argument(
    "-e",
    "--edges",
    nargs="+",
    type=int,
    help="Pairs of numbers. Optional for sub selecting indices.",
)
parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.8,
    help="Fraction of (non-test) data used for training. Rest goes to validation. Default 0.8.",
)
parser.add_argument(
    "-t",
    "--test_ratio",
    type=float,
    default=0.0,
    help="Fraction held out as test set (excluded from JARVIS training/val). Default 0.0 means no test set.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for the train/val/test shuffle.",
)

args = parser.parse_args()
project_dir = args.project_dir
output_folder = args.output_folder
select_indices = args.select_indices
margin_pixel = args.margin

label_folder = os.path.join(project_dir, "labeled_data")
redproj = os.path.join(project_dir, "project.redproj")
with open(redproj, "r") as f:
    project = json.load(f)
calibration_folder = project["calibration_folder"]
video_folder = project["media_folder"]

datetime_pattern = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")

matching_folders = [
    name
    for name in os.listdir(label_folder)
    if os.path.isdir(os.path.join(label_folder, name))
    and datetime_pattern.match(name)
]
matching_folders.sort()
select_folder = matching_folders[-1]
print("Select most recent label: {}".format(select_folder))

select_folder_path = os.path.join(label_folder, select_folder)
cameras = get_all_cams_in_labeled_folder(select_folder_path)

if project["load_skeleton_from_json"]:
    skeleton = project["skeleton_file"]
    keypoints_names, skeleton_names, num_keypoints = (
        load_skeleton_json_format_for_jarvis(skeleton)
    )
else:
    skeleton = project["skeleton_name"]
    keypoints_names, skeleton_names, num_keypoints = skeleton_selector[
        skeleton
    ]()

selected_kp_3d = os.path.join(select_folder_path, "keypoints3d.csv")
world_labels_all = csv_reader_red3d(selected_kp_3d)

if not select_indices:  # None or empty list (the argparse default)
    world_labels = world_labels_all
else:
    world_labels = {}
    for key, value in world_labels_all.items():
        world_labels[key] = value[select_indices]

# filter out invalid lables
world_labels_filterd = {}
for name, value in world_labels.items():
    if not np.any(np.isnan(value)):
        world_labels_filterd[name] = value

labels_frames = np.asarray(list(world_labels_filterd.keys()))
total_num_labels = len(labels_frames)

id_shuffled = np.arange(total_num_labels)
rng = np.random.default_rng(seed=args.seed)
rng.shuffle(id_shuffled)

num_test = int(np.floor(total_num_labels * args.test_ratio))
num_remaining = total_num_labels - num_test
num_train = int(np.floor(num_remaining * args.train_ratio))
num_val = num_remaining - num_train
print(
    "Train: {}, Val: {}, Test: {} (out of {} labeled frames).".format(
        num_train, num_val, num_test, total_num_labels
    )
)
test_ids = np.sort(id_shuffled[:num_test])
train_ids = np.sort(id_shuffled[num_test : num_test + num_train])
val_ids = np.sort(id_shuffled[num_test + num_train :])
# split frames to train, val, test
train_image_frames = labels_frames[train_ids]
val_image_frames = labels_frames[val_ids]
test_image_frames = labels_frames[test_ids]

trial_name = select_folder

image_width = {}
image_height = {}
for cam in cameras:
    input_file_name = calibration_folder + "/{}.yaml".format(cam)
    fs = cv.FileStorage(input_file_name, cv.FILE_STORAGE_READ)
    if fs.isOpened():
        image_width[cam] = int(fs.getNode("image_width").real())
        image_height[cam] = int(fs.getNode("image_height").real())


annotations, images, frame_set_one = process_one_session(
    trial_name,
    select_folder_path,
    num_keypoints,
    train_image_frames,
    cameras,
    image_width,
    image_height,
    select_keypoints_idx=select_indices,
    margin_pixel=margin_pixel,
)
# Collect per-axis bbox/image ratios per annotation for the CenterDetect
# bbox-size check at the end. JARVIS does a non-uniform stretch resize
# to (IMAGE_SIZE, IMAGE_SIZE), so we have to track both axes independently.
# Train alone is representative because val/test see the same cameras.
img_dims_by_id = {im["id"]: (im["width"], im["height"]) for im in images}
train_bbox_axis_ratios = []  # list of (w/img_w, h/img_h)
train_bbox_max_pixels = []  # list of max(w, h) in raw pixels
for ann in annotations:
    if "bbox" not in ann:
        continue
    _, _, w, h = ann["bbox"]
    dims = img_dims_by_id.get(ann["image_id"])
    if dims and (w > 0 or h > 0):
        img_w, img_h = dims
        train_bbox_axis_ratios.append((w / img_w, h / img_h))
        train_bbox_max_pixels.append(max(w, h))

# ---------------------------------------------------------------------------
# JARVIS training-config suggestions
# ---------------------------------------------------------------------------
# All in one place so the user sees them as a coherent block during config.
import itertools

print("\n=== JARVIS training-config suggestions ===")

# HybridNet: ROI_CUBE_SIZE from per-frame max-axis-span. Conservative —
# uses max across frames so no frame ever exceeds the cube. JARVIS would
# silently drop any frame that does (or warn, with our patched JARVIS).
spans = []  # per-frame max-axis-span; reused downstream for px/mm scale
for kps in world_labels_filterd.values():
    if kps.shape[0] >= 2:
        spans.append(float(np.max(np.ptp(kps, axis=0))))
spans.sort()
if spans:
    n = len(spans)
    p95 = spans[int(0.95 * (n - 1))]
    p99 = spans[int(0.99 * (n - 1))]
    max_e = spans[-1]
    print(
        f"3D label extent (max axis span per frame, mm): "
        f"median={spans[n // 2]:.0f}, p95={p95:.0f}, p99={p99:.0f}, "
        f"max={max_e:.0f}"
    )
    # Conservative: round up (max × 1.20) to a multiple of 4 * grid_spacing,
    # so every frame fits with 20% safety margin (no silent drops).
    target = max_e * 1.20
    print("HYBRIDNET.ROI_CUBE_SIZE (1.20 x max axis span, no dropped frames):")
    for gs in (4, 6, 8):
        divisor = 4 * gs
        suggested = int(np.ceil(target / divisor)) * divisor
        print(f"  for GRID_SPACING={gs}: {suggested} mm")

# HybridNet: GT_SIGMA_MM and GRID_SPACING from the *structurally* closest
# pair of keypoints. For each pair (i, j), compute the median distance
# across frames; then take the minimum of those medians. This is robust
# to transient overlaps (two keypoints occasionally touching) — only
# pairs that are usually close get picked.
n_frames_collected = 0
n_kps_seen = None
pair_dists = {}  # (i, j) -> list of distances
for kps in world_labels_filterd.values():
    if kps.shape[0] < 2:
        continue
    if n_kps_seen is None:
        n_kps_seen = kps.shape[0]
        for i, j in itertools.combinations(range(n_kps_seen), 2):
            pair_dists[(i, j)] = []
    if kps.shape[0] != n_kps_seen:
        continue  # skip frames with different keypoint counts
    for i, j in itertools.combinations(range(n_kps_seen), 2):
        pair_dists[(i, j)].append(float(np.linalg.norm(kps[i] - kps[j])))
    n_frames_collected += 1

if pair_dists and n_frames_collected:
    structural = []
    for (i, j), dists in pair_dists.items():
        dists.sort()
        structural.append((dists[len(dists) // 2], i, j))
    structural.sort()

    def _name(k):
        return keypoints_names[k] if k < len(keypoints_names) else f"kp{k}"

    # Approximate pixels-per-mm at typical animal scale, using the median
    # bbox size in pixels divided by the median 3D extent in mm.
    px_per_mm = None
    if train_bbox_max_pixels and spans:
        median_bbox_px = sorted(train_bbox_max_pixels)[
            len(train_bbox_max_pixels) // 2
        ]
        median_extent_mm = spans[len(spans) // 2]
        if median_extent_mm > 0:
            px_per_mm = median_bbox_px / median_extent_mm

    def _mm_to_px(mm):
        return f", ~{mm * px_per_mm:.1f}px" if px_per_mm else ""

    print(f"\nClosest-pair distribution (median across frames):")
    for rank, (dist, i, j) in enumerate(structural[:5], start=1):
        print(f"  #{rank}: {dist:5.1f} mm  ({_name(i)} <-> {_name(j)})")

    closest, ki, kj = structural[0]
    suggested_sigma = max(1, int(round(closest / 2)))
    suggested_grid = max(1, int(round(suggested_sigma / 2)))
    print(
        f"HYBRIDNET.GT_SIGMA_MM:  {suggested_sigma} mm"
        f"{_mm_to_px(suggested_sigma)}  (closest-pair / 2)"
    )
    print(
        f"HYBRIDNET.GRID_SPACING: {suggested_grid} mm"
        f"{_mm_to_px(suggested_grid)}  "
        "(sigma / 2; finer = better localization, ~8x memory per halving)"
    )
    # Practicality check: grid < ~3 mm explodes memory on most GPUs.
    # 480 mm cube at grid 1 mm = 110M voxels per frame; at grid 3 mm = 4M.
    if suggested_grid < 3:
        practical_grid = 3
        practical_sigma = max(suggested_sigma, practical_grid * 2)
        print(
            f"\n  WARN: the suggested values are derived literally but are "
            "impractical."
        )
        print(
            f"  Closely-spaced keypoints ({_name(ki)} and {_name(kj)} are "
            f"only {closest:.1f} mm apart) force a {suggested_grid} mm "
            "voxel grid — that's tens of millions of voxels in a 480 mm "
            "cube, which won't fit on most GPUs."
        )
        print("  Three choices:")
        print(
            f"    A) Use practical values "
            f"(sigma={practical_sigma}{_mm_to_px(practical_sigma)}, "
            f"grid={practical_grid}{_mm_to_px(practical_grid)}) and accept "
            "some confusion between this pair; post-process if needed."
        )
        print(
            f"    B) Drop or merge one of '{_name(ki)}'/'{_name(kj)}' from "
            "your labels — for many rigs they're functionally redundant. "
            "Look at the distribution above to see what the next-closest "
            "pairs would force."
        )
        print(
            "    C) Try the literal values and see if HybridNet OOMs "
            "(probably will at batch >= 1)."
        )

# CenterDetect: IMAGE_SIZE from smallest worst-axis bbox at CD input.
if train_bbox_axis_ratios:
    smallest_max_ratio = min(max(rw, rh) for rw, rh in train_bbox_axis_ratios)
    required = 32.0 / smallest_max_ratio if smallest_max_ratio > 0 else 0
    suggested_image_size = max(64, ((int(required) + 63) // 64) * 64)
    smallest_at_suggested = smallest_max_ratio * suggested_image_size
    print(
        f"CENTERDETECT.IMAGE_SIZE: {suggested_image_size}  "
        f"(smallest animal at CD input = {smallest_at_suggested:.0f} px; "
        ">= 32 px reliable. JARVIS uses non-uniform stretch resize, so "
        "this checks the worst-squashed axis.)"
    )

set_of_frames = {trial_name: frame_set_one}
if select_indices:
    keypoints_names_selected = [keypoints_names[i] for i in select_indices]
    annotation_num_kps = len(select_indices)
    if args.edges:
        skeleton_edge_pairs = list(zip(args.edges[::2], args.edges[1::2]))
        skeleton_names = edges_to_jarvis_skeleton(
            skeleton_edge_pairs, keypoints_names_selected
        )
    else:
        print("You might need to update skeleton. Specify new edges with -e.")
else:
    keypoints_names_selected = keypoints_names
    annotation_num_kps = num_keypoints

annotation_json_train = generate_annotation_file(
    trial_name,
    keypoints_names_selected,
    skeleton_names,
    annotation_num_kps,
    cameras,
    annotations,
    images,
    set_of_frames,
)

annotations, images, frame_set_one = process_one_session(
    trial_name,
    select_folder_path,
    num_keypoints,
    val_image_frames,
    cameras,
    image_width,
    image_height,
    select_keypoints_idx=select_indices,
    margin_pixel=margin_pixel,
)
set_of_frames = {trial_name: frame_set_one}
annotation_json_val = generate_annotation_file(
    trial_name,
    keypoints_names_selected,
    skeleton_names,
    annotation_num_kps,
    cameras,
    annotations,
    images,
    set_of_frames,
)


annotation_path = os.path.join(output_folder, "annotations/")
os.makedirs(annotation_path, exist_ok=True)

with open(annotation_path + "instances_train.json", "w") as f:
    json.dump(annotation_json_train, f)

with open(annotation_path + "instances_val.json", "w") as f:
    json.dump(annotation_json_val, f)

if num_test > 0:
    annotations, images, frame_set_one = process_one_session(
        trial_name,
        select_folder_path,
        num_keypoints,
        test_image_frames,
        cameras,
        image_width,
        image_height,
        select_keypoints_idx=select_indices,
        margin_pixel=margin_pixel,
    )
    set_of_frames = {trial_name: frame_set_one}
    annotation_json_test = generate_annotation_file(
        trial_name,
        keypoints_names_selected,
        skeleton_names,
        annotation_num_kps,
        cameras,
        annotations,
        images,
        set_of_frames,
    )
    with open(annotation_path + "instances_test.json", "w") as f:
        json.dump(annotation_json_test, f)
    # Save just the held-out frame indices for downstream evaluation.
    with open(os.path.join(output_folder, "test_frames.json"), "w") as f:
        json.dump(
            {
                "trial_name": trial_name,
                "frames": [int(x) for x in test_image_frames.tolist()],
            },
            f,
            indent=2,
        )

print("\n=== Prepared dataset at {}. ===".format(output_folder))

# save calibration
save_calib_folder = os.path.join(output_folder, "calib_params", trial_name)
os.makedirs(save_calib_folder, exist_ok=True)

# Jarvis uses crazy transpose for some reason
for cam in cameras:
    input_file_name = calibration_folder + "/{}.yaml".format(cam)
    fs = cv.FileStorage(input_file_name, cv.FILE_STORAGE_READ)
    intrinsicMatrix = fs.getNode("camera_matrix").mat()
    intrinsicMatrix = intrinsicMatrix.T
    distortionCoefficients = fs.getNode("distortion_coefficients").mat()
    distortionCoefficients = distortionCoefficients.T
    R = fs.getNode("rc_ext").mat()
    R = R.T
    T = fs.getNode("tc_ext").mat()

    output_filename = save_calib_folder + "/{}.yaml".format(cam)
    s = cv.FileStorage(output_filename, cv.FileStorage_WRITE)
    s.write("image_width", image_width[cam])
    s.write("image_height", image_height[cam])
    s.write("intrinsicMatrix", intrinsicMatrix)
    s.write("distortionCoefficients", distortionCoefficients)
    s.write("R", R)
    s.write("T", T)
    s.release()


# save jpeg images
map_frame_to_mode = {}
all_image_frames = []

for img in train_image_frames:
    map_frame_to_mode[img] = "train"
    all_image_frames.append(img)

for img in val_image_frames:
    map_frame_to_mode[img] = "val"
    all_image_frames.append(img)

if num_test > 0:
    for img in test_image_frames:
        map_frame_to_mode[img] = "test"
        all_image_frames.append(img)

all_image_frames = np.asarray(all_image_frames)
all_image_frames = np.sort(all_image_frames)
all_jobs = []

for camera in cameras:
    all_jobs.append(
        [
            trial_name,
            camera,
            video_folder,
            output_folder,
            map_frame_to_mode,
            all_image_frames,
        ]
    )

num_jobs = len(all_jobs)

if platform.system() == "Darwin":  # fix for macOS
    print("threadpooling for macos")
    with get_context("fork").Pool(num_jobs) as p:
        p.map(multiprocess_save_jpegs_opencv, all_jobs)
else:
    with Pool(num_jobs) as p:
        p.map(multiprocess_save_jpegs_opencv, all_jobs)
