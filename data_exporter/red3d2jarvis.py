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

# get skeleton name
selected_kp_3d = os.path.join(select_folder_path, "keypoints3d.csv")
skeleton = get_skeleton_name(selected_kp_3d)
print("Skeleton:", skeleton)

if skeleton.endswith("json"):
    keypoints_names, skeleton_names, num_keypoints = (
        load_skeleton_json_format_for_jarvis(skeleton)
    )
else:
    keypoints_names, skeleton_names, num_keypoints = skeleton_selector[
        skeleton
    ]()


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

# ---------------------------------------------------------------------------
# JARVIS training-config suggestions (HybridNet)
# ---------------------------------------------------------------------------
# Print stats from the labeled keypoints so the user can pick
# HYBRIDNET.ROI_CUBE_SIZE, GT_SIGMA_MM, and GRID_SPACING during JARVIS
# config setup. JARVIS silently drops any frame whose 3D extent exceeds
# ROI_CUBE_SIZE, so under-sizing the cube is a quiet way to lose training
# data.
spans = []
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
        "3D label extent (max axis span per frame, mm): "
        f"median={spans[n // 2]:.0f}, p95={p95:.0f}, p99={p99:.0f}, max={max_e:.0f}"
    )
    # JARVIS requires ROI_CUBE_SIZE % (4 * GRID_SPACING) == 0.
    # Round up (max + 20% margin) to the right multiple for each common spacing.
    target = int(max_e * 1.20)
    print("Suggested HYBRIDNET.ROI_CUBE_SIZE (20% margin over max):")
    for gs in (4, 6, 8):
        divisor = 4 * gs
        suggested = ((target + divisor - 1) // divisor) * divisor
        print(f"  for GRID_SPACING={gs}: {suggested} mm")

# Closest-pair distance per frame → suggests GT sigma and grid spacing.
# Sigma should be ~half the smallest distance the model needs to resolve.
import itertools
min_dists = []
for kps in world_labels_filterd.values():
    if kps.shape[0] < 2:
        continue
    pair_dists = [
        float(np.linalg.norm(kps[i] - kps[j]))
        for i, j in itertools.combinations(range(kps.shape[0]), 2)
    ]
    if pair_dists:
        min_dists.append(min(pair_dists))
if min_dists:
    min_dists.sort()
    closest = min_dists[len(min_dists) // 2]  # median across frames
    suggested_sigma = max(1, int(round(closest / 2)))
    suggested_grid = max(1, int(round(suggested_sigma / 2)))
    print(
        "Closest-pair 3D distance (median min across frames): "
        f"{closest:.1f} mm"
    )
    print(
        f"Suggested HYBRIDNET.GT_SIGMA_MM:    {suggested_sigma} mm  "
        "(closest-pair / 2)"
    )
    print(
        f"Suggested HYBRIDNET.GRID_SPACING:   {suggested_grid} mm  "
        "(sigma / 2; finer = better localization, ~8x memory per halving)"
    )

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
for ann in annotations:
    if "bbox" not in ann:
        continue
    _, _, w, h = ann["bbox"]
    dims = img_dims_by_id.get(ann["image_id"])
    if dims and (w > 0 or h > 0):
        img_w, img_h = dims
        train_bbox_axis_ratios.append((w / img_w, h / img_h))
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

print("Prepared dataset at {}.".format(output_folder))

# CenterDetect bbox-size check: how many pixels does the animal occupy
# at the CenterDetect input resolution? JARVIS does a non-uniform stretch
# to (IMAGE_SIZE, IMAGE_SIZE) — so each axis is squashed independently.
# We report the bbox's largest stretched dimension (max of the two) and
# flag based on the smallest such value across all annotations.
if train_bbox_axis_ratios:
    # Per annotation: bbox's "characteristic size" at CD input is
    # max(rw * IMAGE_SIZE, rh * IMAGE_SIZE). The smallest of these across
    # the dataset determines reliability. We want smallest >= 32 px.
    smallest_max_ratio = min(
        max(rw, rh) for rw, rh in train_bbox_axis_ratios
    )
    # Required IMAGE_SIZE: 32 / smallest_max_ratio, rounded up to multiple of 64.
    required = 32.0 / smallest_max_ratio if smallest_max_ratio > 0 else 0
    suggested_image_size = max(64, ((int(required) + 63) // 64) * 64)
    smallest_at_suggested = smallest_max_ratio * suggested_image_size
    print(
        f"\nSuggested CENTERDETECT.IMAGE_SIZE: {suggested_image_size}  "
        f"(smallest animal at CD input = {smallest_at_suggested:.0f} px; "
        "must be >= 32 px for reliable centroid detection. JARVIS uses "
        "non-uniform stretch resize, so this checks the worst-squashed axis.)"
    )


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
