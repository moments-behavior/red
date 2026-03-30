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
    "-w",
    "--working_dir",
    type=str,
    required=True,
    help="labeled_data and project file (project.json or project.redproj) need to be under this directory.",
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
    "-l",
    "--label_subfolder",
    type=str,
    default=None,
    help="Use this subfolder of labeled_data (e.g. '.' for flat layout, or a datetime folder). If not set, auto-selects most recent datetime subfolder or flat labeled_data.",
)

args = parser.parse_args()
working_dir = os.path.abspath(args.working_dir)
output_folder = args.output_folder
select_indices = args.select_indices
margin_pixel = args.margin
label_subfolder_arg = args.label_subfolder

label_folder = os.path.join(working_dir, "labeled_data")
# Support both project.json (new) and project.redproj (old)
project_file = os.path.join(working_dir, "project.json")
if not os.path.isfile(project_file):
    project_file = os.path.join(working_dir, "project.redproj")
if not os.path.isfile(project_file):
    raise SystemExit(
        "No project file found. Expected project.json or project.redproj in: {}".format(working_dir)
    )
with open(project_file, "r") as f:
    project = json.load(f)

# Support older project files: resolve paths relative to working_dir
calibration_folder = project.get("calibration_folder", "")
video_folder = project.get("media_folder", "")
if not calibration_folder:
    raise SystemExit("Project file missing 'calibration_folder'. Check {}.".format(project_file))
if not video_folder:
    raise SystemExit("Project file missing 'media_folder'. Check {}.".format(project_file))
if not os.path.isabs(calibration_folder):
    calibration_folder = os.path.normpath(os.path.join(working_dir, calibration_folder))
if not os.path.isabs(video_folder):
    video_folder = os.path.normpath(os.path.join(working_dir, video_folder))

datetime_pattern = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")

if label_subfolder_arg is not None:
    if label_subfolder_arg == "." or label_subfolder_arg == "":
        select_folder_path = label_folder
        select_folder = os.path.basename(os.path.normpath(working_dir)) + "_labels"
    else:
        select_folder_path = os.path.join(label_folder, label_subfolder_arg)
        select_folder = label_subfolder_arg
    if not os.path.isdir(select_folder_path):
        raise SystemExit("Label subfolder does not exist: {}".format(select_folder_path))
    print("Using label folder: {}".format(select_folder_path))
else:
    matching_folders = [
        name
        for name in os.listdir(label_folder)
        if os.path.isdir(os.path.join(label_folder, name))
        and datetime_pattern.match(name)
    ]
    matching_folders.sort()
    if matching_folders:
        select_folder = matching_folders[-1]
        select_folder_path = os.path.join(label_folder, select_folder)
        print("Select most recent label: {}".format(select_folder))
    else:
        # Older RED: flat labeled_data (no datetime subfolders)
        flat_kp3d = os.path.join(label_folder, "keypoints3d.csv")
        if os.path.isfile(flat_kp3d):
            select_folder_path = label_folder
            select_folder = os.path.basename(os.path.normpath(working_dir)) + "_labels"
            print("Using flat labeled_data (no datetime subfolders): {}".format(select_folder_path))
        else:
            raise SystemExit(
                "No datetime subfolders in labeled_data and no keypoints3d.csv in labeled_data. "
                "Use -l . to use labeled_data as label folder, or -l <subfolder_name>."
            )

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

if select_indices is None:
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
rng = np.random.default_rng(seed=42)
np.random.shuffle(id_shuffled)
num_train = int(np.floor(total_num_labels * 0.9))
print(
    "Train set: {}, validation set: {}.".format(
        num_train, total_num_labels - num_train
    )
)
train_ids = id_shuffled[:num_train]
train_ids = np.sort(train_ids)
val_ids = id_shuffled[num_train:]
val_ids = np.sort(val_ids)
# split frames to train and val
train_image_frames = labels_frames[train_ids]
val_image_frames = labels_frames[val_ids]

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

print("Prepared dataset at {}.".format(output_folder))


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
    print(output_filename)


# save jpeg images
map_frame_to_mode = {}
all_image_frames = []

for img in train_image_frames:
    map_frame_to_mode[img] = "train"
    all_image_frames.append(img)

for img in val_image_frames:
    map_frame_to_mode[img] = "val"
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
