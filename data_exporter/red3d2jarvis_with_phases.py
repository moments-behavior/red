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
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-i", "--label_folder", type=str, required=True)
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
)
parser.add_argument("-p", "--phase", type=str, required=True)

args = parser.parse_args()
label_folder = args.label_folder
label_folder = os.path.normpath(label_folder)
output_folder = args.output_folder
select_indices = args.select_indices
margin_pixel = args.margin
phase = args.phase
config = args.config

with open(config, "r") as file:
    config = json.load(file)

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
# cameras = get_all_cams_in_labeled_folder(select_folder_path)
if phase == "bridge":
    cameras = [
        "Cam710038",
        "Cam2002480",
        "Cam2002485",
        "Cam2002483",
        "Cam2002492",
    ]
elif phase == "arena":
    cameras = [
        "Cam710038",
        "Cam2002490",
        "Cam2002496",
        "Cam2002488",
        "Cam2002489",
    ]
elif phase == "arena24":
    cameras = [
        "Cam710038",
        "Cam2002490",
        "Cam2002496",
        "Cam2002488",
        "Cam2002489",
        "Cam2002479",
        "Cam2002484",
        "Cam2002495",
        "Cam2002493",
        "Cam2002481",
        "Cam2002482",
        "Cam2002494",
        "Cam2002491",
    ]
else:
    print("Unrecognized phase. ")
    sys.exit(1)

# get skeleton name
selected_kp_3d = os.path.join(select_folder_path, "keypoints3d.csv")
if config["load_skeleton_from_json"]:
    skeleton = config["skeleton_file"]
    keypoints_names, skeleton_names, num_keypoints = (
        load_skeleton_json_format_for_jarvis(skeleton)
    )
else:
    skeleton = config["skeleton_name"]
    keypoints_names, skeleton_names, num_keypoints = skeleton_selector[
        skeleton
    ]()

print("Skeleton:", skeleton)
world_labels_all = csv_reader_red3d(selected_kp_3d)

# filter out invalid lables
if phase == "bridge":
    world_labels_filterd = {}
    for name, value in world_labels_all.items():
        if not np.any(np.isnan(value)):
            # check if on the bridge
            if np.all(value[:, 0] > 500) and np.all(np.abs(value[:, 1]) < 250):
                world_labels_filterd[name] = value
else:
    world_labels_filterd = {}
    for name, value in world_labels_all.items():
        if not np.any(np.isnan(value)):
            if not (
                np.all(value[:, 0] > 500) and np.all(np.abs(value[:, 1]) < 250)
            ):
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

calibration_folder = config["calibration_folder"]
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
video_folder = config["media_folder"]

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
