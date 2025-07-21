import cv2 as cv
import os
from utils import *
from keypoints import *
import argparse
from datetime import datetime
import yaml
import random
from multiprocessing import Pool
import platform
from multiprocessing import get_context
import numpy as np
import re

parser = argparse.ArgumentParser()
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
    "-d",
    "--d_ball",
    type=float,
    help="Size of the ball in the same unit of calibration.",
)

args = parser.parse_args()
label_folder = args.label_folder
label_folder = os.path.normpath(label_folder)
output_folder = args.output_folder
select_indices = args.select_indices
d_ball = args.d_ball

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

if not select_indices:
    world_labels = world_labels_all
else:
    world_labels = {}
    for key, value in world_labels_all.items():
        world_labels[key] = value[select_indices]

# filter out invalid lables
world_labels_filter = {}
for name, value in world_labels.items():
    if not np.any(np.isnan(value)):
        # if all values are valid
        world_labels_filter[name] = value

calibration_folder = os.path.join(
    "/".join(label_folder.split("/")[:-1]), "calibration"
)

image_width = {}
image_height = {}
camera_params = {}
for cam in cameras:
    input_file_name = calibration_folder + "/{}.yaml".format(cam)
    fs = cv.FileStorage(input_file_name, cv.FILE_STORAGE_READ)
    if fs.isOpened():
        intrinsicMatrix = fs.getNode("camera_matrix").mat()
        rc_ext = fs.getNode("rc_ext").mat()
        tc_ext = fs.getNode("tc_ext").mat()
        camera_params[cam] = {
            "camera_matrix": intrinsicMatrix,
            "R": rc_ext,
            "T": tc_ext,
        }
        image_width[cam] = int(fs.getNode("image_width").real())
        image_height[cam] = int(fs.getNode("image_height").real())

# estimate r_ball per camera
ball_centers3d = np.squeeze(np.array(list(world_labels_filter.values())))
ball_frames = np.array(list(world_labels_filter.keys()))
d_ball_pxs = {}
for cam in cameras:
    rot_matrix = camera_params[cam]["R"]
    ball_c = rot_matrix @ ball_centers3d.T + camera_params[cam]["T"]
    z_c = ball_c[2]
    fx = float(camera_params[cam]["camera_matrix"][0, 0])
    d_ball_pixel = fx * d_ball / z_c
    d_ball_pixel_dict = dict(zip(ball_frames, d_ball_pixel))
    d_ball_pxs[cam] = d_ball_pixel_dict

labels_frames = np.asarray(list(world_labels_filter.keys()))
total_num_labels = len(labels_frames)

id_shuffled = np.arange(total_num_labels)
rng = np.random.default_rng(seed=42)
rng.shuffle(id_shuffled)

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


annotations_train = process_one_session_ball(
    select_folder_path,
    num_keypoints,
    train_image_frames,
    cameras,
    d_ball_pxs,
    0,
    image_width,
    image_height,
    select_indices,
)

annotations_val = process_one_session_ball(
    select_folder_path,
    num_keypoints,
    val_image_frames,
    cameras,
    d_ball_pxs,
    0,
    image_width,
    image_height,
    select_indices,
)


def save_yolo_data_opencv(output_folder, annotations, dset_mode, video_folder):
    for which_cam in annotations:
        dir_labels = os.path.join(
            output_folder,
            "labels",
            dset_mode,
        )
        dir_images = os.path.join(output_folder, "images", dset_mode)
        os.makedirs(dir_labels, exist_ok=True)
        os.makedirs(dir_images, exist_ok=True)
        for i in range(len(annotations[which_cam]["entry"])):
            fname_label = os.path.join(
                dir_labels,
                annotations[which_cam]["fname_annot"][i],
            )
            with open(fname_label, "w") as f:
                f.write(annotations[which_cam]["entry"][i])
        print("Saving jpeg for {} ...".format(which_cam))
        video_file = os.path.join(video_folder, "{}.mp4".format(which_cam))
        cap = cv.VideoCapture(video_file)
        image_frames = annotations[which_cam]["frame"]
        cap.set(cv.CAP_PROP_POS_FRAMES, image_frames[0])

        frame_num = image_frames[0]
        while frame_num >= image_frames[0] and frame_num <= image_frames[-1]:
            ret, frame = cap.read()
            if not ret:
                print("Missing fame: {}".format(frame_num))
                cap.release()
                break
            else:
                if frame_num in image_frames:
                    index = image_frames.index(frame_num)
                    fname_image = os.path.join(
                        dir_images, annotations[which_cam]["fname_img"][index]
                    )
                    cv.imwrite(fname_image, frame)
                frame_num = frame_num + 1
        cap.release()


# save jpeg images
video_folder = "/".join(label_folder.split("/")[:-1])

final_output_folder = os.path.join(output_folder, select_folder)
save_yolo_data_opencv(
    final_output_folder, annotations_train, "train", video_folder
)
save_yolo_data_opencv(
    final_output_folder, annotations_val, "val", video_folder
)

# write ymal config file
config = {
    "path": final_output_folder,
    "train": "images/train",
    "val": "images/val",
    "names": {0: "ball"},
}

with open(output_folder + "/config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)
