import cv2
import os
import argparse
import re
import numpy as np
import ctypes as C
import PyNvVideoCodec as nvc
from utils import *
import yaml


def cast_address_to_1d_bytearray(base_address, size):
    return np.ctypeslib.as_array(
        C.cast(base_address, C.POINTER(C.c_uint8)), shape=(size,)
    )


def save_image_label(
    simple_decoder,
    metadata,
    labels,
    image_frames,
    image_save_dir,
    label_save_dir,
):
    frames = simple_decoder.get_batch_frames_by_index(image_frames)
    for frame_idx in range(len(image_frames)):
        frame_num = image_frames[frame_idx]
        luma_base_addr = frames[frame_idx].GetPtrToPlane(0)
        new_array = cast_address_to_1d_bytearray(
            base_address=luma_base_addr, size=frames[frame_idx].framesize()
        )

        img = new_array.reshape(
            (metadata.height, metadata.width, -3)
        )  # or (height, width) for grayscale
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_file_name = f"Frame_{frame_num}_{which_cam}.jpg"
        img_save_to = os.path.join(image_save_dir, img_file_name)
        cv2.imwrite(img_save_to, img_bgr)

        label_one_frame = labels[frame_num]
        with open(
            os.path.join(
                label_save_dir,
                f"Frame_{frame_num}_{which_cam}" + ".txt",
            ),
            "w",
        ) as f:
            # rat keypoints
            line = create_yolo_rat_line(
                label_one_frame[:4],
                metadata.width,
                metadata.height,
                40,
            )
            f.write(line)
            f.write("\n")

            # ball keypoint
            line = create_yolo_ball_line(
                label_one_frame[4],
                metadata.width,
                metadata.height,
                120,
                4,
            )
            f.write(line)
            f.write("\n")


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--label_folder", type=str, required=True)
parser.add_argument("-o", "--output_folder", type=str, required=True)

args = parser.parse_args()
label_folder = args.label_folder
output_folder = args.output_folder

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

selected_kp_3d = os.path.join(select_folder_path, "keypoints3d.csv")
world_labels = csv_reader_red3d(selected_kp_3d)
world_labels_filter = {}
for name, value in world_labels.items():
    if not np.any(np.isnan(value)):
        # if all values are valid
        world_labels_filter[name] = value

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

video_folder = "/".join(label_folder.split("/")[:-1])
final_output_folder = os.path.join(output_folder, select_folder)
image_save_dir_train = final_output_folder + "/images/train/"
label_save_dir_train = final_output_folder + "/labels/train/"
image_save_dir_val = final_output_folder + "/images/val/"
label_save_dir_val = final_output_folder + "/labels/val/"
os.makedirs(image_save_dir_train, exist_ok=True)
os.makedirs(label_save_dir_train, exist_ok=True)
os.makedirs(image_save_dir_val, exist_ok=True)
os.makedirs(label_save_dir_val, exist_ok=True)


def create_yolo_rat_line(rat_keypoints, frame_width, frame_height, margin):
    rat_x = rat_keypoints[:, 0] / frame_width
    rat_y = rat_keypoints[:, 1] / frame_height

    num_rat_keypoints = rat_x.shape[0]

    margin_x = margin / frame_width
    margin_y = margin / frame_height
    rat_x_min = np.max((0, np.min(rat_x) - margin_x))
    rat_x_max = np.min((frame_width, np.max(rat_x) + margin_x))
    rat_y_min = np.max((0, np.min(rat_y) - margin_y))
    rat_y_max = np.min((frame_height, np.max(rat_y) + margin_y))
    rat_center_x = (rat_x_min + rat_x_max) / 2.0
    rat_center_y = (rat_y_min + rat_y_max) / 2.0
    rat_w = rat_x_max - rat_x_min
    rat_h = rat_y_max - rat_y_min

    line = "0 {} {} {} {}".format(rat_center_x, rat_center_y, rat_w, rat_h)
    for i in range(num_rat_keypoints):
        line += " {} {} {}".format(rat_x[i], rat_y[i], 2)
    return line


def create_yolo_ball_line(
    ball_keypoints, frame_width, frame_height, ball_bb_size, num_zero_padding
):
    ball_center_x = ball_keypoints[0] / frame_width
    ball_center_y = ball_keypoints[1] / frame_height
    ball_w = ball_bb_size / frame_width
    ball_h = ball_bb_size / frame_height
    line = "1 {} {} {} {}".format(ball_center_x, ball_center_y, ball_w, ball_h)
    for i in range(num_zero_padding):
        if i == 0:
            line += " {} {} {}".format(ball_center_x, ball_center_y, 2)
        else:
            line += " 0 0 0"
    return line


use_device_memory = 0  # 0 = system memory, 1 = device memory
for which_cam in cameras:
    # Initialize decoder
    video_file_name = video_folder + "/{}.mp4".format(which_cam)
    print(video_file_name)
    simple_decoder = nvc.SimpleDecoder(
        video_file_name,
        gpu_id=0,
        use_device_memory=use_device_memory,
        output_color_type=nvc.OutputColorType.RGB,
    )

    # Get video metadata
    metadata = simple_decoder.get_stream_metadata()
    # print(metadata)
    print(f"Total frames: {metadata.num_frames}")
    print(f"FPS: {metadata.average_fps}")

    labels = csv_reader_red2d(
        select_folder_path + "/{}.csv".format(which_cam),
        metadata.height,
    )

    save_image_label(
        simple_decoder,
        metadata,
        labels,
        train_image_frames,
        image_save_dir_train,
        label_save_dir_train,
    )

    save_image_label(
        simple_decoder,
        metadata,
        labels,
        val_image_frames,
        image_save_dir_val,
        label_save_dir_val,
    )

# write ymal config file
config = {
    "path": final_output_folder,
    "train": "images/train",
    "val": "images/val",
    "kpt_shape": [4, 3],
    "names": {0: "rat", 1: "ball"},
}

with open(output_folder + "/config_{}.yaml".format(select_folder), "w") as f:
    yaml.dump(config, f, default_flow_style=False)
