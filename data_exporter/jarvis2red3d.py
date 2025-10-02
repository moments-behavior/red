import cv2 as cv
import os
from utils import *
from keypoints import *
import argparse
from datetime import datetime
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_jarvis_folder", type=str, required=True)
parser.add_argument(
    "-s", "--skeleton", type=str, required=True, help="Name of the skeleton."
)
parser.add_argument("-f", "--filter", type=int, default=1)
parser.add_argument("-t", "--threshold", type=float, default=0.6)
parser.add_argument("-o", "--output_dir", type=str, default="predictions")


args = parser.parse_args()
threshold = args.threshold
input_jarvis_folder = args.input_jarvis_folder
skeleton = args.skeleton
use_filter = args.filter
output_dir = args.output_dir

labels_raw = load_jarvis_3d_csv_rats(input_jarvis_folder + "/data3D.csv")
print("Number of raw labels: {}".format(len(labels_raw)))

first_value = next(iter(labels_raw.values()))
num_keypoints = first_value.shape[0]

labels = {}
for key, value in labels_raw.items():
    if use_filter:
        if value[:, 3].mean() > threshold and value[:, 2].mean() < 500:
            labels[key] = value[:, :3]
    else:
        labels[key] = value[:, :3]
print("Number of valid labels: {}".format(len(labels)))

with open(input_jarvis_folder + "/info.yaml") as stream:
    try:
        jarvis_info = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

output_folder = os.path.join(
    jarvis_info["recording_path"], "{}".format(output_dir)
)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
calibration_folder = os.path.join(jarvis_info["recording_path"], "calibration")

# project to 2d using calibrations, save 2d points
cam_names = []
for file in glob.glob(calibration_folder + "/*.yaml"):
    file_name = file.split("/")
    cam_names.append(file_name[-1].split(".")[0])
cam_names.sort()

cam_list = []
for i in range(len(cam_names)):
    cam_params = {}
    filename = calibration_folder + "/{}.yaml".format(cam_names[i])
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
    cam_params["camera_matrix"] = fs.getNode("camera_matrix").mat()
    cam_params["distortion_coefficients"] = fs.getNode(
        "distortion_coefficients"
    ).mat()
    cam_params["tc_ext"] = fs.getNode("tc_ext").mat()
    cam_params["rc_ext"] = fs.getNode("rc_ext").mat()
    cam_params["image_width"] = int(fs.getNode("image_width").real())
    cam_params["image_height"] = int(fs.getNode("image_height").real())

    cam_list.append(cam_params)

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
output_folder_name = os.path.join(output_folder, dt_string)
os.makedirs(output_folder_name, exist_ok=True)

# save 2d keypoints
for camera_id in range(len(cam_names)):
    labels_converted = {}
    output_file_name = "{}.csv".format(cam_names[camera_id])
    for frame_id in labels.keys():
        projected_2d_keypoints = Project(
            labels[frame_id],
            cam_list[camera_id]["camera_matrix"],
            cam_list[camera_id]["distortion_coefficients"],
            cam_list[camera_id]["rc_ext"],
            cam_list[camera_id]["tc_ext"],
        )
        projected_2d_keypoints[:, 1] = (
            cam_list[camera_id]["image_height"] - projected_2d_keypoints[:, 1]
        )
        labels_converted[frame_id] = projected_2d_keypoints

    output_file = os.path.join(output_folder_name, output_file_name)
    print(output_file)
    with open(output_file, "w", newline="") as f:
        keypoint_writer = csv.writer(f, delimiter=",")
        keypoint_writer.writerow(["{}".format(skeleton)])
        for key, value in labels_converted.items():
            one_row = [key]
            for i in range(num_keypoints):
                one_row.append(i)
                one_row.append(value[i][0])
                one_row.append(value[i][1])
            keypoint_writer.writerow(one_row)

# save 3d keypoints
output_file_name = "keypoints3d.csv"
output_file = os.path.join(output_folder_name, output_file_name)
print(output_file)
with open(output_file, "w", newline="") as f:
    keypoint_writer = csv.writer(f, delimiter=",")
    keypoint_writer.writerow(["{}".format(skeleton)])
    for key, value in labels.items():
        one_row = [key]
        for i in range(num_keypoints):
            one_row.append(i)
            one_row.append(value[i][0])
            one_row.append(value[i][1])
            one_row.append(value[i][2])
        keypoint_writer.writerow(one_row)
