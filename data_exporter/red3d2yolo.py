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


np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--label_folder', type=str, required=True)
parser.add_argument('-m', '--mode', type=str, default='point2bbox')
parser.add_argument('-o', '--output_folder', type=str, required=True)
parser.add_argument('-d', '--d_ball', type=int, default=100)


args = parser.parse_args()
label_folder = args.label_folder
label_folder = os.path.normpath(label_folder)
det_mode = args.mode
output_folder = args.output_folder

cameras = get_all_cams_in_labeled_folder(label_folder)


# Save annotations
world_point_folder = label_folder + "/worldKeyPoints"
all_files = glob.glob(world_point_folder + "/*")
all_files.sort()
select_most_recent_labels = all_files[-1]
print("Select most recent label: {}".format(select_most_recent_labels))

selected_annotation = select_most_recent_labels.split("/")[-1][10:]
selected_annotation = selected_annotation.split(".")[0]
world_labels = csv_reader_rats(select_most_recent_labels, 1, three_d=True)

# filter out invalid lables
world_labels_filterd = {}
for name, value in world_labels.items():
    if not np.any(value == 1E7):
        world_labels_filterd[name] = value

labels_frames = np.asarray(list(world_labels_filterd.keys()))
total_num_labels = len(labels_frames)

id_shuffled = np.arange(total_num_labels)
np.random.shuffle(id_shuffled)
num_train = int(np.floor(total_num_labels * 0.7))
num_validation = int(np.floor(total_num_labels * 0.2))
print("Train set: {}, Validation set: {}, Test set: {}".format(
    num_train, num_validation, total_num_labels - num_train - num_validation))
train_ids = id_shuffled[:num_train]
train_ids = np.sort(train_ids)
val_ids = id_shuffled[num_train:num_train + num_validation]
val_ids = np.sort(val_ids)
test_ids = id_shuffled[num_train + num_validation:]
test_ids = np.sort(test_ids)
# split frames to train and val
train_image_frames = labels_frames[train_ids]
val_image_frames = labels_frames[val_ids]
test_image_frames = labels_frames[test_ids]

trial_name = selected_annotation

num_keypoints = 1
id_ball = 0
d_ball = args.d_ball

image_width = 1464
image_height= 1936

# export annotations
annotations = process_one_session_ball(trial_name, label_folder, num_keypoints,
                                       selected_annotation, train_image_frames, cameras, d_ball, id_ball, image_width, image_height)

create_yolo_annotation_files(
    output_folder, trial_name, annotations, cameras, "train")

annotations = process_one_session_ball(trial_name, label_folder, num_keypoints,
                                       selected_annotation, val_image_frames, cameras, d_ball, id_ball, image_width, image_height)
create_yolo_annotation_files(
    output_folder, trial_name, annotations, cameras, "valid")

annotations = process_one_session_ball(trial_name, label_folder, num_keypoints,
                                       selected_annotation, test_image_frames, cameras, d_ball, id_ball, image_width, image_height)
create_yolo_annotation_files(
    output_folder, trial_name, annotations, cameras, "test")

# save calibration files
calibration_folder = os.path.join(
    "/".join(label_folder.split("/")[:-1]), "calibration")

save_calib_folder = os.path.join(output_folder, trial_name, "calib_params")
os.makedirs(save_calib_folder, exist_ok=True)

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
    s.write('intrinsicMatrix', intrinsicMatrix)
    s.write('distortionCoefficients', distortionCoefficients)
    s.write('R', R)
    s.write('T', T)
    s.release()
    print(output_filename)

# save jpeg images
video_folder = "/".join(label_folder.split("/")[:-1])

map_frame_to_mode = {}
all_image_frames = []

for img in train_image_frames:
    map_frame_to_mode[img] = 'train'
    all_image_frames.append(img)

for img in val_image_frames:
    map_frame_to_mode[img] = 'valid'
    all_image_frames.append(img)

for img in test_image_frames:
    map_frame_to_mode[img] = 'test'
    all_image_frames.append(img)

all_image_frames = np.asarray(all_image_frames)
all_image_frames = np.sort(all_image_frames)


all_jobs = []

save_folder = os.path.join(output_folder, trial_name)
for camera in cameras:
    all_jobs.append([trial_name, camera, video_folder, save_folder,
                    map_frame_to_mode, all_image_frames, "yolo"])

num_jobs = len(all_jobs)
# print(all_jobs[0])
# exit()

if platform.system() == 'Darwin':  # fix for macOS
    print("threadpooling for macos")
    with get_context("fork").Pool(num_jobs) as p:
        p.map(multiprocess_save_jpegs, all_jobs)
else:
    with Pool(num_jobs) as p:
        p.map(multiprocess_save_jpegs, all_jobs)

generate_master_yaml_tennisball(save_folder)