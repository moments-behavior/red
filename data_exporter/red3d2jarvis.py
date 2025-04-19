from keypoints import *
import argparse
import numpy as np
import json
from utils import *
import glob
import cv2 as cv
from multiprocessing import Pool
import platform
from multiprocessing import get_context

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--label_folder', type=str, required=True)
parser.add_argument('-o', '--output_folder', type=str, required=True)
parser.add_argument('-s', '--select_indices', nargs='+', type=int, help='List of numbers', default=[])

args = parser.parse_args()
label_folder = args.label_folder
label_folder = os.path.normpath(label_folder)
output_folder = args.output_folder
select_indices = args.select_indices

world_point_folder = label_folder + "/worldKeyPoints"
all_files = glob.glob(world_point_folder + "/*")
all_files.sort()
select_most_recent_labels = all_files[-1]
print("Select most recent label: {}".format(select_most_recent_labels))
cameras = get_all_cams_in_labeled_folder(label_folder)

# get skeleton name
skeleton = get_skeleton_name(select_most_recent_labels)
print(skeleton)
keypoints_names, skeleton_names, num_keypoints = skeleton_selector[skeleton]()

selected_annotation = select_most_recent_labels.split("/")[-1][10:]
selected_annotation = selected_annotation.split(".")[0]
world_labels = csv_reader_rats(select_most_recent_labels, num_keypoints, True, select_keypoints_idx=select_indices) 

# filter out invalid lables
world_labels_filterd = {}
for name, value in world_labels.items():
    if not np.any(np.isnan(value)):
        world_labels_filterd[name] = value

labels_frames = np.asarray(list(world_labels_filterd.keys()))
total_num_labels = len(labels_frames)

id_shuffled = np.arange(total_num_labels)
np.random.shuffle(id_shuffled)
num_train = int(np.floor(total_num_labels * 0.9))
print("Train set: {}, validation set: {}.".format(num_train, total_num_labels - num_train))
train_ids = id_shuffled[:num_train]
train_ids = np.sort(train_ids)
val_ids = id_shuffled[num_train:]
val_ids = np.sort(val_ids)
## split frames to train and val
train_image_frames = labels_frames[train_ids]
val_image_frames = labels_frames[val_ids]

trial_name = selected_annotation

calibration_folder = os.path.join("/".join(label_folder.split("/")[:-1]), "calibration")
image_width = {}
image_height = {}
for cam in cameras:
    input_file_name = calibration_folder + "/{}.yaml".format(cam)
    fs = cv.FileStorage(input_file_name, cv.FILE_STORAGE_READ)
    if fs.isOpened():
        image_width[cam] = int(fs.getNode("image_width").real())
        image_height[cam] = int(fs.getNode("image_height").real())

annotations, images, frame_set_one = process_one_session(trial_name, 
                                                         label_folder, 
                                                         num_keypoints, 
                                                         selected_annotation, 
                                                         train_image_frames, 
                                                         cameras, 
                                                         image_width,
                                                         image_height,
                                                         select_keypoints_idx=select_indices)
set_of_frames = {
    trial_name: frame_set_one
}
if select_indices:
    keypoints_names_selected = [keypoints_names[i] for i in select_indices]
    annotation_num_kps = len(select_indices)
else:
    keypoints_names_selected = keypoints_names
    annotation_num_kps = num_keypoints
annotation_json_train = generate_annotation_file(trial_name, keypoints_names_selected, skeleton_names, annotation_num_kps, cameras, annotations, images, set_of_frames)


annotations, images, frame_set_one = process_one_session(trial_name, 
                                                         label_folder, 
                                                         num_keypoints, 
                                                         selected_annotation, 
                                                         val_image_frames, 
                                                         cameras, 
                                                         image_width,
                                                         image_height
                                                         select_keypoints_idx=select_indices)
set_of_frames = {
    trial_name: frame_set_one
}
annotation_json_val = generate_annotation_file(trial_name, keypoints_names_selected, skeleton_names, annotation_num_kps, cameras, annotations, images, set_of_frames)


annotation_path = os.path.join(output_folder, "annotations/") 
os.makedirs(annotation_path, exist_ok=True)   

with open(annotation_path + "instances_train.json", 'w') as f:
    json.dump(annotation_json_train, f)
    
with open(annotation_path + "instances_val.json", 'w') as f:
    json.dump(annotation_json_val, f)

print("Prepared dataset at {}.".format(output_folder))


## save calibration
save_calib_folder = os.path.join(output_folder, "calib_params", trial_name)
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


    output_filename =  save_calib_folder + "/{}.yaml".format(cam)
    s = cv.FileStorage(output_filename, cv.FileStorage_WRITE)
    s.write('intrinsicMatrix', intrinsicMatrix)
    s.write('distortionCoefficients', distortionCoefficients)
    s.write('R', R)
    s.write('T', T)
    s.release()
    print(output_filename)


## save jpeg images
video_folder = "/".join(label_folder.split("/")[:-1])

map_frame_to_mode = {}
all_image_frames = []

for img in train_image_frames:
    map_frame_to_mode[img] = 'train'
    all_image_frames.append(img)
    
for img in val_image_frames:
    map_frame_to_mode[img] = 'val'
    all_image_frames.append(img)

all_image_frames = np.asarray(all_image_frames)
all_image_frames = np.sort(all_image_frames)


all_jobs = []

for camera in cameras:
    all_jobs.append([trial_name, camera, video_folder, output_folder, map_frame_to_mode, all_image_frames, 'jarvis'])    

num_jobs = len(all_jobs)

if platform.system() == 'Darwin':  # fix for macOS
    print("threadpooling for macos")
    with get_context("fork").Pool(num_jobs) as p:
        p.map(multiprocess_save_jpegs, all_jobs)
else:
    with Pool(num_jobs) as p:
        p.map(multiprocess_save_jpegs, all_jobs)




