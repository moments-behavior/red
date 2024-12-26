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
parser.add_argument('-s', '--skeleton', type=str, required=True)
parser.add_argument('-o', '--output_folder', type=str, required=True)

args = parser.parse_args()
label_folder = args.label_folder
label_folder = os.path.normpath(label_folder)
skeleton = args.skeleton
output_folder = args.output_folder
_, _, num_keypoints = skeleton_selector[skeleton]()

cameras = get_all_cams_in_labeled_folder(label_folder)

# Save annotations
world_point_folder = label_folder + "/worldKeyPoints"
all_files = glob.glob(world_point_folder + "/*")
all_files.sort()
select_most_recent_labels = all_files[-1]
print("Select most recent label: {}".format(select_most_recent_labels))

selected_annotation = select_most_recent_labels.split("/")[-1][10:]
selected_annotation = selected_annotation.split(".")[0]
world_labels = csv_reader_rats(select_most_recent_labels, num_keypoints, True) 

# filter out invalid lables
world_labels_filterd = {}
for name, value in world_labels.items():
    if not np.any(value==1E7):
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

## 
annotations, images, frame_set_one = process_one_session(trial_name, label_folder, num_keypoints, selected_annotation, train_image_frames, cameras)
set_of_frames = {
    trial_name: frame_set_one
}
annotation_json_train = generate_annotation_file(trial_name, skeleton, cameras, annotations, images, set_of_frames)


annotations, images, frame_set_one = process_one_session(trial_name, label_folder, num_keypoints, selected_annotation, val_image_frames, cameras)
set_of_frames = {
    trial_name: frame_set_one
}
annotation_json_val = generate_annotation_file(trial_name, skeleton, cameras, annotations, images, set_of_frames)


annotation_path = os.path.join(output_folder, "annotations/") 
os.makedirs(annotation_path, exist_ok=True)   

with open(annotation_path + "instances_train.json", 'w') as f:
    json.dump(annotation_json_train, f)
    
with open(annotation_path + "instances_val.json", 'w') as f:
    json.dump(annotation_json_val, f)

print("Prepared dataset at {}.".format(output_folder))


## save calibration
calibration_folder = os.path.join("/".join(label_folder.split("/")[:-1]), "calibration")

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

# exit()

for camera in cameras:
    all_jobs.append([trial_name, camera, video_folder, output_folder, map_frame_to_mode, all_image_frames])    

num_jobs = len(all_jobs)
if platform.system() == 'Darwin':  # fix for macOS
    print("threadpooling for macos")
    with get_context("fork").Pool(num_jobs) as p:
        p.map(multiprocess_save_jpegs, all_jobs)
else:
    with Pool(num_jobs) as p:
        p.map(multiprocess_save_jpegs, all_jobs)

# train_jobs = []
# for camera in cameras:
#     train_jobs.append([trial_name, camera, video_folder, output_folder, 'train', train_image_frames])
# print(train_image_frames)

# valid_jobs = []
# for camera in cameras:
#     valid_jobs.append([trial_name, camera, video_folder, output_folder, 'val', val_image_frames])
# print(val_image_frames)

# exit()

# num_jobs = len(valid_jobs)

# if platform.system() == 'Darwin':  # fix for macOS
#     print("threadpooling for macos")
#     with get_context("fork").Pool(num_jobs) as p:
#         p.map(multiprocess_save_jpegs, valid_jobs)
# else:
#     with Pool(num_jobs) as p:
#         p.map(multiprocess_save_jpegs, valid_jobs)
        
# num_jobs = len(train_jobs)    
# if platform.system() == 'Darwin':  # fix for macOS
#     print("threadpooling for macos")
#     with get_context("fork").Pool(num_jobs) as p:
#         p.map(multiprocess_save_jpegs, train_jobs)
# else:
#     with Pool(num_jobs) as p:
#         p.map(multiprocess_save_jpegs, train_jobs)        




