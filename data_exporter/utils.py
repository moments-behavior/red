import os
import glob
import cv2 as cv
from keypoints import *
import csv
import numpy as np

def csv_reader_rats(file_name, num_keypoints, three_d=False, select_keypoints_idx=[], prediction_file=False):
    labels = {}
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #print(f'{", ".join(row)}')
                line_count += 1
            else:
                if prediction_file:
                    keypoints = [float(x) for x in row[1:]]
                else:
                    keypoints = [float(x) for x in row[1:-1]]
                keypoints = np.asarray(keypoints)
                if three_d:
                    keypoints = keypoints.reshape([num_keypoints, 4])
                else:
                    keypoints = keypoints.reshape([num_keypoints, 3])
                keypoints = keypoints[:, 1:]
                if not three_d:
                    keypoints[:, 1] = 2200 - keypoints[:, 1]  
                if (len(select_keypoints_idx) > 0):
                    keypoints = keypoints[select_keypoints_idx, :]                
                labels[int(row[0])] = keypoints               
                line_count += 1
    return labels

def get_subfolders(path):
    """Gets a list of subfolder names in the given directory."""

    subfolders = []
    for entry in os.scandir(path):
        if entry.is_dir():
            subfolders.append(entry.name)
    return subfolders

def get_all_cams_in_labeled_folder(label_folder):
    all_folders = get_subfolders(label_folder)
    cameras = []
    for one_folder in all_folders:
        if one_folder != "worldKeyPoints":
            cameras.append(one_folder)
    return cameras

def process_one_session(trial_name, load_file_path, num_keypoints, annotation_file, all_image_frames, cameras, select_keypoints_idx=[]):
    """
    args:
        all_image_frames: list of all the frames annotated in 3d
    return: (int) number of pictures in the session
    """
    set_of_frames = {}
    annotations = []
    images = []
    annotation_frame_id = 0
    image_frame_id = 0
    for which_cam in cameras:

        labels = csv_reader_rats(load_file_path + "/{}/{}_{}.csv".format(which_cam, which_cam, annotation_file), num_keypoints, three_d=False, select_keypoints_idx=select_keypoints_idx)   

        file_dir = trial_name + "/{}/".format(which_cam)
        all_2d_labeled_frames = labels.keys()
        for frame_num in all_image_frames:
            ## each frame 
            file_name_annotation = file_dir + "Frame_" + str(int(frame_num)) + '.jpg'

            image_entry = {
                "coco_url":"",
                "date_captured":"",
                "file_name":file_name_annotation,
                "flickr_url":"",
                "height":2200,
                "id":image_frame_id,
                "width":3208
            }

            if frame_num in all_2d_labeled_frames:
                bbox = []
                x_min = labels[frame_num][:, 0].min()
                x_size = labels[frame_num][:, 0].max() - labels[frame_num][:, 0].min()
                y_min = labels[frame_num][:, 1].min()
                y_size = labels[frame_num][:, 1].max() - labels[frame_num][:, 1].min()
                bbox = [x_min, y_min, x_size, y_size]

                keypoints = []
                for keypoint_idx in range(num_keypoints):
                    x = int(labels[frame_num][keypoint_idx, 0])
                    y = int(labels[frame_num][keypoint_idx, 1])
                    keypoints.extend([x, y, 1])
                
                annotation_entry = {
                    "bbox": bbox,
                    "category_id": 1,
                    "id": annotation_frame_id,
                    "image_id": image_frame_id, 
                    "iscrowd":0,
                    "keypoints": keypoints,
                    "num_keypoints":num_keypoints,
                    "segmentation":[]
                }

            
            # for create framesets
            frame_num_int = int(frame_num)
            
            if frame_num_int not in set_of_frames.keys():
                set_of_frames[frame_num_int] = [image_frame_id]
            else:
                set_of_frames[frame_num_int].append(image_frame_id)

            if frame_num in all_2d_labeled_frames:
                annotations.append(annotation_entry)
                annotation_frame_id = annotation_frame_id + 1

            images.append(image_entry)
            image_frame_id = image_frame_id + 1
    return annotations, images, set_of_frames

def generate_framesets(dataset_name, set_of_frames, framesets):
    """
    set_of_frames: dictionary of framesets
    """
    for trial_name, frameset in set_of_frames.items():
        for frame_num, image_ids in frameset.items():    
            one_frameset_entry = {
                "datasetName": dataset_name,
                "frames": image_ids
            }

            entry_name = trial_name + "/Frame_{}".format(frame_num)            
            framesets[entry_name] = one_frameset_entry

def geneate_annotation_file(trial_name, skeleton_name, cameras, annotations, images, set_of_frames):
    
    keypoint_names, skeleton, num_keypoints = skeleton_selector[skeleton_name]()

    categories = [{"id":0,"name":"Rat","num_keypoints":num_keypoints,"supercategory":"None"}]
    root_json = {}
    root_json["keypoint_names"] = keypoint_names
    root_json["skeleton"] = skeleton
    root_json["categories"] = categories
    root_json["annotations"] = annotations
    root_json["images"] = images

    calib_file_dict = {}
    for cam in cameras:
        calib_file_dict["{}".format(cam)] = "calib_params/{}/{}.yaml".format(trial_name, cam)
    root_json["calibrations"] = {"{}".format(trial_name): calib_file_dict}

    framesets = {}
    generate_framesets(trial_name, set_of_frames, framesets)
    root_json["framesets"] = framesets
    return root_json


def multiprocess_save_jpegs(input_args):
    trial_name, cam_name, video_folder_name, save_folder, set_mode, all_image_frames = input_args

    print("Saving jpeg for {} ...".format(cam_name))
    file_dir =  trial_name + "/{}/".format(cam_name)

    for frame_num in all_image_frames:
        ## each frame
        video_file = os.path.join(video_folder_name, "{}.mp4".format(cam_name))
        cap = cv.VideoCapture(video_file)
        cap.set(1, frame_num); 
        ret, frame = cap.read() # frame shape (2200, 3208, 3)    
        if ret == False:
            print("Missing fame: {}".format(frame_num))
            
        dir_name = os.path.join(save_folder, "{}/".format(set_mode), file_dir)
        os.makedirs(dir_name, exist_ok=True)   
        
        frame_name = dir_name + "Frame_" + str(int(frame_num)) + '.jpg'
        cv.imwrite(frame_name, frame)