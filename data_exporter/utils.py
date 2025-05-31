import os
import glob
import cv2 as cv
from keypoints import *
import csv
import numpy as np

def get_skeleton_name(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file)
        first_row = next(csv_reader) 
    return first_row[0]

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
                keypoints[keypoints == 1e7] = np.nan
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

def process_one_session(trial_name, 
                        load_file_path, 
                        num_keypoints, 
                        annotation_file, 
                        all_image_frames, 
                        cameras, 
                        image_width,
                        image_height,
                        select_keypoints_idx=[]):
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
                "height":image_height[which_cam],
                "id":image_frame_id,
                "width":image_width[which_cam]
            }

            if frame_num in all_2d_labeled_frames:
                is_any_nan = np.any(np.isnan(labels[frame_num]))
                if not is_any_nan:
                    if select_keypoints_idx: 
                        annotation_num_kp = len(select_keypoints_idx)
                    else:
                        annotation_num_kp = num_keypoints
                    bbox = []
                    x_min = np.min(labels[frame_num][:, 0])
                    x_size = np.max(labels[frame_num][:, 0]) - np.min(labels[frame_num][:, 0])
                    y_min = np.min(labels[frame_num][:, 1])
                    y_size = np.max(labels[frame_num][:, 1]) - np.min(labels[frame_num][:, 1])
                    bbox = [x_min, y_min, x_size, y_size]

                    keypoints = []
                    for keypoint_idx in range(annotation_num_kp):
                        if np.any(np.isnan(labels[frame_num][keypoint_idx])):
                            x = 0
                            y = 0
                        else:
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
                        "num_keypoints":annotation_num_kp,
                        "segmentation":[]
                    }
            
            # for create framesets
            frame_num_int = int(frame_num)
            
            if frame_num_int not in set_of_frames.keys():
                set_of_frames[frame_num_int] = [image_frame_id]
            else:
                set_of_frames[frame_num_int].append(image_frame_id)

            if frame_num in all_2d_labeled_frames:
                is_any_nan = np.any(np.isnan(labels[frame_num]))
                if not is_any_nan:
                    annotations.append(annotation_entry)
                    annotation_frame_id = annotation_frame_id + 1

            images.append(image_entry)
            image_frame_id = image_frame_id + 1
    return annotations, images, set_of_frames

def process_one_session_ball(trial_name, load_file_path, num_keypoints, annotation_file, all_image_frames, cameras, d_ball,label_id,select_keypoints_idx=[]):
    
    annotations = {}
    for which_cam in cameras:

        # print(select_keypoints_idx)
        labels = csv_reader_rats(load_file_path + "/{}/{}_{}.csv".format(which_cam, which_cam, annotation_file), num_keypoints, three_d=False, select_keypoints_idx=select_keypoints_idx)   

        file_dir = trial_name + "/{}/".format(which_cam)
        all_2d_labeled_frames = labels.keys()
        
        annotation_entry = []
        file_name_annotation = []
        file_name_image = []

        for frame_num in all_image_frames:
            ## each frame 
            
            img_height=2200
            img_width =3208
            bbox_size=d_ball
            
            
            if frame_num in all_2d_labeled_frames:
                bbox = []            
                x_min = (labels[frame_num][:, 0].min())/img_width
                x_size = bbox_size/img_width
                y_min = labels[frame_num][:, 1].min()/img_height
                y_size = bbox_size/img_height
                bbox= [f"{label_id} {x_min} {y_min} {x_size} {y_size}"]

                # print(bbox)
                
                
                annotation_entry.extend(bbox)
                file_name_annotation.extend([f"Frame_{frame_num}.txt"])
                file_name_image.extend([f"Frame_{frame_num}.jpg"])
                
                # file_name_image.extend("Frame_" + str(int(frame_num)) + '.jpg')

        annotations[which_cam,"entry"] = annotation_entry
        annotations[which_cam,"fname_annot"] = file_name_annotation
        annotations[which_cam,"fname_img"] = file_name_image
    
    return annotations


def create_yolo_annotation_files(output_folder, trial_name, annotations,cameras,dset_mode):
    
    for which_cam in cameras:
        dir_labels = os.path.join(output_folder,trial_name,which_cam,dset_mode,"labels")
        os.makedirs(dir_labels, exist_ok=True)
        for i in range(len(annotations[which_cam,"entry"])):  

            # print(annotations[which_cam,"fname_img"][i])
            fname_label =  os.path.join(dir_labels,annotations[which_cam,'fname_annot'][i])
            with open(fname_label, "w") as f:
                f.write(annotations[which_cam,"entry"][i])


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

def generate_annotation_file(trial_name, keypoint_names, skeleton, num_keypoints, cameras, annotations, images, set_of_frames):
    
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
    trial_name, cam_name, video_folder_name, save_folder, map_frame_to_mode, all_image_frames, export_mode = input_args

    print("Saving jpeg for {} ...".format(cam_name))
    file_dir =  trial_name + "/{}/".format(cam_name)

    # make directories for images
    if(export_mode == "jarvis"):
        dir_name = os.path.join(save_folder, "train", file_dir)
        os.makedirs(dir_name, exist_ok=True)     
        dir_name = os.path.join(save_folder, "val", file_dir)
        os.makedirs(dir_name, exist_ok=True) 
    elif(export_mode == "yolo"):
        dir_name = os.path.join(save_folder, "train", "images")
        os.makedirs(dir_name, exist_ok=True)     
        dir_name = os.path.join(save_folder, "valid", "images")
        os.makedirs(dir_name, exist_ok=True) 
        

    print(all_image_frames)
    # exit()    
    
    video_file = os.path.join(video_folder_name, "{}.mp4".format(cam_name))
    cap = cv.VideoCapture(video_file)
    cap.set(cv.CAP_PROP_POS_FRAMES, all_image_frames[0]-1); 

    start_frame = np.min(all_image_frames)-1
    end_frame = np.max(all_image_frames)

    frame_num = start_frame    
    
    while (frame_num >= start_frame and frame_num <= end_frame):
        ret, frame = cap.read() # frame shape (2200, 3208, 3)    
        if ret == False:
            print("Missing fame: {}".format(frame_num))
        else:
            frame_num = frame_num + 1
            if frame_num in all_image_frames:    
                set_mode = map_frame_to_mode[frame_num]        
                if(export_mode == "jarvis"):
                    dir_name = os.path.join(save_folder, "{}/".format(set_mode), file_dir)                  
                    frame_filename = dir_name + "Frame_" + str(int(frame_num)) + '.jpg'
                elif(export_mode == "yolo"):
                    frame_filename = os.path.join(save_folder, set_mode,"images",f"Frame_{frame_num}.jpg")                  
                    # frame_filename = dir_name + "Frame_" + str(int(frame_num)) + '.jpg'

                cv.imwrite(frame_filename, frame)

        if(frame_num % 1000 == 0):
            print(f"Processed frame: {frame_num} for {cam_name}")

def load_jarvis_3d_csv_rats(file_name, num_keypoints):
    labels = {}
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count not in [0, 1]:
                if 'NaN' not in row:        
                    keypoints = [float(x) for x in row]
                    keypoints = np.asarray(keypoints)
                    keypoints = keypoints.reshape([num_keypoints, 4])                                        
                    labels[line_count-2] = keypoints
            line_count += 1
    return labels

def merge_json_annotations(json_data):
   
    root_json = {}
    root_json["keypoint_names"] = json_data[0]["keypoint_names"]
    root_json["skeleton"] = json_data[0]["skeleton"]
    root_json["categories"] = json_data[0]["categories"]

    # Initialize the "calibrations" key in root_json if it doesn't exist
    root_json["calibrations"] = {}
    root_json["images"] = []
    root_json["annotations"] = []
    root_json["framesets"] = {}
      
    img_idx_offset = 0 #need to offset the image ids for each dataset
    
    for dsets in json_data:
        
        # calibrations
        calib_key = list(dsets["calibrations"].keys())[0]
        for calib_key, calib_value in dsets["calibrations"].items():
            root_json["calibrations"][calib_key] = calib_value
            
        #images
        for img in dsets["images"]:
            img["id"] = img["id"] + img_idx_offset
            root_json["images"].append(img)
            
        
        #annotations
        for annots in dsets["annotations"]:
            annots["id"] = annots["id"] + img_idx_offset
            annots["image_id"] = annots["image_id"] + img_idx_offset
            root_json["annotations"].append(annots)
            
        # frameset
        for frameset_key, frameset_value in dsets["framesets"].items():
            frameset_value["frames"] = [x + img_idx_offset for x in frameset_value["frames"]]
            root_json["framesets"][frameset_key] = frameset_value
            

        img_idx_offset += len(dsets["images"])
        

    return root_json

def Project(points, intrinsic, distortion, rotation_matrix, tvec):
    result = []
    if len(points) > 0:
        result, _ = cv.projectPoints(points.astype(float), rotation_matrix, tvec, intrinsic, distortion)
    return np.squeeze(result, axis=1)