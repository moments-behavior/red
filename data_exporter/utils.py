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


def csv_reader_red2d(
    file_name,
    img_height,
):
    labels = {}
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'{", ".join(row)}')
                line_count += 1
            else:
                keypoints = [float(x) for x in row[1:]]
                keypoints = np.asarray(keypoints)
                keypoints = keypoints.reshape([-1, 3])
                keypoints = keypoints[:, 1:]
                keypoints[keypoints == 1e7] = np.nan
                keypoints[:, 1] = img_height - keypoints[:, 1]
                labels[int(row[0])] = keypoints
                line_count += 1
    return labels


def csv_reader_red3d(file_name):
    labels = {}
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'{", ".join(row)}')
                line_count += 1
            else:
                keypoints = [float(x) for x in row[1:]]
                keypoints = np.asarray(keypoints)
                keypoints = keypoints.reshape([-1, 4])
                keypoints = keypoints[:, 1:]
                keypoints[keypoints == 1e7] = np.nan
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


def get_all_cams_in_labeled_folder(folder):
    cameras = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(folder, "*.csv"))
        if os.path.basename(f) != "keypoints3d.csv"
    ]
    return cameras


def get_all_cams_in_labeled_folder_depreciate(label_folder):
    all_folders = get_subfolders(label_folder)
    cameras = []
    for one_folder in all_folders:
        if one_folder != "worldKeyPoints":
            cameras.append(one_folder)
    return cameras


def process_one_session(
    trial_name,
    load_file_path,
    num_keypoints,
    all_image_frames,
    cameras,
    image_width,
    image_height,
    select_keypoints_idx=[],
    margin_pixel=None,
):
    set_of_frames = {}
    annotations = []
    images = []
    annotation_frame_id = 0
    image_frame_id = 0
    for which_cam in cameras:
        labels_all = csv_reader_red2d(
            load_file_path + "/{}.csv".format(which_cam),
            img_height=image_height[which_cam],
        )

        if not select_keypoints_idx:
            labels = labels_all
        else:
            labels = {}
            for key, value in labels_all.items():
                labels[key] = value[select_keypoints_idx]

        file_dir = trial_name + "/{}/".format(which_cam)
        all_2d_labeled_frames = labels.keys()
        for frame_num in all_image_frames:
            # each frame
            file_name_annotation = (
                file_dir + "Frame_" + str(int(frame_num)) + ".jpg"
            )

            image_entry = {
                "coco_url": "",
                "date_captured": "",
                "file_name": file_name_annotation,
                "flickr_url": "",
                "height": image_height[which_cam],
                "id": image_frame_id,
                "width": image_width[which_cam],
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
                    x_max = np.max(labels[frame_num][:, 0])

                    y_min = np.min(labels[frame_num][:, 1])
                    y_max = np.max(labels[frame_num][:, 1])

                    if margin_pixel is not None:
                        x_min = np.clip(x_min - margin_pixel, 0, None)
                        x_max = np.clip(
                            x_max + margin_pixel, None, image_width[which_cam]
                        )

                        y_min = np.clip(y_min - margin_pixel, 0, None)
                        y_max = np.clip(
                            y_max + margin_pixel, None, image_height[which_cam]
                        )

                    x_size = x_max - x_min
                    y_size = y_max - y_min
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
                        "iscrowd": 0,
                        "keypoints": keypoints,
                        "num_keypoints": annotation_num_kp,
                        "segmentation": [],
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


def process_one_session_ball(
    load_file_path,
    num_keypoints,
    all_image_frames,
    cameras,
    d_ball_pxs,
    label_id,
    image_width,
    image_height,
    select_keypoints_idx=[],
):
    annotations = {}
    for which_cam in cameras:
        img_width = image_width[which_cam]
        img_height = image_height[which_cam]
        labels_all = csv_reader_red2d(
            load_file_path + "/{}.csv".format(which_cam),
            img_height,
        )

        if not select_keypoints_idx:
            labels = labels_all
        else:
            labels = {}
            for key, value in labels_all.items():
                labels[key] = value[select_keypoints_idx]

        all_2d_labeled_frames = labels.keys()
        annotation_entry = []
        file_name_annotation = []
        file_name_image = []
        frame_per_cam = []
        for frame_num in all_image_frames:
            is_any_nan = np.any(np.isnan(labels[frame_num]))
            if not is_any_nan:
                if frame_num in all_2d_labeled_frames:
                    bbox = []
                    bbox_size = float(d_ball_pxs[which_cam][frame_num])
                    x_min = (labels[frame_num][:, 0].min()) / img_width
                    x_size = bbox_size / img_width
                    y_min = labels[frame_num][:, 1].min() / img_height
                    y_size = bbox_size / img_height
                    bbox = [f"{label_id} {x_min} {y_min} {x_size} {y_size}"]

                    # print(bbox)
                    annotation_entry.extend(bbox)
                    file_name_annotation.extend(
                        [f"Frame_{frame_num}_{which_cam}.txt"]
                    )
                    file_name_image.extend(
                        [f"Frame_{frame_num}_{which_cam}.jpg"]
                    )
                    frame_per_cam.extend([frame_num])

        annotations[which_cam] = {
            "frame": frame_per_cam,
            "entry": annotation_entry,
            "fname_annot": file_name_annotation,
            "fname_img": file_name_image,
        }
    return annotations


def generate_framesets(dataset_name, set_of_frames, framesets):
    """
    set_of_frames: dictionary of framesets
    """
    for trial_name, frameset in set_of_frames.items():
        for frame_num, image_ids in frameset.items():
            one_frameset_entry = {
                "datasetName": dataset_name,
                "frames": image_ids,
            }

            entry_name = trial_name + "/Frame_{}".format(frame_num)
            framesets[entry_name] = one_frameset_entry


def generate_annotation_file(
    trial_name,
    keypoint_names,
    skeleton,
    num_keypoints,
    cameras,
    annotations,
    images,
    set_of_frames,
):

    categories = [
        {
            "id": 0,
            "name": "Rat",
            "num_keypoints": num_keypoints,
            "supercategory": "None",
        }
    ]
    root_json = {}
    root_json["keypoint_names"] = keypoint_names
    root_json["skeleton"] = skeleton
    root_json["categories"] = categories
    root_json["annotations"] = annotations
    root_json["images"] = images

    calib_file_dict = {}
    for cam in cameras:
        calib_file_dict["{}".format(cam)] = "calib_params/{}/{}.yaml".format(
            trial_name, cam
        )
    root_json["calibrations"] = {"{}".format(trial_name): calib_file_dict}

    framesets = {}
    generate_framesets(trial_name, set_of_frames, framesets)
    root_json["framesets"] = framesets
    return root_json


def multiprocess_save_jpegs_opencv(input_args):
    (
        trial_name,
        cam_name,
        video_folder_name,
        save_folder,
        map_frame_to_mode,
        all_image_frames,
    ) = input_args

    print("Saving jpeg for {} ...".format(cam_name))
    file_dir = trial_name + "/{}/".format(cam_name)

    # make directories for images (test only created if any frame is mapped to it)
    dir_name = os.path.join(save_folder, "train", file_dir)
    os.makedirs(dir_name, exist_ok=True)
    dir_name = os.path.join(save_folder, "val", file_dir)
    os.makedirs(dir_name, exist_ok=True)
    if "test" in set(map_frame_to_mode.values()):
        dir_name = os.path.join(save_folder, "test", file_dir)
        os.makedirs(dir_name, exist_ok=True)

    print(all_image_frames)

    video_file = os.path.join(video_folder_name, "{}.mp4".format(cam_name))
    cap = cv.VideoCapture(video_file)
    cap.set(cv.CAP_PROP_POS_FRAMES, all_image_frames[0])
    frame_num = all_image_frames[0]

    while (
        frame_num >= all_image_frames[0] and frame_num <= all_image_frames[-1]
    ):
        ret, frame = cap.read()
        if not ret:
            print("Missing fame: {}".format(frame_num))
            cap.release()
            break
        else:
            if frame_num in all_image_frames:
                set_mode = map_frame_to_mode[frame_num]
                dir_name = os.path.join(
                    save_folder, "{}/".format(set_mode), file_dir
                )
                frame_filename = (
                    dir_name + "Frame_" + str(int(frame_num)) + ".jpg"
                )
                cv.imwrite(frame_filename, frame)
            frame_num = frame_num + 1

        if frame_num % 1000 == 0:
            print(f"Processed frame: {frame_num} for {cam_name}")
    cap.release()


def load_jarvis_3d_csv_rats(file_name):
    labels = {}
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count not in [0, 1]:
                if "NaN" not in row:
                    keypoints = [float(x) for x in row]
                    keypoints = np.asarray(keypoints)
                    keypoints = keypoints.reshape([-1, 4])
                    labels[line_count - 2] = keypoints
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

    img_idx_offset = 0  # need to offset the image ids for each dataset

    for dsets in json_data:

        # calibrations
        calib_key = list(dsets["calibrations"].keys())[0]
        for calib_key, calib_value in dsets["calibrations"].items():
            root_json["calibrations"][calib_key] = calib_value

        # images
        for img in dsets["images"]:
            img["id"] = img["id"] + img_idx_offset
            root_json["images"].append(img)

        # annotations
        for annots in dsets["annotations"]:
            annots["id"] = annots["id"] + img_idx_offset
            annots["image_id"] = annots["image_id"] + img_idx_offset
            root_json["annotations"].append(annots)

        # frameset
        for frameset_key, frameset_value in dsets["framesets"].items():
            frameset_value["frames"] = [
                x + img_idx_offset for x in frameset_value["frames"]
            ]
            root_json["framesets"][frameset_key] = frameset_value

        img_idx_offset += len(dsets["images"])

    return root_json


def Project(points, intrinsic, distortion, rotation_matrix, tvec):
    result = []
    if len(points) > 0:
        result, _ = cv.projectPoints(
            points.astype(float), rotation_matrix, tvec, intrinsic, distortion
        )
    return np.squeeze(result, axis=1)


# Jarvis helper
class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.cycles = []
        self.max_len = 0

    def get_cycles(self):
        for edge in self.graph:
            for node in edge:
                self.findNewCycles([node])
        return self.cycles

    def findNewCycles(self, path):
        start_node = path[0]
        next_node = None
        sub = []
        # visit each edge and each node of each edge
        for edge in self.graph:
            node1, node2 = edge
            if start_node in edge:
                if node1 == start_node:
                    next_node = node2
                else:
                    next_node = node1
                if not self.visited(next_node, path):
                    # neighbor node not on path yet
                    sub = [next_node]
                    sub.extend(path)
                    # explore extended path
                    self.findNewCycles(sub)
                elif len(path) > 2 and next_node == path[-1]:
                    # cycle found
                    p = self.rotate_to_smallest(path)
                    inv = self.invert(p)
                    if self.isNew(p) and self.isNew(inv):
                        overlaps = self.overlapping(p)
                        if len(overlaps) > 0:
                            max_len = 0
                            for overlap in overlaps:
                                if len(overlap) > max_len:
                                    max_len = len(overlap)
                            if len(p) > max_len:
                                self.cycles.append(p)
                                for overlap in overlaps:
                                    self.cycles.remove(overlap)
                        else:
                            self.cycles.append(p)

    def invert(self, path):
        return self.rotate_to_smallest(path[::-1])

    def rotate_to_smallest(self, path):
        n = path.index(min(path))
        return path[n:] + path[:n]

    def isNew(self, path):
        return not path in self.cycles

    def overlapping(self, path):
        overlaps = []
        for cycle in self.cycles:
            for point in path:
                if point in cycle:
                    overlaps.append(cycle)
                    break
        return overlaps

    def visited(self, node, path):
        return node in path


def part_of_cycle(cycles, idx):
    for cycle in cycles:
        if idx in cycle:
            return True
    return False


def get_skeleton(skeleton, keypoint_names):
    base_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (0, 140, 255),
        (140, 255, 0),
        (255, 140, 0),
        (0, 255, 140),
        (255, 140, 140),
        (140, 255, 140),
        (140, 140, 255),
        (140, 140, 140),
    ]
    gray_color = (100, 100, 100)
    color_idx = 0
    colors = []
    connections = np.zeros(len(keypoint_names), dtype=int)
    for keypoint in keypoint_names:
        colors.append(gray_color)

    line_idxs = []
    starting_idxs = []

    for bone in skeleton:
        index_start = keypoint_names.index(bone[0])
        starting_idxs.append(index_start)
        index_stop = keypoint_names.index(bone[1])
        line_idxs.append([index_start, index_stop])
        connections[index_start] += 1
        connections[index_stop] += 1

    seeds = np.nonzero(connections == 1)[0]

    unconnected = np.nonzero(connections == 0)[0]
    graph = Graph(line_idxs)
    cycles = graph.get_cycles()

    accounted_for = []

    for cycle in cycles:
        for point in cycle:
            colors[point] = base_colors[color_idx]
        color_idx = (color_idx + 1) % len(base_colors)

    for seed in seeds:
        if seed in starting_idxs:
            idx = seed
            colors[idx] = base_colors[color_idx]
            accounted_for.append(idx)
            conn_idxs = [line[1] for line in line_idxs if line[0] == idx]
            backward_idx = [line[0] for line in line_idxs if line[1] == idx]
            while len(conn_idxs) == 1 and len(backward_idx) < 2:
                idx = conn_idxs[0]
                if connections[idx] < 3 or part_of_cycle(cycles, idx):
                    if idx in accounted_for:
                        colors[idx] = gray_color
                    else:
                        colors[idx] = base_colors[color_idx]
                        accounted_for.append(idx)
                conn_idxs = [line[1] for line in line_idxs if line[0] == idx]
                backward_idx = [
                    line[0] for line in line_idxs if line[1] == idx
                ]
            color_idx = (color_idx + 1) % len(base_colors)

    for point in unconnected:
        colors[point] = base_colors[color_idx]
        color_idx = (color_idx + 1) % len(base_colors)
    return colors, line_idxs
