import argparse
import cv2
import os
import json
from collections import defaultdict
import numpy as np
from utils import get_skeleton

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--root_dir", type=str, required=True)
parser.add_argument("-s", "--set_name", type=str, default="train")

args = parser.parse_args()
root_dir = args.root_dir
set_name = args.set_name
dataset_file = open(
    os.path.join(root_dir, "annotations", "instances_" + set_name + ".json")
)
dataset = json.load(dataset_file)

num_keypoints = []
for category in dataset["categories"]:
    num_keypoints.append(category["num_keypoints"])

image_ids = [img["id"] for img in dataset["images"]]

annotations, categories, imgs = dict(), dict(), dict()
imgToAnns = defaultdict(list)

if "annotations" in dataset:
    for ann in dataset["annotations"]:
        imgToAnns[ann["image_id"]].append(ann)
        annotations[ann["id"]] = ann

if "images" in dataset:
    for img in dataset["images"]:
        imgs[img["id"]] = img

if "categories" in dataset:
    for cat in dataset["categories"]:
        categories[cat["id"]] = cat

keypoint_names = dataset["keypoint_names"]
skeleton = []
for component in dataset["skeleton"]:
    skeleton.append([component["keypointA"], component["keypointB"]])


# load annotations
def load_annotations(image_index, dataset_annotations):
    annotations_ids = [ann["id"] for ann in imgToAnns[image_ids[image_index]]]
    annotations = np.zeros((0, 5))
    keypoints = np.zeros((0, num_keypoints[0] * 3))

    if len(annotations_ids) == 0:
        annotations = np.zeros((1, 5))
        annotations[0][4] = -1
        keypoints = np.zeros((1, num_keypoints[0] * 3))
    else:
        coco_annotations = [dataset_annotations[id] for id in annotations_ids]
        for idx, a in enumerate(coco_annotations):
            annotation = np.zeros((1, 5))
            annotation[0, :4] = a["bbox"]
            for i in range(len(annotation)):
                annotation[i] /= 1
            annotation[0, 4] = a["category_id"] - 1
            annotations = np.append(annotations, annotation, axis=0)
            keypoint = np.array(a["keypoints"]).reshape(
                1, num_keypoints[0] * 3
            )
            keypoints = np.append(keypoints, keypoint, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
    return annotations, keypoints


colors, line_idxs = get_skeleton(skeleton, keypoint_names)


def _parse_camera_frame(file_name):
    """Extract (camera, frame) from a JARVIS image path like 'CamXXXX/0.jpg'
    or 'CamXXXX_0.jpg'. Falls back to the raw filename if it can't be parsed."""
    base = os.path.splitext(file_name)[0]
    if "/" in base:
        camera, frame = base.rsplit("/", 1)
    elif "_" in base:
        camera, frame = base.rsplit("_", 1)
    else:
        return file_name, 0
    try:
        frame = int(frame)
    except ValueError:
        frame = 0
    return camera, frame


# Iterate frame-major: show frame 0 from every camera, then frame 1, etc.
parsed = {i: _parse_camera_frame(imgs[i]["file_name"]) for i in range(len(imgs))}
sorted_image_indices = sorted(
    range(len(imgs)), key=lambda i: (parsed[i][1], parsed[i][0])
)

for image_index in sorted_image_indices:
    file_name = imgs[image_index]["file_name"]
    path = os.path.join(root_dir, set_name, file_name)
    img = cv2.imread(path)
    bbox, keypoints = load_annotations(image_index, annotations)
    if bbox[0][-1] != -1:
        top_left = bbox[0][:2]
        top_left = top_left.astype(int)
        bottom_right = bbox[0][2:4]
        bottom_right = bottom_right.astype(int)
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    keypoints = keypoints.reshape(-1, 3)
    for i, keypoint in enumerate(keypoints):
        if keypoint[0] + keypoint[1] != 0:
            img = cv2.circle(
                img, (int(keypoint[0]), int(keypoint[1])), 4, colors[i], 6
            )
    for line in line_idxs:
        if (
            keypoints[line[0]][0] + keypoints[line[0]][1] != 0
            and keypoints[line[1]][0] + keypoints[line[1]][1] != 0
        ):
            cv2.line(
                img,
                (int(keypoints[line[0]][0]), int(keypoints[line[0]][1])),
                (int(keypoints[line[1]][0]), int(keypoints[line[1]][1])),
                colors[line[1]],
                2,
            )

    img = cv2.resize(img, None, fx=1 / 3, fy=1 / 3)

    cv2.putText(
        img,
        file_name,
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    cv2.imshow("preview", img)
    # Wait for a key press (0 = wait indefinitely)
    key = cv2.waitKey()
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
