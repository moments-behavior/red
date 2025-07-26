import glob
import argparse
import cv2
import os
import csv
import yaml
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image to meet YOLO input requirements."""
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Compute padding
    pad_top = (new_shape[0] - nh) // 2
    pad_bottom = new_shape[0] - nh - pad_top
    pad_left = (new_shape[1] - nw) // 2
    pad_right = new_shape[1] - nw - pad_left

    # Pad image
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return img_padded, scale, (pad_left, pad_top)


def get_colors(n):
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    for i in range(n):
        h = int(i * 180 / n)  # evenly spaced hue in [0, 179]
        hsv[i, 0] = [h, 255, 255]  # full saturation and value
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return [tuple(int(c) for c in color[0]) for color in bgr]


parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml_file", type=str, required=True)
parser.add_argument("-s", "--set_name", type=str, default="train")
parser.add_argument("-d", "--downsample", type=int, default="640")

args = parser.parse_args()
yaml_file = args.yaml_file
set_name = args.set_name
downsample = args.downsample

with open(yaml_file, "r") as f:
    config = yaml.safe_load(f)

class_names = config["names"]
num_classes = len(class_names)
colors = get_colors(num_classes)

# Access values
print("Dataset path:", config["path"])
draw_keypoint = True


def convert_yolobbox_cv_rectangle(yolo_bbox, frame_width, frame_height):
    center_x, center_y, w, h = yolo_bbox
    top_left_x = int((center_x - w / 2.0) * frame_width)
    top_left_y = int((center_y - h / 2.0) * frame_height)

    bottom_right_x = int((center_x + w / 2.0) * frame_width)
    bottom_right_y = int((center_y + h / 2.0) * frame_height)
    return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)


dataset_folder = config["path"]
image_folder = os.path.join(dataset_folder, "images", "{}".format(set_name))
label_folder = os.path.join(dataset_folder, "labels", "{}".format(set_name))
image_files = glob.glob(image_folder + "/*.jpg")
image_files.sort()
image_name = []
for image in image_files:
    image_name.append(image.split("/")[-1].split(".")[0])


sel_idx = 0
while True:
    sel_image = os.path.join(
        image_folder, "{}.{}".format(image_name[sel_idx], "jpg")
    )
    sel_label = os.path.join(
        label_folder, "{}.{}".format(image_name[sel_idx], "txt")
    )

    img = cv2.imread(sel_image)
    height, width = img.shape[:2]

    label = {}
    with open(sel_label) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=" ")
        line_count = 0
        bbox = []
        for row in csv_reader:
            class_id = int(row[0])
            label[class_id] = [float(x) for x in row[1:]]

    if draw_keypoint:
        for key, value in label.items():
            bbox = value[:4]
            top_left, bottom_right = convert_yolobbox_cv_rectangle(
                bbox, width, height
            )
            thickness = 5
            img = cv2.rectangle(
                img, top_left, bottom_right, colors[key], thickness
            )

            text_pos = (top_left[0], top_left[1] - 5)
            cv2.putText(
                img,
                class_names[key],
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                colors[key],
                2,
                cv2.LINE_AA,
            )
            keypoints = np.asarray(value[4:])
            keypoints = keypoints.reshape([-1, 3])
            for i in range(keypoints.shape[0]):
                x, y, visible = keypoints[i]
                if visible:
                    point = (int(round(x * width)), int(round(y * height)))
                    cv2.circle(
                        img,
                        point,
                        radius=10,
                        color=colors[key],
                        thickness=-1,
                    )

    img, scale, pad = letterbox(img, (downsample, downsample))

    cv2.putText(
        img,
        sel_image,
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
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
    elif key == ord("n"):
        sel_idx = sel_idx + 1
    elif key == ord("t"):
        draw_keypoint = not draw_keypoint
