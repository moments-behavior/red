import glob
import argparse
import cv2
import os
import csv
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml_file", type=str, required=True)
parser.add_argument("-s", "--set_name", type=str, default="train")

args = parser.parse_args()
yaml_file = args.yaml_file
set_name = args.set_name

with open(yaml_file, "r") as f:
    config = yaml.safe_load(f)

# Access values
print("Dataset path:", config["path"])


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
            bbox = [float(x) for x in row[1:]]
            label[class_id] = bbox

    top_left, bottom_right = convert_yolobbox_cv_rectangle(
        label[0], width, height
    )
    color = (255, 0, 255)
    thickness = 5
    img = cv2.rectangle(img, top_left, bottom_right, color, thickness)

    img = cv2.resize(img, None, fx=1 / 2, fy=1 / 2)

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
    sel_idx = sel_idx + 1
