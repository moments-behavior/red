import os
import sys
import cv2
import matplotlib.pyplot as plt

def read_yolo_labels(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, w, h = map(float, parts)
                boxes.append((cls, cx, cy, w, h))
    return boxes

def plot_boxes_on_image(img_path, label_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]
    boxes = read_yolo_labels(label_path)
    for cls, cx, cy, w, h in boxes:
        print("Relative Center:", cx, cy)
        x_center = cx * w_img
        y_center = cy * h_img
        print("Absolute Center:", x_center, y_center)
        box_w = w * w_img
        box_h = h * h_img
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def main(master_dir):
    images_dir = os.path.join(master_dir, "images")
    labels_dir = os.path.join(master_dir, "labels")
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print("Could not find 'images' or 'labels' subfolders.")
        return
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            print(f"Plotting {img_file} with labels...")
            plot_boxes_on_image(img_path, label_path)
        else:
            print(f"No label file for {img_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_yolo_labels.py <master_directory>")
    else:
        main(sys.argv[1])
