# based on some functions in https://github.com/JohnsonLabJanelia/red/data_exporter/utils.py  

import cv2 as cv
import os
from utils import *
from keypoints import *
import argparse
from datetime import datetime
import shutil
import json
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--source_folder', type=str, required=True)
parser.add_argument('-o', '--output_folder', type=str, required=True)

args = parser.parse_args()

# directory with source datasets
src_dir = args.source_folder

# create parent directory where merged dataset will be created
output_dataset = args.output_folder
if not os.path.exists(output_dataset):
    os.makedirs(output_dataset)

# get the list of datasets in the source directory
input_datasets = sorted(
    o for o in os.listdir(src_dir)
    if os.path.isdir(os.path.join(src_dir, o))
)
print(f"Merging {len(input_datasets)} source datasets")


# copy calibration files
output_calib_dir = os.path.join(output_dataset, "calib_params")
os.makedirs(output_calib_dir, exist_ok=True)
for dset in input_datasets:
    all_calibs = sorted(glob.glob(os.path.join(src_dir, dset,
                                               "calib_params") + "/*"))
    if all_calibs:
        shutil.copytree(os.path.dirname(all_calibs[-1]),
                        output_calib_dir, dirs_exist_ok=True)


# copy train/val/test images
split_image_counts = {}
for image_set in ["train", "val", "test"]:
    output_img_dir = os.path.join(output_dataset, image_set)
    os.makedirs(output_img_dir, exist_ok=True)
    total = 0
    for dset in input_datasets:
        src_image_dir = os.path.join(src_dir, dset, image_set)
        if not os.path.isdir(src_image_dir):
            continue
        all_imagesets = sorted(glob.glob(src_image_dir + "/*"))
        if not all_imagesets:
            continue
        chosen = all_imagesets[-1]
        total += len([f for f in os.listdir(chosen)
                      if os.path.isfile(os.path.join(chosen, f))])
        shutil.copytree(os.path.dirname(chosen), output_img_dir,
                        dirs_exist_ok=True)
    split_image_counts[image_set] = total


# merge annotations
output_annot_dir = os.path.join(output_dataset, "annotations")
os.makedirs(output_annot_dir, exist_ok=True)
split_ann_counts = {}
for image_set in ["train", "val", "test"]:
    json_data = []
    for dset in input_datasets:
        src_json_file = os.path.join(src_dir, dset, "annotations",
                                     f"instances_{image_set}.json")
        if os.path.isfile(src_json_file):
            with open(src_json_file, 'r') as file:
                json_data.append(json.load(file))
    if not json_data:
        continue
    merged_json = merge_json_annotations(json_data)
    with open(os.path.join(output_dataset, "annotations",
                           f"instances_{image_set}.json"), 'w') as file:
        json.dump(merged_json, file, indent=4)
    split_ann_counts[image_set] = len(merged_json.get("annotations", []))


# summary
print(f"\nOutput: {output_dataset}")
for sp in ["train", "val", "test"]:
    if sp in split_image_counts or sp in split_ann_counts:
        imgs = split_image_counts.get(sp, 0)
        anns = split_ann_counts.get(sp, 0)
        print(f"  {sp}: {imgs} images, {anns} annotations")