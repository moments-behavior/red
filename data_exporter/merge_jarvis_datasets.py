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
input_datasets = [o for o in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, o))]
print(input_datasets)


####################################
# copy calibration files
####################################

# create parent output directory for calibration files
output_calib_dir = os.path.join(output_dataset, "calib_params")
if not os.path.exists(output_calib_dir):
    os.makedirs(output_calib_dir)
    
# loop over all datasets    
for dset in input_datasets:
    
    src_calib_dir = os.path.join(src_dir, dset, "calib_params")
    all_calibs = glob.glob(src_calib_dir + "/*")
    all_calibs.sort()
    select_most_recent_calib = all_calibs[-1]
    print("Select most recent label: {}".format(select_most_recent_calib))
    src_calib_dir = select_most_recent_calib
    
    
    # create output directory for dataset and copy calibration files
    shutil.copytree(os.path.dirname(src_calib_dir), output_calib_dir, dirs_exist_ok=True)
    print(f"will copy all from {os.path.dirname(src_calib_dir)} to {output_calib_dir}")


####################################
# copy train/val/test images
####################################
# Test is optional — only datasets exported with --test_ratio > 0 have it.

for image_set in ["train", "val", "test"]:

    # create the output directory for this split
    output_img_dir = os.path.join(output_dataset, image_set)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    # loop over all datasets
    for dset in input_datasets:

        # find the most recent image dataset
        src_image_dir = os.path.join(src_dir, dset, image_set)
        if not os.path.isdir(src_image_dir):
            print(f"  skip {image_set} for {dset} (no {src_image_dir})")
            continue
        all_imagesets = glob.glob(src_image_dir + "/*")
        if not all_imagesets:
            print(f"  skip {image_set} for {dset} (empty {src_image_dir})")
            continue
        all_imagesets.sort()
        select_most_recent_imageset = all_imagesets[-1]
        src_image_dir = select_most_recent_imageset


        print(f"will copy all from {os.path.dirname(src_image_dir)} to {output_img_dir}")
        shutil.copytree(os.path.dirname(src_image_dir), output_img_dir, dirs_exist_ok=True)


####################################
# copy and merge annotations
####################################

# create annotations directory in the parent folder
output_annot_dir = os.path.join(output_dataset, "annotations")
if not os.path.exists(output_annot_dir):
    os.makedirs(output_annot_dir)

# for train, val, and (optionally) test
for image_set in ["train", "val", "test"]:

    json_data = []
    # loop over all datasets
    for dset in input_datasets:

        src_json_file = os.path.join(src_dir, dset, "annotations", f"instances_{image_set}.json")
        if not os.path.isfile(src_json_file):
            print(f"  skip {image_set} annotations for {dset} (no {src_json_file})")
            continue
        with open(src_json_file, 'r') as file:
            json_data.append(json.load(file))

    if not json_data:
        # test split absent across all source datasets; skip writing it
        continue

    merged_json = merge_json_annotations(json_data)

    output_json_file = os.path.join(output_dataset, "annotations", f"instances_{image_set}.json")
    with open(output_json_file, 'w') as file:
        json.dump(merged_json, file, indent=4)
    
    
print("Merged dataset created at: {}".format(output_dataset))