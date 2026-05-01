# Format data for deep learning  

Create conda python virtual environment

```
conda create -n red_exporter python=3.10
conda activate red_exporter
conda install numpy
conda install -c conda-forge opencv
pip install pyyaml
pip install PyNvVideoCodec
```

## YOLO Detection Dataset Export

Export bounding box data for YOLO object detection training:

```bash
python export_yolo_detection.py -i /path/to/labeled/data -v /path/to/videos -o /path/to/output -c class_names.txt
```

Parameters:
- `-i, --label_dir`: Directory containing timestamped label folders from RED
- `-v, --video_dir`: Directory containing video files (should match camera names)
- `-o, --output_dir`: Output directory for YOLO dataset
- `-c, --class_file`: Optional text file containing class names (one per line)
- `--train_ratio`: Fraction of data for training (default: 0.7)
- `--val_ratio`: Fraction of data for validation (default: 0.2) 
- `--test_ratio`: Fraction of data for testing (default: 0.1)
- `--seed`: Random seed for reproducible splits (default: 42)

This creates a YOLO-format dataset with:
- Train/val/test splits with images and labels
- `data.yaml` configuration file
- Normalized bounding box coordinates in YOLO format

## YOLO Pose Dataset Export

Export bounding box and keypoint data for YOLOv8 pose estimation training:

```bash
python export_yolo_pose.py -i /path/to/labeled/data -v /path/to/videos -o /path/to/output -s skeleton.json
```

Parameters:
- `-i, --label_dir`: Directory containing timestamped label folders from RED
- `-v, --video_dir`: Directory containing video files (should match camera names)  
- `-o, --output_dir`: Output directory for YOLO pose dataset
- `-s, --skeleton_file`: JSON file defining skeleton structure (from RED skeleton creator)
- `--train_ratio`: Fraction of data for training (default: 0.7)
- `--val_ratio`: Fraction of data for validation (default: 0.2)
- `--test_ratio`: Fraction of data for testing (default: 0.1)
- `--seed`: Random seed for reproducible splits (default: 42)

This creates a YOLO pose dataset with:
- Train/val/test splits with images and labels
- `data.yaml` configuration file for pose training
- Normalized bounding boxes and keypoint coordinates
- Skeleton configuration file
- README with keypoint names and format documentation

### Example Usage

```bash
# Export detection dataset
python export_yolo_detection.py \
    -i /path/to/red/exports \
    -v /path/to/videos \
    -o /path/to/yolo_detection_dataset \
    -c class_names.txt

# Export pose dataset  
python export_yolo_pose.py \
    -i /path/to/red/exports \
    -v /path/to/videos \
    -o /path/to/yolo_pose_dataset \
    -s /path/to/skeleton.json
```

### Requirements

Both export scripts require:
- Video files named to match camera names (e.g., `cam1.mp4`, `cam2.mp4`)
- Exported label data from RED (with bounding boxes and/or keypoints)
- PyNvVideoCodec for GPU-accelerated video decoding (optional, falls back to OpenCV)

## Jarvis
### Generate training data 

```
conda activate red_exporter
cd data_exporter
```

```
python red3d2jarvis.py -p project_path -o output_folder -m margin_for_bounding_box_in_pixels [-s subset of keypoints index] [-e new skeleton edges if use subset of keypoints]
```
Note, `project_path` is the `red` project folder — it should contain `labeled_data` and `project.redproj`.

You should check your dataset to ensure it is properly exported,

```
python check_jarvis_dataset.py -i jarvis_dataset [-s train/valid]
```

If the skeleton is not in the `keypoints.py`, please add your skeleton manually. We will provide utils for generating from skeleton json file soon.  


### Load predictions in RED

We provide a script to convert JARVIS predictions back to RED format for visualizing in RED.

```
python jarvis2red3d.py -i /path/to/predictions_3D_folder/ -p /path/to/red_project
```

The converted predictions land in `<project>/predictions/`. Open the project in RED and use **Load From Selected** to load the predictions, then scrub through the predicted poses across all views.

A confidence filter is applied by default (drops predictions below 0.7 confidence and z > 500mm, since rats are not that tall). Disable with `--filter=0`.



### Merge multiple jarvis datasets (WIP)
You can merge multiple jarvis projects into a single projects (we assume both projects have same camera set, image resolutions, & XX) 

- first prepare datasets by converting multiple annotated projects from `red` into `jarvis` projects using the script `red3d2jarvis.py`
- collate the different `jarvis` projects into a single directory (say `~/data/jarvis_merge` with projects `dataset_11_06` and `dataset_11_25`)
- folder tree should look like this:

```
jarvis_merge
├── dataset_11_06
│   ├── annotations
│   ├── calib_params
│   ├── train
│   └── val
├── dataset_11_25
│   ├── annotations
│   ├── calib_params
│   ├── train
│   └── val
```
- run the script `merge_jarvis_datasets.py` as below to merge them into a single project
```
python merge_jarvis_datasets.py -i ~/data/jarvis_merge -o ~/data/test_merge
```

## YOLO 
### Generate training data 
```
python red3d2yolo.py -i path/to/labels -o output_dir -d 40
```
d is the diameter of the ball in the same unit of calibration, for instance, mm. The scrip automatically scale the bounding box size based on the depth to the camera. 

### To visualize a dataset
```
python check_yolo_dataset.py -y path/to/config.yaml [-s train/val]
```
