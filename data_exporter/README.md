# Format data for deep learning  

Create conda python virtual environment

```
conda create -n red_exporter python=3.9
conda activate red_exporter
conda install numpy
conda install -c menpo opencv
conda install pyyaml
```
## Jarvis
### Generate training data 

```
conda activate red_exporter
cd data_exporter
```

```
python red3d2jarvis.py -i [label_folder] -o [output_folder]
```

For instance, 

```
python red3d2jarvis.py -i /home/user/example_for_data_exporting/labeled_data/ -o /home/user/example_for_data_exporting/jarvis_test -m 40 -s 0 1 2 3
```
To check the dataset,

```
python check_jarvis_dataset.py -i /nfs/exports/ratlv/exp_2025_03/remy_emilie/2025_04_04_14_26_47/jarvis_test -s train
```

If the skeleton is not in the `keypoints.py`, please add your skeleton manually. We will provide utils for generating from skeleton json file soon.  


### Load predictions in RED

We provide script to convert JARVIS prediction back to RED format for visualizing in RED. 

```
python jarvis2red3d.py -i /path/to/predictions_3D_folder/ -s [skeleton name] -o [output_folder]
```
Note, there is a filter applied to filter out predictions with confidence score less than 0.7, and z > 500mm (since rats are not that tall). The fiter could be disabled by `--filter=0`. 



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
python red3d2yolo.py -i [path/to/labels] -o output_dir -d 100
```
### To visualize a dataset
```
python check_yolo_dataset.py -y [path/to/config.yaml] -s [train or val]
```
