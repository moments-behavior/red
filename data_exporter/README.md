# Format data for deep learning training 

Create conda python virtual environment

```
conda create -n red_exporter python=3.9
conda activate red_exporter
conda install numpy
conda install -c menpo opencv
conda install pyyaml
```

## Generate training data 

```
conda activate red_exporter
cd data_exporter
```

```
python red3d2jarvis.py -i [label_folder] -s [skeleton name] -o [output_folder]
```

For instance, 

```
python red3d2jarvis.py -i /home/user/example_for_data_exporting/labeled_data/ -s rat24 -o /home/user/example_for_data_exporting/jarvis_test
```

If the skeleton is not in the `keypoints.py`, please add your skeleton manually. We will provide utils for generating from skeleton json file soon.  


## Load model predictions 

We provide script to convert JARVIS prediction back to RED format for visualizing in RED. 

```
python jarvis2red3d.py -i /path/to/predictions_3D_folder/ -s [skeleton name] -o [output_folder]
```
Note, there is a filter applied to filter out predictions with confidence score less than 0.7, and z > 500mm (since rats are not that tall). The fiter could be disabled by `--filter=0`. 
