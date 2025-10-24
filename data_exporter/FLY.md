## export annotations from red -> jarvis
1. navigate to the `dlt` branch of `red`
```
cd ~/src2/red/data_exporter
```
2. activate (or setup) conda environment for data export -- see [README.md](README.md) for how to setup the conda environment
```
conda activate red_exporter
```
3. run the python script [`red3d2jarvis.py`](red3d2jarvis.py) with proper arguments
```
python red3d2jarvis.py -i label_folder -o output_folder --scale_10x 1(if yes)
```
For example, 
```
python red3d2jarvis.py  -i /home/user/fly_label/fly_courtship/2025_10_20_13_20_04/labeled_data  -o /home/user/fly_label/fly_courtship/2025_10_20_13_20_04/jarvis_exports/v0_1023/ --scale_10x 1
```


## running jarvis

1. start jarvis in command line
```
cd ~/src/JARVIS-HybridNet
conda activate jarvis
jarvis-local launch-cli
```

2. create project --> point to the `output_folder` used in Step 3 of [export annotations from red -> jarvis](#export-annotations-from-red---jarvis) 

3. train
-  use values suggested by network 
-  grid spacing should be automatically suggested to 1mm
-  test a quick training using very few iterations to verify that all networks are trained
-  train with 50,100,and 50 iterations for center/keypoint/hybrid nets respectively

