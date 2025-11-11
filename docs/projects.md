# projects

### creating project file (.redproj)
- Project files (`.redproj`) usually lives in `/home/$USER/red_data/projects` -- create the directory if not present
- You can modify the [example file](example/fly.redproj) and save it in the above directory

### modifying project file 

I'm not sure what all the different paths are, but some of them share the same value
- `project_root_path` -- not sure, but I think this is where all the project files are saved 
    - overwritten to be `/home/$USER/red_data` in [red.cpp](../src/red.cpp)? 
    - The `.redproj` usually lives here so that it can be easily opened

- `project_path` -- timestamped folder with recorded video is (`YYYY_MM_DD_HH_mm_ss`)

- `load_skeleton_from_json` -- set to true if using JSON skeleton

- `skeleton_file` -- JSON file with skeleton structure 
    - **IMPORTANT** -- make sure that the `has_bbox` inside the skeleton JSON file is set to `false` -- current version does not show annotation when this flag is set to `true` (fix is a TODO)

- `calibration_folder` -- directory where the calibration files are (`*.yaml` or `*_dlt.csv`)

- `keypoints_root_folder` -- directory containing `labeled_data` for each videos 
    - usually same as the folder with videos -- see example file [here](../example/fly.redproj)

- `camera_names` -- list camera names as per the video name 
    - for example, if the videos are saved as `Cam2005325.mp4` and`Cam2006051.mp4`, this field should have `Cam2005325` and `Cam2006051` 

- `skeleton_name` -- this is used when you are using one of the skeletons in the source code and not using a JSON 
    - for example, `Rat24`, `Rat4` are built-in to the source code

- `media_folder` -- folder where the videos live (I think) -- same as the `project path`?